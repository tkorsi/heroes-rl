"""Headless environment contract: visibility, isolation, boundaries and scoring.

These tests use synthetic messages and a miniature fake engine. They establish
the adapter's behaviour, not real VCMI gameplay, completion or throughput.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_ENGINE = ROOT / "tests/fixtures/headless/fake_engine.py"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through the module registry
    spec.loader.exec_module(module)
    return module


def learner_contract():
    """Codex's player-history v1 contract when present (``HEROES_LEARNER_PATH`` names its parent)."""
    path = Path(os.environ.get("HEROES_LEARNER_PATH", ROOT / "experiments")) / "learner" / "contract.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("learner_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["learner_contract"] = module
    spec.loader.exec_module(module)
    return module


V1_KEYS = {"view": {"schema", "episode", "player", "seq", "day", "kind", "view"},
           "decision": {"schema", "episode", "player", "seq", "day", "kind", "view", "mask"},
           "action": {"schema", "episode", "player", "seq", "day", "kind", "action", "decision_seq"},
           "result": {"schema", "episode", "player", "seq", "day", "kind", "action", "accepted", "decision_seq"}}


env_module = load_module("headless_env", "experiments/headless/env.py")
agents = load_module("headless_agents", "experiments/headless/agents.py")
bridge_protocol = sys.modules["bridge_protocol"]
player_history = sys.modules["player_history"]
ProtocolError = bridge_protocol.ProtocolError
COLORS = env_module.COLORS


def obj(oid, x, y, owner=1, kind=34):
    return {"id": oid, "x": x, "y": y, "z": 0, "type": kind, "subtype": 0, "owner": owner}


def tile(position, terrain=2, blocked=False, visitable=False):
    x, y, z = position
    return {"x": x, "y": y, "z": z, "terrain": terrain, "blocked": blocked, "visitable": visitable}


OWN = {"hero": [0, 0, 0], "movement": 1, "movement_max": 1, "resources": [0] * 7,
       "army": [{"slot": 0, "creature": 1, "count": 10}], "towns": [], "mines": 0}


def update(sequence, appeared=(), changed=(), disappeared=(), revealed=(), hidden=(), day=1, tiles_changed=(),
           own=None):
    return {"kind": "visible_update", "sequence": sequence, "day": day, "appeared": list(appeared),
            "changed": list(changed), "disappeared": list(disappeared),
            "revealed": [t if isinstance(t, dict) else tile(t) for t in revealed],
            "tiles_changed": list(tiles_changed), "hidden": [list(t) for t in hidden], "own": own}


def columns(x0, x1, height=3):
    return [(x, y, 0) for y in range(height) for x in range(x0, x1 + 1)]


def observation(action_id, tiles, objects, events, day=1, hero=1, facts=None, own=OWN):
    by_tile = {}
    for entry in objects:
        by_tile.setdefault((entry["x"], entry["y"]), []).append(
            {"id": entry["id"], "type": entry["type"], "subtype": entry["subtype"], "owner": entry["owner"]})
    ordered = sorted(tiles, key=lambda t: (t[2], t[1], t[0]))
    facts = facts or {}
    state = {"day": day, "hero_id": hero, "own_heroes": 1, "own_towns": len(own["towns"]), "own_mines": own["mines"],
             "resources": own["resources"], "army": own["army"], "towns": own["towns"], "movement": own["movement"],
             "movement_max": own["movement_max"]}
    if own["hero"] is not None:
        state.update(x=own["hero"][0], y=own["hero"][1], z=own["hero"][2])
    return {"type": "observation", "protocol_version": 3, "player": 0, "episode": "1", "action_id": action_id,
            "day": day, "hero_id": hero, "x": 0, "y": 0, "z": 0, "width": 6, "height": 3, "levels": 1,
            "movement": 1, "movement_max": 1, "action_mask": [True] * 9, "events": events, "state": state,
            "tiles": [{**facts.get((x, y, z), tile((x, y, z))), "objects": by_tile.get((x, y), [])} for x, y, z in ordered]}


def history(player=0, color="red"):
    result = player_history.PlayerHistory(player, color, "1")
    result.start([6, 3, 1])
    return result


class PlayerHistoryTests(unittest.TestCase):
    """The four handoff scenarios plus last-seen and handle discipline."""

    def setUp(self):
        self.history = history()
        self.own = obj(1, 0, 0, owner=0)
        self.history.ingest(observation(0, columns(0, 2), [self.own, obj(2, 2, 1)],
                                        [update(0, appeared=[self.own, obj(2, 2, 1)], revealed=columns(0, 2), own=OWN)]))

    def test_visible_to_hidden_move_is_a_bare_disappearance(self):
        record = self.history.ingest(observation(1, columns(0, 2), [self.own], [update(1, disappeared=[2])]))
        event = record["events"][0]
        self.assertEqual(event["disappeared"], [2])
        self.assertEqual(event["appeared"] + event["changed"], [])
        memory = {entry["handle"]: entry for entry in self.history.memory()}
        self.assertEqual(memory[2]["fact"]["x"], 2)  # last seen where it was visible, never the hidden endpoint
        self.assertFalse(memory[2]["visible"])

    def test_transient_sighting_between_decisions_enters_history(self):
        records = [self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)], [
            update(1, appeared=[obj(3, 2, 2)]), update(2, disappeared=[3])]))]
        events = records[0]["events"]
        self.assertEqual(events[0]["appeared"], [{"handle": 3, "x": 2, "y": 2, "z": 0, "type": 34, "subtype": 0, "owner": 1}])
        self.assertEqual(events[1]["disappeared"], [3])
        # The snapshot no longer shows it; only the event stream carries the sighting.
        self.assertNotIn(3, [o["handle"] for tile in records[0]["tiles"] for o in tile["objects"]])
        self.assertEqual(self.history.memory()[2]["fact"], {"x": 2, "y": 2, "z": 0, "type": 34, "subtype": 0, "owner": 1})

    def test_hidden_removal_and_unseen_objects_never_reach_the_learner(self):
        with self.assertRaisesRegex(ProtocolError, "never observed"):
            self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)], [update(1, disappeared=[99])]))
        fresh = history()
        with self.assertRaisesRegex(ProtocolError, "does not see"):
            fresh.ingest(observation(0, columns(0, 2), [], [update(0, appeared=[obj(9, 5, 1)], revealed=columns(0, 2), own=OWN)]))
        self.assertEqual(fresh.handle_count(), 0)

    def test_last_seen_never_refreshes_from_hidden_state(self):
        self.history.ingest(observation(1, columns(0, 2), [self.own], [update(1, disappeared=[2])]))
        self.history.ingest(observation(2, columns(0, 2), [self.own], []))  # the enemy moves while hidden: no event
        self.assertEqual(self.history.memory()[1]["fact"]["x"], 2)
        self.history.ingest(observation(3, columns(0, 2), [self.own, obj(2, 1, 2)], [update(2, appeared=[obj(2, 1, 2)])]))
        entry = self.history.memory()[1]
        self.assertEqual((entry["fact"]["x"], entry["fact"]["y"], entry["visible"]), (1, 2, True))
        self.assertEqual(entry["handle"], 2)  # the same handle: identity is kept, not re-issued

    def test_snapshot_that_disagrees_with_the_event_stream_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "diverged"):
            self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 1, 1)], []))

    def test_independent_player_histories_and_handles(self):
        red, blue = history(), history(1, "blue")
        red.ingest(observation(0, columns(0, 2), [obj(7, 0, 0, 0), obj(8, 2, 2)],
                               [update(0, appeared=[obj(7, 0, 0, 0), obj(8, 2, 2)], revealed=columns(0, 2), own=OWN)], hero=7))
        blue_message = observation(0, columns(3, 5), [obj(9, 4, 1, 1), obj(8, 3, 2)],
                                   [update(0, appeared=[obj(8, 3, 2), obj(9, 4, 1, 1)], revealed=columns(3, 5), own=OWN)], hero=9)
        blue_message["player"] = 1
        blue.ingest(blue_message)
        self.assertEqual(red._handles, {7: 1, 8: 2})
        # Same engine object, unrelated per-player handle; within one update
        # handles follow the scan order of visible facts, never engine IDs.
        self.assertEqual(blue._handles, {9: 1, 8: 2})
        self.assertNotIn(9, red._handles)

    def test_tile_facts_and_own_state_must_match_the_snapshot(self):
        blocked = tile((1, 1, 0), blocked=True)
        richer = {**OWN, "resources": [0, 0, 0, 0, 0, 0, 500]}
        events = [update(1, tiles_changed=[blocked], own=richer)]
        good = observation(1, columns(0, 2), [self.own, obj(2, 2, 1)], events, facts={(1, 1, 0): blocked}, own=richer)
        for facts, own, label in (({}, richer, "tile"), ({(1, 1, 0): blocked}, OWN, "own")):
            with self.subTest(label), self.assertRaisesRegex(ProtocolError, "diverged"):
                copy.deepcopy(self.history).ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)],
                                                               copy.deepcopy(events), facts=facts, own=own))
        record = self.history.ingest(good)
        self.assertEqual(record["events"][0]["tiles_changed"], [blocked])
        self.assertEqual(self.history.own["resources"][6], 500)

    def test_tile_facts_only_for_visible_tiles_and_own_state_first(self):
        with self.assertRaisesRegex(ProtocolError, "does not see"):
            self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)],
                                            [update(1, tiles_changed=[tile((5, 1, 0), blocked=True)])]))
        with self.assertRaisesRegex(ProtocolError, "own state"):
            history().ingest(observation(0, columns(0, 2), [], [update(0, revealed=columns(0, 2))]))

    def test_learner_records_contain_no_engine_ids(self):
        record = self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)], []))
        self.assertEqual(agents.engine_id_paths(record), [])
        self.assertEqual(record["hero"], 1)


class HistoryV1Tests(unittest.TestCase):
    """The projection onto the learner's heroes-player-history-v1 frames."""

    def setUp(self):
        self.history = history()
        self.own = obj(1, 0, 0, owner=0)
        self.history.ingest(observation(0, columns(0, 2), [self.own, obj(2, 2, 1), obj(3, 1, 2, owner=255, kind=53)], [
            update(0, appeared=[self.own, obj(2, 2, 1), obj(3, 1, 2, owner=255, kind=53)], revealed=columns(0, 2), own=OWN)]))

    def result(self, action, accepted, events):
        state = {"day": 1, "hero_id": 1, "resources": [0] * 7}
        return {"type": "action_result", "protocol_version": 3, "player": 0, "episode": "1", "action_id": 0,
                "action": action, "accepted": accepted, "before": state, "after": state, "events": events}

    def test_frames_follow_the_contract_order_with_relative_owners_and_handles(self):
        self.history.act(2)
        self.history.ingest(self.result(2, True, [update(1, disappeared=[2])]))
        frames = self.history.v1.frames
        self.assertEqual([f["kind"] for f in frames], ["view", "decision", "action", "view", "result"])
        self.assertEqual([f["seq"] for f in frames], list(range(5)))
        self.assertEqual((frames[2]["decision_seq"], frames[4]["decision_seq"], frames[4]["accepted"]), (1, 1, True))
        for frame in frames:
            self.assertEqual(set(frame), V1_KEYS[frame["kind"]])
            self.assertEqual((frame["schema"], frame["episode"], frame["player"], frame["day"]),
                             ("heroes-player-history-v1", "1", 0, 1))
        owners = {o["handle"]: o["owner"] for o in frames[1]["view"]["objects"]}
        self.assertEqual(owners, {"1": "self", "2": "opponent", "3": "neutral"})
        self.assertNotIn("2", [o["handle"] for o in frames[3]["view"]["objects"]])  # gone, never refreshed
        self.assertEqual(frames[1]["view"]["size"], [6, 3, 1])
        self.assertEqual(len(frames[1]["view"]["tiles"]), 9)
        self.assertEqual(frames[1]["mask"], [True] * 9)
        self.assertEqual(agents.engine_id_paths(frames), [])
        self.assertTrue(self.history.v1.summary()["valid"])

    def test_forbidden_order_invalidates_only_the_projection(self):
        self.history.ingest({"type": "terminal", "protocol_version": 3, "player": 0, "episode": "1", "outcome": "loss",
                             "state": {"day": 1, "hero_id": -1}, "events": [update(1, disappeared=[2])]})
        summary = self.history.v1.summary()
        self.assertFalse(summary["valid"])
        self.assertIn("after an unacted decision", summary["error"])
        self.assertEqual(summary["frames"], 2)  # the projection stops; the full record stream continues
        self.assertEqual(self.history.records[-1]["type"], "terminal")

    def test_unrepresented_visible_events_are_counted(self):
        self.history.act(8)
        self.history.ingest(self.result(8, True, [{"kind": "battle_start", "side": 0, "opponent": "neutral"},
                                                  {"kind": "battle_complete", "won": True},
                                                  {"kind": "visit", "object_id": 3, "object_type": 53,
                                                   "object_subtype": 0, "owner": 255}]))
        self.assertEqual(self.history.v1.summary()["unrepresented_events"],
                         {"battle_complete": 1, "battle_start": 1, "visit": 1})

    @unittest.skipIf(learner_contract() is None, "learner contract not present; set HEROES_LEARNER_PATH")
    def test_frames_decode_with_the_learner_contract(self):
        contract = learner_contract()
        self.history.act(2)
        self.history.ingest(self.result(2, False, [update(1, disappeared=[2])]))
        with self.assertRaises(ValueError):  # the 6x3 fixture map is outside the authored 20x20x1 scenario
            contract.decode_frame(self.history.v1.frames[0])


class BridgeValidatorTests(unittest.TestCase):
    def ready(self, **fields):
        return {"type": "ready", "protocol_version": 3, "player": 0, "episode": "1", "adventure_actions": 9,
                "observation_events": "visible_update-v2", "map_size": [6, 3, 1], **fields}

    def test_handshake_requires_protocol_three_and_the_event_stream(self):
        for message in (self.ready(protocol_version=2), self.ready(observation_events=None), self.ready(player=1),
                        self.ready(observation_events="visible_update-v1"), self.ready(map_size=None)):
            with self.subTest(message=message), self.assertRaises(ProtocolError):
                bridge_protocol.BridgeValidator(0, "1").check(message)

    def test_ordering_and_gaps(self):
        validator = bridge_protocol.BridgeValidator(0, "1")
        validator.check(self.ready())
        first = observation(0, columns(0, 0), [], [update(0, revealed=columns(0, 0), own=OWN)])
        validator.check(first)
        with self.assertRaises(ValueError):
            validator.respond(9)
        self.assertEqual(validator.respond(8), {"action_id": 0, "action": 8})
        result = {"type": "action_result", "protocol_version": 3, "player": 0, "episode": "1", "action_id": 0,
                  "action": 8, "accepted": True, "before": {}, "after": {}, "events": [update(2, hidden=columns(0, 0))]}
        with self.assertRaisesRegex(ProtocolError, "gap"):
            validator.check(result)
        early = bridge_protocol.BridgeValidator(0, "1")
        early.check(self.ready())
        with self.assertRaisesRegex(ProtocolError, "out of order"):
            early.check(observation(1, columns(0, 0), [], []))

    def test_snapshot_period_is_declared_and_enforced(self):
        def result(action_id):
            return {"type": "action_result", "protocol_version": 3, "player": 0, "episode": "1", "action_id": action_id,
                    "action": 0, "accepted": True, "before": {}, "after": {}, "events": []}
        first = observation(0, columns(0, 0), [], [update(0, revealed=columns(0, 0), own=OWN)])
        for declared, second_has_tiles, fails in ((1, False, True), (2, False, False), (2, True, True), (0, True, True)):
            with self.subTest(declared=declared, second_has_tiles=second_has_tiles):
                validator = bridge_protocol.BridgeValidator(0, "1")
                validator.check(self.ready(snapshot_every=declared))
                opening = dict(first)
                if declared == 0:
                    del opening["tiles"]
                validator.check(opening)
                validator.respond(0)
                validator.check(result(0))
                second = observation(1, columns(0, 0), [], [])
                if not second_has_tiles:
                    del second["tiles"]
                if fails:
                    with self.assertRaisesRegex(ProtocolError, "tile list"):
                        validator.check(second)
                else:
                    validator.check(second)
        with self.assertRaises(ProtocolError):
            bridge_protocol.BridgeValidator(0, "1").check(self.ready(snapshot_every=-1))


class FakeLauncher:
    def __init__(self, **scenario):
        self.scenario = scenario

    def launch(self, directory, seed, environment, log):
        (directory / "engine").mkdir(parents=True, exist_ok=True)
        inherited = {k: v for k, v in os.environ.items() if not k.startswith(("EXPLORER_RL_", "VCMI_"))}
        launch_env = {**environment, "VCMI_EXPERIMENT_DIR": str(directory / "engine"),
                      "FAKE_ENGINE_SCENARIO": json.dumps(self.scenario), "FAKE_ENGINE_SEED": str(seed)}
        process = subprocess.Popen([sys.executable, str(FAKE_ENGINE)], env={**inherited, **launch_env},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process, {"command": ["fake_engine.py"], "environment": launch_env}

    def identity(self):
        return {"fake_engine": True}


class ScriptedAgent:
    def __init__(self, actions=()):
        self.actions = list(actions)

    def act(self, request):
        if request.kind == "decision":
            return request.choices[0]
        while self.actions:
            action = self.actions.pop(0)
            if request.mask[action]:
                return action
        return 8


def play(env, players, episode=1, seed=7):
    requests = []
    step = env.reset(episode, seed)
    while not step.done:
        for record in step.since_last:
            assert agents.engine_id_paths(record) == [], agents.engine_id_paths(record)
        requests.append(step)
        step = env.step(players[step.color].act(step))
    return step, requests


class HeadlessEnvTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-headless-test-")
        self.output = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def make(self, limits=None, *, snapshot_every=1, **scenario):
        env = env_module.HeadlessSelfPlayEnv(FakeLauncher(**scenario), self.output / str(len(list(self.output.iterdir()))),
                                             limits or env_module.Limits(watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10),
                                             snapshot_every=snapshot_every)
        self.addCleanup(env.close)
        return env

    def random_players(self):
        return {"red": agents.UniformRandomAgent(1), "blue": agents.UniformRandomAgent(2)}

    def test_natural_episode_is_verified_and_rewarded_and_resets_cleanly(self):
        env = self.make(mode="natural", end_day=3)
        first, requests = play(env, self.random_players())
        self.assertEqual((first.status, first.termination_reason), ("completed", "natural_terminal"))
        self.assertEqual(first.rewards, {"red": 1, "blue": -1})
        self.assertTrue(all(request.reward == 0 for request in requests))
        self.assertEqual(first.manifest["returncode"], 0)
        self.assertIsNotNone(first.manifest["engine_evidence"])
        self.assertFalse(any(hasattr(r, "env_step") for r in requests))  # a global step count would reveal opponent activity
        self.assertEqual(first.final_records["blue"][-1]["type"], "terminal")
        red_history = env.history("red")
        second, _ = play(env, self.random_players(), episode=2, seed=8)
        self.assertEqual(second.status, "completed")
        # A reset starts a fresh process, fresh handles and an independent history.
        self.assertEqual(env.history("red")[0]["index"], 0)
        self.assertEqual(env.sides[0].history.handle_count(), len({h for h in env.sides[0].history._handles.values()}))
        self.assertTrue((env.output / "episode-000001/manifest.json").is_file())
        self.assertTrue((env.output / "episode-000002/manifest.json").is_file())
        self.assertNotEqual(red_history, env.history("red")) if len(red_history) != len(env.history("red")) else None

    def test_history_v1_frames_reach_each_player_and_decode(self):
        env = self.make(mode="natural", end_day=3, blue_start=[4, 2])
        end, requests = play(env, {"red": ScriptedAgent([6, 0]), "blue": ScriptedAgent([6, 2])})
        self.assertEqual(end.status, "completed")
        contract = learner_contract()
        for color in ("red", "blue"):
            frames = env.history_v1(color)
            delivered = [f for r in requests if r.color == color for f in r.frames] + end.final_frames[color]
            self.assertEqual(json.loads(json.dumps(delivered)), frames)
            for request in (r for r in requests if r.color == color and r.kind == "action"):
                self.assertEqual(request.frames[-1]["kind"], "decision")
                self.assertEqual(request.frames[-1]["mask"], request.mask)
            self.assertEqual([f["seq"] for f in frames], list(range(len(frames))))
            summary = end.manifest["players"][color]["history_v1"]
            self.assertEqual((summary["valid"], summary["frames"]), (True, len(frames)))
            self.assertEqual(agents.engine_id_paths(frames), [])
            if contract is not None:
                cursor = contract.HistoryCursor("1", COLORS.index(color))
                for frame in frames:
                    cursor.consume(contract.decode_frame(frame))
        red_views = [f["view"] for f in env.history_v1("red") if f["kind"] == "view"]
        sightings = [o for view in red_views for o in view["objects"] if o["owner"] == "opponent" and o["kind"] == 34]
        self.assertEqual({tuple(o["position"]) for o in sightings}, {(3, 2, 0)})  # only where Red saw it

    def test_transient_enemy_sighting_reaches_only_the_observer(self):
        env = self.make(mode="natural", end_day=2, blue_start=[4, 2])
        end, requests = play(env, {"red": ScriptedAgent(), "blue": ScriptedAgent([6, 2])})
        self.assertEqual(end.status, "completed")
        red_day_two = next(r for r in requests if r.color == "red" and r.record["day"] == 2)
        events = [e for record in red_day_two.since_last for e in record.get("events", []) if e["kind"] == "visible_update"]
        appeared = [o for e in events for o in e["appeared"] if o["type"] == 34 and o["owner"] == 1]
        self.assertEqual([(o["x"], o["y"]) for o in appeared], [(3, 2)])
        handle = appeared[0]["handle"]
        self.assertIn(handle, [h for e in events for h in e["disappeared"]])
        snapshot = [o["handle"] for tile in red_day_two.record["tiles"] for o in tile["objects"]]
        self.assertNotIn(handle, snapshot)
        memory = {entry["handle"]: entry for entry in env.memory("red")}
        self.assertEqual((memory[handle]["fact"]["x"], memory[handle]["visible"]), (3, False))
        # Blue's hero start and return tiles are hidden to Red: never in Red's history.
        red_positions = {(o["x"], o["y"]) for record in env.history("red") for e in record.get("events", [])
                         if e["kind"] == "visible_update" for o in e["appeared"] + e["changed"] if o["handle"] == handle}
        self.assertEqual(red_positions, {(3, 2)})

    def test_horizons_truncate_without_reward_and_keep_a_bootstrap_boundary(self):
        for limits, reason, players in ((env_module.Limits(max_days=1), "game_day_limit", 2),
                                        (env_module.Limits(max_actions=2), "agent_action_limit", 1)):
            with self.subTest(reason=reason):
                env = self.make(limits, mode="natural", end_day=5)
                end, _ = play(env, self.random_players())
                self.assertEqual((end.status, end.termination_reason), ("truncated", reason))
                self.assertEqual(end.rewards, {"red": None, "blue": None})
                # A day horizon gives each player its own first decision past it; the action guard only one.
                boundary = {color: record for color, record in end.bootstrap.items() if record}
                self.assertEqual(len(boundary), players)
                for color, record in boundary.items():
                    self.assertEqual(record["type"], "observation")
                    seq = end.manifest["players"][color]["cutoff"]["history_v1_seq"]
                    frame = env.history_v1(color)[seq]
                    self.assertEqual((frame["kind"], frame["mask"], frame["day"]), ("decision", record["action_mask"], record["day"]))
                    if reason == "game_day_limit":
                        self.assertEqual(record["day"], 2)

    def test_requests_past_a_players_horizon_are_marked(self):
        env = self.make(env_module.Limits(max_days=1), mode="natural", end_day=5)
        end, requests = play(env, self.random_players())
        self.assertEqual(end.termination_reason, "game_day_limit")
        for color in ("red", "blue"):
            marks = [(r.record["day"], r.past_horizon) for r in requests if r.color == color and r.kind == "action"]
            self.assertTrue(marks and all(past == (day > 1) for day, past in marks), marks)
            self.assertEqual(end.bootstrap_v1_seq[color], end.manifest["players"][color]["cutoff"]["history_v1_seq"])

    def test_endings_during_the_horizon_continuation(self):
        # Red passes the day horizon first and plays on until Blue reaches its own boundary.
        fast = env_module.Limits(max_days=1, watchdog_seconds=1.5, startup_seconds=20, shutdown_seconds=1.5)
        crash, _ = play(self.make(fast, mode="crash", crash_day=2), self.random_players())
        self.assertEqual((crash.status, crash.termination_reason), ("failed", "process_exit_without_verified_terminal"))
        self.assertEqual(crash.bootstrap, {"red": None, "blue": None})
        natural, _ = play(self.make(fast, mode="natural", end_day=2), self.random_players())
        self.assertEqual((natural.status, natural.termination_reason), ("truncated", "game_day_limit"))
        self.assertEqual(natural.rewards, {"red": None, "blue": None})
        self.assertEqual(natural.manifest["natural_terminal_after_horizon"], {"red": "win", "blue": "loss"})
        self.assertIsNotNone(natural.bootstrap["red"])
        self.assertIsNone(natural.bootstrap["blue"])  # Blue never reached a boundary of its own
        evidence, _ = play(self.make(fast, mode="missing_metadata", end_day=5), self.random_players())
        self.assertEqual((evidence.status, evidence.termination_reason), ("failed", "engine_evidence_invalid"))
        self.assertEqual(evidence.bootstrap, {"red": None, "blue": None})

    def test_technical_failures_are_never_scored(self):
        fast = env_module.Limits(watchdog_seconds=1.5, startup_seconds=20, shutdown_seconds=1.5)
        cases = [({"mode": "crash"}, "process_exit_without_verified_terminal"),
                 ({"mode": "hang"}, "watchdog_timeout"),
                 ({"mode": "natural", "action_id_offset": 1}, "protocol_error"),
                 ({"mode": "engine_error"}, "policy_error"),
                 ({"mode": "unclean_exit"}, "unclean_exit_after_terminals"),
                 ({"mode": "no_shutdown"}, "shutdown_timeout"),
                 ({"mode": "missing_terminal"}, "unverified_agent_terminal"),
                 ({"mode": "missing_metadata"}, "engine_evidence_invalid"),
                 ({"mode": "leak"}, "protocol_error"),
                 ({"mode": "exit_early"}, "agent_never_acted"),
                 ({"mode": "natural", "observation_events": "none"}, "protocol_error")]
        for scenario, reason in cases:
            with self.subTest(**scenario):
                end, _ = play(self.make(fast, **scenario), self.random_players())
                self.assertEqual((end.status, end.termination_reason), ("failed", reason))
                self.assertEqual(end.rewards, {"red": None, "blue": None})
                self.assertEqual(end.outcomes, {"red": None, "blue": None})

    def test_interleaved_decision_for_the_waiting_player(self):
        env = self.make(mode="decision")
        end, requests = play(env, {"red": ScriptedAgent([2]), "blue": ScriptedAgent()})
        self.assertEqual(end.status, "completed")
        self.assertEqual(end.rewards, {"red": -1, "blue": 1})
        order = [(r.color, r.kind) for r in requests[:3]]
        self.assertEqual(order, [("red", "action"), ("blue", "decision"), ("red", "action")])
        decision = requests[1].record
        self.assertEqual(decision["hero"], 1)  # Blue's own hero handle, never its engine ID
        self.assertNotIn("hero_id", decision)
        # A level-up without a query is reported, never mistaken for a dialog.
        level_ups = [e for record in env.history("blue") for e in record.get("events", []) if e["kind"] == "level_up"]
        self.assertEqual(level_ups, [{"kind": "level_up", "primary_skill": 1, "secondary_skills": [7]}])
        self.assertEqual(end.manifest["players"]["blue"]["history_v1"]["unrepresented_events"]["level_up"], 1)

    def test_periodic_snapshots_still_verify_and_never_thin_the_events(self):
        full_env = self.make(mode="natural", end_day=3)
        full, _ = play(full_env, self.random_players())
        periodic_env = self.make(mode="natural", end_day=3, snapshot_every=3)
        periodic, _ = play(periodic_env, self.random_players())
        self.assertEqual((periodic.status, periodic.termination_reason), ("completed", "natural_terminal"))
        strip = lambda records: [{k: v for k, v in r.items() if k != "tiles"} for r in records]
        for color in ("red", "blue"):
            observations = [r for r in periodic_env.history(color) if r["type"] == "observation"]
            self.assertGreater(len(observations), 3)
            self.assertEqual([r["tiles"] is not None for r in observations], [i % 3 == 0 for i in range(len(observations))])
            # Same seeds, same events: only the redundant tile lists are thinned.
            self.assertEqual(strip(periodic_env.history(color)), strip(full_env.history(color)))
            self.assertEqual(periodic_env.history_v1(color), full_env.history_v1(color))
            self.assertTrue(periodic.manifest["players"][color]["history_v1"]["valid"])

    def test_terminal_racing_an_action_result_keeps_the_history_gap_free(self):
        end, _ = play(self.make(mode="terminal_race", end_day=2), {"red": ScriptedAgent([8, 2]), "blue": ScriptedAgent()})
        self.assertEqual(end.status, "completed")
        red = end.final_records["red"]
        self.assertEqual(red[-1]["type"], "terminal")
        self.assertTrue(any(e["kind"] == "visible_update" for e in red[-1]["events"]))

    def test_invalid_agent_choice_is_rejected_before_sending(self):
        env = self.make(mode="natural", end_day=2)
        request = env.reset(1, 7)
        blocked = [index for index, allowed in enumerate(request.mask) if not allowed]
        with self.assertRaises(ValueError):
            env.step(blocked[0] if blocked else 9)
        self.assertIs(env.request, request)
        while not request.done:
            request = env.step(8 if request.kind == "action" else request.choices[0])
        self.assertEqual(request.status, "completed")


class HeadlessScoringTests(unittest.TestCase):
    def test_hero_death_is_not_a_loss_and_only_horizons_truncate(self):
        scoring = env_module.scoring
        executions = {c: {"ready": True, "actions_sent": 1, "error": None, "limit": None, "terminal": None} for c in ("red", "blue")}
        self.assertEqual(scoring.classify_headless_self_play(0, None, "hero died", executions)[:3],
                         ("failed", {"red": None, "blue": None}, "agent_never_acted" if False else "unverified_self_play_terminal"))
        for reason, status in (("game_day_limit", "truncated"), ("agent_action_limit", "truncated"),
                               ("watchdog_timeout", "failed"), ("protocol_error", "failed")):
            self.assertEqual(scoring.classify_headless_self_play(0, reason, "", executions)[0], status)


if __name__ == "__main__":
    unittest.main()
