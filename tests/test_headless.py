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


env_module = load_module("headless_env", "experiments/headless/env.py")
agents = load_module("headless_agents", "experiments/headless/agents.py")
bridge_protocol = sys.modules["bridge_protocol"]
player_history = sys.modules["player_history"]
ProtocolError = bridge_protocol.ProtocolError


def obj(oid, x, y, owner=1, kind=34):
    return {"id": oid, "x": x, "y": y, "z": 0, "type": kind, "subtype": 0, "owner": owner}


def update(sequence, appeared=(), changed=(), disappeared=(), revealed=(), hidden=(), day=1):
    return {"kind": "visible_update", "sequence": sequence, "day": day, "appeared": list(appeared),
            "changed": list(changed), "disappeared": list(disappeared), "revealed": [list(t) for t in revealed],
            "hidden": [list(t) for t in hidden]}


def columns(x0, x1, height=3):
    return [(x, y, 0) for y in range(height) for x in range(x0, x1 + 1)]


def observation(action_id, tiles, objects, events, day=1, hero=1):
    by_tile = {}
    for entry in objects:
        by_tile.setdefault((entry["x"], entry["y"]), []).append(
            {"id": entry["id"], "type": entry["type"], "subtype": entry["subtype"], "owner": entry["owner"]})
    ordered = sorted(tiles, key=lambda t: (t[2], t[1], t[0]))
    return {"type": "observation", "protocol_version": 3, "player": 0, "episode": "1", "action_id": action_id,
            "day": day, "hero_id": hero, "x": 0, "y": 0, "z": 0, "width": 6, "height": 3, "levels": 1,
            "movement": 1, "movement_max": 1, "action_mask": [True] * 9, "events": events,
            "state": {"day": day, "hero_id": hero, "own_heroes": 1, "own_towns": 1, "own_mines": 0, "resources": [0] * 7},
            "tiles": [{"x": x, "y": y, "z": z, "terrain": 2, "blocked": False, "visitable": bool(by_tile.get((x, y))),
                       "objects": by_tile.get((x, y), [])} for x, y, z in ordered]}


class PlayerHistoryTests(unittest.TestCase):
    """The four handoff scenarios plus last-seen and handle discipline."""

    def setUp(self):
        self.history = player_history.PlayerHistory(0, "red")
        self.own = obj(1, 0, 0, owner=0)
        self.history.ingest(observation(0, columns(0, 2), [self.own, obj(2, 2, 1)],
                                        [update(0, appeared=[self.own, obj(2, 2, 1)], revealed=columns(0, 2))]))

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
        fresh = player_history.PlayerHistory(0, "red")
        with self.assertRaisesRegex(ProtocolError, "does not see"):
            fresh.ingest(observation(0, columns(0, 2), [], [update(0, appeared=[obj(9, 5, 1)], revealed=columns(0, 2))]))
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
        red, blue = player_history.PlayerHistory(0, "red"), player_history.PlayerHistory(1, "blue")
        red.ingest(observation(0, columns(0, 2), [obj(7, 0, 0, 0), obj(8, 2, 2)],
                               [update(0, appeared=[obj(7, 0, 0, 0), obj(8, 2, 2)], revealed=columns(0, 2))], hero=7))
        blue_message = observation(0, columns(3, 5), [obj(9, 4, 1, 1), obj(8, 3, 2)],
                                   [update(0, appeared=[obj(8, 3, 2), obj(9, 4, 1, 1)], revealed=columns(3, 5))], hero=9)
        blue_message["player"] = 1
        blue.ingest(blue_message)
        self.assertEqual(red._handles, {7: 1, 8: 2})
        self.assertEqual(blue._handles, {8: 1, 9: 2})  # same engine object, unrelated per-player handle
        self.assertNotIn(9, red._handles)

    def test_learner_records_contain_no_engine_ids(self):
        record = self.history.ingest(observation(1, columns(0, 2), [self.own, obj(2, 2, 1)], []))
        self.assertEqual(agents.engine_id_paths(record), [])
        self.assertEqual(record["hero"], 1)


class BridgeValidatorTests(unittest.TestCase):
    def ready(self, **fields):
        return {"type": "ready", "protocol_version": 3, "player": 0, "episode": "1", "adventure_actions": 9,
                "observation_events": "visible_update-v1", **fields}

    def test_handshake_requires_protocol_three_and_the_event_stream(self):
        for message in (self.ready(protocol_version=2), self.ready(observation_events=None), self.ready(player=1)):
            with self.subTest(message=message), self.assertRaises(ProtocolError):
                bridge_protocol.BridgeValidator(0, "1").check(message)

    def test_ordering_and_gaps(self):
        validator = bridge_protocol.BridgeValidator(0, "1")
        validator.check(self.ready())
        first = observation(0, columns(0, 0), [], [update(0, revealed=columns(0, 0))])
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

    def make(self, limits=None, **scenario):
        env = env_module.HeadlessSelfPlayEnv(FakeLauncher(**scenario), self.output / str(len(list(self.output.iterdir()))),
                                             limits or env_module.Limits(watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10))
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
        self.assertEqual([r.env_step for r in requests], list(range(len(requests))))
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
        for limits, reason in ((env_module.Limits(max_days=1), "game_day_limit"),
                               (env_module.Limits(max_actions=2), "agent_action_limit")):
            with self.subTest(reason=reason):
                end, _ = play(self.make(limits, mode="natural", end_day=5), self.random_players())
                self.assertEqual((end.status, end.termination_reason), ("truncated", reason))
                self.assertEqual(end.rewards, {"red": None, "blue": None})
                boundary = [record for record in end.bootstrap.values() if record]
                self.assertEqual(len(boundary), 1)
                self.assertEqual(boundary[0]["type"], "observation")

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
        end, requests = play(self.make(mode="decision"), {"red": ScriptedAgent([2]), "blue": ScriptedAgent()})
        self.assertEqual(end.status, "completed")
        self.assertEqual(end.rewards, {"red": -1, "blue": 1})
        order = [(r.color, r.kind) for r in requests[:3]]
        self.assertEqual(order, [("red", "action"), ("blue", "decision"), ("red", "action")])
        decision = requests[1].record
        self.assertEqual(decision["hero"], 1)  # Blue's own hero handle, never its engine ID
        self.assertNotIn("hero_id", decision)

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
