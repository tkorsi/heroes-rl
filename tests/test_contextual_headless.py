"""The opt-in adventure-windows-v1 environment against a miniature fake engine.

The fake (tests/fixtures/headless/fake_contextual_engine.py) speaks bridge
protocol 4 like the native contextual interface. These tests establish the
adapter's contract handling: requests, history-v2 production, prefix
immutability, cutoffs, failures and the manifest the independent auditor binds.
They prove nothing about VCMI gameplay; the native loop needs the Mac acceptance.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/headless"))
sys.path.insert(0, str(ROOT / "validation/probes"))

import audit_contextual_histories as auditor  # noqa: E402
import env as env_module  # noqa: E402
import replay  # noqa: E402
from bridge_protocol import ProtocolError  # noqa: E402
from contextual_bridge import ContextualPlayer  # noqa: E402
from experiments.learner import contextual as contract  # noqa: E402

FAKE_ENGINE = ROOT / "tests/fixtures/headless/fake_contextual_engine.py"
Action = contract.Action


class FakeContextualLauncher:
    def __init__(self, game_map, **scenario):
        self.scenario, self.game_map = scenario, game_map

    def launch(self, directory, seed, environment, log):
        (directory / "engine").mkdir(parents=True, exist_ok=True)
        inherited = {k: v for k, v in os.environ.items() if not k.startswith(("EXPLORER_RL_", "VCMI_"))}
        launch_env = {**environment, "VCMI_EXPERIMENT_DIR": str(directory / "engine"),
                      "FAKE_ENGINE_SCENARIO": json.dumps(self.scenario), "FAKE_ENGINE_SEED": str(seed)}
        process = subprocess.Popen([sys.executable, str(FAKE_ENGINE)], env={**inherited, **launch_env},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process, {"command": ["fake_contextual_engine.py"], "environment": launch_env}

    def identity(self):
        return {"fake_engine": True}


def observation(request):
    return contract.decode_frame(request.frames[-1]).observation


def end_turn(obs):
    return Action("close") if obs.context.kind != "map" else Action("end_turn")


class Script:
    """Per-color action lists; each entry maps the current observation to an action. Then end turns."""

    def __init__(self, red=(), blue=()):
        self.plans = {"red": list(red), "blue": list(blue)}

    def __call__(self, request):
        obs = observation(request)
        plan = self.plans[request.color]
        return plan.pop(0)(obs) if plan else end_turn(obs)


class ContextualEnvTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="heroes-contextual-test-")
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.game_map = self.output / "fake.vmap"
        self.game_map.write_bytes(b"fake map")

    def make(self, limits=None, **scenario):
        env = env_module.HeadlessSelfPlayEnv(
            FakeContextualLauncher(self.game_map, **scenario), self.output / f"run-{len(list(self.output.iterdir()))}",
            limits or env_module.Limits(watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10),
            interface=contract.ENVIRONMENT)
        self.addCleanup(env.close)
        return env

    def play(self, env, policy, episode=1, seed=7):
        state, requests = env.reset(episode=episode, seed=seed), []
        while not state.done:
            requests.append(state)
            self.check_request(env, state)
            state = env.step(policy(state))
        return state, requests

    def check_request(self, env, request):
        self.assertEqual(request.reward, 0)
        frames = [contract.decode_frame(raw) for raw in request.frames]
        self.assertTrue(all(frame.player == request.player for frame in frames))
        self.assertEqual(frames[-1].kind, "decision")
        self.assertEqual(env.history_v2(request.color)[-1], request.frames[-1])
        self.assertEqual(frames[-1].offers, contract.Offers.from_observation(frames[-1].observation))

    def audit(self, env):
        directory = env.episode_dir
        return auditor.audit(directory / "agents/red/history-v2.jsonl", directory / "agents/blue/history-v2.jsonl",
                             directory / "manifest.json")

    def frames(self, env, color):
        return [contract.decode_frame(raw) for raw in env.history_v2(color)]

    def test_default_interface_is_unchanged_and_others_are_refused(self):
        movement = env_module.HeadlessSelfPlayEnv(FakeContextualLauncher(self.game_map), self.output / "movement")
        self.assertIs(type(movement), env_module.HeadlessSelfPlayEnv)
        self.assertEqual(movement.interface, "movement-v1")
        with self.assertRaises(ValueError):
            movement.history_v2("red")
        with self.assertRaises(ValueError):
            env_module.HeadlessSelfPlayEnv(FakeContextualLauncher(self.game_map), self.output / "x", interface="windows")
        self.assertIsInstance(self.make(), env_module.ContextualSelfPlayEnv)

    def test_natural_episode_navigation_visit_and_audit(self):
        hero = lambda obs: obs.heroes[0].handle  # noqa: E731
        script = Script(red=[
            lambda obs: Action("open_hero", hero(obs)),
            lambda obs: Action("close"),
            lambda obs: Action("open_town", obs.towns[0].handle),
            lambda obs: Action("close"),
            lambda obs: Action("move_to", hero(obs), 3, 1),  # the gold pile two steps east
            lambda obs: Action("end_turn"),
            lambda obs: Action("move_to", hero(obs), *obs.heroes[0].position[:2]),
            lambda obs: Action("move_to", hero(obs), 11, 9),  # never seen
        ])
        env = self.make()
        end, requests = self.play(env, script)
        self.assertEqual((end.status, end.termination_reason), ("completed", "natural_terminal"))
        self.assertEqual(end.rewards, {"red": 1, "blue": -1})
        red = self.frames(env, "red")
        contexts = [frame.observation.context for frame in red if frame.kind == "decision"]
        self.assertEqual([c.kind for c in contexts[:5]], ["map", "hero", "map", "town", "map"])
        windows = [frame.observation.window for frame in red if frame.kind == "decision" and frame.observation.window]
        self.assertEqual(windows[0].level, 1)
        self.assertEqual(windows[1].faction, 2)
        results = [frame.result for frame in red if frame.kind == "result"]
        self.assertEqual([r.status for r in results[:5]], ["opened", "closed", "opened", "closed", "interaction"])
        self.assertEqual(results[6].status, "reached")
        self.assertEqual(results[7].status, "no_route_known")
        # The visit's cause precedes its result, and no command follows the stop cause.
        visit = next(i for i, frame in enumerate(red) if frame.kind == "event" and frame.event.kind == "notice")
        result = next(i for i, frame in enumerate(red) if frame.kind == "result" and frame.result.status == "interaction")
        self.assertLess(visit, result)
        self.assertFalse(any(frame.kind == "issued" for frame in red[visit:result]))
        self.assertEqual(sum(frame.kind == "issued" for frame in red[:result]), 2)
        notice = red[visit].event.notice
        self.assertEqual(notice.components[0].resource, "gold")
        # Routing handles only: no engine identifier reaches a frame.
        self.assertNotIn('"id"', json.dumps(env.history_v2("red")))
        # Every delivered prefix stays exactly as delivered.
        for color in ("red", "blue"):
            delivered = [frame for request in requests if request.color == color for frame in request.frames]
            delivered += end.final_frames[color]
            self.assertEqual(json.loads(json.dumps(delivered)), env.history_v2(color))
        report = self.audit(env)
        self.assertTrue(report["consistent"] and report["manifest_bound"] and report["scoring_binding_verified"],
                        report["failures"])
        self.assertTrue(report["learner_eligible"])
        manifest = end.manifest
        self.assertEqual(manifest["kind"], "contextual-headless-self-play")
        self.assertEqual(manifest["contextual_replay"]["status"], "unsupported")
        self.assertFalse(manifest["deterministic_replay_verified"])
        with self.assertRaisesRegex(replay.ReplayError, "not supported"):
            replay.manifest_labels(manifest, env.episode_dir)

    def test_rejected_step_interrupts_the_route(self):
        hero = lambda obs: obs.heroes[0].handle  # noqa: E731
        env = self.make(mode="reject_step")
        end, _ = self.play(env, Script(red=[lambda obs: Action("move_to", hero(obs), 3, 3)]))
        self.assertEqual(end.status, "completed")
        red = self.frames(env, "red")
        acknowledgements = [frame.accepted for frame in red if frame.kind == "acknowledged"]
        self.assertEqual(acknowledgements[:2], [True, False])
        first = next(frame.result for frame in red if frame.kind == "result")
        self.assertEqual((first.status, first.accepted), ("interrupted", True))
        self.assertTrue(self.audit(env)["consistent"])

    def test_terminal_during_a_route_carries_its_result(self):
        hero = lambda obs: obs.heroes[0].handle  # noqa: E731
        env = self.make(mode="terminal_mid_route")
        end, _ = self.play(env, Script(red=[lambda obs: Action("end_turn"), lambda obs: Action("move_to", hero(obs), 3, 3)]))
        self.assertEqual((end.status, end.rewards), ("completed", {"red": -1, "blue": 1}))
        red = self.frames(env, "red")
        kinds = [frame.kind for frame in red]
        self.assertEqual(kinds[-4:], ["acknowledged", "result", "terminal", "finalized"])
        transitions = env.sides[0].history.transitions()
        self.assertEqual((transitions[-1].reward, transitions[-1].successor_seq), (-1, None))
        self.assertTrue(self.audit(env)["consistent"])

    def test_day_horizon_cuts_each_player_at_its_own_decision(self):
        env = self.make(env_module.Limits(max_days=1, watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10))
        end, requests = self.play(env, Script())
        self.assertEqual((end.status, end.termination_reason), ("truncated", "game_day_limit"))
        self.assertEqual(end.rewards, {"red": None, "blue": None})
        self.assertEqual([request.color for request in requests], ["red", "blue"])
        for color in ("red", "blue"):
            frames = self.frames(env, color)
            self.assertEqual([frame.kind for frame in frames[-3:]], ["decision", "cutoff", "cutoff_finalized"])
            self.assertEqual(end.bootstrap[color]["kind"], "decision")
            self.assertEqual(frames[-3].day, 2)
        self.assertEqual(end.manifest["players"]["red"]["continuation_steps"], 1)
        report = self.audit(env)
        self.assertTrue(report["consistent"] and report["manifest_bound"], report["failures"])
        self.assertTrue(all(player["boundary"] == "cutoff" for player in report["players"].values()))

    def test_declared_collection_window_and_invalid_actions(self):
        env = self.make()
        state = env.reset(episode=1, seed=3)
        before = env.history_v2("red")
        obs = observation(state)
        for invalid in (Action("move_to", obs.heroes[0].handle, obs.size[0], 0), Action("close"), "end_turn",
                        Action("open_town", obs.heroes[0].handle)):
            with self.assertRaises(ValueError):
                env.step(invalid)
            self.assertIs(env.request, state)
            self.assertEqual(env.history_v2("red"), before)
        state = env.step(Action("open_hero", obs.heroes[0].handle))
        with self.assertRaises(ValueError):
            env.step(Action("end_turn"))  # a window offers close only
        end = env.truncate("collection_window")
        self.assertEqual((end.status, end.termination_reason), ("truncated", "collection_window"))
        red = self.frames(env, "red")
        self.assertEqual(red[-1].kind, "cutoff_finalized")
        self.assertEqual(red[-3].observation.context.kind, "hero")
        self.assertTrue(self.audit(env)["consistent"])

    def test_unsupported_interaction_fails_both_histories(self):
        env = self.make(mode="unsupported")
        end, _ = self.play(env, Script(red=[lambda obs: Action("open_town", obs.towns[0].handle)]))
        self.assertEqual((end.status, end.termination_reason), ("failed", "unsupported_interaction"))
        self.assertEqual(end.rewards, {"red": None, "blue": None})
        for color in ("red", "blue"):
            self.assertEqual(self.frames(env, color)[-1].kind, "failure")
        report = self.audit(env)
        self.assertTrue(report["consistent"], report["failures"])
        self.assertFalse(report["learner_eligible"])

    def test_engine_exit_is_observed_before_a_response_is_sent(self):
        env = self.make(mode="hang_after_decision")
        state = env.reset(episode=1, seed=3)
        while contract.decode_frame(state.frames[-1]).day < 2 or state.color != "red":
            state = env.step(end_turn(observation(state)))
        os.kill(env.process.pid, signal.SIGKILL)
        env.process.wait(timeout=10)
        end = env.step(Action("end_turn"))
        self.assertEqual(end.status, "failed")
        red = self.frames(env, "red")
        self.assertEqual(red[-1].kind, "failure")
        self.assertEqual(red[-2].kind, "decision")  # never submitted: the engine was already gone
        self.assertTrue(self.audit(env)["consistent"])

    def test_fold_disagreeing_with_the_decision_state_is_a_protocol_error(self):
        env = self.make(mode="leak")
        end, _ = self.play(env, Script())
        self.assertEqual((end.status, end.termination_reason), ("failed", "protocol_error"))
        self.assertIn("fold disagrees", end.manifest["failure_detail"])


class ContextualPlayerTests(unittest.TestCase):
    def ready(self, player):
        player.check({"type": "ready", "protocol_version": 4, "player": 0, "episode": "1",
                      "interface": "adventure-windows-v1", "action_schema": "heroes-context-actions-v2",
                      "observation_events": "context_update-v1", "routing": "player-knowledge-current-turn-v1",
                      "notices": "audited-notice-v1", "battle_ai": "BattleAI", "profiling": False,
                      "snapshot_every": 0, "map_size": [4, 4, 1]})

    def update(self, sequence, **fields):
        event = {"kind": "context_update", "sequence": sequence, "day": 1, "appeared": [], "changed": [],
                 "disappeared": [], "revealed": [], "tiles_changed": [], "hidden": [], "own": None}
        event.update(fields)
        return event

    def first(self):
        own = {"resources": [0] * 7, "heroes": [{"id": 7, "position": [1, 1, 0], "movement": 5, "movement_max": 5}],
               "towns": []}
        return self.update(0, revealed=[{"x": 1, "y": 1, "z": 0, "terrain": 1, "road": 0, "blocked": False,
                                         "visitable": True}],
                           appeared=[{"id": 7, "x": 1, "y": 1, "z": 0, "type": 34, "subtype": 2, "owner": 0}], own=own)

    def decision(self, events, decision_id=0, revision=1):
        return {"type": "decision", "protocol_version": 4, "player": 0, "episode": "1", "decision_id": decision_id,
                "revision": revision, "context": {"kind": "map", "subject": None}, "day": 1, "events": events}

    def test_unanswered_decision_is_superseded_before_terminal_events(self):
        player = ContextualPlayer(0, "red", "1")
        self.ready(player)
        player.check(self.decision([self.first()]))
        player.check({"type": "terminal", "protocol_version": 4, "player": 0, "episode": "1", "outcome": "loss",
                      "events": [{"kind": "battle_start", "side": "defender"}]})
        kinds = [frame.kind for frame in player.history.frames]
        self.assertEqual(kinds, ["view", "decision", "superseded", "event", "terminal"])
        self.assertEqual(player.history.frames[0].observation.heroes[0].handle, "object-1")

    def test_stream_violations_are_protocol_errors(self):
        player = ContextualPlayer(0, "red", "1")
        self.ready(player)
        with self.assertRaises(ProtocolError):
            player.check(self.decision([self.update(1)]))  # sequence gap
        player = ContextualPlayer(0, "red", "1")
        self.ready(player)
        with self.assertRaises(ProtocolError):
            player.check(self.decision([self.first(), {"kind": "issued", "command_index": 0}]))  # no action in flight
        player = ContextualPlayer(0, "red", "1")
        self.ready(player)
        player.check(self.decision([self.first()]))
        with self.assertRaises(ProtocolError):
            player.check({"type": "result", "protocol_version": 4, "player": 0, "episode": "1", "decision_id": 0,
                          "events": [], "result": {"status": "turn_ended", "accepted": True}})  # never answered


if __name__ == "__main__":
    unittest.main()
