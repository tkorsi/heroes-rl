"""Protocol fixtures for the local random-policy server, never VCMI/game evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest


SPEC = importlib.util.spec_from_file_location(
    "heroes_random_policy", Path(__file__).resolve().parents[1] / "experiments/explorer/random_policy.py")
policy_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_module)
RandomPolicy = policy_module.RandomPolicy


class ProtocolFixture:
    """An explicit protocol peer over loopback; it does not simulate game rules."""

    def __init__(self, **options):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.policy = RandomPolicy(seed=options.pop("seed", 1), step_delay=0, **options)
        self.environment = self.policy.start(self.output, "protocol-fixture", 0)
        self.socket = socket.create_connection(("127.0.0.1", int(self.environment["EXPLORER_RL_PORT"])), timeout=2)
        self.stream = self.socket.makefile("rb")

    @staticmethod
    def message(message_type, **fields):
        return {"type": message_type, "player": 0, "episode": "protocol-fixture", **fields}

    @classmethod
    def observation(cls, action_id=0, day=1, mask=None, **fields):
        return cls.message("observation", action_id=action_id, day=day,
                           action_mask=mask if mask is not None else [False] * 8 + [True],
                           tiles=[{"x": 3, "y": 4, "z": 0, "terrain": "grass"}],
                           x=3, y=4, z=0, **fields)

    def send(self, *messages):
        self.socket.sendall(b"".join((json.dumps(message) + "\n").encode() for message in messages))

    def ready(self):
        self.send(self.message("ready", protocol_version=3))

    def receive(self):
        line = self.stream.readline()
        if not line:
            raise AssertionError("Policy disconnected before its expected response")
        return json.loads(line)

    def wait(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.policy.snapshot()
            if predicate(state):
                return state
            time.sleep(0.01)
        raise AssertionError(f"Protocol fixture timed out: {self.policy.snapshot()}")

    def close(self):
        self.policy.close()
        self.stream.close()
        self.socket.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RandomChoiceTests(unittest.TestCase):
    def test_rejects_invalid_limits_and_preserves_existing_ledger(self):
        for options in ({"seed": True}, {"seed": 1.5}, {"max_actions": True}, {"max_actions": 1.5},
                        {"max_actions": 0}, {"max_days": True}, {"max_days": 1.5}, {"max_days": 0},
                        {"step_delay": -1}, {"step_delay": 6}, {"step_delay": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RandomPolicy(**{"seed": 1, **options})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = "existing protocol evidence\n"
            (output / "policy.jsonl").write_text(original)
            policy = RandomPolicy(1)
            try:
                with self.assertRaises((ValueError, FileExistsError)):
                    policy.start(output, "protocol-fixture", 0)
                self.assertIsNone(policy.listener)
                self.assertIsNone(policy.thread)
                self.assertEqual((output / "policy.jsonl").read_text(), original)
            finally:
                policy.close()

    def test_seeded_choices_depend_on_attempt_mask_not_map_targets(self):
        first, second = RandomPolicy(712), RandomPolicy(712)
        mask = [True, False, True, False, False, True, False, False, True]
        plain = {"action_mask": mask, "tiles": [], "x": 1, "y": 2, "resources": [0] * 7}
        different_visible_state = {"action_mask": mask, "x": 99, "y": 45,
                                   "resources": [1000] * 7,
                                   "tiles": [{"x": 99, "y": 46, "objects": ["mine", "enemy_hero"]}]}
        left = [first.choose_action(plain) for _ in range(200)]
        right = [second.choose_action(different_visible_state) for _ in range(200)]
        self.assertEqual(left, right)
        self.assertEqual(set(left), {0, 2, 5, 8})
        other = RandomPolicy(713)
        self.assertNotEqual(left, [other.choose_action(plain) for _ in range(200)])

    def test_rejects_malformed_or_unsafe_attempt_masks(self):
        for mask in (None, [], [True] * 8, [True] * 10, [1] * 9, [False] * 9):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                RandomPolicy(1).choose_action({"action_mask": mask})
        self.assertEqual(RandomPolicy(1).choose_action({"action_mask": [False] * 8 + [True]}), 8)

    def test_player_state_metrics_and_video_label_do_not_claim_learning(self):
        state = {"day": 2, "own_mines": 1, "own_towns": 1, "own_heroes": 1,
                 "completed_battles": 1, "visible_tiles": 15, "resources": [3] * 7}
        summary = RandomPolicy.observation_summary({"state": state, "tiles": [{}] * 15})
        self.assertEqual(summary, state)
        policy = RandomPolicy(1)
        self.assertIs(policy.description["learning"], False)
        self.assertIsNone(policy.description["checkpoint"])
        spec = importlib.util.spec_from_file_location(
            "heroes_random_video_test", Path(__file__).resolve().parents[1] / "experiments/recording/video.py")
        video = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(video)
        for mode, view in (("player", "FOG OF WAR"), ("omniscient", "FULL MAP")):
            source = {"kind": "random-baseline", "episode": 1, "status": "timeout", "outcome": None,
                      "outcome_player": "red", "policy": policy.description,
                      "perspective": {"mode": mode, "player": "red" if mode == "player" else None}}
            lines = video.overlay_lines(source)
            self.assertIn("RANDOM BASELINE (NO TRAINING)", lines[0])
            self.assertIn(view, lines[1])
            self.assertIn("UNKNOWN", lines[-1])
            self.assertNotIn("RL POLICY EVALUATION", " ".join(lines))


class LoopbackProtocolTests(unittest.TestCase):
    def test_fragmented_handshake_actions_dialog_and_natural_win(self):
        with ProtocolFixture() as peer:
            self.assertEqual(peer.policy.listener.getsockname()[0], "127.0.0.1")
            self.assertEqual(peer.environment["EXPLORER_RL_PLAYER"], "0")
            self.assertEqual(peer.environment["EXPLORER_RL_EPISODE"], "protocol-fixture")
            handshake = json.dumps(peer.message("ready", protocol_version=3)).encode()
            peer.socket.sendall(handshake[:13])
            peer.socket.sendall(handshake[13:] + b"\n" +
                                (json.dumps(peer.observation(mask=[False, False, True] + [False] * 5 + [True])) + "\n").encode())
            first = peer.receive()
            self.assertEqual(first, {"action_id": 0, "action": 2})
            peer.send(peer.message("decision", decision_id=0, kind="level_up",
                                   choices=[{"choice": 0, "skill": "offense"}, {"choice": 1, "skill": "wisdom"}]))
            decision = peer.receive()
            self.assertEqual(decision["decision_id"], 0)
            self.assertIn(decision["choice"], (0, 1))
            peer.send(peer.message("action_result", **first, accepted=True,
                                   before={"x": 3, "y": 4, "z": 0}, after={"x": 4, "y": 4, "z": 0}),
                      peer.observation(action_id=1))
            second = peer.receive()
            self.assertEqual(second, {"action_id": 1, "action": 8})
            terminal_state = {"day": 2, "own_heroes": 1, "own_towns": 2, "visible_tiles": 12}
            terminal_events = [{"kind": "battle_complete"}]
            peer.send(peer.message("action_result", **second, accepted=True,
                                   before={"x": 3, "y": 4, "z": 0}, after={"x": 3, "y": 4, "z": 0}),
                      peer.message("turn_end"), peer.message("terminal", outcome="win",
                                                               state=terminal_state, events=terminal_events))
            peer.socket.shutdown(socket.SHUT_WR)
            peer.policy.thread.join(timeout=3)
            self.assertFalse(peer.policy.thread.is_alive())
            state = peer.policy.snapshot()
            self.assertIsNone(state["error"])
            self.assertEqual(state["terminal"], {"outcome": "win",
                                                "state": terminal_state, "events": terminal_events})
            self.assertEqual((state["actions_sent"], state["actions_accepted"], state["decisions_sent"]), (2, 2, 1))
            self.assertEqual((state["successful_displacements"], state["completed_turns"]), (1, 1))
            ledger = [json.loads(line) for line in (peer.output / "policy.jsonl").read_text().splitlines()]
            self.assertEqual([entry["message"] for entry in ledger if entry["direction"] == "policy"],
                             [first, decision, second])
            self.assertEqual(ledger[-1]["message"]["type"], "terminal")
            elapsed = [entry["elapsed_seconds"] for entry in ledger]
            self.assertEqual(elapsed, sorted(elapsed))

    def test_terminal_is_reported_without_assigning_reward(self):
        with ProtocolFixture() as peer:
            peer.ready()
            peer.send(peer.message("terminal", outcome="loss"))
            state = peer.wait(lambda state: state["terminal"] is not None)
            self.assertEqual(state["terminal"]["outcome"], "loss")
            self.assertNotIn("reward", state["terminal"])
            self.assertIsNone(state["error"])
            self.assertEqual(state["actions_sent"], 0)

    def test_action_and_game_day_limits_remain_unscored(self):
        with ProtocolFixture(max_actions=1) as peer:
            peer.ready()
            peer.send(peer.observation())
            response = peer.receive()
            peer.send(peer.message("action_result", **response, accepted=True, before={}, after={}),
                      peer.message("turn_end"), peer.observation(action_id=1))
            state = peer.wait(lambda state: state["limit"] is not None)
            self.assertEqual(state["limit"], "agent_action_limit")
            self.assertEqual(state["actions_sent"], 1)
            self.assertIsNone(state["terminal"])
            self.assertIsNone(state["error"])
            self.assertEqual(peer.policy.stop_reason, "agent_action_limit")
        with ProtocolFixture(max_days=1) as peer:
            peer.ready()
            peer.send(peer.observation(day=2))
            state = peer.wait(lambda state: state["limit"] is not None)
            self.assertEqual(state["limit"], "game_day_limit")
            self.assertEqual(state["actions_sent"], 0)
            self.assertIsNone(state["terminal"])
            self.assertIsNone(state["error"])

    def assert_protocol_error(self, messages, expected):
        with ProtocolFixture() as peer:
            peer.send(*messages)
            state = peer.wait(lambda state: state["error"] is not None)
            self.assertIn(expected, state["error"])
            self.assertIsNone(state["terminal"])
            self.assertEqual(peer.policy.stop_reason, "policy_error")

    def test_rejects_wrong_identity_and_handshake_before_play(self):
        ready = ProtocolFixture.message("ready", protocol_version=3)
        cases = [([dict(ready, player=1)], "identity"),
                 ([dict(ready, player=False)], "identity"),
                 ([dict(ready, episode="another-run")], "identity"),
                 ([ProtocolFixture.observation()], "ready"),
                 ([dict(ready, protocol_version=1)], "handshake"),
                 ([dict(ready, protocol_version=2)], "handshake"),
                 ([ready, ready], "handshake")]
        for messages, expected in cases:
            with self.subTest(messages=messages):
                self.assert_protocol_error(messages, expected)

    def test_rejects_out_of_order_observations_and_results(self):
        ready = ProtocolFixture.message("ready", protocol_version=3)
        observation = ProtocolFixture.observation()
        cases = [([ready, ProtocolFixture.observation(action_id=1)], "sequence"),
                 ([ready, observation, observation], "sequence"),
                 ([ready, observation, ProtocolFixture.message("action_result", action_id=9, action=8, accepted=True)], "match"),
                 ([ready, observation, ProtocolFixture.message("action_result", action_id=0, action=8, accepted=1)], "acceptance"),
                 ([ready, observation, ProtocolFixture.message("action_result", action_id=False, action=8, accepted=True)], "match")]
        for messages, expected in cases:
            with self.subTest(messages=messages):
                self.assert_protocol_error(messages, expected)

    def test_rejects_invalid_decisions_and_repeated_decision_ids(self):
        ready = ProtocolFixture.message("ready", protocol_version=3)
        decision = ProtocolFixture.message("decision", decision_id=0, choices=[{"choice": 1}, {"choice": 2}])
        for messages, expected in (([ready, dict(decision, choices=[])], "choices"),
                                   ([ready, dict(decision, choices=[{"choice": True}])], "choices"),
                                   ([ready, dict(decision, choices=[{"choice": 1}, {"choice": 1}])], "Duplicate"),
                                   ([ready, decision, decision], "increase")):
            with self.subTest(messages=messages):
                self.assert_protocol_error(messages, expected)

    def test_turn_end_requires_one_completed_end_turn_action(self):
        ready = ProtocolFixture.message("ready", protocol_version=3)
        observation = ProtocolFixture.observation()
        result = ProtocolFixture.message("action_result", action_id=0, action=8, accepted=True,
                                          before={}, after={})
        end = ProtocolFixture.message("turn_end")
        for messages in ([ready, end], [ready, observation, end],
                         [ready, observation, result, end, end]):
            with self.subTest(messages=messages):
                self.assert_protocol_error(messages, "turn")

    def test_bridge_errors_and_unsupported_terminal_results_are_unscored(self):
        ready = ProtocolFixture.message("ready", protocol_version=3)
        for message, expected in ((ProtocolFixture.message("error", message="fixture transport error"), "fixture transport error"),
                                  (ProtocolFixture.message("terminal", outcome="timeout"), "natural win or loss"),
                                  (ProtocolFixture.message("unknown"), "Unsupported")):
            with self.subTest(message=message):
                self.assert_protocol_error([ready, message], expected)

    def test_unexpected_disconnect_is_an_error_and_deliberate_stop_is_not(self):
        with ProtocolFixture() as peer:
            peer.ready()
            peer.wait(lambda state: state["ready"])
            peer.socket.shutdown(socket.SHUT_WR)
            state = peer.wait(lambda state: state["error"] is not None)
            self.assertIn("disconnected", state["error"])
            self.assertIsNone(state["terminal"])
        with ProtocolFixture() as peer:
            peer.ready()
            peer.wait(lambda state: state["ready"])
            peer.policy.prepare_stop()
            peer.socket.shutdown(socket.SHUT_WR)
            peer.policy.thread.join(timeout=3)
            self.assertFalse(peer.policy.thread.is_alive())
            state = peer.policy.snapshot()
            self.assertIsNone(state["error"])
            self.assertIsNone(state["terminal"])


if __name__ == "__main__":
    unittest.main()
