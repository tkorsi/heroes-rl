"""Self-play scoring, capture and loopback protocol fixtures; no gameplay evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import socket
import struct
import tempfile
import time
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


record = load_module("self_play_record", "experiments/recording/record.py")
RandomPolicy = load_module("self_play_policy", "experiments/explorer/random_policy.py").RandomPolicy


def terminal_log(red="won", blue="lost"):
    return (f"Experiment player 0 {red}. Terminal received.\n"
            f"Experiment player 1 {blue}. Terminal received.\n"
            "Experiment self-play complete. Ending game.")


class SelfPlayScoringTests(unittest.TestCase):
    def setUp(self):
        self.executions = {
            color: {"ready": True, "actions_sent": 3, "error": None, "limit": None,
                    "terminal": {"outcome": outcome}}
            for color, outcome in (("red", "win"), ("blue", "loss"))}

    def assert_unscored(self, result, status="failed"):
        self.assertEqual(result[0], status)
        self.assertEqual(result[1], {"red": None, "blue": None})
        self.assertEqual(result[3], {"red": None, "blue": None})

    def test_both_sides_need_matching_complementary_natural_results(self):
        for red, blue in (("won", "lost"), ("lost", "won")):
            outcomes = {"red": "win" if red == "won" else "loss", "blue": "win" if blue == "won" else "loss"}
            for color, outcome in outcomes.items():
                self.executions[color]["terminal"]["outcome"] = outcome
            result = record.classify_self_play_run(0, None, terminal_log(red, blue), self.executions)
            self.assertEqual(result, ("completed", outcomes, "natural_terminal",
                                      {color: 1 if outcome == "win" else -1 for color, outcome in outcomes.items()}))

    def test_missing_duplicate_or_conflicting_terminal_evidence_is_unscored(self):
        for log in ("", terminal_log().splitlines()[0],
                    terminal_log() + "\nExperiment player 0 won. Terminal received.",
                    terminal_log() + "\nExperiment player 0 lost. Terminal received.",
                    terminal_log() + "\nExperiment self-play complete. Ending game."):
            with self.subTest(log=log):
                self.assert_unscored(record.classify_self_play_run(0, None, log, self.executions))
        for terminal in (None, {"outcome": "win"}):
            changed = copy.deepcopy(self.executions)
            changed["blue"]["terminal"] = terminal
            with self.subTest(terminal=terminal):
                self.assert_unscored(record.classify_self_play_run(0, None, terminal_log(), changed))
        self.executions["blue"]["terminal"]["outcome"] = "win"
        result = record.classify_self_play_run(0, None, terminal_log("won", "won"), self.executions)
        self.assert_unscored(result)
        self.assertEqual(result[2], "noncomplementary_agent_terminals")

    def test_each_agent_must_have_acted_without_protocol_errors(self):
        self.assert_unscored(record.classify_self_play_run(0, None, terminal_log(), {"red": self.executions["red"]}))
        for color in self.executions:
            for update in ({"ready": False}, {"actions_sent": 0}, {"error": "synthetic identity mismatch"}):
                with self.subTest(color=color, update=update):
                    changed = copy.deepcopy(self.executions)
                    changed[color].update(update)
                    self.assert_unscored(record.classify_self_play_run(0, None, terminal_log(), changed))

    def test_limits_or_crashes_never_score_even_after_both_terminals(self):
        for reason in ("wall_time_limit", "frame_limit", "agent_action_limit", "game_day_limit"):
            with self.subTest(reason=reason):
                self.assert_unscored(record.classify_self_play_run(0, reason, terminal_log(), self.executions), "timeout")
                changed = copy.deepcopy(self.executions)
                changed["blue"]["limit"] = reason
                self.assert_unscored(record.classify_self_play_run(0, None, terminal_log(), changed), "timeout")
        self.assert_unscored(record.classify_self_play_run(-6, None, terminal_log(), self.executions))
        self.assert_unscored(record.classify_self_play_run(0, "recorder_error", terminal_log(), self.executions))


def synthetic_png(path):
    """Two-by-two solid pixels, explicitly not a native game frame."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = (b"\0" + b"\x40\x80\xc0" * 2) * 2
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class SelfPlayCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.metadata = {
            "schema_version": 1, "phase": "before_first_turn", "game_day": 0,
            "cheats_allowed": False, "requested_no_cheats": True, "difficulty_index": 0,
            "map_width": 4, "map_height": 4, "map_levels": 1, "requested_agent_players": [0, 1],
            "players": [{"player_index": i, "team_index": i, "human": False,
                         "adventure_ai": "ExplorerRL", "visible_tiles": 3 + 4 * i, "total_tiles": 16}
                        for i in (0, 1)]}
        self.manifest = {
            "schema_version": 3, "kind": "random-self-play", "seed": 12345,
            "status": "completed", "termination_reason": "natural_terminal",
            "outcomes_by_player": {"red": "win", "blue": "loss"},
            "rewards_by_player": {"red": 1, "blue": -1}, "agent_player_indices": [0, 1],
            "recording_valid": False, "recordings": {}}
        self.log = "Experiment engine RNG: initial_seed=12345 initial_sample=1272378730"
        self.streams = {}
        for observer, color in enumerate(("red", "blue")):
            for view in ("omniscient", "player"):
                key = f"{view}-{color}"
                directory = self.output / "frames" / key
                directory.mkdir(parents=True)
                self.manifest["recordings"][key] = {
                    "capture": {"fps": 6, "frames": 0, "directory": f"frames/{key}"},
                    "video": None, "recording_valid": False,
                    "perspective": {"mode": view, "player": color if view == "player" else None,
                                    "player_index": observer if view == "player" else None,
                                    "observer_player": color, "observer_player_index": observer}}
                entries = []
                for sample in (1, 2):
                    filename = f"frame{sample:06d}.png"
                    synthetic_png(directory / filename)
                    entries.append({"index": sample, "file": filename, "width": 2, "height": 2,
                                    "sample_id": sample, "elapsed_seconds": (sample - 1) / 6,
                                    "game_day": sample - 1, "view": view,
                                    "player_index": observer if view == "player" else None,
                                    "observer_player_index": observer, "visible_tiles": sample + 2 + 4 * observer})
                self.streams[key] = entries
        self.write_evidence()

    def tearDown(self):
        self.temp.cleanup()

    def write_evidence(self):
        (self.output / "frames" / "experiment-metadata.json").write_text(json.dumps(self.metadata))
        for key, entries in self.streams.items():
            (self.output / "frames" / key / "frames.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    def assert_capture_rejected(self, expected):
        with self.assertRaisesRegex((ValueError, OSError), expected):
            record.finalize_recordings(self.manifest, self.output, 0, self.log, no_encode=True)
        saved = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["recording_valid"])
        self.assertEqual(saved["outcomes_by_player"], {"red": None, "blue": None})
        self.assertEqual(saved["rewards_by_player"], {"red": None, "blue": None})
        self.assertTrue(all(not stream["recording_valid"] for stream in saved["recordings"].values()))

    def test_effective_routing_requires_two_explorer_agents_on_opposing_teams(self):
        evidence = record.validate_experiment_metadata(self.metadata, 0, self.log, agent_players=[0, 1])
        self.assertEqual(evidence["ai_players"], [])
        self.assertEqual(evidence["initial_visibility_by_player"], {"red": 3, "blue": 7})
        for field, value in (("adventure_ai", "Nullkiller"), ("human", True), ("team_index", 0)):
            changed = copy.deepcopy(self.metadata)
            changed["players"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                record.validate_experiment_metadata(changed, 0, self.log, agent_players=[0, 1])
        changed = copy.deepcopy(self.metadata)
        changed["requested_agent_players"] = [0]
        with self.assertRaises(ValueError):
            record.validate_experiment_metadata(changed, 0, self.log, agent_players=[0, 1])
        with self.assertRaises(ValueError):
            record.validate_experiment_metadata(self.metadata, 0, self.log + "\nExperiment AI visibility: player=1 cheats_allowed=0 open_map=0 use_object_graph=0",
                                                agent_players=[0, 1])

    def test_all_four_streams_validate_without_equating_player_fog(self):
        record.finalize_recordings(self.manifest, self.output, 0, self.log, no_encode=True)
        self.assertTrue(self.manifest["recording_valid"])
        self.assertEqual(self.manifest["effective_engine_rng"]["initial_seed"], 12345)
        self.assertEqual(set(self.manifest["paired_capture_validation_by_player"]), {"red", "blue"})
        self.assertEqual(self.manifest["synchronized_capture_validation"]["frames"], 2)
        for key, stream in self.manifest["recordings"].items():
            self.assertTrue(stream["recording_valid"])
            self.assertIsNone(stream["video"])
            self.assertEqual(stream["capture"]["initial_visible_tiles"], 3 if key.endswith("red") else 7)

    def test_missing_stream_invalidates_both_rewards(self):
        del self.manifest["recordings"]["player-blue"]
        self.assert_capture_rejected("both perspectives")

    def test_missing_blue_frame_invalidates_both_rewards(self):
        (self.output / "frames" / "player-blue" / "frame000002.png").unlink()
        self.assert_capture_rejected("frame000002")

    def test_cross_observer_time_mismatch_is_rejected_even_when_each_pair_matches(self):
        for key in ("omniscient-blue", "player-blue"):
            for entry in self.streams[key]:
                entry["elapsed_seconds"] += 0.01
        self.write_evidence()
        self.assert_capture_rejected("timestamps")

    def test_red_streams_cannot_be_replaced_with_consistently_labelled_blue_streams(self):
        for view in ("omniscient", "player"):
            key = f"{view}-red"
            self.manifest["recordings"][key]["perspective"].update(
                observer_player="blue", observer_player_index=1,
                player="blue" if view == "player" else None, player_index=1 if view == "player" else None)
            for entry in self.streams[key]:
                entry.update(observer_player_index=1, player_index=1 if view == "player" else None,
                             visible_tiles=7)
        self.write_evidence()
        self.assert_capture_rejected("identity|observer|perspective")

    def test_authoritative_engine_seed_must_match_before_scoring(self):
        self.log = self.log.replace("initial_seed=12345", "initial_seed=54321")
        self.assert_capture_rejected("engine seed")


class TwoPolicyPeers:
    """Two real loopback connections with synthetic protocol messages only."""
    def __init__(self, root):
        self.root = root
        self.policies, self.sockets, self.readers = {}, {}, {}
        for player in (0, 1):
            output = root / str(player)
            output.mkdir()
            policy = RandomPolicy(17 + player, step_delay=0)
            environment = policy.start(output, "synthetic-self-play", player)
            peer = socket.create_connection(("127.0.0.1", int(environment["EXPLORER_RL_PORT"])), timeout=3)
            self.policies[player], self.sockets[player] = policy, peer
            self.readers[player] = peer.makefile("rb")

    def send(self, player, kind, **fields):
        message = {"type": kind, "episode": "synthetic-self-play", "player": player, **fields}
        self.sockets[player].sendall((json.dumps(message) + "\n").encode())

    def action(self, player, action_id):
        self.send(player, "observation", action_id=action_id, day=1, x=player, y=0, z=0,
                  tiles=[{"synthetic_owner": player}], action_mask=[True] * 9,
                  state={"visible_tiles": 3 + player, "own_heroes": 1})
        response = json.loads(self.readers[player].readline())
        self.send(player, "action_result", action_id=action_id, action=response["action"], accepted=True, after={})
        if response["action"] == 8:
            self.send(player, "turn_end")
        return response["action"]

    def wait(self, player, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.policies[player].snapshot()
            if predicate(state):
                return state
            time.sleep(0.005)
        raise AssertionError(f"Policy {player} did not reach the expected state: {state}")

    def close(self):
        for policy in self.policies.values():
            policy.close()
        for reader in self.readers.values():
            reader.close()
        for peer in self.sockets.values():
            peer.close()


class IndependentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.peers = TwoPolicyPeers(Path(self.temp.name))

    def tearDown(self):
        self.peers.close()
        self.temp.cleanup()

    def test_interleaved_agents_keep_rng_observations_and_ledgers_separate(self):
        for player in (0, 1):
            self.peers.send(player, "ready", protocol_version=2)
        expected = {player: random.Random(17 + player) for player in (0, 1)}
        counts = {0: 0, 1: 0}
        for player in (0, 0, 1, 0, 1, 1):
            actual = self.peers.action(player, counts[player])
            self.assertEqual(actual, expected[player].choice(list(range(9))))
            counts[player] += 1
        for player, outcome in ((0, "win"), (1, "loss")):
            self.peers.send(player, "terminal", outcome=outcome, state={"own_heroes": 1 - player})
            state = self.peers.wait(player, lambda state: state["terminal"] is not None)
            self.assertEqual(state["actions_sent"], 3)
            self.assertEqual(state["terminal"]["outcome"], outcome)
            self.assertEqual(state["initial_observation"]["visible_tiles"], 3 + player)
            ledger = [json.loads(line) for line in (Path(self.temp.name) / str(player) / "policy.jsonl").read_text().splitlines()]
            self.assertTrue(all(entry["message"]["player"] == player for entry in ledger if entry["direction"] == "engine"))

    def test_wrong_player_message_fails_only_its_connection(self):
        self.peers.sockets[0].sendall((json.dumps({"type": "ready", "episode": "synthetic-self-play",
                                                  "player": 1, "protocol_version": 2}) + "\n").encode())
        red = self.peers.wait(0, lambda state: state["error"] is not None)
        self.assertIn("identity", red["error"])
        self.peers.send(1, "ready", protocol_version=2)
        self.peers.action(1, 0)
        self.peers.send(1, "terminal", outcome="loss")
        blue = self.peers.wait(1, lambda state: state["terminal"] is not None)
        self.assertIsNone(blue["error"])
        self.assertEqual(blue["actions_sent"], 1)
        self.assertIsNone(red["terminal"])

    def test_freezing_evidence_stops_delayed_writer_before_hashing(self):
        policy = self.peers.policies[0]
        policy.step_delay = 2.5
        self.peers.send(0, "ready", protocol_version=2)
        self.peers.send(0, "observation", action_id=0, day=1, tiles=[],
                        action_mask=[False] * 8 + [True], x=0, y=0, z=0)
        self.peers.wait(0, lambda state: state["actions_chosen"] == 1)
        policy.prepare_stop()
        self.peers.sockets[0].shutdown(socket.SHUT_RDWR)
        snapshot, ledger_hash = record.freeze_policy_evidence(policy)
        self.assertFalse(policy.thread.is_alive())
        self.assertEqual(snapshot["actions_chosen"], 1)
        self.assertEqual(snapshot["actions_sent"], 0)
        self.assertIsNone(snapshot["terminal"])
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot, policy.snapshot())
        self.assertEqual(ledger_hash, record.digest(policy.path))
        # The interrupted response must not appear in the frozen action ledger.
        entries = [json.loads(line) for line in policy.path.read_text().splitlines()]
        self.assertTrue(all(entry["direction"] == "engine" for entry in entries))

    def test_freezing_evidence_drains_delivered_natural_terminal(self):
        self.peers.send(1, "ready", protocol_version=2)
        self.peers.action(1, 0)
        self.peers.send(1, "terminal", outcome="win", state={"own_heroes": 1})
        self.peers.sockets[1].shutdown(socket.SHUT_WR)
        policy = self.peers.policies[1]
        snapshot, ledger_hash = record.freeze_policy_evidence(policy)
        self.assertFalse(policy.thread.is_alive())
        self.assertEqual(snapshot["terminal"]["outcome"], "win")
        self.assertEqual(snapshot["actions_sent"], 1)
        self.assertIsNone(snapshot["error"])
        self.assertEqual(ledger_hash, record.digest(policy.path))
        entries = [json.loads(line) for line in policy.path.read_text().splitlines()]
        self.assertEqual(entries[-1]["message"]["type"], "terminal")


if __name__ == "__main__":
    unittest.main()
