"""Replay-verification tooling: observation chain, log trimming, tamper cases, verifier plumbing.

The fake engine and the stub verifier exercise the Python side only. They do
not replay VCMI and prove nothing about real-game determinism.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_headless import FakeLauncher, ScriptedAgent, agents, env_module, play  # noqa: E402

SPEC = importlib.util.spec_from_file_location("headless_replay_tool", ROOT / "experiments/headless/replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

STUB_VERIFIER = '''#!{python}
"""Stub with the vcmireplay interface ("heroes-replay-verification-v1"); not VCMI."""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
value = lambda name: args[args.index(name) + 1]
home = os.environ.get("VCMI_USER_DATA_DIR") or os.path.join(os.environ["XDG_DATA_HOME"], "vcmi")
assert Path(home, "Maps", "headless.vmap").is_file() and Path(home, "Data").is_dir()
records = [json.loads(line) for line in open(value("--log"))]
terminals, boundaries = [], {{}}
for record in records:
    if record["kind"] == "applied" and record["boundary"]:
        boundaries[record["boundary"]] = boundaries.get(record["boundary"], 0) + 1
        terminals += record["terminals"]
diverged = any(record.get("tampered") for record in records)
report = {{"schema": "heroes-replay-verification-v1", "verified": not diverged,
          "divergence": {{"stage": "applied 1", "field": "digest", "expected": "x", "actual": "y"}} if diverged else None,
          "summary": {{"inputs": (len(records) - 2) // 2, "prefix": "--prefix" in args, "terminals": terminals,
                      "boundaries": boundaries}}}}
Path(value("--report")).write_text(json.dumps(report))
sys.exit(1 if diverged else 0)
'''


def replay_log(terminals=True, pairs=4):
    records = [{"kind": "init", "schema": replay.SCHEMA, "seed": 7, "digest": "00"},
               {"kind": "start", "out": 0, "client_out": 0, "chain": "00", "digest": "00"}]
    for seq in range(pairs):
        last = seq == pairs - 1
        records.append({"kind": "input", "seq": seq, "connection": 1, "player": seq % 2, "request": seq, "type": 5,
                        "out": seq, "pack": f"0a0b{seq:02x}"})
        records.append({"kind": "applied", "seq": seq, "blocked": False, "result": True, "out": seq + 1,
                        "client_out": seq + 1, "chain": "11",
                        "boundary": "game_end" if last and terminals else ("end_turn" if seq % 2 else None),
                        "digest": "22" if (last and terminals) or seq % 2 else None,
                        "terminals": [{"player": 0, "victory": True}, {"player": 1, "victory": False}]
                        if last and terminals else []})
    return records


class ReplayToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-replay-test-")
        self.output = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def episode(self, *, replay_enabled=True, limits=None, **scenario):
        env = env_module.HeadlessSelfPlayEnv(
            FakeLauncher(**scenario), self.output / f"run-{len(list(self.output.iterdir()))}",
            limits or env_module.Limits(watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10),
            replay=replay_enabled)
        self.addCleanup(env.close)
        end, _ = play(env, {"red": agents.UniformRandomAgent(1), "blue": agents.UniformRandomAgent(2)})
        return env, end

    def test_received_updates_match_the_engine_capture_digests(self):
        env, end = self.episode(mode="natural", end_day=3)
        self.assertEqual(end.status, "completed")
        self.assertEqual(env.launch_info["environment"]["VCMI_EXPERIMENT_REPLAY"], "1")
        files = end.manifest["replay"]["files"]
        self.assertEqual(sorted(files), ["engine/observations-p0.jsonl", "engine/observations-p1.jsonl"])
        self.assertEqual(end.manifest["replay"]["missing"], ["engine/replay.jsonl"])  # the fake has no server log
        for index, color in enumerate(("red", "blue")):
            check = replay.cross_check_player(env.episode_dir / "agents" / color / "bridge.jsonl",
                                              env.episode_dir / "engine" / f"observations-p{index}.jsonl", cut_short=False)
            self.assertTrue(check["verified"], check)
            self.assertGreater(check["captures_checked"], 2)
            self.assertGreater(check["messages_checked"], check["captures_checked"] // 2)

    def test_an_altered_received_update_or_engine_digest_breaks_the_chain(self):
        env, _ = self.episode(mode="natural", end_day=3)
        bridge = env.episode_dir / "agents" / "red" / "bridge.jsonl"
        ledger = env.episode_dir / "engine" / "observations-p0.jsonl"
        entries = [json.loads(line) for line in bridge.read_text().splitlines()]
        for mutate in ("object", "own", "hidden"):
            changed = json.loads(json.dumps(entries))
            done = False
            for entry in changed:
                for event in entry["message"].get("events", []) if entry["direction"] == "engine" else []:
                    if done or event.get("kind") != "visible_update":
                        continue
                    if mutate == "object" and event["appeared"]:
                        event["appeared"][0]["owner"] += 1
                        done = True
                    elif mutate == "own" and event["own"]:
                        event["own"]["resources"][0] += 1
                        done = True
                    elif mutate == "hidden" and event["revealed"]:
                        event["revealed"][0]["terrain"] += 1
                        done = True
            self.assertTrue(done, mutate)
            path = self.output / f"bridge-{mutate}.jsonl"
            replay.write_jsonl(path, changed)
            self.assertFalse(replay.cross_check_player(path, ledger, cut_short=False)["verified"], mutate)
        lines = [json.loads(line) for line in ledger.read_text().splitlines()]
        lines[-1]["digest"] = "0" * 16
        path = self.output / "observations-tampered.jsonl"
        replay.write_jsonl(path, lines)
        self.assertFalse(replay.cross_check_player(bridge, path, cut_short=False)["verified"])
        # A complete episode cannot have engine evidence the policy never received.
        lines = [json.loads(line) for line in ledger.read_text().splitlines()]
        lines.append({"kind": "message", "type": "terminal", "index": sum(l["kind"] == "message" for l in lines),
                      "pack": 0, "pack_type": 0, "digest": lines[-1]["digest"]})
        replay.write_jsonl(path, lines)
        self.assertFalse(replay.cross_check_player(bridge, path, cut_short=False)["verified"])
        self.assertTrue(replay.cross_check_player(bridge, path, cut_short=True)["verified"])

    def test_cut_short_recordings_keep_only_complete_records(self):
        path = self.output / "log.jsonl"
        path.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
        with self.assertRaises(replay.ReplayError):
            replay.read_jsonl(path, allow_partial_tail=False)
        self.assertEqual(replay.read_jsonl(path, allow_partial_tail=True), ([{"a": 1}, {"b": 2}], 5))
        records = replay_log()
        trimmed, dropped = replay.complete_replay_records(records + [records[2]])
        self.assertEqual((len(trimmed), dropped), (len(records), 1))
        with self.assertRaises(replay.ReplayError):
            replay.complete_replay_records(records[1:])

    def test_every_tamper_case_changes_the_recording(self):
        records = replay_log(pairs=6)
        observations = {"observations-p0.jsonl": [{"kind": "capture", "sequence": 0, "pack": 1, "pack_type": 3,
                                                   "digest": "ab" * 8}]}
        cases = replay.tamper_cases(records, observations)
        self.assertEqual({name for name, *_ in cases}, {"seed", "pack_byte", "state_digest", "drop_input", "swap_inputs",
                                                        "observation_digest_observations-p0"})
        for name, _, changed_records, changed_observations in cases:
            self.assertTrue(changed_records != records or changed_observations != observations, name)
        dropped = next(changed for name, _, changed, _ in cases if name == "drop_input")
        inputs = [record["seq"] for record in dropped[2:] if record["kind"] == "input"]
        self.assertEqual(inputs, list(range(len(inputs))))  # renumbered: only content can reveal the removal
        flipped = next(changed for name, _, changed, _ in cases if name == "pack_byte")
        target = next(i for i in range(2, len(records), 2) if flipped[i]["pack"] != records[i]["pack"])
        self.assertEqual(records[target + 1]["boundary"], "end_turn")

    def stub_binary(self):
        path = self.output / "vcmireplay"
        path.write_text(STUB_VERIFIER.format(python=sys.executable))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def prepared(self, terminals=True, **scenario):
        env, end = self.episode(**scenario)
        episode = env.episode_dir
        replay.write_jsonl(episode / "engine" / "replay.jsonl", replay_log(terminals=terminals))
        maps = episode / "profile" / "data" / "vcmi" / "Maps"
        maps.mkdir(parents=True)
        (maps / "headless.vmap").write_bytes(b"authored map stand-in")
        data = self.output / "assets"
        (data / "Data").mkdir(parents=True, exist_ok=True)
        return episode, end, data

    def test_verify_episode_combines_engine_replay_observations_and_terminals(self):
        episode, end, data = self.prepared(mode="natural", end_day=3)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["verified"], result)
        self.assertFalse(result["cut_short"])
        self.assertFalse(result["engine"]["summary"]["prefix"])
        self.assertTrue(result["terminal_evidence"]["matches"])
        self.assertEqual(result["terminal_evidence"]["game_end_boundaries"], 1)
        self.assertTrue(all(player["verified"] for player in result["players"].values()))
        self.assertTrue((episode / "verification" / "replay-verification.json").is_file())
        self.assertFalse((episode / "verification" / "profile").exists())

    def test_a_natural_episode_without_replayed_terminals_is_not_verified(self):
        episode, _, data = self.prepared(terminals=False, mode="natural", end_day=3)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertFalse(result["verified"])
        self.assertFalse(result["terminal_evidence"]["matches"])

    def test_truncated_episodes_replay_as_a_prefix(self):
        episode, end, data = self.prepared(terminals=False, mode="natural", end_day=99,
                                           limits=env_module.Limits(max_days=2, watchdog_seconds=10,
                                                                    startup_seconds=20, shutdown_seconds=10))
        self.assertEqual(end.status, "truncated")
        with (episode / "engine" / "replay.jsonl").open("a") as log:
            log.write('{"kind":"input","seq":4')  # cut by the stop
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["verified"], result)
        self.assertTrue(result["engine"]["summary"]["prefix"])
        self.assertGreater(result["dropped"]["replay_partial_tail_bytes"], 0)
        self.assertIsNone(result["terminal_evidence"]["expected"])

    def test_divergence_is_reported_not_hidden(self):
        episode, _, data = self.prepared(mode="natural", end_day=3)
        records = replay_log()
        records[3]["tampered"] = True
        replay.write_jsonl(episode / "engine" / "replay.jsonl", records)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertFalse(result["verified"])
        self.assertEqual(result["engine"]["divergence"]["field"], "digest")

    def test_series_finds_episodes_of_every_worker(self):
        for path in ("episodes/episode-000002", "worker-0/episodes/episode-000001", "worker-1/episodes/episode-000003"):
            (self.output / "run" / path).mkdir(parents=True)
            (self.output / "run" / path / "manifest.json").write_text("{}")
        self.assertEqual([path.name for path in replay.episode_directories(self.output / "run")],
                         ["episode-000001", "episode-000002", "episode-000003"])

    def test_replay_needs_the_bridge_ledger(self):
        with self.assertRaises(ValueError):
            env_module.HeadlessSelfPlayEnv(FakeLauncher(), self.output / "x", ledger=False, replay=True)


if __name__ == "__main__":
    unittest.main()
