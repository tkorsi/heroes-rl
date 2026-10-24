"""Replay-verification tooling: observation chain, log trimming, tamper cases, verifier plumbing.

The fake engine and the stub verifier exercise the Python side only. They do
not replay VCMI and prove nothing about real-game determinism.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_headless import FakeLauncher, ScriptedAgent, agents, env_module, play  # noqa: E402

SPEC = importlib.util.spec_from_file_location("headless_replay_tool", ROOT / "experiments/headless/replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

# Observation records follow the vcmireplay contract: a pack position of at
# least 1, non-decreasing per player, checked up to the replayed client packs
# and otherwise unchecked (--prefix) or a divergence.
STUB_VERIFIER = '''#!{python}
"""Stub with the vcmireplay interface ("heroes-replay-verification-v1"); not VCMI."""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
value = lambda name: args[args.index(name) + 1]
home = os.environ.get("VCMI_USER_DATA_DIR") or os.path.join(os.environ["XDG_DATA_HOME"], "vcmi")
assert Path(home, "Maps", "headless.vmap").is_file() and Path(home, "Data").is_dir()
records = [json.loads(line) for line in open(value("--log"))]
prefix = "--prefix" in args
terminals, boundaries = [], {{}}
for record in records:
    if record["kind"] == "applied" and record["boundary"]:
        boundaries[record["boundary"]] = boundaries.get(record["boundary"], 0) + 1
        terminals += record["terminals"]
client_packs = records[-1]["client_out"]
divergence = ({{"stage": "applied 1", "field": "digest", "expected": "x", "actual": "y"}}
              if any(record.get("tampered") for record in records) else None)
loaded, checked, unchecked = {{}}, {{}}, {{}}
for player in range(8):
    path = Path(value("--observations"), f"observations-p{{player}}.jsonl")
    if not path.is_file():
        continue
    packs = [json.loads(line).get("pack") for line in path.read_text().splitlines()]
    key = str(player)
    loaded[key] = len(packs)
    checked[key] = sum(type(pack) is int and 1 <= pack <= client_packs for pack in packs)
    unchecked[key] = sum(type(pack) is int and pack > client_packs for pack in packs)
    if divergence is None and (any(type(pack) is not int or pack < 1 for pack in packs) or packs != sorted(packs)
                               or (unchecked[key] and not prefix) or checked[key] + unchecked[key] != loaded[key]):
        divergence = {{"stage": f"observation p{{player}}", "field": "pack", "expected": "", "actual": ""}}
report = {{"schema": "heroes-replay-verification-v1", "verified": divergence is None, "divergence": divergence,
          "summary": {{"inputs": (len(records) - 2) // 2, "prefix": prefix, "terminals": terminals,
                      "boundaries": boundaries, "client_packs": client_packs, "observations_loaded": loaded,
                      "observations_checked": checked, "observations_unchecked": unchecked,
                      "observations_after_prefix": sum(unchecked.values())}}}}
Path(value("--report")).write_text(json.dumps(report))
sys.exit(1 if divergence else 0)
'''


def replay_log(terminals=True, pairs=4, client_packs=None):
    records = [{"kind": "init", "schema": replay.SCHEMA, "seed": 7, "digest": "00"},
               {"kind": "start", "out": 0, "client_out": 0, "chain": "00", "digest": "00"}]
    for seq in range(pairs):
        last = seq == pairs - 1
        records.append({"kind": "input", "seq": seq, "connection": 1, "player": seq % 2, "request": seq, "type": 5,
                        "out": seq, "pack": f"0a0b{seq:02x}"})
        records.append({"kind": "applied", "seq": seq, "blocked": False, "result": True, "out": seq + 1,
                        "client_out": (seq + 1) * client_packs // pairs if client_packs else seq + 1, "chain": "11",
                        "boundary": "game_end" if last and terminals else ("end_turn" if seq % 2 else None),
                        "digest": "22" if (last and terminals) or seq % 2 else None,
                        "terminals": [{"player": 0, "victory": True}, {"player": 1, "victory": False}]
                        if last and terminals else []})
    return records


def own_only_stream(movements):
    """One policy message of own-only visible updates, and the ledger a faithful engine writes for it."""
    fold = replay.VisibleFold([4, 4, 1])
    events, ledger = [], []
    for sequence, movement in enumerate(movements):
        event = {"kind": "visible_update", "sequence": sequence, "day": 1, "appeared": [], "changed": [],
                 "disappeared": [], "revealed": [], "tiles_changed": [], "hidden": [], "own": {"movement": movement}}
        fold.apply(event)
        events.append(event)
        ledger.append({"kind": "capture", "sequence": sequence, "pack": sequence + 1, "pack_type": 0,
                       "digest": fold.digest()})
    ledger.append({"kind": "message", "type": "observation", "index": 0, "pack": len(movements), "pack_type": 0,
                   "digest": fold.digest()})
    bridge = [{"t": 0, "direction": "engine", "message": {"type": "ready", "map_size": [4, 4, 1]}},
              {"t": 1, "direction": "engine", "message": {"type": "observation", "events": events}}]
    return bridge, ledger


class RecorderContractTests(unittest.TestCase):
    def test_recorder_records_the_neutral_player_the_host_connection_owns(self):
        # The live server lets the host connection act for PlayerColor::NEUTRAL (neutral battle stacks). A recorder
        # that enumerates another value records connections without it, and every replay then rejects the first
        # neutral battle action.
        patch = (ROOT / "engine/patches/0013-experiment-replay.patch").read_text()
        line = next(line for line in patch.splitlines() if "constexpr int PLAYER_IDS[]" in line)
        players = [int(value) for value in line.split("{", 1)[1].split("}", 1)[0].split(",")]
        self.assertEqual(players, [0, 1, 2, 3, 4, 5, 6, 7, -1])
        identifiers = ROOT / "vcmi/lib/constants/EntityIdentifiers.cpp"
        if identifiers.is_file():  # the prepared engine, where present, must agree
            self.assertIn("const PlayerColor PlayerColor::NEUTRAL(-1);", identifiers.read_text())


class ReplayToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-replay-test-")
        self.output = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def episode(self, *, replay_enabled=True, limits=None, output=None, **scenario):
        env = env_module.HeadlessSelfPlayEnv(
            FakeLauncher(**scenario), output or self.output / f"run-{len(list(self.output.iterdir()))}",
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
            ledger = env.episode_dir / "engine" / f"observations-p{index}.jsonl"
            check = replay.cross_check_player(env.episode_dir / "agents" / color / "bridge.jsonl", ledger,
                                              cut_short=False)
            self.assertTrue(check["verified"], check)
            self.assertGreater(check["captures_checked"], 2)
            self.assertGreater(check["messages_checked"], check["captures_checked"] // 2)
            packs = [record["pack"] for record in replay.read_jsonl(ledger, allow_partial_tail=False)[0]]
            self.assertGreaterEqual(min(packs), 1)  # as vcmireplay requires
            self.assertEqual(packs, sorted(packs))
            self.assertGreater(packs[-1], packs[0])

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
                      "pack": lines[-1]["pack"], "pack_type": 0, "digest": lines[-1]["digest"]})
        replay.write_jsonl(path, lines)
        self.assertFalse(replay.cross_check_player(bridge, path, cut_short=False)["verified"])
        self.assertTrue(replay.cross_check_player(bridge, path, cut_short=True)["verified"])
        # A stopped engine has at most one message in flight.
        lines.append({**lines[-1], "index": lines[-1]["index"] + 1})
        replay.write_jsonl(path, lines)
        self.assertFalse(replay.cross_check_player(bridge, path, cut_short=True)["verified"])

    def test_a_cut_short_cross_check_needs_every_received_update(self):
        env, _ = self.episode(mode="natural", end_day=3)
        bridge = env.episode_dir / "agents" / "red" / "bridge.jsonl"
        ledger = env.episode_dir / "engine" / "observations-p0.jsonl"
        entries = [json.loads(line) for line in bridge.read_text().splitlines()]
        path = self.output / "bridge-cut.jsonl"
        ready = [entry for entry in entries if entry["message"].get("type") == "ready"]
        for name, kept in (("empty", []), ("ready only", ready), ("no ready", [e for e in entries if e not in ready])):
            replay.write_jsonl(path, kept)
            check = replay.cross_check_player(path, ledger, cut_short=True)
            self.assertFalse(check["verified"], name)
            self.assertIsNotNone(check["first_mismatch"], name)
        # An own state replaces the previous one in full, so only the numbering
        # shows that update 1 never reached the policy.
        stream, lines = own_only_stream([3, 2, 1])
        dropped = json.loads(json.dumps(stream))
        del dropped[1]["message"]["events"][1]
        replay.write_jsonl(path, stream)
        replay.write_jsonl(ledger, lines)
        self.assertTrue(replay.cross_check_player(path, ledger, cut_short=False)["verified"])
        replay.write_jsonl(path, dropped)
        check = replay.cross_check_player(path, ledger, cut_short=True)
        self.assertFalse(check["verified"])
        self.assertEqual(check["first_mismatch"]["at"], "visible update 1 sequence")
        # Dropping capture 1 as well leaves a gap on both sides, also in a complete episode.
        replay.write_jsonl(ledger, [line for line in lines if line.get("sequence") != 1])
        self.assertEqual(replay.cross_check_player(path, ledger, cut_short=False)["first_mismatch"]["at"],
                         "capture 1 sequence")

    def test_cut_short_recordings_keep_only_complete_records(self):
        path = self.output / "log.jsonl"
        path.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
        with self.assertRaises(replay.ReplayError):
            replay.read_jsonl(path, allow_partial_tail=False)
        self.assertEqual(replay.read_jsonl(path, allow_partial_tail=True), ([{"a": 1}, {"b": 2}], 5))
        records = replay_log()
        trimmed, dropped = replay.complete_replay_records(records + [records[2]], cut_short=True)
        self.assertEqual((len(trimmed), dropped), (len(records), 1))
        self.assertEqual(replay.complete_replay_records(records, cut_short=False), (records, 0))
        with self.assertRaises(replay.ReplayError):
            replay.complete_replay_records(records[1:], cut_short=True)
        # A stopped engine leaves at most its last input unpaired; nothing else is trimmed.
        retyped = json.loads(json.dumps(records))
        retyped[5]["kind"] = "input"
        for broken in (records + [records[2]] * 2, records + [records[3]], records[:3] + records[2:], retyped):
            with self.assertRaises(replay.ReplayError):
                replay.complete_replay_records(broken, cut_short=True)
        with self.assertRaises(replay.ReplayError):
            replay.complete_replay_records(records + [records[2]], cut_short=False)

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

    def record_evidence(self, episode):
        """The manifest entry the environment writes when the engine left this evidence."""
        manifest = json.loads((episode / "manifest.json").read_text())
        files = [episode / "engine" / "replay.jsonl", *sorted((episode / "engine").glob("observations-p*.jsonl"))]
        manifest["replay"] = {"recorded": True, "missing": [], "files": {
            path.relative_to(episode).as_posix(): {"sha256": replay.scoring.digest(path), "bytes": path.stat().st_size}
            for path in files}}
        replay.scoring.write_json(episode / "manifest.json", manifest)
        return manifest

    def prepared(self, terminals=True, pairs=4, uncovered=0, output=None, **scenario):
        env, end = self.episode(output=output, **scenario)
        episode = env.episode_dir
        packs = max(record["pack"] for path in (episode / "engine").glob("observations-p*.jsonl")
                    for record in replay.read_jsonl(path, allow_partial_tail=True)[0])
        replay.write_jsonl(episode / "engine" / "replay.jsonl",
                           replay_log(terminals=terminals, pairs=pairs, client_packs=packs - uncovered))
        self.record_evidence(episode)
        maps = episode / "profile" / "data" / "vcmi" / "Maps"
        maps.mkdir(parents=True)
        (maps / "headless.vmap").write_bytes(b"authored map stand-in")
        data = self.output / "assets"
        (data / "Data").mkdir(parents=True, exist_ok=True)
        return episode, end, data

    def truncated(self, **options):
        return self.prepared(terminals=False, mode="natural", end_day=99,
                             limits=env_module.Limits(max_days=2, watchdog_seconds=10, startup_seconds=20,
                                                      shutdown_seconds=10), **options)

    def test_verify_episode_combines_engine_replay_observations_and_terminals(self):
        episode, end, data = self.prepared(mode="natural", end_day=3)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["verified"], result)
        self.assertFalse(result["cut_short"])
        self.assertFalse(result["engine"]["summary"]["prefix"])
        self.assertTrue(result["terminal_evidence"]["matches"])
        self.assertEqual(result["terminal_evidence"]["game_end_boundaries"], 1)
        self.assertTrue(all(player["verified"] for player in result["players"].values()))
        self.assertTrue(result["observation_coverage"]["ok"])
        self.assertEqual(set(result["observation_coverage"]["players"]), {"0", "1"})
        self.assertTrue((episode / "verification" / "replay-verification.json").is_file())
        self.assertFalse((episode / "verification" / "profile").exists())

    def test_a_natural_episode_without_replayed_terminals_is_not_verified(self):
        episode, _, data = self.prepared(terminals=False, mode="natural", end_day=3)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertFalse(result["verified"])
        self.assertFalse(result["terminal_evidence"]["matches"])

    def test_truncated_episodes_replay_as_a_prefix(self):
        episode, end, data = self.truncated()
        self.assertEqual(end.status, "truncated")
        with (episode / "engine" / "replay.jsonl").open("a") as log:
            log.write('{"kind":"input","seq":4')  # cut by the stop
        self.record_evidence(episode)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["verified"], result)
        self.assertTrue(result["engine"]["summary"]["prefix"])
        self.assertGreater(result["dropped"]["replay_partial_tail_bytes"], 0)
        self.assertIsNone(result["terminal_evidence"]["expected"])
        self.assertEqual(result["observation_coverage"]["after_prefix"], 0)

    def test_a_cut_short_episode_is_verified_only_if_the_replay_reaches_every_capture(self):
        episode, end, data = self.truncated(uncovered=1)
        self.assertEqual(end.status, "truncated")
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["engine"]["verified"])  # --prefix only counts the records it never reached
        self.assertTrue(all(player["verified"] for player in result["players"].values()))
        self.assertFalse(result["verified"])
        coverage = result["observation_coverage"]
        self.assertFalse(coverage["ok"])
        self.assertGreater(coverage["after_prefix"], 0)
        self.assertTrue(any(player["unchecked"] for player in coverage["players"].values()), coverage)

    def test_observation_counts_must_match_the_prepared_ledgers(self):
        written = {"0": 5, "1": 4}
        complete = {"observations_loaded": dict(written), "observations_checked": dict(written),
                    "observations_unchecked": {"0": 0, "1": 0}, "observations_after_prefix": 0}
        self.assertTrue(replay.observation_coverage(complete, written)["ok"])
        # A record neither checked nor counted past the prefix, as the old verifier left one at pack 0.
        blind = {**complete, "observations_checked": {"0": 4, "1": 4}}
        self.assertEqual(replay.observation_coverage(blind, written)["players"]["0"],
                         {"written": 5, "loaded": 5, "checked": 4, "unchecked": 0})
        for summary in (blind, {**complete, "observations_loaded": {"0": 4, "1": 4}, "observations_checked": {"0": 4, "1": 4}},
                        {**complete, "observations_after_prefix": 1},
                        {key: value for key, value in complete.items() if key != "observations_loaded"},
                        {key: value for key, value in complete.items() if key != "observations_after_prefix"}):
            self.assertFalse(replay.observation_coverage(summary, written)["ok"], summary)
        # End to end: a verifier report that leaves a record unchecked does not verify the episode.
        episode, _, data = self.prepared(mode="natural", end_day=3)
        run_replay = replay.run_replay

        def skipping_one(*args, **kwargs):
            outcome = run_replay(*args, **kwargs)
            outcome["summary"]["observations_checked"]["0"] -= 1
            return outcome

        with mock.patch.object(replay, "run_replay", skipping_one):
            result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertTrue(result["engine"]["verified"])
        self.assertFalse(result["verified"])
        self.assertFalse(result["observation_coverage"]["ok"])

    def test_evidence_must_be_the_files_the_manifest_hashed(self):
        episode, end, data = self.truncated()
        binary = self.stub_binary()
        log = episode / "engine" / "replay.jsonl"
        records = replay.read_jsonl(log, allow_partial_tail=False)[0]
        self.assertTrue(replay.verify_episode(episode, binary=binary, data=data)["verified"])
        # Cut at a record boundary after the episode: the shorter prefix would replay cleanly.
        replay.write_jsonl(log, records[:4])
        result = replay.verify_episode(episode, binary=binary, data=data)
        self.assertFalse(result["verified"])
        self.assertIn("changed after the environment hashed it", result["error"])
        replay.write_jsonl(log, records)
        manifest = self.record_evidence(episode)
        for name, change in (("absent", None), ("missing", {**manifest["replay"], "missing": ["engine/replay.jsonl"]}),
                             ("unlisted", {**manifest["replay"], "files": {
                                 k: v for k, v in manifest["replay"]["files"].items() if "p1" not in k}})):
            replay.scoring.write_json(episode / "manifest.json", {**manifest, "replay": change})
            result = replay.verify_episode(episode, binary=binary, data=data)
            self.assertFalse(result["verified"], name)
            self.assertIn("ReplayError", result["error"], name)
        # A broken pair the engine could not have written is an error, not a shorter prefix.
        replay.write_jsonl(log, records[:3] + records[2:])
        self.record_evidence(episode)
        result = replay.verify_episode(episode, binary=binary, data=data)
        self.assertFalse(result["verified"])
        self.assertIn("breaks the input/applied pairs", result["error"])

    def test_divergence_is_reported_not_hidden(self):
        episode, _, data = self.prepared(mode="natural", end_day=3)
        records = replay.read_jsonl(episode / "engine" / "replay.jsonl", allow_partial_tail=False)[0]
        records[3]["tampered"] = True
        replay.write_jsonl(episode / "engine" / "replay.jsonl", records)
        self.record_evidence(episode)
        result = replay.verify_episode(episode, binary=self.stub_binary(), data=data)
        self.assertFalse(result["verified"])
        self.assertEqual(result["engine"]["divergence"]["field"], "digest")

    def test_series_finds_episodes_of_every_worker(self):
        for path in ("episodes/episode-000002", "worker-0/episodes/episode-000001", "worker-1/episodes/episode-000003"):
            (self.output / "run" / path).mkdir(parents=True)
            (self.output / "run" / path / "manifest.json").write_text("{}")
        self.assertEqual([path.name for path in replay.episode_directories(self.output / "run")],
                         ["episode-000001", "episode-000002", "episode-000003"])

    def test_series_traces_and_tamper_tests_verified_episodes_with_battles_first(self):
        run = self.output / "selection"
        # episode: (status, battles, verifies)
        plan = {1: ("completed", 0, True), 2: ("truncated", 3, True), 3: ("completed", 2, False),
                4: ("completed", 1, True), 5: ("completed", 0, True), 6: ("completed", 4, True)}
        for number, (status, battles, _) in plan.items():
            path = run / "episodes" / f"episode-{number:06d}"
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps({"status": status, "players": {
                "red": {"battles_started": battles}, "blue": {"battles_started": 0}}}))
        calls = []

        def verify(episode, *, trace, **kwargs):
            number = int(episode.name.split("-")[1])
            calls.append((number, trace))
            return {"episode": number, "status": plan[number][0], "termination_reason": None, "verified": plan[number][2]}

        def tamper(episode, **kwargs):
            return {"cases": [], "all_detected": True, "episode": episode.name}

        with mock.patch.object(replay, "verify_episode", verify), mock.patch.object(replay, "tamper_episode", tamper), \
                contextlib.redirect_stdout(io.StringIO()):
            summary = replay.verify_series(run, binary=Path("unused"), data=Path("unused"), count=4, tamper=2, trace=2,
                                           allow_platform_mismatch=False, prefer_natural=True)
        # Natural endings with a battle come first; episode 3 fails, so tracing moves on until two verified traces exist.
        self.assertEqual(calls, [(3, True), (4, True), (6, True), (1, False)])
        self.assertEqual([row["episode"] for row in summary["episodes"] if "tamper" in row], [4, 6])

    def test_tamper_failures_are_recorded_and_the_series_report_is_written(self):
        run = self.output / "series"
        episode, _, data = self.truncated(pairs=2, output=run / "episodes")
        binary = self.stub_binary()

        def series():
            with contextlib.redirect_stdout(io.StringIO()):
                return replay.verify_series(run, binary=binary, data=data, count=None, tamper=1, trace=0,
                                            allow_platform_mismatch=False, prefer_natural=False)

        summary = series()  # a cut-short episode with two complete inputs
        self.assertTrue((run / "replay-series.json").is_file())
        row = summary["episodes"][0]
        self.assertTrue(row["verified"], row)
        self.assertIn("tamper tests need at least 4", row["tamper"]["skipped"])
        self.assertEqual((summary["aggregate"]["tamper_runs"], summary["aggregate"]["tamper_skipped"]), (0, 1))
        self.assertIsNone(summary["aggregate"]["tamper_all_detected"])
        (run / "replay-series.json").unlink()

        def failing(*args, **kwargs):
            raise OSError("disk full")

        with mock.patch.object(replay, "tamper_episode", failing):
            summary = series()
        self.assertTrue((run / "replay-series.json").is_file())
        self.assertEqual(summary["episodes"][0]["tamper"], {"error": "OSError: disk full", "all_detected": False})
        self.assertIs(summary["aggregate"]["tamper_all_detected"], False)

    def test_a_verifier_crash_on_a_tamper_case_is_reported_apart_from_a_divergence(self):
        episode, _, data = self.prepared(mode="natural", end_day=3)
        run_replay = replay.run_replay

        def crashing(binary, environment, log, *args, **kwargs):
            if log.parent.name == "seed":
                raise replay.ReplayError("vcmireplay exited -11 without a report")
            if log.parent.name == "pack_byte":
                raise subprocess.TimeoutExpired("vcmireplay", 1)
            return run_replay(binary, environment, log, *args, **kwargs)

        with mock.patch.object(replay, "run_replay", crashing):
            report = replay.tamper_episode(episode, binary=self.stub_binary(), data=data)
        cases = {case["case"]: case for case in report["cases"]}
        self.assertEqual(report["crashed_cases"], ["seed", "pack_byte"])
        for name in ("seed", "pack_byte"):
            self.assertTrue(cases[name]["detected"])  # fail-closed
            self.assertIsNone(cases[name]["divergence"])
        self.assertIn("-11", cases["seed"]["crashed"])
        self.assertIn("TimeoutExpired", cases["pack_byte"]["crashed"])
        self.assertIsNone(cases["state_digest"]["crashed"])
        self.assertIsNone(report["skipped"])

    def test_replay_needs_the_bridge_ledger(self):
        with self.assertRaises(ValueError):
            env_module.HeadlessSelfPlayEnv(FakeLauncher(), self.output / "x", ledger=False, replay=True)


if __name__ == "__main__":
    unittest.main()
