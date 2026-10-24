"""Series runner and benchmark harness over the real launcher and a fake client.

A stand-in ``vcmiclient`` script runs the miniature fake engine, so the real
launcher, profile isolation, seed schedule, warm-up, synchronized measured
windows and aggregation are exercised. tools/validate_headless.sh runs in a
copy of the tree with the real series driver, cost breakdown and compactor and
stand-ins for the replay verifier and the benchmark. Nothing here measures VCMI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_ENGINE = ROOT / "tests/fixtures/headless/fake_engine.py"

CLIENT = f'''#!{sys.executable}
# Test stand-in for the patched client; preflight looks for VCMI_EXPERIMENT_DIR, EXPLORER_RL_PLAYERS and VCMI_EXPERIMENT_REPLAY.
import json, os, sys
config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.environ["VCMI_USER_DATA_DIR"], "..")
settings = os.path.join(config, "vcmi", "settings.json") if "XDG_CONFIG_HOME" in os.environ \\
    else os.path.join(os.environ["VCMI_USER_DATA_DIR"], "config", "settings.json")
os.environ["FAKE_ENGINE_SEED"] = str(json.load(open(settings))["server"]["seed"])
os.execv(sys.executable, [sys.executable, {str(FAKE_ENGINE)!r}])
'''

STUB_REPLAY = '''import json, os, sys
from pathlib import Path
# Test stand-in for replay.py series: every episode verifies and every tamper case is rejected unless STUB_REPLAY_FAIL.
run, verified = Path(sys.argv[2]), "STUB_REPLAY_FAIL" not in os.environ
rows = [{"episode": int(path.name.split("-")[1]), "verified": verified}
        for path in sorted(run.glob("worker-*/episodes/episode-*"))]
(run / "replay-series.json").write_text(json.dumps({"episodes": rows, "aggregate": {
    "verified": sum(row["verified"] for row in rows), "attempted": len(rows), "tamper_all_detected": verified}}))
sys.exit(0 if verified else 1)
'''

STUB_BENCHMARK = '''import json, os, sys
from pathlib import Path
# Test stand-in for benchmark.py: STUB_BENCHMARK_FAIL stops it before its last configuration.
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.mkdir(parents=True, exist_ok=False)
report = {"configurations": [], "complete": False, "finished_at": None}
(output / "benchmark.json").write_text(json.dumps(report))
if "STUB_BENCHMARK_FAIL" in os.environ:
    sys.exit(1)
report.update(complete=True, finished_at="2026-01-01T00:00:00+00:00")
(output / "benchmark.json").write_text(json.dumps(report))
'''

INTERRUPT_SECOND_CONFIGURATION = '''import sys
sys.path.insert(0, sys.argv[1])
import benchmark
real = benchmark.run_configuration

def run_configuration(args, workers, repetition, directory):
    if workers > 1:
        raise KeyboardInterrupt  # Ctrl-C during the second configuration
    return real(args, workers, repetition, directory)

benchmark.run_configuration = run_configuration
sys.argv = ["benchmark.py", *sys.argv[2:]]
benchmark.main()
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-runner-test-")
        base = Path(self.directory.name)
        self.binary = base / "build/bin/vcmiclient"
        (self.binary.parent / "AI").mkdir(parents=True)
        self.binary.write_text(CLIENT)
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IEXEC)
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        (self.binary.parent / "AI" / f"libExplorerRL{suffix}").write_bytes(b"stand-in visible_update-v2 observations-p")
        (base / "data/Data").mkdir(parents=True)
        self.map = base / "scenario.vmap"
        self.map.write_bytes(b"stand-in map")
        self.base = base
        self.common = ["--binary", str(self.binary), "--data", str(base / "data"), "--map", str(self.map), "--skip-check"]

    def tearDown(self):
        self.directory.cleanup()

    def run_tool(self, name, *arguments, timeout=120):
        return subprocess.run([sys.executable, str(ROOT / "experiments/headless" / name), *arguments, *self.common],
                              capture_output=True, text=True, timeout=timeout)

    def validation_tree(self) -> Path:
        tree = self.base / "tree"
        (tree / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "tools/validate_headless.sh", tree / "tools")
        headless = tree / "experiments/headless"
        headless.mkdir(parents=True)
        for name in ("run_series.py", "cost_breakdown.py", "compact_evidence.py"):
            (headless / name).symlink_to(ROOT / "experiments/headless" / name)
        (headless / "replay.py").write_text(STUB_REPLAY)
        (headless / "benchmark.py").write_text(STUB_BENCHMARK)
        return tree

    def validate(self, tree: Path, *stages, **environment):
        settings = {key: value for key, value in os.environ.items() if key != "OUT"}
        settings.update({"DATA": str(self.base / "data"), "BINARY": str(self.binary), "MAP": str(self.map),
                         "PYTHON": sys.executable, "ATTEMPTS": "2", "WORKERS": "1", **environment})
        return subprocess.run([str(tree / "tools/validate_headless.sh"), *stages], env=settings,
                              capture_output=True, text=True, timeout=300)

    def test_series_follows_its_frozen_schedule_slice_after_warmup(self):
        output = self.base / "series"
        result = self.run_tool("run_headless.py", "--output", str(output), "--episodes", "3", "--schedule-start", "1",
                               "--schedule-step", "2", "--warmup-episodes", "1", "--max-days", "3", "--raw-latencies")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual([e["index"] for e in summary["schedule"]], [1, 3, 5])
        self.assertEqual([row["episode"] for row in summary["episodes"]], [2, 4, 6])
        self.assertEqual(len(summary["warmup"]), 1)
        self.assertEqual(summary["aggregate"]["failures"], 0)
        for row in summary["episodes"]:
            self.assertEqual(len(row["step_latencies"]), row["env_steps"])
            self.assertTrue(row["red"]["history_v1"]["valid"] and row["blue"]["history_v1"]["valid"])
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["resolved_actions"], aggregate["accepted_actions"] + aggregate["rejected_attempts"])
        # Summed over both players, the battle count is named for what it counts.
        self.assertNotIn("battles", aggregate)
        self.assertEqual(aggregate["battle_participations_red_plus_blue"],
                         sum(row[color]["battles"] for row in summary["episodes"] for color in ("red", "blue")))

    def test_benchmark_aligns_windows_and_reports_end_to_end_rates(self):
        output = self.base / "bench"
        result = self.run_tool("benchmark.py", "--output", str(output), "--workers", "1,2", "--repetitions", "1",
                               "--duration", "4", "--warmup-allowance", "3", "--max-days", "3", timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "benchmark.json").read_text())
        self.assertEqual([c["workers"] for c in report["configurations"]], [1, 2])
        self.assertTrue(report["complete"])
        self.assertIsNotNone(report["finished_at"])
        for configuration in report["configurations"]:
            entry = configuration["repetitions"][0]
            self.assertEqual(entry["failures"], 0)
            self.assertGreater(entry["decision_cycles"], 0)
            self.assertEqual(entry["decision_cycles"], entry["accepted_actions"] + entry["rejected_attempts"])
            self.assertLess(entry["measured_start_spread_seconds"], 1.0)
            self.assertIsNotNone(entry["scaling_efficiency"])
            self.assertIn("battle_participations_red_plus_blue", entry["breakdown"])
            self.assertFalse(configuration["stretch_target_1000_per_second_met"])
        two = report["configurations"][1]["repetitions"][0]
        indices = sorted(row for worker in (output / "workers-2/rep-0").glob("worker-[0-9]")
                         for row in [e["index"] for e in json.loads((worker / "schedule.json").read_text())["schedule"][:2]])
        self.assertEqual(indices, [0, 1, 2, 3])  # the same schedule, interleaved across workers
        # The long benchmark plan stays out of the summary that is rewritten after every measured episode.
        self.assertIsNone(json.loads((output / "workers-2/rep-0/worker-0/summary.json").read_text())["schedule"])

    def test_attempt_series_preserves_every_planned_attempt_across_workers(self):
        output = self.base / "attempts"
        result = self.run_tool("run_series.py", "--output", str(output), "--attempts", "5", "--workers", "2",
                               "--max-days", "3", "--profile", timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        series = json.loads((output / "series.json").read_text())
        self.assertEqual(sorted(row["episode"] for row in series["episodes"]), [1, 2, 3, 4, 5])
        self.assertEqual((series["attempts_missing"], series["attempts_unplanned"]), ([], []))
        self.assertEqual(series["natural_completions"] + series["truncations"] + series["failures"], 5)
        self.assertEqual(sorted(row["engine_seed"] for row in series["episodes"]), list(range(40001, 40006)))
        costs = subprocess.run([sys.executable, str(ROOT / "experiments/headless/cost_breakdown.py"),
                                str(output / "worker-0")], capture_output=True, text=True, timeout=60, check=True)
        breakdown = json.loads(costs.stdout)
        self.assertEqual(breakdown["episodes"], 3)
        self.assertGreater(breakdown["adapter_totals"]["ingest"], 0)
        self.assertNotIn("battles", breakdown)
        self.assertIn("battle_participations_red_plus_blue", breakdown)

    def test_interrupted_benchmark_stays_incomplete_in_the_evidence(self):
        output = self.base / "bench"
        result = subprocess.run([sys.executable, "-c", INTERRUPT_SECOND_CONFIGURATION, str(ROOT / "experiments/headless"),
                                 "--output", str(output), "--workers", "1,2", "--repetitions", "1", "--duration", "2",
                                 "--warmup-allowance", "2", "--max-days", "3", *self.common],
                                capture_output=True, text=True, timeout=300)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KeyboardInterrupt", result.stderr)
        report = json.loads((output / "benchmark.json").read_text())
        self.assertEqual([c["workers"] for c in report["configurations"]], [1])
        self.assertEqual((report["complete"], report["finished_at"]), (False, None))
        evidence = self.base / "evidence.json"
        subprocess.run([sys.executable, str(ROOT / "experiments/headless/compact_evidence.py"), "--output", str(evidence),
                        "--benchmark", str(output / "benchmark.json")], capture_output=True, timeout=60, check=True)
        compact = json.loads(evidence.read_text())
        self.assertFalse(compact["benchmark"]["complete"])
        self.assertTrue(any("benchmark is incomplete" in note for note in compact["notes"]))

    def test_attempt_series_reports_steady_state_per_worker_and_series_wide(self):
        output = self.base / "attempts"
        result = self.run_tool("run_series.py", "--output", str(output), "--attempts", "4", "--workers", "2",
                               "--max-days", "3", timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        series = json.loads((output / "series.json").read_text())
        aggregate = series["aggregate"]
        stepping = sum(row["engine_wait_seconds"] + row["agent_seconds"] for row in series["episodes"])
        resolved = aggregate["resolved_actions"]
        self.assertNotIn("steady_state_resolved_actions_per_second", aggregate)
        self.assertEqual(aggregate["steady_state_resolved_actions_per_second_per_worker"], round(resolved / stepping, 3))
        self.assertEqual(aggregate["steady_state_resolved_actions_per_second_all_workers"],
                         round(resolved / (stepping / 2), 3))
        # Steady state leaves out launch and reset time, so the series-wide rate is at least the end-to-end rate.
        self.assertGreaterEqual(aggregate["steady_state_resolved_actions_per_second_all_workers"] + 0.001,
                                aggregate["end_to_end_resolved_actions_per_second"])

    def test_replay_record_without_ledger_is_refused_before_any_output(self):
        output = self.base / "series"
        refused = self.run_tool("run_headless.py", "--output", str(output), "--ledger", "none")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("use --ledger full", refused.stderr)
        self.assertFalse(output.exists())
        corrected = self.run_tool("run_headless.py", "--output", str(output), "--ledger", "none", "--replay", "none",
                                  "--max-days", "3")
        self.assertEqual(corrected.returncode, 0, corrected.stdout + corrected.stderr)
        attempts = self.base / "attempts"
        refused = self.run_tool("run_series.py", "--output", str(attempts), "--attempts", "2", "--workers", "2",
                                "--ledger", "none")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("use --ledger full", refused.stderr)
        self.assertFalse(attempts.exists())

    def test_validation_stops_after_a_failed_smoke_and_redoes_it(self):
        tree = self.validation_tree()
        out = self.base / "validation"
        failed = self.validate(tree, "smoke", "series", OUT=str(out), STUB_REPLAY_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("smoke: 0 of 4 selected attempts verified", failed.stderr)
        self.assertIn("Smoke failed", failed.stderr)
        self.assertFalse((out / "smoke/ok").exists())
        self.assertFalse((out / "attempts").exists())  # the long stages never started
        passed = self.validate(tree, "smoke", "series", OUT=str(out))
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertEqual(len(list(out.glob("smoke.interrupted-*/replay-series.json"))), 1)  # the failed smoke is kept
        self.assertTrue((out / "smoke/ok").is_file())
        self.assertEqual(json.loads((out / "attempts/series.json").read_text())["attempts_played"], 2)
        self.assertIn("smoke: done", self.validate(tree, "smoke", OUT=str(out)).stdout)
        self.assertFalse((tree / "runs/validation-current").exists())  # an explicit OUT is not pinned

    def test_validation_resumes_interrupted_stages_without_losing_evidence(self):
        tree = self.validation_tree()
        out = self.base / "validation"
        # A series interrupted mid-run leaves its directory without series.json.
        (out / "attempts/worker-0").mkdir(parents=True)
        (out / "attempts/worker-0/partial").write_text("kept")
        # A profile series that returned 1 under set -e leaves series.json but no costs.json.
        profile = self.run_tool("run_series.py", "--output", str(out / "profile"), "--attempts", "2", "--workers", "1",
                                "--max-days", "3", "--profile")
        self.assertEqual(profile.returncode, 0, profile.stdout + profile.stderr)
        stopped = self.validate(tree, "benchmark", OUT=str(out), STUB_BENCHMARK_FAIL="1")
        self.assertNotEqual(stopped.returncode, 0)
        result = self.validate(tree, "profile", "series", "benchmark", "evidence", OUT=str(out))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((out / "costs.json").read_text())["episodes"], 2)  # from the existing profile
        moved = list(out.glob("attempts.interrupted-*"))
        self.assertEqual(len(moved), 1)
        self.assertEqual((moved[0] / "worker-0/partial").read_text(), "kept")
        self.assertEqual(json.loads((out / "attempts/series.json").read_text())["attempts_played"], 2)
        self.assertEqual(len(list(out.glob("benchmark.interrupted-*/benchmark.json"))), 1)
        self.assertTrue(json.loads((out / "benchmark/benchmark.json").read_text())["complete"])
        self.assertTrue(json.loads((out / "evidence.json").read_text())["benchmark"]["complete"])

    def test_validation_redoes_a_profile_whose_breakdown_failed(self):
        tree = self.validation_tree()
        out = self.base / "validation"
        broken = self.validate(tree, "profile", OUT=str(out), BINARY=str(self.base / "missing/vcmiclient"))
        self.assertNotEqual(broken.returncode, 0)
        self.assertIn("cost_breakdown.py failed", broken.stderr)
        self.assertFalse((out / "costs.json").exists())
        redone = self.validate(tree, "profile", OUT=str(out))
        self.assertEqual(redone.returncode, 0, redone.stdout + redone.stderr)
        self.assertEqual(len(list(out.glob("profile.interrupted-*/series.json"))), 1)
        self.assertEqual(json.loads((out / "costs.json").read_text())["episodes"], 16)

    def test_validation_pins_its_default_output_across_dates(self):
        tree = self.validation_tree()
        first = self.validate(tree, "smoke")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        pinned = (tree / "runs/validation-current").read_text().strip()
        self.assertRegex(pinned, r"^runs/validation-\d{4}-\d{2}-\d{2}$")
        self.assertIn(f"Output: {pinned}", first.stdout)
        self.assertTrue((tree / pinned / "smoke/ok").is_file())
        # A validation started on an earlier UTC date resumes there instead of starting over.
        (tree / pinned).rename(tree / "runs/validation-2000-01-01")
        (tree / "runs/validation-current").write_text("runs/validation-2000-01-01\n")
        resumed = self.validate(tree, "smoke")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertIn("Output: runs/validation-2000-01-01", resumed.stdout)
        self.assertIn("smoke: done", resumed.stdout)
        self.assertFalse((tree / pinned).exists())

    def test_validation_keeps_every_test_run_in_the_evidence(self):
        tree, out = self.validation_tree(), self.base / "validation"
        (tree / "tests").mkdir()
        (tree / "tests/test_stub.py").write_text(
            "import os, unittest\n\nclass Stub(unittest.TestCase):\n"
            "    def test_stub(self):\n        self.assertNotIn('STUB_TESTS_FAIL', os.environ)\n")
        failed = self.validate(tree, "tests", OUT=str(out), STUB_TESTS_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Tests failed; full log:", failed.stderr)
        passed = self.validate(tree, "tests", "evidence", OUT=str(out))
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        # A resumed run adds its log instead of replacing the failed run's.
        self.assertEqual(len(list(out.glob("tests-*.log"))), 2)
        combined = (out / "tests.log").read_text()
        self.assertEqual([line.rsplit(" ", 2)[-2:] for line in combined.splitlines() if line.startswith("== ")],
                         [["(exit", "1)"], ["(exit", "0)"]])
        record = json.loads((out / "evidence.json").read_text())["tests_log"]
        self.assertEqual((record["tests_run"], record["result"]), (1, "OK"))


if __name__ == "__main__":
    unittest.main()
