"""Series runner and benchmark harness over the real launcher and a fake client.

A stand-in ``vcmiclient`` script runs the miniature fake engine, so the real
launcher, profile isolation, seed schedule, warm-up, synchronized measured
windows and aggregation are exercised. Nothing here measures VCMI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE_ENGINE = ROOT / "tests/fixtures/headless/fake_engine.py"

CLIENT = f'''#!{sys.executable}
# Test stand-in for the patched client; preflight looks for VCMI_EXPERIMENT_DIR and EXPLORER_RL_PLAYERS.
import json, os, sys
config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.environ["VCMI_USER_DATA_DIR"], "..")
settings = os.path.join(config, "vcmi", "settings.json") if "XDG_CONFIG_HOME" in os.environ \\
    else os.path.join(os.environ["VCMI_USER_DATA_DIR"], "config", "settings.json")
os.environ["FAKE_ENGINE_SEED"] = str(json.load(open(settings))["server"]["seed"])
os.execv(sys.executable, [sys.executable, {str(FAKE_ENGINE)!r}])
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
        (self.binary.parent / "AI" / f"libExplorerRL{suffix}").write_bytes(b"stand-in visible_update-v2")
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

    def test_benchmark_aligns_windows_and_reports_end_to_end_rates(self):
        output = self.base / "bench"
        result = self.run_tool("benchmark.py", "--output", str(output), "--workers", "1,2", "--repetitions", "1",
                               "--duration", "4", "--warmup-allowance", "3", "--max-days", "3", timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "benchmark.json").read_text())
        self.assertEqual([c["workers"] for c in report["configurations"]], [1, 2])
        for configuration in report["configurations"]:
            entry = configuration["repetitions"][0]
            self.assertEqual(entry["failures"], 0)
            self.assertGreater(entry["decision_cycles"], 0)
            self.assertEqual(entry["decision_cycles"], entry["accepted_actions"] + entry["rejected_attempts"])
            self.assertLess(entry["measured_start_spread_seconds"], 1.0)
            self.assertIsNotNone(entry["scaling_efficiency"])
            self.assertFalse(configuration["stretch_target_1000_per_second_met"])
        two = report["configurations"][1]["repetitions"][0]
        indices = sorted(row for worker in (output / "workers-2/rep-0").glob("worker-[0-9]")
                         for row in [e["index"] for e in json.loads((worker / "summary.json").read_text())["schedule"][:2]])
        self.assertEqual(indices, [0, 1, 2, 3])  # the same schedule, interleaved across workers


if __name__ == "__main__":
    unittest.main()
