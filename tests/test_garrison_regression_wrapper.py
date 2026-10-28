"""Offline wrapper checks; no engine process, gameplay or PPO is launched."""
import fcntl
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import validate_garrison_regressions as wrapper


class GarrisonRegressionWrapperTests(unittest.TestCase):
    def test_busy_official_lock_and_existing_evidence_prevent_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lockpath = root / "owner.lock"
            lockpath.touch()
            args = SimpleNamespace(official_lock=lockpath, output=root / "attempt")
            with lockpath.open("r") as owner, patch.object(wrapper, "guarded_popen") as launch:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    wrapper.supervise(args)
                self.assertFalse(args.output.exists())
                launch.assert_not_called()
            args.output.mkdir()
            with patch.object(wrapper, "guarded_popen") as launch:
                with self.assertRaises(FileExistsError):
                    wrapper.supervise(args)
                launch.assert_not_called()

    def test_worker_keeps_all_controls_and_adds_only_guarded_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "supervision.json"
            plan = {"schema": "heroes-garrison-regression-supervision-v1",
                    "absolute_deadline_unix": time.time() + 60,
                    "hard_deadline_monotonic": time.monotonic() + 60,
                    "binary": "/unused/client", "replay_binary": "/unused/replay", "data": "/unused/data",
                    "seed": 83070000, "case_seconds": 120}
            path.write_text(json.dumps(plan))
            original_run = wrapper.capability.CapabilityRun
            original_identity = wrapper.capability.capability_implementation_identity
            calls = []

            def inspect_only(argv):
                calls.append(argv)
                self.assertTrue(issubclass(wrapper.capability.CapabilityRun, wrapper.garrison.GarrisonRun))
                self.assertIs(wrapper.capability.capability_implementation_identity, wrapper.implementation_identity)
                self.assertEqual(argv[argv.index("--case") + 1], "all")
                self.assertEqual(argv[argv.index("--player") + 1], "both")
                self.assertEqual(argv[argv.index("--seed-count") + 1], "1")
                self.assertEqual(argv[argv.index("--seed") + 1], "83070000")
                schedule = wrapper.capability.schedule(tuple(wrapper.capability.CASES),
                    wrapper.capability.fixtures.COLORS, plan["seed"], 1)
                self.assertEqual(len(schedule), 28)
                self.assertEqual(len({(row["case"], row["player"]) for row in schedule}), 28)
                for row in schedule:
                    wrapper.capability.fixture_identity(row["fixture"], row["player"], root / "maps")
                return 0

            with patch.object(wrapper, "check_bindings") as bindings, \
                    patch.object(wrapper.capability, "main", side_effect=inspect_only), \
                    patch.object(wrapper.garrison.environment.VcmiLauncher, "launch") as native_launch:
                self.assertEqual(wrapper.run_worker(path, wrapper.digest(path)), 0)
                self.assertEqual(bindings.call_count, 2)
                native_launch.assert_not_called()
            self.assertEqual(len(calls), 1)
            self.assertIs(wrapper.capability.CapabilityRun, original_run)
            self.assertIs(wrapper.capability.capability_implementation_identity, original_identity)
            with patch.object(wrapper, "check_bindings"), patch.object(wrapper.capability, "main") as rerun:
                with self.assertRaises(FileExistsError):
                    wrapper.run_worker(path, wrapper.digest(path))
                rerun.assert_not_called()

    def test_changed_plan_or_expired_window_never_generates_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "supervision.json"
            path.write_text(json.dumps({"schema": "heroes-garrison-regression-supervision-v1",
                "absolute_deadline_unix": time.time() - 1, "hard_deadline_monotonic": time.monotonic() + 60}))
            with patch.object(wrapper, "check_bindings"), patch.object(wrapper.capability, "main") as run:
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    wrapper.run_worker(path, "0" * 64)
                with self.assertRaises(TimeoutError):
                    wrapper.run_worker(path, wrapper.digest(path))
                run.assert_not_called()
            self.assertFalse((root / "maps").exists())


if __name__ == "__main__":
    unittest.main()
