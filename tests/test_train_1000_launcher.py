"""Exercise the manual launcher with disposable controllers, never official training."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SHA256 = "919951d970eefa356ed19c901ddc3441e108035da186957b5cbe107b00ace623"
MOCK_CONTROLLER = r'''
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["MOCK_ROOT"])
(root / "arguments.json").write_text(json.dumps(sys.argv[1:]))
mode = os.environ.get("MOCK_MODE", "exit")
status = int(os.environ.get("MOCK_STATUS", "0"))
if mode == "signal_self":
    os.kill(os.getpid(), int(os.environ["MOCK_SIGNAL"]))
if mode == "wait_for_stop":
    signals = []
    def stop(signum, _frame):
        signals.append(signum)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (root / "ready").write_text(str(os.getpid()))
    deadline = time.monotonic() + 5
    while not signals and time.monotonic() < deadline:
        time.sleep(0.01)
    if not signals:
        raise SystemExit(97)
    time.sleep(0.15)
    (root / "cleaned.json").write_text(json.dumps({"signals": signals, "status": status}))
print("Mock controller completed", flush=True)
raise SystemExit(status)
'''


class Train1000LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="heroes-manual-launcher-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        self.wrapper = self.root / "train-1000.sh"
        shutil.copyfile(ROOT / self.wrapper.name, self.wrapper)
        shutil.copyfile(ROOT / "tools/train_1000_games.py", self.root / "tools/train_1000_games.py")
        controller = self.root / "continue-original-native.sh"
        controller.write_text(f"#!{sys.executable}\n" + MOCK_CONTROLLER)
        controller.chmod(0o755)
        self.env = {**os.environ, "HOME": str(self.root), "HEROES_PYTHON": sys.executable,
                    "MOCK_ROOT": str(self.root)}

    def run_wrapper(self, *args, **environment):
        return subprocess.run(["sh", str(self.wrapper), *args], cwd=self.root,
                              env={**self.env, **environment}, text=True,
                              capture_output=True, timeout=8)

    def called(self):
        return json.loads((self.root / "arguments.json").read_text())

    def assert_runtime(self, result, status):
        self.assertEqual(result.returncode, status, result.stderr)
        self.assertRegex(result.stdout, rf"Total running time: \d+:\d{{2}}:\d{{2}} "
                         rf"\(\d+\.\d{{2}} seconds\); exit status: {status}\.")

    def test_fixed_game_target_native_binding_and_default_hours(self):
        result = self.run_wrapper()
        self.assert_runtime(result, 0)
        args = self.called()
        expected = {"--session-games": "1000", "--native-sha256": NATIVE_SHA256,
                    "--native-learner": str(self.root / "Projects/Heroes-artifacts/"
                        "native-learner-builds/mac-utilization-v1-attempt6/manifest.json")}
        for flag, value in expected.items():
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(float(args[args.index("--hours") + 1]), 10)
        self.assertNotIn("--cpu-cap", args)
        self.assertNotIn("--check", args)
        self.assertNotIn("--parent", args)
        self.assertFalse(any("1229" in arg for arg in args))
        self.assertIn("250 complete PPO updates", result.stdout)

    def test_check_cpu_cap_hours_and_recovery_are_forwarded(self):
        result = self.run_wrapper("--check", "--hours", "1.5", "--cpu-cap", "8",
                                  "--recovery-receipt", "reviewed receipt.json")
        self.assert_runtime(result, 0)
        args = self.called()
        self.assertEqual(args.count("--check"), 1)
        for flag, value in (("--session-games", "1000"), ("--hours", "1.5"),
                            ("--cpu-cap", "8"), ("--recovery-receipt",
                             str((self.root / "reviewed receipt.json").resolve()))):
            self.assertEqual(args.count(flag), 1)
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_conflicting_and_unknown_arguments_never_invoke_controller(self):
        for args in (("--session-games", "4"), ("--games", "4"), ("--parent", "old.json"),
                     ("--native-learner", "/other/manifest.json"), ("--native-sha256", "a" * 64),
                     ("--config", "other.json"), ("--max-updates", "1"),
                     ("--experiment", "shared_graph_encodings"), ("--collection-workers", "1"),
                     ("--ho", "1"), ("start",), ("--hours",)):
            with self.subTest(args=args):
                result = self.run_wrapper(*args)
                self.assert_runtime(result, 2)
                self.assertFalse((self.root / "arguments.json").exists())

    def test_duplicate_options_never_invoke_controller(self):
        for args in (("--check", "--check"), ("--hours", "1", "--hours=2"),
                     ("--cpu-cap=4", "--cpu-cap=8"),
                     ("--recovery-receipt", "one.json", "--recovery-receipt", "two.json")):
            with self.subTest(args=args):
                result = self.run_wrapper(*args)
                self.assert_runtime(result, 2)
                self.assertIn("repeated options", result.stderr)
                self.assertFalse((self.root / "arguments.json").exists())

    def test_invalid_hours_and_cpu_caps_never_invoke_controller(self):
        cases = [("--hours", value) for value in ("0", "-1", "0.01", "337", "nan", "inf", "oops")]
        cases += [("--cpu-cap", value) for value in ("0", "-1", "257", "1.5", "oops")]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_wrapper(*args)
                self.assert_runtime(result, 2)
                self.assertFalse((self.root / "arguments.json").exists())

    def test_help_prints_runtime_without_invoking_controller(self):
        result = self.run_wrapper("--help")
        self.assert_runtime(result, 0)
        self.assertIn("target_games_met", result.stdout)
        self.assertFalse((self.root / "arguments.json").exists())

    def test_child_failure_status_is_preserved_and_runtime_is_last(self):
        result = self.run_wrapper(MOCK_STATUS="7")
        self.assert_runtime(result, 7)
        self.assertLess(result.stdout.index("Mock controller completed"),
                        result.stdout.index("Total running time"))
        self.assertNotIn("1000 games completed", result.stdout)

    def test_missing_controller_prints_elapsed_time_and_start_error(self):
        (self.root / "continue-original-native.sh").unlink()
        result = self.run_wrapper()
        self.assert_runtime(result, 2)
        self.assertIn("Cannot start manual continuation", result.stderr)

    def test_negative_child_signal_status_maps_to_shell_status(self):
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signum=signum):
                result = self.run_wrapper(MOCK_MODE="signal_self", MOCK_SIGNAL=str(signum))
                self.assert_runtime(result, 128 + signum)

    def check_forwarded_stop(self, signum, *, process_group, child_status):
        process = subprocess.Popen(["sh", str(self.wrapper)], cwd=self.root,
            env={**self.env, "MOCK_MODE": "wait_for_stop", "MOCK_STATUS": str(child_status)},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / "ready").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    self.fail("Disposable controller did not become ready before its deadline")
                time.sleep(0.01)
            sent = time.monotonic()
            if process_group:
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
            stdout, stderr = process.communicate(timeout=8)
            result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            self.assert_runtime(result, child_status)
            self.assertGreaterEqual(time.monotonic() - sent, 0.12)
            cleaned = json.loads((self.root / "cleaned.json").read_text())
            self.assertIn(signum, cleaned["signals"])
            self.assertEqual(cleaned["status"], child_status)
            self.assertLess(stdout.index("Mock controller completed"), stdout.index("Total running time"))
            self.assertEqual(len(re.findall("Total running time:", stdout)), 1)
        finally:
            if process.poll() is None:
                # This group was created above exclusively for disposable mocks.
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)

    def test_direct_pid_sigterm_reaches_controller_and_waits_for_cleanup(self):
        self.check_forwarded_stop(signal.SIGTERM, process_group=False, child_status=0)

    def test_terminal_group_sigint_waits_for_cleanup_and_preserves_status(self):
        self.check_forwarded_stop(signal.SIGINT, process_group=True, child_status=9)


if __name__ == "__main__":
    unittest.main()
