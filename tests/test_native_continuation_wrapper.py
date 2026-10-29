"""Manual wrapper argument and ownership boundaries; no runner or game starts."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


class NativeContinuationWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wrapper = self.root / "continue-original-native.sh"
        shutil.copyfile(ROOT / self.wrapper.name, self.wrapper)
        self.args_file = self.root / "called-arguments"
        child = self.root / "train-local.sh"
        child.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENTS_FILE"\n')
        child.chmod(0o755)
        self.campaign = self.root / "Projects/Heroes-artifacts/local-script-training"
        (self.campaign / "original").mkdir(parents=True)
        self.locks = (self.campaign / "owner.lock", self.campaign / "original/owner.lock")
        for lock in self.locks:
            lock.write_text("unchanged-owner\n")
        self.env = {**os.environ, "HOME": str(self.root), "ARGUMENTS_FILE": str(self.args_file)}
        self.required = ("--native-learner", "/private/native build/manifest.json", "--native-sha256", SHA)

    def run_wrapper(self, *args):
        return subprocess.run(["sh", str(self.wrapper), *args], env=self.env,
                              text=True, capture_output=True, timeout=5)

    def called(self):
        return self.args_file.read_text().splitlines()

    def test_default_uses_current_continuation_and_explicit_native_identity(self):
        result = self.run_wrapper(*self.required)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.called()
        self.assertEqual(args[0], "start")
        self.assertIn("--fast-path", args)
        for key, expected in (("--native-learner", self.required[1]), ("--native-sha256", SHA),
                              ("--hours", "1"), ("--profile", "original"),
                              ("--interaction-guard", "visible-interaction-budget-v1")):
            self.assertEqual(args[args.index(key) + 1], expected)
        self.assertEqual(args[args.index("--continue-original") + 1],
                         str(self.campaign / "original/run-20261026T181133Z-062cd75a"))
        for forbidden in ("--games", "--parent", "--parent-final", "--recovery-receipt", "--max-updates",
                          "--prefix-helpers", "--replay-threads", "--experiment", "--collection-workers"):
            self.assertNotIn(forbidden, args)
        self.assertFalse(any("1229" in value for value in args))
        for lock in self.locks:
            self.assertEqual(lock.read_text(), "unchanged-owner\n")

    def test_check_hours_cpu_cap_and_explicit_recovery_are_forwarded_once(self):
        result = self.run_wrapper(*self.required, "--check", "--hours", "10", "--cpu-cap", "8",
                                  "--recovery-receipt", "/private/new recovery/receipt.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.called()
        self.assertEqual(args[0], "check")
        for key, expected in (("--hours", "10"), ("--cpu-cap", "8"),
                              ("--recovery-receipt", "/private/new recovery/receipt.json")):
            self.assertEqual(args.count(key), 1)
            self.assertEqual(args[args.index(key) + 1], expected)

    def test_help_does_not_need_locks_or_native_identity(self):
        for lock in self.locks:
            lock.unlink()
        result = self.run_wrapper("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("latest complete state", result.stdout)
        self.assertFalse(self.args_file.exists())
        self.assertTrue(all(not lock.exists() for lock in self.locks))

    def test_unknown_conflicting_and_duplicate_flags_never_reach_runner(self):
        invalid = [("--config", "/private/other.json"), ("--games", "4"), ("--parent", "old.json"),
                   ("--continue-original", "old-run"), ("--collection-workers", "1"),
                   ("--experiment", "shared_graph_encodings"), ("--overlap-verification",),
                   ("--replay-threads", "2"), ("_launch",), ("start",), ("--",),
                   ("--hours", "1", "--hours", "10"), ("--check", "--check"),
                   ("--native-sha256", SHA), ("--hours",), ("--cpu-cap", "--check")]
        for args in invalid:
            with self.subTest(args=args):
                result = self.run_wrapper(*self.required, *args)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.args_file.exists())

    def test_missing_or_malformed_native_binding_rejects(self):
        for args in ((), ("--native-learner", "relative.json", "--native-sha256", SHA),
                     ("--native-learner", "/manifest.json"),
                     ("--native-learner", "/manifest.json", "--native-sha256", "A" * 64),
                     ("--native-learner", "/manifest.json", "--native-sha256", "a" * 63)):
            with self.subTest(args=args):
                self.assertEqual(self.run_wrapper(*args).returncode, 2)
                self.assertFalse(self.args_file.exists())

    def test_missing_symlinked_or_readonly_locks_block_start_and_check_unchanged(self):
        if os.geteuid() == 0:
            self.skipTest("Root can bypass the cooperative-user filesystem fence")
        for lock in self.locks:
            for mode in ("readonly", "missing", "symlink"):
                original = lock.read_bytes()
                if mode == "readonly":
                    lock.chmod(0o444)
                else:
                    lock.unlink()
                    if mode == "symlink":
                        target = self.root / "other-owner"
                        target.write_bytes(original)
                        lock.symlink_to(target)
                try:
                    for command in ((), ("--check",)):
                        with self.subTest(lock=lock, mode=mode, command=command):
                            result = self.run_wrapper(*self.required, *command)
                            self.assertEqual(result.returncode, 2)
                            self.assertIn("ownership", result.stderr)
                            self.assertFalse(self.args_file.exists())
                    if mode == "missing":
                        self.assertFalse(lock.exists())
                    else:
                        self.assertEqual(lock.read_bytes(), original)
                        if mode == "readonly": self.assertEqual(lock.stat().st_mode & 0o777, 0o444)
                        else: self.assertTrue(lock.is_symlink())
                finally:
                    if lock.is_symlink(): lock.unlink()
                    if lock.exists(): lock.chmod(0o644)
                    lock.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
