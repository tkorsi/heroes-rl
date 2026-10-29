"""Check interpreter selection and argument preservation without running training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainLocalLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="heroes-local-launcher-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.checkout = self.base / "checkout with spaces"
        self.checkout.mkdir()
        self.wrapper = self.checkout / "train-local.sh"
        shutil.copyfile(ROOT / "train-local.sh", self.wrapper)
        self.home = self.base / "home with spaces"
        self.record = self.base / "called.json"
        recorder = self.base / "record.py"
        recorder.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['MOCK_RECORD']).write_text(json.dumps({"
            "'interpreter': sys.argv[1], 'arguments': sys.argv[2:], 'cwd': os.getcwd()}))\n"
            "raise SystemExit(int(os.environ.get('MOCK_STATUS', '0')))\n"
        )
        self.old_python = self.home / "Projects/Heroes/.venv/bin/python"
        self.new_python = self.home / (
            "Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"
        )
        self.override_python = self.base / "override interpreter"
        for path, identity in ((self.old_python, "old"), (self.new_python, "new"),
                               (self.override_python, "override")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#!/bin/sh\nexec "$MOCK_REAL_PYTHON" "$MOCK_RECORDER" '
                            + identity + ' "$@"\n')
            path.chmod(0o755)
        self.environment = {
            **os.environ, "HOME": str(self.home), "MOCK_REAL_PYTHON": sys.executable,
            "MOCK_RECORDER": str(recorder), "MOCK_RECORD": str(self.record),
        }
        self.environment.pop("HEROES_PYTHON", None)

    def assert_launch(self, identity, arguments, *, environment=None, status=0):
        result = subprocess.run(
            ["sh", str(self.wrapper), *arguments], cwd=self.base,
            env={**self.environment, **(environment or {})}, text=True,
            capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, status, result.stderr)
        self.assertEqual(json.loads(self.record.read_text()), {
            "interpreter": identity,
            "arguments": ["tools/run_local_training.py", *arguments],
            "cwd": str(self.checkout.resolve()),
        })

    def test_default_keeps_historical_interpreter(self):
        self.assert_launch("old", ["check"])

    def test_easier_profile_keeps_historical_interpreter(self):
        for profile in (["--profile", "easier8x8"], ["--profile=easier8x8"]):
            with self.subTest(profile=profile):
                self.assert_launch("old", ["check", *profile])

    def test_original_profile_uses_prepared_environment(self):
        self.assert_launch("new", ["check", "--profile", "original"])

    def test_equals_original_profile_preserves_all_arguments(self):
        self.assert_launch("new", ["check", "--profile=original", "--config",
                                   "custom configuration.json", "", "literal*", "line\nbreak"])

    def test_override_wins_for_every_profile_and_custom_configuration(self):
        for profile in ([], ["--profile", "easier8x8"], ["--profile", "original"],
                        ["--profile=original"]):
            with self.subTest(profile=profile):
                self.assert_launch("override", ["check", *profile, "--config", "custom.json"],
                                   environment={"HEROES_PYTHON": str(self.override_python)})

    def test_empty_override_uses_profile_default(self):
        self.assert_launch("old", ["check"], environment={"HEROES_PYTHON": ""})
        self.assert_launch("new", ["check", "--profile=original"],
                           environment={"HEROES_PYTHON": ""})

    def test_last_explicit_profile_matches_argument_parser_selection(self):
        self.assert_launch("old", ["check", "--profile=original", "--profile", "easier8x8"])
        self.assert_launch("new", ["check", "--profile", "easier8x8", "--profile=original"])

    def test_option_terminator_does_not_select_an_interpreter(self):
        self.assert_launch("old", ["check", "--", "--profile=original"])

    def test_custom_configuration_does_not_select_an_executable(self):
        config = self.base / "custom.json"
        config.write_text(json.dumps({"python": str(self.override_python)}))
        self.assert_launch("old", ["check", "--config", str(config)])

    def test_interpreter_exit_status_is_preserved(self):
        self.assert_launch("new", ["check", "--profile=original"],
                           environment={"MOCK_STATUS": "7"}, status=7)


if __name__ == "__main__":
    unittest.main()
