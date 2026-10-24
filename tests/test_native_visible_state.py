"""Compile the production visibility tracker against synthetic team-fog worlds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeVisibleStateTests(unittest.TestCase):
    def test_sanitized_changes_reconstruct_only_the_players_own_view(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-visible-state-test-") as directory:
            executable = Path(directory) / "visible-state-regression"
            # The header is the exact overlay file ExplorerRL compiles; it has
            # no engine dependency, so no prepared VCMI checkout is needed.
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/visible_state.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True, timeout=10)
            self.assertIn("7 scenarios, 200 randomized folds", result.stdout)


if __name__ == "__main__":
    unittest.main()
