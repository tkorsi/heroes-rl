"""Compile the production context state against audited and near-miss templates."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeContextStateTests(unittest.TestCase):
    def test_change_stream_rebuilds_every_capture(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-context-state-test-") as directory:
            executable = Path(directory) / "context-state-regression"
            # The header is the exact overlay file ExplorerRL compiles; it has
            # no engine dependency, so no prepared VCMI checkout is needed.
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/context_state.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertRegex(result.stdout, r"context state: 3 groups, \d+ checks passed")


if __name__ == "__main__":
    unittest.main()
