"""Compile production garrison permission rules without starting VCMI."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeGarrisonAccessTests(unittest.TestCase):
    def test_garrison_ownership_and_lock_boundaries(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-garrison-access-") as directory:
            executable = Path(directory) / "garrison-access"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/garrison_access.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("garrison ownership, locking, gifts and last-stack boundaries passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
