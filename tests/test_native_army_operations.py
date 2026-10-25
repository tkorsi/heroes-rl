"""Compile production army arithmetic; these are not simulated VCMI episodes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeArmyOperationRulesTests(unittest.TestCase):
    def test_exact_native_effects_and_boundaries(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-army-rules-") as directory:
            executable = Path(directory) / "army-rules"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/army_operation_rules.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("explicit army operations, garrison effects and upgrade costs passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
