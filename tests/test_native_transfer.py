"""Compile the exact production transfer arithmetic; no fake engine or game."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeTransferRulesTests(unittest.TestCase):
    def test_exact_transfer_and_native_integer_boundaries(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-transfer-test-") as directory:
            executable = Path(directory) / "transfer-rules"
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"), str(ROOT / "tests/native/transfer_rules.cpp"),
                            "-o", str(executable)], check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("exact split, merge, capacity, minimum army and int32 boundaries passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
