"""Compile the production knowledge grid against synthetic fogged worlds."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeKnowledgeGridTests(unittest.TestCase):
    def test_guards_and_steps_come_only_from_explored_tiles(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-knowledge-grid-test-") as directory:
            executable = Path(directory) / "knowledge-grid-regression"
            # The header is the exact overlay file the planner compiles; it has
            # no engine dependency, so no prepared VCMI checkout is needed.
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/knowledge_grid.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("knowledge grid: 5 groups, 200 hidden-variant folds passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
