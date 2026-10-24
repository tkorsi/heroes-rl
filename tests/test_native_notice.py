"""Compile the production notice parser against audited and near-miss templates."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeNoticeParserTests(unittest.TestCase):
    def test_only_audited_templates_become_typed_notices(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-notice-parser-test-") as directory:
            executable = Path(directory) / "notice-parser-regression"
            # The header is the exact overlay file ExplorerRL compiles; it has
            # no engine dependency, so no prepared VCMI checkout is needed.
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/notice_parser.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertRegex(result.stdout, r"notice parser: 6 groups, \d+ checks passed")


if __name__ == "__main__":
    unittest.main()
