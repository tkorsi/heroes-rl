"""Compile the production request-queue template in isolation, with no game launch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeShutdownTests(unittest.TestCase):
    def test_pending_and_late_requests_observe_termination(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        header = ROOT / "vcmi/client/Client.h"
        if compiler is None or not header.is_file():
            self.skipTest("Requires a C++ compiler and the prepared VCMI dependency")
        source = header.read_text()
        begin = source.index("template<typename T>\nclass ThreadSafeVector")
        end = source.index("\nclass CPlayerEnvironment", begin)
        with tempfile.TemporaryDirectory(prefix="heroes-shutdown-test-") as directory:
            path = Path(directory)
            # Extract the exact production template, rather than copying its logic
            # into a test. The surrounding fixture only supplies standard helpers.
            (path / "threadsafevector_under_test.h").write_text(source[begin:end])
            executable = path / "shutdown-regression"
            subprocess.run([compiler, "-std=c++20", "-O2", "-DNDEBUG", "-pthread", "-I", str(path),
                            str(ROOT / "tests/native/threadsafevector_shutdown.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=5)
            self.assertIn("512 racing senders", result.stdout)


if __name__ == "__main__":
    unittest.main()
