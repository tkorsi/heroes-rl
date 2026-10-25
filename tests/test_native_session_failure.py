"""Failure admission uses production C++ helpers/channel, without game assets."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("clang++") or shutil.which("g++")


@unittest.skipUnless(COMPILER, "requires a C++20 compiler")
class NativeSessionFailureTests(unittest.TestCase):
    def test_failed_reset_and_late_shutdown_cannot_admit_a_new_episode(self):
        with tempfile.TemporaryDirectory(prefix="native-session-failure-") as temporary:
            binary = Path(temporary) / "lifecycle"
            built = subprocess.run([COMPILER, "-std=c++20", "-pthread", "-Wall", "-Wextra", "-Werror",
                "-I" + str(ROOT / "engine/NativeSession"),
                str(ROOT / "tests/native/session_failure_state.cpp"), "-o", str(binary)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("100 failure/reset", result.stdout)

    def test_real_channel_failure_is_not_a_clean_cancellation(self):
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if not library.exists():
            self.skipTest("requires prepared VCMI and a built libvcmi for the actual JSON implementation")
        with tempfile.TemporaryDirectory(prefix="native-session-channel-failure-") as temporary:
            folder = Path(temporary)
            overlay = folder / "nativesession"
            overlay.mkdir()
            for name in ("PolicyTransport.h", "PolicyTransport.cpp", "StdInc.h"):
                shutil.copyfile(ROOT / "engine/NativeSession" / name, overlay / name)
            for name in ("Global.h", "lib"):
                (folder / name).symlink_to(engine / name)
            includes = [overlay, engine, engine / "include", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            binary = folder / "channel"
            built = subprocess.run([COMPILER, "-std=c++20", "-pthread", "-Wall", "-Wextra", "-Werror",
                "-Wno-unused-parameter", "-Wno-reorder", *("-I" + str(p) for p in includes),
                str(ROOT / "tests/native/session_failure_channel.cpp"), str(overlay / "PolicyTransport.cpp"),
                "-L" + str(library.parent), "-lvcmi", "-Wl,-rpath," + str(library.parent), "-o", str(binary)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("100 error-before-close", result.stdout)


if __name__ == "__main__":
    unittest.main()
