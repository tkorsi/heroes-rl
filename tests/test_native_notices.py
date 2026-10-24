"""Compile the current notice parser against the real, locally built VCMI library."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeNoticeTests(unittest.TestCase):
    def test_exact_templates_pools_and_relative_fields(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists() or not (engine / "lib/texts/MetaString.h").exists():
            self.skipTest("Requires a C++ compiler and the prepared, locally built real VCMI library")
        with tempfile.TemporaryDirectory(prefix="heroes-native-notices-") as directory:
            temporary = Path(directory)
            overlay = temporary / "AI/ExplorerRL"
            overlay.mkdir(parents=True)
            # Always compile the current canonical parser, never a stale copied
            # AI overlay. Only real engine headers/library are external inputs.
            for name in ("Notice.h", "Notice.cpp", "StdInc.h"):
                shutil.copyfile(ROOT / "engine/ExplorerRL" / name, overlay / name)
            for name in ("lib", "Global.h"):
                (temporary / name).symlink_to(engine / name)
            executable = temporary / "notices"
            includes = [overlay, engine, engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            command = [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare",
                       *(f"-I{path}" for path in includes),
                       str(ROOT / "tests/native/notices.cpp"), str(overlay / "Notice.cpp"),
                       f"-L{library.parent}", "-lvcmi", f"-Wl,-rpath,{library.parent}",
                       "-o", str(executable)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            self.assertIn("native notice parser: 98 checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
