"""Compile the current notice parser against the real, locally built VCMI library."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeNoticeTests(unittest.TestCase):
    def test_exact_templates_pools_and_relative_fields(self):
        self._check_native(trophies=False)

    def test_trophies_with_actual_static_game_content(self):
        self._check_native(trophies=True)

    def _check_native(self, *, trophies):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists() or not (engine / "lib/texts/MetaString.h").exists():
            self.skipTest("Requires a C++ compiler and the prepared, locally built real VCMI library")
        data = Path(os.environ.get("HEROES_VCMI_DATA", Path.home() / (
            "Library/Application Support/vcmi" if sys.platform == "darwin" else ".local/share/vcmi")))
        if trophies and not (data / "Data").is_dir():
            self.skipTest("Requires original local game Data for static content; set HEROES_VCMI_DATA to its parent")
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
            if trophies:
                # Keep static-library settings/cache writes in this temporary
                # profile. No game, engine executable, map or bridge is run.
                profile = temporary / "profile"
                if sys.platform == "darwin":
                    data_root = profile
                    profile_env = {"VCMI_USER_DATA_DIR": str(profile)}
                else:
                    data_root = profile / "data/vcmi"
                    profile_env = {"XDG_DATA_HOME": str(profile / "data"),
                                   "XDG_CONFIG_HOME": str(profile / "config"),
                                   "XDG_CACHE_HOME": str(profile / "cache")}
                data_root.mkdir(parents=True)
                (data_root / "Data").symlink_to((data / "Data").resolve(), target_is_directory=True)
                inherited = {key: value for key, value in os.environ.items()
                             if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
                result = subprocess.run([str(executable), "--trophies"], cwd=library.parent,
                                        env={**inherited, **profile_env}, check=True,
                                        capture_output=True, text=True, timeout=30)
                self.assertIn("native trophy notice parser: 42 checks passed", result.stdout)
            else:
                result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
                self.assertIn("native notice parser: 98 checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
