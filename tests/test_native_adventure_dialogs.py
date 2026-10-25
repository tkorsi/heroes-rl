"""Compile canonical v7 normalizers against real VCMI; never launch a game."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments.learner import contextual as c

ROOT = Path(__file__).resolve().parents[1]


class NativeAdventureDialogTests(unittest.TestCase):
    def test_dialog_families_flags_pools_and_legacy_notice_isolation(self):
        self.check_native(static=False)

    def test_displayed_scholar_and_artifact_content_without_latent_data(self):
        self.check_native(static=True)

    def check_native(self, *, static):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.is_file():
            self.skipTest("Requires the prepared real VCMI library and C++ compiler")
        data = Path(os.environ.get("HEROES_VCMI_DATA", Path.home() / (
            "Library/Application Support/vcmi" if sys.platform == "darwin" else ".local/share/vcmi")))
        if static and not (data / "Data").is_dir():
            self.skipTest("Requires original local Data to resolve static displayed content")
        with tempfile.TemporaryDirectory(prefix="heroes-native-dialogs-") as directory:
            temporary = Path(directory)
            overlay = temporary / "AI/ExplorerRL"; overlay.mkdir(parents=True)
            for name in ("Dialog.h", "Dialog.cpp", "Notice.h", "Notice.cpp", "StdInc.h"):
                shutil.copyfile(ROOT / "engine/ExplorerRL" / name, overlay / name)
            for name in ("lib", "Global.h"):
                (temporary / name).symlink_to(engine / name)
            executable = temporary / "adventure-dialogs"
            includes = [overlay, engine, engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").is_dir(): includes.append(Path("/opt/homebrew/include"))
            command = [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare",
                       *(f"-I{path}" for path in includes), str(ROOT / "tests/native/adventure_dialogs.cpp"),
                       str(overlay / "Dialog.cpp"), str(overlay / "Notice.cpp"),
                       f"-L{library.parent}", "-lvcmi", f"-Wl,-rpath,{library.parent}", "-o", str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            environment = dict(os.environ)
            if static:
                profile = temporary / "profile"
                data_root = profile if sys.platform == "darwin" else profile / "data/vcmi"
                data_root.mkdir(parents=True)
                (data_root / "Data").symlink_to((data / "Data").resolve(), target_is_directory=True)
                environment = {key: value for key, value in environment.items()
                               if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
                environment.update({"VCMI_USER_DATA_DIR": str(profile)} if sys.platform == "darwin" else
                                   {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
                                    "XDG_CACHE_HOME": str(profile / "cache")})
            result = subprocess.run([str(executable), *(["--static"] if static else [])], cwd=library.parent,
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertRegex(result.stdout, r"native adventure normalizers: [1-9][0-9]+ checks passed")
            payloads = json.loads(result.stdout.split("PAYLOADS_BEGIN\n", 1)[1].split("\nPAYLOADS_END", 1)[0])
            for panel in payloads["dialogs"]:
                c.decode_dialog_panel(panel, schema=c.DIALOGS_SCHEMA)
            for notice in payloads["notices"]:
                c.decode_notice(notice, schema=c.DIALOGS_SCHEMA)


if __name__ == "__main__": unittest.main()
