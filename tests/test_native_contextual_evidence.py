"""Exercise production evidence ownership against real libvcmi, without a game."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeContextualEvidenceTests(unittest.TestCase):
    def test_explicit_release_and_process_teardown(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists() or not (engine / "lib/callback/ExperimentHooks.h").exists():
            self.skipTest("Requires a C++ compiler and prepared, locally built real VCMI library")
        with tempfile.TemporaryDirectory(prefix="heroes-evidence-lifetime-") as directory:
            temporary = Path(directory)
            overlay = temporary / "AI/ExplorerRL"
            overlay.mkdir(parents=True)
            # Compile canonical production source, never a stale generated copy.
            for name in ("ContextualEvidence.h", "ContextualEvidence.cpp", "StdInc.h"):
                shutil.copyfile(ROOT / "engine/ExplorerRL" / name, overlay / name)
            for name in ("lib", "Global.h"):
                (temporary / name).symlink_to(engine / name)
            includes = [overlay, engine, engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            executable = temporary / "evidence-lifetime"
            command = [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare", "-Wno-reorder", "-pthread",
                       *(f"-I{path}" for path in includes),
                       str(ROOT / "tests/native/contextual_evidence_lifetime.cpp"),
                       str(overlay / "ContextualEvidence.cpp"),
                       f"-L{library.parent}", "-lvcmi", f"-Wl,-rpath,{library.parent}", "-o", str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("EXPLORER_RL_", "VCMI_EXPERIMENT_", "VCMI_RECORD_"))}
            for mode in ("clear", "registry-exit", "live-exit", "session-close"):
                with self.subTest(mode=mode):
                    result = subprocess.run([str(executable), mode, str(temporary)], env=env,
                                            capture_output=True, text=True, timeout=10)
                    # The message precedes static destruction; process exit is
                    # required too, so a teardown crash cannot count as success.
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"contextual evidence lifetime: {mode} passed", result.stdout)
            header = json.loads((temporary / "contextual-replay.jsonl").read_text())
            self.assertEqual(header["schema"], "heroes-contextual-replay-v1")
            self.assertEqual(header["episode"], "lifetime-only")


if __name__ == "__main__":
    unittest.main()
