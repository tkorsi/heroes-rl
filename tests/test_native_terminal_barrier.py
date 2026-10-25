"""Actual native transaction hooks and evidence writer; no game is launched."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeTerminalBarrierTests(unittest.TestCase):
    def test_callback_transactions_and_failure_latching(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists() or not (engine / "lib/callback/ExperimentHooks.h").exists():
            self.skipTest("Requires a C++ compiler and prepared, locally built real VCMI library")
        with tempfile.TemporaryDirectory(prefix="heroes-terminal-hooks-") as directory:
            temporary = Path(directory)
            includes = [engine, engine / "AI/ExplorerRL", engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            executable = temporary / "terminal-hooks"
            command = [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare", "-Wno-reorder", "-pthread",
                       *(f"-I{path}" for path in includes), str(ROOT / "tests/native/terminal_barrier.cpp"),
                       f"-L{library.parent}", "-lvcmi", f"-Wl,-rpath,{library.parent}", "-o", str(executable)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("EXPLORER_RL_", "VCMI_EXPERIMENT_", "VCMI_RECORD_"))}
            result = subprocess.run([str(executable), str(temporary)], env=env, check=True,
                                    capture_output=True, text=True, timeout=15)
            self.assertIn("native terminal barrier hooks: 23 checks passed", result.stdout)
            server, client = (json.loads((temporary / f"terminal-barrier-{side}.jsonl").read_text())
                              for side in ("server", "client"))
            self.assertEqual(server["schema"], "heroes-contextual-terminal-barrier-v1")
            self.assertEqual(server["outcomes"], ["win", "loss"])
            self.assertEqual(server["request"], {"player": -1, "id": 7, "type": 198})
            self.assertTrue(client.pop("callbacks_complete"))
            self.assertEqual(client.pop("observed_count"), server["client_packets"])
            self.assertEqual(client.pop("kind"), "client")
            self.assertEqual(server.pop("kind"), "server")
            self.assertEqual(client, server)


if __name__ == "__main__":
    unittest.main()
