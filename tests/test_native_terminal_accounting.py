"""Compile actual terminal request admission and native codecs; no game/replay."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_builtin_native_contract import function

ROOT = Path(__file__).resolve().parents[1]


class NativeTerminalAccountingTests(unittest.TestCase):
    def test_exact_native_packets_and_terminal_suffix_mutations(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists():
            self.skipTest("Requires compiler and prepared, built libvcmi")
        source = (engine / "server/ExperimentReplay.cpp").read_text()
        # Actual production buffer classes and packet codecs, not surrogate
        # identity parsing. Linking libvcmi does not run an engine or a map.
        codec = "\n".join(function(source, signature) + (";" if signature.startswith("class") else "")
            for signature in ("class BufferWriter", "class BufferReader", "std::vector<std::byte> encodePack(",
                              "std::unique_ptr<CPack> decodePack(", "std::string toHex(", "std::vector<std::byte> fromHex("))
        allowed = function((ROOT / "engine/ExperimentReplay/ContextualReplay.cpp").read_text(), "bool automaticAllowed(")
        fixture = (ROOT / "tests/native/terminal_request_accounting.cpp").read_text()
        with tempfile.TemporaryDirectory(prefix="heroes-terminal-accounting-") as directory:
            temporary = Path(directory)
            overlay = temporary / "ExperimentReplay"
            overlay.mkdir()
            for name in ("StdInc.h", "RequestAccounting.h"):
                shutil.copyfile(ROOT / "engine/ExperimentReplay" / name, overlay / name)
            for name in ("lib", "Global.h"):
                (temporary / name).symlink_to(engine / name)
            path = overlay / "test.cpp"
            path.write_text(fixture.replace("// PRODUCTION_PACKET_CODEC", codec).replace("// PRODUCTION_AUTOMATIC_ALLOWED", allowed))
            binary = temporary / "test"
            includes = [overlay, engine, engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            built = subprocess.run([compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare", "-Wno-reorder",
                *(f"-I{p}" for p in includes), str(path), f"-L{library.parent}", "-lvcmi",
                f"-Wl,-rpath,{library.parent}", "-o", str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("460 checks passed", result.stdout)

    def test_verified_range_is_granted_only_after_final_native_state_and_effect_checks(self):
        source = (ROOT / "engine/ExperimentReplay/ReplayMain.cpp").read_text()
        handoff = source.index("contextual->verifiedTerminalInput(")
        self.assertLess(source.index("expectChain(after, applied, server.outgoing)"), handoff)
        self.assertLess(source.index('expectEqual(after, "terminals"'), handoff)
        self.assertLess(source.index("expectState(after, applied, *gameHandler)"), handoff)
        guards = source[source.index("if(contextual && !prefix", source.index("expectState(after, applied")):handoff]
        for expected in ('index + 2 == records.size()', 'std::string(boundary) == "game_end"',
                         'gameHandler->experimentTerminalBarrierSent()'):
            self.assertIn(expected, guards)
        client = (ROOT / "engine/ExperimentReplay/ContextualReplay.cpp").read_text()
        self.assertIn('result["request_accounting"] = accounting', client)
        self.assertIn('client.acceptExperimentReplayAutomaticRequest(*request)', client)
        self.assertNotIn('handleReceivedPack(', client)


if __name__ == "__main__":
    unittest.main()
