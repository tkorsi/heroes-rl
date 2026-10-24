"""Compile production routing and terminal aggregation without launching the game."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeSelfPlayTests(unittest.TestCase):
    def test_strict_player_routing_and_all_terminal_barrier(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        implementation = ROOT / "vcmi/client/Client.cpp"
        if compiler is None or not implementation.is_file():
            self.skipTest("Requires a C++ compiler and the prepared VCMI dependency")
        source = implementation.read_text()
        begin = source.index("std::optional<PlayerColor> CClient::configuredExplorerPlayer() const")
        end = source.index("\nstd::string CClient::aiNameForPlayer", begin)
        with tempfile.TemporaryDirectory(prefix="heroes-self-play-test-") as directory:
            path = Path(directory)
            # Compile the exact production methods. The fixture supplies only a
            # player value type and logging, not replacement routing logic.
            (path / "self_play_under_test.h").write_text(source[begin:end])
            executable = path / "self-play-regression"
            subprocess.run([compiler, "-std=c++20", "-O2", "-DNDEBUG", "-I", str(path),
                            str(ROOT / "tests/native/self_play_routing.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True, timeout=5)
            self.assertIn("both terminal orders", result.stdout)


if __name__ == "__main__":
    unittest.main()
