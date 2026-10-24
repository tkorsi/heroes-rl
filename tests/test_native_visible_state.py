"""Compile the production visibility tracker against synthetic team-fog worlds."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("headless_replay", ROOT / "experiments/headless/replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def two_sided_world_update() -> dict:
    """tests/native/visible_state.cpp twoSidedWorld() as the bridge reports its first capture."""
    objects = {1: (1, 1, 0, 34, 0, 0), 2: (2, 3, 0, 34, 1, 1), 4: (1, 4, 0, 53, 0, -1)}
    occupied = {(x, y) for x, y, *_ in objects.values()}
    revealed = [{"x": x, "y": y, "z": 0, "terrain": (x + y) % 10, "blocked": False, "visitable": (x, y) in occupied}
                for y in range(5) for x in range(3)]
    return {"kind": "visible_update", "sequence": 0, "day": 1,
            "appeared": [dict(zip(("id", "x", "y", "z", "type", "subtype", "owner"), (key, *value)))
                         for key, value in objects.items()],
            "changed": [], "disappeared": [], "revealed": revealed, "tiles_changed": [], "hidden": [],
            "own": {"hero": [1, 1, 0], "movement": 1500, "movement_max": 1500,
                    "resources": [20, 5, 20, 5, 5, 5, 7500], "army": [{"slot": 0, "creature": 1, "count": 10}],
                    "towns": [], "mines": 0}}


class NativeVisibleStateTests(unittest.TestCase):
    def test_sanitized_changes_reconstruct_only_the_players_own_view(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("Requires a C++ compiler")
        with tempfile.TemporaryDirectory(prefix="heroes-visible-state-test-") as directory:
            executable = Path(directory) / "visible-state-regression"
            # The header is the exact overlay file ExplorerRL compiles; it has
            # no engine dependency, so no prepared VCMI checkout is needed.
            subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/ExplorerRL"),
                            str(ROOT / "tests/native/visible_state.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True, timeout=10)
            self.assertIn("10 scenarios, 200 randomized folds", result.stdout)
            native = re.search(r"capture digest ([0-9a-f]{16})", result.stdout).group(1)
            fold = replay.VisibleFold([6, 5, 1])
            fold.apply(two_sided_world_update())
            # The replay verifier ties received updates to engine captures with this mirror.
            self.assertEqual(fold.digest(), native)


if __name__ == "__main__":
    unittest.main()
