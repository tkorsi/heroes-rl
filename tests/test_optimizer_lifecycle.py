"""Compile and exercise the real Linux process reset primitive without assets."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('g++'), 'requires Linux and g++')
class ForkLifecycleTests(unittest.TestCase):
    def test_process_memory_rng_descriptors_deadlines_and_reaping(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'lifecycle'
            asset = Path(directory) / 'readonly.txt'
            asset.write_text('immutable resource')
            subprocess.run(['g++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-pthread',
                            '-I', str(ROOT / 'engine/ExperimentReplay'),
                            str(ROOT / 'tests/native/fork_episode.cpp'), '-o', str(binary)],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(binary), str(asset)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
