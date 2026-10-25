"""Actual native producer/task drain order; no simulated gameplay or Data."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeHeadlessShutdownOrderTests(unittest.TestCase):
    def test_ai_tasks_drain_before_network_producer_exits(self):
        binary = ROOT / 'vcmi/build/bin/native_headless_shutdown_order'
        if not binary.exists():
            self.skipTest('build native_headless_shutdown_order against the prepared client')
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('100 actual headless shutdown order checks passed', result.stdout)
        self.assertIn('no gameplay simulated', result.stdout)


if __name__ == '__main__':
    unittest.main()
