"""Offline benchmark CLI deadlines; all work is mocked, no native replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import validate_rollout_storage as benchmark


class StorageBenchmarkDeadlineTests(unittest.TestCase):
    def test_minimum_parent_deadline_can_reach_its_worker_after_cleanup_reserve(self):
        # A 60-second parent gives its child 50 seconds and retains ten for cleanup.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['validate_rollout_storage.py', '--worker', '--run', str(root / 'source'),
                    '--output', str(root / 'copy'), '--seconds', '50']
            with patch('sys.argv', argv), patch.object(benchmark, 'private_path', return_value=root / 'copy'), \
                 patch.object(benchmark, 'worker', return_value={'status': 'passed'}) as worker:
                self.assertEqual(benchmark.main(), 0)
                self.assertEqual(worker.call_args.args[2], 50)
            for mode, seconds in (([], '50'), (['--worker'], '49'), ([], 'nan'), ([], '901')):
                argv = ['validate_rollout_storage.py', *mode, '--run', str(root / 'source'),
                        '--output', str(root / 'copy'), '--seconds', seconds]
                with self.subTest(mode=mode, seconds=seconds), patch('sys.argv', argv), \
                     patch.object(benchmark, 'worker') as worker, self.assertRaises(ValueError):
                    benchmark.main()
                worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
