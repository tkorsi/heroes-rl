"""The fixed-cohort launcher uses disposable controller processes only."""
import os
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path

from tests import test_train_1000_launcher as fixtures


ROOT = Path(__file__).resolve().parents[1]


class Continue1585LauncherTests(unittest.TestCase):
    run_wrapper = fixtures.Train1000LauncherTests.run_wrapper
    called = fixtures.Train1000LauncherTests.called
    assert_runtime = fixtures.Train1000LauncherTests.assert_runtime
    check_forwarded_stop = fixtures.Train1000LauncherTests.check_forwarded_stop

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heroes-1176-launcher-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tools').mkdir()
        self.wrapper = self.root / 'continue-1585.sh'
        for name in ('continue-1585.sh', 'tools/continue_1585_games.py', 'tools/train_1000_games.py'):
            shutil.copyfile(ROOT / name, self.root / name)
        controller = self.root / 'continue-original-native.sh'
        controller.write_text(f'#!{sys.executable}\n' + fixtures.MOCK_CONTROLLER)
        controller.chmod(0o755)
        self.env = {**os.environ, 'HOME': str(self.root), 'HEROES_PYTHON': sys.executable,
                    'MOCK_ROOT': str(self.root)}

    def test_default_fixes_goal_and_native_build_with_ten_hour_limit(self):
        result = self.run_wrapper()
        self.assert_runtime(result, 0)
        args = self.called()
        for key, expected in (('--session-goal', 'checkpoint1585-remaining1176-v1'),
                              ('--native-sha256', fixtures.NATIVE_SHA256), ('--hours', '10'),
                              ('--interaction-guard', 'visible-interaction-budget-revealed-v1')):
            self.assertEqual(args.count(key), 1)
            self.assertEqual(args[args.index(key)+1], expected)
        self.assertNotIn('--session-games', args)
        self.assertNotIn('--parent', args)
        self.assertIn('294 complete PPO updates', result.stdout)

    def test_check_hours_and_cpu_cap_forward_once(self):
        result = self.run_wrapper('--check', '--hours', '1.5', '--cpu-cap', '8')
        self.assert_runtime(result, 0)
        args = self.called()
        self.assertEqual(args.count('--check'), 1)
        self.assertEqual(args[args.index('--hours')+1], '1.5')
        self.assertEqual(args[args.index('--cpu-cap')+1], '8')

    def test_overrides_duplicates_and_invalid_limits_fail_before_controller(self):
        for args in (('--session-games', '773'), ('--session-goal', 'other'), ('--parent', 'old'),
                     ('--games', '1176'), ('--interaction-guard', 'visible-interaction-budget-v1'), ('--config', 'other'), ('--recovery-receipt', 'receipt'),
                     ('--collection-workers', '1'), ('--hours', '0'), ('--hours', 'nan'),
                     ('--cpu-cap', '257'), ('--check', '--check'), ('--hours', '1', '--hours=2')):
            with self.subTest(args=args):
                result = self.run_wrapper(*args)
                self.assert_runtime(result, 2)
                self.assertFalse((self.root / 'arguments.json').exists())

    def test_help_does_not_invoke_controller(self):
        result = self.run_wrapper('--help')
        self.assert_runtime(result, 0)
        self.assertIn('same 1176-game goal', result.stdout)
        self.assertFalse((self.root / 'arguments.json').exists())

    def test_failure_status_and_elapsed_time_survive(self):
        result = self.run_wrapper(MOCK_STATUS='7')
        self.assert_runtime(result, 7)
        self.assertLess(result.stdout.index('Mock controller completed'), result.stdout.index('Total running time'))

    def test_owned_controller_receives_sigterm_and_finishes_cleanup(self):
        self.check_forwarded_stop(signal.SIGTERM, process_group=False, child_status=0)


if __name__ == '__main__':
    unittest.main()
