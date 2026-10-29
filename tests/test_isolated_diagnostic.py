import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.isolated_diagnostic import diagnostic_lease, private_output


class DiagnosticLeaseTests(unittest.TestCase):
    def test_control_clis_keep_private_output_spelling_for_alias_rejection(self):
        from tools import validate_garrison_regressions as regression
        from validation.probes import validate_garrison_dialogs as garrison
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root/'alias').symlink_to(root, target_is_directory=True)
            output = root/'alias/new'
            arguments = ['--binary', '/unused/client', '--replay-binary', '/unused/replay',
                         '--data', '/unused/data', '--output', str(output), '--seed', '1850000000',
                         '--isolated-diagnostic']
            for module, extra in ((regression, []), (garrison, ['--maps', '/unused/maps'])):
                with self.subTest(module=module.__name__), patch.object(module, 'supervise', return_value=0) as launch:
                    self.assertEqual(module.main(arguments+extra), 0)
                    self.assertEqual(launch.call_args.args[0].output, output)

    def test_isolated_lease_does_not_open_official_lock_and_serializes_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            output = root/'run-investigations/probe'
            with diagnostic_lease(output, isolated=True, artifacts=root) as proof:
                self.assertFalse(proof['official_lock_opened'])
                self.assertFalse((root/'local-script-training').exists())
                self.assertFalse(output.exists())
                with self.assertRaises(BlockingIOError):
                    with diagnostic_lease(output, isolated=True, artifacts=root):
                        self.fail('Concurrent private diagnostic acquired the lease')

    def test_default_preserves_official_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); owner = root/'owner.lock'; owner.touch()
            with owner.open('r') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    with diagnostic_lease(root/'output', official_lock=owner):
                        self.fail('Official exclusion was bypassed')
            with diagnostic_lease(root/'output', official_lock=owner) as proof:
                self.assertEqual(proof['mode'], 'official-training-excluded')

    def test_private_mode_rejects_existing_aliased_and_official_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); private = root/'run-investigations'; private.mkdir()
            (private/'old').mkdir(); (private/'alias').symlink_to(root, target_is_directory=True)
            for output in (root/'local-script-training/probe', private, private/'old', private/'alias/probe',
                           private/'missing/../probe'):
                with self.subTest(path=output), self.assertRaises(ValueError):
                    private_output(output, artifacts=root)
            with self.assertRaises(ValueError):
                with diagnostic_lease(private/'new', isolated=True, official_lock=root/'fake.lock', artifacts=root):
                    self.fail('Custom official lock accepted')

    def test_private_mode_rejects_symlink_or_hardlink_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); private = root/'run-investigations'; private.mkdir()
            target = root/'target'; target.touch(); lease = private/'.isolated-native-diagnostics.lock'
            lease.symlink_to(target)
            with self.assertRaises(OSError):
                with diagnostic_lease(private/'new', isolated=True, artifacts=root):
                    self.fail('Symlink lease accepted')
            lease.unlink(); lease.hardlink_to(target)
            with self.assertRaises(ValueError):
                with diagnostic_lease(private/'new', isolated=True, artifacts=root):
                    self.fail('Hardlink lease accepted')


if __name__ == '__main__':
    unittest.main()
