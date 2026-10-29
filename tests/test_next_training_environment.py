"""The next-launch preparation cannot reuse or upgrade the live environment."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from tools import prepare_next_training_environment as environment


class EnvironmentSeparationTests(unittest.TestCase):
    def test_only_fresh_durable_environment_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            artifacts, source = root/'artifacts', root/'live-venv'
            (artifacts/'python').mkdir(parents=True)
            source.mkdir()
            fresh = artifacts/'python/next-launch-v1'
            self.assertEqual(environment.validate_destination(fresh, source, artifacts), fresh)
            for bad in (source, source/'nested', root/'other', artifacts/'python'):
                with self.subTest(path=str(bad)), self.assertRaises(ValueError):
                    environment.validate_destination(bad, source, artifacts)
            fresh.mkdir()
            with self.assertRaisesRegex(ValueError, 'new directory'):
                environment.validate_destination(fresh, source, artifacts)
            link = artifacts/'python/redirect'
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                environment.validate_destination(link/'next', source, artifacts)

    def fixture(self):
        before = dict(prefix='/live', packages={'torch': '2.9.1', 'sympy': '1.14.0'},
            base_prefix='/python', python_version='3.13.5', python_soabi='cpython-313-darwin',
            machine='arm64', torch_version='2.9.1', torch_config='same',
            torch_libraries={'libtorch.dylib': 'a'*64}, numeric_smoke={'values': [1, 2, 3]},
            numpy_version=None, numpy_bridge=False, warnings=['Failed to initialize NumPy'])
        expected = {**before['packages'], 'numpy': '2.3.5'}
        candidate = {**deepcopy(before), 'prefix': '/separate', 'packages': expected,
                     'numpy_version': '2.3.5', 'numpy_bridge': True, 'warnings': []}
        return before, candidate, expected

    def test_numpy_only_addition_with_exact_tensor_runtime_passes(self):
        before, candidate, expected = self.fixture()
        environment.verify_transition(before, candidate, expected)

    def test_source_reuse_runtime_changes_or_failed_bridge_are_rejected(self):
        before, candidate, expected = self.fixture()
        changes = [dict(prefix=before['prefix']), dict(numpy_bridge=False),
                   dict(numpy_version='2.3.4'), dict(warnings=['Failed to initialize NumPy'])]
        changes += [{key: 'changed'} for key in ('base_prefix', 'python_version', 'python_soabi',
                    'machine', 'torch_version', 'torch_config', 'torch_libraries', 'numeric_smoke')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                environment.verify_transition(before, {**candidate, **change}, expected)

    def test_locked_package_upgrade_or_extra_package_is_rejected(self):
        before, candidate, expected = self.fixture()
        for changed in ({**expected, 'torch': '2.10.0'}, {**expected, 'extra': '1.0'},
                        {**expected, 'sympy': '1.15.0'}):
            with self.subTest(expected=changed), self.assertRaises(ValueError):
                environment.verify_transition(before, {**candidate, 'packages': changed}, changed)

    def test_reviewed_versions_and_duplicate_lock_entries_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lock'
            path.write_text('torch==2.9.1\nnumpy==2.3.5\ntyping_extensions==4.16.0\n')
            self.assertEqual(environment.locked_versions(path)['typing-extensions'], '4.16.0')
            for invalid in ('torch==2.10.0\nnumpy==2.3.5\n', 'torch==2.9.1\n',
                            'torch==2.9.1\nnumpy==2.3.5\nnumpy==2.3.5\n'):
                path.write_text(invalid)
                with self.subTest(lock=invalid), self.assertRaises(ValueError):
                    environment.locked_versions(path)


if __name__ == '__main__':
    unittest.main()
