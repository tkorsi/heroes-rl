"""Separate-interpreter selection remains bound to the reviewed NumPy-only change."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools import training_environment as environment


class TrainingEnvironmentTests(unittest.TestCase):
    def fixture(self, root):
        (root/'dependencies').mkdir()
        lock = root/'dependencies/learner.lock'
        lock.write_text('torch==2.9.1\nnumpy==2.3.5\n')
        def write(name, value):
            path = root/name
            path.write_text(json.dumps(value, sort_keys=True))
            raw = path.read_bytes()
            return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        before = dict(executable='/old/bin/python', prefix='/old', base_prefix='/base',
            python_version='same Python', python_soabi='same ABI', machine='arm64',
            torch_version='2.9.1', torch_config='same config', torch_libraries={'libtorch': 'a'*64},
            numeric_smoke={'exact': True}, packages={'torch': '2.9.1'})
        candidate = {**before, 'executable': '/new/bin/python', 'prefix': '/new',
            'packages': {'torch': '2.9.1', 'numpy': '2.3.5'}, 'numpy_version': '2.3.5',
            'numpy_bridge': True, 'warnings': []}
        receipt = dict(schema='heroes-next-training-environment-v1',
            existing_packages_unchanged=True, numeric_smoke_exact=True, numpy_bridge_verified=True,
            python_and_torch_abi_unchanged=True, source_environment_unchanged=True, torch_libraries_identical=True,
            initialization_warnings=[], selected_for_training=False,
            official_state_modified=False, training_launched=False,
            source_before=write('before.json', before), source_after=write('after.json', before),
            candidate=write('candidate.json', candidate),
            dependency_lock=dict(sha256=hashlib.sha256(lock.read_bytes()).hexdigest(), bytes=lock.stat().st_size),
            python=candidate['executable'], source_python=before['executable'])
        return write('receipt.json', receipt), before, candidate

    def test_current_probe_and_both_machine_identities_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); identity, old, new = self.fixture(root)
            with patch.object(environment, 'ROOT', root), patch.object(environment.sys, 'executable', new['executable']), \
                    patch.object(environment.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(new), stderr='')) as probe:
                environment.verify_environment(identity, previous_python=environment.machine_python(old),
                    current_python=environment.machine_python(new))
                probe.assert_called_once()
                self.assertEqual(probe.call_args.kwargs['timeout'], 45)
                self.assertTrue(all(probe.call_args.kwargs['env'][name] == '1' for name in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')))
                for key, kwargs in (('old', {'previous_python': {'executable': '/unexpected'}}),
                                    ('new', {'current_python': {'executable': '/unexpected'}})):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        environment.verify_environment(identity, **kwargs)
                probe.return_value.stdout = json.dumps({**new, 'numpy_bridge': False})
                with self.assertRaisesRegex(ValueError, 'changed after acceptance'):
                    environment.verify_environment(identity)

    def test_wrong_interpreter_tampered_evidence_and_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); identity, old, new = self.fixture(root)
            with patch.object(environment, 'ROOT', root), patch.object(environment.sys, 'executable', old['executable']), \
                    patch.object(environment.subprocess, 'run') as probe:
                with self.assertRaisesRegex(ValueError, 'explicitly prepared Python'):
                    environment.verify_environment(identity)
                probe.assert_not_called()
                (root/'dependencies/learner.lock').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'lock differs'):
                    environment.verify_environment(identity, probe_current=False)
            Path(identity['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                environment.bound_json(identity)

    def test_receipt_cannot_be_redirected_aliased_or_malformed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); identity, _, _ = self.fixture(root)
            linked = root/'linked.json'; os.link(identity['path'], linked)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                environment.bound_json(identity)
            linked.unlink()
            redirect = root/'redirect'; redirect.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                environment.bound_json({**identity, 'path': str(redirect/'receipt.json')})
            for change in ({'bytes': True}, {'sha256': 'wrong'}, {'path': 42}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'binding'):
                    environment.bound_json({**identity, **change})


if __name__ == '__main__':
    unittest.main()
