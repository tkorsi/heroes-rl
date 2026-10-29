"""Copied diagnostics never reuse writable training inputs or official leases."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import benchmark_copied_training as copied


class CopiedInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.artifacts = self.root/'artifacts'
        self.artifacts.mkdir()
        self.patch = patch.object(copied, 'ARTIFACTS', self.artifacts)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def snapshot(self):
        root = self.artifacts/'snapshot'; root.mkdir()
        file = root/'payload/parent/policy.pt'; file.parent.mkdir(parents=True); file.write_bytes(b'original')
        (root/'lease.lock').touch()
        value = dict(schema=copied.SCHEMA, diagnostic_only=True, official_samples_admitted=0,
            official_updates=0, admission_capability_created=False, original_inputs_unchanged=True,
            inventory={'payload/parent/policy.pt': copied.binding(file)})
        copied.write_new(root/'snapshot.json', value)
        digest = copied.binding(root/'snapshot.json')['sha256']
        return root, file, digest

    def test_copy_has_independent_inode_and_verified_bytes(self):
        source = self.root/'source'; source.write_bytes(b'checkpoint')
        target = self.artifacts/'copy'
        original = copied.binding(source)
        receipt = copied.copy_verified(source, target, original)
        self.assertEqual(receipt['source'], original)
        self.assertEqual(source.read_bytes(), target.read_bytes())
        self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
        self.assertEqual(target.stat().st_nlink, 1)
        target.write_bytes(b'local diagnostic changes')
        self.assertEqual(copied.binding(source), original)
        with self.assertRaises(ValueError):
            copied.copy_verified(source, self.artifacts/'wrong', {**original, 'sha256': '0'*64})
        self.assertFalse((self.artifacts/'wrong').exists())

    def test_symlinks_hardlinks_and_parent_redirection_are_rejected(self):
        source = self.root/'source'; source.write_bytes(b'checkpoint')
        alias = self.root/'alias'; alias.symlink_to(source)
        with self.assertRaises(ValueError): copied.no_links(alias, regular=True)
        alias.unlink(); os.link(source, alias)
        with self.assertRaises(ValueError): copied.no_links(source, regular=True)
        with self.assertRaises(ValueError): copied.copy_verified(alias, self.artifacts/'copy')
        alias.unlink()
        redirect = self.artifacts/'redirect'; redirect.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): copied.no_links(redirect/'source', regular=True)

    def test_snapshot_rejects_mutation_unknown_files_and_manifest_replacement(self):
        root, file, digest = self.snapshot()
        copied.verify_snapshot(root, digest)
        file.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Copied input changed'):
            copied.verify_snapshot(root, digest)
        file.write_bytes(b'original')
        unexpected = root/'unbound.pt'; unexpected.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'unbound or missing'):
            copied.verify_snapshot(root, digest)
        unexpected.unlink()
        (root/'snapshot.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            copied.verify_snapshot(root, digest)

    def test_snapshot_hardlink_or_escaping_member_is_rejected(self):
        root, file, digest = self.snapshot()
        outside = self.artifacts/'alias'; os.link(file, outside)
        with self.assertRaisesRegex(ValueError, 'hard links'):
            copied.verify_snapshot(root, digest)
        outside.unlink()
        for name in ('../source', '/absolute', 'payload/../escape', 'payload//file', './file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                copied.relative_file(root, name)

    def test_snapshot_local_lease_blocks_a_second_diagnostic(self):
        root, _, _ = self.snapshot()
        with copied.snapshot_lease(root):
            with copied.snapshot_lease(root, worker=True):
                with self.assertRaises(BlockingIOError):
                    with copied.snapshot_lease(root):
                        self.fail('A second diagnostic acquired the same snapshot')
        with copied.snapshot_lease(root):
            pass

    def test_official_paths_and_repository_outputs_are_rejected(self):
        for path in (self.artifacts, self.artifacts/'local-script-training/test',
                     self.artifacts/'releases/test', self.root/'outside'):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                copied.private_path(path)
        repo = self.artifacts/'repo'; repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        with self.assertRaisesRegex(ValueError, 'Git checkout'):
            copied.private_path(repo/'output')
        output = self.artifacts/'fresh'
        self.assertEqual(copied.private_path(output, fresh=True), output)
        output.mkdir()
        with self.assertRaises(ValueError): copied.private_path(output, fresh=True)

    def test_output_cannot_modify_the_input_snapshot(self):
        root, file, digest = self.snapshot()
        original = file.read_bytes()
        for output in (root, root/'results', root.parent, root/'../escape'):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                copied.disjoint_output(output, root)
        copied.disjoint_output(self.artifacts/'separate-output', root)
        self.assertEqual(file.read_bytes(), original)
        copied.verify_snapshot(root, digest)

    def test_timeout_still_closes_only_owned_process_and_writes_shutdown(self):
        output = self.artifacts/'timed-out'
        args = SimpleNamespace(checkout=self.root, output_dir=output, wall_seconds=30)
        process = SimpleNamespace(returncode=None)
        def wait(**_): raise subprocess.TimeoutExpired('diagnostic', 20)
        process.wait = wait
        def close(actual, **_):
            self.assertIs(actual, process)
            actual.returncode = -9
        deadline = SimpleNamespace(guarded_popen=lambda *a, **k: process,
                                   close_owned_process=close)
        identity = {'path': str(self.root), 'commit': 'a'*40}
        with patch.object(copied, 'checkout_identity', return_value=identity), \
                patch.object(copied, 'load_helpers', return_value=(None, deadline)), \
                redirect_stdout(io.StringIO()):
            result = copied.supervise(args)
        self.assertNotEqual(result, 0)
        shutdown = json.loads((output/'shutdown.json').read_text())
        self.assertTrue(shutdown['cleanup_verified'])
        self.assertEqual(shutdown['status'], 'timed_out_or_interrupted')
        self.assertFalse(shutdown['official_owner_lock_acquired'])
        self.assertEqual(shutdown['official_updates'], 0)
        self.assertAlmostEqual(shutdown['deadline_unix']-shutdown['started_unix'], 30)


if __name__ == '__main__':
    unittest.main()
