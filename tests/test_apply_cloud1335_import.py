"""Disposable-file controls for the fenced cloud1335 selection transaction."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import apply_cloud1335_import as apply


class ApplyCloudImportTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.artifacts = Path(temporary).resolve() / 'artifacts'
        self.root = self.artifacts / 'local-script-training/original'
        self.root.mkdir(parents=True)
        self.private = self.artifacts / 'run-investigations/current'
        self.private.mkdir(parents=True)
        self.config = {'artifact_root': str(self.artifacts), 'run_root': str(self.root), 'default_profile': 'original'}
        self.config_path = self.private / 'machine.json'
        self.config_path.write_bytes(apply.encoded(self.config))
        self.before = {name: apply.encoded({'old': name}).decode() for name in apply.MOVING}
        for name, value in self.before.items():
            (self.root / name).write_text(value)
        self.base = self.private / 'base-recovery.json'
        self.base.write_bytes(apply.encoded({'verification': {'original_moving_snapshots': self.before}}))
        self.receipt = self.private / 'import.json'
        self.receipt.write_bytes(apply.encoded({'base_recovery_receipt': apply.cloud.binding(self.base)}))
        self.lock_paths = [self.artifacts / 'local-script-training/owner.lock', self.root / 'owner.lock']
        locks = []
        for path in self.lock_paths:
            path.write_bytes(b'30058')
            path.chmod(0o444)
            metadata = path.stat()
            locks.append({'binding': apply.cloud.binding(path), 'device': metadata.st_dev,
                          'inode': metadata.st_ino, 'uid': metadata.st_uid, 'original_mode': '0644'})
        self.fence = self.artifacts / apply.FENCE_RELATIVE
        self.fence.parent.mkdir(parents=True)
        self.fence.write_bytes(apply.encoded({'schema': 'heroes-mac-cloud-handoff-fence-v1',
            'handoff_id': apply.cloud.HANDOFF, 'uid': os.getuid(), 'current_mode': '0444',
            'locks_held_through_fence': True, 'locks': locks}))
        self.audit = {'selected_update': 1335, 'progress': {'updates': 1335, 'rollouts': 5340, 'decisions': 680677},
                      'ledger': {'next_sequence': 975, 'consumed': [{'sequence': 974}]},
                      'receipt': {**apply.cloud.binding(self.receipt), 'path': str(self.root / 'cloud-import.json')},
                      'parent': {'policy': 'verified-private-policy'}, 'import_run': str(self.root / 'cloud-import-1335')}
        self.output = self.private / 'application.json'
        self.stack.enter_context(patch.object(apply, 'FENCE_SHA256', apply.cloud.binding(self.fence)['sha256']))
        self.source_check = self.stack.enter_context(patch.object(apply, 'reviewed_source'))
        self.stack.enter_context(patch.object(apply, 'local_process_closure', return_value={'matching_official_processes': []}))
        self.audit_call = self.stack.enter_context(patch.object(apply.cloud, 'audit_import', return_value=self.audit))
        self.continuation_call = self.stack.enter_context(patch.object(apply.runner, 'original_continuation',
                                                                        side_effect=self.continuation))

    def modes(self):
        return [stat.S_IMODE(path.stat().st_mode) for path in self.lock_paths]

    def continuation(self, anchor, config):
        self.assertEqual(self.modes(), [0o444, 0o444])
        self.assertEqual((self.root / 'cloud-import.json').read_bytes(), self.receipt.read_bytes())
        for name, expected in apply.cloud.moving_pointers(self.audit).items():
            self.assertEqual(apply.cloud.read(self.root / name), expected)
        return self.audit['parent'], 2176, {'admitted_before_session': 2824, 'remaining_before_session': 2176,
                                          'next_sequence': 975, 'seed_ledger_snapshot': self.audit['ledger']}

    def run_apply(self, **kwargs):
        return apply.apply_import(self.config_path, self.receipt, apply=True, output=self.output, **kwargs)

    def snapshot(self):
        return {str(path): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
                for path in self.artifacts.rglob('*') if path.is_file()}

    def test_default_check_changes_nothing(self):
        before = self.snapshot()
        result = apply.apply_import(self.config_path, self.receipt)
        self.assertEqual(result['status'], 'check_passed')
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.modes(), [0o444, 0o444])
        self.continuation_call.assert_not_called()
        self.source_check.assert_called_once()
        self.assertTrue(self.audit_call.call_args.kwargs['preview'])

    def test_success_verifies_installed_continuation_before_unlocking(self):
        observed = []
        original = apply.restore_lock_mode
        def restore(fd, mode):
            self.assertEqual(self.continuation_call.call_count, 1)
            self.assertEqual(apply.cloud.read(self.output.with_name(self.output.name + '.journal') / 'transaction.json')['status'],
                             'verified_before_unlock')
            observed.append(self.modes())
            original(fd, mode)
        with patch.object(apply, 'restore_lock_mode', side_effect=restore):
            result = self.run_apply()
        self.assertEqual(observed, [[0o444, 0o444], [0o644, 0o444]])
        self.assertEqual(self.modes(), [0o644, 0o644])
        self.assertEqual(result['status'], 'applied')
        self.assertTrue(result['verified_before_unlock'])
        self.assertEqual(result['remaining_original_games'], 2176)
        journal = apply.cloud.read(result['journal']['path'])
        self.assertEqual(journal['status'], 'applied')
        for name, old in self.before.items():
            self.assertEqual(journal['before'][str(self.root / name)]['utf8'], old)
            self.assertEqual(journal['after'][str(self.root / name)]['utf8'], (self.root / name).read_text())
        summary = apply.cloud.read(result['summary']['path'])
        self.assertEqual(summary['status'], 'imported_verified_cloud_state')
        self.assertEqual(summary['new_local_games'], 0)
        self.assertFalse(summary['training_launched'])

    def test_unrelated_current_pointer_change_is_rejected_without_writes(self):
        (self.root / 'sequence-ledger.json').write_text('{"unexpected":true}\n')
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'Official state changed'):
            self.run_apply()
        self.assertEqual(self.snapshot(), before)
        self.audit_call.assert_not_called()

    def test_exact_original_bytes_required_not_just_equivalent_json(self):
        name = 'latest-run.json'
        (self.root / name).write_text(json.dumps(json.loads(self.before[name])))
        with self.assertRaisesRegex(ValueError, 'Official state changed'):
            self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])

    def test_existing_marker_rejected_without_overwrite(self):
        marker = self.root / 'cloud-import.json'
        marker.write_text('unrelated')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_apply()
        self.assertEqual(marker.read_text(), 'unrelated')
        self.assertEqual(self.modes(), [0o444, 0o444])

    def test_mid_pointer_write_failure_stays_fenced_and_preserves_journal(self):
        original = apply.atomic_install
        def install(path, payload, previous):
            if path == self.root / 'latest-checkpoint.json':
                raise OSError('injected pointer write failure')
            return original(path, payload, previous)
        with patch.object(apply, 'atomic_install', side_effect=install):
            with self.assertRaisesRegex(OSError, 'injected'):
                self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])
        self.assertTrue((self.root / 'cloud-import.json').exists())
        self.assertNotEqual((self.root / 'latest-run.json').read_text(), self.before['latest-run.json'])
        self.assertEqual((self.root / 'latest-checkpoint.json').read_text(), self.before['latest-checkpoint.json'])
        journal = apply.cloud.read(self.output.with_name(self.output.name + '.journal') / 'transaction.json')
        self.assertEqual(journal['status'], 'failed_fenced_manual_review_required')
        self.assertEqual(set(journal['before']), set(journal['after']))
        self.assertEqual(len(journal['before']), 5)
        self.continuation_call.assert_not_called()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'journal already exists'):
            self.run_apply()

    def test_failure_between_lock_restorations_refences_both(self):
        original = apply.restore_lock_mode
        count = 0
        def restore(fd, mode):
            nonlocal count
            count += 1
            if count == 2:
                self.assertEqual(self.modes(), [0o644, 0o444])
                raise OSError('injected second unlock failure')
            original(fd, mode)
        with patch.object(apply, 'restore_lock_mode', side_effect=restore):
            with self.assertRaisesRegex(OSError, 'second unlock'):
                self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])
        journal = apply.cloud.read(self.output.with_name(self.output.name + '.journal') / 'transaction.json')
        self.assertEqual(journal['status'], 'failed_fenced_manual_review_required')
        self.assertFalse(self.output.exists())

    def test_failed_continuation_never_unlocks(self):
        self.continuation_call.side_effect = ValueError('injected restore mismatch')
        with patch.object(apply, 'restore_lock_mode') as unlock:
            with self.assertRaisesRegex(ValueError, 'restore mismatch'):
                self.run_apply()
        unlock.assert_not_called()
        self.assertEqual(self.modes(), [0o444, 0o444])

    def test_journal_parent_sync_failure_prevents_all_official_writes(self):
        before = {name: (self.root / name).read_bytes() for name in apply.MOVING}
        original = apply.fsync_directory
        def sync(path):
            if path == self.private:
                raise OSError('injected journal parent fsync failure')
            original(path)
        with patch.object(apply, 'fsync_directory', side_effect=sync):
            with self.assertRaisesRegex(OSError, 'journal parent'):
                self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])
        self.assertFalse((self.root / 'cloud-import.json').exists())
        self.assertEqual({name: (self.root / name).read_bytes() for name in apply.MOVING}, before)

    def test_failure_after_both_unlocks_refences_before_propagating(self):
        original = apply.atomic_install
        def install(path, payload, previous):
            if path == self.output:
                self.assertEqual(self.modes(), [0o644, 0o644])
                raise OSError('injected final receipt failure')
            return original(path, payload, previous)
        with patch.object(apply, 'atomic_install', side_effect=install):
            with self.assertRaisesRegex(OSError, 'final receipt'):
                self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])

    def test_lock_mode_or_inode_change_rejected(self):
        self.lock_paths[0].chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'identity, bytes or mode'):
            self.run_apply()
        self.lock_paths[0].chmod(0o444)
        payload = self.lock_paths[1].read_bytes()
        self.lock_paths[1].unlink()
        self.lock_paths[1].write_bytes(payload)
        self.lock_paths[1].chmod(0o444)
        with self.assertRaisesRegex(ValueError, 'identity, bytes or mode'):
            self.run_apply()

    def test_fence_hash_change_rejected(self):
        self.fence.write_text(self.fence.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'pinned Mac handoff fence'):
            self.run_apply()
        self.assertEqual(self.modes(), [0o444, 0o444])

    def test_existing_campaign_owner_rejected(self):
        with apply.recovery.existing_campaign_locks(self.config):
            with self.assertRaisesRegex(ValueError, 'currently holds'):
                self.run_apply()

    def test_apply_requires_private_output(self):
        with self.assertRaisesRegex(ValueError, 'fresh private output'):
            apply.apply_import(self.config_path, self.receipt, apply=True)
        with self.assertRaisesRegex(ValueError, 'private investigation'):
            apply.apply_import(self.config_path, self.receipt, apply=True, output=self.root / 'bad.json')


class ProcessClosureTests(unittest.TestCase):
    def test_current_training_command_rejected(self):
        command = f'123 1 123 {os.getuid()} /usr/bin/python tools/run_local_training.py _worker\n'
        with patch.object(apply.subprocess, 'run', return_value=SimpleNamespace(stdout=command)):
            with self.assertRaisesRegex(ValueError, 'official training processes remain'):
                apply.local_process_closure({'artifact_root': '/private/artifacts'})

    def test_unrelated_process_is_not_signalled_or_rejected(self):
        command = f'123 1 123 {os.getuid()} /usr/bin/python unrelated.py\n'
        with patch.object(apply.subprocess, 'run', return_value=SimpleNamespace(stdout=command)):
            self.assertEqual(apply.local_process_closure({'artifact_root': '/private/artifacts'})['signals_sent'], 0)

    def test_orphan_official_collector_is_rejected(self):
        command = (f'123 1 123 {os.getuid()} /usr/bin/python -m experiments.learner.parallel_collection '
                   'worker --plan /private/artifacts/local-script-training/original/run/batch/plan.json\n')
        with patch.object(apply.subprocess, 'run', return_value=SimpleNamespace(stdout=command)):
            with self.assertRaisesRegex(ValueError, 'official training processes remain'):
                apply.local_process_closure({'artifact_root': '/private/artifacts'})

    def test_atomic_absent_install_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'marker'
            path.write_bytes(b'existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                apply.atomic_install(path, b'new', None)
            self.assertEqual(path.read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main()
