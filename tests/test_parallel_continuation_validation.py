"""Offline checks of diagnostic isolation, fresh seeds and complete parent state."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import validate_parallel_continuation as probe


class ParallelContinuationValidationTests(unittest.TestCase):
    def test_fresh_sequences_cover_both_seats_without_reusing_arm_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(probe, 'VALIDATION_ROOT', root):
                first = probe.reserve_sequences(root/'first')
                second = probe.reserve_sequences(root/'second')
            rows = [row for study in (first, second) for arm in study['arms'] for row in arm['schedule']]
            self.assertEqual(len(rows), 32)
            self.assertEqual(len({row['engine_seed'] for row in rows}), 32)
            self.assertEqual(len({row['policy_seed'] for row in rows}), 32)
            self.assertEqual(rows[0]['engine_seed'], 1_700_000_000)
            self.assertEqual(rows[0]['policy_seed'], 1_710_000_000)
            self.assertTrue(all(row['engine_seed'] < 2**31 for row in rows))
            self.assertEqual([(a['workers'], a['seat']) for a in first['arms']], list(probe.ARMS))
            self.assertTrue(all(len(arm['schedule']) == 4 for arm in first['arms']))
            ledger = json.loads((root/'seed-ledger.json').read_text())
            ledger['next_sequence'] = probe.FIRST_SEQUENCE
            (root/'seed-ledger.json').write_text(json.dumps(ledger))
            with mock.patch.object(probe, 'VALIDATION_ROOT', root):
                with self.assertRaises(ValueError): probe.reserve_sequences(root/'reused')

    def test_output_must_be_new_and_symlinks_cannot_escape_validation_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root/'allowed'; allowed.mkdir()
            outside = root/'outside'; outside.mkdir()
            (allowed/'link').symlink_to(outside, target_is_directory=True)
            with mock.patch.object(probe, 'VALIDATION_ROOT', allowed):
                self.assertEqual(probe.output_path(allowed/'new'), (allowed/'new').resolve())
                for path in (allowed, outside/'new', allowed/'link'/'new'):
                    with self.assertRaises(ValueError): probe.output_path(path)

    def test_all_five_parent_files_are_copied_exactly_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = {}
            for name in ('plan', 'report', 'policy', 'resume', 'initial_checkpoint'):
                source = root/(name+'.bin'); source.write_bytes(name.encode())
                parent[name] = probe.binding(source)
            copied = probe.copy_parent(parent, root/'diagnostic')
            self.assertEqual(set(copied), set(parent))
            for key in parent:
                self.assertNotEqual(copied[key]['path'], parent[key]['path'])
                self.assertEqual(copied[key]['sha256'], parent[key]['sha256'])
            with self.assertRaises(FileExistsError): probe.copy_parent(parent, root/'diagnostic')
            with self.assertRaises(ValueError): probe.copy_parent({'policy':parent['policy']}, root/'partial')
            Path(parent['resume']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError): probe.copy_parent(parent, root/'changed')

    def test_full_state_fingerprint_detects_tensor_rng_and_structural_changes(self):
        import torch
        value = dict(model={'weight':torch.tensor([1.,2.])}, optimizer={'step':torch.tensor(4.)},
                     generator=torch.tensor([1,2], dtype=torch.uint8), progress={'updates':1095})
        before = probe.fingerprint(value)
        self.assertEqual(before, probe.fingerprint(value))
        value['generator'][0] = 2
        self.assertNotEqual(before, probe.fingerprint(value))
        self.assertNotEqual(probe.fingerprint([1,2]), probe.fingerprint((1,2)))
        self.assertNotEqual(probe.fingerprint(torch.tensor([1.,2.])),
                            probe.fingerprint(torch.tensor([[1.,2.]])))

    def test_closed_parent_accepts_verified_budget_stop_but_refuses_running_or_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = dict(run_dir=str(root), profile='original')
            progress = {'updates':1095, 'rollouts':4380, 'decisions':597916}
            report = dict(status='stopped_budget', owned_cleanup_verified=True, initial_state_unchanged=True,
                          full_final_restore={'exact':True,'progress':progress}, latest_progress=progress,
                          latest_parent={'policy':'fixture'})
            shutdown = dict(owned_cleanup_verified=True, within_hard_deadline=True, worker_returncode=0)
            for name, value in [('session.json',session),('campaign-report.json',report),('shutdown.json',shutdown)]:
                (root/name).write_text(json.dumps(value))
            self.assertEqual(probe.closed_parent(root,1095), report['latest_parent'])
            report['status'] = 'running'
            (root/'campaign-report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): probe.closed_parent(root,1095)
            report['status'] = 'stopped_budget'; report['full_final_restore']['exact'] = False
            (root/'campaign-report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): probe.closed_parent(root,1095)


if __name__ == '__main__':
    unittest.main()
