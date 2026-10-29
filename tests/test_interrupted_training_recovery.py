"""Adversarial read-only recovery controls; no native games or official files."""
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tools import recover_interrupted_training as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')
        return path

    def test_exclusive_receipt_is_complete_and_cannot_overwrite(self):
        path = self.root/'receipt.json'
        recovery.write_exclusive(path, {'verified': True})
        self.assertEqual(recovery.read(path), {'verified': True})
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            recovery.write_exclusive(path, {'verified': False})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_publication_link_failure_leaves_no_partial_receipt(self):
        path = self.root/'receipt.json'
        with patch.object(recovery.os, 'link', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                recovery.write_exclusive(path, {'large': [1, 2, 3]})
        self.assertEqual(list(self.root.iterdir()), [])

    def locks(self):
        config = {'artifact_root': str(self.root), 'run_root': str(self.root/'local-script-training/original')}
        paths = [self.root/'local-script-training/owner.lock', Path(config['run_root'])/'owner.lock']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('old-owner-123\n')
        return config, paths

    def test_existing_locks_preserve_bytes_and_do_not_create_missing_files(self):
        config, paths = self.locks()
        before = [p.read_bytes() for p in paths]
        with recovery.existing_campaign_locks(config) as unchanged:
            unchanged()
            self.assertEqual([p.read_bytes() for p in paths], before)
        paths[-1].unlink()
        with self.assertRaises(FileNotFoundError):
            with recovery.existing_campaign_locks(config):
                self.fail('Missing lock accepted')
        self.assertFalse(paths[-1].exists())

    def test_active_shared_or_profile_lock_refuses_without_truncation(self):
        config, paths = self.locks()
        for path in paths:
            with path.open('rb') as owner:
                recovery.fcntl.flock(owner, recovery.fcntl.LOCK_EX | recovery.fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'currently holds'):
                    with recovery.existing_campaign_locks(config):
                        self.fail('Active owner accepted')
                self.assertEqual(path.read_text(), 'old-owner-123\n')
        with recovery.existing_campaign_locks(config):
            pass

    def test_lock_replacement_and_symlink_are_rejected(self):
        config, paths = self.locks()
        with self.assertRaisesRegex(ValueError, 'path changed'):
            with recovery.existing_campaign_locks(config):
                paths[0].unlink(); paths[0].write_text('replacement')
        paths[0].unlink(); paths[0].symlink_to(paths[1])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            with recovery.existing_campaign_locks(config):
                pass

    def deadlines(self, rows=None):
        if rows is None:
            rows = [dict(event='launch_requested', writer_pid=1001),
                    dict(event='wrapper_started', writer_pid=1002, leader_pid=1002, process_group=1002),
                    dict(event='guardian_armed', writer_pid=1003, leader_pid=1002, guardian_pid=1003,
                         process_group=1002)]
        path = self.root/'run/deadline.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps({'schema': 'heroes-process-deadline-v1', **row})+'\n' for row in rows))
        self.write('run/session.json', {'supervisor_pid': 1000})
        return [recovery.binding(path)]

    def ps_output(self, rows):
        return rows + f'{os.getpid()} {os.getpgrp()} S recovery-observer\n'

    def test_process_scan_is_read_only_and_rejects_reused_old_pid_or_group(self):
        deadlines = self.deadlines()
        for stdout in ('1002 999 S unrelated-new-app\n', '999 1002 S unrelated-new-app\n',
                       '1000 999 S unrelated-new-app\n', '1003 999 S unrelated-new-app\n'):
            with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout=self.ps_output(stdout))) as ps:
                with patch.object(os, 'kill') as kill, patch.object(os, 'killpg') as killpg:
                    with self.assertRaisesRegex(ValueError, 'Live or ambiguous'):
                        recovery.process_closure(deadlines, self.root/'run')
                    kill.assert_not_called(); killpg.assert_not_called()
                    self.assertEqual(ps.call_args.kwargs['timeout'], 10)

    def test_process_scan_rejects_run_specific_unrecorded_process(self):
        deadlines = self.deadlines()
        stdout = f'999 999 S worker --run {self.root}/run\n'
        with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout=self.ps_output(stdout))):
            with self.assertRaisesRegex(ValueError, 'Live or ambiguous'):
                recovery.process_closure(deadlines, self.root/'run')

    def test_process_scan_accepts_absence_and_zombie_without_signals(self):
        deadlines = self.deadlines()
        with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout=self.ps_output('999 999 S editor\n1002 1002 Z dead\n'))):
            result = recovery.process_closure(deadlines, self.root/'run')
        self.assertEqual(result['matched_processes'], [])
        self.assertEqual(result['recorded_groups'], 1)
        self.assertEqual(result['signals_sent'], 0)
        self.assertTrue(result['observer_present'])
        self.assertEqual(result['observer_pid'], os.getpid())

    def test_process_scan_rejects_empty_missing_or_nonlive_observer(self):
        deadlines = self.deadlines()
        for stdout in ('', '  \n', None, '999 999 S editor\n',
                       f'{os.getpid()} {os.getpgrp()} Z impossible-dead-observer\n'):
            with self.subTest(stdout=stdout):
                with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout=stdout)):
                    with patch.object(os, 'kill') as kill, patch.object(os, 'killpg') as killpg:
                        with self.assertRaisesRegex(ValueError, 'inventory'):
                            recovery.process_closure(deadlines, self.root/'run')
                        kill.assert_not_called(); killpg.assert_not_called()

    def test_malformed_deadline_rows_and_supervisor_identity_fail_cleanly(self):
        deadlines = self.deadlines()
        path = Path(deadlines[0]['path'])
        for malformed in ([], None, 123, 'deadline', {'schema': ['heroes-process-deadline-v1']}):
            path.write_text(json.dumps(malformed)+'\n')
            with self.subTest(malformed=malformed):
                with patch.object(recovery.subprocess, 'run') as ps:
                    with self.assertRaisesRegex(ValueError, 'malformed deadline'):
                        recovery.process_closure([recovery.binding(path)], self.root/'run')
                    ps.assert_not_called()
        deadlines = self.deadlines()
        for malformed in ([], None, {'supervisor_pid': True}, {'supervisor_pid': '1000'}):
            self.write('run/session.json', malformed)
            with self.subTest(supervisor=malformed):
                with patch.object(recovery.subprocess, 'run') as ps:
                    with self.assertRaises(ValueError):
                        recovery.process_closure(deadlines, self.root/'run')
                    ps.assert_not_called()

    def test_unresolved_launch_bad_wrapper_and_changed_evidence_rejected_before_ps(self):
        for rows in ([dict(event='launch_requested', writer_pid=1001)],
                     [dict(event='wrapper_started', writer_pid=1001, leader_pid=1002, process_group=1002)]):
            deadlines = self.deadlines(rows)
            with patch.object(recovery.subprocess, 'run') as ps:
                with self.assertRaises(ValueError):
                    recovery.process_closure(deadlines, self.root/'run')
                ps.assert_not_called()
        deadlines = self.deadlines()
        Path(deadlines[0]['path']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            recovery.process_closure(deadlines, self.root/'run')

    def test_inventory_binds_every_file_and_rejects_symlinks(self):
        first = self.write('inventory/a.json', {'x': 1})
        self.write('inventory/b.json', {'x': 2})
        before = recovery.inventory(first.parent)
        first.write_text('{}')
        self.assertNotEqual(recovery.json_digest(before), recovery.json_digest(recovery.inventory(first.parent)))
        (first.parent/'escape').symlink_to(self.root/'elsewhere')
        (self.root/'elsewhere').write_text('x')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            recovery.inventory(first.parent)

    def state(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        generator = torch.Generator().manual_seed(5)
        pool = SimpleNamespace(state_dict=lambda: {'entries': ()})
        return SimpleNamespace(model=model, optimizer=optimizer, generator=generator, pool=pool,
            progress=SimpleNamespace(updates=1229, rollouts=4916, decisions=627977),
            sampler_state={'next_schedule_index': 4, 'optimizer_steps': 21238},
            optimizer_steps=21238, weights_sha256='reviewed-weights', checkpoint='checkpoint1229')

    def test_complete_restore_compares_adam_rng_sampler_pool_and_preserves_caller_rng(self):
        state = self.state()
        torch_state = torch.Generator().manual_seed(17).get_state()
        python_state = random.Random(17).getstate()
        snapshot = dict(model=dict(state.model.state_dict()), optimizer=state.optimizer.state_dict(),
            generator=state.generator.get_state(), pool=state.pool.state_dict(), progress=vars(state.progress),
            sampler=state.sampler_state, torch_rng=torch_state, python_rng=python_state)
        expected = dict(weights_sha256=state.weights_sha256, checkpoint=state.checkpoint,
            full_state_sha256=recovery.fingerprint(snapshot),
            component_sha256={k: recovery.fingerprint(v) for k,v in snapshot.items()})
        original_torch, original_python = torch.get_rng_state().clone(), random.getstate()
        def restore(descriptor):
            torch.set_rng_state(torch_state); random.setstate(python_state)
            return state
        with patch('experiments.learner.local_script_training.restore_parent', side_effect=restore) as loader:
            result = recovery.verify_state({'canonical': 1}, {'backup': 1}, expected)
            self.assertEqual(loader.call_count, 2)
        self.assertTrue(result['canonical_and_backup_exact'])
        self.assertTrue(torch.equal(torch.get_rng_state(), original_torch))
        self.assertEqual(random.getstate(), original_python)
        for component in snapshot:
            changed = dict(expected, component_sha256={**expected['component_sha256'], component:'tampered'})
            with self.subTest(component=component):
                with patch('experiments.learner.local_script_training.restore_parent', side_effect=restore):
                    with self.assertRaisesRegex(ValueError, 'Complete model/Adam/RNG'):
                        recovery.verify_state({}, {}, changed)
                self.assertTrue(torch.equal(torch.get_rng_state(), original_torch))
                self.assertEqual(random.getstate(), original_python)

    def test_wrong_run_and_profile_fail_before_state_loading(self):
        profile = dict(schema=recovery.SCHEMA+'-profile', run_dir=str(self.root/'run'))
        for run, config in [(self.root/'other', {'default_profile': 'original'}),
                            (self.root/'run', {'default_profile': 'easier8x8'})]:
            with patch.object(recovery, 'reviewed_profile', return_value=profile):
                with patch.object(recovery, 'verify_state') as state:
                    with self.assertRaisesRegex(ValueError, 'scoped'):
                        recovery.audit_current(run, config)
                    state.assert_not_called()

    def test_receipt_recomputes_evidence_and_keeps_moving_files_out_of_future_bindings(self):
        profile = self.write('profile.json', {'scope': 'reviewed'})
        immutable = self.write('run/shutdown.json', {'within_hard_deadline': False})
        actual = dict(parent={'policy': 'fixed'}, progress={'updates': 1229},
            state={'full_state_sha256': 'full-state'}, immutable_evidence={'shutdown': recovery.binding(immutable)},
            original_moving_snapshots={'sequence-ledger.json': '{"next_sequence":869}'},
            process_closure={'observed_utc': 'new', 'signals_sent': 0, 'matched_processes': []}, **recovery.FLAGS)
        identity = {k:v for k,v in recovery.binding(profile).items() if k != 'path'}
        receipt = self.write('receipt.json', dict(schema=recovery.SCHEMA, profile=identity,
            creation_process_closure={'matched_processes': [], 'signals_sent': 0},
            verification={k:v for k,v in actual.items() if k != 'process_closure'}))
        with patch.object(recovery, 'PROFILE', profile), patch.object(recovery, 'audit_current', return_value=actual) as audit:
            result = recovery.validate_receipt(receipt, self.root/'run', {})
            audit.assert_called_once()
            self.assertEqual(set(result['evidence']), {'shutdown', 'recovery_receipt'})
            self.assertEqual(result['progress']['updates'], 1229)
            relocated = self.write('other-checkout/profile.json', {'scope': 'reviewed'})
            with patch.object(recovery, 'PROFILE', relocated):
                relocated_result = recovery.validate_receipt(receipt, self.root/'run', {})
                self.assertEqual(result['parent'], relocated_result['parent'])
                relocated.write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'receipt profile'):
                    recovery.validate_receipt(receipt, self.root/'run', {})
            value = recovery.read(receipt);value['verification']['progress']['updates'] = 770
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'facts differ'):
                recovery.validate_receipt(receipt, self.root/'run', {})

    def test_stale_root_and_live_process_errors_cannot_be_overridden_by_receipt(self):
        profile = self.write('profile.json', {})
        identity = {k:v for k,v in recovery.binding(profile).items() if k != 'path'}
        receipt = self.write('receipt.json', {'schema': recovery.SCHEMA, 'profile': identity})
        with patch.object(recovery, 'PROFILE', profile):
            for reason in ('latest pointer advanced', 'consumed seed ledger changed', 'live old process'):
                with patch.object(recovery, 'audit_current', side_effect=ValueError(reason)):
                    with self.assertRaisesRegex(ValueError, reason):
                        recovery.validate_receipt(receipt, self.root/'run', {})

    def moving_fixture(self):
        run = self.root/'original/run'
        paths = {name:self.write('original/'+name, {'reviewed': name}) for name in recovery.MOVING}
        profile = {'moving_evidence': {name:recovery.binding(path) for name,path in paths.items()}}
        snapshots = {name:path.read_text() for name,path in paths.items()}
        return run, paths, profile, snapshots

    def test_inherited_snapshots_allow_current_roots_to_advance_but_initial_mode_rejects(self):
        run, paths, profile, snapshots = self.moving_fixture()
        self.assertEqual(recovery.read_moving_inputs(profile,run), snapshots)
        for name,path in paths.items():
            with self.subTest(name=name):
                path.write_text('{"later_clean_session":true}\n')
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    recovery.read_moving_inputs(profile,run)
                self.assertEqual(recovery.read_moving_inputs(profile,run,snapshots), snapshots)
                path.unlink()
                self.assertEqual(recovery.read_moving_inputs(profile,run,snapshots), snapshots)
                path.write_text(snapshots[name])

    def test_inherited_snapshots_reject_changed_bytes_omissions_and_wrong_types(self):
        run, paths, profile, snapshots = self.moving_fixture()
        for name in recovery.MOVING:
            for replacement in ('{}', snapshots[name]+' '):
                with self.subTest(name=name,replacement=replacement):
                    changed = {**snapshots, name:replacement}
                    with self.assertRaisesRegex(ValueError, 'snapshot bytes changed'):
                        recovery.read_moving_inputs(profile,run,changed)
        for snapshots_invalid in ({}, [], {**snapshots,'latest-run.json':None},
                                  {**snapshots,'unexpected':'{}'}):
            with self.assertRaisesRegex(ValueError, 'all exact original'):
                recovery.read_moving_inputs(profile,run,snapshots_invalid)

    def test_inherited_receipt_explicitly_selects_snapshots_and_still_runs_full_audit(self):
        run, paths, profile, snapshots = self.moving_fixture()
        profile_path = self.write('profile.json', profile)
        actual = dict(parent={'policy':'fixed'},progress={'updates':1229},
            state={'full_state_sha256':'state'},immutable_evidence={},
            original_moving_snapshots=snapshots,process_closure={'matched_processes':[]})
        receipt = self.write('receipt.json',dict(schema=recovery.SCHEMA,
            profile={k:v for k,v in recovery.binding(profile_path).items() if k != 'path'},
            verification={k:v for k,v in actual.items() if k != 'process_closure'},
            creation_process_closure={'matched_processes':[],'signals_sent':0}))
        paths['latest-run.json'].write_text('{"later_clean_session":true}\n')
        def audit(run,config,**kwargs):
            recovery.read_moving_inputs(profile,run,kwargs.get('moving_snapshots'))
            return actual
        with patch.object(recovery,'PROFILE',profile_path), patch.object(recovery,'audit_current',side_effect=audit) as invoked:
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                recovery.validate_receipt(receipt,run,{})
            result=recovery.validate_receipt(receipt,run,{},inherited=True)
            self.assertEqual(result['progress'],{'updates':1229})
            self.assertEqual(invoked.call_args.kwargs,{'moving_snapshots':snapshots})
            changed=recovery.read(receipt)
            changed['verification']['original_moving_snapshots']['sequence-ledger.json']='{}'
            receipt.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'snapshot bytes changed'):
                recovery.validate_receipt(receipt,run,{},inherited=True)
            with self.assertRaisesRegex(ValueError,'explicit Boolean'):
                recovery.validate_receipt(receipt,run,{},inherited=1)

    def test_cli_cannot_write_inside_historical_run(self):
        config = self.write('machine.json', {'artifact_root': str(self.root)})
        profile = self.write('profile.json', {'run_dir': str(self.root/'local-script-training/original/run')})
        with patch.object(recovery, 'PROFILE', profile), patch.object(recovery, 'audit_current') as audit:
            with self.assertRaisesRegex(ValueError, 'outside historical runs'):
                recovery.main(['--config', str(config), '--output', str(self.root/'local-script-training/original/run/recovery.json')])
            audit.assert_not_called()

    def test_read_only_verification_checks_wall_and_monotonic_deadlines(self):
        for clock in ('monotonic', 'time'):
            with self.subTest(clock=clock):
                def expired(*args, **kwargs):
                    with patch.object(recovery.time, clock, return_value=10**20):
                        recovery.binding(self.write('deadline.json', {}))
                with patch.object(recovery, '_audit_current', side_effect=expired):
                    with self.assertRaisesRegex(TimeoutError, '120-second'):
                        recovery.audit_current(self.root, {})
                self.assertIsNone(recovery.VERIFY_DEADLINE.get())


if __name__ == '__main__':
    unittest.main()
