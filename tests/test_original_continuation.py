"""Continuation target, ancestry and deadline guards; no native games."""
from copy import deepcopy
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import local_script_training as adapter

SPEC = importlib.util.spec_from_file_location('continuation_runner', Path(__file__).parents[1]/'tools/run_local_training.py')
from experiments.learner.builtin_training_resume import binding as runner_binding

runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ManualWindowTests(unittest.TestCase):
    def test_generic_cap_stays48_and_only_continuation_allows336(self):
        with patch.object(runner.time, 'time', return_value=1000), patch.object(runner.time, 'monotonic', return_value=2000):
            ordinary = runner.budget(48)
            continuation = runner.budget(336, continuation=True)
        self.assertEqual(ordinary['absolute_deadline_unix'], 1000+48*3600)
        self.assertEqual(continuation['absolute_deadline_unix'], 1000+336*3600)
        self.assertEqual(continuation['hard_deadline_monotonic'], 2000+336*3600)
        self.assertEqual(continuation['compute_deadline_unix'], continuation['absolute_deadline_unix']-300)
        self.assertEqual(continuation['compute_deadline_monotonic'], continuation['hard_deadline_monotonic']-300)
        runner.budget(48, continuation=True)  # Explicit shorter manual window remains possible.
        for invalid in (49, 336):
            with self.assertRaises(ValueError):
                runner.budget(invalid)
        for invalid in (0, 337, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                runner.budget(invalid, continuation=True)

    def test_legacy_wrapper_cannot_reset_an_existing_official_run(self):
        source = Path(__file__).parents[1]/'train-original-5000.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wrapper = root/'train-original-5000.sh'; wrapper.write_bytes(source.read_bytes())
            child = root/'train-local.sh'
            child.write_text('#!/bin/sh\nprintf launched > "$SENTINEL"\n'); child.chmod(0o755)
            original = root/'Projects/Heroes-artifacts/local-script-training/original'
            original.mkdir(parents=True)
            for name in ('latest-run.json', 'latest-checkpoint.json', 'sequence-ledger.json'):
                with self.subTest(name=name):
                    marker = original/name; marker.write_text('{}')
                    result = subprocess.run(['sh', str(wrapper)], env={**os.environ, 'HOME':str(root),
                        'SENTINEL':str(root/'launched')}, text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('continue-original-4720.sh', result.stderr)
                    self.assertFalse((root/'launched').exists())
                    marker.unlink()


class OperationalBudgetTests(unittest.TestCase):
    def test_only_shared_and_update_wall_offsets_change(self):
        cfg = dict(interface=adapter.INTERFACE, max_days=365, ordinary_choice_limit=4960,
            reserved_dialog_choices=160, max_total_choices=5120, max_updates=1,
            allow_verified_training_prefix=False, shared_wall_seconds=3600,
            phase_deadlines_seconds=dict(training=2400, update=3540, evaluation=3540),
            attempt_wall_seconds=600, cleanup_reserve_seconds=30, max_optimizer_steps=640,
            ppo={'epochs': 1}, adam={'lr': .0003})
        with patch.object(adapter, 'read_descriptor', return_value=(dict(configuration=cfg), {})):
            changed = adapter.operational_configuration({}, 10800)
            self.assertEqual(cfg['shared_wall_seconds'], 3600)
            self.assertEqual(changed['shared_wall_seconds'], 10800)
            self.assertEqual(changed['phase_deadlines_seconds']['update'], 10740)
            changed['shared_wall_seconds'] = 3600
            changed['phase_deadlines_seconds']['update'] = 3540
            self.assertEqual(changed, cfg)
            self.assertEqual(adapter.operational_configuration({}, None), cfg)
            for invalid in (0, 3599, 21601, True, float('inf'), 10800.0):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    adapter.operational_configuration({}, invalid)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.anchor = self.root/'original'
        self.anchor.mkdir()
        self.parent = {'policy': {'sha256': '045d48215be54bd2bd3f438a12f2626ffe83b2a53445a6a8623254feeb0ac046'},
                       'resume': {'sha256': 'c5f8e601040e738bbda9fee987f996829107dda3737e2954e1cc4ba371c0afcf'},
                       'plan': {'sha256': '89b8b69bf2b451e62de73b36f80e3201724259ff15f77649762cfa5760b8b6a6'}}
        self.progress = dict(updates=699, rollouts=2796, decisions=366475)
        self.config = dict(run_root=str(self.root), default_profile='original')
        self.session = dict(run_dir=str(self.anchor), run_root=str(self.root), profile='original', target_games=5000)
        self.report = dict(status='stopped_failure', owned_cleanup_verified=True, initial_state_unchanged=True,
            full_final_restore=dict(exact=True, progress=self.progress), latest_parent=self.parent,
            latest_progress=self.progress, batches=[])
        for sequence in range(266, 337):
            folder = self.anchor/f'batch-{sequence}'; folder.mkdir()
            schedule = [dict(engine_seed=40_000_000+4*sequence+i, policy_seed=50_000_000+4*sequence+i,
                             learner_player=sequence % 2) for i in range(4)]
            self.write(folder/'plan.json', dict(batch_sequence=sequence, schedule=schedule,
                parent_descriptor=self.parent, execution_output=str(folder/'execution')))
            plan_binding = runner_binding(folder/'plan.json')
            self.write(folder/'report.json', dict(plan_sha256=plan_binding['sha256'], update_status='rolled_back',
                training_performed=False, optimizer_steps=0, optimizer_updates=0, training_samples=0, after_checkpoint=None))
            self.report['batches'].append(dict(sequence=sequence, mode='training',
                status='passed' if sequence<336 else 'failed', checkpoint_verified=sequence<336,
                plan=plan_binding, report=runner_binding(folder/'report.json')))
        self.shutdown = dict(owned_cleanup_verified=True, within_hard_deadline=True, worker_returncode=1)
        self.write(self.anchor/'session.json', self.session)
        self.write(self.anchor/'campaign-report.json', self.report)
        self.write(self.anchor/'shutdown.json', self.shutdown)
        self.write(self.root/'latest-run.json', dict(run_dir=str(self.anchor), session=runner_binding(self.anchor/'session.json')))
        self.write(self.root/'latest-checkpoint.json', dict(parent=self.parent, progress=self.progress, run_dir=str(self.anchor)))
        self.write(self.root/'sequence-ledger.json', dict(next_sequence=337, consumed=[dict(sequence=n) for n in range(266,337)]))
        self.reader = patch.object(adapter, 'read_descriptor', return_value=({}, self.valid_report()))
        self.reader.start(); self.addCleanup(self.reader.stop)

    @staticmethod
    def valid_report():
        return dict(status='finished', worker_complete=True, supervisor_cleanup_verified=True, within_shared_budget=True,
                    optimizer_updates=1, update_status='checkpoint_verified',
                    attempts=[dict(status='passed', completion_receipt=True, cleanup_verified=True,
                                   identity_verified=True, replay_verified=True)]*4)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def select(self):
        return runner.original_continuation(self.anchor, self.config)

    def recovery_receipt(self, path, ledger):
        self.write(path, dict(schema='heroes-interrupted1229-recovery-v1', verification={
            'original_moving_snapshots': {'latest-run.json': '{}', 'latest-checkpoint.json': '{}',
                                        'sequence-ledger.json': json.dumps(ledger)}}))
        return runner_binding(path)

    def test_failed_but_fully_rolled_back_closed_anchor_continues_at337(self):
        parent, remaining, proof = self.select()
        self.assertEqual(parent, self.parent)
        self.assertEqual(remaining, 4720)
        self.assertEqual(proof['next_sequence'], 337)
        self.assertEqual(proof['admitted_before_session'], 280)
        self.assertIn(336, [x['sequence'] for x in proof['seed_ledger_snapshot']['consumed']])

    def test_incomplete_shutdown_or_restore_rejected(self):
        for which, field in [('shutdown.json', 'owned_cleanup_verified'), ('shutdown.json', 'within_hard_deadline'),
                              ('campaign-report.json', 'initial_state_unchanged')]:
            with self.subTest(field=field):
                source = self.shutdown if which == 'shutdown.json' else self.report
                self.write(self.anchor/which, {**source, field: False})
                with self.assertRaisesRegex(ValueError, 'verified closed'):
                    self.select()
                self.write(self.anchor/which, source)

    def test_newest_attempt_failure_before_first_update_cannot_bypass_cleanup(self):
        failed = self.root/'newest-failed'; failed.mkdir()
        self.write(failed/'session.json', {**self.session, 'run_dir': str(failed), 'target_games': 4720})
        self.write(failed/'campaign-report.json', {**self.report, 'batches': []})
        self.write(failed/'shutdown.json', {**self.shutdown, 'owned_cleanup_verified': False})
        self.write(self.root/'latest-run.json', dict(run_dir=str(failed), session=runner_binding(failed/'session.json')))
        with self.assertRaisesRegex(ValueError, 'verified closed'):
            self.select()
        self.write(failed/'shutdown.json', self.shutdown)
        self.assertEqual(self.select()[1], 4720)

    def test_null_final_restore_is_an_unclosed_run(self):
        self.write(self.anchor/'campaign-report.json', {**self.report, 'full_final_restore': None})
        with self.assertRaisesRegex(ValueError, 'verified closed'):
            self.select()

    def test_seed_ledger_cannot_reset_reuse_or_omit_failed336(self):
        for ledger in [dict(next_sequence=336, consumed=[dict(sequence=335)]),
                       dict(next_sequence=337, consumed=[dict(sequence=336), dict(sequence=336)]),
                       dict(next_sequence=336, consumed=[dict(sequence=336)])]:
            with self.subTest(ledger=ledger):
                self.write(self.root/'sequence-ledger.json', ledger)
                with self.assertRaisesRegex(ValueError, 'seed ledger'):
                    self.select()

    def descendant(self, foreign=False):
        child = self.root/'continued'; child.mkdir()
        parent = dict(plan=dict(sha256='child'), policy=dict(sha256='child-policy'), resume=dict(sha256='child-resume'))
        progress = dict(updates=700, rollouts=2800, decisions=366500)
        session = {**self.session, 'run_dir': str(child), 'target_games': 4720,
                   'original_continuation': dict(origin_run='foreign' if foreign else str(self.anchor))}
        self.write(child/'session.json', session)
        self.write(child/'campaign-report.json', {**self.report, 'latest_parent': parent,
            'full_final_restore': dict(exact=True, progress=progress), 'latest_progress': progress})
        self.write(child/'shutdown.json', self.shutdown)
        self.write(self.root/'latest-checkpoint.json', dict(parent=parent, progress=progress, run_dir=str(child)))
        ledger = runner.read(self.root/'sequence-ledger.json')
        ledger['consumed'].append(dict(sequence=337)); ledger['next_sequence']=338
        self.write(self.root/'sequence-ledger.json', ledger)
        plan = dict(profile='original', mode='training', scope=adapter.SCOPE, batch_sequence=337,
                    parent_descriptor=self.parent, local_session=runner_binding(child/'session.json'),
                    schedule=[dict(engine_seed=40_000_000+4*337+i, policy_seed=50_000_000+4*337+i,
                                   learner_player=1) for i in range(4)])
        report = dict(status='finished', worker_complete=True, supervisor_cleanup_verified=True, within_shared_budget=True,
                      optimizer_updates=1, update_status='checkpoint_verified',
                      attempts=[dict(status='passed', completion_receipt=True, cleanup_verified=True,
                                     identity_verified=True, replay_verified=True)]*4)
        return parent, plan, report

    def test_repeated_manual_resume_reduces_target(self):
        child, plan, report = self.descendant()
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d: (plan, report) if d == child else ({}, self.valid_report())):
            parent, remaining, proof = self.select()
        self.assertEqual(remaining, 4716)
        self.assertEqual(proof['next_sequence'], 338)
        self.assertEqual(proof['admitted_before_session'], 284)

    def test_failed336_admission_or_after_checkpoint_rejected(self):
        row = self.report['batches'][-1]
        report_path = Path(row['report']['path'])
        report = json.loads(report_path.read_text())
        report['training_samples'] = 1
        self.write(report_path, report)
        row['report'] = runner_binding(report_path)
        self.write(self.anchor/'campaign-report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'fully rolled back'):
            self.select()

    def test_mutated_descendant_session_or_reused_seed_rejected(self):
        child, plan, report = self.descendant()
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d: (plan, report) if d == child else ({}, self.valid_report())):
            plan['schedule'][0]['engine_seed'] -= 4
            with self.assertRaisesRegex(ValueError, 'training seeds'):
                self.select()
            plan['schedule'][0]['engine_seed'] += 4
            path = Path(plan['local_session']['path'])
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'session binding'):
                self.select()

    def test_foreign_or_benchmark_descendant_rejected(self):
        child, plan, report = self.descendant(foreign=True)
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d: (plan, report) if d == child else ({}, self.valid_report())):
            with self.assertRaisesRegex(ValueError, 'foreign/benchmark'):
                self.select()

    def test_recovery_is_explicit_and_keeps_historical_failure_immutable(self):
        child, plan, report = self.descendant()
        run = self.root/'continued'
        historical = runner.read(run/'campaign-report.json')
        historical.update(owned_cleanup_verified=False, initial_state_unchanged=False,
                          full_final_restore=None)
        self.write(run/'campaign-report.json', historical)
        self.write(run/'shutdown.json', dict(owned_cleanup_verified=False,
                                           within_hard_deadline=False, worker_returncode=-9))
        self.write(self.root/'latest-run.json', dict(run_dir=str(run), session=runner_binding(run/'session.json')))
        receipt = self.root/'recovery.json'; self.write(receipt, {'separate': 'receipt'})
        before = {p: p.read_bytes() for p in (run/'campaign-report.json', run/'shutdown.json')}
        proof = dict(parent=child, progress=historical['latest_progress'],
                     evidence={'receipt': runner_binding(receipt)})
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())), \
                patch('tools.recover_interrupted_training.validate_receipt', return_value=proof) as validate:
            with self.assertRaisesRegex(ValueError, 'verified closed'):
                self.select()
            selected, remaining, continuation = runner.original_continuation(self.anchor, self.config, receipt)
            self.assertEqual(selected, child)
            self.assertEqual(remaining, 4716)
            self.assertEqual(continuation['evidence']['recovery/receipt'], runner_binding(receipt))
            validate.assert_called_once_with(receipt, run, self.config)
            proof['progress'] = dict(updates=699)
            with self.assertRaisesRegex(ValueError, 'Recovery checkpoint differs'):
                runner.original_continuation(self.anchor, self.config, receipt)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_receipt_cannot_reset_a_later_clean_descendant(self):
        child, plan, report = self.descendant()
        inherited = self.root/'recovery-proof.json'; self.write(inherited, {'historical': 'proof'})
        session_path = Path(plan['local_session']['path'])
        session = runner.read(session_path)
        session['original_continuation']['evidence'] = {'recovery/receipt': runner_binding(inherited)}
        self.write(session_path, session)
        plan['local_session'] = runner_binding(session_path)
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())), \
                patch('tools.recover_interrupted_training.validate_receipt') as validate:
            selected, remaining, proof = runner.original_continuation(self.anchor, self.config, self.root/'old-recovery.json')
            self.assertEqual(proof['evidence']['recovery/receipt'], runner_binding(inherited))
            inherited.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Inherited recovery evidence changed'):
                self.select()
        self.assertEqual(selected, child)
        self.assertEqual(remaining, 4716)
        validate.assert_not_called()

    def test_clean_zero_update_run_can_inherit_recovered_checkpoint_owner(self):
        child, plan, report = self.descendant()
        interrupted = self.root/'continued'
        historical = runner.read(interrupted/'campaign-report.json')
        self.write(interrupted/'campaign-report.json', {**historical,
            'owned_cleanup_verified': False, 'full_final_restore': None})
        self.write(interrupted/'shutdown.json', {**self.shutdown, 'within_hard_deadline': False})
        receipt = self.root/'recovery.json'
        self.recovery_receipt(receipt, runner.read(self.root/'sequence-ledger.json'))
        newest = self.root/'clean-zero-update'; newest.mkdir()
        session = {**self.session, 'run_dir': str(newest), 'initial_parent': child,
            'original_continuation': {'origin_run': str(self.anchor),
                'evidence': {'recovery/recovery_receipt': runner_binding(receipt)}}}
        self.write(newest/'session.json', session)
        self.write(newest/'campaign-report.json', {**historical, 'batches': []})
        self.write(newest/'shutdown.json', self.shutdown)
        self.write(self.root/'latest-run.json', dict(run_dir=str(newest), session=runner_binding(newest/'session.json')))
        proof = dict(parent=child, progress=historical['latest_progress'],
                     evidence={'recovery_receipt': runner_binding(receipt)})
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())), \
                patch('tools.recover_interrupted_training.validate_receipt', return_value=proof) as validate:
            selected, remaining, continuation = self.select()
            self.assertEqual(selected, child)
            self.assertEqual(remaining, 4716)
            self.assertEqual(continuation['next_sequence'], 338)
            validate.assert_called_once_with(str(receipt), interrupted, self.config, inherited=True)
            # A newer report cannot authorize rolling the pointer backward.
            self.write(newest/'campaign-report.json', {**historical, 'latest_parent': {'later': 'checkpoint'}})
            with self.assertRaisesRegex(ValueError, 'cannot reset a later checkpoint'):
                self.select()

    def test_recovery_receipt_ledger_requires_exact_append_including_failed555(self):
        child, plan, report = self.descendant()
        ledger = runner.read(self.root/'sequence-ledger.json')
        ledger['consumed'] += [dict(sequence=555, reserved_utc='failed reservation'),
                               dict(sequence=556, reserved_utc='later reservation')]
        ledger['next_sequence'] = 557
        receipt = self.root/'recovery.json'
        bound = self.recovery_receipt(receipt, ledger)
        session_path = Path(plan['local_session']['path'])
        session = runner.read(session_path)
        session['original_continuation']['evidence'] = {'recovery/recovery_receipt': bound}
        self.write(session_path, session); plan['local_session'] = runner_binding(session_path)
        appended = deepcopy(ledger)
        appended['consumed'].append(dict(sequence=557, reserved_utc='new reservation'))
        appended['next_sequence'] = 558
        self.write(self.root/'sequence-ledger.json', appended)
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())):
            self.assertEqual(self.select()[2]['next_sequence'], 558)
            dropped = deepcopy(appended)
            dropped['consumed'] = [row for row in dropped['consumed'] if row['sequence'] != 555]
            self.write(self.root/'sequence-ledger.json', dropped)
            with self.assertRaisesRegex(ValueError, 'exact append.*recovery receipt'):
                self.select()
            changed = deepcopy(appended)
            changed['consumed'][-3]['reserved_utc'] = 'changed existing reservation'
            self.write(self.root/'sequence-ledger.json', changed)
            with self.assertRaisesRegex(ValueError, 'exact append.*recovery receipt'):
                self.select()

    def test_ancestor_session_snapshot_preserves_prior_failed_reservations(self):
        child, plan, report = self.descendant()
        ledger = runner.read(self.root/'sequence-ledger.json')
        ledger['consumed'] += [dict(sequence=555), dict(sequence=556)]
        ledger['next_sequence'] = 557
        session_path = Path(plan['local_session']['path'])
        session = runner.read(session_path)
        session['original_continuation']['seed_ledger_snapshot'] = deepcopy(ledger)
        self.write(session_path, session); plan['local_session'] = runner_binding(session_path)
        ledger['consumed'] = [row for row in ledger['consumed'] if row['sequence'] != 555]
        self.write(self.root/'sequence-ledger.json', ledger)
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())):
            with self.assertRaisesRegex(ValueError, 'exact append.*ancestor session'):
                self.select()

    def test_clean_zero_update_run_cannot_reuse_failed_or_preparing869(self):
        child, plan, report = self.descendant()
        interrupted = self.root/'continued'
        historical = runner.read(interrupted/'campaign-report.json')
        self.write(interrupted/'campaign-report.json', {**historical,
            'owned_cleanup_verified': False, 'full_final_restore': None})
        self.write(interrupted/'shutdown.json', {**self.shutdown, 'within_hard_deadline': False})
        ledger = runner.read(self.root/'sequence-ledger.json')
        receipt = self.root/'recovery.json'; receipt_binding = self.recovery_receipt(receipt, ledger)
        newest = self.root/'clean-zero-update'; newest.mkdir()
        session = {**self.session, 'run_dir': str(newest), 'initial_parent': child,
            'original_continuation': {'origin_run': str(self.anchor), 'seed_ledger_snapshot': ledger,
                'evidence': {'recovery/recovery_receipt': receipt_binding}}}
        self.write(newest/'session.json', session); self.write(newest/'shutdown.json', self.shutdown)
        self.write(self.root/'latest-run.json', dict(run_dir=str(newest), session=runner_binding(newest/'session.json')))
        proof = dict(parent=child, progress=historical['latest_progress'],
                     evidence={'recovery_receipt': receipt_binding})
        with patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                          (plan, report) if d == child else ({}, self.valid_report())), \
                patch('tools.recover_interrupted_training.validate_receipt', return_value=proof):
            for status in ('failed', 'preparing'):
                with self.subTest(status=status):
                    self.write(newest/'campaign-report.json', {**historical, 'batches': [
                        dict(mode='training', sequence=869, status=status, checkpoint_verified=False)]})
                    with self.assertRaisesRegex(ValueError, 'omitted a consumed training attempt'):
                        self.select()
                    advanced = deepcopy(ledger)
                    advanced['consumed'].append(dict(sequence=869, reserved_utc='consumed before failure'))
                    advanced['next_sequence'] = 870
                    self.write(self.root/'sequence-ledger.json', advanced)
                    self.assertEqual(self.select()[2]['next_sequence'], 870)
                    self.write(self.root/'sequence-ledger.json', ledger)


if __name__ == '__main__':
    unittest.main()
