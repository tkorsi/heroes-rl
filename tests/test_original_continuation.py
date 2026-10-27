"""Continuation target, ancestry and deadline guards; no native games."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import local_script_training as adapter

SPEC = importlib.util.spec_from_file_location('continuation_runner', Path(__file__).parents[1]/'tools/run_local_training.py')
from experiments.learner.builtin_training_resume import binding as runner_binding

runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


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


if __name__ == '__main__':
    unittest.main()
