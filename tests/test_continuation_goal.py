"""Fixed cohort accounting and controller handoff without games or PPO."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.learner import local_script_training as adapter
from experiments.learner import builtin_first_run as pilot, inference
from tools import continuation_goal as goal
from tests import test_local_session_game_cap as fixtures

runner = fixtures.runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


class GoalAncestryTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run_root = self.root / 'local-script-training/original'
        self.config = {'artifact_root': str(self.root), 'run_root': str(self.run_root),
                       'default_profile': 'original'}
        self.descriptors = {}
        self.anchor = self.checkpoint('anchor', None)
        self.value = {'schema': 'heroes-manual-continuation-goal-v1', 'id': 'test-goal',
            'run_root_relative': 'local-script-training/original', 'origin_run': 'origin',
            'anchor_parent': {k: self.relative(v) for k, v in self.anchor.items()},
            'anchor_progress': {'updates': 1392, 'rollouts': 5568, 'decisions': 729985},
            'additional_games': 772, 'admitted_original_at_anchor': 3052,
            'minimum_next_sequence': 1033, 'failed_sequence': 1032}
        failure_plan = write(self.run_root / 'failed/plan.json', {'sequence': 1032})
        self.value['failed_plan'] = self.relative(failure_plan)
        self.value['failed_report'] = self.relative(write(self.run_root / 'failed/report.json',
                                                         {'status': 'stopped'}))
        self.failure = dict(plan_sha256=failure_plan['sha256'], training_performed=False,
            optimizer_updates=0, optimizer_steps=0, training_samples=0, update_status='not_started',
            attempts=[dict(completion_receipt=False, batch_admission='rejected') for _ in range(4)])
        self.value['failed_worker'] = self.relative(write(self.run_root / 'failed/worker.json', self.failure))
        self.proof = dict(origin_run=str(self.run_root / 'origin'), admitted_before_session=3052,
            remaining_before_session=1948, next_sequence=1033, target_games=5000,
            seed_ledger_snapshot=dict(next_sequence=1033, consumed=[{'sequence': 1032}]))
        self.pin()
        self.addCleanup(patch.stopall)
        patch.object(goal, 'ROOT', self.root).start()
        patch.object(adapter, 'read_descriptor', side_effect=lambda d: self.descriptors[d['plan']['sha256']]).start()

    def relative(self, value):
        return {**value, 'path': str(Path(value['path']).relative_to(self.run_root))}

    def checkpoint(self, name, parent):
        path = self.run_root / name
        session = write(path / 'session.json', {'run_root': str(self.run_root)})
        plan = dict(name=name, parent_descriptor=parent, profile='original', mode='training',
                    scope=adapter.SCOPE, local_session=session)
        report = dict(status='finished', worker_complete=True, supervisor_cleanup_verified=True,
            within_shared_budget=True, optimizer_updates=1, update_status='checkpoint_verified',
            attempts=[dict(status='passed', completion_receipt=True, cleanup_verified=True,
                           identity_verified=True, replay_verified=True) for _ in range(4)])
        result = {k: write(path / (k+'.json'), value) for k, value in (
            ('plan', plan), ('report', report), ('policy', name), ('resume', name), ('initial_checkpoint', name))}
        self.descriptors[result['plan']['sha256']] = (plan, report)
        return result

    def pin(self):
        pin = write(self.root / 'goal.json', self.value)
        goal.GOALS['test-goal'] = ('goal.json', pin['sha256'])
        self.addCleanup(lambda: goal.GOALS.pop('test-goal', None))

    def audit(self, parent=None):
        return goal.remaining_games('test-goal', self.config, parent or self.anchor,
                                    self.proof['remaining_before_session'], self.proof)

    def advance(self, count):
        cursor = self.anchor
        for i in range(count):
            cursor = self.checkpoint(str(i), cursor)
        self.proof['admitted_before_session'] += 4*count
        self.proof['remaining_before_session'] -= 4*count
        return cursor

    def test_initial_and_resumed_goal_count_only_verified_descendants(self):
        self.assertEqual(self.audit()[0], 772)
        cursor = self.advance(2)
        remaining, proof = self.audit(cursor)
        self.assertEqual(remaining, 764)
        self.assertEqual(proof['admitted_since_anchor'], 8)
        self.assertEqual(proof['additional_games'], 772)
        # Another failed reservation changes neither admission nor the goal.
        self.proof['seed_ledger_snapshot']['consumed'].append({'sequence': 1033})
        self.proof['seed_ledger_snapshot']['next_sequence'] = self.proof['next_sequence'] = 1034
        self.assertEqual(self.audit(cursor)[0], 764)

    def test_exact_goal_completion_returns_zero_and_excess_descendant_refuses(self):
        self.assertEqual(self.audit(self.advance(193))[0], 0)
        self.proof.update(admitted_before_session=3052, remaining_before_session=1948)
        with self.assertRaisesRegex(ValueError, 'within its game budget'):
            self.audit(self.advance(194))

    def test_missing_anchor_unverified_update_and_foreign_session_fail(self):
        foreign = self.checkpoint('foreign', None)
        with self.assertRaises((ValueError, TypeError)):
            self.audit(foreign)
        cursor = self.advance(1)
        plan, report = self.descriptors[cursor['plan']['sha256']]
        report['attempts'][0]['completion_receipt'] = False
        with self.assertRaisesRegex(ValueError, 'fully admitted'):
            self.audit(cursor)
        report['attempts'][0]['completion_receipt'] = True
        plan['local_session'] = write(self.run_root / 'outside-session.json', {'run_root': str(self.root / 'other')})
        with self.assertRaisesRegex(ValueError, 'another campaign'):
            self.audit(cursor)

    def test_changed_goal_and_anchor_evidence_fail(self):
        path = self.root / 'goal.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'goal changed'):
            self.audit()
        self.pin()
        Path(self.anchor['resume']['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.audit()

    def test_non_batch_target_foreign_root_missing_reservation_and_count_mismatch_fail(self):
        self.value['additional_games'] = 773
        self.pin()
        with self.assertRaisesRegex(ValueError, 'positive multiple of four'):
            self.audit()
        self.value['additional_games'] = 772
        self.pin()
        self.config['run_root'] = str(self.root / 'foreign')
        with self.assertRaisesRegex(ValueError, 'campaign root'):
            self.audit()
        self.config['run_root'] = str(self.run_root)
        self.proof['seed_ledger_snapshot']['consumed'] = []
        with self.assertRaisesRegex(ValueError, 'consumed failed'):
            self.audit()
        self.proof['seed_ledger_snapshot']['consumed'] = [{'sequence': 1032}]
        self.proof['admitted_before_session'] += 4
        with self.assertRaisesRegex(ValueError, 'independently verified'):
            self.audit()


class GoalControllerTests(unittest.TestCase):
    setUp = fixtures.SessionGameCapTests.setUp
    argv = fixtures.SessionGameCapTests.argv
    arguments = fixtures.SessionGameCapTests.arguments
    start_patches = fixtures.SessionGameCapTests.start_patches

    def test_goal_conflicts_never_invoke_start(self):
        import sys
        for extra, continuation in ((('--session-goal', 'checkpoint1392-remaining772-v1'), False),
            (('--session-goal', 'unknown'), True),
            (('--session-goal', 'checkpoint1392-remaining772-v1', '--session-games', '772'), True)):
            with self.subTest(extra=extra), patch.object(sys, 'argv', self.argv(*extra, continuation=continuation)), \
                    patch.object(runner, 'start') as start, self.assertRaises(SystemExit), \
                    __import__('contextlib').redirect_stderr(io.StringIO()):
                runner.main()
            start.assert_not_called()

    def test_goal_identifier_is_rechecked_in_frozen_release_and_complete_is_noop(self):
        args = self.arguments('--session-goal', 'checkpoint1392-remaining772-v1')
        release = self.root / 'frozen-release'
        class Reentered(BaseException):
            pass
        with ExitStack() as stack:
            self.start_patches(stack)
            audit = stack.enter_context(patch.object(goal, 'remaining_games', return_value=(772, {'remaining_games': 772})))
            freeze = stack.enter_context(patch.object(runner, 'release_checkout', return_value=(release, 'f'*40)))
            stack.enter_context(patch.object(runner.os, 'chdir'))
            reenter = stack.enter_context(patch.object(runner.os, 'execv', side_effect=Reentered))
            with self.assertRaises(Reentered):
                runner.start(args)
            command = reenter.call_args.args[1]
            self.assertEqual(command[command.index('--session-goal')+1], 'checkpoint1392-remaining772-v1')
            self.assertNotIn('--session-games', command)
            audit.return_value = (0, {'remaining_games': 0})
            freeze.reset_mock()
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(runner.start(args), 0)
            freeze.assert_not_called()
            # The second ownership acquisition recalculates rather than forwarding 772.
            stack.enter_context(patch.object(runner, 'selected_execution', return_value={}))
            stack.enter_context(patch.object(adapter, 'check_policy_support', return_value=None))
            restore = stack.enter_context(patch.object(adapter, 'restore_parent'))
            self.assertEqual(runner.launch_owned(args), 0)
            restore.assert_not_called()
        self.assertFalse(self.run_root.exists())

    def test_owned_launch_records_recomputed_goal_and_frozen_bindings(self):
        args = self.arguments('--session-goal', 'checkpoint1392-remaining772-v1')
        execution = dict(overlapped_verification=False, collection_workers=4,
                         prefix_helpers=0, text_leaf_replay=False, experiments=[])
        with ExitStack() as stack:
            self.start_patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            audit = stack.enter_context(patch.object(goal, 'remaining_games',
                return_value=(768, {'remaining_games': 768, 'admitted_since_anchor': 4})))
            stack.enter_context(patch.object(runner, 'selected_execution', return_value=execution))
            stack.enter_context(patch.object(adapter, 'check_policy_support', return_value=None))
            stack.enter_context(patch.object(adapter, 'restore_parent', return_value=self.state))
            stack.enter_context(patch.object(adapter, 'read_descriptor', return_value=({}, {})))
            stack.enter_context(patch.object(adapter, 'check_lineage'))
            stack.enter_context(patch.object(adapter, 'operational_configuration', return_value={'preserved': True}))
            stack.enter_context(patch.object(inference, 'source_identity', return_value={'tracked_worktree_dirty': False}))
            stack.enter_context(patch.object(pilot, 'implementation', return_value={}))
            stack.enter_context(patch.object(pilot, 'prerequisites', return_value={}))
            supervise = stack.enter_context(patch.object(runner, 'supervise', return_value=0))
            self.assertEqual(runner.launch_owned(args), 0)
        session = runner.read(supervise.call_args.args[0])
        self.assertEqual(session['target_games'], 768)
        self.assertEqual(session['session_goal'], audit.return_value[1])
        self.assertEqual(session['configuration'], {'preserved': True})
        for name in ('tools/continuation_goal.py', 'configs/checkpoint1392-remaining772.json',
                     'continue-772.sh', 'tools/continue_772_games.py'):
            self.assertIn(name, session['frozen_files'])
        self.assertFalse((self.run_root / 'sequence-ledger.json').exists())


if __name__ == '__main__':
    unittest.main()
