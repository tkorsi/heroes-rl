"""A completed-run anchor never resets the model, goal or consumed seeds."""
import unittest

from tools import continuation_goal as goal
from tests import test_continuation_goal as fixtures

write = fixtures.write


class CompletedGoalTests(unittest.TestCase):
    relative = fixtures.GoalAncestryTests.relative
    checkpoint = fixtures.GoalAncestryTests.checkpoint
    pin = fixtures.GoalAncestryTests.pin
    audit = fixtures.GoalAncestryTests.audit

    def setUp(self):
        fixtures.GoalAncestryTests.setUp(self)
        self.value.update(schema=goal.COMPLETED_ANCHOR_SCHEMA,
            anchor_progress={'updates': 1585, 'rollouts': 6340, 'decisions': 922041},
            admitted_original_at_anchor=3824, additional_games=1176,
            minimum_next_sequence=1226, anchor_sequence=1225,
            required_policy_support='visible-interaction-budget-revealed-v1')
        for name in ('failed_sequence', 'failed_worker', 'failed_plan', 'failed_report'):
            self.value.pop(name)
        self.run = self.run_root / 'anchor'
        self.session = {'run_root': str(self.run_root), 'run_dir': str(self.run),
                        'profile': 'original', 'target_games': 4,
                        'original_continuation': {'origin_run': str(self.run_root / 'origin'),
                                                  'admitted_before_session': 3820}}
        session = write(self.run / 'session.json', self.session)
        plan, report = self.descriptors[self.anchor['plan']['sha256']]
        plan.update(local_session=session, batch_sequence=1225)
        report['cumulative_progress'] = self.value['anchor_progress'].copy()
        self.anchor['plan'] = write(self.run / 'plan.json', plan)
        self.anchor['report'] = write(self.run / 'report.json', report)
        self.descriptors[self.anchor['plan']['sha256']] = plan, report
        self.value['anchor_parent'] = {k: self.relative(v) for k, v in self.anchor.items()}
        self.campaign = dict(status='finished', stop_reason='training_game_target_reached',
            profile='original', target_games=4, latest_parent=self.anchor,
            latest_progress=self.value['anchor_progress'].copy(), owned_cleanup_verified=True,
            initial_state_unchanged=True, full_final_restore={'exact': True,
                'progress': self.value['anchor_progress'].copy()},
            original_continuation=self.session['original_continuation'],
            batches=[dict(mode='training', status='passed', completed_games=4,
                checkpoint_verified=True, sequence=1225, plan=self.anchor['plan'],
                report=self.anchor['report'])])
        self.shutdown = dict(owned_cleanup_verified=True, worker_returncode=0, within_hard_deadline=True)
        self.proof.update(admitted_before_session=3824, remaining_before_session=1176,
            next_sequence=1226, seed_ledger_snapshot={'next_sequence': 1226, 'consumed': [{'sequence': 1225}]})
        self.repin_completion()

    def repin_completion(self):
        self.value['completed_run'] = {name: self.relative(write(self.run / name, value)) for name, value in (
            ('session.json', self.session), ('campaign-report.json', self.campaign), ('shutdown.json', self.shutdown))}
        self.pin()

    def advance(self, count):
        cursor = fixtures.GoalAncestryTests.advance(self, count)
        for plan, _ in self.descriptors.values():
            if plan.get('parent_descriptor') is not None:
                plan['policy_support'] = self.value['required_policy_support']
        return cursor

    def test_anchor_and_verified_descendants_recalculate_same_original_goal(self):
        self.assertEqual(self.audit()[0], 1176)
        cursor = self.advance(2)
        remaining, proof = self.audit(cursor)
        self.assertEqual(remaining, 1168)
        self.assertEqual(proof['schema'], 'heroes-completed-run-continuation-goal-audit-v1')
        self.assertEqual(proof['required_policy_support'], 'visible-interaction-budget-revealed-v1')
        self.assertNotIn('failed_plan', proof)
        self.proof['seed_ledger_snapshot']['consumed'].append({'sequence': 1226})
        self.proof['next_sequence'] = self.proof['seed_ledger_snapshot']['next_sequence'] = 1227
        self.assertEqual(self.audit(cursor)[0], 1168)

    def test_target_completion_is_zero_and_one_extra_update_refuses(self):
        self.assertEqual(self.audit(self.advance(294))[0], 0)
        self.proof.update(admitted_before_session=3824, remaining_before_session=1176)
        with self.assertRaisesRegex(ValueError, 'within its game budget'):
            self.audit(self.advance(295))

    def test_changed_final_evidence_and_dirty_shutdown_refuse(self):
        path = self.run / 'shutdown.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.audit()
        self.shutdown['worker_returncode'] = 1
        self.repin_completion()
        with self.assertRaisesRegex(ValueError, 'cleanly finalized'):
            self.audit()

    def test_progress_target_and_last_anchor_must_match(self):
        self.campaign['latest_progress']['updates'] -= 1
        self.repin_completion()
        with self.assertRaisesRegex(ValueError, 'incomplete progress'):
            self.audit()
        self.campaign['latest_progress'] = self.value['anchor_progress'].copy()
        self.campaign['batches'][0]['sequence'] -= 1
        self.repin_completion()
        with self.assertRaisesRegex(ValueError, 'finalized admitted target'):
            self.audit()

    def test_consumed_anchor_missing_duplicate_or_reused_seed_refuses(self):
        for sequences, next_sequence in (([], 1226), ([1225, 1225], 1226), ([1225], 1225), ([1226], 1227)):
            with self.subTest(sequences=sequences, next_sequence=next_sequence):
                self.proof['seed_ledger_snapshot'] = dict(next_sequence=next_sequence,
                    consumed=[{'sequence': value} for value in sequences])
                self.proof['next_sequence'] = next_sequence
                with self.assertRaisesRegex(ValueError, 'consumed reservation'):
                    self.audit()

    def test_descendant_requires_explicit_revealed_guard_and_verified_admission(self):
        cursor = self.advance(1)
        plan, report = self.descriptors[cursor['plan']['sha256']]
        plan['policy_support'] = 'visible-interaction-budget-v1'
        with self.assertRaisesRegex(ValueError, 'policy support intervention'):
            self.audit(cursor)
        plan['policy_support'] = self.value['required_policy_support']
        report['attempts'][0]['completion_receipt'] = False
        with self.assertRaisesRegex(ValueError, 'fully admitted'):
            self.audit(cursor)


if __name__ == '__main__':
    unittest.main()
