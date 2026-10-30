"""Diagnostic source/output isolation and distinct seed ownership; no gameplay."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import benchmark_pipeline_collection as diagnostic


class PipelineDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base = self.root/'run-investigations/pipeline-collection'
        patcher = patch.object(diagnostic, 'BASE', self.base)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(diagnostic.copies, 'ARTIFACTS', self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_separate_reservations_never_reuse_consumed_seeds_and_cover_both_seats(self):
        first = diagnostic.reserve_schedule(self.base/'first')
        second = diagnostic.reserve_schedule(self.base/'second')
        keys = ('engine_seed', 'policy_seed')
        left = {row[key] for row in first['schedule'] for key in keys}
        right = {row[key] for row in second['schedule'] for key in keys}
        self.assertEqual(len(left), 16)
        self.assertFalse(left & right)
        self.assertEqual([row['learner_player'] for row in first['schedule']], [0]*4+[1]*4)
        self.assertEqual(first['paired_arms'], [name for name, _ in diagnostic.ARMS])
        self.assertFalse(first['retries_permitted'])
        ledger = json.loads((self.base/'seed-ledger.json').read_text())
        self.assertEqual(ledger['next_study'], 2)

    def test_output_cannot_be_official_state_or_an_existing_attempt(self):
        with self.assertRaises(ValueError):
            diagnostic.diagnostic_output(self.root/'local-script-training/new', fresh=True)
        output = self.base/'existing'; output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            diagnostic.diagnostic_output(output, fresh=True)

    def test_output_cannot_follow_a_symlink(self):
        self.base.mkdir(parents=True)
        (self.base/'redirect').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            diagnostic.diagnostic_output(self.base/'redirect/new', fresh=True)

    def test_changed_copied_inputs_are_rejected_before_a_worker_runs(self):
        path = self.root/'policy.pt'; path.write_bytes(b'original')
        launch = dict(inputs={'policy': diagnostic.binding(path)})
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Copied diagnostic input changed'):
            diagnostic.verify_inputs(launch)

    def test_private_lease_rejects_nonempty_or_replaced_identity(self):
        output = self.base/'lease'; output.mkdir(parents=True)
        path = output/'diagnostic.lock'; path.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'lease file changed'):
            with diagnostic.local_lease(output):
                self.fail('nonempty lease accepted')
        path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'lease identity changed'):
            with diagnostic.local_lease(output):
                path.unlink(); path.touch()

    def test_collector_requires_the_explicit_excluded_scope(self):
        from experiments.learner import parallel_collection as parallel
        args = SimpleNamespace(plan=self.root/'plan.json', sha256='x')
        with patch.object(parallel, '_read_plan', return_value={}), patch.object(parallel, '_collect_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'isolated collector scope'):
                diagnostic.collect_child(args)
            worker.assert_not_called()


class RevealedPipelineDiagnosticTests(unittest.TestCase):
    def test_new_study_is_explicit_and_the_historical_contract_is_unchanged(self):
        from experiments.learner import interaction_guard
        old = diagnostic.study_contract(diagnostic.LEGACY_RELEASE)
        new = diagnostic.study_contract(diagnostic.REVEALED_RELEASE)
        self.assertEqual((old['schema'], old['policy_support'], old['policy_checkpoint_update']),
                         (diagnostic.SCHEMA, interaction_guard.VERSION, 1392))
        self.assertEqual((new['schema'], new['policy_support'], new['policy_checkpoint_update']),
                         (diagnostic.REVEALED_SCHEMA, interaction_guard.REVEALED_VERSION, 1585))
        launch = dict(new, schema=new['schema']+'-launch', study_release=diagnostic.REVEALED_RELEASE)
        self.assertEqual(diagnostic.launch_contract(launch), new)
        for changed in (dict(checkpoint_selection='completed_before'), dict(policy_checkpoint_update=1584),
                        dict(policy_support=interaction_guard.VERSION), dict(study_release='unreviewed')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                diagnostic.launch_contract(dict(launch, **changed))

    def test_new_study_selects_the_verified_after_state_with_its_initial_dependency(self):
        from experiments.learner import local_script_training as local
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp).resolve()/'plan.json'; plan_path.write_text('{}')
            before = {'checkpoint': 'before1584'}
            initial = {'path': 'before1584.pt', 'sha256': 'initial', 'bytes': 1}
            original = {'parent_descriptor': before, 'checkpoint': initial}
            report = {'cumulative_progress': {'updates': 1585},
                      'after_checkpoint': {'policy': {'after': 'policy'}, 'resume': {'after': 'resume'}}}
            after = dict(plan=diagnostic.binding(plan_path), report={'after': 'report'},
                         initial_checkpoint=initial, **report['after_checkpoint'])
            with patch.object(local, 'descriptor_from_batch', return_value=after) as select, \
                    patch.object(local, 'read_descriptor', return_value=(original, report)):
                self.assertEqual(diagnostic.selected_parent(plan_path, original, report,
                    diagnostic.REVEALED_RELEASE), after)
                select.assert_called_once_with(plan_path)
                for wrong in ({**report, 'cumulative_progress': {'updates': 1584}},
                              {**report, 'cumulative_progress': {'updates': 1586}}):
                    with patch.object(local, 'read_descriptor', return_value=(original, wrong)):
                        with self.assertRaisesRegex(ValueError, 'wrong complete checkpoint'):
                            diagnostic.selected_parent(plan_path, original, wrong, diagnostic.REVEALED_RELEASE)
                with patch.object(local, 'descriptor_from_batch', return_value=dict(after, initial_checkpoint={'wrong': True})):
                    with self.assertRaisesRegex(ValueError, 'initial dependency'):
                        diagnostic.selected_parent(plan_path, original, report, diagnostic.REVEALED_RELEASE)

    def test_historical_study_keeps_the_before_parent(self):
        from experiments.learner import local_script_training as local
        descriptor = {'historical': 'parent1392'}
        with patch.object(local, 'descriptor_from_batch') as after, \
                patch.object(local, 'read_descriptor', return_value=({}, {'cumulative_progress': {'updates': 1392}})):
            self.assertEqual(diagnostic.selected_parent(Path('plan.json'), {'parent_descriptor': descriptor}, {},
                diagnostic.LEGACY_RELEASE), descriptor)
            after.assert_not_called()

    @staticmethod
    def episode(target=(2, 3), *, tiles=((2, 3, 1),), hero=True):
        action = SimpleNamespace(kind='move_to', subject='own', x=target[0], y=target[1])
        observation = SimpleNamespace(heroes=[SimpleNamespace(handle='own', position=(0, 0, 1))] if hero else [],
                                      tiles=[SimpleNamespace(position=tile) for tile in tiles])
        return SimpleNamespace(history=[SimpleNamespace(seq=7, kind='decision', observation=observation)],
                               decisions=[SimpleNamespace(decision_seq=7, action=action)])

    def test_movement_audit_uses_the_actual_decision_observation_and_hero_level(self):
        self.assertEqual(diagnostic.revealed_movement_audit(self.episode()),
                         dict(movement_choices=1, revealed_targets=1, unrevealed_targets=0))
        for episode in (self.episode(target=(9, 9)), self.episode(tiles=((2, 3, 0),))):
            with self.assertRaisesRegex(ValueError, 'unrevealed movement target'):
                diagnostic.revealed_movement_audit(episode)
        with self.assertRaisesRegex(ValueError, 'own hero'):
            diagnostic.revealed_movement_audit(self.episode(hero=False))


if __name__ == '__main__':
    unittest.main()
