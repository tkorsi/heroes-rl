"""Synthetic repair scope checks; no engine or optimizer is launched."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.learner import builtin_v8_session as scope
from experiments.learner.builtin_training_resume import binding


class RepairTests(unittest.TestCase):
    def test_exact_original_budget_and_immutable_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root/'original/campaign-session.json'
            failure_path = root/'runs/v8-update92-heldout-092/execution/report.json'
            cleanup_path = original_path.parent/'controller-result.json'
            original = dict(started_unix=1000, started_monotonic=100,
                            absolute_deadline_unix=8200, compute_deadline_unix=7900)
            failure = dict(training_samples=0, optimizer_steps=0,
                           attempts=[dict(sampled_choices=0), dict(status='not_started')])
            for path, value in [(original_path,original),(failure_path,failure),
                                (cleanup_path,dict(controller_cleanup_verified=True,cleanup_incomplete=[]))]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            repair = dict(attempt=2,original_session=binding(original_path),failure=binding(failure_path),
                          controller_result=binding(cleanup_path),absolute_deadline_unix=8200)
            release = dict(repair=repair, source={'synthetic':True})
            release_path = root/'release.json';release_path.write_text(json.dumps(release))
            session_path = root/'repair/campaign-session.json'
            with patch.object(scope,'DURABLE_ROOT',root), patch.object(scope,'ORIGINAL_SESSION',original_path), \
                 patch.object(scope,'ORIGINAL_SESSION_SHA',binding(original_path)['sha256']), \
                 patch.object(scope,'ORIGINAL_FAILURE_SHA',binding(failure_path)['sha256']), \
                 patch.object(scope,'REPAIR','attempt2'), patch.object(scope,'SESSION',session_path), \
                 patch.object(scope,'durable_path',side_effect=Path), \
                 patch.object(scope,'validate_release',return_value=release), \
                 patch.object(scope.time,'time',return_value=2000), patch.object(scope.time,'monotonic',return_value=1100):
                self.assertEqual(scope.repair_origin(release),original)
                session=scope.create_session(release_path)
                self.assertEqual(session['attempt_started_unix'],2000)
                for key,value in original.items():self.assertEqual(session[key],value)
                scope.session_for(session_path)
                with self.assertRaises(ValueError):scope.create_session(release_path)
                repair['absolute_deadline_unix']=9200
                with self.assertRaisesRegex(ValueError,'extend'):scope.repair_origin(release)
                repair['absolute_deadline_unix']=8200
                with patch.object(scope.time,'time',return_value=7900):
                    with self.assertRaisesRegex(ValueError,'expired'):scope.repair_origin(release)
                with patch.object(scope.time,'monotonic',return_value=7000):
                    with self.assertRaisesRegex(ValueError,'expired'):scope.repair_origin(release)
                failure_path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'failed baseline'):scope.repair_origin(release)
            self.assertEqual(json.loads(original_path.read_text()),original)

    def test_failed_training_seed_tuple_stays_consumed(self):
        from experiments.learner import builtin_v8_training as training
        from experiments.learner.builtin_training import schedule
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prior=root/'runs/failed-original/plan.json'
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({'schedule':schedule(94)}))
            with patch.object(training,'DURABLE_ROOT',root):
                with self.assertRaises(ValueError):training.unused_schedule(94)
                training.unused_schedule(95)

    def test_release_mode_is_explicit(self):
        with patch.object(scope,'REPAIR',''):
            with self.assertRaises(ValueError):scope.repair_origin({'repair':{'attempt':2}})
        with patch.object(scope,'REPAIR','attempt2'):
            with self.assertRaises(ValueError):scope.repair_origin({})


class ArtifactRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = dict(started_unix=1000, started_monotonic=100,
                             absolute_deadline_unix=8200, compute_deadline_unix=7900)
        self.original_path = self.root/'original/campaign-session.json'
        self.failure_path = self.root/'runs/v8-update92-heldout-092/execution/report.json'
        original_bound = self.save(self.original_path, self.original)
        failure_bound = self.save(self.failure_path, dict(training_samples=0, optimizer_steps=0,
            attempts=[dict(sampled_choices=0), dict(status='not_started')]))
        cleanup_bound = self.save(self.original_path.parent/'controller-result.json',
            dict(controller_cleanup_verified=True, cleanup_incomplete=[]))
        self.progress = dict(updates=92, rollouts=368, decisions=21062)
        self.previous = {
            'session': dict(self.original, attempt=2),
            'failure': dict(training_samples=0, optimizer_steps=0, optimizer_updates=0,
                training_performed=False, update_status='not_started',
                attempts=[dict(status='failed', sampled_choices=39, training_samples=0, optimizer_steps=0)]
                    +[dict(status='not_started', training_samples=0, optimizer_steps=0) for _ in range(3)],
                parent=dict(progress=self.progress, optimizer_steps=783)),
        }
        session_bound = self.save(self.root/'previous/session.json', self.previous['session'])
        self.previous.update(
            controller_result=dict(session=session_bound, controller_cleanup_verified=True, cleanup_incomplete=[]),
            campaign_report=dict(session=session_bound, status='stopped_failure', latest_verified_progress=self.progress))
        self.repair = dict(attempt=3, original_session=original_bound, failure=failure_bound,
            controller_result=cleanup_bound, absolute_deadline_unix=8200)
        self.release = dict(repair=self.repair, source={'synthetic': True})
        self.release_path = self.root/'release.json'
        self.session_path = self.root/'repair3/campaign-session.json'
        self.pins = {}
        self.bind_previous()
        stack = ExitStack()
        self.addCleanup(stack.close)
        for attribute, value in dict(DURABLE_ROOT=self.root, ORIGINAL_SESSION=self.original_path,
            ORIGINAL_SESSION_SHA=original_bound['sha256'], ORIGINAL_FAILURE_SHA=failure_bound['sha256'],
            ATTEMPT2_PINS=self.pins, REPAIR='attempt3', SESSION=self.session_path).items():
            stack.enter_context(patch.object(scope, attribute, value))
        stack.enter_context(patch.object(scope, 'durable_path', side_effect=Path))
        stack.enter_context(patch.object(scope, 'validate_release', return_value=self.release))
        stack.enter_context(patch.object(scope.time, 'time', return_value=2000))
        stack.enter_context(patch.object(scope.time, 'monotonic', return_value=1100))

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return binding(path)

    def bind_previous(self):
        bound = {key: self.save(self.root/f'previous/{key}.json', value)
                 for key, value in self.previous.items()}
        self.pins.update({key: (str(Path(value['path']).relative_to(self.root)), value['sha256'])
                          for key, value in bound.items()})
        self.repair['previous_attempt'] = bound
        self.save(self.release_path, self.release)

    def test_fresh_single_use_attempt_retains_all_original_clocks(self):
        session = scope.create_session(self.release_path)
        self.assertEqual(session['attempt'], 3)
        self.assertEqual(session['attempt_started_unix'], 2000)
        for key, value in self.original.items():
            self.assertEqual(session[key], value)
        scope.session_for(self.session_path)
        with self.assertRaisesRegex(ValueError, 'cannot restart'):
            scope.create_session(self.release_path)
        with patch.object(scope.time, 'time', return_value=7900):
            with self.assertRaisesRegex(ValueError, 'expired'):
                scope.repair_origin(self.release)
        with patch.object(scope.time, 'monotonic', return_value=7000):
            with self.assertRaisesRegex(ValueError, 'expired'):
                scope.repair_origin(self.release)
        self.assertEqual(json.loads(self.original_path.read_text()), self.original)

    def test_all_prior_evidence_required_and_immutable(self):
        del self.repair['previous_attempt']['campaign_report']
        with self.assertRaisesRegex(ValueError, 'complete prior'):
            scope.repair_origin(self.release)
        self.bind_previous()
        (self.root/'previous/failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            scope.repair_origin(self.release)

    def test_prior_admitted_experience_cannot_be_discarded_by_repair(self):
        self.previous['failure']['training_samples'] = 39
        self.bind_previous()
        with self.assertRaisesRegex(ValueError, 'unadmitted failed batch'):
            scope.repair_origin(self.release)

    def test_prior_cleanup_and_original_budget_required(self):
        self.previous['controller_result']['cleanup_incomplete'] = ['native-group']
        self.bind_previous()
        with self.assertRaisesRegex(ValueError, 'completely stopped'):
            scope.repair_origin(self.release)
        self.previous['controller_result']['cleanup_incomplete'] = []
        self.previous['session']['started_monotonic'] += 1
        self.bind_previous()
        with self.assertRaisesRegex(ValueError, 'reset the original budget'):
            scope.repair_origin(self.release)

    def test_release_attempt_must_match_explicit_mode(self):
        self.repair['attempt'] = 2
        with self.assertRaisesRegex(ValueError, 'attempt3 release required'):
            scope.repair_origin(self.release)


if __name__ == '__main__':unittest.main()
