"""Synthetic repair scope checks; no engine or optimizer is launched."""
import json
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


if __name__ == '__main__':unittest.main()
