"""Offline controller tests. No native game or real PPO update is launched."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import unittest
from unittest.mock import Mock, patch
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_controller', ROOT/'tools/run_local_training.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    def test_both_clocks_and_reserve(self):
        with patch.object(c.time, 'time', return_value=1000.), patch.object(c.time, 'monotonic', return_value=30.):
            session = c.budget(2)
        self.assertEqual(session['compute_deadline_unix'], 7900)
        self.assertEqual(session['hard_deadline_monotonic'], 7230)
        with patch.object(c.time, 'time', return_value=8000.), patch.object(c.time, 'monotonic', return_value=30.):
            self.assertLess(c.left(session), 0)
        with patch.object(c.time, 'time', return_value=1.), patch.object(c.time, 'monotonic', return_value=8000.):
            self.assertLess(c.left(session), 0)

    def test_invalid_budgets(self):
        for hours in (0, -1, float('nan'), float('inf'), 25):
            with self.subTest(hours=hours), self.assertRaises(ValueError): c.budget(hours)

    def test_failed_attempt_consumes_sequence_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(c.reserve_sequence(tmp), 266)
            self.assertEqual(c.reserve_sequence(tmp), 267)
            self.assertEqual(c.read(Path(tmp)/'sequence-ledger.json')['next_sequence'], 268)

    def test_lock_blocks_second_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'owner.lock'
            code = ('import fcntl,sys; f=open(sys.argv[1],"a+"); '
                    'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
            with c.exclusive_lock(path):
                result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual(subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True).returncode, 0)

    def test_no_partial_or_unverified_batch(self):
        def report():
            return dict(status='finished', worker_complete=True, supervisor_cleanup_verified=True,
                        within_shared_budget=True, attempts=[dict(status='passed', completion_receipt=True,
                        cleanup_verified=True, identity_verified=True, replay_verified=True) for _ in range(4)])
        self.assertTrue(c.verified_batch(report(), 0))
        r=report(); r['attempts'].pop(); self.assertFalse(c.verified_batch(r, 0))
        r=report(); r['attempts'][2]['replay_verified']=False; self.assertFalse(c.verified_batch(r, 0))
        self.assertFalse(c.verified_batch(report(), -11))

    def test_summary_never_admits_failed_or_evaluation_games(self):
        value = dict(status='stopped_failure', latest_parent={}, batches=[
            dict(mode='training', status='passed', checkpoint_verified=True, outcomes=['win']*4),
            dict(mode='training', status='failed', outcomes=['win']*3),
            dict(mode='baseline', status='passed', outcomes=['loss']*4)])
        summary = c.summarize(value)
        self.assertEqual(summary['new_ppo_updates'], 1)
        self.assertEqual(summary['admitted_training_games'], 4)
        self.assertEqual(summary['training_outcomes'], {'win':4})
        self.assertEqual(summary['evaluation_outcomes'], {'loss':4})
        self.assertEqual(len(summary['failed_batches']), 1)

    def test_pair_budget_reserves_both_seats(self):
        with patch.object(c, 'left', return_value=250):
            self.assertTrue(c.enough({}, [60], gamesets=1))
            self.assertFalse(c.enough({}, [60], gamesets=2))

    def test_worker_baseline_failure_never_calls_training(self):
        from experiments.learner import local_script_training as adapter
        import experiments.learner.builtin_training_resume as resume
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parent={'policy':{'path':'placeholder'}}
            session=dict(run_dir=str(root),run_root=str(root),initial_parent=parent,supervisor_pid=1,next_sequence=266)
            c.atomic(root/'session.json',session)
            state=SimpleNamespace(progress=SimpleNamespace(updates=412,rollouts=1648,decisions=95836),optimizer_steps=3545)
            modes=[]
            def fail(session_path,parent,seq,seat,mode,value):
                modes.append(mode)
                raise RuntimeError('Native acceptance failed')
            with patch.object(c.os,'getppid',return_value=1), patch.object(adapter,'restore_parent',return_value=state), \
                    patch.object(resume,'binding',return_value={'path':'placeholder'}), patch.object(c,'execute_batch',side_effect=fail):
                self.assertEqual(c.worker(root/'session.json'),1)
            self.assertEqual(modes,['baseline'])
            summary=c.read(root/'summary.json')
            self.assertEqual(summary['new_ppo_updates'],0)
            self.assertEqual(summary['admitted_training_games'],0)
            self.assertEqual(summary['status'],'stopped_failure')
            self.assertFalse((root/'latest-checkpoint.json').exists())


class BoundedCleanupTests(unittest.TestCase):
    def test_all_evidence_checks_share_the_original_deadline(self):
        from experiments.headless import process_deadline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(3): (root/f'native-deadline-{n}.jsonl').touch()
            session = dict(absolute_deadline_unix=1002., hard_deadline_monotonic=52.)
            elapsed = [0.]
            calls = []
            def close(process, **kwargs):
                calls.append(('close', kwargs)); elapsed[0] += 1.75
            def evidence(path, **kwargs):
                calls.append(('evidence', kwargs)); elapsed[0] += .25
            with patch.object(c.time, 'time', side_effect=lambda: 1000.+elapsed[0]), \
                    patch.object(c.time, 'monotonic', side_effect=lambda: 50.+elapsed[0]), \
                    patch.object(process_deadline, 'close_owned_process', side_effect=close), \
                    patch.object(process_deadline, 'cleanup_from_evidence', side_effect=evidence):
                receipt = c.cleanup_owned(session, object(), root)
            self.assertEqual([kind for kind, _ in calls], ['close', 'evidence'])
            self.assertEqual(calls[0][1]['timeout'], 2)
            self.assertEqual(calls[1][1]['timeout'], .25)
            for _, kwargs in calls:
                self.assertEqual(kwargs['deadline_monotonic'], 52.)
                self.assertEqual(kwargs['deadline_wall'], 1002.)
            self.assertFalse(receipt['owned_cleanup_verified'])
            self.assertTrue(receipt['hard_deadline_exhausted'])
            self.assertIn('remaining_deadline_evidence', [e['check'] for e in receipt['errors']])

    def test_exhaustion_still_releases_the_owned_worker(self):
        from experiments.headless import process_deadline
        session = dict(absolute_deadline_unix=1000., hard_deadline_monotonic=50.)
        with patch.object(c, 'left', return_value=-1), \
                patch.object(process_deadline, 'close_owned_process', side_effect=TimeoutError('closure unverified')) as close:
            receipt = c.cleanup_owned(session, object())
        self.assertEqual(close.call_args.kwargs['timeout'], 0)
        self.assertEqual(close.call_args.kwargs['deadline_wall'], 1000.)
        self.assertFalse(receipt['owned_cleanup_verified'])

    def test_caffeinate_wait_cannot_start_an_extra_shutdown_window(self):
        awake = Mock()
        awake.poll.return_value = None
        remaining = [.2]
        def wait(timeout):
            self.assertLessEqual(timeout, remaining[0])
            remaining[0] = 0
            raise subprocess.TimeoutExpired('caffeinate', timeout)
        awake.wait.side_effect = wait
        session = dict(absolute_deadline_unix=1000., hard_deadline_monotonic=50.)
        with patch.object(c, 'left', side_effect=lambda *a, **k: remaining[0]):
            receipt = c.cleanup_owned(session, awake=awake)
        awake.terminate.assert_called_once()
        awake.kill.assert_called_once()
        self.assertEqual([call.kwargs['timeout'] for call in awake.wait.call_args_list], [.2, 0])
        self.assertFalse(receipt['owned_cleanup_verified'])

    def test_one_cleanup_failure_does_not_skip_remaining_owned_evidence(self):
        from experiments.headless import process_deadline
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'worker-deadline.jsonl'; path.touch()
            with patch.object(process_deadline, 'close_owned_process', side_effect=RuntimeError('live worker')), \
                    patch.object(process_deadline, 'cleanup_from_evidence') as evidence:
                receipt = c.cleanup_owned(c.budget(1), object(), tmp)
            evidence.assert_called_once()
            self.assertFalse(receipt['owned_cleanup_verified'])


class SupervisorFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.session = dict(run_dir=str(self.run), run_root=str(self.root), initial_parent={}, **c.budget(1))
        self.path = self.run/'session.json'; c.atomic(self.path, self.session)

    def test_launch_failure_is_reported_without_checkpoint_admission(self):
        from experiments.headless import process_deadline
        with patch.object(process_deadline, 'guarded_popen', side_effect=OSError('synthetic spawn failure')):
            self.assertEqual(c.supervise(self.path), 1)
        shutdown = c.read(self.run/'shutdown.json')
        summary = c.read(self.run/'summary.json')
        self.assertIsNone(shutdown['worker_returncode'])
        self.assertIn('synthetic spawn failure', shutdown['supervisor_error'])
        self.assertEqual(summary['status'], 'stopped_failure')
        self.assertEqual(summary['admitted_training_games'], 0)
        self.assertFalse((self.root/'latest-checkpoint.json').exists())

    def test_failed_cleanup_overrides_an_existing_finished_report(self):
        from experiments.headless import process_deadline
        c.atomic(self.run/'campaign-report.json', dict(status='finished', batches=[], latest_parent={}))
        (self.run/'RESULTS.md').write_text('Worker completion before supervisor cleanup')
        checkpoint = {'parent': {'unchanged': True}}
        c.atomic(self.root/'latest-checkpoint.json', checkpoint)
        child = Mock(); child.wait.return_value = 0
        with patch.object(c.sys, 'platform', 'linux'), \
                patch.object(process_deadline, 'guarded_popen', return_value=child), \
                patch.object(process_deadline, 'close_owned_process', side_effect=RuntimeError('synthetic still-live group')):
            self.assertEqual(c.supervise(self.path), 1)
        shutdown = c.read(self.run/'shutdown.json')
        self.assertFalse(shutdown['owned_cleanup_verified'])
        self.assertEqual(c.read(self.run/'summary.json')['status'], 'stopped_failure')
        self.assertIn('synthetic still-live group', c.read(self.run/'summary.json')['error'])
        self.assertEqual(c.read(self.root/'latest-checkpoint.json'), checkpoint)

    def test_worker_wait_exception_retains_failure_and_restores_signal_handlers(self):
        from experiments.headless import process_deadline
        child = Mock(); child.wait.side_effect = subprocess.TimeoutExpired('worker', 1)
        previous = {sig: c.signal.getsignal(sig) for sig in (c.signal.SIGINT, c.signal.SIGTERM)}
        with patch.object(c.sys, 'platform', 'linux'), \
                patch.object(process_deadline, 'guarded_popen', return_value=child), \
                patch.object(process_deadline, 'close_owned_process'):
            self.assertEqual(c.supervise(self.path), 1)
        self.assertLessEqual(child.wait.call_args.kwargs['timeout'], 3600)
        self.assertIn('TimeoutExpired', c.read(self.run/'shutdown.json')['supervisor_error'])
        self.assertEqual(previous, {sig: c.signal.getsignal(sig) for sig in previous})

    def test_late_supervisor_completion_is_not_reported_as_success(self):
        from experiments.headless import process_deadline
        c.atomic(self.run/'campaign-report.json', dict(status='finished', batches=[], latest_parent={}))
        (self.run/'RESULTS.md').write_text('Worker finished')
        child = Mock(); child.wait.return_value = 0
        receipt = dict(owned_cleanup_verified=True, errors=[])
        with patch.object(c.sys, 'platform', 'linux'), \
                patch.object(process_deadline, 'guarded_popen', return_value=child), \
                patch.object(c, 'cleanup_owned', return_value=receipt), \
                patch.object(c, 'left', return_value=-.1):
            self.assertEqual(c.supervise(self.path), 1)
        self.assertFalse(c.read(self.run/'shutdown.json')['within_hard_deadline'])
        self.assertEqual(c.read(self.run/'summary.json')['status'], 'stopped_failure')
        self.assertIn('original hard deadline', c.read(self.run/'summary.json')['error'])
