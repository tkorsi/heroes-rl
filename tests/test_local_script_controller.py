"""Offline controller tests. No native game or real PPO update is launched."""
import importlib.util
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
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
        for hours in (0, -1, float('nan'), float('inf'), 49):
            with self.subTest(hours=hours), self.assertRaises(ValueError): c.budget(hours)

    def test_48_hour_cap_retains_five_minute_finalization_reserve(self):
        session = c.budget(48)
        self.assertEqual(session['absolute_deadline_unix']-session['started_unix'], 172800)
        self.assertAlmostEqual(session['hard_deadline_monotonic']-session['started_monotonic'], 172800)
        self.assertEqual(session['absolute_deadline_unix']-session['compute_deadline_unix'], 300)

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


class CampaignOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.easy = dict(artifact_root=str(self.root), run_root=str(self.root/'local-script-training'))
        self.original = dict(artifact_root=str(self.root), run_root=str(self.root/'original-training'))

    def second_process(self, config):
        code = ('import json,sys\nfrom tools.run_local_training import campaign_locks\n'
                'try:\n with campaign_locks(json.loads(sys.argv[1])): pass\n'
                'except RuntimeError: sys.exit(72)\n')
        return subprocess.run([sys.executable, '-c', code, json.dumps(config)],
                              cwd=ROOT, capture_output=True, text=True, timeout=10)

    def test_running_easy_profile_blocks_original_in_a_separate_process(self):
        with c.campaign_locks(self.easy):
            blocked = self.second_process(self.original)
        self.assertEqual(blocked.returncode, 72, blocked.stderr)
        allowed = self.second_process(self.original)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_original_profile_blocks_easy_and_another_original(self):
        with c.campaign_locks(self.original):
            for other in (self.easy, self.original):
                result = self.second_process(other)
                self.assertEqual(result.returncode, 72, result.stderr)

    def test_failed_profile_lock_releases_the_shared_lock(self):
        with c.exclusive_lock(Path(self.original['run_root'])/'owner.lock'):
            with self.assertRaisesRegex(RuntimeError, 'owns'):
                with c.campaign_locks(self.original): pass
            # The second acquisition failed; ExitStack must release the first one.
            result = self.second_process(self.easy)
            self.assertEqual(result.returncode, 0, result.stderr)


class FinalParentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/'finished-easy-run'; self.run.mkdir()
        self.parent = {'policy': {'path': 'synthetic-final-policy'}, 'resume': {'path': 'synthetic-final-resume'}}
        self.session = dict(run_dir=str(self.run))
        self.report = dict(status='finished', latest_parent=self.parent, latest_progress={'updates': 600},
            owned_cleanup_verified=True, initial_state_unchanged=True,
            full_final_restore={'exact': True, 'progress': {'updates': 600}})
        self.shutdown = dict(owned_cleanup_verified=True, worker_returncode=0, within_hard_deadline=True)
        self.write()

    def write(self):
        for name, value in [('session.json', self.session), ('campaign-report.json', self.report),
                            ('shutdown.json', self.shutdown)]:
            c.atomic(self.run/name, value)

    def test_uses_final_report_and_ignores_stale_or_moving_latest_pointer(self):
        for pointer in ({'parent': {'checkpoint': 412}}, {'parent': {'checkpoint': 700}}):
            c.atomic(self.root/'latest-checkpoint.json', pointer)
            self.assertEqual(c.final_parent(self.run), self.parent)
            self.assertEqual(c.read(self.root/'latest-checkpoint.json'), pointer)

    def test_clean_requested_stop_is_a_finalized_parent(self):
        self.report['status'] = 'stopped_requested'; self.write()
        self.assertEqual(c.final_parent(self.run), self.parent)

    def test_active_or_failed_parent_cannot_use_a_stale_successful_shutdown(self):
        for status in ('running', 'stopped_failure', 'interrupted'):
            self.report['status'] = status; self.write()
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'finalized'):
                c.final_parent(self.run)

    def test_requires_complete_cleanup_and_restore_proofs(self):
        report = deepcopy(self.report); shutdown = deepcopy(self.shutdown)
        cases = [('report', 'owned_cleanup_verified'), ('report', 'initial_state_unchanged'),
                 ('report', 'full_final_restore'), ('shutdown', 'owned_cleanup_verified'),
                 ('shutdown', 'within_hard_deadline')]
        for area, key in cases:
            for invalid in (False, None, 'missing'):
                self.report, self.shutdown = deepcopy(report), deepcopy(shutdown)
                target = self.report if area == 'report' else self.shutdown
                if invalid == 'missing': target.pop(key)
                elif key == 'full_final_restore': target[key] = {'exact': invalid}
                else: target[key] = invalid
                self.write()
                with self.subTest(area=area, key=key, invalid=invalid), self.assertRaisesRegex(ValueError, 'finalized'):
                    c.final_parent(self.run)

    def test_missing_shutdown_or_nonzero_worker_exit_never_falls_back_to_latest(self):
        c.atomic(self.root/'latest-checkpoint.json', {'parent': self.parent})
        (self.run/'shutdown.json').unlink()
        with self.assertRaises(FileNotFoundError): c.final_parent(self.run)
        self.shutdown['worker_returncode'] = 1; self.write()
        with self.assertRaisesRegex(ValueError, 'finalized'): c.final_parent(self.run)

    def test_session_must_bind_the_requested_campaign_directory(self):
        self.session['run_dir'] = str(self.root/'another-run'); self.write()
        with self.assertRaisesRegex(ValueError, 'path differs'): c.final_parent(self.run)


class CountArgumentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root/'machine.json'
        c.atomic(self.config, {'artifact_root': str(self.root), 'default_profile': 'original'})

    def arguments(self, *extra):
        return ['run_local_training.py', '--config', str(self.config), '--profile', 'original', *extra]

    def test_count_mode_defaults_to_48_hours_and_100_update_evaluation(self):
        argv = self.arguments('--games', '5000', '--parent-final', str(self.root/'previous-run'))
        with patch.object(c.sys, 'argv', argv), patch.dict(c.os.environ), patch.object(c, 'start', return_value=0) as start:
            self.assertEqual(c.main(), 0)
        args = start.call_args.args[0]
        self.assertEqual((args.games, args.hours, args.eval_every), (5000, 48, 100))
        self.assertEqual(args.parent_final, self.root/'previous-run')
        self.assertEqual(args.profile, 'original')

    def test_legacy_time_mode_and_explicit_count_caps_remain_distinct(self):
        for extra, expected in [((), (2, 5)), (('--games', '5000', '--hours', '48', '--eval-every', '50'), (48, 50))]:
            with self.subTest(extra=extra), patch.object(c.sys, 'argv', self.arguments(*extra)), \
                    patch.dict(c.os.environ), patch.object(c, 'start', return_value=0) as start:
                self.assertEqual(c.main(), 0)
                args = start.call_args.args[0]
                self.assertEqual((args.hours, args.eval_every), expected)

    def test_invalid_count_or_conflicting_controls_fail_before_start(self):
        cases = [('--games', '0'), ('--games', '-4'), ('--games', '4999'), ('--games', '4.0'),
                 ('--games', '5000', '--max-updates', '1250'), ('--eval-every', '0'),
                 ('--parent', 'a', '--parent-final', 'b')]
        for extra in cases:
            with self.subTest(extra=extra), patch.object(c.sys, 'argv', self.arguments(*extra)), \
                    patch.object(c, 'start') as start, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped: c.main()
                self.assertEqual(stopped.exception.code, 2)
                start.assert_not_called()

    def test_count_mode_cannot_exceed_48_hour_safety_limit(self):
        with patch.object(c.sys, 'argv', self.arguments('--games', '5000', '--hours', '49')), \
                patch.dict(c.os.environ), patch.object(c, 'start') as start:
            with self.assertRaisesRegex(ValueError, '48'): c.main()
            start.assert_not_called()


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
