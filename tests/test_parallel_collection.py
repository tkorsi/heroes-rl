"""Parallel collection orchestration using harmless subprocesses, never VCMI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.learner import builtin_training as runner
from experiments.learner import builtin_gameplay as gameplay
from experiments.headless import process_deadline

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and hasattr(os, 'fork'), 'POSIX collection guardians')
class ParallelSupervisorTests(unittest.TestCase):
    def setUp(self):
        from experiments.learner import parallel_collection
        self.parallel = parallel_collection
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.processes = []
        self.addCleanup(self.close_children)

    def close_children(self):
        for process in self.processes:
            process_deadline.close_owned_process(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def jobs(self, count=8):
        return [self.parallel._Job(index, f'{index:064x}') for index in range(count)]

    def budget(self, seconds=12):
        return runner.pilot.Budget(time.monotonic()+seconds, time.time()+seconds)

    def launch(self, job, code, budget):
        job.process = process_deadline.guarded_popen([sys.executable, '-c', code],
            deadline_monotonic=budget.deadline, deadline_wall=budget.wall_deadline,
            evidence_path=self.root/f'worker-{job.index}.jsonl', cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.processes.append(job.process)
        job.process.stdin.close()

    def assert_closed(self, jobs):
        for job in jobs:
            if job.process is not None:
                self.assertIsNotNone(job.process.poll())
                self.assertIsNone(job.process._deadline_owner_fd)
                self.assertFalse(process_deadline.group_live_members(job.process.pid))

    def test_eight_workers_finish_out_of_order_but_results_follow_schedule(self):
        jobs, budget = self.jobs(), self.budget()
        finished = []

        def launch(job):
            code = f'''
import pathlib, time
root = pathlib.Path({str(self.root)!r})
(root / '{job.index}.ready').touch()
while len(list(root.glob('*.ready'))) != 8:
    time.sleep(.01)
time.sleep({.075*(7-job.index)!r})
print({job.index}, flush=True)
'''
            self.launch(job, code, budget)

        def finish(job):
            value = int(job.process.stdout.read())
            finished.append(value)
            return value

        result = self.parallel._supervise(jobs, workers=8, budget=budget, launch=launch, finish=finish)
        self.assertEqual(result, tuple(range(8)))
        self.assertEqual(len(finished), 8)
        self.assertEqual(finished[0], 7)
        self.assertNotEqual(finished, list(range(8)))
        self.assert_closed(jobs)

    def test_serial_mode_never_launches_next_slot_before_prior_completion(self):
        jobs, budget = self.jobs(3), self.budget()
        events = []

        def launch(job):
            events.append(('launch', job.index))
            self.launch(job, f'print({job.index}, flush=True)', budget)

        def finish(job):
            events.append(('finish', job.index))
            return int(job.process.stdout.read())

        result = self.parallel._supervise(jobs, workers=1, budget=budget, launch=launch, finish=finish)
        self.assertEqual(result, (0, 1, 2))
        self.assertEqual(events, [(kind, index) for index in range(3) for kind in ('launch', 'finish')])
        self.assert_closed(jobs)

    def test_worker_failure_cancels_sibling_and_native_descendant_without_touching_other_group(self):
        jobs, budget = self.jobs(3), self.budget()
        native_evidence = self.root/'native-deadline.jsonl'
        ready = self.root/'native-ready'
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                     start_new_session=True)
        try:
            def launch(job):
                if job.index == 0:
                    native = f'from pathlib import Path;import time;Path({str(ready)!r}).touch();time.sleep(30)'
                    code = f'''
import subprocess, sys, time
from experiments.headless.process_deadline import guarded_popen
child = guarded_popen([sys.executable, '-c', {native!r}],
    deadline_monotonic={budget.deadline!r}, deadline_wall={budget.wall_deadline!r},
    evidence_path={str(native_evidence)!r}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(30)
'''
                else:
                    code = f'''
from pathlib import Path
import time
while not Path({str(ready)!r}).exists():
    time.sleep(.01)
raise SystemExit(7)
'''
                self.launch(job, code, budget)

            with self.assertRaisesRegex(ValueError, 'status 7'):
                self.parallel._supervise(jobs, workers=2, budget=budget, launch=launch,
                                         finish=lambda job: self.fail('Failed workers cannot finish'))
            self.assertIsNone(jobs[2].process)
            self.assert_closed(jobs)
            process_deadline.cleanup_from_evidence(native_evidence, timeout=5)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_partial_launch_failure_releases_all_started_processes(self):
        jobs, budget = self.jobs(3), self.budget()

        def launch(job):
            self.launch(job, 'import time;time.sleep(30)', budget)
            if job.index == 1:
                raise OSError('synthetic error after Popen')

        with self.assertRaisesRegex(OSError, 'after Popen'):
            self.parallel._supervise(jobs, workers=8, budget=budget, launch=launch,
                                     finish=lambda job: None)
        self.assertIsNone(jobs[2].process)
        self.assert_closed(jobs)

    def test_completion_validation_failure_stops_remaining_workers(self):
        jobs, budget = self.jobs(2), self.budget()

        def launch(job):
            self.launch(job, 'print("done")' if job.index == 0 else 'import time;time.sleep(30)', budget)

        def finish(job):
            raise ValueError('synthetic wrong completion token')

        with self.assertRaisesRegex(ValueError, 'wrong completion token'):
            self.parallel._supervise(jobs, workers=8, budget=budget, launch=launch, finish=finish)
        self.assert_closed(jobs)

    def test_deadline_stops_all_workers_and_keeps_pending_slots_unstarted(self):
        jobs, budget = self.jobs(3), self.budget(.8)

        def launch(job):
            self.launch(job, 'import time;time.sleep(30)', budget)

        started = time.monotonic()
        with self.assertRaises((TimeoutError, ValueError)):
            self.parallel._supervise(jobs, workers=2, budget=budget, launch=launch,
                finish=lambda job: self.fail('Timed-out worker returned a result'), cleanup_reserve=.4)
        self.assertLess(time.monotonic()-started, 5)
        self.assertIsNone(jobs[2].process)
        self.assert_closed(jobs)

    def test_expired_hard_budget_releases_owners_without_claiming_verified_cleanup(self):
        jobs, budget = self.jobs(2), self.budget(.3)

        def launch(job):
            self.launch(job, 'import time;time.sleep(30)', budget)

        with self.assertRaises((TimeoutError, ValueError)) as caught:
            self.parallel._supervise(jobs, workers=2, budget=budget, launch=launch,
                                     finish=lambda job: None)
        self.assertTrue(all(job.process._deadline_owner_fd is None for job in jobs))
        self.assertTrue(any('cleanup failures' in note for note in getattr(caught.exception, '__notes__', ())))
        # Observation by this test is separate from the exhausted coordinator
        # budget: guardians still have to stop every released process group.
        for job in jobs:
            job.process.wait(timeout=5)
            process_deadline.cleanup_from_evidence(self.root/f'worker-{job.index}.jsonl', timeout=5)
        self.assert_closed(jobs)


class ParallelRunnerDispatchTests(unittest.TestCase):
    def test_four_worker_execution_dispatch_keeps_configuration_and_schedule(self):
        from experiments.learner import parallel_collection
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cfg = runner.configuration()
            rows = [{**row, 'phase': 'evaluation'} for row in runner.schedule(1)]
            plan = dict(schema='heroes-builtin-checkpoint-evaluation-v1-plan', configuration=cfg,
                execution={'collection_workers': 4}, execution_output=str(output), parent={},
                mode='evaluation', schedule=rows, runtime={}, native={},
                compute_deadline_unix=time.time()+3600)
            report = dict(attempts=[runner.pilot.empty_attempt(row) for row in rows],
                update_status='not_started', optimizer_updates=0, optimizer_steps=0,
                training_performed=False)
            (output/'report.json').write_text(json.dumps(report))
            args = SimpleNamespace(plan=output/'plan.json', sha256='a'*64,
                started=time.monotonic(), wall_started=time.time())
            state = SimpleNamespace(model=object(), checkpoint='synthetic')
            attempts = tuple({**row, 'status': 'passed', 'identity_verified': True,
                              'cleanup_verified': True} for row in rows)
            collected = SimpleNamespace(attempts=attempts, receipts=(), collection_seconds=1., verification_seconds=1.)
            with patch.object(runner.pilot, 'check_identity'), \
                    patch.object(parallel_collection, 'collect_batch', return_value=collected) as parallel, \
                    patch.object(runner.pilot, 'run_attempt') as serial, \
                    patch.object(gameplay, 'native_update') as update:
                result = runner.worker(args, plan_reader=lambda *args: plan, state_reader=lambda plan: state)
            self.assertEqual(parallel.call_args.kwargs['workers'], 4)
            self.assertEqual(result['status'], 'finished')
            self.assertEqual(result['parallel_collection']['games'], 4)
            self.assertEqual(plan['configuration'], runner.configuration())
            self.assertNotIn('parallel_collection', cfg)
            self.assertEqual(cfg['max_optimizer_steps'], 640)
            serial.assert_not_called()
            update.assert_not_called()

    def test_partial_or_unverified_parallel_return_never_reaches_ppo(self):
        from experiments.learner import parallel_collection
        cases = ('missing_game', 'failed_game', 'identity_unchecked', 'cleanup_unchecked', 'missing_receipt')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                cfg = runner.configuration()
                cfg['parallel_collection'] = {'batch_size': 8, 'workers': 8}
                rows = [{**runner.schedule(1)[i % 4], 'episode': i+1} for i in range(8)]
                plan = dict(schema='heroes-builtin-training-v1-plan', configuration=cfg, execution_output=str(output), parent={},
                    mode='training', schedule=rows, runtime={}, native={},
                    compute_deadline_unix=time.time()+3600)
                report = dict(attempts=[runner.pilot.empty_attempt(row) for row in rows],
                    update_status='not_started', optimizer_updates=0, optimizer_steps=0,
                    training_performed=False)
                (output/'report.json').write_text(json.dumps(report))
                args = SimpleNamespace(plan=output/'plan.json', sha256='a'*64,
                    started=time.monotonic(), wall_started=time.time())
                state = SimpleNamespace(model=object(), checkpoint='synthetic')
                attempts = [{**row, 'status': 'passed', 'identity_verified': True,
                             'cleanup_verified': True} for row in rows]
                receipts = [object() for _ in rows]
                if case == 'missing_game':
                    attempts.pop()
                elif case == 'failed_game':
                    attempts[3]['status'] = 'failed'
                elif case == 'identity_unchecked':
                    del attempts[3]['identity_verified']
                elif case == 'cleanup_unchecked':
                    attempts[3]['cleanup_verified'] = False
                else:
                    receipts.pop()
                collected = SimpleNamespace(attempts=tuple(attempts), receipts=tuple(receipts),
                    collection_seconds=1., verification_seconds=1.)
                with patch.object(runner.pilot, 'check_identity'), \
                        patch.object(runner, 'require_heldout'), \
                        patch.object(parallel_collection, 'collect_batch', return_value=collected), \
                        patch.object(runner.pilot, 'run_attempt') as serial, \
                        patch.object(gameplay, 'native_update') as update:
                    result = runner.worker(args, plan_reader=lambda *args: plan,
                                           state_reader=lambda plan: state)
                update.assert_not_called()
                serial.assert_not_called()
                self.assertEqual(result['status'], 'stopped')
                self.assertEqual(result['update_status'], 'not_started')
                self.assertFalse(result['training_performed'])

    def test_parallel_failure_does_not_update_or_retry_through_serial_collector(self):
        from experiments.learner import parallel_collection
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cfg = runner.configuration()
            cfg['parallel_collection'] = {'batch_size': 8, 'workers': 8}
            rows = [{**runner.schedule(1)[i % 4], 'episode': i + 1} for i in range(8)]
            plan = dict(schema='heroes-builtin-training-v1-plan', configuration=cfg, execution_output=str(output), parent={},
                mode='training', schedule=rows, runtime={}, native={},
                compute_deadline_unix=time.time()+3600)
            report = dict(attempts=[runner.pilot.empty_attempt(row) for row in rows],
                update_status='not_started', optimizer_updates=0, optimizer_steps=0,
                training_performed=False)
            (output/'report.json').write_text(json.dumps(report))
            args = SimpleNamespace(plan=output/'plan.json', sha256='a'*64,
                                   started=time.monotonic(), wall_started=time.time())
            state = SimpleNamespace(model=object(), checkpoint='synthetic')

            def failed(path, sha, model, checkpoint, **kwargs):
                self.assertEqual(kwargs['workers'], 8)
                self.assertLessEqual(kwargs['deadline_wall'], plan['compute_deadline_unix'])
                kwargs['save'](3, {**report['attempts'][3], 'status': 'failed',
                                   'error': 'synthetic child failed'})
                raise RuntimeError('synthetic child failed')

            with patch.object(runner.pilot, 'check_identity'), \
                    patch.object(runner, 'require_heldout'), \
                    patch.object(parallel_collection, 'collect_batch', side_effect=failed) as parallel, \
                    patch.object(runner.pilot, 'run_attempt') as serial, \
                    patch.object(gameplay, 'native_update') as update:
                result = runner.worker(args, plan_reader=lambda *args: plan,
                                       state_reader=lambda plan: state)
            parallel.assert_called_once()
            serial.assert_not_called()
            update.assert_not_called()
            self.assertEqual(result['status'], 'stopped')
            self.assertIn('synthetic child failed', result['stop_reason'])
            self.assertEqual(result['attempts'][3]['status'], 'failed')
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertFalse(result['training_performed'])
            self.assertEqual(result['update_status'], 'not_started')


class ParallelChildPlanTests(unittest.TestCase):
    def test_child_activates_declared_support_and_uses_parent_checkpoint_metadata(self):
        from experiments.learner import interaction_guard, local_script_training as local, ppo
        from experiments.learner import parallel_collection as parallel
        from experiments.headless import background_check
        parent = {'source': {'torch': 'historical-version'}}
        report = {'plan_sha256': 'parent-plan-hash'}
        metadata = {'historical': 'exact-saved-checkpoint-metadata'}
        plan = {'scope': local.SCOPE, 'parent_descriptor': {'bound': 'parent'},
                'policy_support': interaction_guard.VERSION,
                'execution': local.execution_choice(overlapped_verification=False, prefix_helpers=0,
                                                     text_leaf_replay=True, replay_threads=2)}
        old = (interaction_guard.ACTIVE_VERSION, background_check.FORK, ppo.PREFIX_HELPERS,
               ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS, ppo.EXPERIMENTS)
        try:
            interaction_guard.ACTIVE_VERSION = None
            with patch.object(local, 'read_descriptor', return_value=(parent, report)) as read, \
                    patch('experiments.learner.builtin_training_resume._metadata', return_value=metadata) as bound:
                source, options = parallel._worker_checkpoint_context(plan)
            self.assertIs(source, metadata)
            self.assertEqual(options, {'historical_torch_version': 'historical-version'})
            read.assert_called_once_with(plan['parent_descriptor'])
            bound.assert_called_once_with(parent, report['plan_sha256'])
            self.assertEqual(interaction_guard.Tracker().version, interaction_guard.VERSION)
            self.assertEqual(ppo.TEXT_REPLAY_WORKERS, 2)
            self.assertFalse(background_check.FORK)
        finally:
            (interaction_guard.ACTIVE_VERSION, background_check.FORK, ppo.PREFIX_HELPERS,
             ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS, ppo.EXPERIMENTS) = old

    def test_manual_four_plan_is_checked_by_current_local_adapter(self):
        from experiments.learner import local_script_training as local, parallel_collection as parallel
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'plan.json'
            plan = {'schema': 'heroes-builtin-training-v1-plan', 'scope': local.SCOPE,
                    'configuration': {'allow_verified_training_prefix': False},
                    'execution': {'collection_workers': 4}, 'execution_output': str(path.parent/'execution'),
                    'schedule': runner.schedule(1)}
            path.write_text(json.dumps(plan))
            digest = runner.pilot.digest(path)
            with patch.object(local, 'read_plan', return_value=plan) as read:
                self.assertEqual(parallel._read_plan(path, digest), plan)
            read.assert_called_once_with(path.resolve(), digest)

    def test_failure_retains_original_error_and_secondary_evidence_errors(self):
        from experiments.learner import parallel_collection as parallel
        args = SimpleNamespace(sha256='plan-hash', index=2)
        budget = SimpleNamespace(remaining=lambda: 4)
        original = RuntimeError('native action rejected')
        def retain(env, collector, attempt, error, **kwargs):
            self.assertIs(error, original)
            attempt.update(episode_directory='/private/failed-episode',
                           failure_finalization_error='native process already exited',
                           partial_evidence_error='synthetic write failure')
        with patch('experiments.learner.builtin_rollout.retain_failed_builtin_evidence', side_effect=retain):
            failure = parallel._retain_worker_failure(args, object(), object(), original, budget)
        self.assertIn('native action rejected', failure['error'])
        self.assertEqual(failure['failure_finalization_error'], 'native process already exited')
        self.assertEqual(failure['partial_evidence_error'], 'synthetic write failure')
        self.assertFalse(failure['training_input'])
        self.assertFalse(failure['completion_receipt'])

    def test_expired_retention_budget_is_recorded_instead_of_silently_dropped(self):
        from experiments.learner import parallel_collection as parallel
        args = SimpleNamespace(sha256='plan-hash', index=0)
        budget = SimpleNamespace(remaining=lambda: (_ for _ in ()).throw(TimeoutError('expired')))
        failure = parallel._retain_worker_failure(args, None, None, RuntimeError('original'), budget)
        self.assertEqual(failure['error'], 'RuntimeError: original')
        self.assertEqual(failure['failure_retention_error'], 'TimeoutError: expired')


class ParallelFailedBatchTests(unittest.TestCase):
    def collect(self, root, *, supervisor_error=None):
        from contextlib import ExitStack
        from experiments.learner import parallel_collection as parallel
        # Import before replacing model identity: the verifier imports that
        # function by name and must not retain this test's temporary mock.
        from validation.probes import validate_hero_policy
        cfg = {'allow_verified_training_prefix': False, 'cleanup_reserve_seconds': 0}
        rows = [{**row, 'phase': 'evaluation'} for row in runner.schedule(1)]
        plan = {'schema': 'heroes-builtin-checkpoint-evaluation-v1-plan', 'configuration': cfg,
                'execution': {'collection_workers': 4}, 'execution_output': str(root),
                'schedule': rows, 'runtime': {}, 'data': '/synthetic/data',
                'replay_binary': {'path': '/synthetic/replay'}}
        replay_error = RuntimeError('synthetic replay rejection')
        def supervise(*args, **kwargs):
            if supervisor_error is not None:
                raise supervisor_error
            for row in rows:
                (root/f"episode-{row['episode']:06d}").mkdir()
        with ExitStack() as stack:
            stack.enter_context(patch.object(parallel, '_read_plan', return_value=plan))
            stack.enter_context(patch.object(parallel, '_check_identity'))
            stack.enter_context(patch('experiments.learner.contextual_model.weights_sha256', return_value='fixed'))
            stack.enter_context(patch.object(parallel, '_supervise', side_effect=supervise))
            stack.enter_context(patch.object(runner.pilot, 'native_opponent', return_value={}))
            stack.enter_context(patch('experiments.learner.builtin_rollout.audit_persisted_builtin_policy', return_value={}))
            stack.enter_context(patch('experiments.learner.builtin_behavior.analyze_episode', return_value={}))
            stack.enter_context(patch('experiments.headless.builtin_replay.verify_episode',
                                      side_effect=[{'verified': True}, replay_error]))
            with self.assertRaises(parallel.CollectionFailed) as caught:
                parallel.collect_batch(root/'plan.json', 'plan-hash', object(), 'checkpoint', workers=4,
                    deadline_monotonic=time.monotonic()+60, deadline_wall=time.time()+60)
        return json.loads((root/'parallel-collection/result.json').read_text()), caught.exception

    def test_later_rejection_revokes_earlier_completion_without_admitting_partial_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, error = self.collect(Path(temporary))
        self.assertEqual(result['attempts'][0]['status'], 'rejected')
        self.assertTrue(result['attempts'][0]['replay_verified'])
        self.assertEqual(result['attempts'][0]['reason'], 'batch_aborted')
        self.assertEqual(result['attempts'][0]['error'], 'RuntimeError: synthetic replay rejection')
        self.assertTrue(all(not item['completion_receipt'] for item in result['attempts']))
        self.assertTrue(all(item['batch_admission'] == 'rejected' for item in result['attempts']))
        self.assertEqual((result['optimizer_updates'], result['training_samples_admitted']), (0, 0))
        self.assertEqual(error.attempts, tuple(result['attempts']))

    def test_later_training_rejection_survives_worker_and_supervisor_report_reader(self):
        from contextlib import ExitStack
        from experiments.learner import parallel_collection as parallel
        from validation.probes import validate_hero_policy
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = runner.schedule(1)
            plan = dict(schema='heroes-builtin-training-v1-plan', mode='training',
                configuration=runner.configuration(), execution={'collection_workers': 4},
                execution_output=str(root), schedule=rows, runtime={}, native={}, source={},
                parent={}, checkpoint={'path': '/synthetic/checkpoint'}, data='/synthetic/data',
                replay_binary={'path': '/synthetic/replay'}, compute_deadline_unix=time.time()+60)
            report = dict(plan_sha256='a'*64, attempts=[runner.pilot.empty_attempt(row) for row in rows],
                update_status='not_started', optimizer_updates=0, optimizer_steps=0,
                training_samples=0, training_performed=False)
            runner.pilot.write(root/'report.json', report)
            args = SimpleNamespace(plan=root/'plan.json', sha256='a'*64,
                started=time.monotonic(), wall_started=time.time())
            state = SimpleNamespace(model=object(), checkpoint='synthetic')
            receipts = [object(), object()]
            for receipt in receipts:
                gameplay._RECEIPTS[id(receipt)] = receipt
                self.addCleanup(gameplay._RECEIPTS.pop, id(receipt), None)

            def supervise(*args, **kwargs):
                for row in rows:
                    (root/f"episode-{row['episode']:06d}").mkdir()

            with ExitStack() as stack:
                stack.enter_context(patch.object(parallel, '_read_plan', return_value=plan))
                stack.enter_context(patch.object(parallel, '_check_identity'))
                stack.enter_context(patch.object(runner.pilot, 'check_identity'))
                stack.enter_context(patch.object(runner.pilot, 'native_opponent', return_value={}))
                stack.enter_context(patch.object(parallel, '_supervise', side_effect=supervise))
                stack.enter_context(patch('experiments.learner.contextual_model.weights_sha256', return_value='fixed'))
                stack.enter_context(patch.object(gameplay.TrainingBinding, 'create', side_effect=lambda **kwargs: object()))
                stack.enter_context(patch.object(gameplay, '_begin_parallel_capture',
                    side_effect=lambda binding: SimpleNamespace(token='synthetic', consumed=True)))
                stack.enter_context(patch.object(gameplay, 'verify_training_capture',
                    side_effect=[*receipts, RuntimeError('synthetic terminal replay rejection')]))
                stack.enter_context(patch('experiments.learner.builtin_behavior.analyze_episode', return_value={}))
                update = stack.enter_context(patch.object(gameplay, 'native_update'))
                serial = stack.enter_context(patch.object(runner.pilot, 'run_attempt'))
                result = runner.worker(args, plan_reader=lambda *args: plan,
                    state_reader=lambda plan: state, heldout_checker=lambda *args: None)
                loaded = runner.pilot.read_worker_report(root/'worker-report.json', plan, args.sha256)
                update.assert_not_called()
                serial.assert_not_called()

            self.assertEqual(loaded, result)
            self.assertEqual(loaded['status'], 'stopped')
            self.assertTrue(loaded['worker_complete'])
            self.assertEqual(loaded['stop_reason'], 'CollectionFailed: synthetic terminal replay rejection')
            self.assertEqual([row['status'] for row in loaded['attempts'][:2]], ['rejected', 'rejected'])
            self.assertTrue(all(row['replay_verified'] for row in loaded['attempts'][:2]))
            self.assertTrue(all(row['completion_receipt'] is False
                                and row['batch_admission'] == 'rejected' for row in loaded['attempts']))
            self.assertEqual((loaded['optimizer_updates'], loaded['optimizer_steps'], loaded['training_samples']), (0, 0, 0))
            self.assertEqual(loaded['update_status'], 'not_started')
            self.assertFalse(loaded['training_performed'])
            self.assertTrue(all(id(receipt) not in gameplay._RECEIPTS for receipt in receipts))

    def test_uncertain_coordinator_cleanup_remains_visible_in_report_and_exception(self):
        failure = TimeoutError('collection deadline')
        failure.collector_cleanup_verified = False
        failure.add_note('Collector cleanup failures: shared deadline exhausted')
        with tempfile.TemporaryDirectory() as temporary:
            result, error = self.collect(Path(temporary), supervisor_error=failure)
        self.assertFalse(result['cleanup_verified'])
        self.assertEqual(result['exception_notes'], failure.__notes__)
        self.assertEqual(error.__notes__, failure.__notes__)


if __name__ == '__main__':
    unittest.main()
