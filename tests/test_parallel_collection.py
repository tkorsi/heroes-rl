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
    def test_partial_or_unverified_parallel_return_never_reaches_ppo(self):
        from experiments.learner import parallel_collection
        cases = ('missing_game', 'failed_game', 'identity_unchecked', 'cleanup_unchecked', 'missing_receipt')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                cfg = runner.configuration()
                cfg['parallel_collection'] = {'batch_size': 8, 'workers': 8}
                rows = [{**runner.schedule(1)[i % 4], 'episode': i+1} for i in range(8)]
                plan = dict(configuration=cfg, execution_output=str(output), parent={},
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
            plan = dict(configuration=cfg, execution_output=str(output), parent={},
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


if __name__ == '__main__':
    unittest.main()
