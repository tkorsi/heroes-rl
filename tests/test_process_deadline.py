"""Real harmless subprocess checks; no VCMI, games or training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from experiments.headless import process_deadline as d

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and hasattr(os, "fork"), "POSIX guardian")
class ProcessDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            d.close_owned_process(process)
        self.directory.cleanup()

    def start(self, code, *, seconds=5, name="guard.jsonl", **kwargs):
        process = d.guarded_popen([sys.executable, "-c", code],
            deadline_monotonic=time.monotonic() + seconds, deadline_wall=time.time() + seconds,
            evidence_path=self.root / name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        self.processes.append(process)
        return process

    def records(self, name="guard.jsonl"):
        path = self.root / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_for(self, predicate, seconds=3):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("Harmless subprocess did not reach expected condition")

    def test_exec_preserves_leader_pid_group_environment_and_exit_code(self):
        out = self.root / "child.json"
        code = "import json,os,pathlib;pathlib.Path('child.json').write_text(json.dumps([os.getpid(),os.getpgrp(),os.getenv('GUARD_TEST')]));raise SystemExit(7)"
        process = self.start(code, cwd=self.root, env={**os.environ, "GUARD_TEST": "preserved"})
        self.assertEqual(process.wait(timeout=3), 7)
        self.assertEqual(json.loads(out.read_text()), [process.pid, process.pid, "preserved"])
        self.wait_for(lambda: any(r['event'] == 'parent_exited' for r in self.records()))
        events = [r['event'] for r in self.records()]
        self.assertLess(events.index('guardian_armed'), events.index('exec_attempt'))
        self.assertFalse(d.group_live_members(process.pid))
        self.assertNotIn('deadline_expired', events)

    def test_deadline_kills_group_children_even_if_term_ignored_and_leaves_other_group(self):
        child_file = self.root / 'descendant'
        code = ("import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "child=os.fork();"
                f"pathlib.Path({str(child_file)!r}).write_text(str(os.getpid())) if child==0 else None;"
                "time.sleep(30)")
        other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        try:
            started = time.monotonic()
            process = self.start(code, seconds=0.6)
            self.wait_for(child_file.exists)
            self.assertEqual(process.wait(timeout=3), -signal.SIGKILL)
            self.assertLess(time.monotonic() - started, 2)
            self.wait_for(lambda: not d.group_live_members(process.pid))
            self.assertIsNone(other.poll())
            event = next(r for r in self.records() if r['event'] == 'deadline_expired')
            self.assertEqual(event['process_group'], process.pid)
            self.assertEqual(event['signal'], 'SIGKILL')
        finally:
            os.killpg(other.pid, signal.SIGKILL)
            other.wait(timeout=3)

    def test_expired_and_invalid_launches_do_not_create_a_process_or_evidence(self):
        for deadline in (time.monotonic() - 1, float('nan'), float('inf'), True):
            with self.subTest(deadline=deadline), patch.object(d.subprocess, 'Popen') as spawn:
                with self.assertRaises((ValueError, TimeoutError)):
                    d.guarded_popen([sys.executable, '-c', 'pass'], deadline_monotonic=deadline,
                                   deadline_wall=time.time(), evidence_path=self.root / 'absent')
                spawn.assert_not_called()
        self.assertFalse((self.root / 'absent').exists())

    def test_deadline_expiring_during_spawn_never_executes_target(self):
        target = self.root / 'must-not-exist'
        real_spawn = subprocess.Popen
        def delayed(*args, **kwargs):
            time.sleep(0.08)
            return real_spawn(*args, **kwargs)
        with patch.object(d.subprocess, 'Popen', side_effect=delayed):
            process = self.start(f"from pathlib import Path;Path({str(target)!r}).touch()", seconds=0.03)
        self.assertEqual(process.wait(timeout=3), 124)
        self.assertFalse(target.exists())
        self.assertIn('expired_before_exec', [r['event'] for r in self.records()])

    def test_expired_wall_clock_rejects_launch_despite_future_monotonic(self):
        with patch.object(d.subprocess, 'Popen') as spawn, self.assertRaises(TimeoutError):
            d.guarded_popen([sys.executable, '-c', 'pass'], deadline_monotonic=time.monotonic()+30,
                           deadline_wall=time.time()-1, evidence_path=self.root / 'absent')
        spawn.assert_not_called()
        self.assertFalse((self.root / 'absent').exists())

    def test_earlier_wall_deadline_kills_command(self):
        p = d.guarded_popen([sys.executable, '-c', 'import time;time.sleep(30)'],
            deadline_monotonic=time.monotonic()+20, deadline_wall=time.time()+0.4,
            evidence_path=self.root / 'guard.jsonl', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p)
        self.assertEqual(p.wait(timeout=3), -signal.SIGKILL)
        self.assertIn('deadline_expired', [r['event'] for r in self.records()])

    def test_owner_crash_kills_native_group_without_supervisor_registration(self):
        evidence = self.root / 'orphan.jsonl'
        ready = self.root / 'owner-ready'
        # The intermediate owner exits with os._exit: no Python finally/atexit
        # cleanup can run, and its child's owner pipe must still close.
        code = (
            "from experiments.headless.process_deadline import guarded_popen;"
            "import os,sys,time;from pathlib import Path;"
            f"p=guarded_popen([sys.executable,'-c','import time;time.sleep(30)'],deadline_monotonic=time.monotonic()+20,deadline_wall=time.time()+20,evidence_path={str(evidence)!r});"
            f"Path({str(ready)!r}).write_text(str(p.pid));"
            "os._exit(0)"
        )
        owner = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(owner.wait(timeout=3), 0)
            child = int(ready.read_text())
            self.wait_for(lambda: any(r['event'] == 'launcher_exited' for r in self.records('orphan.jsonl')))
            self.wait_for(lambda: not d.group_live_members(child))
            self.assertEqual(d.cleanup_from_evidence(evidence), (child,))
        finally:
            if owner.poll() is None:
                owner.kill(); owner.wait(timeout=3)
            d.cleanup_from_evidence(evidence)

    def test_owner_pipe_can_close_after_exec_and_stops_command(self):
        process = self.start('import time;time.sleep(30)', seconds=20)
        self.wait_for(lambda: any(r['event'] == 'exec_attempt' for r in self.records()))
        os.close(process._deadline_owner_fd)
        process._deadline_owner_fd = None
        self.assertEqual(process.wait(timeout=3), -signal.SIGKILL)
        self.assertIn('launcher_exited', [r['event'] for r in self.records()])

    def test_leader_exit_kills_surviving_group_child_without_external_cleanup(self):
        child_file = self.root / 'child-pid'
        code = ("import os,time;from pathlib import Path;child=os.fork();"
                f"Path({str(child_file)!r}).write_text(str(child)) if child else None;"
                "time.sleep(30) if child==0 else None")
        p = self.start(code, seconds=20)
        self.assertEqual(p.wait(timeout=3), 0)
        self.assertGreater(int(child_file.read_text()), 1)
        # The test owner remains alive and has not called close_owned_process.
        self.wait_for(lambda: any(r['event'] == 'parent_exited' for r in self.records()))
        self.wait_for(lambda: not d.group_live_members(p.pid))
        self.assertEqual(next(r for r in self.records() if r['event']=='parent_exited')['cleanup_signal'], 'SIGKILL')

    def test_exec_failure_is_retained_and_guardian_exits(self):
        p = d.guarded_popen([str(self.root / 'absent-program')], deadline_monotonic=time.monotonic()+5,
                           deadline_wall=time.time()+5, evidence_path=self.root / 'guard.jsonl',
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p)
        self.assertEqual(p.wait(timeout=3), 127)
        self.wait_for(lambda: any(r['event'] == 'parent_exited' for r in self.records()))
        self.assertIn('exec_failed', [r['event'] for r in self.records()])

    def test_refuses_shared_group_shell_override_and_overwriting_evidence(self):
        for kwargs in ({'start_new_session': False}, {'shell': True}, {'preexec_fn': lambda: None},
                       {'pass_fds': ()}, {'close_fds': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.start('pass', **kwargs)
        p = self.start('pass')
        p.wait(timeout=3)
        with self.assertRaises(FileExistsError):
            self.start('pass')

    def test_launcher_tracks_and_cleans_private_native_groups(self):
        launcher = d.DeadlineVcmiLauncher(Path(sys.executable), self.root, self.root / 'unused.vmap',
            deadline_monotonic=time.monotonic()+5, deadline_wall=time.time()+5,
            evidence_path=self.root / 'native.jsonl')
        with (self.root / 'native.log').open('w') as log:
            p = launcher._spawn_process([sys.executable, '-c', 'import time;time.sleep(30)'],
                                        {**os.environ, 'VCMI_EXPERIMENT_DIR': str(self.root / 'engine')}, log)
        self.processes.append(p)
        self.wait_for(lambda: any(r['event'] == 'exec_attempt' for r in self.records('native.jsonl')))
        self.assertEqual(launcher.launched_processes, [p])
        self.assertEqual(launcher.supervision_records[0]['process_group'], p.pid)
        with self.assertRaises(RuntimeError):
            launcher.assert_closed()
        launcher.close_owned_groups()
        launcher.assert_closed()
        self.assertEqual(p.returncode, -signal.SIGKILL)

    def test_environment_cleanup_closes_ownership_without_external_group_signals(self):
        import importlib
        # ContextualMixin deliberately imports the legacy top-level env alias.
        # Both module paths must reach the canonical guardian cleanup helper.
        for index, module_name in enumerate(('experiments.headless.env', 'env')):
            with self.subTest(module_name=module_name):
                stop_process_group = importlib.import_module(module_name).stop_process_group
                name = f'cleanup-{index}.jsonl'
                process = self.start('import time;time.sleep(30)', seconds=20, name=name)
                self.wait_for(lambda: any(r['event'] == 'exec_attempt' for r in self.records(name)))
                with patch.object(os, 'killpg') as external_signal:
                    stop_process_group(process)
                external_signal.assert_not_called()
                self.assertIsNone(process._deadline_owner_fd)
                self.assertEqual(process.returncode, -signal.SIGKILL)
                self.assertFalse(d.group_live_members(process.pid))
                self.assertIn('launcher_exited', [r['event'] for r in self.records(name)])

    def test_live_group_scan_ignores_zombies_and_rejects_own_group(self):
        result = subprocess.CompletedProcess([], 0, '71 99 Z\n72 99 S\n73 100 S\n', '')
        with patch.object(d.subprocess, 'run', return_value=result):
            self.assertEqual(d.group_live_members(99), ({'pid': 72, 'process_group': 99, 'state': 'S'},))
        with self.assertRaises(ValueError):
            d.group_live_members(os.getpgrp())

    def test_only_explicit_guarded_processes_use_guardian_cleanup(self):
        from experiments.headless.env import stop_process_group
        process = MagicMock()
        process.pid = 999999
        with patch.object(d, 'close_owned_process') as close, patch.object(os, 'killpg') as signal_group:
            stop_process_group(process)
        close.assert_not_called()
        self.assertEqual(signal_group.call_count, 2)
        with self.assertRaises(ValueError):
            d.close_owned_process(process)

    def test_stale_receipt_never_signals_numeric_process_group(self):
        path = self.root / 'stale.jsonl'
        path.write_text(json.dumps({'schema': d.SCHEMA, 'event': 'wrapper_started',
                                   'writer_pid': 99, 'leader_pid': 99, 'process_group': 99})+'\n')
        with patch.object(d, 'group_live_members', return_value=({'pid': 99},)), \
                patch.object(d.os, 'killpg') as kill, self.assertRaises(TimeoutError):
            d.cleanup_from_evidence(path, timeout=0.01)
        kill.assert_not_called()

    def test_expired_cleanup_releases_and_polls_without_new_wait_or_scan(self):
        for monotonic, wall in ((99, 1005), (105, 999)):
            with self.subTest(monotonic=monotonic, wall=wall):
                reader, writer = os.pipe()
                process = MagicMock()
                process._deadline_guarded = True
                process._deadline_owner_fd = writer
                try:
                    with patch.object(d.time, 'monotonic', return_value=100), \
                            patch.object(d.time, 'time', return_value=1000), \
                            patch.object(d, 'group_live_members') as scan, \
                            self.assertRaisesRegex(TimeoutError, 'not verified'):
                        d.close_owned_process(process, timeout=20,
                            deadline_monotonic=monotonic, deadline_wall=wall)
                    self.assertIsNone(process._deadline_owner_fd)
                    self.assertEqual(os.read(reader, 1), b'')
                    process.poll.assert_called()
                    process.wait.assert_not_called()
                    scan.assert_not_called()
                finally:
                    os.close(reader)

    def test_cleanup_uses_actual_submillisecond_remainder_and_rejects_late_receipt(self):
        clock = [100.0, 1000.0]
        process = MagicMock()
        process._deadline_guarded = True
        process._deadline_owner_fd = None
        def elapsed(seconds):
            clock[0] += seconds
            clock[1] += seconds
        process.wait.side_effect = lambda **kwargs: elapsed(.9995)
        def scan(group, *, timeout):
            self.assertGreater(timeout, 0)
            self.assertLess(timeout, .001)
            elapsed(.001)  # An empty scan completed after the fixed deadline.
            return ()
        with patch.object(d.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(d.time, 'time', side_effect=lambda: clock[1]), \
                patch.object(d, 'group_live_members', side_effect=scan) as observed, \
                self.assertRaises(TimeoutError):
            d.close_owned_process(process, timeout=1)
        process.wait.assert_called_once_with(timeout=1)
        observed.assert_called_once()

    def test_multiple_receipt_groups_share_one_budget_without_postexpiry_scans(self):
        path = self.root / 'many-groups.jsonl'
        path.write_text(''.join(json.dumps({'schema': d.SCHEMA, 'event': 'wrapper_started',
            'writer_pid': group, 'leader_pid': group, 'process_group': group}) + '\n'
            for group in (71, 72, 73)))
        clock = [100.0, 1000.0]
        timeouts = []
        def scan(group, *, timeout):
            timeouts.append(timeout)
            clock[0] += .6
            clock[1] += .6
            return ()
        with patch.object(d.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(d.time, 'time', side_effect=lambda: clock[1]), \
                patch.object(d, 'group_live_members', side_effect=scan) as observed, \
                self.assertRaises(TimeoutError):
            d.cleanup_from_evidence(path, timeout=30, deadline_monotonic=101, deadline_wall=1001)
        self.assertEqual(observed.call_count, 2)
        self.assertAlmostEqual(timeouts[0], 1)
        self.assertAlmostEqual(timeouts[1], .4)

    def test_unresolved_launch_receipt_cannot_certify_absent_native_group(self):
        path = self.root / 'unresolved-launch.jsonl'
        requested = {'schema': d.SCHEMA, 'event': 'launch_requested', 'command': ['unused']}
        path.write_text(json.dumps(requested) + '\n')
        with patch.object(d, 'group_live_members') as scan, patch.object(d.os, 'killpg') as signal_group, \
                self.assertRaisesRegex(RuntimeError, 'Unresolved process launch'):
            d.cleanup_from_evidence(path)
        scan.assert_not_called()
        signal_group.assert_not_called()
        self.assertEqual(path.read_text(), json.dumps(requested) + '\n')

    def test_failed_spawn_receipt_needs_no_process_group_cleanup(self):
        evidence = self.root / 'spawn-failure.jsonl'
        with patch.object(d.subprocess, 'Popen', side_effect=OSError('synthetic failed spawn')), \
                self.assertRaisesRegex(OSError, 'failed spawn'):
            d.guarded_popen([sys.executable, '-c', 'pass'], deadline_monotonic=time.monotonic()+5,
                deadline_wall=time.time()+5, evidence_path=evidence)
        self.assertEqual([row['event'] for row in self.records('spawn-failure.jsonl')],
                         ['launch_requested', 'launch_failed'])
        with patch.object(d, 'group_live_members') as scan, patch.object(d.os, 'killpg') as signal_group:
            self.assertEqual(d.cleanup_from_evidence(evidence), ())
        scan.assert_not_called()
        signal_group.assert_not_called()

    def test_exhausted_launcher_releases_every_real_group_before_any_wait(self):
        launcher = d.DeadlineVcmiLauncher(Path(sys.executable), self.root, self.root / 'unused.vmap',
            deadline_monotonic=time.monotonic()+20, deadline_wall=time.time()+20)
        for index in range(2):
            name = f'multiple-{index}.jsonl'
            process = self.start('import time;time.sleep(30)', seconds=20, name=name)
            launcher.launched_processes.append(process)
            self.wait_for(lambda name=name: any(row['event'] == 'exec_attempt' for row in self.records(name)))
        with patch.object(d, 'group_live_members') as scan, self.assertRaises(TimeoutError):
            launcher.close_owned_groups(timeout=0)
        scan.assert_not_called()
        self.assertTrue(all(process._deadline_owner_fd is None for process in launcher.launched_processes))
        # This is test-harness observation, not an extension of the expired
        # cleanup budget. Guardians must stop both groups on owner-pipe EOF.
        for process in launcher.launched_processes:
            self.assertEqual(process.wait(timeout=3), -signal.SIGKILL)
            self.wait_for(lambda process=process: not d.group_live_members(process.pid))


if __name__ == '__main__':
    unittest.main()
