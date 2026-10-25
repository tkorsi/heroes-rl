"""Compile and exercise the real Linux process reset primitive without assets."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import json
import os
import signal
import time
from unittest.mock import patch
from tools.benchmark_native_lifecycle import streaming_worker, measure, SCHEMA


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('g++'), 'requires Linux and g++')
class ForkLifecycleTests(unittest.TestCase):
    def test_process_memory_rng_descriptors_deadlines_and_reaping(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'lifecycle'
            asset = Path(directory) / 'readonly.txt'
            asset.write_text('immutable resource')
            subprocess.run(['g++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-pthread',
                            '-I', str(ROOT / 'engine/ExperimentReplay'),
                            str(ROOT / 'tests/native/fork_episode.cpp'), '-o', str(binary)],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(binary), str(asset)], check=True, timeout=10)


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('g++'), 'requires Linux and g++')
class NativeWorkerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='optimizer-protocol-')
        cls.binary = Path(cls.temporary.name) / 'worker-harness'
        engine = ROOT / 'vcmi'
        if not (engine / 'build/bin/libvcmi.so').exists():
            cls.temporary.cleanup()
            raise unittest.SkipTest('Requires prepared and built VCMI JSON library')
        # Compile canonical production code through an overlay, not a copied
        # protocol implementation. This also checks the root-owned source.
        overlay = Path(cls.temporary.name) / 'experimentreplay'
        overlay.mkdir()
        for name in ('WorkerProtocol.h', 'WorkerProtocol.cpp', 'ForkEpisode.h', 'StdInc.h'):
            (overlay / name).symlink_to(ROOT / 'engine/ExperimentReplay' / name)
        (overlay.parent / 'lib').symlink_to(engine / 'lib')
        (overlay.parent / 'Global.h').symlink_to(engine / 'Global.h')
        command = ['g++', '-std=c++20', '-O1', '-pthread', '-I' + str(overlay),
                   '-I' + str(engine), '-I' + str(engine / 'include'), '-I' + str(engine / 'build'),
                   str(ROOT / 'tests/native/worker_protocol.cpp'), str(overlay / 'WorkerProtocol.cpp'),
                   '-L' + str(engine / 'build/bin'), '-lvcmi', '-lboost_filesystem',
                   '-Wl,-rpath,' + str(engine / 'build/bin'), '-o', str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def launch(self, limit=100):
        return subprocess.Popen([str(self.binary), '--experimental-worker', str(limit)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def request(self, i, mode='ok', report='/tmp/unused-worker-report'):
        return {'schema': SCHEMA, 'kind': 'replay', 'job_id': i,
                'args': ['--log', mode, '--report', report]}

    def finish(self, process, data):
        output, errors = process.communicate(data, timeout=4)
        return [json.loads(line) for line in output.splitlines()], errors.decode()

    def test_runtime_admission_drift_stops_next_worker_dispatch(self):
        from tools.native_lifecycle_inventory import SCHEMA as INVENTORY_SCHEMA, require_inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}')
            spec = {'schema': INVENTORY_SCHEMA, 'data': [str(config)],
                    'configuration': [str(config)], 'mods': [], 'additional_libraries': []}
            expected = require_inventory(spec)
            calls = []
            def admission(stage):
                calls.append(stage)
                if len(calls) == 2:
                    config.write_text('{"drift":true}')
                require_inventory(spec, expected)
            commands = [self.request(i)['args'] for i in range(3)]
            with self.assertRaisesRegex(ValueError, 'config.json'):
                streaming_worker(self.binary, commands, root, .5, 1, 4, admission=admission)
            state = json.loads((root / 'service.json').read_text())
            self.assertEqual([row['status'] for row in state['jobs']],
                             ['completed', 'not_started', 'not_started'])
            self.assertEqual(len(state['jobs'][0]['receipts']), 2)
            self.assertTrue(state['cleanup']['worker_reaped'])
            self.assertEqual(state['status'], 'failed')
            # Actual production protocol callback only; no verified game report.

    def test_actual_protocol_crash_timeout_reuse_and_order(self):
        process = self.launch()
        requests = [self.request(i, mode) for i, mode in enumerate(['ok', 'crash', 'hang', 'ok', 'throw', 'ok'])]
        rows, errors = self.finish(process, ''.join(json.dumps(r) + '\n' for r in requests).encode())
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual(rows[0]['schema'], SCHEMA)
        self.assertEqual(rows[-1], {'schema': SCHEMA, 'kind': 'closed', 'jobs': 6})
        completed = [row for row in rows if row['kind'] == 'completed']
        self.assertEqual([r['job_id'] for r in completed], list(range(6)))
        self.assertEqual([r['exit_code'] for r in completed], [0, -1, -1, 0, 125, 0])
        self.assertEqual(completed[1]['signal'], signal.SIGKILL)
        self.assertTrue(completed[2]['timed_out'])
        self.assertEqual(len([r for r in rows if r['kind'] == 'started']), 6)

    def test_malformed_oversized_partial_and_duplicate_requests_fail_closed(self):
        good = json.dumps(self.request(0)).encode() + b'\n'
        cases = [b'{bad}\n', b'x' * 65537, good[:-1], b'\n',
                 json.dumps(self.request(1)).encode() + b'\n', good + good,
                 b'{"schema":"x","schema":"y"}\n',
                 json.dumps({**self.request(0), 'extra': 1}).encode() + b'\n',
                 json.dumps({**self.request(0), 'job_id': True}).encode() + b'\n',
                 json.dumps({**self.request(0), 'args': ['--log', 'ok']}).encode() + b'\n']
        for data in cases:
            with self.subTest(data=data[:60]):
                process = self.launch()
                rows, errors = self.finish(process, data)
                self.assertEqual(process.returncode, 3, errors)
                self.assertFalse(any(r['kind'] == 'closed' for r in rows))
                self.assertIn('Worker protocol failure', errors)

    def test_empty_eof_closes_and_broken_output_is_nonzero(self):
        process = self.launch()
        rows, errors = self.finish(process, b'')
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual([r['kind'] for r in rows], ['ready', 'closed'])
        process = self.launch()
        process.stdout.close()
        process.stdin.close()
        process.wait(timeout=3)
        self.assertNotEqual(process.returncode, 0)
        process.stderr.close()

    def inactive(self, pid):
        try:
            stat = Path(f'/proc/{pid}/stat').read_text()
            return stat[stat.rfind(')') + 2:].split()[0] == 'Z'
        except FileNotFoundError:
            return True

    def await_inactive(self, pid):
        end = time.monotonic() + 2
        while not self.inactive(pid) and time.monotonic() < end:
            time.sleep(.005)
        self.assertTrue(self.inactive(pid), f'Live descendant {pid}')

    def test_ordinary_descendant_cleanup_on_completion_timeout_and_parent_loss(self):
        for cause in ('completion', 'crash', 'timeout', 'parent_loss'):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / 'pid'
                process = self.launch(250)
                self.assertEqual(json.loads(process.stdout.readline())['kind'], 'ready')
                mode = {'completion': 'descendant', 'crash': 'descendant_crash'}.get(cause, 'descendant_hang')
                process.stdin.write((json.dumps(self.request(0, mode, str(report))) + '\n').encode())
                process.stdin.flush()
                started = json.loads(process.stdout.readline())
                end = time.monotonic() + 2
                while (not report.exists() or not report.read_text().strip()) and time.monotonic() < end:
                    time.sleep(.005)
                descendant = int(report.read_text())
                before = time.monotonic()
                if cause == 'parent_loss':
                    process.kill()
                else:
                    receipt = json.loads(process.stdout.readline())
                    self.assertEqual(receipt['timed_out'], cause == 'timeout')
                    self.assertTrue(receipt['cleanup_complete'])
                    self.assertFalse(Path(f'/proc/{descendant}').exists(), 'Service must reap ordinary descendants')
                    process.stdin.close()
                process.wait(timeout=2)
                self.await_inactive(started['pid'])
                self.await_inactive(descendant)
                self.assertLess(time.monotonic() - before, 2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if not stream.closed: stream.close()

    def test_streaming_native_success_and_failure_preserve_attribution(self):
        for mode in ('ok', 'fail', 'hang'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                commands = [self.request(0)['args'], self.request(1, mode)['args'], self.request(2)['args']]
                if mode == 'ok':
                    result = streaming_worker(self.binary, commands, output, .3, 1, 3)
                    self.assertEqual(result['status'], 'protocol_completed')
                else:
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        streaming_worker(self.binary, commands, output, .15, 1, 3)
                state = json.loads((output / 'service.json').read_text())
                self.assertEqual(state['jobs'][0]['status'], 'completed')
                self.assertEqual(state['jobs'][2]['status'], 'completed' if mode == 'ok' else 'not_started')
                self.assertTrue(state['cleanup']['worker_reaped'])
                self.assertTrue((output / 'wire.jsonl').read_bytes())

    def test_startup_and_overall_deadlines_persist_state(self):
        for startup in (True, False):
            with self.subTest(startup=startup), tempfile.TemporaryDirectory() as directory:
                with patch.dict(os.environ, {'HARNESS_INIT_MS': '500' if startup else '0'}):
                    with self.assertRaises(TimeoutError):
                        streaming_worker(self.binary, [self.request(0, 'hang')['args']], Path(directory),
                                         2, .05 if startup else 1, .1)
                state = json.loads((Path(directory) / 'service.json').read_text())
                self.assertNotEqual(state['status'], 'protocol_completed')
                self.assertTrue(state['cleanup']['worker_reaped'])
                self.assertEqual(len(state['jobs']), 1)

    def test_supervisor_rejects_corrupt_ready_missing_duplicate_and_partial_responses(self):
        ready = {'schema': SCHEMA, 'kind': 'ready', 'load_seconds': 0}
        started = {'schema': SCHEMA, 'kind': 'started', 'job_id': 0, 'pid': 999999999}
        completed = {'schema': SCHEMA, 'kind': 'completed', 'job_id': 0, 'exit_code': 0,
                     'signal': 0, 'timed_out': False, 'seconds': 0, 'cleanup_complete': True}
        closed = {'schema': SCHEMA, 'kind': 'closed', 'jobs': 1}
        transcripts = [json.dumps({**ready, 'schema': 'old'}) + '\n',
                       json.dumps({**ready, 'load_seconds': True}) + '\n',
                       json.dumps(ready) + '\n',
                       '\n'.join(map(json.dumps, [ready, completed])) + '\n',
                       '\n'.join(map(json.dumps, [ready, started, {**completed, 'job_id': 1}])) + '\n',
                       '\n'.join(map(json.dumps, [ready, started, completed, completed, closed])) + '\n',
                       '\n'.join(map(json.dumps, [ready, started])) + '\n{"partial":',
                       '{"schema":"x","schema":"y"}\n', 'x' * 65537 + '\n']
        for transcript in transcripts:
            with self.subTest(transcript=transcript[:60]), tempfile.TemporaryDirectory() as directory:
                peer = Path(directory) / 'corrupt-peer'
                # Fixed corrupt transcripts exercise admission failures. This
                # peer implements no request parser or successful service.
                peer.write_text('#!' + sys.executable + '\nimport sys\nsys.stdout.write(' + repr(transcript)
                                + ')\nsys.stdout.flush()\nsys.stdin.buffer.read()\n')
                peer.chmod(0o755)
                with self.assertRaises((RuntimeError, ValueError, TimeoutError)):
                    streaming_worker(peer, [self.request(0)['args']], Path(directory), .15, .15, .5)
                state = json.loads((Path(directory) / 'service.json').read_text())
                self.assertNotEqual(state['status'], 'protocol_completed')
                self.assertTrue(state['cleanup']['worker_reaped'])

    def test_supervisor_interruptions_keep_completed_receipts_and_clean_groups(self):
        for interruption in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=interruption), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                descendant_path = output / 'descendant'
                commands = [self.request(0)['args'], self.request(1, 'descendant_hang', str(descendant_path))['args'],
                            self.request(2)['args']]
                script = ('from tools.benchmark_native_lifecycle import streaming_worker\n'
                          'from pathlib import Path\nimport sys,json\n'
                          'streaming_worker(Path(sys.argv[1]),json.loads(sys.argv[3]),Path(sys.argv[2]),10,2,20)\n')
                with (output / 'supervisor.log').open('wb') as log:
                    supervisor = subprocess.Popen([sys.executable, '-c', script, str(self.binary), directory,
                                                   json.dumps(commands)], cwd=ROOT, stdout=log, stderr=log)
                    end = time.monotonic() + 4
                    state = None
                    while time.monotonic() < end:
                        path = output / 'service.json'
                        if path.exists():
                            state = json.loads(path.read_text())
                            if state['jobs'][1]['status'] == 'started' and descendant_path.exists(): break
                        time.sleep(.01)
                    try:
                        self.assertIsNotNone(state)
                        self.assertEqual(state['jobs'][1]['status'], 'started')
                        descendant = int(descendant_path.read_text())
                        before = time.monotonic()
                        supervisor.send_signal(interruption)
                        supervisor.wait(timeout=3)
                        self.assertLess(time.monotonic() - before, 3)
                        state = json.loads((output / 'service.json').read_text())
                        self.assertEqual(state['status'], 'incomplete' if interruption == signal.SIGKILL else 'interrupted')
                        self.assertEqual([r['status'] for r in state['jobs']], ['completed', 'started' if interruption == signal.SIGKILL else 'incomplete', 'not_started'])
                        if interruption != signal.SIGKILL:
                            self.assertTrue(state['cleanup']['worker_reaped'])
                        self.await_inactive(state['worker_pid'])
                        self.await_inactive(descendant)
                    finally:
                        if supervisor.poll() is None:
                            supervisor.kill()
                            supervisor.wait(timeout=2)


    def test_broken_output_after_started_cleans_episode_group(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'pid'
            process = self.launch(150)
            process.stdout.readline()
            process.stdin.write((json.dumps(self.request(0, 'descendant_hang', str(report))) + '\n').encode())
            process.stdin.flush()
            started = json.loads(process.stdout.readline())
            end = time.monotonic() + 1
            while (not report.exists() or not report.read_text().strip()) and time.monotonic() < end: time.sleep(.005)
            descendant = int(report.read_text())
            process.stdout.close()
            process.wait(timeout=2)
            self.assertEqual(process.returncode, 3)
            self.await_inactive(started['pid'])
            self.await_inactive(descendant)
            process.stdin.close()
            process.stderr.close()

    def test_transport_completion_without_verified_native_report_is_never_admitted(self):
        for mode in ('cold', 'worker'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    measure(self.binary, [['--log', 'ok']], Path(directory) / 'run', mode, .3, 1, 2)
                # The harness deliberately supplies no fabricated Heroes report.
                path = Path(directory) / 'run' / ('service.json' if mode == 'worker' else 'cold.json')
                state = json.loads(path.read_text())
                self.assertNotIn(state['status'], ('verified_report_equivalence', 'success'))


if __name__ == '__main__':
    unittest.main()
