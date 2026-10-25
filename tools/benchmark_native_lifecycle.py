#!/usr/bin/env python3
"""Compare fresh-process and experimental preloaded native replay lifecycles.

No collection, inference, optimization, or synthetic game rules. Each manifest
row is an argument array containing --log and optionally --contextual,
--observations and --prefix. Include A/B/A and natural/prefix recordings to test
reset independence. Input captures are never modified. Outputs must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import math
import os
import select
import signal

try:
    from tools.native_lifecycle_inventory import runtime_inventory, require_inventory
except ModuleNotFoundError:  # Direct script execution.
    from native_lifecycle_inventory import runtime_inventory, require_inventory


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def semantic(value):
    """Only the newly additive timing fields are excluded from exact comparison."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k != 'timing_seconds'}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Expected a nonempty list of replay argument arrays')
    for job in jobs:
        if not isinstance(job, list) or any(type(v) is not str or '\0' in v for v in job):
            raise ValueError('Job arguments must be strings without NUL')
        seen = set()
        index = 0
        while index < len(job):
            option = job[index]
            if option in seen:
                raise ValueError('Duplicate option: ' + option)
            seen.add(option)
            if option in ('--log', '--contextual', '--observations'):
                index += 1
                if index >= len(job) or not Path(job[index]).exists():
                    raise ValueError('Missing input for ' + option)
                path = Path(job[index])
                if option == '--observations':
                    if not path.is_dir() or not list(path.glob('observations-p*.jsonl')):
                        raise ValueError('Observation directory needs observations-p*.jsonl: ' + str(path))
                elif not path.is_file():
                    raise ValueError('Expected regular file for ' + option)
            elif option != '--prefix':
                raise ValueError('Unsupported manifest option: ' + option)
            index += 1
        if '--log' not in seen:
            raise ValueError('Every row needs --log')
    return jobs


def input_identity(jobs):
    paths = set()
    for job in jobs:
        for index, option in enumerate(job):
            if option in ('--log', '--contextual'):
                paths.add(Path(job[index + 1]).resolve())
            elif option == '--observations':
                paths.update(Path(job[index + 1]).resolve().glob('observations-p*.jsonl'))
    return {str(path): digest(path) for path in sorted(paths)}


def library_identity(binary):
    # These are the directly linked runtime libraries, not a complete inventory
    # of mods, Data or libraries opened later by dlopen.
    output = subprocess.check_output(['ldd', str(binary)], text=True, stderr=subprocess.STDOUT)
    if 'not found' in output:
        raise ValueError('Unresolved linked libraries: ' + output.strip())
    paths = set()
    for line in output.splitlines():
        for token in line.split():
            if token.startswith('/') and Path(token).is_file():
                paths.add(Path(token).resolve())
    return {str(path): digest(path) for path in sorted(paths)}


SCHEMA = 'heroes-replay-worker-v2'


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate response field')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Wire:
    def __init__(self, process, journal):
        self.process, self.journal = process, journal
        self.buffer = bytearray()
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)

    def read(self, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('Worker response deadline')
        while b'\n' not in self.buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('Worker response deadline')
            if not select.select([self.process.stdout], [], [], left)[0]:
                raise TimeoutError('Worker response deadline')
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                if self.buffer:
                    raise RuntimeError('Partial response at EOF')
                return None
            self.journal.write(chunk)
            self.journal.flush()
            os.fsync(self.journal.fileno())
            self.buffer.extend(chunk)
            if len(self.buffer.split(b'\n', 1)[0]) > 65536:
                raise RuntimeError('Oversized worker response')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return strict_json(line.decode('utf-8'))

    def write(self, request, deadline):
        payload = (json.dumps(request, allow_nan=False) + '\n').encode()
        if len(payload) > 65537:
            raise ValueError('Oversized worker request')
        while payload:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([], [self.process.stdin], [], max(0, left))[1]:
                raise TimeoutError('Worker request deadline')
            try:
                count = os.write(self.process.stdin.fileno(), payload)
                payload = payload[count:]
            except BlockingIOError:
                continue


def validate_response(row, kind, fields, job_id=None):
    if (type(row) is not dict or set(row) != {'schema', 'kind', *fields}
            or row['schema'] != SCHEMA or row['kind'] != kind):
        raise RuntimeError('Unexpected response schema/kind/fields')
    if job_id is not None and (type(row['job_id']) is not int or row['job_id'] != job_id):
        raise RuntimeError('Missing, duplicate or mismatched job response')
    for key in ('seconds', 'load_seconds'):
        if key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0):
            raise RuntimeError('Invalid response timing')
    return row


def streaming_worker(binary, commands, output, job_seconds, startup_seconds,
                     overall_seconds, cleanup_seconds=2, admission=None):
    """One outstanding job. Persist attribution before dispatch and each receipt.

    Callbacks must remain in the episode process group. No retries. A killed
    supervisor cannot persist its final exception, so the initial journal/state
    deliberately remains incomplete until orderly protocol closure and reaping.
    """
    output = Path(output)
    state_path = output / 'service.json'
    state = {'schema': SCHEMA, 'status': 'incomplete', 'ready': None, 'closed': None,
             'jobs': [{'job_id': i, 'args': args, 'status': 'not_started', 'receipts': []}
                      for i, args in enumerate(commands)], 'cleanup': None}
    save_state(state_path, state)
    process = None
    started = time.monotonic()
    deadline = started + overall_seconds
    active_group = None
    active_birth = None
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    old_term = signal.signal(signal.SIGTERM, terminate)
    try:
        with (output / 'native.log').open('ab') as logs, (output / 'wire.jsonl').open('ab') as journal:
            process = subprocess.Popen([str(binary), '--experimental-worker', str(int(job_seconds * 1000))],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logs,
                                       start_new_session=True)
            state['worker_pid'] = process.pid
            save_state(state_path, state)
            wire = Wire(process, journal)
            state['ready'] = validate_response(wire.read(min(deadline, started + startup_seconds)),
                                               'ready', {'load_seconds'})
            save_state(state_path, state)
            for row in state['jobs']:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Overall worker deadline')
                if admission:
                    admission('before_worker_job')
                row['status'] = 'requested'
                save_state(state_path, state)
                job_deadline = min(deadline, time.monotonic() + job_seconds)
                wire.write({'schema': SCHEMA, 'kind': 'replay', 'job_id': row['job_id'], 'args': row['args']},
                           job_deadline)
                receipt = validate_response(wire.read(job_deadline), 'started', {'job_id', 'pid'}, row['job_id'])
                if type(receipt['pid']) is not int or receipt['pid'] <= 1:
                    raise RuntimeError('Invalid episode process group')
                active_group = receipt['pid']
                active_birth = None
                try:
                    fields = Path(f'/proc/{active_group}/stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[1]) != process.pid or int(fields[2]) != active_group:
                        raise RuntimeError('Episode is not a direct owned process group')
                    active_birth = fields[19]
                except FileNotFoundError:
                    pass  # A fast completed episode may already have been reaped.
                row['receipts'].append(receipt)
                row['status'] = 'started'
                save_state(state_path, state)
                receipt = validate_response(wire.read(job_deadline), 'completed',
                                            {'job_id', 'exit_code', 'signal', 'timed_out', 'seconds', 'cleanup_complete'}, row['job_id'])
                if (type(receipt['exit_code']) is not int or type(receipt['signal']) is not int
                        or type(receipt['timed_out']) is not bool or not -1 <= receipt['exit_code'] <= 125
                        or receipt['signal'] < 0 or receipt['cleanup_complete'] is not True):
                    raise RuntimeError('Invalid completion status')
                active_group = None
                row['receipts'].append(receipt)
                row['status'] = 'completed' if (receipt['exit_code'] == 0 and receipt['signal'] == 0
                                               and receipt['timed_out'] is False) else 'failed'
                save_state(state_path, state)
                if row['status'] == 'failed':
                    raise RuntimeError('Native job failed; no replacement or retry')
            process.stdin.close()
            state['closed'] = validate_response(wire.read(deadline), 'closed', {'jobs'})
            if type(state['closed']['jobs']) is not int or state['closed']['jobs'] != len(commands):
                raise RuntimeError('Missing or extra completion count')
            if wire.read(deadline) is not None:
                raise RuntimeError('Unexpected response after closure')
            process.wait(timeout=max(0, deadline - time.monotonic()))
            if process.returncode != 0:
                raise RuntimeError('Worker exited unsuccessfully')
            state['status'] = 'protocol_completed'
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        state['error'] = f'{type(error).__name__}: {error}'
        for row in state['jobs']:
            if row['status'] in ('requested', 'started'):
                row['status'] = 'incomplete'
        raise
    finally:
        # One absolute cleanup budget, including the worker wait. The native
        # episode reserves SIGUSR2 for parent loss; the supervisor also kills
        # the known active group before terminating the service.
        cleanup_end = min(deadline, time.monotonic()) + cleanup_seconds
        cleanup = {'worker_reaped': process is None, 'budget_seconds': cleanup_seconds}
        if process is not None:
            if process.poll() is None:
                if active_group is not None and active_birth is not None:
                    try:
                        fields = Path(f'/proc/{active_group}/stat').read_text().rsplit(')', 1)[1].split()
                        if (fields[19] == active_birth and int(fields[1]) == process.pid
                                and int(fields[2]) == active_group):
                            os.killpg(active_group, signal.SIGKILL)
                    except (ProcessLookupError, FileNotFoundError):
                        pass
                    except OSError as error:
                        cleanup['group_error'] = str(error)
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=max(0, cleanup_end - time.monotonic()))
                cleanup['worker_reaped'] = True
            except subprocess.TimeoutExpired:
                cleanup['worker_reaped'] = False
            for stream in (process.stdin, process.stdout):
                if stream and not stream.closed:
                    stream.close()
        cleanup['deadline_exceeded'] = time.monotonic() > cleanup_end
        state['cleanup'] = cleanup
        if not cleanup['worker_reaped'] or cleanup['deadline_exceeded'] or 'group_error' in cleanup:
            state['status'] = 'cleanup_failed'
        state['wall_seconds'] = time.monotonic() - started
        save_state(state_path, state)
        signal.signal(signal.SIGTERM, old_term)
    if state['status'] != 'protocol_completed':
        raise RuntimeError('Incomplete worker cleanup')
    return state


def measure(binary, jobs, output, mode, job_seconds, startup_seconds, overall_seconds=None, cleanup_seconds=2, admission=None):
    output.mkdir()
    commands = [job + ['--report', str(output / f'{i}.json')] for i, job in enumerate(jobs)]
    started = time.monotonic()
    receipts = []
    with (output / 'native.log').open('wb') as logs:
        if mode == 'cold':
            run_end = started + (overall_seconds or startup_seconds + len(jobs) * job_seconds)
            cold = {'status': 'incomplete', 'jobs': [{'job_id': i, 'status': 'not_started'}
                                                    for i in range(len(commands))]}
            path = output / 'cold.json'
            save_state(path, cold)
            try:
                for row, command in zip(cold['jobs'], commands):
                    if time.monotonic() >= run_end:
                        raise TimeoutError('Overall cold replay deadline')
                    if admission:
                        admission('before_cold_job')
                    before = time.monotonic()
                    row['status'] = 'requested'
                    save_state(path, cold)
                    process = None
                    try:
                        process = subprocess.Popen([str(binary), *command], stdout=logs, stderr=logs,
                                                   start_new_session=True)
                        process.wait(timeout=max(0, min(run_end, before + job_seconds + startup_seconds)
                                                 - time.monotonic()))
                        row.update(status='completed' if process.returncode == 0 else 'failed',
                                   exit_code=process.returncode, seconds=time.monotonic() - before)
                        receipts.append(dict(row))
                        if process.returncode:
                            raise RuntimeError('Cold replay failed')
                    except BaseException:
                        if row['status'] == 'requested': row['status'] = 'incomplete'
                        raise
                    finally:
                        cleanup_end = min(run_end, time.monotonic()) + cleanup_seconds
                        if process is not None and process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=max(0, cleanup_end - time.monotonic()))
                        save_state(path, cold)
                cold['status'] = 'processes_completed'
            except BaseException as error:
                cold['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
                cold['error'] = str(error)
                raise
            finally:
                save_state(path, cold)
        else:
            service = streaming_worker(binary, commands, output, job_seconds, startup_seconds,
                                       overall_seconds or startup_seconds + len(jobs) * job_seconds + 5,
                                       cleanup_seconds, admission)
            receipts = [service['ready'], *[receipt for row in service['jobs'] for receipt in row['receipts']],
                        service['closed']]
    elapsed = time.monotonic() - started
    reports = [json.loads((output / f'{i}.json').read_text()) for i in range(len(jobs))]
    if any(report.get('verified') is not True for report in reports):
        raise RuntimeError('Unverified native replay')
    return {'mode': mode, 'wall_seconds': elapsed, 'receipts': receipts,
            'native_timings': [report.get('timing_seconds') for report in reports],
            'native_episode_timings': [report.get('summary', {}).get('timing_seconds',
                                      report.get('summary', {}).get('server', {}).get('timing_seconds'))
                                      for report in reports],
            'semantic_reports': [semantic(report) for report in reports]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--runtime-spec', type=Path, help='Explicit real-path inventory specification')
    parser.add_argument('--runtime-manifest', type=Path, help='Previously reviewed inventory.json to enforce')
    parser.add_argument('--inventory-only', action='store_true', help='Write runtime inventory/missing-input report without launching')
    parser.add_argument('--jobs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--job-seconds', type=float, default=120)
    parser.add_argument('--startup-seconds', type=float, default=120)
    parser.add_argument('--overall-seconds', type=float, default=3600)
    parser.add_argument('--cleanup-seconds', type=float, default=2)
    args = parser.parse_args()
    if (args.repeats < 1 or not 0.001 <= args.job_seconds <= 3600
            or any(not math.isfinite(v) or v <= 0 for v in (args.startup_seconds, args.overall_seconds, args.cleanup_seconds))):
        parser.error('repeats and deadlines must be positive')
    if args.inventory_only:
        if not args.runtime_spec or args.runtime_manifest:
            parser.error('--inventory-only requires --runtime-spec and no --runtime-manifest')
        args.output.mkdir()
        try:
            inventory = runtime_inventory(strict_json(args.runtime_spec.read_text()))
        except (OSError, ValueError) as error:
            inventory = {'admitted': False, 'errors': [str(error)]}
        save_state(args.output / 'inventory.json', inventory)
        if not inventory['admitted']:
            parser.exit(2, 'Missing runtime prerequisites; see ' + str(args.output / 'inventory.json') + '\n')
        return
    if not args.binary or not args.jobs:
        parser.error('--binary and --jobs are required for a benchmark')
    if bool(args.runtime_spec) != bool(args.runtime_manifest):
        parser.error('Use --runtime-spec and --runtime-manifest together')
    args.output = args.output.resolve()
    args.output.mkdir()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).splitlines()
    report = {'schema': 'heroes-native-lifecycle-benchmark-v1', 'status': 'incomplete',
              'source_revision': revision, 'source_dirty': dirty,
              'runtime_admission': {'enabled': bool(args.runtime_spec), 'checks': []},
              'workload': 'native replay verification; not gameplay throughput', 'runs': [],
              'wall_timing_scope': 'Includes synchronous inventory/admission overhead; native phase timers are unchanged',
              'unmeasured': ['complete_game_throughput', 'policy_inference', 'live_transport',
                             'live_opponent_work', 'pathfinding_and_autobattle_breakdown',
                             'end_to_end_training_experiment']}
    destination = args.output / 'summary.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    campaign_deadline = time.monotonic() + args.overall_seconds
    try:
        spec = None
        if args.runtime_spec:
            spec = strict_json(args.runtime_spec.read_text())
            expected = strict_json(args.runtime_manifest.read_text())
            report['runtime_admission']['expected'] = expected
            spec_sha = digest(args.runtime_spec)
            manifest_sha = digest(args.runtime_manifest)
        def admission(stage):
            check_started = time.monotonic()
            check = {'stage': stage, 'admitted': False}
            report['runtime_admission']['checks'].append(check)
            try:
                if spec is not None:
                    current = runtime_inventory(spec)
                    check['runtime'] = current
                    require_inventory(spec, expected, current)
                    if digest(args.runtime_spec) != spec_sha or digest(args.runtime_manifest) != manifest_sha:
                        raise ValueError('Runtime spec or reviewed manifest drift')
                    check.pop('runtime')
                actual = {'binary_sha256': digest(binary), 'jobs_sha256': digest(args.jobs),
                          'input_files': input_identity(jobs), 'linked_libraries': library_identity(binary)}
                changed = [key for key in actual if actual[key] != report[key]]
                if changed:
                    check['actual_input_identity'] = actual
                    raise ValueError('Input identity drift: ' + ', '.join(changed))
                if time.monotonic() >= campaign_deadline:
                    raise TimeoutError('Overall deadline during admission; no dispatch/acceptance')
                check['admitted'] = True
            except BaseException as error:
                check['error'] = str(error)
                raise
            finally:
                check['seconds'] = time.monotonic() - check_started
                save_state(destination, report)
        # All admission failures remain attributable even before a native process exists.
        jobs = validate_jobs(strict_json(args.jobs.read_text()))
        binary = args.binary.resolve(strict=True)
        report.update(binary_sha256=digest(binary), jobs_sha256=digest(args.jobs),
                      input_files=input_identity(jobs), linked_libraries=library_identity(binary))
        admission('preflight')
        reference = None
        for repeat in range(args.repeats):
            for mode in (('cold', 'worker') if repeat % 2 == 0 else ('worker', 'cold')):
                if time.monotonic() >= campaign_deadline:
                    raise TimeoutError('Overall campaign deadline; no further run started')
                admission('before_mode_launch')
                run = measure(binary, jobs, args.output / f'{repeat}-{mode}', mode,
                              args.job_seconds, args.startup_seconds,
                              max(0.001, campaign_deadline - time.monotonic()), args.cleanup_seconds, admission)
                admission('before_comparison_acceptance')
                semantics = run.pop('semantic_reports')
                if reference is None:
                    reference = semantics
                if reference != semantics:
                    raise RuntimeError('Fresh and reset replay reports differ')
                report['runs'].append(run)
        admission('final_acceptance')
        report['status'] = 'verified_report_equivalence'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        report['error'] = str(error)
        raise
    finally:
        save_state(destination, report)


if __name__ == '__main__':
    main()
