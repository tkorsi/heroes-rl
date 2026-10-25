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
            elif option != '--prefix':
                raise ValueError('Unsupported manifest option: ' + option)
            index += 1
        if '--log' not in seen:
            raise ValueError('Every row needs --log')
    return jobs


def measure(binary, jobs, output, mode, job_seconds, startup_seconds):
    output.mkdir()
    commands = [job + ['--report', str(output / f'{i}.json')] for i, job in enumerate(jobs)]
    started = time.monotonic()
    receipts = []
    with (output / 'native.log').open('wb') as logs:
        if mode == 'cold':
            for command in commands:
                before = time.monotonic()
                process = subprocess.run([str(binary), *command], stdout=logs, stderr=logs,
                                         timeout=job_seconds + startup_seconds)
                receipts.append({'exit_code': process.returncode, 'seconds': time.monotonic() - before})
                if process.returncode:
                    raise RuntimeError('Cold replay failed; inspect ' + str(output))
        else:
            process = subprocess.run([str(binary), '--experimental-worker', str(int(job_seconds * 1000))],
                                     input=''.join(json.dumps(row) + '\n' for row in commands),
                                     stdout=subprocess.PIPE, stderr=logs, text=True,
                                     timeout=startup_seconds + len(jobs) * job_seconds + 5)
            replies = [json.loads(line) for line in process.stdout.splitlines()]
            (output / 'receipts.json').write_text(json.dumps(replies, indent=2) + '\n')
            if (process.returncode or len(replies) != len(jobs) + 1
                    or replies[0].get('kind') != 'ready'
                    or any(row.get('kind') != 'completed' or row.get('exit_code') != 0
                           or row.get('signal') != 0 or row.get('timed_out') is not False for row in replies[1:])):
                raise RuntimeError('Worker replay incomplete or failed; inspect ' + str(output))
            receipts = replies
    elapsed = time.monotonic() - started
    reports = [json.loads((output / f'{i}.json').read_text()) for i in range(len(jobs))]
    if any(report.get('verified') is not True for report in reports):
        raise RuntimeError('Unverified native replay')
    return {'mode': mode, 'wall_seconds': elapsed, 'receipts': receipts,
            'native_timings': [report.get('timing_seconds') for report in reports],
            'native_episode_timings': [report.get('summary', {}).get('timing_seconds') for report in reports],
            'semantic_reports': [semantic(report) for report in reports]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--job-seconds', type=float, default=120)
    parser.add_argument('--startup-seconds', type=float, default=120)
    args = parser.parse_args()
    if args.repeats < 1 or args.job_seconds <= 0 or args.startup_seconds <= 0:
        parser.error('repeats and deadlines must be positive')
    jobs = validate_jobs(json.loads(args.jobs.read_text()))
    binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir()
    report = {'schema': 'heroes-native-lifecycle-benchmark-v1', 'status': 'incomplete',
              'binary_sha256': digest(binary), 'jobs_sha256': digest(args.jobs),
              'workload': 'native replay verification; not gameplay throughput', 'runs': [],
              'unmeasured': ['complete_game_throughput', 'policy_inference', 'live_transport',
                             'live_opponent_work', 'pathfinding_and_autobattle_breakdown',
                             'end_to_end_training_experiment']}
    destination = args.output / 'summary.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    try:
        reference = None
        for repeat in range(args.repeats):
            for mode in (('cold', 'worker') if repeat % 2 == 0 else ('worker', 'cold')):
                run = measure(binary, jobs, args.output / f'{repeat}-{mode}', mode,
                              args.job_seconds, args.startup_seconds)
                semantics = run.pop('semantic_reports')
                if reference is None:
                    reference = semantics
                if reference != semantics:
                    raise RuntimeError('Fresh and reset replay reports differ')
                if digest(binary) != report['binary_sha256'] or digest(args.jobs) != report['jobs_sha256']:
                    raise RuntimeError('Binary or job manifest drift')
                report['runs'].append(run)
        report['status'] = 'verified_report_equivalence'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
