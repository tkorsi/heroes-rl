#!/usr/bin/env python3
"""Compare Python and C++ PPO on disposable, audited historical state copies.

No game is launched, no seed is reserved, and no historical experience is
admitted again. Both paths use identical parent state and capture bytes. A
single fixed deadline includes audits, both updates and owned cleanup. Outputs
contain private model state and must remain outside the public repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
from experiments.learner.resources import make_resource_plan


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def remaining(deadline, wall):
    return min(deadline - time.monotonic(), wall - time.time())


def outside(path, root):
    path, root = path.resolve(), root.resolve()
    return not (str(path).casefold().rstrip('/') + '/').startswith(str(root).casefold().rstrip('/') + '/')


def summarize(results, order):
    if set(results) != {'python', 'native'}:
        return {'status': 'incomplete', 'completed_paths': list(results), 'order': list(order)}
    reference, native = (results[name]['repetitions'][0] for name in ('python', 'native'))
    fields = ('before_full_state_sha256', 'model_sha256', 'gradient_sha256', 'adam_sha256',
              'generator_sha256', 'full_state_sha256', 'sampler_order', 'report', 'state_parts_sha256')
    exact = {name: reference[name] == native[name] for name in fields}
    for name in ('audit', 'parent', 'checkout_head', 'driver'):
        exact[name] = results['python'][name] == results['native'][name]
    return {
        'status': 'passed' if all(exact.values()) else 'numerical_mismatch',
        'order': list(order),
        'exact': exact,
        'python_seconds': reference['seconds'],
        'native_seconds': native['seconds'],
        'kernel_speedup': reference['seconds'] / native['seconds'],
        'python_average_cpu_cores': reference.get('average_cpu_cores'),
        'native_average_cpu_cores': native.get('average_cpu_cores'),
        'scope': 'One offline PPO update per path, excluding gameplay and native replay wall time.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True, help='historical batch execution directory')
    parser.add_argument('--output', type=Path, required=True, help='fresh private output directory outside the repository')
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--native-manifest-sha256', required=True)
    parser.add_argument('--cpu-cap', type=int)
    parser.add_argument('--wall-seconds', type=float, default=120, help='whole comparison bound, at most 120 seconds')
    parser.add_argument('--order', choices=('PN', 'NP'), default='PN', help='Python/native order; reverse for a separate paired repetition')
    parser.add_argument('--profile', action='store_true', help='capture cProfile output, which affects measured timings')
    args = parser.parse_args(argv)
    if not math.isfinite(args.wall_seconds) or not 30 < args.wall_seconds <= 120:
        parser.error('--wall-seconds must be greater than 30 and no greater than 120')
    resource_plan = make_resource_plan(args.cpu_cap)
    output, batch = args.output.expanduser().resolve(), args.batch.expanduser().resolve()
    official = Path.home() / 'Projects/Heroes-artifacts/local-script-training'
    if not outside(output, ROOT) or not outside(output, official):
        parser.error('Private benchmark output must be outside the source repository and official training lineage')
    if output.exists():
        parser.error('Never reuse a diagnostic output directory')
    manifest = binding(args.native_manifest)
    if manifest['sha256'] != args.native_manifest_sha256:
        parser.error('Native manifest differs from its expected SHA256')
    plan_path = batch.parent / 'plan.json'
    plan = json.loads(plan_path.read_text())
    parent = plan['parent_descriptor']
    paths = {name: Path(parent[name]['path']) for name in ('plan', 'report', 'policy', 'resume')}
    order = tuple({'P': 'python', 'N': 'native'}[letter] for letter in args.order)
    start_wall, start_mono = time.time(), time.monotonic()
    deadline_wall, deadline = start_wall + args.wall_seconds, start_mono + args.wall_seconds
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    request = dict(schema='heroes-native-training-benchmark-v1',
        started_utc=datetime.fromtimestamp(start_wall, timezone.utc).isoformat(),
        hard_deadline_unix=deadline_wall, hard_deadline_monotonic=deadline,
        resources=resource_plan, order=list(order), batch=str(batch), capture_plan=binding(plan_path),
        parent={name: binding(path) for name, path in paths.items()}, native_manifest=manifest,
        driver=binding(__file__), kernel_driver=binding(ROOT / 'tools/acceleration3/ppo_bench.py'),
        profiling=args.profile, official_samples_admitted=0, official_updates=0)
    write_new(output / 'request.json', request)
    completed, attempts = {}, []
    error = None
    try:
        for name in order:
            if remaining(deadline, deadline_wall) <= 25:
                raise TimeoutError('Insufficient original comparison budget for another bounded path')
            result = output / (name + '.json')
            command = [sys.executable, str(ROOT / 'tools/acceleration3/ppo_bench.py'),
                '--checkout', str(ROOT), '--parent-plan', str(paths['plan']), '--parent-report', str(paths['report']),
                '--policy', str(paths['policy']), '--resume', str(paths['resume']), '--batch', str(batch),
                '--output', str(result), '--save-state', str(output / (name + '-state.pt')),
                '--wall-seconds', str(args.wall_seconds), '--deadline-unix', repr(deadline_wall - 5),
                '--deadline-monotonic', repr(deadline - 5), '--prefix-helpers', '0',
                '--text-replay-workers', str(max(0, min(2, resource_plan['budget_cpus'] - 1)))]
            if args.profile:
                command += ['--profile', str(output / (name + '.pstats'))]
            if name == 'native':
                command += ['--native-manifest', manifest['path'], '--native-manifest-sha256', manifest['sha256'],
                            '--native-prefix-threads', str(resource_plan['native_prefix_threads'])]
            evidence = output / (name + '-supervisor.jsonl')
            attempt = dict(path=name, command=command, status='started')
            attempts.append(attempt)
            with (output / (name + '.log')).open('xb') as log:
                process = guarded_popen(command, deadline_monotonic=deadline - 5, deadline_wall=deadline_wall - 5,
                                        evidence_path=evidence, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
                attempt['pid'] = process.pid
                try:
                    code = process.wait(timeout=max(0, remaining(deadline, deadline_wall) - 5))
                    attempt['returncode'] = code
                finally:
                    close_owned_process(process, timeout=5, deadline_monotonic=deadline, deadline_wall=deadline_wall)
                    cleanup_from_evidence(result.with_suffix('.json.guard.jsonl'), timeout=5,
                                          deadline_monotonic=deadline, deadline_wall=deadline_wall)
                    attempt['owned_cleanup_verified'] = True
            if code != 0:
                attempt['status'] = 'failed'
                raise RuntimeError(f'{name} path exited {code}; inspect its retained private log')
            completed[name] = json.loads(result.read_text())
            attempt['status'] = 'completed'
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
    summary = {**summarize(completed, order), 'schema': request['schema'], 'attempts': attempts,
               'error': error, 'wall_seconds': time.monotonic() - start_mono,
               'official_samples_admitted': 0, 'official_updates': 0,
               'limitations': ['A single order is not a thermal/cache-counterbalanced throughput estimate.',
                               'These disposable updates never advance the official checkpoint or consume game seeds.']}
    if error is not None:
        summary['status'] = 'failed'
    write_new(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
