#!/usr/bin/env python3
"""Measure the accepted native PPO workload using private disposable state.

Runs one native/Python comparison, bounded to 120 seconds, on archived batch846.
No gameplay, seed reservation or official update occurs. Reports PPO CPU use and
process lifetime peak RSS; it is not a full-campaign or artificial stress test.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_result(request, summary, native):
    attempts = summary.get('attempts', [])
    if (summary.get('status') != 'passed' or len(summary.get('exact', {})) != 13
            or not all(value is True for value in summary['exact'].values())
            or len(attempts) != 2 or {row.get('path') for row in attempts} != {'native', 'python'}
            or not all(row.get('status') == 'completed' and row.get('returncode') == 0
                       and row.get('owned_cleanup_verified') is True for row in attempts)):
        raise ValueError('The load comparison did not pass exact state and cleanup checks')
    plan = request['resources']
    repetition = native['repetitions'][0]
    cores = repetition['average_cpu_cores']
    capacity = plan['resources']['available_cpus']
    rss = repetition['process_lifetime_peak_rss_bytes']
    return dict(schema='heroes-native-load-test-v1',
                available_cpus=capacity, cpu_budget=plan['budget_cpus'],
                native_mean_cpu_cores=cores, available_cpu_capacity_percent=100 * cores / capacity,
                native_process_lifetime_peak_rss_bytes=rss,
                native_peak_concurrent_text_replays=repetition['timings']['native_peak_parallel_replays'],
                native_ppo_seconds=summary['native_seconds'], python_ppo_seconds=summary['python_seconds'],
                ppo_speedup=summary['kernel_speedup'], exact_state_checks_passed=13,
                owned_cleanup_verified=True, official_updates=0, seed_reservations=0,
                limitations=['CPU percentage is this PPO process against available CPU capacity, not total system activity.',
                             'Peak RSS includes this process lifetime through PPO, including imports, restoration and audits.',
                             'This test does not measure GPU use or full collection/replay campaign throughput.',
                             'One paired execution order is not a counterbalanced performance study.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-cap', type=int, help='Limit native workers; default is all available CPUs')
    parser.add_argument('--order', choices=('NP', 'PN'), default='NP', help='Native or Python first (default NP)')
    args = parser.parse_args(argv)
    from experiments.learner.native_execution import native_execution_choice
    from tools import benchmark_native_training as benchmark
    artifacts = Path.home() / 'Projects/Heroes-artifacts'
    manifest = artifacts / 'native-learner-builds/mac-native-v1-attempt6/manifest.json'
    accepted = json.loads((ROOT / 'validation/native-mac-learner-2026-10-29.json').read_text())
    sha = accepted['native_manifest_sha256']
    # Preflight official backend acceptance, even though this command is diagnostic.
    selected = native_execution_choice(manifest, sha, cpu_cap=args.cpu_cap)
    batch = artifacts / 'local-script-training/original/run-20261028T192106Z-44be2ecc/batches/train-000846-seat0/execution'
    if not (batch.parent / 'plan.json').is_file():
        raise FileNotFoundError('The preserved batch846 fixture is missing; no replacement or current run is selected')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = artifacts / 'native-load-tests' / (stamp + '-' + uuid.uuid4().hex[:10])
    console = output.with_suffix('.log')
    for path in (output, console):
        if not all(benchmark.outside(path, forbidden) for forbidden in (ROOT, artifacts / 'local-script-training')):
            raise ValueError('Load-test output must stay outside source and official training, including symlinks')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f"Testing archived PPO on disposable state, {args.order} order, at most 120 seconds.", flush=True)
    print(f"Available CPUs: {selected['resource_plan']['resources']['available_cpus']}; "
          f"native budget: {selected['resource_plan']['budget_cpus']}. Evidence: {output}", flush=True)
    command = ['--batch', str(batch), '--output', str(output), '--native-manifest', str(manifest),
               '--native-manifest-sha256', sha, '--wall-seconds', '120', '--order', args.order]
    if args.cpu_cap is not None:
        command += ['--cpu-cap', str(args.cpu_cap)]
    with console.open('x') as stream, redirect_stdout(stream):
        console.chmod(0o600)
        code = benchmark.main(command)
    if code:
        print(f"Load test failed (exit {code}). Preserved evidence: {output}; log: {console}", file=sys.stderr)
        return code
    read = lambda name: json.loads((output / name).read_text())
    result = load_result(read('request.json'), read('summary.json'), read('native.json'))
    benchmark.write_new(output / 'load-report.json', result)
    print(f"Native PPO: {result['native_ppo_seconds']:.2f}s; Python: {result['python_ppo_seconds']:.2f}s; "
          f"speedup: {result['ppo_speedup']:.2f}x.")
    print(f"Native mean CPU: {result['native_mean_cpu_cores']:.2f} of {result['available_cpus']} cores "
          f"({result['available_cpu_capacity_percent']:.1f}% of available CPU capacity).")
    print(f"Peak concurrent text replays: {result['native_peak_concurrent_text_replays']:.0f}.")
    print(f"Native process lifetime peak RSS: {result['native_process_lifetime_peak_rss_bytes'] / 2**30:.2f} GiB.")
    print('Exact state: 13/13 checks passed. Owned workers cleaned up. Official updates and new seeds: 0.')
    print('These are PPO measurements, not full-system, GPU or full-campaign utilization.')
    print(f"Report: {output / 'load-report.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
