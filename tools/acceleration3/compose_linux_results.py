#!/usr/bin/env python3
"""Compose the compact Linux validation record of the original-map local runner acceleration.

    compose_linux_results.py --bench BENCH_ROOT --campaigns m1-B m3-RH ... \
        --optimizer-only LABEL=RESULT.json ... --out validation/original5000-local-acceleration-linux-2026-10-27.json

Reads only compact records: each campaign's ``bench-run.json``, ``summary.json`` and ``resources.jsonl``
(through ``analyze_campaigns``), the optimizer-only results of ``ppo_bench.py`` (timings and state digests,
no samples), and the accepted path's own per-game deadline evidence for its stopped batch. No raw game
data, capture or checkpoint is copied; states are bound by their SHA-256 digests.

Projection for the manual run (``--games 5000 --eval-every 100``: 1,250 training updates; eight evaluation
games before the first update, after every 100 updates and at the end, 26 evaluation batches of one seat):
    wall = cold start + 1,250 x (mean measured training batch) + 26 x (mean measured evaluation batch)
from batches of this host that start at the 629 parent. Later batches depend on how long the policy's games
become; nothing here predicts that, so the projection is a scale, not a forecast.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_campaigns as analysis  # noqa: E402

UPDATES, EVALUATION_BATCHES = 1250, 26


def optimizer_only(label, path):
    data = json.loads(Path(path).read_text())
    rows = data['repetitions']
    first = rows[0]
    return dict(label=label, commit=data['checkout_head'], prefix_helpers=data.get('prefix_helpers', 0),
                experiments=data.get('experiments', []), samples=data['samples'], decisions=data.get('decisions'),
                seconds=[r['seconds'] for r in rows], timings=first['timings'],
                weights_sha256=first['weights_sha256'], adam_sha256=first['adam_sha256'],
                gradient_sha256=first.get('gradient_sha256'), reference_equal=first.get('reference_equal'),
                difference=first.get('difference'), identical_across_repetitions=data.get('identical_across_repetitions'))


def accepted_stopped_batch(bench, campaign, sequence):
    """Start of PPO in the accepted path's stopped batch: the end of its last game (its own deadline evidence)."""
    run = next((Path(bench) / 'art' / 'runs' / campaign).glob('run-*'))
    folder = next(run.glob(f'batches/train-{sequence:06d}-seat*'))
    execution = folder / 'execution'
    start = json.loads((execution / 'worker-deadline.jsonl').read_text().splitlines()[0])['wall_time']
    ends = []
    for path in sorted(execution.glob('native-deadline-*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ends.append(rows[-1]['wall_time'] - start)
    report = json.loads((execution / 'worker-report.json').read_text())
    return dict(campaign=campaign, sequence=sequence, games_end_seconds_after_worker_start=max(ends),
                attempted_optimizer_steps=report.get('attempted_optimizer_steps'),
                execution_wall_seconds=report.get('execution_wall_seconds'))


def host():
    info = {}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                info['cpu_model'] = line.split(':', 1)[1].strip()
                break
        info['vcpus'] = os.cpu_count()
        mem = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal'))
        info['memory_gib'] = round(int(mem.split()[1]) / 2 ** 20, 1)
    except (OSError, StopIteration):
        pass
    info.update(kernel=platform.release(), python=platform.python_version())
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bench', type=Path, default=Path('/home/claude/bench3'))
    p.add_argument('--campaigns', nargs='+', required=True)
    p.add_argument('--optimizer-only', nargs='*', default=[], metavar='LABEL=PATH')
    p.add_argument('--accepted-stopped', nargs='*', default=['m1-B:266'], metavar='CAMPAIGN:SEQUENCE',
                   help='the accepted stopped batches')
    p.add_argument('--notes', type=Path, help='JSON object merged in as "notes" (commits, checks, limits)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    campaigns = {}
    for name in a.campaigns:
        try:
            campaigns[name] = analysis.campaign(a.bench, name)
        except (OSError, KeyError, ValueError, StopIteration) as error:
            campaigns[name] = dict(error=repr(error))
    per_variant = {}
    for name, c in campaigns.items():
        if 'variant' in c:
            per_variant.setdefault(c['variant'], []).append(name)
    variants = {}
    for variant, names in per_variant.items():
        passed = [b for n in names for b in campaigns[n]['training_batches'] if b['status'] == 'passed']
        evals = [b['seconds'] for n in names for b in campaigns[n]['evaluation_batches'] if b['status'] == 'passed']
        cold = [campaigns[n]['cold_start_seconds'] for n in names if campaigns[n].get('cold_start_seconds')]
        stopped = [dict(campaign=n, sequence=b['sequence'], seconds=b['seconds'], error=b['error'])
                   for n in names for b in campaigns[n]['training_batches'] if b['status'] != 'passed']
        by_sequence = {}
        for b in passed:
            by_sequence.setdefault(b['sequence'], []).append(b['seconds'])
        row = dict(campaigns=names, commits=sorted({campaigns[n]['git_head'] for n in names}),
                   training_batch_seconds=analysis.describe([b['seconds'] for b in passed]),
                   training_batch_seconds_by_sequence={str(k): analysis.describe(v) for k, v in sorted(by_sequence.items())},
                   evaluation_batch_seconds=analysis.describe(evals), cold_start_seconds=analysis.describe(cold),
                   stopped_training_batches=stopped)
        if passed:
            mean_train = statistics.fmean(b['seconds'] for b in passed)
            mean_eval = statistics.fmean(evals) if evals else 0.0
            start = statistics.fmean(cold) if cold else 0.0
            row['verified_updates_per_hour'] = 3600 / mean_train
            row['projection'] = dict(
                cold_start_seconds=start, mean_training_batch_seconds=mean_train, mean_evaluation_batch_seconds=mean_eval,
                hours=(start + UPDATES * mean_train + EVALUATION_BATCHES * mean_eval) / 3600,
                hours_if_every_batch_were_the_fastest=(start + UPDATES * min(b['seconds'] for b in passed)
                                                       + EVALUATION_BATCHES * mean_eval) / 3600,
                hours_if_every_batch_were_the_slowest=(start + UPDATES * max(b['seconds'] for b in passed)
                                                       + EVALUATION_BATCHES * mean_eval) / 3600,
                batches=len(passed), distinct_sequences=sorted(by_sequence))
        variants[variant] = row

    # Paired comparisons, as analyze_campaigns computes them.
    baselines = [n for n, c in campaigns.items() if c.get('variant') == 'B' and 'training_batches' in c]
    pairs, evaluation = [], []
    for name, c in campaigns.items():
        if c.get('variant') in (None, 'B'):
            continue
        block = name.split('-')[0]
        bases = [f'{block}-B'] if f'{block}-B' in baselines else baselines
        for b in c['training_batches']:
            for base in bases:
                for r in campaigns[base]['training_batches']:
                    if (r['sequence'], r['seat']) != (b['sequence'], b['seat']):
                        continue
                    both = r['status'] == b['status'] == 'passed'
                    censored = r['status'] != 'passed' and b['status'] == 'passed'
                    pairs.append(dict(campaign=name, baseline=base, variant=c['variant'], sequence=b['sequence'],
                                      seat=b['seat'], baseline_status=r['status'], candidate_status=b['status'],
                                      baseline_seconds=r['seconds'], candidate_seconds=b['seconds'],
                                      ratio=r['seconds'] / b['seconds'] if both else None,
                                      ratio_lower_bound=r['seconds'] / b['seconds'] if censored else None,
                                      identical_games=analysis.same([r['games'], b['games']])))
        for b in c.get('evaluation_batches', []):
            for base in bases:
                for r in campaigns[base].get('evaluation_batches', []):
                    if r['seat'] == b['seat'] and r['status'] == b['status'] == 'passed':
                        evaluation.append(dict(campaign=name, baseline=base, variant=c['variant'], seat=b['seat'],
                                               baseline_seconds=r['seconds'], candidate_seconds=b['seconds'],
                                               ratio=r['seconds'] / b['seconds'],
                                               identical_games=analysis.same([r.get('games'), b.get('games')])))

    record = dict(schema='original5000-local-acceleration-linux-v1',
                  generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                  host=host(), measurement_scope=('Linux benchmark host, not the Mac. Campaigns: one at a time on CPUs '
                                                  '1-3 from disposable copies of the final 629 package, benchmark seed '
                                                  'namespace, first sequence 266, original hero-management-v2. '
                                                  'Optimizer-only jobs: one at a time on CPU 0.'),
                  projection_method=__doc__.split('Projection for the manual run')[1].strip(),
                  variants=variants, paired_training_batches=pairs, paired_evaluation_batches=evaluation,
                  optimizer_only=[optimizer_only(*item.split('=', 1)) for item in a.optimizer_only],
                  campaigns=campaigns)
    stopped = []
    for item in a.accepted_stopped:
        name, sequence = item.split(':')
        try:
            stopped.append(accepted_stopped_batch(a.bench, name, int(sequence)))
        except (OSError, StopIteration, KeyError, ValueError) as error:
            stopped.append(dict(campaign=name, sequence=int(sequence), error=repr(error)))
    record['accepted_stopped_batches'] = stopped
    if a.notes:
        record['notes'] = json.loads(a.notes.read_text())
    a.out.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(json.dumps({v: {k: r.get(k) for k in ('training_batch_seconds', 'verified_updates_per_hour')}
                      | {'projection_hours': (r.get('projection') or {}).get('hours')} for v, r in variants.items()},
                     indent=1))


if __name__ == '__main__':
    main()
