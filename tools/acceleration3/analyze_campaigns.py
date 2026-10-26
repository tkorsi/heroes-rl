#!/usr/bin/env python3
"""Compact results of the Linux measurement schedule: per-variant times, paired ratios, projection.

    analyze_campaigns.py --bench BENCH_ROOT --campaigns m1-B m1-CH ... [--out FILE]

Reads each campaign's ``bench-run.json`` (launch wall, process-tree CPU, resource peaks),
``summary.json`` (``campaign_summary.py``: batches, game lengths, digests, PPO timings) and
``resources.jsonl`` (memory, descriptors and threads over time). No raw game data is copied.

Projection for the manual run (1,250 training updates and 112 evaluation games, as scheduled by
``--games 5000 --eval-every 100``: eight baseline games, eight every 100 updates, eight final):
  wall = cold start + 1,250 x (mean training batch) + 26 x (mean evaluation batch of one seat)
The batch means come from this host's batches from the 629 parent; the stated range uses the
fastest and slowest measured training batch of each variant. Future batches depend on how long the
policy's games become, which nothing here predicts.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from campaign_summary import same


def ts(value):
    return datetime.fromisoformat(value).timestamp()


def resources(path):
    peak = dict(total_rss=0, total_pss=0, fds=0, threads=0, processes=0)
    first = last = None
    for line in path.open():
        row = json.loads(line)
        peak = {k: max(peak[k], row.get(k, 0) if k != 'processes' else len(row['processes'])) for k in peak}
        top = row['processes'][0] if row['processes'] else None
        if top:
            first = first or top
            last = top
    return dict(peak=peak, supervisor_rss_first=first and first['rss'], supervisor_rss_last=last and last['rss'],
                supervisor_fds_first=first and first['fds'], supervisor_fds_last=last and last['fds'])


def campaign(bench, name):
    folder = Path(bench) / 'art' / 'campaigns' / name
    run = json.loads((folder / 'bench-run.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    batches = summary['campaigns'][name]['batches']
    launch = ts(run['started_utc'])
    training = [b for b in batches if b['mode'] == 'training']
    first = training[0] if training else None
    evals = [b for b in batches if b['mode'] != 'training']
    result = dict(variant=run['variant'], git_head=run['git_head'], status=summary['campaigns'][name]['status'],
                  launch_seconds=run.get('launch_seconds'), child_cpu_seconds=run.get('child_cpu_seconds'),
                  cold_start_seconds=(ts(first['started_utc']) - launch) if first else None,
                  first_checkpoint_seconds=(ts(first['finished_utc']) - launch) if first and first['status'] == 'passed' else None,
                  evaluation_batches=[dict(seat=b['seat'], seconds=b['seconds'], choices=b['choices'], status=b['status'])
                                      for b in evals],
                  training_batches=[dict(sequence=b['sequence'], seat=b['seat'], status=b['status'], error=b.get('error'),
                                         seconds=b['seconds'], decisions=b['training_samples'], choices=b['choices'],
                                         game_seconds=b['game_seconds'],
                                         ppo_seconds=(b.get('ppo') or {}).get('timings_seconds'),
                                         state=b.get('state'), games=b['games']) for b in training])
    path = folder / 'resources.jsonl'
    if path.exists():
        result['resources'] = resources(path)
    return result


def describe(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values), min=values[0], max=values[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bench', type=Path, default=Path('/home/claude/bench3'))
    p.add_argument('--campaigns', nargs='+', required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    campaigns = {}
    for name in a.campaigns:
        try:
            campaigns[name] = campaign(a.bench, name)
        except (OSError, KeyError, ValueError) as error:
            campaigns[name] = dict(error=repr(error))
    variants = {}
    for name, c in campaigns.items():
        if 'variant' in c:
            variants.setdefault(c['variant'], []).append(name)
    per_variant = {}
    for variant, names in variants.items():
        passed = [b for n in names for b in campaigns[n]['training_batches'] if b['status'] == 'passed']
        evals = [b['seconds'] for n in names for b in campaigns[n]['evaluation_batches'] if b['status'] == 'passed']
        cold = [campaigns[n]['cold_start_seconds'] for n in names if campaigns[n]['cold_start_seconds']]
        failed = [dict(campaign=n, sequence=b['sequence'], error=b['error'], seconds=b['seconds'])
                  for n in names for b in campaigns[n]['training_batches'] if b['status'] != 'passed']
        row = dict(campaigns=names, training_batch_seconds=describe([b['seconds'] for b in passed]),
                   evaluation_batch_seconds=describe(evals), cold_start_seconds=describe(cold), failed_training_batches=failed)
        if passed:
            mean_train = statistics.fmean(b['seconds'] for b in passed)
            mean_eval = statistics.fmean(evals) if evals else 0.0
            start = statistics.fmean(cold) if cold else 0.0
            row['verified_updates_per_hour'] = 3600 / mean_train
            row['projection_hours'] = dict(
                mean=(start + 1250 * mean_train + 26 * mean_eval) / 3600,
                fastest_batch=(start + 1250 * min(b['seconds'] for b in passed) + 26 * mean_eval) / 3600,
                slowest_batch=(start + 1250 * max(b['seconds'] for b in passed) + 26 * mean_eval) / 3600)
        per_variant[variant] = row
    # Paired batches: same (sequence, seat) in a baseline and a candidate campaign of the same block.
    pairs = []
    for name, c in campaigns.items():
        if c.get('variant') in (None, 'B'):
            continue
        block = name.split('-')[0]
        base = campaigns.get(f'{block}-B')
        if not base or 'training_batches' not in base:
            continue
        for b in c['training_batches']:
            for r in base['training_batches']:
                if (r['sequence'], r['seat']) == (b['sequence'], b['seat']):
                    pairs.append(dict(block=block, variant=c['variant'], sequence=b['sequence'], seat=b['seat'],
                                      baseline_status=r['status'], candidate_status=b['status'],
                                      baseline_seconds=r['seconds'], candidate_seconds=b['seconds'],
                                      ratio=r['seconds'] / b['seconds'] if b['seconds'] else None,
                                      identical_games=same([r['games'], b['games']]),
                                      identical_state=same([r.get('state'), b.get('state')])))
    result = dict(campaigns=campaigns, per_variant=per_variant, paired_training_batches=pairs)
    text = json.dumps(result, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + '\n')
    print(json.dumps({v: {k: r.get(k) for k in ('training_batch_seconds', 'verified_updates_per_hour', 'projection_hours')}
                      for v, r in per_variant.items()}, indent=1))


if __name__ == '__main__':
    main()
