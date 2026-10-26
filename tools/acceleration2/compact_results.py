"""Compact, path-free phase-2 results from the private campaign and optimizer records.

    compact_results.py --bench ROOT --campaigns NAME... --ab NAME... --output JSON

Summarizes each campaign with ``summarize_campaigns.summarize``, then reports per workload, push
delay and variant the cold first batch, the steady-state batches (2 to 4) and their medians, the
per-update process-tree CPU seconds, and ratios of medians against the baseline variant B. Paired
batches (same sequence and seat) also get per-batch ratios, whether both played games of the same
decision counts, and whether their verified checkpoints have the same weights digest. Optimizer-only records (``ppo-ab/NAME.json``) are reduced to their
timings and digests, with equality against the B record of the same batch. Every raw input is
bound by SHA-256 under a path relative to ROOT; no absolute path is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_campaigns import summarize  # noqa: E402

DIGESTS = ('weights_sha256', 'model_sha256', 'adam_sha256', 'gradient_sha256', 'generator_sha256')


def relative(path, root):
    path = Path(path)
    return str(path.relative_to(root)) if path.is_relative_to(root) else path.name


def batch_row(b, weights):
    ppo = b.get('ppo_timings') or {}
    peaks = b.get('resource_peaks') or {}
    worker = {k: v for k, v in (peaks.get('by_kind') or {}).items() if 'bench_training' in k}
    return dict(sequence=b['sequence'], seat=b['seat'], status=b['status'], samples=b.get('training_samples'),
                decisions=b.get('decisions'), compute_seconds=b.get('compute_cycle_seconds'),
                durable_seconds=b.get('durable_cycle_seconds'), ppo_seconds=ppo.get('total'),
                ppo_components={k: v for k, v in ppo.items() if k != 'total'},
                prepare_seconds=b.get('prepare_seconds'), worker_start_seconds=b.get('worker_start_seconds'),
                game_seconds=b.get('native_game_seconds'), between_games_seconds=b.get('between_games_seconds'),
                after_last_game_seconds=b.get('after_last_game_to_worker_exit_seconds'),
                publication_seconds=b.get('publication_seconds'),
                peak_summed_rss_bytes=peaks.get('total_rss'), peak_processes=peaks.get('processes'),
                worker_peak=next(iter(worker.values()), None), after_weights_sha256=weights.get((b['sequence'], b['seat'])))


def campaign_row(summary, root):
    report = json.loads((root/'campaigns'/summary['campaign']/'campaign-report.json').read_text())
    weights = {(r['sequence'], r['seat']): ((r.get('private_preservation') or {}).get('checkpoint_restore') or {}).get('weights_sha256')
               for r in report['batches']}
    return dict(name=summary['campaign'], variant=summary['variant'], workload=summary['workload'],
                experiments=summary['experiments'], push_delay_seconds=summary['push_delay'], status=summary['status'],
                checkout_head=summary['checkout_head'], launch_seconds=summary['launch_seconds'],
                process_tree_cpu_seconds=summary['child_cpu_seconds'], batches=[batch_row(b, weights) for b in summary['batches']])


def groups(campaigns):
    out = {}
    for c in campaigns:
        out.setdefault((c['workload'], c['push_delay_seconds'] or 0.0), {}).setdefault(c['variant'], []).append(c)
    return out


def aggregate(runs):
    cold = [b for c in runs for b in c['batches'][:1]]
    steady = [b for c in runs for b in c['batches'][1:]]
    passed = [c for c in runs if c['status'] == 'finished' and all(b['status'] == 'passed' for b in c['batches'])]
    updates = sum(1 for c in runs for b in c['batches'] if b['status'] == 'passed')
    def med(rows, key):
        values = [b[key] for b in rows if b['status'] == 'passed' and b[key] is not None]
        return median(values) if values else None
    return dict(campaigns=[c['name'] for c in runs], verified_updates=updates,
                failed_batches=sum(1 for c in runs for b in c['batches'] if b['status'] != 'passed'),
                all_campaigns_complete=len(passed) == len(runs),
                cold_compute_seconds=[b['compute_seconds'] for b in cold], cold_durable_seconds=[b['durable_seconds'] for b in cold],
                steady_compute_seconds=[b['compute_seconds'] for b in steady], steady_durable_seconds=[b['durable_seconds'] for b in steady],
                steady_median_compute_seconds=med(steady, 'compute_seconds'),
                steady_median_durable_seconds=med(steady, 'durable_seconds'),
                steady_median_ppo_seconds=med(steady, 'ppo_seconds'),
                all_median_compute_seconds=med(cold + steady, 'compute_seconds'),
                all_median_durable_seconds=med(cold + steady, 'durable_seconds'),
                process_tree_cpu_seconds_per_verified_update=(sum(c['process_tree_cpu_seconds'] or 0 for c in runs) / updates
                                                              if updates else None),
                verified_updates_per_hour_steady=(3600 / med(steady, 'durable_seconds') if med(steady, 'durable_seconds') else None))


def ratio(base, other):
    return base / other if base and other else None


def paired(base_runs, runs):
    by_key = {(b['sequence'], b['seat']): b for c in base_runs for b in c['batches']}
    rows = []
    for c in runs:
        for b in c['batches']:
            a = by_key.get((b['sequence'], b['seat']))
            if a is None:
                continue
            rows.append(dict(sequence=b['sequence'], seat=b['seat'], same_decision_counts=a['decisions'] == b['decisions'],
                             same_after_weights=a['after_weights_sha256'] is not None and a['after_weights_sha256'] == b['after_weights_sha256'],
                             statuses=[a['status'], b['status']],
                             compute_ratio=ratio(a['compute_seconds'], b['compute_seconds']) if a['status'] == b['status'] == 'passed' else None,
                             durable_ratio=ratio(a['durable_seconds'], b['durable_seconds']) if a['status'] == b['status'] == 'passed' else None))
    return rows


def optimizer(root, names, bound):
    records = {}
    for name in names:
        path = root/'ppo-ab'/f'{name}.json'
        bound[relative(path, root)] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        records[name] = record
    out = []
    for name, record in records.items():
        batch, variant = name.split('-', 1)
        reference = records.get(f'{batch}-B')
        rows = []
        for r in record['repetitions']:
            rows.append(dict(seconds=r['seconds'], timings=r['timings'], **{k: r[k] for k in DIGESTS},
                             equal_to_baseline=None if reference is None else all(
                                 r[k] == reference['repetitions'][-1][k] for k in (*DIGESTS, 'sampler_order', 'report')),
                             equal_to_archived_reference=r.get('reference_equal'), difference_from_baseline=r.get('difference')))
        out.append(dict(name=name, batch=batch, variant=variant, experiments=record['experiments'],
                        prefix_helpers=record.get('prefix_helpers', 0), samples=record['samples'], frames=record['frames'],
                        decisions=record['decisions'], torch=record['torch'],
                        identical_across_repetitions=record['identical_across_repetitions'], repetitions=rows))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', type=Path, required=True); p.add_argument('--campaigns', nargs='+', required=True)
    p.add_argument('--ab', nargs='*', default=[]); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.bench.resolve(); bound = {}
    campaigns = []
    for name in a.campaigns:
        summary = summarize(root/'campaigns'/name)
        bound.update({relative(k, root): v for k, v in summary['inputs_sha256'].items()})
        campaigns.append(campaign_row(summary, root))
    comparisons = []
    for (workload, delay), variants in sorted(groups(campaigns).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        stats = {v: aggregate(runs) for v, runs in sorted(variants.items())}
        base = stats.get('B')
        ratios = {v: dict(steady_compute=ratio(base['steady_median_compute_seconds'], s['steady_median_compute_seconds']),
                          steady_durable=ratio(base['steady_median_durable_seconds'], s['steady_median_durable_seconds']),
                          cpu_per_update=ratio(base['process_tree_cpu_seconds_per_verified_update'],
                                               s['process_tree_cpu_seconds_per_verified_update']))
                  for v, s in stats.items() if base and v != 'B'}
        pairs = {v: paired(variants['B'], runs) for v, runs in variants.items() if 'B' in variants and v != 'B'}
        comparisons.append(dict(workload=workload, push_delay_seconds=delay, variants=stats,
                                ratio_of_medians_vs_baseline=ratios, paired_batches=pairs))
    result = dict(schema='heroes-training-acceleration-phase2-results-v1',
                  definitions=dict(compute='batch intent to verified preserved checkpoint',
                                   durable='batch intent to confirmed remote acknowledgement',
                                   steady='batches 2 to 4 of each four-batch campaign; batch 1 is the cold start',
                                   ratio='baseline median divided by variant median (above 1 is faster)'),
                  campaigns=campaigns, comparisons=comparisons, optimizer_only=optimizer(root, a.ab, bound),
                  inputs_sha256=dict(sorted(bound.items())))
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    for c in comparisons:
        print(c['workload'], c['push_delay_seconds'], {v: (s['steady_median_compute_seconds'], s['steady_median_durable_seconds'],
              s['verified_updates'], s['failed_batches']) for v, s in c['variants'].items()}, c['ratio_of_medians_vs_baseline'])


if __name__ == '__main__':
    main()
