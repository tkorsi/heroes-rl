"""Compact per-batch timelines and resource peaks of benchmark campaigns.

    summarize_campaigns.py CAMPAIGN_DIR... [--output JSON]

Reads only what each campaign recorded: the controller timeline, the supervisor, worker and
native deadline journals (one system-wide monotonic clock), the execution report with its PPO
timings, the run record and the /proc resource samples. Every input file is bound by SHA-256 in
the output so a compact result can be traced to its private raw evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        rows.setdefault(row['event'], row['monotonic'])
    return rows


def batch_summary(campaign, row, bound):
    folder = campaign/'batches'/f"train-{row['sequence']:04d}-seat{row['seat']}"
    t = row['timeline']
    out = dict(sequence=row['sequence'], seat=row['seat'], status=row['status'],
               training_samples=row.get('training_samples'), optimizer_steps=row.get('optimizer_steps'),
               outcomes=row.get('outcomes'))
    def span(a, b):
        return t[b]-t[a] if a in t and b in t else None
    out['compute_cycle_seconds'] = span('intent', 'verified')
    out['durable_cycle_seconds'] = span('intent', 'acknowledged')
    out['prepare_seconds'] = span('intent', 'prepared')
    out['supervisor_seconds'] = span('supervisor_launch', 'supervisor_exit')
    out['post_supervisor_verification_seconds'] = span('supervisor_exit', 'verified')
    out['publication_seconds'] = span('cleanup_verified', 'acknowledged')
    execution = folder/'execution'
    report_path = execution/'report.json'
    if report_path.exists():
        bound[str(report_path)] = digest(report_path)
        report = json.loads(report_path.read_text())
        out['ppo_timings'] = (report.get('ppo') or {}).get('timings_seconds')
        out['decisions'] = [a.get('sampled_choices') for a in report['attempts']]
        out['attempt_wall_seconds'] = [a.get('wall_seconds') for a in report['attempts']]
        out['execution_wall_seconds'] = report.get('execution_wall_seconds')
    journals = {}
    for name in ('supervisor-deadline.jsonl',):
        if (folder/name).exists(): journals['supervisor'] = events(folder/name)
    if (execution/'worker-deadline.jsonl').exists(): journals['worker'] = events(execution/'worker-deadline.jsonl')
    natives = sorted(execution.glob('native-deadline-*.jsonl')) if execution.exists() else []
    games = [events(p) for p in natives]
    for p in [folder/'supervisor-deadline.jsonl', execution/'worker-deadline.jsonl', *natives]:
        if p.exists(): bound[str(p)] = digest(p)
    if 'worker' in journals and games:
        w, s = journals['worker'], journals.get('supervisor', {})
        out['supervisor_start_seconds'] = w['launch_requested']-s['launch_requested'] if s else None
        out['worker_start_seconds'] = games[0]['launch_requested']-w['launch_requested']
        out['native_game_seconds'] = [g['parent_exited']-g['launch_requested'] for g in games if 'parent_exited' in g]
        out['between_games_seconds'] = [b['launch_requested']-a['parent_exited'] for a, b in zip(games, games[1:]) if 'parent_exited' in a]
        last = games[-1].get('parent_exited')
        if last is not None and 'parent_exited' in w:
            out['after_last_game_to_worker_exit_seconds'] = w['parent_exited']-last
        if s and 'parent_exited' in s and 'parent_exited' in w:
            out['worker_exit_to_supervisor_exit_seconds'] = s['parent_exited']-w['parent_exited']
    return out


def resources(path, rows):
    """Peak RSS, descriptors and threads per batch window and per process kind."""
    peaks = {}
    windows = [(r['sequence'], r['timeline'].get('intent'), r['timeline'].get('acknowledged')) for r in rows]
    for line in Path(path).read_text().splitlines():
        sample = json.loads(line)
        for sequence, start, end in windows:
            if start is not None and end is not None and start <= sample['t'] <= end:
                entry = peaks.setdefault(sequence, dict(total_rss=0, processes=0, by_kind={}))
                entry['total_rss'] = max(entry['total_rss'], sample['total_rss'])
                entry['processes'] = max(entry['processes'], len(sample['processes']))
                for p in sample['processes']:
                    words = p['command'].split(' ')
                    kind = 'vcmi:' + Path(words[0]).name if 'vcmi' in words[0] else (
                        'python:' + next((w for w in words[1:] if not w.startswith('-')), '?').split('/')[-1])
                    k = entry['by_kind'].setdefault(kind, dict(rss=0, fds=0, threads=0))
                    for key in k: k[key] = max(k[key], p[key])
    return peaks


def summarize(campaign):
    campaign = Path(campaign).resolve(); bound = {}
    report_path = campaign/'campaign-report.json'; bound[str(report_path)] = digest(report_path)
    report = json.loads(report_path.read_text())
    context = json.loads((campaign/'context.json').read_text()); bound[str(campaign/'context.json')] = digest(campaign/'context.json')
    run = json.loads((campaign/'bench-run.json').read_text()) if (campaign/'bench-run.json').exists() else {}
    if run: bound[str(campaign/'bench-run.json')] = digest(campaign/'bench-run.json')
    rows = report['batches']
    result = dict(campaign=campaign.name, variant=context.get('variant'), workload=context.get('workload'),
                  experiments=context.get('experiments', []), checkout_head=run.get('git_head'), runtime=run.get('runtime'),
                  push_delay=run.get('push_delay'), status=report.get('status'),
                  launch_seconds=run.get('launch_seconds'), child_cpu_seconds=run.get('child_cpu_seconds'),
                  batches=[batch_summary(campaign, r, bound) for r in rows])
    if (campaign/'resources.jsonl').exists():
        bound[str(campaign/'resources.jsonl')] = digest(campaign/'resources.jsonl')
        peaks = resources(campaign/'resources.jsonl', rows)
        for b in result['batches']: b['resource_peaks'] = peaks.get(b['sequence'])
    result['inputs_sha256'] = bound
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('campaigns', nargs='+', type=Path); p.add_argument('--output', type=Path)
    a = p.parse_args()
    results = [summarize(c) for c in a.campaigns]
    text = json.dumps(results, indent=1) + '\n'
    if a.output: a.output.write_text(text)
    for r in results:
        print(r['campaign'], r['variant'], r['workload'], r['status'])
        for b in r['batches']:
            ppo = (b.get('ppo_timings') or {}).get('total')
            print('  seq', b['sequence'], 'seat', b['seat'], b['status'], 'samples', b['training_samples'],
                  'compute %.1f' % (b['compute_cycle_seconds'] or -1), 'durable %.1f' % (b['durable_cycle_seconds'] or -1),
                  'ppo %.1f' % (ppo or -1), 'prepare %.1f' % (b['prepare_seconds'] or -1),
                  'worker_start %.1f' % (b.get('worker_start_seconds') or -1),
                  'games', [round(x, 1) for x in b.get('native_game_seconds', [])],
                  'gaps', [round(x, 1) for x in b.get('between_games_seconds', [])],
                  'publish %.1f' % (b['publication_seconds'] or -1))


if __name__ == '__main__':
    main()
