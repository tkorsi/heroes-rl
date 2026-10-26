#!/usr/bin/env python3
"""Summarize and compare local-runner benchmark campaigns (Linux or Mac), from the runner's own records.

    campaign_summary.py RUN_ROOT [RUN_ROOT ...] [--out FILE] [--no-state]

For each campaign run root (the folder holding ``latest-run.json``): every batch from
``research-log.jsonl`` (intent to result, which includes preparation, the guarded supervisor and
worker, four games with all checks, PPO, checkpoint round trip, preservation and the independent
restore of the preserved copy), the worker's own report (execution wall, per-game choices, wall
and outcome, PPO result), a path-independent digest of every game's decision record (see game_digest), and with Torch a digest of every admitted checkpoint's full state (model, Adam, all RNG,
progress, opponent pool; metadata such as source digests and paths excluded). Batches of different
campaigns with the same (sequence, seat, mode) are paired: identical game digests mean identical
games; identical state digests mean bitwise-identical learner state.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


DECISION_KINDS = ('decision', 'submitted', 'consumed', 'view', 'context', 'terminal')


def game_digest(episode):
    """The learner's decision-relevant record: every decision, submission, observation, context and
    the terminal result, without record numbers. Raw histories and native replays of identical
    games still differ in the engine's start time and in the interleaving of asynchronous notices."""
    paths = sorted(Path(episode).glob('agents/*/history-v8.jsonl'))
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            if row.get('kind') in DECISION_KINDS:
                row.pop('seq', None)
                digest.update(json.dumps(row, sort_keys=True).encode() + b'\n')
    return digest.hexdigest()


def raw_digest(episode):
    episode = Path(episode)
    files = sorted(episode.glob('agents/*/history-v8.jsonl')) + [episode / 'engine' / 'replay.jsonl']
    digest = hashlib.sha256()
    for path in files:
        if not path.exists():
            return None
        digest.update(path.relative_to(episode).as_posix().encode() + b'\0' + bytes.fromhex(sha(path)))
    return digest.hexdigest()


def state_digest(resume_path):
    import torch
    raw = torch.load(resume_path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()

    def feed(value, key=''):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            digest.update(f'{key}|T|{t.dtype}|{tuple(t.shape)}|'.encode())
            flat = t.reshape(-1).contiguous()
            flat = flat.to(torch.uint8) if flat.dtype == torch.bool else flat
            digest.update(bytes(flat.view(torch.uint8).tolist()))  # exact bytes, no NumPy needed
        elif isinstance(value, dict):
            for k in sorted(value, key=str):
                feed(value[k], f'{key}/{k}')
        elif isinstance(value, (list, tuple)):
            digest.update(f'{key}|L{len(value)}|'.encode())
            for i, item in enumerate(value):
                feed(item, f'{key}[{i}]')
        else:
            digest.update(f'{key}|{value!r}|'.encode())

    resume = raw.get('resume', {})
    parts = dict(model=raw['model_state'], optimizer=resume.get('optimizer_state'), rng=raw.get('rng'),
                 progress=resume.get('progress'), pool=raw.get('opponent_pool'),
                 sampler=resume.get('sampler_state'))
    feed(parts)
    return digest.hexdigest()


def batches(run_root, with_state=True):
    run = Path(json.loads((Path(run_root) / 'latest-run.json').read_text())['run_dir'])
    rows = []
    for line in (run / 'research-log.jsonl').open():
        row = json.loads(line)
        if row['event'] != 'batch_result':
            continue
        report = json.loads(Path(row['report']['path']).read_text()) if row.get('report') else {}
        attempts = report.get('attempts', [])
        start, end = (datetime.fromisoformat(row[k]) for k in ('started_utc', 'finished_utc'))
        ppo = report.get('ppo') or {}
        item = dict(sequence=row['sequence'], seat=row['seat'], mode=row['mode'], status=row['status'],
                    error=row.get('error'), seconds=(end - start).total_seconds(),
                    started_utc=row['started_utc'], finished_utc=row['finished_utc'],
                    execution_wall_seconds=report.get('execution_wall_seconds'),
                    choices=[a.get('sampled_choices') for a in attempts],
                    game_seconds=[a.get('wall_seconds') for a in attempts],
                    outcomes=row.get('outcomes'),
                    games=[game_digest(a['episode_directory']) if a.get('episode_directory') else None for a in attempts],
                    raw=[raw_digest(a['episode_directory']) if a.get('episode_directory') else None for a in attempts],
                    optimizer_steps=report.get('optimizer_steps'), training_samples=report.get('training_samples'),
                    ppo={k: v for k, v in ppo.items() if not isinstance(v, (list, dict)) or k.endswith('seconds')},
                    after_weights=(report.get('after_checkpoint') or {}).get('weights_sha256'))
        if with_state and row['mode'] == 'training' and row.get('status') == 'passed':
            item['state'] = state_digest(run / 'checkpoints' / Path(row['plan']['path']).parent.name / 'after-resume.pt')
        rows.append(item)
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return dict(run_root=str(run_root), run_dir=str(run), status=summary.get('status'), batches=rows)


def compare(campaigns):
    keyed = {}
    for name, campaign in campaigns.items():
        for b in campaign['batches']:
            keyed.setdefault((b['sequence'], b['seat'], b['mode']), {})[name] = b
    pairs = []
    for key, by in sorted(keyed.items(), key=lambda kv: (kv[0][2] != 'baseline', kv[0][0], kv[0][1])):
        if len(by) < 2:
            continue
        games = {n: tuple(b['games']) for n, b in by.items()}
        states = {n: b.get('state') for n, b in by.items() if b.get('state')}
        pairs.append(dict(sequence=key[0], seat=key[1], mode=key[2], seconds={n: b['seconds'] for n, b in by.items()},
                          games_identical=len(set(games.values())) == 1,
                          raw_records_identical=len({tuple(b['raw']) for b in by.values()}) == 1,
                          states_identical=(len(set(states.values())) == 1) if len(states) == len(by) else None,
                          choices={n: b['choices'] for n, b in by.items()}))
    return pairs


def describe(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values), min=values[0], max=values[-1],
                mean=statistics.fmean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_roots', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--no-state', action='store_true', help='skip checkpoint state digests (no Torch needed)')
    a = p.parse_args()
    campaigns = {root.name: batches(root, not a.no_state) for root in a.run_roots}
    for campaign in campaigns.values():
        campaign['per_mode'] = {mode: describe([b['seconds'] for b in campaign['batches'] if b['mode'] == mode
                                                and b['status'] == 'passed'])
                                for mode in sorted({b['mode'] for b in campaign['batches']})}
    result = dict(campaigns=campaigns, pairs=compare(campaigns))
    text = json.dumps(result, indent=2, sort_keys=True)
    if a.out:
        a.out.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
