#!/usr/bin/env python3
"""Summarize and compare local-runner benchmark campaigns (Linux or Mac), from the runner's own records.

    campaign_summary.py RUN_ROOT [RUN_ROOT ...] [--out FILE] [--no-state]

For each campaign run root (the folder holding ``latest-run.json``): every batch from
``research-log.jsonl`` (intent to result, which includes preparation, the guarded supervisor and
worker, four games with all checks, PPO, checkpoint round trip, preservation and the independent
restore of the preserved copy), the worker's own report (execution wall, per-game choices, wall
and outcome, PPO result), a path-independent digest of every game's observations and decisions
(see game_digest; read from the episode folder or its compressed archive), and with Torch a digest
of every admitted checkpoint's full state (model, Adam, all RNG, progress, opponent pool, sampler;
metadata such as source digests and paths excluded). Batches of different campaigns with the same
(sequence, seat, mode) are paired: identical game digests mean identical games; identical state
digests mean bitwise-identical learner state; a missing digest makes the pair not comparable (null).
"""
import argparse
from datetime import datetime
import fnmatch
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# Bookkeeping frames of the history (contextual_model._METADATA): issue/cutoff/finalization records
# that carry no game content.
METADATA_KINDS = frozenset(('issued', 'cutoff', 'cutoff_finalized', 'finalized', 'failure', 'superseded'))


def episode_files(episode, pattern):
    """(relative name, bytes) of the episode's files matching pattern, from the folder or its archive."""
    episode = Path(episode)
    if episode.is_dir():
        return [(p.relative_to(episode).as_posix(), p.read_bytes()) for p in sorted(episode.glob(pattern))]
    archive = episode.with_name(episode.name + '.tar.gz')
    if not archive.is_file():
        raise FileNotFoundError(f'episode {episode} has neither its folder nor its archive')
    with tarfile.open(archive, 'r:gz') as tar:
        return sorted((m.name.split('/', 1)[1], tar.extractfile(m).read()) for m in tar
                      if m.isfile() and fnmatch.fnmatchcase(m.name.split('/', 1)[1], pattern))


def game_digest(episode):
    """Every game frame of both agents' histories (all but METADATA_KINDS), without record numbers.
    Raw histories and native replays of identical games still differ in the engine's start time."""
    histories = [h for h in episode_files(episode, 'agents/*/history-v8.jsonl')]
    if not histories:
        raise FileNotFoundError(f'episode {episode} has no agent history')
    digest = hashlib.sha256()
    for name, data in histories:
        digest.update(name.encode() + b'\0')
        for line in data.splitlines():
            row = json.loads(line)
            if row.get('kind') not in METADATA_KINDS:
                row.pop('seq', None)
                digest.update(json.dumps(row, sort_keys=True).encode() + b'\n')
    return digest.hexdigest()


def raw_digest(episode):
    files = episode_files(episode, 'agents/*/history-v8.jsonl') + episode_files(episode, 'engine/replay.jsonl')
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode() + b'\0' + hashlib.sha256(data).digest())
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


def present(digest, episode):
    """A digest of an episode, or None (recorded as not comparable) when its files are gone."""
    try:
        return digest(episode) if episode else None
    except FileNotFoundError:
        return None


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
                    games=[present(game_digest, a.get('episode_directory')) for a in attempts],
                    raw=[present(raw_digest, a.get('episode_directory')) for a in attempts],
                    optimizer_steps=report.get('optimizer_steps'), training_samples=report.get('training_samples'),
                    ppo={k: v for k, v in ppo.items() if not isinstance(v, (list, dict)) or k.endswith('seconds')},
                    after_weights=(report.get('after_checkpoint') or {}).get('weights_sha256'))
        if with_state and row['mode'] == 'training' and row.get('status') == 'passed':
            item['state'] = state_digest(run / 'checkpoints' / Path(row['plan']['path']).parent.name / 'after-resume.pt')
        rows.append(item)
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return dict(run_root=str(run_root), run_dir=str(run), status=summary.get('status'), batches=rows)


def same(values):
    """True or False when every value is present; None (not comparable) when any is missing."""
    values = [tuple(v) if isinstance(v, list) else v for v in values]
    if len(values) < 2 or any(v is None or (isinstance(v, tuple) and (not v or None in v)) for v in values):
        return None
    return len(set(values)) == 1


def compare(campaigns):
    keyed = {}
    for name, campaign in campaigns.items():
        for b in campaign['batches']:
            keyed.setdefault((b['sequence'], b['seat'], b['mode']), {})[name] = b
    pairs = []
    for key, by in sorted(keyed.items(), key=lambda kv: (kv[0][2] != 'baseline', kv[0][0], kv[0][1])):
        if len(by) < 2:
            continue
        pairs.append(dict(sequence=key[0], seat=key[1], mode=key[2], seconds={n: b['seconds'] for n, b in by.items()},
                          statuses={n: b['status'] for n, b in by.items()},
                          games_identical=same(b['games'] for b in by.values()),
                          raw_records_identical=same(b['raw'] for b in by.values()),
                          states_identical=same(b.get('state') for b in by.values()),
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
