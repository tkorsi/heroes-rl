#!/usr/bin/env python3
"""Bounded uniform-random native validation; no learned model or training.

Fold only newly delivered request frames during play. Audit the complete persisted
histories once at the end, avoiding repeated serialization of every old snapshot.
This is correctness/outcome validation, not a simulation throughput benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/headless'))
from env import HeadlessSelfPlayEnv, VcmiLauncher, Limits
from experiments.learner import contextual as contract
from validation.probes.audit_contextual_histories import audit
from validation.probes.validate_contextual_native import random_action


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def run(args, index):
    directory = args.output / f'case-{index + 1:03d}'
    directory.mkdir()
    seeds = {'engine': args.seed + index, 'red': args.red_seed + index, 'blue': args.blue_seed + index}
    rng = {color: random.Random(seeds[color]) for color in ('red', 'blue')}
    env = HeadlessSelfPlayEnv(VcmiLauncher(args.binary, args.data, args.map, 'info'), directory,
        Limits(max_days=args.max_days, max_actions=args.max_actions, watchdog_seconds=30,
               startup_seconds=30, shutdown_seconds=30, bridge_timeout=30),
        interface=contract.ENVIRONMENT, ledger=True, replay=True,
        agents={color: {'kind': 'uniform-contextual-validation', 'policy_seed': seeds[color],
                       'training': False} for color in rng})
    seen = {player: contract.PlayerHistory('1', player) for player in (0, 1)}
    started, count, error, state, audited = time.monotonic(), 0, None, None, None
    try:
        state = env.reset(1, seeds['engine'])
        with (directory / 'choices.jsonl').open('x') as choices:
            while not state.done:
                if time.monotonic() - started > args.wall_seconds:
                    os.kill(env.process.pid, signal.SIGKILL)
                    env.process.wait(timeout=10)
                    state = env.truncate('collection_window')
                    raise RuntimeError('Native validation wall watchdog expired; unscored failure')
                if state.reward != 0 or state.player not in seen:
                    raise ValueError('Invalid ordinary request reward/player')
                for raw in state.frames:
                    seen[state.player].append(contract.decode_frame(raw))
                decision = seen[state.player].frames[-1]
                if decision.kind != 'decision' or decision != seen[state.player].decision:
                    raise ValueError('Request does not end at its actual own decision')
                action = random_action(decision, rng[state.color], contract)
                choices.write(json.dumps({'player': state.player, 'decision_seq': decision.seq,
                    'day': decision.day, 'action': asdict(action), 'training_input': False}, separators=(',', ':')) + '\n')
                count += 1
                state = env.step(action)
        manifest_path = env.episode_dir / 'manifest.json'
        paths = [env.episode_dir / 'agents' / color / 'history-v2.jsonl' for color in ('red', 'blue')]
        for player, path in enumerate(paths):
            frames = [json.loads(line) for line in path.read_text().splitlines()]
            prefix = [contract.encode_frame(frame) for frame in seen[player].frames]
            if frames[:len(prefix)] != prefix:
                raise ValueError('Persisted history changed a previously delivered prefix')
        audited = audit(*paths, manifest_path)
        write(directory / 'history-audit.json', audited)
        if not audited['consistent'] or not audited['manifest_bound']:
            raise ValueError('Independent persisted history/manifest audit failed')
        if state.status == 'completed' and (not audited['scoring_binding_verified'] or set(state.rewards.values()) != {-1, 1}):
            raise ValueError('Natural outcome lacks complementary independently bound rewards')
        if state.status != 'completed' and any(value is not None for value in state.rewards.values()):
            raise ValueError('Unscored boundary has a reward')
        if state.status == 'failed' and (audited['learner_eligible'] or any(state.bootstrap.values())):
            raise ValueError('Failed native run became learner eligible')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if state is not None and not state.done:
            try: state = env.truncate('collection_window')
            except Exception as cleanup: error += f'; finalization: {cleanup}'
    finally:
        env.close()
    row = {'case': index + 1, 'seeds': seeds, 'choices': count, 'seconds': round(time.monotonic()-started, 3),
        'status': state.status if state is not None and state.done else None,
        'termination_reason': state.termination_reason if state is not None and state.done else None,
        'outcomes': state.outcomes if state is not None and state.done else None,
        'rewards': state.rewards if state is not None and state.done else None,
        'audit_passed': bool(audited and audited['consistent'] and audited['manifest_bound']),
        'error': error, 'manifest': str(env.episode_dir / 'manifest.json') if env.episode_dir else None,
        'training_performed': False}
    write(directory / 'report.json', row)
    print(json.dumps(row), flush=True)
    return row


def cohort_report(rows):
    """Keep native outcomes separate from probe exceptions; the counts can overlap."""
    statuses = Counter({status: 0 for status in ('completed', 'truncated', 'failed', 'unfinalized')})
    reasons = Counter()
    natural = cutoffs = failed = 0
    for row in rows:
        statuses[row['status'] or 'unfinalized'] += 1
        reasons[row['termination_reason'] or 'unfinalized'] += 1
        verified = row['audit_passed'] is True and row['error'] is None
        natural += verified and row['status'] == 'completed'
        cutoffs += verified and row['status'] == 'truncated'
        failed += not verified or row['status'] not in ('completed', 'truncated')
    return {'schema': 'heroes-contextual-random-native-validation-v1', 'episodes': rows,
        'native_status_counts': dict(sorted(statuses.items())),
        'termination_reason_counts': dict(sorted(reasons.items())),
        'natural_completions': natural, 'supported_cutoffs': cutoffs,
        'native_failures': statuses['failed'],
        'probe_exceptions': sum(row['error'] is not None for row in rows),
        'failed_cases': failed,
        'training_performed': False, 'throughput_benchmark': False}


def cohort_exit_status(report):
    return int(report['failed_cases'] > 0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binary', type=Path, required=True)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--map', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--episodes', type=int, default=8)
    cli.add_argument('--seed', type=int, default=86001)
    cli.add_argument('--red-seed', type=int, default=92001)
    cli.add_argument('--blue-seed', type=int, default=93001)
    cli.add_argument('--max-days', type=int, default=120)
    cli.add_argument('--max-actions', type=int, default=1500)
    cli.add_argument('--wall-seconds', type=float, default=90)
    args = cli.parse_args()
    if not 1 <= args.episodes <= 20 or min(args.max_days, args.max_actions, args.wall_seconds) <= 0:
        cli.error('Use 1..20 episodes and positive bounds')
    for name in ('binary','data','map','output'): setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output / 'plan.json', {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'policy':'uniform available action type and uniform independent legal parameter domains; windows close only',
        'training_performed':False})
    rows = [run(args,index) for index in range(args.episodes)]
    report = cohort_report(rows)
    write(args.output / 'validation.json',report)
    return cohort_exit_status(report)


if __name__ == '__main__':
    raise SystemExit(main())
