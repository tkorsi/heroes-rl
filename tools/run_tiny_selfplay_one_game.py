#!/usr/bin/env python3
"""One manually launched frozen self-play diagnostic with the untrained 8/4 model.

Default check mode launches nothing. The existing trained policy is protected as
a comparison baseline, never used as the tiny model's parent or initialization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import prepare_tiny_policy as candidate
from tools import run_selfplay_one_game as evaluation
from tools import run_selfplay_training as campaign

require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound
SCHEMA = 'heroes-tiny-one-game-v1'
BASE = campaign.ARTIFACTS / 'selfplay-tiny-one-game'
PREPARATION = campaign.ARTIFACTS / 'tiny-policy-experiments/tiny-v1-initial/preparation.json'
PREPARATION_SHA = 'a9f041cd982df051b68716121d65965df26fc4801440303725ad7acff5b933e2'
POLICY_SHA = '0eaea769ae11ecf4e6c5c02c056acd1c11e94c3939f17b4dbf0943d2271e4ffb'
WEIGHTS_SHA = '55d3bdd61162ae82fa96be7f84baedec77cd1d5df45a38d44ad0bc791c360cca'
CHECKPOINT = 'tiny-v1-untrained-initial'
ENGINE_SEED, POLICY_SEED = 1_974_000_000, 1_976_000_000
BUDGET = evaluation.BUDGET


def source_identity():
    return dict(runtime=evaluation.source_identity(), driver=binding(__file__),
                launcher=binding(ROOT / 'play-one-tiny-selfplay.sh'),
                preparer=binding(candidate.__file__), configuration=binding(candidate.CONFIG))


def expected_metadata():
    spec, _ = candidate.specification()
    return dict(experiment=spec['experiment'], purpose='untrained_architecture_candidate',
                initialized_from_checkpoint=None, configuration=binding(candidate.CONFIG),
                training_games=0, optimizer_updates=0, training_launcher_accepted=False)


def frozen_models(plan):
    from experiments.learner import checkpoints as cp
    _, config = candidate.specification()
    require(plan['tiny_policy']['sha256'] == POLICY_SHA and plan['weights_sha256'] == WEIGHTS_SHA
            and plan['model_checkpoint'] == CHECKPOINT and plan['tiny_metadata'] == expected_metadata(),
            'Tiny initialization identity changed')
    loaded = tuple(cp.load_policy_checkpoint(bound(plan['tiny_policy']),
        expected_source_metadata=expected_metadata(), allowed_provenance=('random_initialization',),
        interface=campaign.INTERFACE) for _ in range(2))
    for item in loaded:
        require(item.provenance == 'random_initialization' and item.checkpoint == CHECKPOINT
                and item.weights_sha256 == WEIGHTS_SHA and item.model.config == config
                and sum(p.numel() for p in item.model.parameters()) == 15460
                and item.opponent_pool is not None
                and item.opponent_pool.identities == ((CHECKPOINT, WEIGHTS_SHA),),
                'Tiny model shape, weights, provenance or initial pool differs')
        item.opponent_pool.assert_unchanged()
    models = tuple(item.model.eval().requires_grad_(False) for item in loaded)
    require(not ({p.data_ptr() for p in models[0].parameters()} & {p.data_ptr() for p in models[1].parameters()}),
            'Tiny players must have independent parameter storage')
    return models


def inspect_candidate():
    baseline = evaluation.inspect_parent(campaign.LINEAGE)
    receipt = binding(PREPARATION)
    require(receipt['sha256'] == PREPARATION_SHA, 'Tiny preparation receipt changed')
    prepared = read(PREPARATION)
    require(prepared['status'] == 'prepared_untrained' and prepared['progress'] == dict(updates=0, rollouts=0, decisions=0)
            and prepared['policy']['sha256'] == POLICY_SHA and prepared['weights_sha256'] == WEIGHTS_SHA
            and prepared['exact_model_rng_pool_restore'] is True and prepared['training_launcher_accepted'] is False,
            'Tiny candidate is not the preserved untrained initialization')
    protected = {**baseline['protected'], **prepared['protected']}
    for value in (receipt, prepared['policy'], binding(candidate.CONFIG)):
        protected[value['path']] = value
    checked = {key: value for key, value in baseline.items()
               if key not in ('source', 'parent', 'checkpoint', 'model_checkpoint', 'weights_sha256', 'optimizer_steps', 'protected')}
    checked.update(source=source_identity(), comparison_baseline=baseline['parent'],
        checkpoint=CHECKPOINT, model_checkpoint=CHECKPOINT, weights_sha256=WEIGHTS_SHA,
        tiny_policy=prepared['policy'], tiny_metadata=expected_metadata(), preparation=receipt,
        protected=protected, initialization='random_initialization', initialized_from_checkpoint=None,
        model_config=prepared['config'], parameters=15460, optimizer_state='not created',
        training_launcher_accepted=False)
    frozen_models(checked)
    campaign.verify_protected(protected)
    return checked


def seed_row(offset):
    require(type(offset) is int and 0 <= offset < 100000, 'Tiny evaluation seed namespace exhausted')
    return dict(episode=1, engine_seed=ENGINE_SEED + offset,
                policy_seeds=[POLICY_SEED + 2 * offset, POLICY_SEED + 2 * offset + 1])


def seed_ledger():
    path = BASE / 'seed-reservations.json'
    if not path.exists():
        require(not BASE.exists() or not any(p.name != 'namespace.lock' for p in BASE.iterdir()),
                'Tiny seed ledger is missing while evidence remains; refusing seed reuse')
        return dict(schema=SCHEMA + '-seeds', reservations=[])
    ledger = read(path)
    require(ledger.get('schema') == SCHEMA + '-seeds' and type(ledger.get('reservations')) is list,
            'Invalid tiny evaluation seed ledger')
    outputs = set()
    for index, row in enumerate(ledger['reservations']):
        require(row.get('offset') == index and row.get('schedule') == seed_row(index)
                and row.get('consumed_including_unstarted') is True and row['output'] not in outputs,
                'Tiny evaluation reservation changed')
        require(read(BASE / f'reserved-{index:06d}.json') == row, 'Tiny reservation receipt changed or missing')
        outputs.add(row['output'])
    require({p.name for p in BASE.glob('reserved-*.json')}
            == {f'reserved-{i:06d}.json' for i in range(len(outputs))}, 'Unmatched tiny reservation history')
    return ledger


def reserve(output):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'namespace.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = seed_ledger()
        require(all(row['output'] != str(output) for row in ledger['reservations']), 'Tiny output already consumed seeds')
        index = len(ledger['reservations'])
        row = dict(offset=index, output=str(output), schedule=seed_row(index), consumed_including_unstarted=True)
        ledger['reservations'].append(row)
        write(BASE / 'seed-reservations.json', ledger)
        receipt = BASE / f'reserved-{index:06d}.json'
        write(receipt, row, new=True)
        return binding(receipt)


def prepare(args):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_candidate()
    output = evaluation.copies.private_path(args.output or BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    reservation = reserve(output)
    output.mkdir(parents=True, mode=0o700)
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=campaign.GUARD,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=time.monotonic() + args.seconds, hard_deadline_unix=time.time() + args.seconds)
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    require(plan['schema'] == SCHEMA + '-plan' and plan['source'] == source_identity(), 'Frozen tiny source changed')
    require(plan['budget'] == BUDGET and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and plan['policy_support'] == campaign.GUARD and plan['checkpoint'] == CHECKPOINT
            and plan['initialization'] == 'random_initialization' and plan['initialized_from_checkpoint'] is None
            and plan['training_launcher_accepted'] is False and plan['parameters'] == 15460
            and plan['model_config'] == candidate.specification()[1].to_dict()
            and plan['optimizer_state'] == 'not created', 'Tiny evaluation contract changed')
    require(runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Tiny evaluation runtime changed')
    bound(plan['replay_binary']); bound(plan['native_manifest'])
    require(bound(plan['preparation']) == PREPARATION.resolve() and plan['preparation']['sha256'] == PREPARATION_SHA,
            'Tiny preparation binding changed')
    bound(plan['tiny_policy'])
    reservation = read(bound(plan['reservation']))
    require(reservation in seed_ledger()['reservations'] and plan['schedule'] == reservation['schedule'],
            'Tiny evaluation seed binding differs')
    campaign.verify_protected(plan['protected'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('command', nargs='?', choices=('_worker',))
    parser.add_argument('--plan', type=Path); parser.add_argument('--sha256')
    args = parser.parse_args(argv)
    if args.command == '_worker':
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check,
                'Tiny worker requires a bound plan')
        return int(evaluation.worker(args.plan, args.sha256, identity_check=check_identity,
            model_loader=frozen_models, result_schema=SCHEMA, checkpoint=CHECKPOINT)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    started = time.monotonic()
    try:
        if not args.run:
            checked = inspect_candidate()
            result = dict(status='passed', checkpoint=CHECKPOINT, initialization='random_initialization',
                model_config=checked['model_config'], parameters=15460, source=checked['source'], budget=BUDGET,
                total_seconds=args.seconds, next_seed=seed_row(len(seed_ledger()['reservations'])),
                gameplay_performed=False, seeds_reserved=False, optimizer_updates=0,
                exact_tiny_policy_and_pool=True, comparison_baseline_1753_full_state_restored=True,
                training_launcher_accepted=False)
        else:
            with campaign.activity_lease():
                plan = prepare(args)
                print('Tiny self-play: hidden 8, features 4, 15,460 parameters; untrained frozen weights.\n'
                      'One game; no optimizer update. Red/Blue each have 9,920 ordinary + 320 reserved choices.', flush=True)
                result = evaluation.supervise(plan, worker_script=Path(__file__).resolve(),
                                              result_schema=SCHEMA, checkpoint=CHECKPOINT)
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
