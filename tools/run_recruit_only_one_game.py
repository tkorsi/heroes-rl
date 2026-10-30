#!/usr/bin/env python3
"""One manual frozen game with trained checkpoint 1753 and recruit-only armies.

The large model, original map and doubled budgets remain unchanged. The explicit
observation/support intervention does not rewrite the checkpoint's provenance.
Check mode starts no game, reserves no seed and performs no optimizer update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from experiments.learner.contextual import OWN_ARMY_INFORMATION
from experiments.learner.interaction_guard import RECRUIT_ONLY_VERSION
from tools import run_selfplay_one_game as evaluation
from tools import run_selfplay_training as campaign
from tools import run_tiny_selfplay_one_game as runtime_validation

require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound
SCHEMA = 'heroes-recruit-only-one-game-v1'
CHECKPOINT = 1753
PARAMETERS = 74344
MODEL_CONFIG = dict(schema='heroes-context-model-v6', hidden_size=64, feature_size=32, max_map_side=256)
BUDGET = evaluation.BUDGET
GUARD = RECRUIT_ONLY_VERSION


def experiment():
    return dict(experiment='trained1753-recruit-only-v1', temporary=True,
                policy_support=GUARD, army_information=OWN_ARMY_INFORMATION,
                checkpoint=CHECKPOINT, model_config=MODEL_CONFIG, parameters=PARAMETERS,
                recruitment='unchanged_native_purchase', whole_hero_garrison='allowed',
                combat='automatic_unchanged', rewards=dict(win=1, loss=-1, ordinary_transition=0),
                training_launcher_accepted=False,
                restoration_reminder='Restore troop-slot management in a separately versioned experiment.')


def source_identity():
    return dict(runtime=evaluation.source_identity(), driver=binding(__file__),
                launcher=binding(ROOT / 'play-one-selfplay.sh'),
                runtime_validator=binding(runtime_validation.__file__))


def frozen_models(plan):
    models = evaluation.frozen_models(plan)
    require(plan['checkpoint'] == CHECKPOINT and plan['model_config'] == MODEL_CONFIG
            and plan['parameters'] == PARAMETERS
            and all(model.config.to_dict() == MODEL_CONFIG
                    and sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
                    for model in models), 'The trained large model architecture changed')
    return models


def inspect_candidate():
    baseline = evaluation.inspect_parent(campaign.LINEAGE)
    require(baseline['checkpoint'] == CHECKPOINT, 'The completed trained checkpoint changed')
    checked = {**baseline, 'source': source_identity(), 'protected': dict(baseline['protected']),
               'model_config': dict(MODEL_CONFIG), 'parameters': PARAMETERS,
               'experiment': experiment(), 'training_launcher_accepted': False}
    # This validator only binds the accepted native runtime. It never loads the
    # retired tiny policy, its preparation receipt or its seed namespace.
    checked.update(runtime_validation.inspect_runtime(baseline))
    checked['protected'].update(checked.pop('runtime_protected'))
    frozen_models(checked)
    campaign.verify_protected(checked['protected'])
    return checked


def prepare(args):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_candidate()
    output = evaluation.copies.private_path(args.output or evaluation.BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    # Share the established large-model ledger, including its consumed attempts.
    reservation = evaluation.reserve(output)
    output.mkdir(parents=True, mode=0o700)
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=GUARD,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=time.monotonic() + args.seconds, hard_deadline_unix=time.time() + args.seconds)
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    require(plan['schema'] == SCHEMA + '-plan' and plan['budget'] == BUDGET
            and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and plan['policy_support'] == GUARD and plan['army_information'] == OWN_ARMY_INFORMATION,
            'Trained recruit-only evaluation contract changed')
    checked = inspect_candidate()
    require(all(plan.get(key) == value for key, value in checked.items()),
            'Frozen trained policy, runtime, source or original-state protection changed')
    reservation = read(bound(plan['reservation']))
    require(reservation in evaluation.seed_ledger()['reservations']
            and reservation['output'] == plan['output'] and plan['schedule'] == reservation['schedule'],
            'Trained evaluation seed binding differs')
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
                'Trained worker requires a bound plan')
        return int(evaluation.worker(args.plan, args.sha256, identity_check=check_identity,
            model_loader=frozen_models, result_schema=SCHEMA, checkpoint=CHECKPOINT)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    started = time.monotonic()
    try:
        if not args.run:
            checked = inspect_candidate()
            result = dict(status='passed', checkpoint=CHECKPOINT, model_checkpoint=checked['model_checkpoint'],
                model_config=MODEL_CONFIG, parameters=PARAMETERS, source=checked['source'], budget=BUDGET,
                policy_support=GUARD, army_information=OWN_ARMY_INFORMATION, experiment=experiment(),
                total_seconds=args.seconds, next_seed=evaluation.seed_row(len(evaluation.seed_ledger()['reservations'])),
                gameplay_performed=False, seeds_reserved=False, optimizer_updates=0,
                exact_model_adam_rng_progress_sampler_pool=True, training_launcher_accepted=False)
        else:
            with campaign.activity_lease():
                plan = prepare(args)
                print('Trained self-play: checkpoint 1753, hidden 64, features 32, 74,344 parameters.\n'
                      'Temporary recruit-only armies: purchases and whole-hero garrison changes remain; '
                      'troop-slot operations and stack upgrades are disabled.\n'
                      'Own-army composition and combat statistics are visible without clicks.\n'
                      'One frozen game; no optimizer update. Red/Blue each have '
                      '9,920 ordinary + 320 reserved choices.', flush=True)
                result = evaluation.supervise(plan, worker_script=Path(__file__).resolve(),
                                              result_schema=SCHEMA, checkpoint=CHECKPOINT)
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
