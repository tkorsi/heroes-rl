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
GUARD = 'visible-interaction-budget-recruit-only-v1'
EXPERIMENT_CONFIG = ROOT / 'configs/tiny-recruit-only-v1.json'
RUNTIME = campaign.ARTIFACTS / 'runtime/local-v8-passive-armies-v1'
RUNTIME_FREEZE_SHA = 'PENDING_REVIEW'
RUNTIME_PREFLIGHT_SHA = 'PENDING_REVIEW'
REQUIRED_NATIVE_VALIDATION = ROOT / 'validation/passive-armies-native-2026-10-30.json'
REQUIRED_NATIVE_VALIDATION_SHA = 'PENDING_REVIEW'


def experiment():
    from experiments.learner.contextual import OWN_ARMY_INFORMATION
    from experiments.learner.interaction_guard import RECRUIT_ONLY_VERSION, RECRUIT_ONLY_BLOCKED_KINDS
    spec = read(EXPERIMENT_CONFIG)
    require(GUARD == RECRUIT_ONLY_VERSION and spec == dict(
        schema='heroes-tiny-recruit-only-experiment-v1', experiment='tiny-recruit-only-v1',
        initial_policy=CHECKPOINT, policy_support=GUARD, temporary=True,
        army_information=OWN_ARMY_INFORMATION,
        blocked_actions=['open_stack', 'split_stack', 'merge_stacks', 'swap_stacks',
                         'transfer_stack', 'select_upgrade', 'upgrade_stack'],
        recruitment='unchanged_native_purchase', combat='automatic_unchanged',
        rewards=dict(win=1, loss=-1, ordinary_transition=0),
        restoration_reminder='Restore manual army arrangement and stack inspection in a separately versioned experiment.',
        training_launcher_accepted=False) and set(spec['blocked_actions']) == RECRUIT_ONLY_BLOCKED_KINDS,
        'Temporary tiny army restriction contract changed')
    return spec


def source_identity():
    return dict(runtime=evaluation.source_identity(), driver=binding(__file__),
                launcher=binding(ROOT / 'play-one-tiny-selfplay.sh'),
                preparer=binding(candidate.__file__), configuration=binding(candidate.CONFIG),
                experiment_configuration=binding(EXPERIMENT_CONFIG))


def inspect_runtime(baseline):
    """Bind the new observation runtime without changing the comparison baseline."""
    from experiments.episodes.scoring import engine_recipe_digests
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.learner.contextual import OWN_ARMY_INFORMATION
    from tools.freeze_local_runtime import ARTIFACTS as REQUIRED_ARTIFACTS
    directory = RUNTIME.resolve()
    require(directory == RUNTIME and not RUNTIME.is_symlink(), 'Passive-army runtime path changed')
    protected = {}
    def keep(path, expected_sha=None, expected_bytes=None):
        value = binding(path)
        require(expected_sha is None or value['sha256'] == expected_sha, 'Passive-army runtime binding changed')
        require(expected_bytes is None or value['bytes'] == expected_bytes, 'Passive-army runtime file size changed')
        protected[value['path']] = value
        return value
    freeze_binding = keep(directory / 'freeze-receipt.json', RUNTIME_FREEZE_SHA)
    preflight_binding = keep(directory / 'preflight.json', RUNTIME_PREFLIGHT_SHA)
    freeze, preflight = read(bound(freeze_binding)), read(bound(preflight_binding))
    binary_dir = directory / 'bin'
    require(freeze.get('schema') == 'heroes-local-runtime-freeze-v2'
            and freeze.get('source_recipe_check') is True and freeze.get('source_dirty') is False
            and freeze.get('donor_load_references_removed') is True and freeze.get('old_runtime_modified') is False
            and freeze.get('gameplay_started') is False and freeze.get('training_started') is False
            and freeze.get('frozen_bin') == str(binary_dir), 'Passive-army runtime freeze is incomplete')
    recipe = engine_recipe_digests(ROOT)
    require(recipe and all(freeze['source_inputs'].get(name, {}).get('sha256') == digest
                           for name, digest in recipe.items()), 'Passive-army engine recipe differs from its frozen build')
    names = [item['path'] for item in freeze['artifacts']]
    require(len(names) == len(set(names)) and set(names) == set(REQUIRED_ARTIFACTS),
            'Passive-army frozen binary inventory changed')
    for item in freeze['artifacts']:
        require(item.get('ad_hoc_signature_verified') is True, 'Frozen native signature was not verified')
        keep(binary_dir / item['path'], item['frozen_sha256'], item['bytes'])
    actual_resources = set()
    for name in ('config', 'Mods'):
        folder = binary_dir / name
        require(folder.is_dir() and not folder.is_symlink(), 'Passive-army resource directory changed')
        for path in folder.rglob('*'):
            require(not path.is_symlink(), 'Passive-army runtime resource symlink introduced')
            if path.is_file(): actual_resources.add(path.relative_to(binary_dir).as_posix())
    resources = [item['path'] for item in freeze['resources']]
    require(len(resources) == len(set(resources)) and set(resources) == actual_resources,
            'Passive-army resource inventory changed')
    for item in freeze['resources']:
        keep(binary_dir / item['path'], item['sha256'], item['bytes'])
    for item in freeze['external_dependencies']:
        require(str(Path(item['path']).resolve()) == item['resolved'], 'Frozen external dependency target changed')
        keep(item['path'], item['sha256'], item['bytes'])
    dependency = freeze['dependency_check']
    keep(directory / 'dependency-check.stdout', dependency['sha256'], dependency['bytes'])
    game_map = keep(freeze['map']['path'], freeze['map']['sha256'], freeze['map']['bytes'])
    require(Path(game_map['path']).is_relative_to(directory / 'maps')
            and game_map['sha256'] == binding(baseline['game_map'])['sha256'], 'Passive-army runtime changed the original map')
    runtime = runtime_identity(binary_dir)
    native = preflight.get('native', {})
    require(preflight.get('schema') == 'heroes-local-runtime-preflight-v2'
            and preflight.get('launcher_preflight_passed') is True and preflight.get('client_help_exit_code') == 0
            and preflight.get('donor_load_detected') is False and preflight.get('gameplay_started') is False
            and preflight.get('training_started') is False and preflight.get('runtime') == runtime
            and native.get('binary') == str(binary_dir / 'vcmiclient')
            and native.get('binary_sha256') == binding(binary_dir / 'vcmiclient')['sha256']
            and native.get('explorer_rl_sha256') == runtime['AI/libExplorerRL.dylib']
            and native.get('map', {}).get('sha256') == game_map['sha256']
            and preflight.get('map') == freeze['map']
            and any(str(binary_dir / 'libvcmi.dylib') in line for line in preflight.get('actual_client_libvcmi_load', [])),
            'Passive-army actual runtime loader preflight differs')
    acceptance_binding = keep(REQUIRED_NATIVE_VALIDATION, REQUIRED_NATIVE_VALIDATION_SHA)
    acceptance = read(bound(acceptance_binding))
    require(acceptance.get('schema') == 'heroes-passive-armies-native-v1' and acceptance.get('status') == 'passed'
            and acceptance.get('army_information') == OWN_ARMY_INFORMATION and acceptance.get('policy_support') == GUARD
            and acceptance.get('runtime') == runtime
            and acceptance.get('source_implementation') == evaluation.source_identity()['implementation']
            and acceptance.get('players') == [0, 1] and type(acceptance.get('controls')) is int and acceptance['controls'] > 0
            and acceptance.get('native_replay_verified') is True and acceptance.get('both_policies_reconstructed') is True
            and acceptance.get('owned_cleanup_verified') is True and acceptance.get('optimizer_updates') == 0
            and acceptance.get('training_samples_admitted') == 0,
            'Passive-army runtime lacks complete two-seat native validation')
    return dict(binary=str(binary_dir / 'vcmiclient'), replay_binary=binding(binary_dir / 'vcmireplay'),
                game_map=game_map['path'], runtime=runtime, native=native, army_information=OWN_ARMY_INFORMATION,
                runtime_origin=dict(freeze=freeze_binding, preflight=preflight_binding, native_validation=acceptance_binding),
                runtime_protected=protected)


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
    spec = experiment()
    baseline = evaluation.inspect_parent(campaign.LINEAGE)
    receipt = binding(PREPARATION)
    require(receipt['sha256'] == PREPARATION_SHA, 'Tiny preparation receipt changed')
    prepared = read(PREPARATION)
    require(prepared['status'] == 'prepared_untrained' and prepared['progress'] == dict(updates=0, rollouts=0, decisions=0)
            and prepared['policy']['sha256'] == POLICY_SHA and prepared['weights_sha256'] == WEIGHTS_SHA
            and prepared['exact_model_rng_pool_restore'] is True and prepared['training_launcher_accepted'] is False,
            'Tiny candidate is not the preserved untrained initialization')
    protected = {**baseline['protected'], **prepared['protected']}
    for value in (receipt, prepared['policy'], binding(candidate.CONFIG), binding(EXPERIMENT_CONFIG)):
        protected[value['path']] = value
    checked = {key: value for key, value in baseline.items()
               if key not in ('source', 'parent', 'checkpoint', 'model_checkpoint', 'weights_sha256', 'optimizer_steps', 'protected')}
    checked.update(source=source_identity(), comparison_baseline=baseline['parent'],
        checkpoint=CHECKPOINT, model_checkpoint=CHECKPOINT, weights_sha256=WEIGHTS_SHA,
        tiny_policy=prepared['policy'], tiny_metadata=expected_metadata(), preparation=receipt,
        protected=protected, initialization='random_initialization', initialized_from_checkpoint=None,
        model_config=prepared['config'], parameters=15460, optimizer_state='not created',
        training_launcher_accepted=False, experiment=spec,
        experiment_configuration=binding(EXPERIMENT_CONFIG), runtime_comparison_map=binding(baseline['game_map']))
    checked.update(inspect_runtime(baseline))
    checked['protected'].update(checked.pop('runtime_protected'))
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
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=GUARD,
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
            and plan['policy_support'] == GUARD and plan['checkpoint'] == CHECKPOINT
            and plan['initialization'] == 'random_initialization' and plan['initialized_from_checkpoint'] is None
            and plan['training_launcher_accepted'] is False and plan['parameters'] == 15460
            and plan['model_config'] == candidate.specification()[1].to_dict()
            and plan['optimizer_state'] == 'not created', 'Tiny evaluation contract changed')
    require(plan['experiment'] == experiment()
            and plan['experiment_configuration'] == binding(EXPERIMENT_CONFIG),
            'Temporary tiny army restriction binding changed')
    runtime = inspect_runtime(dict(game_map=str(bound(plan['runtime_comparison_map']))))
    require(all(plan.get(key) == value for key, value in runtime.items() if key != 'runtime_protected')
            and all(plan['protected'].get(key) == value for key, value in runtime['runtime_protected'].items()),
            'Passive-army runtime selection or protection changed')
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
                policy_support=GUARD, experiment=experiment(), army_information=experiment()['army_information'],
                total_seconds=args.seconds, next_seed=seed_row(len(seed_ledger()['reservations'])),
                gameplay_performed=False, seeds_reserved=False, optimizer_updates=0,
                exact_tiny_policy_and_pool=True, comparison_baseline_1753_full_state_restored=True,
                training_launcher_accepted=False)
        else:
            with campaign.activity_lease():
                plan = prepare(args)
                print('Tiny self-play: hidden 8, features 4, 15,460 parameters; untrained frozen weights.\n'
                      'Temporary recruit-only armies: no stack inspection, upgrades, rearrangement or transfers.\n'
                      'Own army composition and combat statistics are delivered without opening stack windows.\n'
                      'One game; no optimizer update. Red/Blue each have 9,920 ordinary + 320 reserved choices.', flush=True)
                result = evaluation.supervise(plan, worker_script=Path(__file__).resolve(),
                                              result_schema=SCHEMA, checkpoint=CHECKPOINT)
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
