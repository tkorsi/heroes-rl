#!/usr/bin/env python3
"""Prepare an untrained small policy separately; never launch games or training.

Dimensions change only. The existing visible encoder, memory, typed action
heads, support restrictions and learner implementation remain unchanged.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/selfplay-tiny-v1.json'


def specification(path=CONFIG):
    from experiments.learner import contextual as c, contextual_model as m
    from tools import run_selfplay_training as campaign
    spec = campaign.read(path)
    campaign.require(spec['schema'] == 'heroes-tiny-policy-experiment-v1'
        and spec['initialization'] == 'fresh_random_weights' and spec['parent_checkpoint'] is None
        and spec['training_launcher_accepted'] is False, 'Tiny model requires an independent untrained experiment')
    campaign.require(spec['interface'] == c.EXPANDED_ENVIRONMENT and spec['policy_support'] == campaign.GUARD
        and spec['rewards'] == dict(win=1, loss=-1, ordinary_transition=0)
        and spec['training_games_per_update'] == 4, 'Existing learning contract must remain explicit')
    config = m.ModelConfig(schema=m.model_schema_for(spec['interface']), **spec['model'])
    campaign.require(config.hidden_size == 8 and config.feature_size == 4 and config.max_map_side == 256
        and spec['expected_parameters'] == 15460, 'Unknown tiny candidate')
    return spec, config


def prepare(output):
    import torch
    from experiments.learner import checkpoints as cp, contextual_model as m
    from tools import run_selfplay_training as campaign
    spec, config = specification()
    output = output.expanduser().resolve()
    campaign.require(output.is_relative_to(campaign.ARTIFACTS / 'tiny-policy-experiments') and not output.exists(),
                     'Use a fresh durable tiny-policy-experiments directory')
    # The complete published lineage is protected before any candidate is made.
    lineage = campaign.inspect_lineage(campaign.LINEAGE)
    protected = dict(lineage['protected'])
    for value in (lineage['lineage'], lineage['seed_ledger']): protected[value['path']] = value
    campaign.require(lineage['selected']['checkpoint']['progress']['updates'] == 1753
        and lineage['admitted_games'] == 100, 'Unexpected completed comparison baseline')
    original_torch, original_python = torch.get_rng_state(), random.getstate()
    try:
        torch.manual_seed(spec['torch_seed']); random.seed(spec['python_seed'])
        model = m.ContextualActorCritic(config)
        count = sum(p.numel() for p in model.parameters())
        campaign.require(count == spec['expected_parameters'], 'Tiny parameter count changed')
        generator = torch.Generator().manual_seed(spec['minibatch_seed'])
        pool = cp.FrozenOpponentPool(); pool.add('tiny-v1-untrained-initial', model)
        output.mkdir(parents=True, mode=0o700)
        metadata = dict(experiment=spec['experiment'], purpose='untrained_architecture_candidate',
            initialized_from_checkpoint=None, configuration=campaign.binding(CONFIG), training_games=0,
            optimizer_updates=0, training_launcher_accepted=False)
        path = output / 'initial-policy.pt'
        cp.save_policy_checkpoint(path, model, checkpoint='tiny-v1-untrained-initial',
            source_metadata=metadata, generators={'minibatch': generator}, opponent_pool=pool,
            interface=spec['interface'])
        saved_rng = torch.get_rng_state().clone(); saved_python = random.getstate()
        restored_generator = torch.Generator()
        loaded = cp.load_policy_checkpoint(path, expected_source_metadata=metadata, interface=spec['interface'],
            restore_rng=True, generators={'minibatch': restored_generator})
        campaign.require(loaded.model.config == config and loaded.weights_sha256 == m.weights_sha256(model)
            and loaded.provenance == 'random_initialization'
            and torch.equal(saved_rng, torch.get_rng_state()) and saved_python == random.getstate()
            and torch.equal(generator.get_state(), restored_generator.get_state())
            and loaded.opponent_pool.identities == pool.identities, 'Untrained candidate restore differed')
        campaign.verify_protected(protected)
        result = dict(schema='heroes-tiny-policy-preparation-v1', status='prepared_untrained',
            experiment=spec['experiment'], config=config.to_dict(), parameters=count,
            baseline_parameters=74344, parameter_reduction_percent=100*(1-count/74344),
            policy=campaign.binding(path), weights_sha256=loaded.weights_sha256,
            exact_model_rng_pool_restore=True, protected=protected, protected_files_unchanged=len(protected),
            optimizer_state='not created; no update has run', progress=dict(updates=0,rollouts=0,decisions=0),
            initialization_seeds={k:spec[k] for k in ('torch_seed','python_seed','minibatch_seed')},
            baseline_checkpoint=1753, baseline_used_as_initial_weights=False, training_launcher_accepted=False,
            gameplay_performed=False, training_seeds_reserved=False, optimizer_updates=0,
            limits=['Width reduction only; encoders, CNN and GRU remain.',
                    'Native engine gameplay and full training-launch acceptance have not been run for this candidate.',
                    'No end-to-end speed or playing-strength improvement is established.'])
        campaign.write(output / 'preparation.json', result, new=True)
        return result
    finally:
        torch.set_rng_state(original_torch); random.setstate(original_python)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.output is None: parser.error('--prepare requires --output')
        result = prepare(args.output)
    else:
        spec, config = specification()
        result = dict(status='configuration_checked', model=config.to_dict(), parameters=spec['expected_parameters'],
                      gameplay_performed=False, optimizer_updates=0, training_launcher_accepted=False)
    print(json.dumps(result, indent=2))
