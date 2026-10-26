"""Optimizer-only PPO benchmark on disposable exact copies.

Restores a parent checkpoint pair (model, Adam, minibatch generator, global RNG) with the strict
loaders, rebuilds the four verified rollouts of one completed batch from its preserved capture
files, and runs the unchanged PPO kernel (``ppo._update``) once per repetition from the identical
restored state. Every repetition records the PPO report, a digest of every resulting tensor
(model, gradients, Adam state, generator) and the sampled minibatch order, so implementations can
be compared bit for bit. Nothing is written into any checkpoint lineage.

    ppo_bench.py --parent-plan P --parent-report R --policy F --resume F --batch DIR [--repeat N]
                 [--reference-resume F] [--save-state F] [--compare-state F]
                 [--experiment shared_graph_encodings] [--prefix-helpers N] [--profile OUT] [--output JSON]
"""
import argparse
import cProfile
import ctypes
import hashlib
import json
from pathlib import Path
import pstats
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from experiments.learner import checkpoints as cp
from experiments.learner.builtin_training_resume import _completion, _metadata, _exact, require
from experiments.learner.builtin_rollout import load_builtin_episode
from experiments.learner.builtin_gameplay import _samples
from experiments.learner import ppo
from experiments.learner.ppo import PPOConfig, _update
from experiments.headless.contextual_evidence import strict_json


class Rollout:
    """The fields of a verified training rollout that the PPO kernel reads."""
    def __init__(self, experience, samples):
        self.experience, self.samples = experience, samples
        self.history, self.decisions = experience.history, experience.decisions


def tensor_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().contiguous(); digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()) if data.numel() else b'')
        elif isinstance(item, dict):
            for key in sorted(item, key=str): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item: visit(element)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def difference(reference, state):
    """Largest absolute and relative element differences per part, and how many tensors differ at all."""
    def tensors(value):
        if isinstance(value, torch.Tensor): yield value
        elif isinstance(value, dict):
            for key in sorted(value, key=str): yield from tensors(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: yield from tensors(item)
    result = {}
    for part in ('model', 'gradients', 'adam'):
        absolute = relative = 0.0; differing = total = 0
        for left, right in zip(tensors(reference[part]), tensors(state[part]), strict=True):
            if not left.is_floating_point(): continue
            total += 1
            if torch.equal(left, right): continue
            differing += 1
            delta = (left.double() - right.double()).abs()
            absolute = max(absolute, delta.max().item())
            relative = max(relative, (delta / left.double().abs().clamp_min(1e-30)).max().item())
        result[part] = dict(tensors=total, differing=differing, max_absolute=absolute, max_relative=relative)
    return result


def restore(plan_path, report_path, policy_path, resume_path):
    plan, report = json.loads(Path(plan_path).read_text()), json.loads(Path(report_path).read_text())
    metadata = _metadata(plan, report['plan_sha256'])
    allowed = (cp.BUILTIN_GAMEPLAY_PROVENANCE,)
    interface = plan['source']['interface']
    policy = cp.load_policy_checkpoint(policy_path, expected_source_metadata=metadata, allowed_provenance=allowed, interface=interface)
    model = policy.model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
    generator = torch.Generator()
    state = cp.load_checkpoint(resume_path, model, optimizer, expected_ppo_config=plan['configuration']['ppo'],
                               expected_source_metadata=metadata, generators={'minibatch': generator}, interface=interface,
                               allowed_provenance=allowed, at_rollout_boundary=True)
    progress, steps, sampler = _completion(plan, report)
    require(state.progress == progress and state.sampler_state == sampler, 'Parent progress differs')
    return model, optimizer, generator, plan


def rollouts(batch):
    result = []
    for folder in sorted(Path(batch).glob('episode-*')):
        raw = strict_json((folder/'builtin-training-rollout.json').read_bytes())
        experience = load_builtin_episode(raw['experience'])
        receipt = strict_json((folder/'training-receipt.json').read_bytes())
        reference = ('training-verification/builtin-replay-verification.json#sha256='
                     + receipt['files']['training-verification/builtin-replay-verification.json']['sha256'])
        samples = _samples(experience, reference)
        require(len(samples) == receipt['samples'], 'Rebuilt samples differ from the receipt')
        result.append(Rollout(experience, samples))
    require(len(result) == 4, 'Expected the four verified games of one batch')
    return tuple(result)


def run_once(args, items, profile=None):
    model, optimizer, generator, plan = restore(args.parent_plan, args.parent_report, args.policy, args.resume)
    torch.set_num_threads(1)
    config = PPOConfig(**plan['configuration']['ppo'])
    order_state = generator.get_state().clone()
    model.train().requires_grad_(True)
    started = time.perf_counter()
    if profile is not None: profile.enable()
    report = _update(model, optimizer, items, config=config, generator=generator,
                     max_optimizer_steps=plan['configuration']['max_optimizer_steps'])
    if profile is not None: profile.disable()
    seconds = time.perf_counter() - started
    order_generator = torch.Generator(); order_generator.set_state(order_state)
    windows = sum((len(item.samples) + config.decisions_per_window - 1) // config.decisions_per_window for item in items)
    order = torch.randperm(windows, generator=order_generator).tolist()
    model.eval().requires_grad_(False)
    return dict(seconds=seconds, report={k: v for k, v in vars(report).items() if k != 'timings_seconds'},
                timings=report.timings_seconds, sampler_order=order,
                model_sha256=tensor_digest(dict(model.state_dict())), weights_sha256=cp.model_weight_digest(model),
                gradient_sha256=tensor_digest([p.grad for p in model.parameters()]),
                adam_sha256=tensor_digest(optimizer.state_dict()), generator_sha256=tensor_digest(generator.get_state()),
                model=model, optimizer=optimizer, generator=generator)


def main():
    p = argparse.ArgumentParser()
    for name in ('parent-plan', 'parent-report', 'policy', 'resume', 'batch'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1); p.add_argument('--reference-resume', type=Path)
    p.add_argument('--profile', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--save-state', type=Path, help='write the last repetition\'s model, gradients and Adam state')
    p.add_argument('--compare-state', type=Path, help='report differences from a state saved by --save-state')
    p.add_argument('--experiment', action='append', default=[], choices=['shared_graph_encodings'],
                   help='enable a separate numerical experiment (never a default)')
    p.add_argument('--prefix-helpers', type=int, default=0, help='exact forked prefix-encoding helpers (0: serial)')
    a = p.parse_args()
    ppo.EXPERIMENTS = frozenset(a.experiment)
    ppo.PREFIX_HELPERS = a.prefix_helpers
    items = rollouts(a.batch)
    results = []
    for index in range(a.repeat):
        profile = cProfile.Profile() if a.profile and index == a.repeat-1 else None
        result = run_once(a, items, profile)
        if profile is not None:
            profile.dump_stats(str(a.profile))
            pstats.Stats(str(a.profile)).sort_stats('cumulative').print_stats(40)
        if a.reference_resume:
            raw = torch.load(a.reference_resume, map_location='cpu', weights_only=True)
            checks = {}
            for name, left, right in (('model', dict(result['model'].state_dict()), raw['model_state']),
                                      ('adam', result['optimizer'].state_dict(), raw['resume']['optimizer_state']),
                                      ('minibatch_generator', result['generator'].get_state(), raw['rng']['torch_generators']['minibatch'])):
                try: _exact(left, right); checks[name] = True
                except ValueError: checks[name] = False
            result['reference_equal'] = checks
        state = dict(model=dict(result['model'].state_dict()), gradients=[p.grad for p in result['model'].parameters()],
                     adam=result['optimizer'].state_dict())
        if a.save_state and index == a.repeat-1: torch.save(state, a.save_state)
        if a.compare_state: result['difference'] = difference(torch.load(a.compare_state, weights_only=True), state)
        for key in ('model', 'optimizer', 'generator'): result.pop(key)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('seconds', 'weights_sha256', 'model_sha256', 'adam_sha256', 'gradient_sha256')}
                         | {'reference_equal': result.get('reference_equal'), 'difference': result.get('difference'), 'timings': result['timings']}))
    record = dict(experiments=sorted(ppo.EXPERIMENTS), prefix_helpers=ppo.PREFIX_HELPERS, samples=sum(len(i.samples) for i in items), frames=[len(i.history) for i in items],
                  decisions=[len(i.decisions) for i in items], torch=str(torch.__version__), repetitions=results,
                  identical_across_repetitions=len({r['model_sha256'] + r['adam_sha256'] + r['gradient_sha256'] for r in results}) == 1)
    if a.output: a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('samples', 'frames', 'decisions', 'identical_across_repetitions')}))


if __name__ == '__main__':
    main()
