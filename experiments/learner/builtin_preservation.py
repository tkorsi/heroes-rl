"""Verify and atomically preserve a completed batch outside managed worktrees.

Canonical runs must already live under the durable root. Absolute bindings in
plans/reports are retained; this helper never rewrites evidence or game data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from .builtin_training_resume import binding, require, _exact, _metadata, _completion

DURABLE_ROOT = Path('/Users/orsi/Projects/Heroes-artifacts')


def durable_path(path, root=DURABLE_ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root) and path != root, 'Artifact must be inside the durable root')
    require('/.codex/worktrees/' not in str(path) and '/.codex/worktrees/' not in str(root),
            'Durable artifacts cannot live inside managed worktrees')
    return path


def verify_pair(resume_path, policy_path, plan, report):
    import torch
    from . import checkpoints as cp
    original_torch, original_python = torch.get_rng_state(), random.getstate()
    try:
        progress, _, sampler = _completion(plan, report)
        metadata = _metadata(plan, report['plan_sha256'])
        allowed = (cp.BUILTIN_GAMEPLAY_PROVENANCE,)
        policy = cp.load_policy_checkpoint(policy_path, expected_source_metadata=metadata,
            allowed_provenance=allowed, interface=plan['source']['interface'])
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
        generator = torch.Generator()
        state = cp.load_checkpoint(resume_path, model, optimizer,
            expected_ppo_config=plan['configuration']['ppo'], expected_source_metadata=metadata,
            generators={'minibatch': generator}, interface=plan['source']['interface'],
            allowed_provenance=allowed, at_rollout_boundary=True)
        raw = torch.load(resume_path, weights_only=True, map_location='cpu')
        require(state.progress == progress and state.sampler_state == sampler, 'Progress differs')
        require(state.checkpoint == policy.checkpoint and cp.model_weight_digest(model) == policy.weights_sha256
                == report['after_checkpoint']['weights_sha256'], 'Checkpoint pair differs')
        _exact(dict(model.state_dict()), raw['model_state'])
        _exact(optimizer.state_dict(), raw['resume']['optimizer_state'])
        _exact(state.opponent_pool.state_dict(), policy.opponent_pool.state_dict())
        _exact(torch.get_rng_state(), raw['rng']['torch_cpu'])
        _exact(random.getstate(), raw['rng']['python'])
        _exact(generator.get_state(), raw['rng']['torch_generators']['minibatch'])
        return {'checkpoint': policy.checkpoint, 'weights_sha256': policy.weights_sha256,
                'progress': vars(progress), 'exact_model_adam_rng_progress_pool': True}
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)


def preserve(plan_path, report_path, destination, *, root=DURABLE_ROOT):
    plan_path, report_path, destination = [durable_path(p, root) for p in (plan_path, report_path, destination)]
    require(not destination.exists(), 'Preservation destination already exists')
    plan, report = [json.loads(p.read_text()) for p in (plan_path, report_path)]
    require(report['plan_sha256'] == binding(plan_path)['sha256'], 'Plan/report mismatch')
    require(report_path == Path(plan['execution_output']) / 'report.json', 'Report path differs')
    _completion(plan, report)
    sources = {'plan.json': plan_path, 'report.json': report_path}
    for kind in ('resume', 'policy'):
        bound = report['after_checkpoint'][kind]
        path = durable_path(bound['path'], root)
        require(binding(path) == bound, 'Checkpoint binding differs')
        sources['after-' + kind + '.pt'] = path
    before = {name: binding(path) for name, path in sources.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.preserve-', dir=destination.parent))
    try:
        for name, path in sources.items():
            shutil.copyfile(path, staging / name)
            with (staging / name).open('rb') as stream:
                os.fsync(stream.fileno())
            require(binding(staging / name)['sha256'] == before[name]['sha256'], 'Copy digest differs')
        proof = verify_pair(staging / 'after-resume.pt', staging / 'after-policy.pt', plan, report)
        require(before == {name: binding(path) for name, path in sources.items()}, 'Source changed during copy')
        receipt = {'schema': 'heroes-durable-batch-v1', 'sources': before,
                   'destination': str(destination), 'restore': proof,
                   'canonical_run': str(plan_path.parent),
                   'note': 'Canonical histories/replays remain at their durable absolute paths; no evidence was rewritten.'}
        with (staging / 'preservation-receipt.json').open('w') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.rename(staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        return receipt
    finally:
        if staging.exists(): shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preserve(args.plan, args.report, args.destination), indent=2))


if __name__ == '__main__':
    main()
