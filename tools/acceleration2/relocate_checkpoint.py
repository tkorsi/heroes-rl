"""Benchmark-only import of a historical checkpoint pair into this machine's Torch build.

The strict loader accepts a checkpoint only under the exact Torch version string that
wrote it (or the reviewed '+cpu' relocation). This machine's PyPI wheel reports
'+cu128' for the same public release, so the archived pair cannot be loaded here
unchanged. This tool writes disposable copies whose payload differs from the original
only in the recorded ``torch_version`` field, then proves that every other field equals
the original file's payload: every tensor bitwise, RNG states, Adam state, progress,
sampler, opponent pool and metadata. The copies are benchmark inputs only. They are
never a trained production checkpoint and are never published as one.

    python tools/acceleration2/relocate_checkpoint.py --plan P --report R --source-dir D --output O
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from experiments.learner import checkpoints as cp
from experiments.learner.builtin_training_resume import binding, require, _exact, _metadata, _completion

KINDS = ('policy', 'resume')


def _raw(path):
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def relocate(plan_path, report_path, source_dir, output):
    plan_binding, report_binding = binding(plan_path), binding(report_path)
    plan, report = json.loads(Path(plan_path).read_text()), json.loads(Path(report_path).read_text())
    require(report['plan_sha256'] == plan_binding['sha256'], 'Report belongs to another plan')
    saved = report['after_checkpoint']
    originals = {}
    for kind in KINDS:
        original = binding(Path(source_dir) / f'after-{kind}.pt')
        require({k: original[k] for k in ('sha256', 'bytes')} == {k: saved[kind][k] for k in ('sha256', 'bytes')},
                'Archived checkpoint bytes differ from the completion report: ' + kind)
        originals[kind] = original
    historical = plan['source']['torch']
    current = str(torch.__version__)
    require(historical.split('+')[0] == current.split('+')[0], 'Relocation needs the same public Torch release')
    output = Path(output)
    require(not output.exists(), 'Relocation output must be new')
    output.mkdir(parents=True)
    relocated = {}
    for kind in KINDS:
        payload = _raw(originals[kind]['path'])
        require(payload['torch_version'] == historical, 'Archived payload Torch version differs from its plan')
        payload['torch_version'] = current
        relocated[kind] = binding(cp._write(output / f'after-{kind}.pt', payload))
    # Every field other than the version string must equal the original bytes' payload.
    for kind in KINDS:
        before, after = _raw(originals[kind]['path']), _raw(relocated[kind]['path'])
        require(set(before) == set(after), 'Relocated payload keys differ')
        for key in before:
            if key != 'torch_version':
                _exact(before[key], after[key])
        require(after['torch_version'] == current, 'Relocated version not recorded')
    # The strict loaders must restore the original state exactly from the copies.
    caller_torch, caller_python = torch.get_rng_state(), random.getstate()
    try:
        progress, steps, sampler = _completion(plan, report)
        metadata = _metadata(plan, report['plan_sha256'])
        allowed = (cp.BUILTIN_GAMEPLAY_PROVENANCE,)
        interface = plan['source']['interface']
        policy = cp.load_policy_checkpoint(relocated['policy']['path'], expected_source_metadata=metadata,
                                           allowed_provenance=allowed, interface=interface)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
        generator = torch.Generator()
        state = cp.load_checkpoint(relocated['resume']['path'], model, optimizer,
                                   expected_ppo_config=plan['configuration']['ppo'], expected_source_metadata=metadata,
                                   generators={'minibatch': generator}, interface=interface, allowed_provenance=allowed,
                                   at_rollout_boundary=True)
        original = _raw(originals['resume']['path'])
        _exact(dict(model.state_dict()), original['model_state'])
        _exact(optimizer.state_dict(), original['resume']['optimizer_state'])
        _exact(torch.get_rng_state(), original['rng']['torch_cpu'])
        _exact(random.getstate(), original['rng']['python'])
        _exact(generator.get_state(), original['rng']['torch_generators']['minibatch'])
        _exact(state.opponent_pool.state_dict(), policy.opponent_pool.state_dict())
        require(state.progress == progress and state.sampler_state == sampler, 'Relocated progress differs')
        require(cp.model_weight_digest(model) == policy.weights_sha256 == saved['weights_sha256'],
                'Relocated weights differ')
    finally:
        torch.set_rng_state(caller_torch)
        random.setstate(caller_python)
    receipt = dict(schema='heroes-benchmark-relocation-v1', scope='benchmark input only; never a production checkpoint',
                   plan=plan_binding, report=report_binding, original=originals, relocated=relocated,
                   historical_torch=historical, current_torch=current, python=sys.version,
                   changed_fields=['torch_version'], weights_sha256=saved['weights_sha256'],
                   progress=vars(progress), optimizer_steps=steps, exact_model_adam_rng_progress_pool=True)
    path = output / 'relocation-receipt.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = relocate(args.plan, args.report, args.source_dir, args.output)
    print(json.dumps({k: receipt[k] for k in ('relocated', 'weights_sha256', 'progress')}, indent=2))


if __name__ == '__main__':
    main()
