#!/usr/bin/env python3
"""Explicit CPU inference-only migration; never changes the original or loads an optimizer.

The public entry point accepts only the handoff's exact original checkpoint.
This produces a separately bound artifact, not a resumption of a Mac plan.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_SHA256 = "4043b58bb3dc4eb6169514ca1ed7c1be0b373a5139efe04b95dfeda1832232c1"
ORIGINAL_BYTES = 629494
INTERFACE = "adventure-dialogs-v1"
SCHEMA = "heroes-linux-policy-portability-v1"


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _convert(source, output, *, expected_sha256, expected_bytes, expected_learner, expected_opponents):
    """Internal conversion also exercised with synthetic, explicitly bound fixtures."""
    import torch
    from experiments.learner.checkpoints import load_policy_checkpoint
    from experiments.learner.gameplay import assert_same_state
    from experiments.learner.inference import source_identity

    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output already exists; never overwrite a migration')
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 required')
    require(str(torch.__version__) == '2.9.1+cpu' and torch.version.cuda is None and torch.version.hip is None,
            'Validated target requires CPU PyTorch 2.9.1+cpu')
    require(source.stat().st_size == expected_bytes and digest(source) == expected_sha256,
            'Original checkpoint size or SHA-256 differs')
    original = torch.load(source, map_location='cpu', weights_only=True)
    require(original.get('torch_version') == '2.9.1', 'Expected original macOS Torch 2.9.1 envelope')
    require(original.get('provenance') == 'random_initialization' and original.get('kind') == 'policy'
            and original.get('resume') is None, 'Only the original frozen policy is portable')
    saved = original['source_metadata']
    current = source_identity(INTERFACE)
    require(saved['torch'] == '2.9.1' and saved['interface'] == INTERFACE,
            'Source runtime/interface differs')
    require(isinstance(saved['python'], str) and saved['python'].startswith('3.13.5 '),
            'Expected the documented original Python 3.13.5 runtime')
    require(bool(saved['source_sha256']), 'Missing original learner source inventory')
    for name, sha in saved['source_sha256'].items():
        require(current['source_sha256'].get(name) == sha, 'Original learner source changed: ' + name)
    require(original['weights_sha256'] == expected_learner, 'Original learner identity differs')
    pool = original['opponent_pool']
    require(isinstance(pool, list) and [(p['checkpoint'], p['weights_sha256']) for p in pool] == expected_opponents,
            'Original opponent identities differ')

    migrated = deepcopy(original)
    migrated['checkpoint'] = original['checkpoint'] + '-linux-cpu-v1'
    migrated['torch_version'] = str(torch.__version__)
    migrated['source_metadata'] = {**current, 'portability': {
        'schema': SCHEMA, 'original_sha256': expected_sha256,
        'original_bytes': expected_bytes, 'original_checkpoint': original['checkpoint'],
        'original_source_metadata': saved, 'purpose': 'frozen_inference_only',
        'optimizer_updates': 0, 'cross_platform_behavior_equivalence': 'not_established'}}
    # The only permitted changes are the new artifact identity and explicit runtime provenance.
    for key in original.keys() - {'checkpoint', 'torch_version', 'source_metadata'}:
        assert_same_state(original[key], migrated[key])
    with tempfile.TemporaryDirectory(prefix='heroes-policy-port-') as temp:
        candidate = Path(temp) / 'fixed-policy-linux.pt'
        torch.save(migrated, candidate)
        # Preserve all standard schema, contract, finite tensor, model shape, hash,
        # provenance and local runtime guards. There is no permissive loader flag.
        loaded = load_policy_checkpoint(candidate, expected_source_metadata=migrated['source_metadata'],
                                        interface=INTERFACE, allowed_provenance=('random_initialization',))
        assert_same_state(original['model_state'], dict(loaded.model.state_dict()))
        require(loaded.opponent_pool is not None, 'Frozen opponent pool missing')
        for entry in pool:
            opponent = loaded.opponent_pool.get(entry['checkpoint'])
            assert_same_state(entry['model_state'], dict(opponent.model.state_dict()))
        loaded.opponent_pool.assert_unchanged()
        require(digest(source) == expected_sha256, 'Original changed during conversion')
        report = {'schema': SCHEMA, 'purpose': 'frozen_inference_only',
            'original': {'sha256': expected_sha256, 'bytes': expected_bytes},
            'converted': {'filename': candidate.name, 'sha256': digest(candidate), 'bytes': candidate.stat().st_size},
            'source_runtime': {'torch': saved['torch'], 'python': saved['python']},
            'target_runtime': {'torch': str(torch.__version__), 'python': sys.version, 'platform': platform.platform()},
            'learner_weights_sha256': loaded.weights_sha256,
            'opponents': [list(item) for item in loaded.opponent_pool.identities],
            'exact_learner_and_opponent_tensors': True, 'standard_loader_passed': True,
            'original_preserved': True, 'optimizer_updates': 0, 'gameplay_performed': False,
            'cross_platform_behavior_equivalence': 'not_established',
            'linux_gameplay_and_native_replay': 'not_run'}
        output.mkdir(parents=True, exist_ok=False)
        with (output / candidate.name).open('xb') as stream:
            stream.write(candidate.read_bytes())
        with (output / 'portability.json').open('x') as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def convert(source, output):
    return _convert(source, output, expected_sha256=ORIGINAL_SHA256, expected_bytes=ORIGINAL_BYTES,
        expected_learner='1c290bcb1ca37cb876841645fab5cd08a94f999bbc57a56fb285243a8e46bbae',
        expected_opponents=[('hero-opponent-7176f1a2e81393cb',
                            '7176f1a2e81393cbfe709238fcd3c0060871a9ddaec1cf4181812f75fb08fc82')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.output), indent=2))
