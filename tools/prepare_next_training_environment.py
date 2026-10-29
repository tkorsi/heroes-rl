#!/usr/bin/env python3
"""Prepare a separate, locked Python environment without modifying a live run.

This only installs into a new directory and records interpreter/runtime checks.
It neither selects the environment nor changes a machine configuration, launch
script, checkpoint, lock or seed ledger. Selection requires separate acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path.home() / 'Projects/Heroes-artifacts'
SOURCE_PYTHON = Path.home() / 'Projects/Heroes/.venv/bin/python'
LOCK = ROOT / 'dependencies/learner.lock'
PROBE = r'''
import hashlib, importlib.metadata, importlib.util, json, platform, sys, sysconfig, warnings
from pathlib import Path
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def norm(name):
    return name.lower().replace('_', '-').replace('.', '-')
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1392)
    layer = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    value = layer(torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float64))
    value.square().sum().backward()
    optimizer.step()
    numeric = {'output': value.detach().tolist(),
               'parameters': [p.detach().tolist() for p in layer.parameters()],
               'gradients': [p.grad.tolist() for p in layer.parameters()],
               'rng_sha256': hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()}
    numpy_version, bridge = None, False
    if importlib.util.find_spec('numpy') is not None:
        import numpy as np
        numpy_version = np.__version__
        array = np.array([1.0, 2.0], dtype=np.float64)
        tensor = torch.from_numpy(array)
        tensor[0] = 3.0
        back = tensor.numpy()
        back[1] = 4.0
        bridge = bool(array.tolist() == tensor.tolist() == back.tolist() == [3.0, 4.0])
packages, records = {}, {}
for distribution in importlib.metadata.distributions():
    name = norm(distribution.metadata['Name'])
    if name in packages:
        raise RuntimeError('Duplicate installed distribution: ' + name)
    packages[name] = distribution.version
    records[name] = hashlib.sha256((distribution.read_text('RECORD') or '').encode()).hexdigest()
torch_root = Path(torch.__file__).resolve().parent
print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix,
    'base_prefix': sys.base_prefix, 'python_version': sys.version,
    'python_soabi': sysconfig.get_config_var('SOABI'), 'machine': platform.machine(),
    'packages': packages, 'package_record_sha256': records,
    'torch_version': torch.__version__, 'torch_config': torch.__config__.show(),
    'torch_libraries': {p.name: sha(p) for p in sorted((torch_root/'lib').iterdir())
                        if p.is_file() and p.name.startswith(('libtorch', 'libc10'))},
    'numeric_smoke': numeric, 'numpy_version': numpy_version, 'numpy_bridge': bridge,
    'warnings': [str(item.message) for item in caught]}, sort_keys=True))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def locked_versions(path):
    rows = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)', Path(path).read_text(), re.MULTILINE)
    result = {re.sub(r'[-_.]+', '-', name).lower(): version for name, version in rows}
    require(len(rows) == len(result), 'Duplicate locked package')
    require(result.get('torch') == '2.9.1' and result.get('numpy') == '2.3.5',
            'The reviewed environment requires Torch 2.9.1 and NumPy 2.3.5')
    return result


def validate_destination(path, source_prefix, artifacts=ARTIFACTS):
    output = Path(path).expanduser().absolute()
    allowed = Path(artifacts).resolve() / 'python'
    require(not output.exists() and not output.is_symlink(), 'Output must be a new directory')
    require(output.resolve().is_relative_to(allowed) and output.resolve() != allowed,
            'Output must be a fresh child of the durable artifacts/python directory')
    require(not output.resolve().is_relative_to(Path(source_prefix).resolve()),
            'Preparation cannot write inside the source Python environment')
    return output


def verify_transition(before, candidate, expected):
    require(candidate['packages'] == expected, 'Prepared packages differ from the exact lock')
    require({name: expected.get(name) for name in before['packages']} == before['packages']
            and set(expected) - set(before['packages']) <= {'numpy'},
            'Preparation may only add NumPy, never upgrade an existing package')
    for key in ('base_prefix', 'python_version', 'python_soabi', 'machine', 'torch_version',
                'torch_config', 'torch_libraries', 'numeric_smoke'):
        require(candidate[key] == before[key], 'Existing tensor runtime changed: ' + key)
    require(candidate['numpy_version'] == expected['numpy'] and candidate['numpy_bridge'] is True
            and candidate['warnings'] == [], 'NumPy bridge or warning-free Torch initialization failed')
    require(candidate['prefix'] != before['prefix'], 'Candidate reused the source environment')


def prepare(output, *, source_python=SOURCE_PYTHON, lock=LOCK, artifacts=ARTIFACTS, timeout=600):
    require(type(timeout) is int and 30 <= timeout <= 1800, 'Preparation timeout must be 30..1800 seconds')
    started = time.monotonic()
    source_python = Path(source_python).expanduser().absolute()
    require(source_python.is_file(), 'Source Python executable is missing')
    uv = shutil.which('uv')
    require(uv is not None, 'Install uv before preparing the separate environment')
    lock_binding = binding(lock)
    expected = locked_versions(lock)
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
                   'UV_CONCURRENT_INSTALLS': '1', 'UV_CONCURRENT_DOWNLOADS': '2'}

    def run(command, log=None):
        remaining = timeout - (time.monotonic() - started)
        require(remaining > 0, 'Environment preparation deadline exhausted')
        result = subprocess.run(command, capture_output=True, text=True, timeout=remaining,
                                env=environment)
        if log is not None:
            log.write_text(result.stdout + result.stderr)
        require(result.returncode == 0, 'Environment preparation command failed: ' + repr(command))
        return result

    before_result = run([str(source_python), '-B', '-I', '-c', PROBE])
    before = json.loads(before_result.stdout)
    output = validate_destination(output, before['prefix'], artifacts)
    require({name: expected.get(name) for name in before['packages']} == before['packages']
            and set(expected) - set(before['packages']) <= {'numpy'},
            'Source environment does not match the preserved locked package versions')
    output.mkdir(parents=True, mode=0o700)
    (output/'source-before.json').write_text(json.dumps(before, indent=2, sort_keys=True)+'\n')
    (output/'source-before.stderr').write_text(before_result.stderr)
    target = output/'venv'
    run([uv, 'venv', '--no-managed-python', '--python', str(source_python), str(target)], output/'venv.log')
    python = target/'bin/python'
    run([uv, 'pip', 'sync', '--python', str(python), '--require-hashes', '--no-build',
         '--link-mode', 'copy', str(lock)], output/'install.log')
    candidate_result = run([str(python), '-B', '-I', '-c', PROBE], output/'candidate-probe.log')
    candidate = json.loads(candidate_result.stdout)
    verify_transition(before, candidate, expected)
    after_result = run([str(source_python), '-B', '-I', '-c', PROBE])
    after = json.loads(after_result.stdout)
    require(before == after, 'Source Python environment changed during preparation')
    require(binding(lock) == lock_binding, 'Dependency lock changed during preparation')
    (output/'candidate.json').write_text(json.dumps(candidate, indent=2, sort_keys=True)+'\n')
    (output/'source-after.json').write_text(json.dumps(after, indent=2, sort_keys=True)+'\n')
    receipt = {'schema': 'heroes-next-training-environment-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'python': str(python),
        'dependency_lock': lock_binding, 'source_python': str(source_python),
        'source_before': binding(output/'source-before.json'),
        'source_after': binding(output/'source-after.json'), 'candidate': binding(output/'candidate.json'),
        'source_environment_unchanged': True, 'existing_packages_unchanged': True,
        'python_and_torch_abi_unchanged': True, 'torch_libraries_identical': True,
        'numeric_smoke_exact': True, 'numpy_bridge_verified': True, 'initialization_warnings': [],
        'selected_for_training': False, 'official_state_modified': False, 'training_launched': False,
        'elapsed_seconds': time.monotonic()-started,
        'scope': 'Separate environment preparation only. Full learner/runtime acceptance and a new immutable machine selection remain required.'}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-python', type=Path, default=SOURCE_PYTHON)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, source_python=args.source_python,
                             timeout=args.timeout_seconds), indent=2))


if __name__ == '__main__':
    main()
