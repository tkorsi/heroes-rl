"""Verify an explicitly accepted, separate Python environment before selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NEXT_LAUNCH_ENVIRONMENT = {
    'path': str(Path.home() / 'Projects/Heroes-artifacts/python/next-launch-numpy235-v1/receipt.json'),
    'sha256': 'ce11a8abb87b091231a09a42552313ef0c3b88db7b9dd6466dbcacf536373964',
    'bytes': 1717,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(identity):
    require(type(identity) is dict and set(identity) == {'path', 'sha256', 'bytes'}
            and type(identity['path']) is str and type(identity['bytes']) is int
            and identity['bytes'] > 0 and type(identity['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', identity['sha256']),
            'Incomplete Python environment evidence binding')
    path = Path(identity['path'])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
            and path.is_file() and path.stat().st_nlink == 1,
            'Python environment evidence must be an absolute regular file')
    raw = path.read_bytes()
    require(len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256'],
            'Python environment evidence changed: ' + str(path))
    return json.loads(raw)


def machine_python(probe):
    return {'executable': probe['executable'], 'version': probe['python_version'],
            'torch': probe['torch_version']}


def verify_environment(identity, *, previous_python=None, current_python=None, probe_current=True):
    """Keep the old machine immutable; require the pinned NumPy-only transition.

    The numerical probe runs in a separate process, so it cannot alter the caller's
    model, random generators or optimizer. No environment is installed or edited.
    """
    from tools.prepare_next_training_environment import PROBE, locked_versions, verify_transition
    receipt = bound_json(identity)
    require(receipt.get('schema') == 'heroes-next-training-environment-v1'
            and all(receipt.get(key) is True for key in (
                'existing_packages_unchanged', 'numeric_smoke_exact', 'numpy_bridge_verified',
                'python_and_torch_abi_unchanged', 'source_environment_unchanged', 'torch_libraries_identical'))
            and receipt.get('initialization_warnings') == []
            and receipt.get('selected_for_training') is False
            and receipt.get('official_state_modified') is False and receipt.get('training_launched') is False,
            'Separate Python environment preparation is incomplete')
    before, after, candidate = [bound_json(receipt[key]) for key in ('source_before', 'source_after', 'candidate')]
    require(before == after, 'Preparation changed the active Python environment')
    lock = ROOT / 'dependencies/learner.lock'
    raw = lock.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == receipt['dependency_lock']['sha256']
            and len(raw) == receipt['dependency_lock']['bytes'], 'Prepared Python lock differs from current source')
    verify_transition(before, candidate, locked_versions(lock))
    require(receipt['python'] == candidate['executable'] and receipt['source_python'] == before['executable'],
            'Prepared Python executable binding differs')
    if previous_python is not None:
        require(previous_python == machine_python(before), 'Historical Python differs from the reviewed source environment')
    if current_python is not None:
        require(current_python == machine_python(candidate), 'Selected machine differs from the reviewed Python environment')
    if probe_current:
        require(str(Path(sys.executable).absolute()) == candidate['executable'],
                'Use the explicitly prepared Python environment for this runtime release')
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-I', '-c', PROBE], capture_output=True,
                                text=True, check=True, timeout=45, env=environment)
        require(result.stderr == '' and json.loads(result.stdout) == candidate,
                'Selected Python packages, tensor runtime or NumPy bridge changed after acceptance')
    return receipt
