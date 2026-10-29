"""Explicit native execution plans; no implicit backend or fallback selection."""
from pathlib import Path
import hashlib
import json
import re
import subprocess


def acceptance_source_hashes():
    """Bind the validated integration as well as the independently built library."""
    root = Path(__file__).resolve().parents[2]
    from .local_script_training import REVIEWED_TREE, discovered_reviewed_sources
    names = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--', *REVIEWED_TREE],
                                       text=True, timeout=30).splitlines())
    names |= discovered_reviewed_sources(root)
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}


def check_native_execution(value):
    fields = {'implementation', 'overlapped_verification', 'prefix_helpers', 'text_leaf_replay',
              'replay_threads', 'experiments', 'collection_workers', 'verification_workers',
              'resource_plan', 'native_learner'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('Unknown native execution choice')
    if (value['implementation'] != 'native-cpu-v1' or value['overlapped_verification'] is not False
            or type(value['prefix_helpers']) is not int or value['prefix_helpers'] != 0
            or value['text_leaf_replay'] is not False or type(value['replay_threads']) is not int
            or value['replay_threads'] != 0 or value['experiments'] != []):
        raise ValueError('Native execution cannot mix fork helpers, Python leaf replay or numerical experiments')
    native, plan = value['native_learner'], value['resource_plan']
    from .resources import make_resource_plan
    try:
        if make_resource_plan(plan.get('requested_cpu_cap'), resources=plan['resources']) != plan:
            raise ValueError('Native resource plan changed after discovery')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed native resource plan') from error
    if (type(native) is not dict or set(native) != {'manifest', 'sha256', 'prefix_threads'}
            or type(native['manifest']) is not str or not Path(native['manifest']).is_absolute()
            or type(native['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', native['sha256'])
            or type(native['prefix_threads']) is not int or not 1 <= native['prefix_threads'] <= 256):
        raise ValueError('Native learner needs an absolute manifest, SHA256 and bounded worker count')
    if (type(plan) is not dict or type(plan.get('budget_cpus')) is not int
            or not 1 <= plan['budget_cpus'] <= 256
            or plan.get('batch_size') != 4 or plan.get('torch_intraop_threads') != 1
            or plan.get('native_prefix_threads') != native['prefix_threads']
            or plan.get('native_text_replay_workers') != max(0, native['prefix_threads'] - 1)
            or native['prefix_threads'] != plan['budget_cpus']):
        raise ValueError('Native resource plan differs from the fixed numerical contract')
    for name in ('collection_workers', 'verification_workers'):
        if (type(value[name]) is not int or value[name] not in (1, 4)
                or value[name] > plan['budget_cpus'] or value[name] != plan.get(name)):
            raise ValueError('Native phase workers exceed the recorded resource budget')
    return value


def require_native_acceptance(manifest, sha256):
    """Require accepted source and actual binary at every official entry point."""
    from .native_backend import verified_manifest
    manifest = Path(manifest).resolve()
    receipt = verified_manifest(manifest, sha256)
    accepted = Path(__file__).resolve().parents[2] / 'validation/native-mac-learner-2026-10-29.json'
    report = json.loads(accepted.read_text())
    if (report.get('accepted_for_manual_training') is not True
            or report.get('native_manifest_sha256') != sha256
            or report.get('acceptance_source_sha256') != acceptance_source_hashes()):
        raise ValueError('This built native learner has not passed the manual-training acceptance gate')
    return receipt


def native_execution_choice(manifest, sha256, *, cpu_cap=None, require_acceptance=True):
    from .native_backend import verified_manifest
    from .resources import make_resource_plan
    manifest = Path(manifest).resolve()
    if require_acceptance:
        require_native_acceptance(manifest, sha256)
    else:
        verified_manifest(manifest, sha256)
    plan = make_resource_plan(cpu_cap)
    value = dict(implementation='native-cpu-v1', overlapped_verification=False, prefix_helpers=0,
                 text_leaf_replay=False, replay_threads=0, experiments=[],
                 collection_workers=plan['collection_workers'], verification_workers=plan['verification_workers'],
                 resource_plan=plan, native_learner=dict(manifest=str(manifest), sha256=sha256,
                                                       prefix_threads=plan['native_prefix_threads']))
    return check_native_execution(value)
