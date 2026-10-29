#!/usr/bin/env python3
"""Read-only audit of the pinned cloud1335 publication failure and preserved state.

This certifies stored evidence, not a new Mac replay of Linux behavior. It never
admits a checkpoint, executes a game/update, reserves a seed or changes a pointer.
Optional raw extraction creates only verified regular files in a fresh directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'heroes-cloud1335-publication-audit-v1'
COMMIT = '89550dba6c481653ffc38cd78997438e846947f2'
SOURCE = 'c6ebc8327e8dfe65dfa384a3fe77da709343a4dd'
PREFIX = 'cloud/checkpoint1229-cloud-a08f55ff0c7548f58ff9d50ef5af8ba7'
CONFIG_SHA = '2dac14048e06f4595d082522811577da93925f526155505ef31b850dbc4aec30'
SOURCE_SHA = '57816bb82e504532b54440bc3850a945f28a722584031b529bca63ef3c3419b4'
MAP_SHA = 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3'
RAW_SHA = '9d25a99722395a570ff58edb7dc82d69028ad7da351ccd1b5cd9198d5542c436'
DATA_TARGET = '/workspace/heroes-task-private/assets/game-data/Data'
DATA_LINKS = {f'batch/execution/episode-{n:06d}/profile/data/vcmi/Data': DATA_TARGET
              for n in range(1, 5)}
NONRAW = {'plan.json', 'report.json', 'after-policy.pt', 'after-resume.pt',
          'initial-checkpoint.pt', 'preservation-receipt.json', 'sequence-ledger.json', 'candidate.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Read-only cloud publication audit exceeded its deadline')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def safe_relative(name, *, root=False):
    require(type(name) is str and name and '\\' not in name and '\0' not in name,
            'Unsafe archive or inventory path')
    parts = name.split('/')
    require(not name.startswith('/') and all(p not in ('', '.', '..') for p in parts)
            and ':' not in parts[0], 'Unsafe archive or inventory path: ' + name)
    require(not root or parts[0] == 'batch', 'Archive member is outside batch/: ' + name)
    return PurePosixPath(name)


def regular_path(path):
    """Reject symlink ancestors as well as a symlink final component."""
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        require(not component.is_symlink(), 'Symlink input or output path: ' + str(component))
    require(stat.S_ISREG(path.stat().st_mode), 'Expected regular file: ' + str(path))
    return path


def binding(path, deadline):
    path = regular_path(path)
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            check_time(deadline)
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'File changed while hashing')
    return {'bytes': size, 'sha256': digest.hexdigest()}


def bound_equal(actual, expected, message):
    require(type(expected) is dict and all(actual[k] == expected.get(k) for k in ('sha256', 'bytes')), message)


def git(repo, args, deadline):
    check_time(deadline)
    env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0', 'GIT_ALLOW_PROTOCOL': ''}
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          env=env, timeout=min(30, deadline - time.monotonic())).stdout


def verify_manifest(repo, relative, deadline, *, include_raw=False):
    folder = repo / relative
    path = regular_path(folder / 'manifest.json')
    recorded = git(repo, ['show', COMMIT + ':' + relative + '/manifest.json'], deadline)
    require(path.read_bytes() == recorded, 'Manifest differs from pinned Git commit')
    manifest = strict_json(recorded)
    require(manifest.get('schema') == 'heroes-linux-private-publication-v1'
            and manifest.get('game_data_symlinks_followed') is False
            and type(manifest.get('entries')) is dict and type(manifest.get('raw_inventory')) is dict,
            'Invalid publication manifest')
    artifacts = {'manifest.json': binding(path, deadline)}
    for name, expected in manifest['entries'].items():
        safe_relative(name)
        require('/' not in name and name != 'manifest.json', 'Unsafe publication entry')
        require(type(expected.get('bytes')) is int and expected['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', expected.get('sha256', '')), 'Invalid publication binding')
        if not name.startswith('raw.part') or include_raw:
            artifacts[name] = binding(folder / name, deadline)
            bound_equal(artifacts[name], expected, 'Publication entry differs: ' + name)
    return manifest, artifacts


def verify_raw_archive(archive, inventory, *, deadline, extract_to=None, allowed_symlinks=None):
    """Verify every tar member; never use extract(), extractall() or follow links."""
    allowed_symlinks = {} if allowed_symlinks is None else allowed_symlinks
    require(type(inventory) is dict and inventory, 'Raw inventory must not be empty')
    expected = {}
    for relative, entry in inventory.items():
        safe_relative(relative)
        require(type(entry) is dict and type(entry.get('bytes')) is int and entry['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', entry.get('sha256', '')), 'Invalid raw inventory binding')
        expected['batch/' + relative] = entry
    archive_before = binding(archive, deadline)
    staging = None
    destination = None
    if extract_to is not None:
        destination = Path(os.path.abspath(extract_to))
        require(not destination.exists() and not destination.is_symlink(), 'Extraction destination must be fresh')
        require(destination.parent.is_dir(), 'Extraction parent must already exist')
        for component in (*reversed(destination.parent.parents), destination.parent):
            require(not component.is_symlink(), 'Extraction parent contains a symlink')
        staging = Path(tempfile.mkdtemp(prefix='.raw-audit-', dir=destination.parent))
    seen, regular, links, directories, size = set(), set(), {}, 0, 0
    try:
        with tarfile.open(regular_path(archive), mode='r|gz') as stream:
            for member in stream:
                check_time(deadline)
                safe_relative(member.name, root=True)
                require(member.name not in seen, 'Duplicate archive member: ' + member.name)
                seen.add(member.name)
                require(not member.sparse, 'Sparse archive members are not supported')
                # No member may appear below a link, regardless of archive ordering.
                require(not any(parent.as_posix() in allowed_symlinks
                                for parent in PurePosixPath(member.name).parents), 'Archive member below a symlink')
                if member.isdir():
                    require(member.size == 0, 'Directory has unexpected payload')
                    directories += 1
                    continue
                if member.issym():
                    require(member.size == 0 and allowed_symlinks.get(member.name) == member.linkname,
                            'Unapproved archive symlink')
                    require(member.name not in expected, 'Inventory cannot contain a symlink')
                    links[member.name] = member.linkname
                    continue
                require(member.isreg(), 'Archive contains a hardlink, device, FIFO or other unsafe member')
                require(member.name in expected and member.size == expected[member.name]['bytes'],
                        'Unexpected archive file or size: ' + member.name)
                source = stream.extractfile(member)
                require(source is not None, 'Missing archive member payload')
                output = None
                if staging is not None:
                    target = staging.joinpath(*PurePosixPath(member.name).parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    output = target.open('xb')
                    os.chmod(target, 0o600)
                digest, actual_size = hashlib.sha256(), 0
                try:
                    while chunk := source.read(1024 * 1024):
                        check_time(deadline)
                        digest.update(chunk)
                        actual_size += len(chunk)
                        if output is not None:
                            output.write(chunk)
                finally:
                    source.close()
                    if output is not None:
                        output.close()
                bound_equal({'bytes': actual_size, 'sha256': digest.hexdigest()}, expected[member.name],
                            'Archive member digest differs: ' + member.name)
                regular.add(member.name)
                size += actual_size
        require(regular == set(expected), 'Archive regular members differ from complete raw inventory')
        require(links == allowed_symlinks, 'Archive did not retain exactly the recorded game-data links')
        require(binding(archive, deadline) == archive_before, 'Archive changed during audit')
        if staging is not None:
            require(not destination.exists(), 'Extraction destination appeared during audit')
            staging.rename(destination)
            staging = None
        return {'archive': archive_before, 'regular_files_verified': len(regular),
                'regular_bytes_verified': size, 'directories': directories,
                'symlinks_recorded_not_followed': links, 'all_regular_members_match_inventory': True,
                'extracted_regular_files': str(destination) if destination else None}
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def full_state_fingerprints(path):
    import torch
    from tools.recover_interrupted_training import fingerprint
    raw = torch.load(path, weights_only=True, map_location='cpu')
    components = {'model': raw['model_state'], 'adam': raw['resume']['optimizer_state'],
                  'rng': raw['rng'], 'progress': raw['resume']['progress'],
                  'sampler': raw['resume']['sampler_state'], 'pool': raw['opponent_pool']}
    return {'full_state_sha256': fingerprint(components),
            'component_sha256': {name: fingerprint(value) for name, value in components.items()}}


def audit(repo, *, deadline, extract_to=None):
    from experiments.learner.builtin_preservation import verify_pair
    repo = Path(os.path.abspath(repo))
    require(git(repo, ['rev-parse', 'HEAD'], deadline).decode().strip() == COMMIT,
            'Evidence checkout is not at the pinned closure commit')
    closure = repo / PREFIX / 'closure'
    _, closure_bindings = verify_manifest(repo, PREFIX + '/closure', deadline, include_raw=True)
    read = lambda path: strict_json(regular_path(path).read_bytes())
    session, campaign, summary, shutdown, latest, pending = [read(closure / name) for name in
        ('session.json', 'campaign-report.json', 'summary.json', 'shutdown.json',
         'latest-checkpoint.json', 'pending-checkpoint.json')]
    require(session['source']['git_commit'] == SOURCE and json_digest(session['source']) == SOURCE_SHA,
            'Frozen cloud source identity changed')
    require(json_digest(session['configuration']) == CONFIG_SHA
            and session['policy_support'] == 'visible-interaction-budget-v1'
            and session['training_engine_seed_base'] == 40000000
            and session['training_policy_seed_base'] == 50000000 and session['next_sequence'] == 869,
            'Frozen configuration, support contract or seed schedule changed')
    for relative, digest in session['source']['source_sha256'].items():
        safe_relative(relative)
        require(hashlib.sha256(git(ROOT, ['show', SOURCE + ':' + relative], deadline)).hexdigest() == digest,
                'Historical source bytes differ: ' + relative)
    require(campaign['status'] == summary['status'] == 'stopped_failure'
            and shutdown['owned_cleanup_verified'] is True and shutdown['worker_returncode'] == 1
            and shutdown['supervisor_error'] is None and shutdown['within_hard_deadline'] is True,
            'Historical shutdown or failure status differs')
    require(summary['new_ppo_updates'] == 105 and summary['admitted_training_games'] == 420
            and summary['original_campaign'] == {'target_games': 5000, 'admitted_games': 2820,
                                                'remaining_games': 2180}, 'Historical admission counters differ')
    rows = [row for row in campaign['batches'] if row['mode'] == 'training']
    require([row['sequence'] for row in rows] == list(range(869, 975)), 'Cloud batch sequence has gaps or repeats')
    previous, previous_progress, previous_steps = session['initial_parent'], None, None
    results, held_bindings = [], {}
    original_ledger = session['original_continuation']['seed_ledger_snapshot']['consumed']
    for row in rows:
        check_time(deadline)
        sequence, seat = row['sequence'], row['sequence'] % 2
        name = f'train-{sequence:06d}-seat{seat}'
        relative = PREFIX + '/batches/' + name
        folder = repo / relative
        manifest, artifacts = verify_manifest(repo, relative, deadline, include_raw=sequence == 974)
        require(set(manifest['entries']) == NONRAW | {'raw.part00000'}, 'Unexpected batch entry set')
        held_bindings[relative] = artifacts
        plan, report, candidate, preservation, ledger = [read(folder / name) for name in
            ('plan.json', 'report.json', 'candidate.json', 'preservation-receipt.json', 'sequence-ledger.json')]
        require(plan['source'] == session['source'] and plan['configuration'] == session['configuration']
                and plan['execution'] == session['execution']
                and plan['policy_support'] == session['policy_support'], 'Batch changed its frozen experiment')
        require(plan['map']['sha256'] == MAP_SHA and plan['map']['seat'] == seat
                and plan['map']['scenario'] == 'hero-management-v2' and plan['map']['fixture'] is False
                and plan['native_opponents'] == read(repo / PREFIX / 'batches/train-000869-seat1/plan.json')['native_opponents'],
                'Map or opponent changed')
        require(plan['batch_sequence'] == sequence and plan['seat'] == row['seat'] == seat,
                'Batch or seat identity differs')
        schedule = [dict(episode=i + 1, phase='training', learner_player=seat,
                         engine_seed=40000000 + 4 * sequence + i, policy_seed=50000000 + 4 * sequence + i)
                    for i in range(4)]
        require(plan['schedule'] == schedule, 'The four fresh training seeds changed')
        require(plan['parent_descriptor'] == previous and row['parent'] == previous, 'Parent descriptor chain differs')
        for kind in ('plan', 'report', 'policy', 'resume'):
            require(plan['parent'][kind] == previous[kind], 'Plan lost parent linkage')
        require(plan['checkpoint'] == previous['policy'], 'Initial policy linkage differs')
        bound_equal(artifacts['initial-checkpoint.pt'], previous['policy'], 'Initial checkpoint bytes differ')
        bound_equal(artifacts['plan.json'], row['plan'], 'Campaign plan binding differs')
        bound_equal(artifacts['report.json'], row['report'], 'Campaign report binding differs')
        require(report['plan_sha256'] == artifacts['plan.json']['sha256'], 'Report plan hash differs')
        bound_equal(closure_bindings['session.json'], plan['local_session'], 'Batch session binding differs')
        for kind in ('policy', 'resume'):
            bound_equal(artifacts['after-' + kind + '.pt'], report['after_checkpoint'][kind], 'Checkpoint binding differs')
        proof = verify_pair(folder / 'after-resume.pt', folder / 'after-policy.pt', plan, report,
                            migration=SimpleNamespace(checkpoint_options={'historical_torch_version': session['source']['torch']}))
        require(proof == preservation['restore'] and proof['progress'] == candidate['progress']
                and proof['progress']['updates'] == 1230 + sequence - 869
                and proof['progress']['rollouts'] == 4 * proof['progress']['updates'], 'Restored progress differs')
        if previous_progress is not None:
            require(plan['parent']['progress'] == previous_progress
                    and plan['parent']['optimizer_steps'] == previous_steps, 'Progress or Adam chain differs')
        for kind, filename in [('plan', 'plan.json'), ('report', 'report.json'), ('policy', 'after-policy.pt'),
                               ('resume', 'after-resume.pt'), ('initial_checkpoint', 'initial-checkpoint.pt')]:
            bound_equal(artifacts[filename], candidate['parent'][kind], 'Candidate descriptor binding differs')
        require(candidate['parent']['initial_checkpoint'] == previous['policy'], 'Candidate initial parent differs')
        require(ledger['next_sequence'] == sequence + 1
                and ledger['consumed'][:len(original_ledger)] == original_ledger
                and [item['sequence'] for item in ledger['consumed'][len(original_ledger):]] == list(range(869, sequence + 1)),
                'Consumed-seed ledger changed or repeated a sequence')
        admitted = sequence < 974
        require(row['status'] == ('passed' if admitted else 'failed') and row['completed_games'] == 4
                and row['owned_cleanup_verified'] is True, 'Campaign admission or cleanup differs')
        if admitted:
            require(row['checkpoint_verified'] is True
                    and row['private_publication']['fresh_readback_verified'] is True
                    and row['private_publication']['checkpoint_roundtrip_verified'] is True
                    and row['private_publication']['manifest_sha256'] == artifacts['manifest.json']['sha256'],
                    'Originally admitted batch lacks publication proof')
        else:
            require('raw.part00000' in row['error'] and 'timed out after 120 seconds' in row['error']
                    and 'private_publication' not in row and 'checkpoint_verified' not in row,
                    'Pending batch was not the recorded publication-only failure')
        result = {'batch': name, 'sequence': sequence, 'originally_admitted': admitted,
                  'artifacts': artifacts, 'restore': proof}
        if sequence >= 973:
            result.update(full_state_fingerprints(folder / 'after-resume.pt'))
        results.append(result)
        previous, previous_progress = candidate['parent'], proof['progress']
        previous_steps = plan['parent']['optimizer_steps'] + report['optimizer_steps']
    require(latest['parent'] == read(repo / PREFIX / 'batches/train-000973-seat1/candidate.json')['parent']
            and latest['progress'] == results[-2]['restore']['progress']
            and pending['parent'] == previous and pending['progress'] == previous_progress
            and campaign['latest_parent'] == latest['parent'] and campaign['latest_progress'] == latest['progress'],
            'Closure latest/pending checkpoint chain differs')
    for prefix, result in [('', results[-2]), ('pending-', results[-1])]:
        for filename in ('plan.json', 'report.json', 'after-policy.pt', 'after-resume.pt', 'initial-checkpoint.pt'):
            closure_name = prefix + filename
            if prefix:
                closure_name = {'after-policy.pt': 'pending-policy.pt', 'after-resume.pt': 'pending-resume.pt',
                                'initial-checkpoint.pt': 'pending-initial_checkpoint.pt'}.get(filename, closure_name)
            require(closure_bindings[closure_name] == result['artifacts'][filename], 'Closure copy differs from batch')
    pending_folder = repo / PREFIX / 'batches/train-000974-seat0'
    raw_manifest = read(pending_folder / 'manifest.json')
    require(raw_manifest['entries']['raw.part00000'] == {'bytes': 5661006, 'sha256': RAW_SHA},
            'Investigated raw archive binding differs')
    raw = verify_raw_archive(pending_folder / 'raw.part00000', raw_manifest['raw_inventory'],
                             deadline=deadline, extract_to=extract_to, allowed_symlinks=DATA_LINKS)
    # Rebind every inspected input after restore and extraction; nothing historical is rewritten.
    for relative, artifacts in {PREFIX + '/closure': closure_bindings, **held_bindings}.items():
        for filename, expected in artifacts.items():
            require(binding(repo / relative / filename, deadline) == expected, 'Audit input changed')
    return {'schema': SCHEMA, 'status': 'passed', 'evidence_commit': COMMIT, 'source_commit': SOURCE,
            'prefix': PREFIX, 'historical_status': 'stopped_failure', 'historical_evidence_unchanged': True,
            'official_state_written': False, 'games_launched': 0, 'optimizer_updates_applied': 0,
            'original_admitted_updates': 105, 'original_admitted_games': 420,
            'latest_admitted_checkpoint': 1334, 'pending_checkpoint': 1335,
            'pending_originally_admitted': False, 'remaining_original_games_before_recovery': 2180,
            'remaining_original_games_if_pending_is_separately_certified': 2176,
            'next_sequence': 975, 'closure_artifacts': closure_bindings, 'batches': results, 'pending_raw': raw,
            'mac_behavior_reconstruction_performed': False,
            'limitations': ['Stored Linux admission evidence is retained; no new native replay or behavior reconstruction ran.',
                            'Fresh remote Git readback and official continuation promotion require separate certification.']}


@contextmanager
def wall_deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError('Read-only cloud publication audit exceeded its hard deadline')
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh private audit report path')
    parser.add_argument('--extract-raw', type=Path, help='Fresh private directory for regular files only')
    parser.add_argument('--deadline-seconds', type=int, default=600, choices=range(1, 601), metavar='1..600')
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), 'Audit output must be fresh')
    started = time.monotonic()
    try:
        with wall_deadline(args.deadline_seconds) as deadline:
            result = audit(args.evidence_repo, deadline=deadline, extract_to=args.extract_raw)
    except Exception as exc:
        result = {'schema': SCHEMA, 'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                  'historical_evidence_rewritten': False, 'official_state_written': False,
                  'games_launched': 0, 'optimizer_updates_applied': 0}
    result['wall_seconds'] = round(time.monotonic() - started, 3)
    with args.output.open('x') as stream:
        os.chmod(args.output, 0o600)
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'wall_seconds': result['wall_seconds'],
                      **({'error': result['error']} if 'error' in result else {})}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
