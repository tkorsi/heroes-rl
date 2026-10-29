#!/usr/bin/env python3
"""Certify a new manual continuation after the reviewed checkpoint1229 sleep stop.

This is deliberately not a general failed-run override. The committed profile
pins the investigated failure and its prior audits. Historical shutdown remains
failed; a separate receipt proves current closure and exact checkpoint recovery.
No game, update, signal, official pointer write or seed reservation occurs here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = ROOT / 'configs/interrupted1229-recovery.json'
SCHEMA = 'heroes-interrupted1229-recovery-v1'
MOVING = ('latest-run.json', 'latest-checkpoint.json', 'sequence-ledger.json')
FLAGS = dict(original_deadline_met=False, historical_shutdown_rewritten=False,
             fresh_manual_session_only=True, training_launched=False,
             games_launched=0, optimizer_updates=0, signals_sent=0)
VERIFY_LIMIT_SECONDS = 120
VERIFY_DEADLINE = ContextVar('interrupted_recovery_deadline', default=None)
ARTIFACT_ACCESS = ContextVar('interrupted_recovery_artifact_access', default=None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def remaining_time():
    deadline = VERIFY_DEADLINE.get()
    remaining = (min(deadline[0]-time.monotonic(), deadline[1]-time.time())
                 if deadline is not None else VERIFY_LIMIT_SECONDS)
    if remaining <= 0:
        raise TimeoutError('Read-only recovery verification exceeded its 120-second deadline')
    return remaining


def read(path):
    from experiments.headless.contextual_evidence import strict_json
    remaining_time()
    access = ARTIFACT_ACCESS.get()
    return access.read(path) if access is not None else strict_json(Path(path).read_bytes())


def binding(path):
    remaining_time()
    access = ARTIFACT_ACCESS.get()
    if access is not None:
        return access.binding(path)
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def profile_identity():
    """Reviewed bytes relocate between main and frozen source checkouts."""
    return {key: value for key, value in binding(PROFILE).items() if key != 'path'}


def bound(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'}
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and type(value['bytes']) is int and value['bytes'] >= 0,
            'Expected an exact absolute recovery artifact binding')
    require(binding(value['path']) == value, 'Recovery evidence changed: ' + value['path'])


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def reviewed_profile():
    for path in (PROFILE, Path(__file__).resolve()):
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(['git', 'show', 'HEAD:'+relative], cwd=ROOT,
                                   capture_output=True, check=True, timeout=min(10, remaining_time())).stdout
        require(committed == path.read_bytes(), 'Recovery helper and profile must be exactly committed')
    return read(PROFILE)


def fingerprint(value):
    """The prior audit's deterministic typed complete-state digest."""
    import torch
    digest = hashlib.sha256()
    def add(item):
        digest.update(type(item).__qualname__.encode() + b'\0')
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode() + b':')
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\xff')
    add(value)
    return digest.hexdigest()


def inventory(root, pattern='*'):
    access = ARTIFACT_ACCESS.get()
    if access is not None:
        return access.inventory(root, pattern)
    root = Path(root)
    paths = sorted(p for p in root.rglob(pattern) if p.is_file())
    require(all(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) for p in paths),
            'Recovery inventory contains a symlink or escaping file')
    return [binding(path) for path in paths]


@contextmanager
def existing_campaign_locks(config):
    """Hold the real exclusion locks without creating or truncating their files."""
    paths = [Path(config['artifact_root'])/'local-script-training/owner.lock',
             Path(config['run_root'])/'owner.lock']
    with ExitStack() as stack:
        held = []
        for path in dict.fromkeys(paths):
            require(not path.is_symlink(), 'Recovery lock cannot be a symlink')
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stream = stack.enter_context(os.fdopen(fd, 'rb'))
            require(stat.S_ISREG(os.fstat(fd).st_mode), 'Recovery lock must be an existing regular file')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('A campaign currently holds ' + str(path)) from exc
            held.append((path, fd, os.fstat(fd)))
        def unchanged():
            for path, fd, original in held:
                current = path.stat(follow_symlinks=False)
                require((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
                        and (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (original.st_dev, original.st_ino),
                        'Recovery lock path changed while held')
        unchanged()
        yield unchanged
        unchanged()


def process_closure(deadlines, run):
    """One bounded read-only snapshot; old numeric identities never grant signals."""
    access = ARTIFACT_ACCESS.get()
    if access is not None:
        return access.process_closure(deadlines, run)
    pids, groups = set(), set()
    for item in deadlines:
        bound(item)
        rows = [json.loads(line) for line in Path(item['path']).read_text().splitlines()]
        require(rows and all(type(row) is dict and row.get('schema') == 'heroes-process-deadline-v1'
                             for row in rows),
                'Missing or malformed deadline evidence')
        starts = [row for row in rows if row.get('event') == 'wrapper_started']
        require(not any(row.get('event') == 'launch_requested' for row in rows) or starts
                or any(row.get('event') == 'launch_failed' for row in rows),
                'Unresolved historical launch cannot establish process closure')
        for row in starts:
            group = row.get('process_group')
            require(type(group) is int and group > 1 and group == row.get('leader_pid') == row.get('writer_pid'),
                    'Deadline wrapper does not identify its private process group')
            groups.add(group)
        for row in rows:
            for key in ('writer_pid', 'leader_pid', 'guardian_pid'):
                value = row.get(key)
                if value is not None:
                    require(type(value) is int and value > 1, 'Invalid historical process identity')
                    pids.add(value)
    session = read(Path(run)/'session.json')
    require(type(session) is dict and type(session.get('supervisor_pid')) is int
            and session['supervisor_pid'] > 1, 'Malformed historical supervisor identity')
    pids.add(session['supervisor_pid'])
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat=,command='],
                            text=True, capture_output=True, check=True, timeout=min(10, remaining_time()))
    remaining_time()
    require(type(result.stdout) is str and result.stdout.strip(),
            'Current process inventory is empty or malformed')
    matches, examined, observed, observer_present = [], 0, set(), False
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 3)
        require(len(fields) == 4, 'Cannot parse current process identity')
        pid, group, status, command = int(fields[0]), int(fields[1]), fields[2], fields[3]
        require(pid > 0 and group >= 0 and pid not in observed,
                'Current process inventory has invalid or duplicate identities')
        observed.add(pid)
        examined += 1
        if 'Z' in status:
            continue
        observer_present = observer_present or pid == os.getpid()
        # Matching historical numbers conservatively fail, even if reused. No
        # attempt is made to infer ownership of an unrelated live process.
        if pid in pids or group in groups or (str(run) in command and pid != os.getpid()):
            matches.append(dict(pid=pid, process_group=group, status=status, command=command))
    require(observer_present, 'Current process inventory omitted its live observer')
    require(not matches, 'Live or ambiguous historical process identities prevent recovery: ' + repr(matches))
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), matched_processes=[],
                processes_examined=examined, observer_pid=os.getpid(), observer_present=True,
                recorded_pids=len(pids), recorded_groups=len(groups),
                historical_id_reuse_policy='fail_closed_without_signals', signals_sent=0)


def verify_batches(run, session, campaign, audit, ledger):
    from experiments.learner import local_script_training as local
    from experiments.learner import builtin_training_resume as resume
    rows = [row for row in campaign['batches'] if row['mode'] == 'training']
    require(len(rows) == 134 and [row['sequence'] for row in rows] == list(range(735, 869)),
            'The exact 134 reviewed training reservations are required')
    sequences = [row['sequence'] for row in ledger['consumed']]
    require(all(type(n) is int for n in sequences) and len(sequences) == len(set(sequences))
            and 336 in sequences and 555 in sequences and ledger['next_sequence'] == max(sequences)+1 == 869,
            'Consumed original seeds or next reservation changed')
    cursor, table, samples, optimizer_steps = session['initial_parent'], [], 0, 0
    for row in rows:
        sequence = row['sequence']
        require(row.get('status') == 'passed' and row.get('checkpoint_verified') is True
                and row.get('owned_cleanup_verified') is True and row['parent'] == cursor
                and sequences.count(sequence) == 1, 'Training admission or immediate ancestry differs')
        bound(row['plan']); bound(row['report'])
        plan, report = read(row['plan']['path']), read(row['report']['path'])
        require(plan.get('scope') == local.SCOPE and plan.get('profile') == 'original'
                and plan.get('mode') == 'training' and plan['parent_descriptor'] == cursor
                and plan['batch_sequence'] == sequence and plan['local_session'] == binding(run/'session.json')
                and len(plan['schedule']) == 4, 'Training plan left its original run or parent')
        for index, item in enumerate(plan['schedule']):
            require(item == dict(episode=index+1, phase='training', learner_player=sequence % 2,
                    engine_seed=40_000_000+4*sequence+index, policy_seed=50_000_000+4*sequence+index),
                    'A training seed or seat was changed')
        cursor = dict(plan=row['plan'], report=row['report'], policy=report['after_checkpoint']['policy'],
                      resume=report['after_checkpoint']['resume'],
                      initial_checkpoint=resume._metadata(plan, row['plan']['sha256'])['initial_checkpoint'])
        access = ARTIFACT_ACCESS.get()
        (access.descriptor if access is not None else local.read_descriptor)(cursor)
        preservation = row['preservation']
        target = run/'checkpoints'/Path(row['plan']['path']).parent.name
        require(Path(preservation['destination']) == target and read(target/'preservation-receipt.json') == preservation,
                'Preservation receipt or destination changed')
        expected = {'plan.json': cursor['plan'], 'report.json': cursor['report'],
                    'after-policy.pt': cursor['policy'], 'after-resume.pt': cursor['resume']}
        require(preservation['sources'] == expected
                and preservation['restore']['exact_model_adam_rng_progress_pool'] is True,
                'Preservation omitted exact full state')
        for name, original in expected.items():
            copied = binding(target/name)
            require(all(copied[k] == original[k] for k in ('sha256', 'bytes')), 'Durable checkpoint copy changed')
        table.append([sequence, row['plan']['sha256'], row['report']['sha256'],
                      binding(target/'preservation-receipt.json')['sha256']])
        samples += report['training_samples']; optimizer_steps += report['optimizer_steps']
    require(table == audit['admission_audit']['receipt_table'] and cursor == audit['latest_parent']
            and samples == 30061 and optimizer_steps == 1104,
            'Reviewed admitted batch totals, evidence or final parent changed')
    return cursor


def verify_state(parent, backup, expected):
    """Production exact restores, with caller RNG restored even on failure."""
    import torch
    from experiments.learner import local_script_training as local
    from experiments.learner.builtin_training_resume import _exact
    original_torch, original_python = torch.get_rng_state().clone(), random.getstate()
    try:
        snapshots = []
        for descriptor in (parent, backup):
            remaining_time()
            access = ARTIFACT_ACCESS.get()
            if access is not None:
                descriptor = access.rebind(descriptor)
            state = local.restore_parent(descriptor)
            remaining_time()
            snapshot = dict(model=dict(state.model.state_dict()), optimizer=state.optimizer.state_dict(),
                generator=state.generator.get_state(), pool=state.pool.state_dict(), progress=vars(state.progress),
                sampler=state.sampler_state, torch_rng=torch.get_rng_state(), python_rng=random.getstate())
            require(state.progress.updates == 1229 and state.optimizer_steps == 21238
                    and state.weights_sha256 == expected['weights_sha256']
                    and state.checkpoint == expected['checkpoint'], 'Restored checkpoint identity differs')
            require(fingerprint(snapshot) == expected['full_state_sha256']
                    and {k: fingerprint(v) for k, v in snapshot.items()} == expected['component_sha256'],
                    'Complete model/Adam/RNG/progress/sampler/pool state differs')
            snapshots.append(snapshot)
        _exact(snapshots[0], snapshots[1])
        return dict(exact=True, canonical_and_backup_exact=True, progress=snapshots[0]['progress'],
                    full_state_sha256=expected['full_state_sha256'],
                    component_sha256=expected['component_sha256'], optimizer_steps=21238)
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)


def read_moving_inputs(profile, run, snapshots=None):
    """Read current roots, or verify the receipt's immutable original bytes.

    Only the runner may select inherited snapshots after proving a clean later
    session carries this exact receipt and still has the recovered parent.
    Current ledger/lineage checks remain the runner's responsibility in that case.
    """
    require(type(profile.get('moving_evidence')) is dict and set(profile['moving_evidence']) == set(MOVING),
            'Recovery profile omitted an original moving input')
    if snapshots is not None:
        require(type(snapshots) is dict and set(snapshots) == set(MOVING)
                and all(type(value) is str for value in snapshots.values()),
                'Inherited recovery requires all exact original moving snapshots')
    result = {}
    for name in MOVING:
        expected = profile['moving_evidence'][name]
        require(Path(expected['path']) == Path(run).resolve().parent/name, 'Original moving input path differs')
        if snapshots is None:
            bound(expected)
            access = ARTIFACT_ACCESS.get()
            contents = (access.read_bytes(expected['path']) if access is not None
                        else Path(expected['path']).read_bytes())
            result[name] = contents.decode('utf-8')
        else:
            result[name] = snapshots[name]
            contents = result[name].encode('utf-8')
        require(len(contents) == expected['bytes'] and hashlib.sha256(contents).hexdigest() == expected['sha256'],
                'Original moving snapshot bytes changed: '+name)
    return result


def audit_current(run, config, *, moving_snapshots=None):
    """Recompute all gates; stored Boolean claims never substitute for evidence."""
    token = VERIFY_DEADLINE.set((time.monotonic()+VERIFY_LIMIT_SECONDS, time.time()+VERIFY_LIMIT_SECONDS))
    try:
        result = _audit_current(run, config, moving_snapshots=moving_snapshots)
        remaining_time()
        return result
    finally:
        VERIFY_DEADLINE.reset(token)


def _audit_current(run, config, *, moving_snapshots=None):
    profile = reviewed_profile()
    run = Path(run).expanduser().resolve()
    require(profile['schema'] == SCHEMA+'-profile' and str(run) == profile['run_dir']
            and config['default_profile'] == 'original' and Path(config['run_root']).resolve() == run.parent
            and Path(config['artifact_root']).resolve() == run.parents[2], 'Recovery is scoped to the reviewed original run1229')
    for item in profile['immutable_evidence'].values():
        bound(item)
    original = read_moving_inputs(profile, run, moving_snapshots)
    audit = read(profile['immutable_evidence']['checkpoint_audit']['path'])
    require(audit['state_audit']['full_state_sha256'] == profile['full_state_sha256'],
            'Reviewed full-state fingerprint differs from the pinned audit')
    session, campaign, shutdown = [read(run/name) for name in ('session.json', 'campaign-report.json', 'shutdown.json')]
    require(campaign['status'] == 'stopped_failure' and campaign.get('full_final_restore') is None
            and shutdown['worker_returncode'] == -9 and shutdown['within_hard_deadline'] is False,
            'The historical failure must remain honestly recorded')
    latest, newest, ledger = [json.loads(original[name]) for name in
                             ('latest-checkpoint.json', 'latest-run.json', 'sequence-ledger.json')]
    require(latest['run_dir'] == newest['run_dir'] == str(run)
            and latest['parent'] == campaign['latest_parent'] == audit['latest_parent']
            and latest['progress'] == campaign['latest_progress'] == audit['state_audit']['progress'],
            'Latest checkpoint or newest attempt differs from reviewed recovery')
    parent = verify_batches(run, session, campaign, audit, ledger)
    pending = run/'batches/periodic-001229-seat1'
    failed_files = inventory(pending)
    deadlines = inventory(run, '*deadline*.jsonl')
    for name, values in (('interrupted_cohort', failed_files), ('deadline_inventory', deadlines)):
        require(len(values) == profile[name]['count'] and json_digest(values) == profile[name]['sha256'],
                'Complete '+name+' evidence inventory changed')
    worker, final = read(pending/'execution/worker-report.json'), read(pending/'execution/report.json')
    require(worker['optimizer_updates'] == worker['training_samples'] == 0
            and worker['worker_complete'] is False and final['status'] == 'running'
            and [row['status'] for row in worker['attempts']] == ['passed']*3+['pending_verification']
            and not (inventory(pending, 'after-*.pt') if ARTIFACT_ACCESS.get() is not None
                     else list(pending.rglob('after-*.pt')))
            and not any(row.get('status') == 'passed' and row['mode'] == 'periodic'
                        and row.get('sequence') == 1229 and row['seat'] == 1 for row in campaign['batches']),
            'Interrupted evaluation cohort must remain excluded without any update')
    identity = audit['source_runtime']
    require(config['runtime_bin'] == session['runtime_bin'] == identity['runtime_bin'], 'Recovery runtime location changed')
    for key in ('machine', 'client', 'acceptance'):
        bound(identity[key])
    require(config == read(identity['machine']['path']), 'Recovery requires the reviewed immutable machine descriptor')
    for item in identity['maps'] + list(identity['frozen_files'].values()):
        bound(item)
    for name, sha in identity['runtime_sha256'].items():
        require(binding(Path(identity['runtime_bin'])/name)['sha256'] == sha, 'Actual runtime binary changed')
    for name, sha in identity['source_sha256'].items():
        require(binding(Path(identity['frozen_release'])/name)['sha256'] == sha, 'Frozen run source changed')
    state = verify_state(parent, audit['backup_parent'], audit['state_audit'])
    process = process_closure(deadlines, run)
    # Initial selection rechecks current roots under the caller's locks. An
    # inherited selection rechecks only the receipt's reviewed original bytes;
    # a subsequent clean session may legitimately advance the current roots.
    require(read_moving_inputs(profile, run, moving_snapshots) == original,
            'Original moving inputs changed during verification')
    for item in profile['immutable_evidence'].values():
        bound(item)
    # Source profile location changes between the implementation checkout, main
    # and a frozen release. The receipt binds its bytes, and the actual launch
    # separately pins its current source copy through reviewed_tree/frozen_files.
    evidence = dict(profile['immutable_evidence'])
    evidence.update({'parent/'+key: value for key, value in parent.items()})
    evidence.update({'backup/'+key: value for key, value in audit['backup_parent'].items()})
    return dict(run_dir=str(run), parent=parent, progress=state['progress'], state=state,
                admitted_original_games=2400, remaining_original_games=2600, next_sequence=869,
                original_moving_bindings=profile['moving_evidence'], original_moving_snapshots=original,
                immutable_evidence=evidence, deadline_inventory=deadlines, interrupted_cohort=failed_files,
                process_closure=process, **FLAGS)


def validate_receipt(receipt_path, run, config, *, inherited=False):
    """Validate under caller locks; inherited requires the runner's clean-session proof.

    The CLI never selects inherited mode. Supplying a receipt alone cannot skip
    the initial exact-current-pointer/ledger checks.
    """
    require(type(inherited) is bool, 'Inherited recovery selection must be explicit Boolean')
    receipt_path = Path(receipt_path).expanduser().resolve()
    saved_binding = binding(receipt_path)
    saved = read(receipt_path)
    require(saved.get('schema') == SCHEMA and saved.get('profile') == profile_identity(),
            'Unknown or changed reviewed recovery receipt profile')
    if inherited:
        verification = saved.get('verification')
        require(type(verification) is dict and type(verification.get('original_moving_snapshots')) is dict,
                'Inherited receipt omitted its original moving snapshots')
        actual = audit_current(run, config, moving_snapshots=verification['original_moving_snapshots'])
    else:
        actual = audit_current(run, config)
    # A new process observation is expected; all factual immutable proof must match.
    expected = {k: v for k, v in actual.items() if k != 'process_closure'}
    require(saved.get('verification') == expected, 'Recovery receipt facts differ from recomputed exact evidence')
    require(saved.get('creation_process_closure', {}).get('matched_processes') == []
            and saved.get('creation_process_closure', {}).get('signals_sent') == 0,
            'Receipt has no separate creation-time process closure')
    bound(saved_binding)
    return dict(parent=actual['parent'], progress=actual['progress'],
                closure_kind='posthoc_recovery', full_state_sha256=actual['state']['full_state_sha256'],
                evidence={**actual['immutable_evidence'], 'recovery_receipt': saved_binding},
                process_closure=actual['process_closure'])


def write_exclusive(path, value):
    """Publish complete bytes with no overwrite, including on interrupted writes."""
    path = Path(path)
    require(path.parent.is_dir() and not path.exists(), 'Recovery output needs an existing directory and a fresh filename')
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config_binding, config = binding(args.config), read(args.config)
    profile = read(PROFILE)
    output = args.output.expanduser().resolve()
    allowed = Path(config['artifact_root'])/'run-investigations'
    require(output.is_relative_to(allowed.resolve()) and output != allowed.resolve(),
            'Recovery receipt must be outside historical runs in the private investigation directory')
    with existing_campaign_locks(config) as check_locks:
        proof = audit_current(profile['run_dir'], config)
        check_locks(); bound(config_binding)
        receipt = dict(schema=SCHEMA, profile=profile_identity(), created_utc=datetime.now(timezone.utc).isoformat(),
            creation_process_closure=proof.pop('process_closure'), verification=proof)
        write_exclusive(output, receipt)
    print(json.dumps(dict(receipt=binding(output), progress=proof['progress'], remaining_games=2600,
                          next_sequence=869, training_launched=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
