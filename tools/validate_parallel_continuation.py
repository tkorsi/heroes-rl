#!/usr/bin/env python3
"""Bounded, disposable validation of four-game manual continuation collection.

This does not invoke the campaign runner or write any official checkpoint pointer.
Four fresh games per seat run through the serial production path and four more
through the parallel path. Every arm starts from the same complete copied parent.
The seeds differ between arms: timings are descriptive, never a paired speedup.
One isolated PPO update per arm and its checkpoint round trip exercise admission;
the resulting checkpoints remain excluded diagnostic evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = Path('/Users/orsi/Projects/Heroes-artifacts')
VALIDATION_ROOT = ARTIFACTS / 'parallel-continuation-validation-20261028'
DEFAULT_PARENT = ARTIFACTS / 'local-script-training/original/run-20261028T173849Z-7ee63337'
MACHINE = ARTIFACTS / 'local-script-training/machine-original.json'
FIRST_SEQUENCE = 415_000_000
SCHEMA = 'heroes-parallel-continuation-validation-v1'
ARMS = ((1, 0), (4, 0), (1, 1), (4, 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def write(path, value, *, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        write(temporary, value, new=True)
        os.replace(temporary, path)


def output_path(value):
    path, base = Path(value).expanduser().resolve(), VALIDATION_ROOT.resolve()
    require(not VALIDATION_ROOT.is_symlink(), 'Validation root must not redirect writes through a symlink')
    require(path != base and path.is_relative_to(base), 'Output must be a fresh child of ' + str(base))
    require(not path.exists(), 'Diagnostic attempts are never overwritten or resumed')
    return path


def reserve_sequences(output):
    """Consume four distant diagnostic sequences before any game; never retry."""
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = VALIDATION_ROOT / 'seed-ledger.json'
    with (VALIDATION_ROOT / 'seed-ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(ledger.read_text()) if ledger.exists() else {
            'schema': SCHEMA + '-seeds', 'next_sequence': FIRST_SEQUENCE, 'reservations': []}
        require(value.get('schema') == SCHEMA + '-seeds', 'Unknown diagnostic seed ledger')
        first = value['next_sequence']
        require(type(first) is int and FIRST_SEQUENCE <= first < FIRST_SEQUENCE + 100_000,
                'Diagnostic namespace exhausted')
        used = {arm['sequence'] for reservation in value['reservations'] for arm in reservation['arms']}
        require(not used.intersection(range(first, first+len(ARMS))), 'Diagnostic sequence was already consumed')
        rows = []
        for index, (workers, seat) in enumerate(ARMS):
            sequence = first + index
            schedule = [dict(episode=i+1, phase='training', learner_player=seat,
                engine_seed=40_000_000+4*sequence+i, policy_seed=50_000_000+4*sequence+i)
                for i in range(4)]
            rows.append(dict(workers=workers, seat=seat, sequence=sequence, schedule=schedule))
        reservation = dict(output=str(output), created_utc=utc(), arms=rows)
        value['next_sequence'] = first + len(ARMS)
        value['reservations'].append(reservation)
        write(ledger, value)
        directory = os.open(VALIDATION_ROOT, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return reservation


def copy_parent(descriptor, output):
    require(set(descriptor) == {'plan', 'report', 'policy', 'resume', 'initial_checkpoint'},
            'A complete five-file parent descriptor is required')
    target = Path(output) / 'parent'
    target.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, expected in descriptor.items():
        source = Path(expected['path'])
        require(binding(source) == expected, 'Original parent bytes changed: ' + name)
        path = target / (name + source.suffix)
        with source.open('rb') as src, path.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        copied[name] = binding(path)
        require(all(copied[name][key] == expected[key] for key in ('bytes', 'sha256')),
                'Copied parent differs: ' + name)
        require(binding(source) == expected, 'Original parent changed during copy: ' + name)
    return copied


def fingerprint(value):
    """Hash types, shapes, values and every nested tensor without pickle metadata."""
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


def official_snapshot(parent_run):
    paths = [MACHINE, parent_run/'session.json', parent_run/'campaign-report.json', parent_run/'shutdown.json']
    paths.extend(ARTIFACTS/'local-script-training/original'/name for name in
                 ('latest-checkpoint.json', 'latest-run.json', 'sequence-ledger.json'))
    return {str(path): binding(path) if path.exists() else None for path in paths}


def clean_source():
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit the complete candidate before native validation')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def closed_parent(run, expected_update):
    """Accept the verified budget stop without treating an active run as final."""
    run = Path(run).resolve()
    session, report, shutdown = [json.loads((run/name).read_text()) for name in
                                ('session.json', 'campaign-report.json', 'shutdown.json')]
    require(Path(session['run_dir']).resolve() == run and session.get('profile') == 'original',
            'Selected parent is not its declared original-map run')
    require(report.get('status') in ('finished', 'stopped_requested', 'stopped_budget')
            and report.get('owned_cleanup_verified') is True
            and report.get('initial_state_unchanged') is True
            and report.get('full_final_restore', {}).get('exact') is True
            and report.get('latest_progress', {}).get('updates') == expected_update
            and report.get('full_final_restore', {}).get('progress') == report.get('latest_progress')
            and shutdown.get('owned_cleanup_verified') is True
            and shutdown.get('within_hard_deadline') is True and shutdown.get('worker_returncode') == 0,
            'Selected parent is not cleanly stopped and completely restore-verified')
    return report['latest_parent']


def session_for(arm, config, descriptor, source, implementation, prerequisites, clock, output):
    from experiments.learner import local_script_training as local
    from experiments.learner import interaction_guard
    run = output / f"serial-seat{arm['seat']}" if arm['workers'] == 1 else output / f"parallel-seat{arm['seat']}"
    execution = local.execution_choice(overlapped_verification=False, prefix_helpers=0,
                                       text_leaf_replay=True, replay_threads=2,
                                       collection_workers=arm['workers'])
    return dict(schema=local.SESSION_SCHEMA, scope=local.SCOPE, source=source,
        implementation=implementation, prerequisites=prerequisites,
        configuration=local.configuration(descriptor), batch_wall_seconds=None,
        execution=execution, policy_support=interaction_guard.VERSION,
        asset_config=binding(output/'machine.json'), runtime_bin=config['runtime_bin'],
        runtime=config['runtime'], data=config['data'], maps=config['maps'],
        artifact_root=str(ARTIFACTS), run_root=str(output), run_dir=str(run),
        initial_parent=descriptor, next_sequence=arm['sequence'], profile='original',
        training_engine_seed_base=40_000_000, training_policy_seed_base=50_000_000,
        authorization='Isolated validation only; no official training or continuation publication',
        diagnostic_only=True, **clock)


def run_child(args):
    os.environ['HEROES_ARTIFACT_ROOT'] = str(ARTIFACTS)
    import torch
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    from experiments.learner import builtin_gameplay as gameplay
    from experiments.learner import builtin_training_resume as resume
    from experiments.learner import interaction_guard
    from experiments.learner.checkpoints import TrainingProgress
    from experiments.learner.ppo import PPOConfig
    from experiments.learner.parallel_collection import collect_batch
    from experiments.learner.contextual_rollout import behavior_guard_version
    from experiments.learner.inference import source_identity
    from tools.validate_interaction_guard import state_snapshot, verify_state_snapshot
    from tools.prepare_local_training import validate_config
    from validation.probes.validate_hero_policy import wall_limit
    from experiments.headless.process_deadline import cleanup_from_evidence
    torch.set_num_threads(1)
    output, parent_run = args.output.resolve(), args.parent_run.resolve()
    launch = json.loads((output/'launch.json').read_text())
    require(binding(__file__) == launch['harness'], 'Validation harness changed after launch')
    report = dict(schema=SCHEMA, status='running', started_utc=utc(), official_training_samples=0,
                  official_optimizer_updates=0, diagnostic_optimizer_updates=0, arms=[],
                  limitations=['Different fresh seeds per arm; no paired workload speedup or policy-equivalence claim.',
                               'Four games per arm exercise one checkpoint on the original map only.',
                               'The parallel evaluation dispatch has offline coverage only; fixed held-out seeds are not replayed here.',
                               'Native game rules and runtime are unchanged; no embedded engine is enabled.'])
    before = official_snapshot(parent_run)
    report['official_before'] = before
    # Admission remembers object identities. Keep consumed receipts alive until
    # every arm finishes so Python cannot reuse an old receipt's identity.
    consumed_receipts = []
    try:
        require(clean_source() == launch['source_commit'], 'Committed validation source changed')
        prepared = subprocess.run([sys.executable, str(ROOT/'tools/prepare_vcmi.py'), '--check'],
            cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
        report['prepared_engine_check'] = dict(returncode=prepared.returncode,
                                               stdout=prepared.stdout, stderr=prepared.stderr)
        config = json.loads(MACHINE.read_text())
        validate_config(config)
        require(config['default_profile'] == 'original'
                and Path(config['runtime_bin']).parent.name == 'local-v8-terminal-replay-repair1',
                'Use the accepted original map and current terminal-replay runtime')
        parent = closed_parent(parent_run, args.parent_update)
        descriptor = copy_parent(parent, output)
        write(output/'machine.json', config, new=True)
        state = local.restore_parent(descriptor)
        require(state.progress.updates == args.parent_update == 1095, 'This validation binds checkpoint 1095')
        initial = state_snapshot(state)
        report.update(parent_original=parent, parent_copy=descriptor,
                      parent_update=state.progress.updates, full_state_sha256=fingerprint(initial),
                      state_component_sha256={key:fingerprint(value) for key,value in initial.items()})
        reservation = reserve_sequences(output)
        report['reservation'] = reservation
        write(output/'preregistered.json', report, new=True)
        clock = {key: launch[key] for key in ('started_unix', 'started_monotonic',
            'compute_deadline_unix', 'compute_deadline_monotonic', 'absolute_deadline_unix', 'hard_deadline_monotonic')}
        budget = pilot.Budget(clock['compute_deadline_monotonic'], clock['compute_deadline_unix'])
        for arm in reservation['arms']:
            budget.remaining(60)
            require(clean_source() == launch['source_commit'], 'Source changed between diagnostic arms')
            session = session_for(arm, config, descriptor, source_identity(local.INTERFACE),
                                  pilot.implementation(), pilot.prerequisites(), clock, output)
            session_path = Path(session['run_dir'])/'session.json'
            write(session_path, session, new=True)
            plan_path = local.prepare_batch(session_path, descriptor, arm['sequence'], arm['seat'])
            # Finalize the generated plan's descriptive scope before binding or launching it.
            plan = json.loads(plan_path.read_text())
            plan.update(diagnostic_only=True,
                authorization={'scope': SCHEMA, 'instruction': 'Excluded diagnostic; never official training'},
                wall_scope='Independent 15-minute diagnostic budget including preparation and cleanup.',
                comparison_scope='Fresh distinct seeds per arm; descriptive timings only.')
            write(plan_path, plan)
            sha = binding(plan_path)['sha256']
            plan = local.read_plan(plan_path, sha)
            require(plan['schedule'] == arm['schedule'], 'Production schedule differs from reserved diagnostic seeds')
            state = local.worker_state(plan)
            verify_state_snapshot(initial, state)
            frozen = state_snapshot(state)
            row = dict(**arm, plan=binding(plan_path), status='running', attempts=[pilot.empty_attempt(r) for r in plan['schedule']])
            report['arms'].append(row)
            write(output/'summary.json', report)
            execution = Path(plan['execution_output']); execution.mkdir(parents=True, exist_ok=False)
            receipts, collected = [], None
            registries = (gameplay._RECEIPTS, gameplay._CAPTURES, gameplay._BINDINGS)
            registry_before = [set(registry) for registry in registries]
            started = time.monotonic()
            def save(index, attempt):
                row['attempts'][index] = dict(attempt)
                write(output/'summary.json', report)
            try:
                if arm['workers'] == 1:
                    for index in range(4):
                        attempt, receipt = pilot.run_attempt(plan, plan_path, sha, index, state.model,
                            state.checkpoint, budget.capped(plan['configuration']['attempt_wall_seconds']),
                            lambda value, i=index: save(i, value))
                        require(attempt['status'] == 'passed' and receipt is not None, 'Serial diagnostic failed')
                        receipts.append(receipt)
                else:
                    collected = collect_batch(plan_path, sha, state.model, state.checkpoint, workers=4,
                        deadline_monotonic=budget.deadline, deadline_wall=budget.wall_deadline, save=save)
                    receipts = list(collected.receipts)
                    row.update(collection_seconds=collected.collection_seconds,
                               verification_seconds=collected.verification_seconds)
                row['verified_batch_seconds'] = time.monotonic()-started
                require(len(receipts) == 4 and all(all(attempt.get(key) is True for key in
                    ('identity_verified', 'cleanup_verified', 'replay_verified', 'completion_receipt'))
                    and attempt['status'] == 'passed' for attempt in row['attempts']),
                    'Complete four-game native and policy verification required')
                require(all(behavior_guard_version(receipt.rollout.decisions) == interaction_guard.VERSION
                            for receipt in receipts), 'Visible interaction budget missing')
                verify_state_snapshot(frozen, state)
                row.update(full_frozen_state_unchanged=True, verified_receipts=4,
                    frozen_state_sha256=fingerprint(frozen),
                    postcollection_state_sha256=fingerprint(state_snapshot(state)),
                    rng_unchanged={'torch_cpu': True, 'python': True, 'minibatch_generator': True})
                cfg = plan['configuration']
                state.model.train().requires_grad_(True)
                update_started = time.monotonic()
                consumed_receipts.extend(receipts)
                with wall_limit(budget.remaining(cfg['cleanup_reserve_seconds'])):
                    result, rollouts = gameplay.native_update(state.model, state.optimizer, tuple(receipts),
                        plan_sha256=sha, config=PPOConfig(**cfg['ppo']), generator=state.generator,
                        remaining=lambda: budget.remaining(cfg['cleanup_reserve_seconds']),
                        max_optimizer_steps=cfg['max_optimizer_steps'])
                progress = TrainingProgress(state.progress.updates+1, state.progress.rollouts+4,
                                            state.progress.decisions+result.samples)
                sampler = dict(next_schedule_index=4, optimizer_steps=state.optimizer_steps+result.optimizer_steps,
                               batch_optimizer_steps=result.optimizer_steps)
                state.model.eval().requires_grad_(False)
                with wall_limit(budget.remaining(cfg['cleanup_reserve_seconds'])):
                    _, checkpoint = pilot.checkpoint_roundtrip(execution, state.model, state.optimizer,
                        state.pool, state.generator, resume._metadata(plan, sha), cfg, progress, sampler, 'diagnostic-after')
                row.update(diagnostic_ppo=asdict(result), diagnostic_checkpoint=checkpoint,
                           ppo_and_roundtrip_seconds=time.monotonic()-update_started)
                report['diagnostic_optimizer_updates'] += 1
                restored = local.restore_parent(descriptor)
                verify_state_snapshot(initial, restored)
                row.update(status='passed', full_original_restore_exact=True,
                           restored_state_sha256=fingerprint(state_snapshot(restored)))
                local.check_identity(plan)
            finally:
                if collected is not None:
                    collected.discard()
                # Revoke only capabilities minted in this diagnostic arm.
                for registry, old in zip(registries, registry_before):
                    for key in set(registry)-old:
                        registry.pop(key, None)
                row['finished_utc'] = utc()
                row['full_cycle_seconds'] = time.monotonic()-started
                write(output/'summary.json', report)
        require(clean_source() == launch['source_commit'], 'Source changed during validation')
        require(all(binding(v['path']) == v for v in parent.values()), 'Official parent changed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if report['arms'] and report['arms'][-1]['status'] == 'running':
            report['arms'][-1].update(status='failed', error=report['error'])
    finally:
        cleanup_errors = []
        for path in sorted(output.rglob('*deadline*.jsonl')):
            if path.name == 'study-deadline.jsonl':
                continue
            try:
                cleanup_from_evidence(path, timeout=2, deadline_wall=launch['absolute_deadline_unix'],
                                      deadline_monotonic=launch['hard_deadline_monotonic'])
            except Exception as error:
                cleanup_errors.append(f'{path.name}: {error}')
        report.update(cleanup_verified=not cleanup_errors, cleanup_errors=cleanup_errors,
                      official_after=official_snapshot(parent_run), finished_utc=utc())
        report['official_files_unchanged'] = before == report['official_after']
        if cleanup_errors or not report['official_files_unchanged']:
            report['status'] = 'failed'
        write(output/'summary.json', report)
    return 0 if report['status'] == 'passed' else 1


def launch(args):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output = output_path(args.output)
    require(360 <= args.total_seconds <= 900, 'Total diagnostic budget must be 360 to 900 seconds')
    source = clean_source()
    started_wall, started_mono = time.time(), time.monotonic()
    wall, mono = started_wall+args.total_seconds, started_mono+args.total_seconds
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=SCHEMA+'-launch', source_commit=source, harness=binding(__file__),
        started_unix=started_wall, started_monotonic=started_mono,
        absolute_deadline_unix=wall, hard_deadline_monotonic=mono,
        compute_deadline_unix=wall-300, compute_deadline_monotonic=mono-300,
        parent_run=str(args.parent_run.resolve()), parent_update=args.parent_update)
    write(output/'launch.json', receipt, new=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--output', str(output),
               '--parent-run', str(args.parent_run.resolve()), '--parent-update', str(args.parent_update)]
    process, rc, errors = None, None, []
    try:
        with (output/'worker.log').open('x') as log:
            process = guarded_popen(command, deadline_wall=wall, deadline_monotonic=mono,
                evidence_path=output/'study-deadline.jsonl', cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            rc = process.wait(timeout=max(.01, min(wall-time.time(), mono-time.monotonic())-10))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    finally:
        if process is not None:
            try:
                close_owned_process(process, timeout=3, deadline_wall=wall, deadline_monotonic=mono)
                for path in output.rglob('*deadline*.jsonl'):
                    cleanup_from_evidence(path, timeout=2, deadline_wall=wall, deadline_monotonic=mono)
            except Exception as error:
                errors.append(str(error))
        result = dict(returncode=rc, cleanup_verified=not errors and process is not None,
                      cleanup_errors=errors, within_hard_deadline=time.time() <= wall, finished_utc=utc())
        write(output/'shutdown.json', result, new=True)
    print(json.dumps(dict(output=str(output), **result), indent=2))
    return rc if rc is not None and result['cleanup_verified'] and result['within_hard_deadline'] else 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=VALIDATION_ROOT / (datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]))
    parser.add_argument('--parent-run', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--parent-update', type=int, default=1095)
    parser.add_argument('--total-seconds', type=int, default=900)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    return run_child(args) if args.child else launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
