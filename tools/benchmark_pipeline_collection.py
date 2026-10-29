#!/usr/bin/env python3
"""Fresh, excluded native pipeline diagnostics on a complete copied checkpoint.

Two four-game batches per arm cover both seats. Baseline and candidate use the
same newly reserved schedule; every actual game gets ordinary exact replay and
policy checks. Fresh receipts are revoked, never admitted or passed to PPO.
This standalone route neither opens official locks nor runs campaign entrypoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_copied_training as copies
from tools.benchmark_parallel_collection import write_new, replace_json, parent_rng_fingerprints

SCHEMA = 'heroes-pipeline-collection-diagnostic-v1'
BASE = copies.ARTIFACTS/'run-investigations/pipeline-collection'
ARMS = (('sequential-verification', False), ('pipelined-verification', True))
require, binding = copies.require, copies.binding


def utc():
    return datetime.now(timezone.utc).isoformat()


def diagnostic_output(path, *, fresh=False):
    path = copies.private_path(path, fresh=fresh)
    require(path != BASE and path.is_relative_to(BASE), 'Use a fresh pipeline diagnostic directory')
    return path


def reserve_schedule(output):
    """Durably consume a separate namespace before any attempt, including failures."""
    BASE.mkdir(parents=True, exist_ok=True)
    ledger = BASE/'seed-ledger.json'
    with (BASE/'seed-ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(ledger.read_text()) if ledger.exists() else dict(schema=SCHEMA+'-seeds', next_study=0, studies=[])
        require(value.get('schema') == SCHEMA+'-seeds', 'Wrong diagnostic seed ledger')
        study = value['next_study']
        require(type(study) is int and 0 <= study < 100_000, 'Diagnostic seed namespace exhausted')
        # Disjoint from official 40M/50M, historical 1.5B/1.6B benchmarks,
        # and the 415M-sequence continuation diagnostics (1.70B/1.71B).
        rows = [dict(episode=i+1, phase='training', learner_player=i//4,
                     engine_seed=1_730_000_000+study*8+i, policy_seed=1_830_000_000+study*8+i)
                for i in range(8)]
        used = {row[k] for old in value['studies'] for row in old['schedule'] for k in ('engine_seed', 'policy_seed')}
        require(not used.intersection(row[k] for row in rows for k in ('engine_seed', 'policy_seed')),
                'Diagnostic seeds were already consumed')
        reserved = dict(study=study, output=str(output), created_utc=utc(), schedule=rows,
                        paired_arms=[name for name, _ in ARMS], retries_permitted=False)
        value['studies'].append(reserved)
        value['next_study'] += 1
        replace_json(ledger, value)
        descriptor = os.open(BASE, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return reserved


@contextmanager
def local_lease(output):
    path = copies.no_links(Path(output)/'diagnostic.lock', regular=True)
    require(path.stat().st_size == 0, 'Diagnostic lease file changed')
    with path.open('rb') as lock:
        identity = os.fstat(lock.fileno())
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            current = path.stat()
            require((current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino),
                    'Diagnostic lease identity changed')


def verify_inputs(launch):
    for item in launch['inputs'].values():
        require(binding(item['path']) == item, 'Copied diagnostic input changed')
    require(copies.checkout_identity(ROOT) == launch['checkout'], 'Frozen diagnostic source changed')
    require(binding(__file__) == launch['harness'], 'Diagnostic driver changed')


def load_launch(output, expected_sha256):
    output = diagnostic_output(output)
    require(binding(output/'launch.json')['sha256'] == expected_sha256, 'Diagnostic launch changed')
    launch = json.loads((output/'launch.json').read_text())
    require(launch.get('schema') == SCHEMA+'-launch' and launch.get('diagnostic_only') is True
            and launch['output'] == str(output) and launch['optimization_eligible'] is False,
            'Only an excluded private diagnostic launch is supported')
    verify_inputs(launch)
    return launch


def collector_command(plan_path, sha, index, deadline, wall):
    return [sys.executable, str(Path(__file__).resolve()), 'collector', '--plan', str(plan_path),
            '--sha256', sha, '--index', str(index), '--deadline-monotonic', str(deadline), '--deadline-wall', str(wall)]


def collect_child(args):
    from experiments.learner import parallel_collection as parallel, interaction_guard
    from experiments.headless import background_check
    plan = parallel._read_plan(args.plan, args.sha256)
    require(plan.get('diagnostic_scope') == SCHEMA and plan.get('optimization_eligible') is False
            and plan.get('policy_support') == interaction_guard.VERSION, 'Wrong isolated collector scope')
    output = diagnostic_output(plan['diagnostic_output'])
    launch = load_launch(output, plan['diagnostic_launch_sha256'])
    require(Path(plan['execution_output']).is_relative_to(output)
            and plan['checkpoint'] == launch['parent_descriptor']['policy'], 'Collector escaped copied inputs/output')
    require(args.deadline_monotonic <= launch['hard_deadline_monotonic']
            and args.deadline_wall <= launch['hard_deadline_unix'], 'Collector cannot extend the study deadline')
    with local_lease(output):
        interaction_guard.ACTIVE_VERSION = interaction_guard.VERSION
        background_check.FORK = False
        # The unchanged production worker owns collection, native cleanup,
        # final identity checks and its private sealed completion protocol.
        return parallel._collect_worker(args)


def prepare(args, lease):
    from experiments.learner import local_script_training as local
    output = diagnostic_output(args.output, fresh=True)
    checkout = copies.checkout_identity(ROOT)
    engine_check = subprocess.run([sys.executable, str(ROOT/'tools/prepare_vcmi.py'), '--check'],
                                  cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    plan_path = copies.no_links(args.completed_plan, regular=True)
    original_plan = json.loads(plan_path.read_text())
    report_path = copies.no_links(plan_path.parent/'execution/report.json', regular=True)
    report = json.loads(report_path.read_text())
    require(report.get('plan_sha256') == binding(plan_path)['sha256'] and report.get('status') == 'finished'
            and report.get('update_status') == 'checkpoint_verified' and report.get('optimizer_updates') == 1
            and report.get('supervisor_cleanup_verified') is True, 'Select a completed verified batch')
    local.read_descriptor(original_plan['parent_descriptor'])
    require(original_plan.get('policy_support') == local.interaction_guard.VERSION,
            'The diagnostic preserves the declared visible interaction budget')
    require(120 <= args.seconds <= 1800, 'Use a bounded 120 to 1800 second study')
    output.mkdir(parents=True, mode=0o700)
    (output/'diagnostic.lock').touch(exist_ok=False)
    inventory, descriptor = {}, {}
    sources = {'completed-plan': binding(plan_path), 'completed-report': binding(report_path),
               'map': binding(original_plan['game_map']), **original_plan['parent_descriptor']}
    for name, source in sources.items():
        target = output/'input'/(name+Path(source['path']).suffix)
        copies.copy_verified(Path(source['path']), target, source)
        inventory[name] = binding(target)
        if name in local.DESCRIPTOR_KEYS:
            descriptor[name] = inventory[name]
    local.read_descriptor(descriptor)
    reservation = reserve_schedule(output)
    now_wall, now_mono = time.time(), time.monotonic()
    launch = dict(schema=SCHEMA+'-launch', diagnostic_only=True, optimization_eligible=False,
        output=str(output), checkout=checkout, harness=binding(__file__), created_utc=utc(),
        diagnostic_lease=lease, prepared_engine_source_check=dict(returncode=engine_check.returncode,
            stdout=engine_check.stdout, stderr=engine_check.stderr),
        inputs=inventory, original_inputs=sources, parent_descriptor=descriptor, reservation=reservation,
        configuration=original_plan['configuration'], policy_support=original_plan['policy_support'],
        binary=original_plan['binary'], replay_binary=original_plan['replay_binary'], data=original_plan['data'],
        runtime=original_plan['runtime'], hard_deadline_monotonic=now_mono+args.seconds,
        hard_deadline_unix=now_wall+args.seconds, maximum_seconds=args.seconds,
        optimizer_updates=0, training_samples_admitted=0)
    write_new(output/'launch.json', launch)
    return output, launch


def enrich_attempt(attempt):
    from experiments.learner.builtin_rollout import load_builtin_episode
    from experiments.learner.contextual_rollout import behavior_guard_version
    from experiments.learner.interaction_guard import VERSION
    folder = Path(attempt['episode_directory'])
    raw = json.loads((folder/'builtin-training-rollout.json').read_text())
    # Loading independently rederives every visible support/action invariant.
    episode = load_builtin_episode(raw['experience'])
    require(behavior_guard_version(episode.decisions) == VERSION, 'Guard missing from captured decisions')
    reconstructed = json.loads((folder/'training-policy-reconstruction.json').read_text())
    require(reconstructed['maximum_absolute_difference'] == 0
            and reconstructed['actions_resampled'] == attempt['sampled_choices'] == len(episode.decisions)
            and reconstructed['values_recomputed'] == len(episode.estimates)
            and reconstructed['private_memory_reset'] is True and reconstructed['weights_unchanged'] is True,
            'Incomplete exact policy reconstruction')
    return dict(attempt, reconstruction=dict(reconstructed, passed=True), guard_invariants_verified=True,
                guarded_decisions=len(episode.decisions), optimization_eligible=False,
                training_samples=0, training_samples_admitted=0, fresh_receipt_discarded=True)


def study_child(args):
    import torch
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    from experiments.learner import parallel_collection as parallel, interaction_guard, builtin_gameplay as gameplay
    from experiments.learner.inference import source_identity
    from experiments.headless import background_check
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    from tools.validate_interaction_guard import state_snapshot, verify_state_snapshot
    from tools.acceleration3.campaign_summary import game_digest
    os.nice(max(0, 15-os.getpriority(os.PRIO_PROCESS, 0)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = diagnostic_output(args.output)
    launch = load_launch(output, args.sha256)
    with local_lease(output):
        write_new(output/'study-claimed.json', dict(pid=os.getpid(), created_utc=utc()))
        interaction_guard.ACTIVE_VERSION = interaction_guard.VERSION
        background_check.FORK = False
        state = local.restore_parent(launch['parent_descriptor'])
        before = state_snapshot(state)
        original_command = parallel._worker_command
        source = source_identity(local.INTERFACE)
        implementation = pilot.implementation()
        for name in ('tools/benchmark_pipeline_collection.py', 'tools/benchmark_copied_training.py',
                     'tools/benchmark_parallel_collection.py', 'tools/isolated_diagnostic.py',
                     'tools/acceleration3/campaign_summary.py'):
            implementation[name] = binding(ROOT/name)['sha256']
        native = VcmiLauncher(Path(launch['binary']), Path(launch['data']), Path(launch['inputs']['map']['path']))
        native.preflight()
        require(runtime_identity(Path(launch['binary']).parent) == launch['runtime'], 'Frozen runtime changed')
        report = dict(schema=SCHEMA, status='running', started_utc=utc(), source=source, implementation=implementation,
            native=native.identity(), runtime=launch['runtime'], plan_sha256=args.sha256,
            launch=binding(output/'launch.json'),
            parent_descriptor=launch['parent_descriptor'], parent_progress=vars(state.progress),
            full_parent_restore=True, full_parent_unchanged=False, sampler_state_exact=False,
            training_performed=False, optimizer_updates=0, optimizer_steps=0, training_samples_admitted=0,
            optimization_eligible=False, policy_support=interaction_guard.VERSION, arms=[],
            hard_deadline_unix=launch['hard_deadline_unix'], cleanup_verified=False,
            limitations=['One baseline then one candidate arm; no counterbalancing or whole-training speed claim.',
                         'An active user run may compete for resources; it is never controlled by this diagnostic.',
                         'Equal seeds may produce different Nullkiller histories; semantic digests are compared.'])
        started = time.monotonic()
        replace_json(output/'report.json', report)
        try:
            parallel._worker_command = collector_command
            for name, pipelined in ARMS:
                arm = dict(name=name, pipeline_verification=pipelined, status='running', batches=[], attempts=[])
                report['arms'].append(arm)
                rng_before = parent_rng_fingerprints(torch)
                for seat in (0, 1):
                    verify_inputs(launch)
                    folder = output/name/f'seat-{seat}'
                    folder.mkdir(parents=True)
                    rows = launch['reservation']['schedule'][4*seat:4*(seat+1)]
                    plan = dict(schema='heroes-builtin-training-v1-plan', mode='training', diagnostic_scope=SCHEMA,
                        diagnostic_output=str(output), diagnostic_launch_sha256=args.sha256, optimization_eligible=False,
                        configuration=launch['configuration'], execution=dict(collection_workers=4, verification_workers=4,
                            pipeline_verification=pipelined), schedule=rows, source=source, implementation=implementation,
                        policy_support=interaction_guard.VERSION, execution_output=str(folder/'execution'),
                        checkpoint=launch['parent_descriptor']['policy'], checkpoint_source=state.metadata,
                        native=report['native'], runtime=launch['runtime'], binary=launch['binary'],
                        replay_binary=launch['replay_binary'], game_map=launch['inputs']['map']['path'], data=launch['data'])
                    write_new(folder/'plan.json', plan)
                    entry = dict(seat=seat, plan=binding(folder/'plan.json'), status='running')
                    arm['batches'].append(entry)
                    batch = None
                    try:
                        batch = parallel.collect_batch(folder/'plan.json', entry['plan']['sha256'], state.model,
                            state.checkpoint, workers=4, verification_workers=4, pipeline_verification=pipelined,
                            deadline_monotonic=launch['hard_deadline_monotonic']-30,
                            deadline_wall=launch['hard_deadline_unix']-30)
                        require(len(batch.receipts) == 4 and all(row['status'] == 'passed' for row in batch.attempts),
                                'A diagnostic batch is incomplete')
                        receipts = batch.receipts
                        batch.discard()
                        require(all(id(item) not in gameplay._RECEIPTS for item in receipts), 'Receipt revocation failed')
                        attempts = [enrich_attempt(row) for row in batch.attempts]
                        arm['attempts'].extend(attempts)
                        entry.update(status='passed', report=binding(folder/'execution/parallel-collection/result.json'),
                            collection_seconds=batch.collection_seconds, verification_seconds=batch.verification_seconds,
                            collection_elapsed_seconds=batch.collection_elapsed_seconds,
                            overlapped_verification_seconds=batch.overlapped_verification_seconds,
                            verified_batch_seconds=batch.verified_batch_seconds,
                            semantic_digests=[game_digest(row['episode_directory']) for row in attempts])
                    except BaseException as error:
                        entry.update(status='failed', error=type(error).__name__+': '+str(error))
                        arm['status'] = 'failed'
                        if hasattr(error, 'attempts'):
                            arm['attempts'].extend(dict(row, optimization_eligible=False,
                                training_samples_admitted=0) for row in error.attempts)
                        raise
                    finally:
                        if batch is not None:
                            batch.discard()
                        collection_report = folder/'execution/parallel-collection/result.json'
                        if collection_report.exists():
                            entry['report'] = binding(collection_report)
                        verify_state_snapshot(before, state)
                        replace_json(output/'report.json', report)
                arm.update(status='passed', frozen_model_unchanged=True,
                           parent_rng_unchanged=parent_rng_fingerprints(torch) == rng_before)
                require(arm['parent_rng_unchanged'], 'Diagnostic changed coordinator RNG state')
                arm['verified_batch_seconds'] = sum(row['verified_batch_seconds'] for row in arm['batches'])
            verify_state_snapshot(before, state)
            restored = local.restore_parent(launch['parent_descriptor'])
            verify_state_snapshot(before, restored)
            verify_inputs(launch)
            require(all(binding(item['path']) == item for item in launch['original_inputs'].values()),
                    'Original immutable inputs changed during the diagnostic')
            left, right = report['arms']
            semantic_match = ([digest for batch in left['batches'] for digest in batch['semantic_digests']] ==
                              [digest for batch in right['batches'] for digest in batch['semantic_digests']])
            ratio = left['verified_batch_seconds']/right['verified_batch_seconds']
            report.update(status='passed', full_parent_unchanged=True, sampler_state_exact=True,
                original_inputs_unchanged=True, diagnostic_lease=launch['diagnostic_lease'],
                comparison=dict(all_semantic_matches=semantic_match, observed_timing_ratio=ratio,
                                identical_workload_speedup=ratio if semantic_match else None))
        except BaseException as error:
            report.update(status='failed', error=type(error).__name__+': '+str(error))
        finally:
            parallel._worker_command = original_command
            from experiments.headless.process_deadline import cleanup_from_evidence
            try:
                for path in sorted(output.rglob('*deadline*.jsonl')):
                    cleanup_from_evidence(path, timeout=2, deadline_monotonic=launch['hard_deadline_monotonic'],
                                          deadline_wall=launch['hard_deadline_unix'])
                report['cleanup_verified'] = True
            except Exception as error:
                report.update(status='failed', cleanup_error=str(error))
            report.update(finished_utc=utc(), wall_seconds=time.monotonic()-started,
                          within_hard_deadline=time.time() <= launch['hard_deadline_unix'])
            replace_json(output/'report.json', report)
        return 0 if report['status'] == 'passed' else 1


def launch_owned(args, lease):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output, plan = prepare(args, lease)
    command = [sys.executable, str(Path(__file__).resolve()), 'study', '--output', str(output),
               '--sha256', binding(output/'launch.json')['sha256']]
    process, status, cleanup = None, None, False
    try:
        with local_lease(output), (output/'study.log').open('x') as log:
            process = guarded_popen(command, deadline_monotonic=plan['hard_deadline_monotonic'],
                deadline_wall=plan['hard_deadline_unix'], evidence_path=output/'study-guard.jsonl',
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status = process.wait(timeout=max(.1, plan['hard_deadline_unix']-time.time()-10))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=5, deadline_monotonic=plan['hard_deadline_monotonic'],
                                    deadline_wall=plan['hard_deadline_unix'])
            for path in output.rglob('*deadline*.jsonl'):
                cleanup_from_evidence(path, timeout=2, deadline_monotonic=plan['hard_deadline_monotonic'],
                                      deadline_wall=plan['hard_deadline_unix'])
            cleanup = True
        finally:
            write_new(output/'shutdown.json', dict(returncode=status, owned_cleanup_verified=cleanup,
                finished_utc=utc(), within_hard_deadline=time.time() <= plan['hard_deadline_unix'],
                optimizer_updates=0, training_samples_admitted=0))
    print(json.dumps(dict(output=str(output), returncode=status, owned_cleanup_verified=cleanup)), flush=True)
    return status if status is not None and cleanup else 124


def launch(args):
    from tools.isolated_diagnostic import diagnostic_lease
    with diagnostic_lease(args.output, isolated=True) as lease:
        return launch_owned(args, lease)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--completed-plan', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seconds', type=int, default=900)
    study = sub.add_parser('study', help=argparse.SUPPRESS)
    study.add_argument('--output', type=Path, required=True)
    study.add_argument('--sha256', required=True)
    child = sub.add_parser('collector', help=argparse.SUPPRESS)
    child.add_argument('--plan', type=Path, required=True)
    child.add_argument('--sha256', required=True)
    child.add_argument('--index', type=int, required=True)
    child.add_argument('--deadline-monotonic', type=float, required=True)
    child.add_argument('--deadline-wall', type=float, required=True)
    args = parser.parse_args()
    return {'run': launch, 'study': study_child, 'collector': collect_child}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
