#!/usr/bin/env python3
"""Eight disposable frozen-policy native diagnostics; no training/update path.

Requires an explicit stopped official parent and a fresh output directory. The
parent is copied byte-for-byte. An independent guard bounds this process and
native children. Native replay and independent policy reconstruction are those
of the accepted built-in-opponent evaluator, without PPO admission.
"""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def state_snapshot(state):
    import torch
    return deepcopy(dict(model=dict(state.model.state_dict()), optimizer=state.optimizer.state_dict(),
                         generator=state.generator.get_state(), pool=state.pool.state_dict(),
                         torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                         sampler=state.sampler_state, progress=vars(state.progress)))


def verify_state_snapshot(before, state):
    from experiments.learner.gameplay import assert_same_state
    assert_same_state(before, state_snapshot(state))


def write_shutdown(output, rc, cleanup):
    write(output/'shutdown.json', dict(returncode=rc, **cleanup))


def configure_artifact_root(value):
    """Bind preservation defaults before importing any learner dependency."""
    root = Path(value).expanduser().resolve()
    if not Path(value).expanduser().is_absolute() or not root.is_dir():
        raise ValueError('Diagnostic artifact root must be an existing absolute directory')
    loaded = sys.modules.get('experiments.learner.builtin_preservation')
    if loaded is not None and loaded.DURABLE_ROOT != root:
        raise ValueError('Preservation was already imported with a different artifact root')
    os.environ['HEROES_ARTIFACT_ROOT'] = str(root)
    return root


def diagnostic_schedule(engine_base, policy_base):
    if (type(engine_base) is not int or type(policy_base) is not int
            or not 83_400_000 <= engine_base < 2**31 - 8 or not 83_400_000 <= policy_base < 2**31 - 8
            or abs(engine_base - policy_base) < 8):
        raise ValueError('Candidate diagnostics require distinct explicit fresh seed ranges starting at83400000 or above')
    return [dict(episode=i+1, phase='evaluation', learner_player=i//4,
                 engine_seed=engine_base+i, policy_seed=policy_base+i) for i in range(8)]


def reserve_diagnostic_seeds(path, schedule, *, output, parent_run, candidate):
    """Consume the diagnostic seed ranges under the already held official lock.

    Failed and unstarted attempts keep their reservations; this ledger never
    participates in the official training sequence namespace.
    """
    path = Path(path)
    old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    keys = ('engine_seed', 'policy_seed')
    wanted = {row[key] for row in schedule for key in keys}
    used = {row[key] for entry in old for row in entry['schedule'] for key in keys}
    if wanted & used:
        raise ValueError('Candidate diagnostic seeds were already consumed; use a fresh explicit range')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(schema='heroes-garrison-policy-seed-reservation-v1', created_utc=utc(),
            output=str(output), parent_run=str(parent_run), runtime_candidate=candidate,
            schedule=schedule), sort_keys=True) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def load_runtime_candidate(path, expected=None):
    from tools.prepare_local_training import binding, candidate_runtime_origin, require
    from experiments.headless.contextual_evidence import runtime_identity
    bound = binding(path)
    require(expected is None or bound == expected, 'Candidate runtime descriptor changed')
    value = json.loads(Path(bound['path']).read_text())
    require(value.get('schema') == 'heroes-local-training-runtime-candidate-v1'
            and value.get('change_classification') in ('adapter-support-extension', 'replay-verifier-support-extension'),
            'Expected an explicit supported runtime candidate descriptor; not machine acceptance')
    runtime_bin = Path(value['runtime_bin']).expanduser().resolve()
    native = runtime_identity(runtime_bin)
    candidate_runtime_origin(runtime_bin, native, value)
    return value, bound


def stopped_candidate_parent(run, config, expected_update=770):
    """Bind one explicitly selected closed parent; caller holds campaign locks."""
    from experiments.learner.builtin_training_resume import require
    require(type(expected_update) is int and expected_update > 0, 'Expected parent update must be a positive integer')
    run = Path(run).expanduser().resolve()
    if run == Path(config['run_root']).resolve() / 'cloud-import-1335':
        from tools import run_local_training as runner
        from tools import cloud_state_import as cloud
        require(expected_update == 1335 and config.get('default_profile') == 'original',
                'Imported candidate selection is pinned to the certified original checkpoint1335')
        parent, _, proof = runner.original_continuation(
            run.parent / 'run-20261026T181133Z-062cd75a', config)
        latest = cloud.read(run.parent / 'latest-checkpoint.json')
        require(latest['run_dir'] == str(run) and latest['parent'] == parent
                and latest['progress']['updates'] == 1335
                and latest['cloud_import'] == proof['evidence'][cloud.RECEIPT_KEY],
                'Frozen diagnostic must select the exact current certified import boundary')
        return parent
    session, report, shutdown = [json.loads((run/name).read_text()) for name in
                                ('session.json', 'campaign-report.json', 'shutdown.json')]
    require(config.get('default_profile') == 'original' and run.parent == Path(config['run_root']).resolve()
            and Path(session['run_dir']).resolve() == run and Path(session['run_root']).resolve() == run.parent
            and session.get('profile') == 'original', 'Candidate parent must be the closed official original-map run')
    require(report.get('status') in ('finished', 'stopped_requested', 'stopped_budget', 'stopped_failure')
            and report.get('owned_cleanup_verified') is True and report.get('initial_state_unchanged') is True
            and report.get('full_final_restore', {}).get('exact') is True
            and report.get('latest_progress', {}).get('updates') == expected_update
            and shutdown.get('owned_cleanup_verified') is True and shutdown.get('within_hard_deadline') is True
            and shutdown.get('worker_returncode') in (0, 1),
            'Candidate requires the exact selected stopped official parent, owned cleanup and full preservation')
    return report['latest_parent']


def check_import_provenance(plan):
    """The copied1335 parent must still be the current selection after lock transfer."""
    from experiments.learner.builtin_training_resume import binding, require
    receipt = plan.get('cloud_import')
    if receipt is None:
        require('cloud_import_moving' not in plan, 'Import moving proof lacks its receipt')
        return
    require(plan.get('runtime_candidate') is not None and plan.get('parent_update') == 1335
            and binding(receipt['path']) == receipt, 'Certified cloud import provenance changed')
    root = Path(receipt['path']).parent
    moving = plan.get('cloud_import_moving')
    require(type(moving) is dict and set(moving) == {'latest-run.json','latest-checkpoint.json','sequence-ledger.json'},
            'Imported diagnostic omitted its current selection bindings')
    require(all(Path(value['path']) == root/name and binding(value['path']) == value
                for name,value in moving.items()), 'Official selection changed during diagnostic lock transfer')


def execute(plan_path, sha):
    content = plan_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sha:
        raise ValueError('Diagnostic plan changed')
    plan = json.loads(content)
    # Fresh worker imports must use the same durable root as preparation; the
    # preservation module captures it in function defaults at import time.
    configure_artifact_root(plan['artifact_root'])
    import torch
    from experiments.learner import builtin_first_run as pilot
    from experiments.learner import interaction_guard as guard
    from experiments.learner import local_script_training as local
    from experiments.learner.gameplay import assert_same_state
    from experiments.learner.builtin_training_resume import binding, require
    from experiments.learner.contextual_rollout import behavior_guard_version
    from experiments.learner.builtin_rollout import load_builtin_episode
    from experiments.headless import background_check
    from experiments.headless.process_deadline import cleanup_from_evidence
    torch.set_num_threads(1)
    require(binding(plan_path)['sha256'] == sha, 'Diagnostic plan changed')
    lock = os.open(plan['official_lock'], os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(plan['policy_support'] == guard.VERSION and plan['mode'] == 'evaluation'
            and len(plan['schedule']) == 8, 'Wrong diagnostic scope')
    candidate = plan.get('runtime_candidate')
    check_import_provenance(plan)
    expected_schedule = (diagnostic_schedule(plan['engine_seed_base'], plan['policy_seed_base']) if candidate else
        [dict(episode=i+1, phase='evaluation', learner_player=i//4,
              engine_seed=82010000+i, policy_seed=82050000+i) for i in range(8)])
    require(plan['schedule'] == expected_schedule, 'Schedule changed')
    if candidate:
        require(binding(ROOT/'tools/prepare_local_training.py') == plan['runtime_gate'], 'Candidate runtime gate changed')
        load_runtime_candidate(candidate['path'], candidate)
    guard.ACTIVE_VERSION = guard.VERSION
    background_check.FORK = False
    output = Path(plan['execution_output'])
    report = dict(schema='heroes-garrison-policy-diagnostic-v1' if candidate else 'heroes-interaction-guard-diagnostic-v1', started_utc=utc(),
                  plan_sha256=sha, status='running', parent_update=plan.get('parent_update', 770) if candidate else 717,
                  training_samples=0, training_performed=False, optimizer_steps=0,
                  attempts=[pilot.empty_attempt(row) for row in plan['schedule']])
    report_path = output / 'report.json'
    write(report_path, report)
    state = local.restore_parent(plan['parent_descriptor'])
    require(state.progress.updates == (plan.get('parent_update', 770) if candidate else 717), 'Frozen diagnostic parent changed')
    before = state_snapshot(state)
    # ParentState's pool field is checked again by the exact historical loader.
    try:
        for i, row in enumerate(plan['schedule']):
            require(binding(plan_path)['sha256'] == sha, 'Diagnostic plan changed')
            require(binding(__file__) == plan['harness'], 'Diagnostic harness changed')
            if candidate:
                require(binding(ROOT/'tools/prepare_local_training.py') == plan['runtime_gate'], 'Candidate runtime gate changed')
                load_runtime_candidate(candidate['path'], candidate)
            pilot.check_identity(plan)
            remaining = min(plan['compute_deadline_unix']-time.time(),
                            plan['compute_deadline_monotonic']-time.monotonic())
            require(remaining > 40, 'Diagnostic reserve reached')
            def save(value):
                report['attempts'][i] = value
                write(report_path, report)
            attempt, receipt = pilot.run_attempt(plan, plan_path, sha, i, state.model,
                state.checkpoint, pilot.Budget(time.monotonic()+min(600, remaining),
                                              time.time()+min(600, remaining)), save)
            require(receipt is None and attempt['training_samples'] == 0, 'Diagnostic became training')
            require(attempt['status'] == 'passed', 'Diagnostic failed; no retry or replacement')
            episode = load_builtin_episode(Path(attempt['episode_directory'])/'builtin-rollout.json')
            require(behavior_guard_version(episode.decisions) == guard.VERSION, 'Guard missing from actual behavior')
            # Loading independently rederives and checks every support/action.
            attempt['guard_invariants_verified'] = True
            attempt['guarded_decisions'] = len(episode.decisions)
            save(attempt)
        verify_state_snapshot(before, state)
        restored = local.restore_parent(plan['parent_descriptor'])
        assert_same_state(before['model'], dict(restored.model.state_dict()))
        assert_same_state(before['optimizer'], restored.optimizer.state_dict())
        if candidate:
            verify_state_snapshot(before, restored)
            require(binding(ROOT/'tools/prepare_local_training.py') == plan['runtime_gate'], 'Candidate runtime gate changed')
            load_runtime_candidate(candidate['path'], candidate)
        check_import_provenance(plan)
        report.update(status='passed', full_parent_unchanged=True, full_parent_restore=True,
                      parent_progress=vars(restored.progress))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        errors = []
        for path in output.glob('native-deadline-*.jsonl'):
            try: cleanup_from_evidence(path, timeout=3,
                    deadline_wall=plan['absolute_deadline_unix'],
                    deadline_monotonic=plan['hard_deadline_monotonic'])
            except Exception as error: errors.append(str(error))
        report.update(finished_utc=utc(), cleanup_verified=not errors, cleanup_errors=errors)
        if errors: report['status'] = 'failed'
        for row in report['attempts']:
            if row['status'] == 'not_started': row['reason'] = report.get('error', 'not_started')
        write(report_path, report)
        os.close(lock)
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parent-run', type=Path)
    parser.add_argument('--machine', type=Path)
    parser.add_argument('--parent-update', type=int, default=770,
                        help='Exact closed candidate parent update; bound into the immutable diagnostic plan')
    parser.add_argument('--runtime-candidate', type=Path,
                        help='Native-validated frozen candidate descriptor; diagnostics only, never machine acceptance')
    parser.add_argument('--engine-seed-base', type=int)
    parser.add_argument('--policy-seed-base', type=int)
    parser.add_argument('--worker-plan', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.worker_plan:
        try:
            return execute(args.worker_plan, args.sha256)
        except Exception as error:
            plan = json.loads(args.worker_plan.read_text())
            write(Path(plan['execution_output'])/'startup-failure.json',
                  dict(timestamp_utc=utc(), error=f'{type(error).__name__}: {error}',
                       training_samples=0, optimizer_steps=0))
            raise
    if not (args.output and args.parent_run and args.machine):
        parser.error('Output, stopped parent and machine are required')
    config = json.loads(args.machine.read_text())
    configure_artifact_root(config['artifact_root'])
    from experiments.learner import local_script_training as local
    from experiments.learner import builtin_first_run as pilot
    from experiments.learner import interaction_guard as guard
    from experiments.learner.builtin_training_resume import binding, require
    from experiments.learner.inference import source_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.builtin_v8_migration import map_identity
    from experiments.headless.process_deadline import guarded_popen, close_owned_process
    from run_local_training import final_parent
    from prepare_local_training import validate_config, original_map_identity, durable
    candidate = None
    schedule = [dict(episode=i+1, phase='evaluation', learner_player=i//4,
                     engine_seed=82010000+i, policy_seed=82050000+i) for i in range(8)]
    if args.runtime_candidate:
        runtime, candidate = load_runtime_candidate(args.runtime_candidate)
        schedule = diagnostic_schedule(args.engine_seed_base, args.policy_seed_base)
        require(config.get('default_profile') == 'original' and len(config.get('maps', [])) == 2
                and config['maps'][0] == config['maps'][1]
                and original_map_identity(config['maps'][0]['path']) == config['maps'][0],
                'Candidate diagnostics require the unchanged official original map')
        artifact_root, run_root = durable(config['artifact_root']), durable(config['run_root'])
        require(run_root.is_relative_to(artifact_root) and run_root != artifact_root,
                'Candidate machine paths are outside the official artifact scope')
        config = {**config, 'runtime_bin': runtime['runtime_bin'],
                  'runtime': {key: value for key, value in runtime['runtime_files'].items() if key != 'vcmiclient'}}
    else:
        require(args.engine_seed_base is None and args.policy_seed_base is None and args.parent_update == 770,
                'Historical diagnostics keep their exact original seed schedule')
        validate_config(config)
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Freeze a clean committed release before diagnostics')
    from tools.recover_interrupted_training import existing_campaign_locks
    with existing_campaign_locks(config):
        parent = stopped_candidate_parent(args.parent_run, config, args.parent_update) if candidate else final_parent(args.parent_run.resolve())
        output = args.output.resolve()
        if candidate:
            durable(output)
            require(not output.exists(), 'Preserve previous diagnostic attempts; output already exists')
            reserve_diagnostic_seeds(Path(config['artifact_root'])/'garrison-policy-diagnostics/seed-ledger.jsonl',
                schedule, output=output, parent_run=args.parent_run.resolve(), candidate=candidate)
        output.mkdir(parents=True, exist_ok=False)
        copied = output/'parent'; copied.mkdir()
        descriptor = {}
        for name, value in parent.items():
            destination = copied/(name + Path(value['path']).suffix)
            shutil.copyfile(value['path'], destination)
            descriptor[name] = binding(destination)
            require(descriptor[name]['sha256'] == value['sha256'], 'Parent copy changed')
        state = local.restore_parent(descriptor)
        require(state.progress.updates == (args.parent_update if candidate else 717), 'This protocol binds its exact stopped parent')
        binary = Path(config['runtime_bin'])/'vcmiclient'
        game_map = Path(config['maps'][0]['path'])
        started_wall, started_mono = time.time(), time.monotonic()
        plan = dict(schema='heroes-garrison-policy-diagnostic-plan-v1' if candidate else 'heroes-interaction-guard-diagnostic-plan-v1', mode='evaluation',
            scope='visible-interaction-budget-v1-diagnostic', policy_support=guard.VERSION,
            source=source_identity(local.INTERFACE), implementation=pilot.implementation(),
            prerequisites=pilot.prerequisites(), configuration=local.configuration(descriptor),
            runtime=config['runtime'], binary=str(binary), data=config['data'], artifact_root=config['artifact_root'],
            game_map=str(game_map), map=map_identity(game_map),
            native=VcmiLauncher(binary, Path(config['data']), game_map).identity(),
            checkpoint=descriptor['policy'], replay_binary=binding(binary.parent/'vcmireplay'),
            parent_descriptor=descriptor, harness=binding(__file__), execution_output=str(output/'execution'),
            official_lock=str(Path(config['artifact_root'])/'local-script-training/owner.lock'),
            schedule=schedule,
            started_unix=started_wall, started_monotonic=started_mono,
            compute_deadline_unix=started_wall+2640, compute_deadline_monotonic=started_mono+2640,
            absolute_deadline_unix=started_wall+2700, hard_deadline_monotonic=started_mono+2700)
        if candidate:
            plan.update(runtime_candidate=candidate, parent_update=args.parent_update, engine_seed_base=args.engine_seed_base,
                        policy_seed_base=args.policy_seed_base, runtime_gate=binding(ROOT/'tools/prepare_local_training.py'),
                        scope=('terminal-replay-frozen-policy-v1-diagnostic'
                               if runtime.get('change_classification') == 'replay-verifier-support-extension'
                               else 'map-garrison-frozen-policy-v1-diagnostic'))
            if args.parent_run.resolve() == Path(config['run_root']).resolve() / 'cloud-import-1335':
                plan['cloud_import'] = binding(Path(config['run_root']) / 'cloud-import.json')
                plan['cloud_import_moving'] = {name:binding(Path(config['run_root'])/name)
                    for name in ('latest-run.json','latest-checkpoint.json','sequence-ledger.json')}
        (output/'execution').mkdir()
        write(output/'plan.json', plan)
        command = [sys.executable, __file__, '--worker-plan', str(output/'plan.json'),
                   '--sha256', binding(output/'plan.json')['sha256']]
        # The independently guarded worker acquires its own lifetime lock before any
        # game. If a user launch wins this transfer race, the diagnostic refuses.
    with (output/'worker.log').open('w') as log:
        process = guarded_popen(command, deadline_wall=plan['absolute_deadline_unix'],
            deadline_monotonic=plan['hard_deadline_monotonic'], evidence_path=output/'deadline.jsonl',
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rc = None
        try: rc = process.wait(timeout=2700)
        finally:
            from run_local_training import cleanup_owned
            cleanup = cleanup_owned(plan, process=process, evidence_root=output)
            write_shutdown(output, rc, cleanup)
            if not cleanup['owned_cleanup_verified']: rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
