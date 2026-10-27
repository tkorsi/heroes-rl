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


def execute(plan_path, sha):
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
    plan = json.loads(plan_path.read_text())
    require(binding(plan_path)['sha256'] == sha, 'Diagnostic plan changed')
    lock = os.open(plan['official_lock'], os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(plan['policy_support'] == guard.VERSION and plan['mode'] == 'evaluation'
            and len(plan['schedule']) == 8, 'Wrong diagnostic scope')
    require(plan['schedule'] == [dict(episode=i+1, phase='evaluation', learner_player=i//4,
            engine_seed=82010000+i, policy_seed=82050000+i) for i in range(8)], 'Schedule changed')
    guard.ACTIVE_VERSION = guard.VERSION
    background_check.FORK = False
    output = Path(plan['execution_output'])
    report = dict(schema='heroes-interaction-guard-diagnostic-v1', started_utc=utc(),
                  plan_sha256=sha, status='running', training_samples=0, optimizer_steps=0,
                  attempts=[pilot.empty_attempt(row) for row in plan['schedule']])
    report_path = output / 'report.json'
    write(report_path, report)
    state = local.restore_parent(plan['parent_descriptor'])
    before = deepcopy(dict(model=state.model.state_dict(), optimizer=state.optimizer.state_dict(),
                           generator=state.generator.get_state(), pool=state.pool.state_dict(),
                           torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                           sampler=state.sampler_state, progress=vars(state.progress)))
    # ParentState's pool field is checked again by the exact historical loader.
    try:
        for i, row in enumerate(plan['schedule']):
            require(binding(plan_path)['sha256'] == sha, 'Diagnostic plan changed')
            require(binding(__file__) == plan['harness'], 'Diagnostic harness changed')
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
        assert_same_state(before['model'], dict(state.model.state_dict()))
        assert_same_state(before['optimizer'], state.optimizer.state_dict())
        assert_same_state(before['generator'], state.generator.get_state())
        assert_same_state(before['pool'], state.pool.state_dict())
        assert_same_state(before['torch_rng'], torch.get_rng_state())
        assert_same_state(before['python_rng'], random.getstate())
        assert_same_state(before['sampler'], state.sampler_state)
        assert_same_state(before['progress'], vars(state.progress))
        restored = local.restore_parent(plan['parent_descriptor'])
        assert_same_state(before['model'], dict(restored.model.state_dict()))
        assert_same_state(before['optimizer'], restored.optimizer.state_dict())
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
    from experiments.learner import local_script_training as local
    from experiments.learner import builtin_first_run as pilot
    from experiments.learner import interaction_guard as guard
    from experiments.learner.builtin_training_resume import binding, require
    from experiments.learner.inference import source_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.builtin_v8_migration import map_identity
    from experiments.headless.process_deadline import guarded_popen, close_owned_process
    from run_local_training import final_parent
    from prepare_local_training import validate_config
    require(args.output and args.parent_run and args.machine, 'Output, stopped parent and machine are required')
    config = json.loads(args.machine.read_text()); validate_config(config)
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Freeze a clean committed release before diagnostics')
    parent = final_parent(args.parent_run.resolve())
    lock = os.open(Path(config['artifact_root'])/'local-script-training/owner.lock', os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = output/'parent'; copied.mkdir()
    descriptor = {}
    for name, value in parent.items():
        destination = copied/(name + Path(value['path']).suffix)
        shutil.copyfile(value['path'], destination)
        descriptor[name] = binding(destination)
        require(descriptor[name]['sha256'] == value['sha256'], 'Parent copy changed')
    state = local.restore_parent(descriptor)
    require(state.progress.updates == 717, 'This protocol binds the stopped update717 parent')
    binary = Path(config['runtime_bin'])/'vcmiclient'
    game_map = Path(config['maps'][0]['path'])
    started_wall, started_mono = time.time(), time.monotonic()
    plan = dict(schema='heroes-interaction-guard-diagnostic-plan-v1', mode='evaluation',
        scope='visible-interaction-budget-v1-diagnostic', policy_support=guard.VERSION,
        source=source_identity(local.INTERFACE), implementation=pilot.implementation(),
        prerequisites=pilot.prerequisites(), configuration=local.configuration(descriptor),
        runtime=config['runtime'], binary=str(binary), data=config['data'],
        game_map=str(game_map), map=map_identity(game_map),
        native=VcmiLauncher(binary, Path(config['data']), game_map).identity(),
        checkpoint=descriptor['policy'], replay_binary=binding(binary.parent/'vcmireplay'),
        parent_descriptor=descriptor, harness=binding(__file__), execution_output=str(output/'execution'),
        official_lock=str(Path(config['artifact_root'])/'local-script-training/owner.lock'),
        schedule=[dict(episode=i+1, phase='evaluation', learner_player=i//4,
                       engine_seed=82010000+i, policy_seed=82050000+i) for i in range(8)],
        started_unix=started_wall, started_monotonic=started_mono,
        compute_deadline_unix=started_wall+2640, compute_deadline_monotonic=started_mono+2640,
        absolute_deadline_unix=started_wall+2700, hard_deadline_monotonic=started_mono+2700)
    (output/'execution').mkdir()
    write(output/'plan.json', plan)
    command = [sys.executable, __file__, '--worker-plan', str(output/'plan.json'),
               '--sha256', binding(output/'plan.json')['sha256']]
    # The independently guarded worker acquires its own lifetime lock before any
    # game. If a user launch wins this transfer race, the diagnostic refuses.
    os.close(lock)
    with (output/'worker.log').open('w') as log:
        process = guarded_popen(command, deadline_wall=plan['absolute_deadline_unix'],
            deadline_monotonic=plan['hard_deadline_monotonic'], evidence_path=output/'deadline.jsonl',
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rc = None
        try: rc = process.wait(timeout=2700)
        finally:
            from run_local_training import cleanup_owned
            cleanup = cleanup_owned(plan, process=process, evidence_root=output)
            write(output/'shutdown.json', dict(finished_utc=utc(), returncode=rc, **cleanup))
            if not cleanup['owned_cleanup_verified']: rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
