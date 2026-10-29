"""Freeze and validate actual Linux binaries; diagnostic games never enter PPO."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    import hashlib
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha)


def write(path, value):
    from tools.run_local_training import atomic
    atomic(path, value)


def read(path):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(Path(path).read_bytes())


def freeze(build, destination, game_map):
    from tools.check_cloud_runtime import inspect
    from experiments.episodes.scoring import engine_recipe_digests
    require(sys.platform == 'linux', 'Linux runtime required')
    build, destination, game_map = map(lambda p: Path(p).resolve(), (build, destination, game_map))
    require(not destination.exists() and not destination.is_relative_to(ROOT), 'Fresh private runtime destination required')
    subprocess.run([sys.executable, 'tools/prepare_vcmi.py', '--check'], cwd=ROOT, check=True)
    destination.mkdir(parents=True)
    target = destination/'bin'; target.mkdir()
    names = ['vcmiclient', 'vcmiserver', 'vcmireplay', 'libvcmi.so',
             *('AI/lib'+name+'.so' for name in ('ExplorerRL', 'BattleAI', 'Nullkiller', 'EmptyAI', 'StupidAI'))]
    artifacts = {}
    for name in names:
        source = build/name
        require(source.is_file() and source.resolve().is_relative_to(build), 'Missing or redirected binary: '+name)
        output = target/name; output.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, output); output.chmod(0o700)
        require(binding(source)['sha256'] == binding(output)['sha256'], 'Frozen binary differs')
        dynamic = subprocess.check_output(['readelf', '-d', str(output)], text=True)
        require(str(build) not in dynamic, 'Binary retains a donor build loader path: '+name)
        artifacts[name] = binding(output)
    for name in ('config', 'Mods'):
        shutil.copytree(build/name, target/name, symlinks=False)
    map_target = destination/'hero-management-v2.vmap'
    shutil.copyfile(game_map, map_target)
    resources = {str(p.relative_to(target)): binding(p) for folder in ('config', 'Mods')
                 for p in sorted((target/folder).rglob('*')) if p.is_file()}
    external = {}
    for name in names:
        loaded = subprocess.check_output(['ldd', str(target/name)], text=True)
        require('not found' not in loaded, 'Unresolved frozen runtime library')
        for value in re.findall(r'=> (/\S+) \(', loaded):
            path = Path(value).resolve()
            if not path.is_relative_to(target): external[str(path)] = binding(path)
    # LD_DEBUG records the library actually initialized, not just an open port
    # or a successful source agreement check.
    with tempfile.TemporaryDirectory() as temporary:
        env = {**os.environ, 'LD_DEBUG': 'libs', 'XDG_DATA_HOME': temporary+'/data',
               'XDG_CONFIG_HOME': temporary+'/config', 'XDG_CACHE_HOME': temporary+'/cache'}
        result = subprocess.run([str(target/'vcmiclient'), '--help'], cwd=target, env=env,
                                capture_output=True, text=True, timeout=30)
    loaded = re.findall(r'calling init:\s*(\S*libvcmi\.so\S*)', result.stderr)
    require(result.returncode == 0 and loaded == [str(target/'libvcmi.so')], 'Actual frozen libvcmi load was not verified')
    runtime = inspect(target)
    require(runtime['passed'], 'Frozen Linux CLI/library checks failed')
    receipt = dict(schema='heroes-linux-runtime-freeze-v1', artifacts=artifacts, resources=resources,
        external_dependencies=external, runtime=runtime['artifacts'], runtime_bin=str(target),
        map=binding(map_target), engine_recipe=engine_recipe_digests(ROOT),
        actual_client_libvcmi_load=loaded, runtime_check=runtime, games_launched=0, optimizer_updates=0)
    receipt['loader_environment'] = {k: os.environ.get(k) for k in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    write(destination/'freeze.json', receipt)
    print(json.dumps({'freeze': binding(destination/'freeze.json'), 'actual_loader_verified': True}))


def check_freeze(path):
    from experiments.episodes.scoring import engine_recipe_digests
    from experiments.headless.contextual_evidence import runtime_identity
    receipt = read(path)
    require(receipt['schema'] == 'heroes-linux-runtime-freeze-v1'
            and receipt['engine_recipe'] == engine_recipe_digests(ROOT)
            and receipt['runtime_check']['passed'] is True, 'Unaccepted Linux freeze')
    require(receipt['loader_environment'] == {k: os.environ.get(k) for k in ('LD_LIBRARY_PATH', 'LD_PRELOAD')},
            'Frozen loader environment changed')
    binary_root = Path(receipt['runtime_bin'])
    expected_runtime = {name: item['sha256'] for name, item in receipt['artifacts'].items()
                        if name not in ('vcmiclient', 'vcmiserver')}
    require(runtime_identity(binary_root) == expected_runtime, 'Frozen module inventory changed')
    resources = set()
    for name in ('config', 'Mods'):
        folder = binary_root/name
        require(folder.is_dir() and not folder.is_symlink(), 'Frozen resource folder redirected')
        for item in folder.rglob('*'):
            require(not item.is_symlink(), 'Frozen resource symlink introduced')
            if item.is_file(): resources.add(str(item.relative_to(binary_root)))
    require(resources == set(receipt['resources']), 'Frozen resource inventory changed')
    for item in [*receipt['artifacts'].values(), *receipt['resources'].values(),
                 *receipt['external_dependencies'].values(), receipt['map']]:
        require(binding(item['path']) == item, 'Frozen runtime, resource or dependency changed')
    return receipt


def policy_worker(path, expected_sha256):
    require(binding(path)['sha256'] == expected_sha256, 'Frozen-policy plan changed')
    plan = read(path)
    os.environ['HEROES_ARTIFACT_ROOT'] = plan['artifact_root']
    import torch
    from experiments.learner import builtin_first_run as pilot, local_script_training as local, interaction_guard
    from experiments.learner.builtin_rollout import load_builtin_episode
    from experiments.learner.contextual_rollout import behavior_guard_version
    from tools.validate_interaction_guard import state_snapshot, verify_state_snapshot
    torch.set_num_threads(1)
    require(plan['mode'] == 'evaluation' and len(plan['schedule']) == 8
            and plan['configuration'] == local.configuration(plan['parent_descriptor']), 'Frozen diagnostic scope changed')
    require(binding(__file__) == plan['harness'], 'Linux diagnostic harness changed')
    check_freeze(plan['freeze']['path'])
    state = local.restore_parent(plan['parent_descriptor'])
    before = state_snapshot(state)
    interaction_guard.ACTIVE_VERSION = interaction_guard.VERSION
    report = dict(schema='heroes-linux-frozen-policy-v1', status='running', training_samples=0,
                  optimizer_steps=0, optimizer_updates=0, attempts=[], parent_progress=vars(state.progress),
                  plan_sha256=binding(path)['sha256'])
    output = Path(plan['execution_output']); output.mkdir()
    try:
        for index, row in enumerate(plan['schedule']):
            require(binding(path)['sha256'] == expected_sha256, 'Frozen-policy plan changed')
            require(binding(__file__) == plan['harness'], 'Linux diagnostic harness changed')
            require(binding(plan['freeze']['path']) == plan['freeze'], 'Linux freeze receipt changed')
            check_freeze(plan['freeze']['path'])
            pilot.check_identity(plan)
            def save(attempt):
                if len(report['attempts']) == index: report['attempts'].append(attempt)
                else: report['attempts'][index] = attempt
                write(output/'report.json', report)
            attempt, receipt = pilot.run_attempt(plan, path, binding(path)['sha256'], index,
                state.model, state.checkpoint,
                pilot.Budget(min(plan['compute_deadline_monotonic'], time.monotonic()+600),
                             min(plan['compute_deadline_unix'], time.time()+600)), save)
            require(receipt is None and attempt['status'] == 'passed' and attempt['training_samples'] == 0,
                    'Frozen Linux diagnostic failed; no replacement or PPO admission')
            episode = load_builtin_episode(Path(attempt['episode_directory'])/'builtin-rollout.json')
            require(behavior_guard_version(episode.decisions) == interaction_guard.VERSION,
                    'Recorded policy support differs')
            attempt['guard_invariants_verified'] = True; save(attempt)
        verify_state_snapshot(before, state)
        restored = local.restore_parent(plan['parent_descriptor'])
        verify_state_snapshot(before, restored)
        report.update(status='passed', full_parent_unchanged=True, full_parent_restore=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        write(output/'report.json', report)
    return 0 if report['status'] == 'passed' else 1


def frozen_policy(args):
    os.environ['HEROES_ARTIFACT_ROOT'] = str(args.artifact_root.resolve())
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    from experiments.learner.inference import source_identity
    from experiments.learner.builtin_v8_migration import map_identity
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.process_deadline import guarded_popen
    from tools.run_local_training import cleanup_owned, exclusive_lock
    from tools.validate_interaction_guard import reserve_diagnostic_seeds
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit the complete diagnostic source first')
    freeze_receipt = check_freeze(args.freeze)
    output = args.output.resolve(); require(not output.exists(), 'Diagnostic output already consumed')
    require(output.is_relative_to(args.artifact_root.resolve()) and not output.is_relative_to(ROOT),
            'Diagnostics must stay in the private artifact root')
    require(all(1_900_000_000 <= seed <= 2_147_480_000-7
                for seed in (args.engine_seed_base, args.policy_seed_base)), 'Use the isolated Linux diagnostic seed range')
    parent = read(args.lineage)['parent']
    output.mkdir(parents=True)
    copied = output/'parent'; copied.mkdir()
    descriptor = {}
    for name, original in parent.items():
        require(binding(original['path']) == original, 'Original parent changed')
        target = copied/(name+Path(original['path']).suffix)
        shutil.copyfile(original['path'], target); descriptor[name] = binding(target)
        require(descriptor[name]['sha256'] == original['sha256'], 'Isolated parent copy differs')
    rows = [dict(episode=i+1, phase='evaluation', learner_player=i//4,
                 engine_seed=args.engine_seed_base+i, policy_seed=args.policy_seed_base+i) for i in range(8)]
    reserve_diagnostic_seeds(args.artifact_root/'linux-diagnostic-seeds.jsonl', rows, output=output,
                             parent_run=Path(parent['plan']['path']).parent, candidate=binding(args.freeze))
    binary = Path(freeze_receipt['runtime_bin'])/'vcmiclient'
    game_map = Path(freeze_receipt['map']['path'])
    wall, mono = time.time(), time.monotonic()
    plan = dict(schema='heroes-linux-frozen-policy-plan-v1', scope='heroes-linux1229-frozen-policy-v1',
        mode='evaluation', policy_support='visible-interaction-budget-v1', source=source_identity(local.INTERFACE),
        implementation=pilot.implementation(), prerequisites=pilot.prerequisites(), configuration=local.configuration(descriptor),
        binary=str(binary), runtime=runtime_identity(binary.parent), data=str(args.data.resolve()),
        game_map=str(game_map), map=map_identity(game_map), native=VcmiLauncher(binary, args.data, game_map).identity(),
        checkpoint=descriptor['policy'], replay_binary=binding(binary.parent/'vcmireplay'), parent_descriptor=descriptor,
        freeze=binding(args.freeze), harness=binding(__file__), artifact_root=str(args.artifact_root.resolve()),
        schedule=rows, execution_output=str(output/'execution'), started_unix=wall, started_monotonic=mono,
        compute_deadline_unix=wall+2640, compute_deadline_monotonic=mono+2640,
        absolute_deadline_unix=wall+2700, hard_deadline_monotonic=mono+2700)
    path = output/'plan.json'; write(path, plan)
    with exclusive_lock(args.artifact_root/'diagnostic-owner.lock'), (output/'worker.log').open('x') as log:
        process = guarded_popen([sys.executable, __file__, '_policy-worker', '--plan', str(path),
                                '--sha256', binding(path)['sha256']],
            deadline_wall=plan['absolute_deadline_unix']-5, deadline_monotonic=plan['hard_deadline_monotonic']-5,
            evidence_path=output/'deadline.jsonl', cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rc = None
        try:
            rc = process.wait(timeout=2700)
        finally:
            cleanup = cleanup_owned(plan, process=process, evidence_root=output)
            write(output/'shutdown.json', dict(returncode=rc, **cleanup))
    require(rc == 0 and cleanup['owned_cleanup_verified'], 'Frozen policy validation or cleanup failed')
    print(json.dumps({'passed': True, 'games': 8, 'training_samples': 0, 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze')
    for name in ('build', 'destination', 'map'): p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('frozen-policy')
    for name in ('freeze', 'lineage', 'artifact-root', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('engine-seed-base', 'policy-seed-base'): p.add_argument('--'+name, type=int, required=True)
    p = sub.add_parser('_policy-worker'); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.build, args.destination, args.map)
    elif args.action == 'frozen-policy': frozen_policy(args)
    else: return policy_worker(args.plan, args.sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
