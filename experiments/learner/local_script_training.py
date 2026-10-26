"""User-launched local batches using the existing collector, verifier and PPO.

Historical checkpoint bytes and metadata are immutable. Explicit local bindings
relocate their files; new plans bind this executable source and native runtime.
The controller owns scheduling, the campaign clock and durable publication.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import torch

from . import builtin_first_run as pilot, builtin_training as batches
from . import builtin_training_resume as resume, checkpoints as cp
from .builtin_preservation import binding, durable_path, verify_pair
from .inference import source_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import VcmiLauncher

ROOT = Path(__file__).resolve().parents[2]
INTERFACE = 'adventure-capabilities-v1'
SCOPE = 'heroes-local-script-training-v1'
SESSION_SCHEMA = 'heroes-local-script-session-v1'
require = resume.require
DESCRIPTOR_KEYS = frozenset(('plan', 'report', 'policy', 'resume', 'initial_checkpoint'))
HISTORICAL_ADAPTERS = frozenset(('experiments/learner/curriculum_training.py',
                               'experiments/learner/overnight_training.py'))
ROUTING_SCOPES = frozenset(('heroes-update146-curriculum-v3',
                          'heroes-overnight-curriculum-20261026', SCOPE))
ROUTING_MODULES = {'heroes-update146-curriculum-v3': 'curriculum_training',
                   'heroes-overnight-curriculum-20261026': 'overnight_training',
                   SCOPE: 'local_script_training'}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def binding_descriptor(plan, report, policy, resume_file, initial_checkpoint):
    return {name: binding(path) for name, path in zip(
        ('plan', 'report', 'policy', 'resume', 'initial_checkpoint'),
        (plan, report, policy, resume_file, initial_checkpoint))}


def _bound(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'},
            'Expected exact local file binding')
    require(binding(value['path']) == value, 'Bound local artifact changed')
    return Path(value['path'])


def _same_bytes(local, historical):
    require(all(local[k] == historical[k] for k in ('sha256', 'bytes')),
            'Relocation cannot change historical bytes')


def read_descriptor(descriptor):
    require(type(descriptor) is dict and set(descriptor) == DESCRIPTOR_KEYS,
            'Checkpoint descriptor requires full state and initial dependency')
    for value in descriptor.values():
        _bound(value)
    plan, report = (json.loads(Path(descriptor[k]['path']).read_text()) for k in ('plan', 'report'))
    require(report['plan_sha256'] == descriptor['plan']['sha256'], 'Parent plan/report mismatch')
    resume._completion(plan, report)
    for kind in ('policy', 'resume'):
        _same_bytes(descriptor[kind], report['after_checkpoint'][kind])
    _same_bytes(descriptor['initial_checkpoint'], resume._metadata(plan, report['plan_sha256'])['initial_checkpoint'])
    require(report['after_checkpoint'].get('exact_model_adam_rng_progress_pool') is True,
            'Parent lacks verified full state')
    require(plan['source']['interface'] == INTERFACE, 'Unsupported learned interface')
    return plan, report


def _without_scope_routes(contents):
    """Permit only the small adapter dispatches, not changes to the collector."""
    tree = ast.parse(contents)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'check_identity':
            while node.body and isinstance(node.body[0], ast.If):
                branch = node.body[0]
                test = branch.test
                if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value in ROUTING_SCOPES
                        and ast.dump(test.left) == ast.dump(ast.parse('plan.get("scope")', mode='eval').body)
                        and not branch.orelse and len(branch.body) == 2
                        and isinstance(branch.body[0], ast.ImportFrom)
                        and isinstance(branch.body[1], ast.Return)):
                    break
                imported, returned = branch.body[0], branch.body[1].value
                if not (imported.level == 1 and imported.module == ROUTING_MODULES[test.comparators[0].value]
                        and len(imported.names) == 1 and imported.names[0].name == 'check_identity'
                        and isinstance(returned, ast.Call) and isinstance(returned.func, ast.Name)
                        and returned.func.id == (imported.names[0].asname or 'check_identity')
                        and len(returned.args) == 1 and isinstance(returned.args[0], ast.Name)
                        and returned.args[0].id == 'plan' and not returned.keywords):
                    break
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def source_compatibility(parent):
    """Every learner/PPO byte remains exact except recognized routing glue."""
    require(parent['interface'] == INTERFACE, 'Interface transition forbidden')
    require(parent['torch'].split('+')[0] == str(torch.__version__).split('+')[0],
            'Unreviewed PyTorch version transition')
    changes = {}
    for name, expected in parent['source_sha256'].items():
        historical = subprocess.check_output(['git', 'show', parent['git_commit'] + ':' + name],
                                             cwd=ROOT, timeout=10)
        require(hashlib.sha256(historical).hexdigest() == expected, 'Historical source is not committed')
        current_path = ROOT / name
        if name in HISTORICAL_ADAPTERS and not current_path.exists():
            changes[name] = 'Historical orchestration is not executed by this local adapter'
            continue
        require(current_path.is_file(), 'Missing learner source: ' + name)
        current = current_path.read_bytes()
        if hashlib.sha256(current).hexdigest() == expected:
            continue
        require(name == 'experiments/learner/builtin_first_run.py'
                and _without_scope_routes(historical) == _without_scope_routes(current),
                'Policy, PPO or collector implementation changed: ' + name)
        changes[name] = 'Recognized check_identity scope routing only; remaining AST exact'
    return {'historical_commit': parent['git_commit'], 'changes': changes,
            'policy_ppo_unchanged': True, 'historical_metadata_rewritten': False}


def configuration(descriptor):
    plan, _ = read_descriptor(descriptor)
    cfg = deepcopy(plan['configuration'])
    require(cfg['interface'] == INTERFACE and cfg['max_days'] == 365
            and cfg['ordinary_choice_limit'] == 4960 and cfg['reserved_dialog_choices'] == 160
            and cfg['max_total_choices'] == 5120 and cfg['max_updates'] == 1
            and cfg['allow_verified_training_prefix'] is False,
            'Only the accepted complete-batch configuration is supported')
    return cfg


def restore_parent(descriptor):
    """Restore full historical state without rewriting paths inside metadata."""
    original_torch, original_python = torch.get_rng_state().clone(), random.getstate()
    try:
        plan, report = read_descriptor(descriptor)
        proof = source_compatibility(plan['source'])
        options = {'historical_torch_version': plan['source']['torch']}
        verify_pair(descriptor['resume']['path'], descriptor['policy']['path'], plan, report,
                    migration=SimpleNamespace(checkpoint_options=options))
        metadata = resume._metadata(plan, report['plan_sha256'])
        progress, steps, sampler = resume._completion(plan, report)
        policy = cp.load_policy_checkpoint(descriptor['policy']['path'], expected_source_metadata=metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE, **options)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
        generator = torch.Generator(device='cpu')
        state = cp.load_checkpoint(descriptor['resume']['path'], model, optimizer,
            expected_ppo_config=plan['configuration']['ppo'], expected_source_metadata=metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE,
            generators={'minibatch': generator}, at_rollout_boundary=True, **options)
        raw = torch.load(descriptor['resume']['path'], map_location='cpu', weights_only=True)
        require(state.progress == progress and state.sampler_state == sampler,
                'Restored progress or sampler differs')
        require(state.checkpoint == policy.checkpoint
                and cp.model_weight_digest(model) == policy.weights_sha256 == report['after_checkpoint']['weights_sha256'],
                'Restored policy/resume pair differs')
        for left, right in ((dict(model.state_dict()), raw['model_state']),
                (optimizer.state_dict(), raw['resume']['optimizer_state']),
                (state.opponent_pool.state_dict(), policy.opponent_pool.state_dict()),
                (torch.get_rng_state(), raw['rng']['torch_cpu']), (random.getstate(), raw['rng']['python']),
                (generator.get_state(), raw['rng']['torch_generators']['minibatch'])):
            resume._exact(left, right)
        require(optimizer.state and any(v['step'].item() > 0 for v in optimizer.state.values()),
                'Restored Adam state is absent')
        for value in descriptor.values():
            _bound(value)
        model.eval().requires_grad_(False)
        links = {k: descriptor[k] for k in ('plan', 'report', 'policy', 'resume')}
        links.update(progress=vars(progress), optimizer_steps=steps, source_proof=proof,
                     runtime_proof={'explicit_local_migration': True, 'historical_metadata_unchanged': True})
        return resume.ParentState(model, optimizer, generator, state.opponent_pool, progress, sampler,
                                  metadata, policy.checkpoint, policy.weights_sha256, links, steps)
    except BaseException:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)
        raise


def _map_payloads(seat):
    from experiments.explorer.create_capability_scenarios import ordinary_payloads
    require(seat in (0, 1), 'Invalid seat')
    result = ordinary_payloads()
    color, other = ('red', 'blue')[seat], ('red', 'blue')[1-seat]
    hero = next(x for x in result['objects.json'] if x['instanceName'] == f'hero_{seat}')
    town = next(x for x in result['objects.json'] if x['instanceName'] == f'town_{1-seat}')
    town.update(x=6, y=5)
    hero.update(x=5, y=6)
    hero['options']['army'] = [{'type': 'core:pikeman', 'amount': 100}]
    result['objects.json'] = [hero, town]
    result['surface_terrain.json'] = [row[:8] for row in result['surface_terrain.json'][:8]]
    header = result['header.json']
    header['mapLevels']['surface'].update(width=8, height=8)
    header['players'][color].pop('mainTown', None)
    header['players'][other].pop('mainHero', None)
    header['players'][other]['heroes'] = {}
    header['players'][other]['mainTown'].update(x=4, y=5)
    header['name'] = f'Curriculum v1 stage3 learner {color}'
    header['description'] = ('Declared initial-state distribution: townless learner near opponent town; '
        'opponent starts without a hero. Native recruitment, fog, actions, battles and elimination rules unchanged.')
    return result


def map_identity(path, seat):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 3, 'Unexpected map archive members')
        actual = {name: json.loads(archive.read(name)) for name in names}
    require(actual == _map_payloads(seat), 'Map differs from the accepted open8x8 distribution')
    return {**binding(path), 'scenario': 'curriculum-v1-open8x8', 'stage': 3, 'seat': seat}


def check_clock(session, *, active=True):
    names = ('started_unix', 'started_monotonic', 'compute_deadline_unix',
             'compute_deadline_monotonic', 'absolute_deadline_unix', 'hard_deadline_monotonic')
    require(all(type(session[k]) in (int, float) and math.isfinite(session[k]) for k in names),
            'Invalid campaign clock')
    wall = session['absolute_deadline_unix'] - session['started_unix']
    mono = session['hard_deadline_monotonic'] - session['started_monotonic']
    require(wall > 300 and abs(wall-mono) < .001
            and abs((session['compute_deadline_unix']-session['started_unix'])
                    -(session['compute_deadline_monotonic']-session['started_monotonic'])) < .001
            and session['started_unix'] < session['compute_deadline_unix'] <= session['absolute_deadline_unix']-300,
            'Campaign clock or finalization reserve changed')
    if active:
        require(session['started_unix'] <= time.time() < session['compute_deadline_unix']
                and session['started_monotonic'] <= time.monotonic() < session['compute_deadline_monotonic'],
                'Campaign compute window expired or not started')


def read_session(path, *, active=True):
    path = durable_path(path)
    session = json.loads(path.read_text())
    require(session['schema'] == SESSION_SCHEMA and session['scope'] == SCOPE, 'Unknown local session')
    require(path == Path(session['run_dir'])/'session.json', 'Session path changed')
    check_clock(session, active=active)
    require(source_identity(INTERFACE) == session['source'] and not session['source']['tracked_worktree_dirty'],
            'Frozen learner source drift')
    require(pilot.implementation() == session['implementation'] and pilot.prerequisites() == session['prerequisites'],
            'Frozen implementation or acceptance evidence drift')
    require(configuration(session['initial_parent']) == session['configuration'], 'PPO configuration changed')
    _bound(session['asset_config'])
    for name, value in session.get('frozen_files', {}).items():
        require((ROOT/name).resolve().is_relative_to(ROOT) and binding(ROOT/name) == value,
                'Frozen controller file changed')
    require(runtime_identity(Path(session['runtime_bin'])) == session['runtime'], 'Runtime changed')
    require(len(session['maps']) == 2, 'Both seat maps required')
    for seat, value in enumerate(session['maps']):
        _bound(value)
        map_identity(value['path'], seat)
    return session


def schedule(session, sequence, seat, evaluate=False):
    require(type(sequence) is int and sequence >= 0 and type(seat) is int and seat in (0, 1),
            'Invalid sequence or seat')
    if not evaluate:
        require(sequence >= session['next_sequence'], 'Training sequence was already consumed')
    engine = 710001 if evaluate else session['training_engine_seed_base'] + 4*sequence
    policy = 720001 if evaluate else session['training_policy_seed_base'] + 4*sequence
    require(type(engine) is int and type(policy) is int and 0 < engine <= 2**31-5 and 0 < policy <= 2**31-5,
            'Seed range exceeds signed native limits')
    require(evaluate or not (set(range(engine, engine+4)) & set(range(710001, 710005))
                            or set(range(policy, policy+4)) & set(range(720001, 720005))),
            'Training seeds overlap held-out seeds')
    return [dict(episode=i+1, phase='evaluation' if evaluate else 'training', learner_player=seat,
                 engine_seed=engine+i, policy_seed=policy+i) for i in range(4)]


def check_identity(plan):
    _bound(plan['local_session'])
    session = read_session(plan['local_session']['path'])
    evaluate = plan['mode'] == 'evaluation'
    require(plan['mode'] in ('training', 'evaluation') and plan['scope'] == SCOPE
            and plan['schema'] == (batches.EVALUATION if evaluate else batches.SCHEMA)+'-plan', 'Unknown batch scope')
    for key in ('source', 'runtime', 'implementation', 'prerequisites', 'configuration', 'absolute_deadline_unix', 'compute_deadline_unix'):
        require(plan[key] == session[key], 'Batch/session identity differs: ' + key)
    require(plan['schedule'] == schedule(session, plan['batch_sequence'], plan['seat'], evaluate), 'Schedule changed')
    require(plan['game_map'] == session['maps'][plan['seat']]['path']
            and plan['map'] == map_identity(plan['game_map'], plan['seat']), 'Seat map changed')
    require(plan['binary'] == str(Path(session['runtime_bin'])/'vcmiclient') and plan['data'] == session['data'],
            'Runtime location changed')
    launcher = VcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']))
    require(launcher.identity() == plan['native'], 'Native binary or map identity changed')
    require(plan['native_opponents'] == [pilot.native_opponent(session['runtime'], seat) for seat in (0, 1)],
            'Native opponent changed')
    require(plan['checkpoint'] == plan['parent_descriptor']['policy'], 'Wrong policy checkpoint')
    for value in (plan['checkpoint'], plan['replay_binary'], *plan['parent_descriptor'].values()):
        _bound(value)
    parent_plan, parent_report = read_descriptor(plan['parent_descriptor'])
    progress, steps, _ = resume._completion(parent_plan, parent_report)
    require(all(plan['parent'][key] == plan['parent_descriptor'][key] for key in ('plan', 'report', 'policy', 'resume'))
            and plan['parent']['progress'] == vars(progress) and plan['parent']['optimizer_steps'] == steps,
            'Parent linkage or cumulative counters changed')
    resume._metadata(plan, '0'*64)  # Validate metadata linkage without creating a circular plan hash.
    require(Path(plan['execution_output']).parent.parent == Path(session['run_dir'])/'batches', 'Foreign batch output')


def prepare_batch(session_path, parent_descriptor, sequence, seat, evaluate=False, label=None):
    session_path = Path(session_path).resolve()
    session = read_session(session_path)
    state = restore_parent(parent_descriptor)
    require(configuration(parent_descriptor) == session['configuration'], 'Parent PPO configuration differs')
    rows = schedule(session, sequence, seat, evaluate)
    label = label or ('eval' if evaluate else 'train')
    require(label in ('train', 'eval', 'baseline', 'final', 'periodic'), 'Unknown batch label')
    require((label == 'train') == (not evaluate), 'Label/mode differs')
    output = Path(session['run_dir'])/'batches'/f'{label}-{sequence:06d}-seat{seat}'
    require(not output.exists(), 'Never overwrite or retry a consumed batch')
    if not evaluate:
        keys = {(r['engine_seed'], r['policy_seed'], r['learner_player']) for r in rows}
        for previous in (Path(session['run_dir'])/'batches').glob('*/plan.json'):
            old = json.loads(previous.read_text())
            require(not keys & {(r['engine_seed'], r['policy_seed'], r['learner_player'])
                                for r in old['schedule'] if r['phase'] == 'training'}, 'Training seeds reused')
    game_map = Path(session['maps'][seat]['path'])
    binary = Path(session['runtime_bin'])/'vcmiclient'
    launcher = VcmiLauncher(binary, Path(session['data']), game_map)
    plan = dict(schema=(batches.EVALUATION if evaluate else batches.SCHEMA)+'-plan', scope=SCOPE,
        mode='evaluation' if evaluate else 'training', seat=seat, batch_sequence=sequence,
        source=session['source'], implementation=session['implementation'], prerequisites=session['prerequisites'],
        configuration=session['configuration'], runtime=session['runtime'], binary=str(binary), data=session['data'],
        game_map=str(game_map), map=map_identity(game_map, seat), native=launcher.identity(),
        replay_binary=binding(binary.parent/'vcmireplay'), native_opponents=[pilot.native_opponent(session['runtime'], s) for s in (0, 1)],
        schedule=rows, execution_output=str(output/'execution'), parent=state.bindings,
        parent_descriptor=parent_descriptor, checkpoint=parent_descriptor['policy'], local_session=binding(session_path),
        absolute_deadline_unix=session['absolute_deadline_unix'], compute_deadline_unix=session['compute_deadline_unix'],
        authorization={'scope': SCOPE, 'instruction': 'Explicit manual local script launch; no LLM supervision or automatic restart'},
        wall_scope='The immutable manual session includes evaluation, collection, checks, PPO, preservation and cleanup.',
        comparison_scope='Same easier8x8 state distribution; fixed paired held-out seeds are not original-map generalization.')
    plan['checkpoint_metadata'] = {**session['source'], 'parent_plan_sha256': state.bindings['plan']['sha256'],
        'parent_report_sha256': state.bindings['report']['sha256'], 'initial_checkpoint': state.bindings['policy'],
        'native_opponent': {k: v for k, v in plan['native_opponents'][0].items() if k != 'player'},
        'archived_neural_pool_executed': False}
    check_identity(plan)
    write_new(output/'plan.json', plan)
    return output/'plan.json'


def read_plan(path, sha):
    path = durable_path(path)
    require(binding(path)['sha256'] == sha, 'Immutable batch plan changed')
    plan = json.loads(path.read_text())
    require(plan['execution_output'] == str(path.parent/'execution'), 'Batch execution path changed')
    check_identity(plan)
    return plan


def valid_evaluation(plan, report):
    return (report.get('schema') == batches.EVALUATION+'-result' and plan.get('mode') == 'evaluation'
        and report.get('status') == 'finished' and report.get('training_performed') is False
        and all(report.get(k) is True for k in ('worker_complete', 'supervisor_cleanup_verified', 'within_shared_budget'))
        and all(report.get(k) == 0 for k in ('optimizer_updates', 'optimizer_steps', 'training_samples'))
        and len(report.get('attempts', [])) == len(plan.get('schedule', [])) == 4
        and all(all(row.get(k) == value for k, value in expected.items())
                and row.get('status') == 'passed' and row.get('training_samples') == 0
                and all(row.get(k) is True for k in ('completion_receipt', 'cleanup_verified', 'identity_verified', 'replay_verified'))
                for expected, row in zip(plan['schedule'], report['attempts'])))


def require_baselines(plan):
    session = read_session(plan['local_session']['path'])
    journal = json.loads((Path(session['run_dir'])/'campaign-report.json').read_text())
    accepted = set()
    for row in journal['batches']:
        if row.get('mode') != 'baseline' or row.get('status') != 'passed':
            continue
        bp = json.loads(_bound(row['plan']).read_text())
        br = json.loads(_bound(row['report']).read_text())
        require(br['plan_sha256'] == row['plan']['sha256'], 'Baseline report/plan mismatch')
        require(bp['local_session'] == plan['local_session'] and bp['checkpoint'] == session['initial_parent']['policy']
                and bp['configuration'] == session['configuration'] and bp['runtime'] == session['runtime']
                and bp['source'] == session['source'] and bp['schedule'] == schedule(session, bp['batch_sequence'], bp['seat'], True)
                and valid_evaluation(bp, br), 'Unverified or foreign baseline')
        accepted.add(bp['seat'])
    require(accepted == {0, 1}, 'Both current-runtime frozen baselines must pass before PPO')


def descriptor_from_batch(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    report_path = Path(plan['execution_output'])/'report.json'
    report = json.loads(report_path.read_text())
    resume._completion(plan, report)
    return binding_descriptor(plan_path, report_path, report['after_checkpoint']['policy']['path'],
                              report['after_checkpoint']['resume']['path'], plan['checkpoint']['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('execute', 'worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--started', type=float)
    parser.add_argument('--wall-started', type=float)
    args = parser.parse_args()
    if args.command == 'worker':
        require(args.started is not None and args.wall_started is not None, 'Worker clock required')
        result = batches.worker(args, plan_reader=read_plan,
            state_reader=lambda plan: restore_parent(plan['parent_descriptor']),
            heldout_checker=lambda *_: require_baselines(read_plan(args.plan, args.sha256)))
    else:
        result = pilot.execute(args, plan_reader=read_plan, worker_module=__spec__.name)
    print(json.dumps({k: result.get(k) for k in ('status', 'stop_reason', 'optimizer_updates', 'optimizer_steps', 'training_samples')}))
    return 0 if result.get('status') == 'finished' else 1


if __name__ == '__main__':
    sys.exit(main())
