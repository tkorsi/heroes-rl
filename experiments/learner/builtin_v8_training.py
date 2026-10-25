"""Released v8 four-game batches using the existing strict admission/PPO worker."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

from . import builtin_training as batches, builtin_first_run as pilot, builtin_training_resume as resume
from . import builtin_v8_migration as migration, builtin_v8_session as scope
from .builtin_preservation import DURABLE_ROOT, durable_path, preserve, verify_pair
from .inference import source_identity


def parent_state(plan):
    parent = plan['parent']
    for name in ('plan', 'report', 'policy', 'resume'):
        resume.require(resume.binding(parent[name]['path']) == parent[name], 'Prepared v8 parent changed')
    raw = json.loads(Path(parent['plan']['path']).read_text())
    if raw['schema'] == migration.SCHEMA:
        state = migration.load_parent(parent['plan']['path'])
        resume.require(raw['source'] == plan['source'] and raw['runtime'] == plan['runtime']
                       and raw['native'] == plan['native'], 'Bootstrap target source/runtime mismatch')
        return state
    return resume.load_parent(parent['plan']['path'], parent['report']['path'], plan['source'],
        current_runtime=plan['runtime'], current_native=plan['native'],
        expected_plan_sha256=parent['plan']['sha256'], expected_report_sha256=parent['report']['sha256'])


def require_heldout(state, runtime, native):
    if state.progress.updates != 92 and state.progress.updates % 5:
        return
    # Only this fresh v8 session can satisfy the new baseline or periodic gate.
    for path in (DURABLE_ROOT/'runs').glob('v8-update92-heldout-*/plan.json'):
        plan = json.loads(path.read_text())
        if (plan.get('schema') != batches.EVALUATION+'-plan'
                or plan.get('parent', {}).get('plan') != state.bindings['plan']
                or plan.get('runtime') != runtime or plan.get('native') != native
                or plan.get('local_session') != resume.binding(scope.SESSION)
                or plan.get('schedule') != batches.schedule(state.progress.updates, True)):
            continue
        report_path = Path(plan['execution_output'])/'report.json'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        if (report.get('plan_sha256') == pilot.digest(path) and report.get('status') == 'finished'
                and all(report.get(k) is True for k in ('worker_complete','supervisor_cleanup_verified','within_shared_budget'))
                and report.get('training_performed') is False and report.get('optimizer_updates') == 0
                and report.get('optimizer_steps') == 0 and len(report.get('attempts', [])) == 2
                and all(all(row.get(k) == v for k, v in expected.items()) and row.get('status') == 'passed'
                    and row.get('training_samples') == 0
                    and all(row.get(k) is True for k in ('completion_receipt','cleanup_verified','identity_verified','replay_verified'))
                    for row, expected in zip(report['attempts'], plan['schedule']))):
            return
    raise ValueError('Fresh v8 both-seat baseline/periodic gate required before further training')


def unused_schedule(sequence):
    resume.require(type(sequence) is int and sequence >= 94, 'Sequences through93 are consumed')
    rows = batches.schedule(sequence)
    keys = {(r['engine_seed'], r['learner_player'], r['policy_seed']) for r in rows}
    for path in (DURABLE_ROOT/'runs').glob('*/plan.json'):
        prior = json.loads(path.read_text())
        used = {(r.get('engine_seed'), r.get('learner_player'), r.get('policy_seed'))
                for r in prior.get('schedule', []) if r.get('phase') == 'training'}
        resume.require(not keys & used, 'Training seeds already scheduled, including failed attempts')
    return rows


def prepare(args):
    session = scope.session_for(args.session)
    resume.require(time.time()+180 < session['compute_deadline_unix'], 'Insufficient batch/preservation reserve')
    parent_path, output = durable_path(args.parent), durable_path(args.output)
    resume.require(not output.exists(), 'Fresh v8 batch directory required')
    raw = json.loads(parent_path.read_text())
    source = source_identity(migration.INTERFACE)
    if raw['schema'] == migration.SCHEMA:
        state = migration.load_parent(parent_path)
    else:
        report_path = Path(raw['execution_output'])/'report.json'
        state = resume.load_parent(parent_path, report_path, source, current_runtime=raw['runtime'], current_native=raw['native'],
            expected_plan_sha256=pilot.digest(parent_path), expected_report_sha256=pilot.digest(report_path))
        backup = DURABLE_ROOT/'checkpoints'/parent_path.parent.name
        receipt = json.loads((backup/'preservation-receipt.json').read_text())
        resume.require(receipt['sources']['report.json'] == resume.binding(report_path), 'Backup report binding differs')
        for name in ('after-policy.pt','after-resume.pt'):
            resume.require(pilot.digest(backup/name) == receipt['sources'][name]['sha256'], 'Backup changed')
        verify_pair(backup/'after-resume.pt', backup/'after-policy.pt', raw, json.loads(report_path.read_text()))
    if not args.evaluate:
        require_heldout(state, raw['runtime'], raw['native'])
    sequence = state.progress.updates if args.evaluate else args.sequence
    resume.require(args.evaluate or sequence >= state.progress.updates, 'Sequence predates parent progress')
    rows = batches.schedule(sequence, True) if args.evaluate else unused_schedule(sequence)
    check = subprocess.run([sys.executable, str(pilot.ROOT/'tools/prepare_vcmi.py'), '--check'],
        cwd=pilot.ROOT, check=True, capture_output=True, text=True, timeout=60)
    plan = {k: deepcopy(raw[k]) for k in ('runtime','native','map','binary','data','game_map','replay_binary','native_opponents')}
    plan.update(schema=(batches.EVALUATION if args.evaluate else batches.SCHEMA)+'-plan',
        mode='evaluation' if args.evaluate else 'training', scope=scope.SCOPE, source=source,
        configuration=scope.configuration(), schedule=rows, batch_sequence=sequence,
        execution_output=str(output/'execution'), parent=state.bindings, checkpoint=state.bindings['policy'],
        local_session=resume.binding(args.session), authorization=scope.AUTHORIZATION,
        absolute_deadline_unix=session['absolute_deadline_unix'], compute_deadline_unix=session['compute_deadline_unix'],
        implementation=pilot.implementation(), prerequisites=pilot.prerequisites(), prepared_check=check.stdout.strip(),
        wall_scope='Whole v8 controller at most7200 seconds, including baseline, batches, verification and preservation.',
        comparison_scope='Fresh v8 both-seat baseline and held-out evaluations; no v7 probability-parity assumption.')
    plan['checkpoint_metadata'] = {**source, 'parent_plan_sha256': state.bindings['plan']['sha256'],
        'parent_report_sha256': state.bindings['report']['sha256'], 'initial_checkpoint': state.bindings['policy'],
        'native_opponent': {k:v for k,v in plan['native_opponents'][0].items() if k!='player'},
        'archived_neural_pool_executed': False}
    pilot.check_identity(plan)
    output.mkdir(parents=True)
    migration.write_new(output/'plan.json', plan)
    return {'plan': str(output/'plan.json'), 'sha256': pilot.digest(output/'plan.json')}


def read_plan(path, sha):
    path = durable_path(path)
    resume.require(pilot.digest(path) == sha, 'V8 plan changed')
    plan = json.loads(path.read_text())
    bound = plan['local_session']
    resume.require(resume.binding(bound['path']) == bound, 'V8 session binding changed')
    session = scope.session_for(bound['path'])
    evaluation = plan['mode'] == 'evaluation'
    resume.require(plan['scope'] == scope.SCOPE and plan['configuration'] == scope.configuration()
        and plan['authorization'] == scope.AUTHORIZATION and plan['source'] == session['source']
        and plan['schema'] == (batches.EVALUATION if evaluation else batches.SCHEMA)+'-plan'
        and plan['execution_output'] == str(path.parent/'execution')
        and plan['schedule'] == batches.schedule(plan['batch_sequence'], evaluation)
        and plan['absolute_deadline_unix'] == session['absolute_deadline_unix']
        and plan['compute_deadline_unix'] == session['compute_deadline_unix'], 'V8 plan scope drift')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for flag in ('parent','output','session'): prep.add_argument('--'+flag, type=Path, required=True)
    prep.add_argument('--sequence', type=int); prep.add_argument('--evaluate', action='store_true')
    for mode in ('execute','worker'):
        item = sub.add_parser(mode); item.add_argument('--plan', type=Path, required=True); item.add_argument('--sha256', required=True)
        if mode == 'worker':
            item.add_argument('--started', type=float, required=True); item.add_argument('--wall-started', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args)
    elif args.command == 'worker':
        result = batches.worker(args, plan_reader=read_plan, state_reader=parent_state, heldout_checker=require_heldout)
    else:
        result = pilot.execute(args, plan_reader=read_plan, worker_module=__spec__.name)
        if result['status'] == 'finished' and result['training_performed']:
            preserve(args.plan, args.plan.parent/'execution/report.json', DURABLE_ROOT/'checkpoints'/args.plan.parent.name)
    print(json.dumps({k: v for k,v in result.items() if k in ('plan','sha256','status','stop_reason','optimizer_updates','optimizer_steps','training_samples')}))


if __name__ == '__main__': main()
