"""Four fresh games per exact-resume batch; separate fixed-seed evaluations.

Uses the pilot's collector, fresh receipts, PPO kernel, native replay, process
supervision and checkpoint roundtrip. No failed attempt is retried or replaced.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

from . import builtin_first_run as pilot
from . import builtin_training_resume as resume
from .builtin_preservation import durable_path, preserve, DURABLE_ROOT
from .inference import source_identity

SCHEMA = resume.CONTINUATION
EVALUATION = 'heroes-builtin-checkpoint-evaluation-v1'


def configuration():
    cfg = deepcopy(pilot.config())
    cfg.update(shared_wall_seconds=600, allow_verified_training_prefix=False,
               phase_deadlines_seconds={'training': 480, 'update': 570, 'evaluation': 570})
    return cfg


def schedule(update, evaluation=False):
    pilot.require(type(update) is int and 1 <= update < 100000, 'Invalid parent update')
    if evaluation:
        cfg = pilot.config()
        return [dict(episode=seat+1, phase='evaluation', engine_seed=cfg['evaluation_engine_seed'],
                     learner_player=seat, policy_seed=cfg['evaluation_policy_seeds'][seat]) for seat in (0, 1)]
    return [dict(episode=i+1, phase='training', engine_seed=170000+2*update+i//2,
                 learner_player=i%2, policy_seed=270000+4*update+i) for i in range(4)]


def parent_state(plan):
    parent = plan['parent']
    return resume.load_parent(parent['plan']['path'], parent['report']['path'], plan['source'],
        current_runtime=plan['runtime'], current_native=plan['native'],
        expected_plan_sha256=parent['plan']['sha256'], expected_report_sha256=parent['report']['sha256'])


def prepare(args):
    import torch
    torch.set_num_threads(1)
    output = durable_path(args.output)
    parent_path = durable_path(args.parent)
    parent_plan = json.loads(parent_path.read_text())
    report_path = Path(parent_plan['execution_output']) / 'report.json'
    source = source_identity(pilot.INTERFACE)
    current_runtime, current_native = parent_plan['runtime'], deepcopy(parent_plan['native'])
    runtime_bin = getattr(args, 'runtime_bin', None)
    if runtime_bin is not None:
        from experiments.headless.contextual_evidence import runtime_identity
        runtime_bin = durable_path(runtime_bin)
        current_runtime = runtime_identity(runtime_bin)
        current_native['binary'] = str(runtime_bin / 'vcmiclient')
        pilot.require(pilot.digest(current_native['binary']) == current_native['binary_sha256'], 'Client bytes changed')
    state = resume.load_parent(parent_path, report_path, source,
        current_runtime=current_runtime, current_native=current_native,
        expected_plan_sha256=pilot.digest(parent_path), expected_report_sha256=pilot.digest(report_path))
    pilot.require(not output.exists(), 'Use a new batch directory')
    # Require the independently restored second copy before any further game.
    receipt_path = DURABLE_ROOT / 'checkpoints' / parent_path.parent.name / 'preservation-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    pilot.require(receipt['sources']['report.json'] == resume.binding(report_path)
                  and receipt['restore']['exact_model_adam_rng_progress_pool'] is True,
                  'Parent has not been durably preserved and restore-tested')
    for name in ('after-resume.pt', 'after-policy.pt'):
        pilot.require(pilot.digest(receipt_path.parent / name) == receipt['sources'][name]['sha256'],
                      'Parent backup changed')
    plan = {key: deepcopy(parent_plan[key]) for key in
            ('runtime', 'native', 'map', 'binary', 'data', 'game_map', 'replay_binary', 'native_opponents')}
    plan.update(runtime=current_runtime, native=current_native, binary=current_native['binary'],
                replay_binary=resume.binding(Path(current_native['binary']).parent/'vcmireplay'))
    sequence = getattr(args, 'sequence', None)
    if args.evaluate: sequence = state.progress.updates
    pilot.require(type(sequence) is int and sequence >= state.progress.updates, 'Choose an explicit unused batch sequence')
    planned_schedule = schedule(sequence, args.evaluate)
    if not args.evaluate:
        keys = {(r['engine_seed'], r['learner_player'], r['policy_seed']) for r in planned_schedule}
        for prior_path in (DURABLE_ROOT/'runs').glob('*/plan.json'):
            prior = json.loads(prior_path.read_text())
            if prior.get('schema') == SCHEMA+'-plan':
                used = {(r['engine_seed'], r['learner_player'], r['policy_seed']) for r in prior['schedule']}
                pilot.require(not keys & used, 'Training seeds already scheduled; choose an unused sequence')
    plan.update(batch_sequence=sequence, schema=(EVALUATION if args.evaluate else SCHEMA)+'-plan', configuration=configuration(),
        schedule=planned_schedule, execution_output=str(output/'execution'),
        checkpoint=state.bindings['policy'], source=source, implementation=pilot.implementation(),
        prerequisites=pilot.prerequisites(), authorization=pilot.AUTHORIZATION, parent=state.bindings,
        mode='evaluation' if args.evaluate else 'training',
        wall_scope='600 seconds including fresh collection, reconstruction/replay, at most one update, checkpoint and cleanup.',
        comparison_scope='Fixed paired held-out seeds and both seats; evaluation supplies no training samples.')
    plan['checkpoint_metadata'] = {**source, 'parent_plan_sha256': state.bindings['plan']['sha256'],
        'parent_report_sha256': state.bindings['report']['sha256'], 'initial_checkpoint': state.bindings['policy'],
        'native_opponent': {k:v for k,v in plan['native_opponents'][0].items() if k!='player'},
        'archived_neural_pool_executed': False}
    pilot.check_identity(plan)
    output.mkdir(parents=True)
    pilot.write(output/'plan.json', plan)
    return {'plan':str(output/'plan.json'), 'sha256':pilot.digest(output/'plan.json')}


def read_plan(path, sha):
    path = durable_path(path)
    pilot.require(pilot.digest(path) == sha, 'Immutable batch plan changed')
    plan = json.loads(path.read_text())
    evaluation = plan.get('mode') == 'evaluation'
    pilot.require(plan['schema'] == (EVALUATION if evaluation else SCHEMA)+'-plan'
        and plan['configuration'] == configuration() and plan['authorization'] == pilot.AUTHORIZATION
        and plan['execution_output'] == str(path.parent/'execution')
        and plan['schedule'] == schedule(plan['batch_sequence'], evaluation), 'Batch scope changed')
    return plan


def worker(args):
    import torch
    from .builtin_gameplay import native_update, UpdateRolledBack
    from .checkpoints import TrainingProgress
    from .ppo import PPOConfig
    from validation.probes.validate_hero_policy import wall_limit
    torch.set_num_threads(1)
    plan = read_plan(args.plan, args.sha256)
    cfg, output = plan['configuration'], Path(plan['execution_output'])
    report = json.loads((output/'report.json').read_text())
    report['parent'] = plan['parent']
    def save():
        report['execution_wall_seconds'] = time.monotonic()-args.started
        pilot.write(output/'worker-report.json', report)
    def budget(phase):
        offset = cfg['phase_deadlines_seconds'][phase]
        return pilot.Budget(args.started+offset, args.wall_started+offset)
    try:
        pilot.check_identity(plan)
        state = parent_state(plan)
        model = state.model
        receipts = []
        for index, row in enumerate(plan['schedule']):
            def save_attempt(value, index=index):
                report['attempts'][index] = value; save()
            phase = 'evaluation' if plan['mode']=='evaluation' else 'training'
            attempt, receipt = pilot.run_attempt(plan, args.plan, args.sha256, index, model, state.checkpoint,
                budget(phase).capped(cfg['attempt_wall_seconds']), save_attempt)
            pilot.require(attempt['status']=='passed', 'Attempt failed; batch stopped without replacement')
            if phase == 'training':
                pilot.require(receipt is not None, 'Missing fresh receipt'); receipts.append(receipt)
        if plan['mode'] == 'training':
            report['update_status']='started'; save()
            model.train().requires_grad_(True)
            update_budget = budget('update')
            try:
                with wall_limit(update_budget.remaining(cfg['cleanup_reserve_seconds'])):
                    result, rollouts = native_update(model, state.optimizer, tuple(receipts), plan_sha256=args.sha256,
                        config=PPOConfig(**cfg['ppo']), generator=state.generator,
                        remaining=lambda:update_budget.remaining(cfg['cleanup_reserve_seconds']),
                        max_optimizer_steps=cfg['max_optimizer_steps'])
            except UpdateRolledBack as error:
                report.update(update_status='rolled_back', attempted_optimizer_steps=error.completed_steps)
                raise
            report.update(update_status='completed', training_performed=True, optimizer_updates=1,
                          optimizer_steps=result.optimizer_steps, training_samples=result.samples, ppo=asdict(result))
            for index, rollout in enumerate(rollouts): report['attempts'][index]['training_samples']=len(rollout.samples)
            progress = TrainingProgress(state.progress.updates+1, state.progress.rollouts+len(receipts),
                                        state.progress.decisions+result.samples)
            report['cumulative_progress']=vars(progress); save()
            model.eval().requires_grad_(False)
            sampler={'next_schedule_index':len(plan['schedule']),
                     'optimizer_steps':state.optimizer_steps+result.optimizer_steps,
                     'batch_optimizer_steps':result.optimizer_steps}
            with wall_limit(update_budget.remaining()):
                _, report['after_checkpoint'] = pilot.checkpoint_roundtrip(output, model, state.optimizer,
                    state.pool, state.generator, resume._metadata(plan,args.sha256),cfg,progress,sampler,'after')
            report['update_status']='checkpoint_verified'; save()
        pilot.check_identity(plan)
        report.update(status='finished', stop_reason=None)
    except Exception as error:
        report.update(status='stopped',stop_reason=f'{type(error).__name__}: {error}')
        if report['update_status']=='started':
            report.update(update_status='interrupted_unknown',optimizer_updates=None,optimizer_steps=None,training_performed=None)
    for row in report['attempts']:
        if row['status']=='not_started': row['reason']=report.get('stop_reason') or 'not_started'
    report['worker_complete']=True; save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--parent',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--evaluate',action='store_true');prep.add_argument('--runtime-bin',type=Path);prep.add_argument('--sequence',type=int)
    for name in ('execute','worker'):
        entry=sub.add_parser(name);entry.add_argument('--plan',type=Path,required=True);entry.add_argument('--sha256',required=True)
        if name=='worker':
            entry.add_argument('--started',type=float,required=True);entry.add_argument('--wall-started',type=float,required=True)
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(args)
    elif args.command=='worker': result=worker(args)
    else:
        result=pilot.execute(args,plan_reader=read_plan,worker_module='experiments.learner.builtin_training')
        if result['status']=='finished' and result['training_performed']:
            preserve(args.plan,Path(args.plan).parent/'execution/report.json',DURABLE_ROOT/'checkpoints'/Path(args.plan).parent.name)
    print(json.dumps(result if args.command=='prepare' else {k:result.get(k) for k in
          ('status','stop_reason','optimizer_updates','optimizer_steps','training_samples','execution_wall_seconds')},indent=2))

if __name__=='__main__': main()
