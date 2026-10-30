"""Four fresh games per exact-resume batch; separate fixed-seed evaluations.

Uses the pilot's collector, fresh receipts, PPO kernel, native replay, process
supervision and checkpoint roundtrip. No failed attempt is retried or replaced.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path
import time

from . import builtin_first_run as pilot
from . import builtin_training_resume as resume
from .builtin_preservation import durable_path, preserve, DURABLE_ROOT
from .inference import source_identity

SCHEMA = resume.CONTINUATION
EVALUATION = 'heroes-builtin-checkpoint-evaluation-v1'
HARD_DEADLINE = datetime(2026, 10, 25, 17, 53, tzinfo=timezone.utc).timestamp()
COMPUTE_DEADLINE = HARD_DEADLINE - 180


def configuration(session=None):
    if session is not None:
        from .builtin_local365 import configuration as local_configuration
        return local_configuration(session)
    cfg = deepcopy(pilot.config())
    cloud = json.loads((pilot.ROOT/'configs/builtin-cloud-training-v1.json').read_text())
    pilot.require(cloud == {'schema': 'heroes-builtin-cloud-training-config-v1',
        'ordinary_choice_limit': 4960, 'reserved_dialog_choices': 160, 'max_total_choices': 5120,
        'max_optimizer_steps': 640, 'max_days': 30, 'training_games': 4,
        'heldout_every_updates': 5, 'allow_verified_training_prefix': False,
        'shared_wall_seconds': 3600, 'attempt_wall_seconds': 600,
        'phase_deadlines_seconds': {'training': 2400, 'update': 3540, 'evaluation': 3540}},
        'Versioned cloud continuation scope changed')
    cfg.update(shared_wall_seconds=3600, attempt_wall_seconds=600, allow_verified_training_prefix=False,
               ordinary_choice_limit=4960, reserved_dialog_choices=160, max_total_choices=5120,
               max_optimizer_steps=640,
               phase_deadlines_seconds={'training': 2400, 'update': 3540, 'evaluation': 3540})
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
    migration = migration_for(plan.get('portable_migration'))
    if migration and plan['mode'] == 'training':
        migration.training_gate(plan['runtime'])
    return resume.load_parent(parent['plan']['path'], parent['report']['path'], plan['source'],
        current_runtime=plan['runtime'], current_native=plan['native'],
        expected_plan_sha256=parent['plan']['sha256'], expected_report_sha256=parent['report']['sha256'], migration=migration)


def migration_for(bound):
    if bound is None:
        return None
    pilot.require(resume.binding(bound['path']) == bound, 'Migration binding changed')
    from .builtin_local365 import migration_class
    return migration_class(bound['path'])(bound['path'], pilot.ROOT)


def require_heldout(state, runtime, native):
    """Do not advance beyond a fifth update before its fixed two-seat check."""
    if state.progress.updates % 5:
        return
    for path in (DURABLE_ROOT/'runs').glob('*/plan.json'):
        plan = json.loads(path.read_text())
        if (plan.get('schema') != EVALUATION+'-plan'
                or plan.get('parent', {}).get('plan', {}).get('sha256') != state.bindings['plan']['sha256']
                or plan.get('runtime') != runtime or plan.get('native') != native
                or plan.get('schedule') != schedule(state.progress.updates, True)):
            continue
        report_path = Path(plan['execution_output'])/'report.json'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        if (report.get('plan_sha256') == pilot.digest(path) and report.get('status') == 'finished'
                and report.get('worker_complete') is True and report.get('supervisor_cleanup_verified') is True
                and report.get('within_shared_budget') is True
                and report.get('training_performed') is False
                and report.get('optimizer_updates') == report.get('optimizer_steps') == 0
                and len(report.get('attempts', [])) == 2
                and all(all(row.get(k) == v for k,v in expected.items())
                        and row.get('status') == 'passed' and row.get('training_samples') == 0
                        and all(row.get(k) is True for k in ('completion_receipt','cleanup_verified',
                                                           'identity_verified','replay_verified'))
                        for expected,row in zip(plan['schedule'],report['attempts']))):
            return
    raise ValueError('The fixed held-out pair for this fifth update must pass before more training')


def prepare(args):
    import torch
    torch.set_num_threads(1)
    from .builtin_local365 import session_for
    session = session_for(getattr(args, 'session', None))
    hard_deadline = session['absolute_deadline_unix'] if session else HARD_DEADLINE
    compute_deadline = session['compute_deadline_unix'] if session else COMPUTE_DEADLINE
    minimum = 300 if not args.evaluate else 180
    pilot.require(time.time()+minimum < compute_deadline, 'Absolute cloud session deadline leaves insufficient execution/preservation reserve')
    output = durable_path(args.output)
    parent_path = durable_path(args.parent)
    parent_plan = json.loads(parent_path.read_text())
    migration = None
    if getattr(args, 'migration', None):
        from .builtin_local365 import migration_class
        migration = migration_class(args.migration)(args.migration, pilot.ROOT)
        if session:
            pilot.require(migration.binding == session['migration'], 'Session migration differs')
    report_path = (migration.resolve(parent_plan['execution_output']) if migration else Path(parent_plan['execution_output'])) / 'report.json'
    from experiments.headless.builtin_evidence import profile_for
    interface = profile_for(parent_plan["source"]["interface"]).environment
    source = source_identity(interface)
    current_runtime, current_native = parent_plan['runtime'], deepcopy(parent_plan['native'])
    runtime_bin = getattr(args, 'runtime_bin', None)
    pilot.require(migration is None or runtime_bin is not None, 'Migration requires an explicit current runtime')
    if runtime_bin is not None:
        from experiments.headless.contextual_evidence import runtime_identity
        runtime_bin = durable_path(runtime_bin)
        current_runtime = runtime_identity(runtime_bin)
        current_native['binary'] = str(runtime_bin / 'vcmiclient')
        if migration:
            from experiments.headless.env import VcmiLauncher
            pilot.require(args.data is not None and args.game_map is not None, 'Migration requires current data and map paths')
            launcher = VcmiLauncher(runtime_bin/'vcmiclient', durable_path(args.data), durable_path(args.game_map))
            launcher.preflight()
            current_native = launcher.identity()
        else:
            pilot.require(pilot.digest(current_native['binary']) == current_native['binary_sha256'], 'Client bytes changed')
    source_check = subprocess.run([sys.executable, str(pilot.ROOT/'tools/prepare_vcmi.py'), '--check'],
        cwd=pilot.ROOT, check=True, capture_output=True, text=True, timeout=60)
    if migration and not args.evaluate:
        migration.training_gate(current_runtime)
    state = resume.load_parent(parent_path, report_path, source,
        current_runtime=current_runtime, current_native=current_native,
        expected_plan_sha256=pilot.digest(parent_path), expected_report_sha256=pilot.digest(report_path), migration=migration)
    if not args.evaluate:
        require_heldout(state, current_runtime, current_native)
    pilot.require(not output.exists(), 'Use a new batch directory')
    # Require the independently restored second copy before any further game.
    receipt_path = ((migration.resolve(migration.old_root/'checkpoints'/parent_path.parent.name/'preservation-receipt.json'))
                    if migration else DURABLE_ROOT/'checkpoints'/parent_path.parent.name/'preservation-receipt.json')
    receipt = json.loads(receipt_path.read_text())
    report_bound = receipt['sources']['report.json']
    pilot.require((migration.artifact(report_bound) if migration else report_bound) == resume.binding(report_path)
                  and receipt['restore']['exact_model_adam_rng_progress_pool'] is True,
                  'Parent has not been durably preserved and restore-tested')
    for name in ('after-resume.pt', 'after-policy.pt'):
        pilot.require(pilot.digest(receipt_path.parent / name) == receipt['sources'][name]['sha256'],
                      'Parent backup changed')
    from .builtin_preservation import verify_pair
    verify_pair(receipt_path.parent/'after-resume.pt', receipt_path.parent/'after-policy.pt',
                parent_plan, json.loads(report_path.read_text()), migration=migration)
    plan = {key: deepcopy(parent_plan[key]) for key in
            ('runtime', 'native', 'map', 'binary', 'data', 'game_map', 'replay_binary', 'native_opponents')}
    plan.update(runtime=current_runtime, native=current_native, binary=current_native['binary'],
                replay_binary=resume.binding(Path(current_native['binary']).parent/'vcmireplay'))
    if migration:
        plan.update(data=str(durable_path(args.data)), game_map=str(durable_path(args.game_map)),
                    portable_migration=migration.binding,
                    native_opponents=[pilot.native_opponent(current_runtime, seat) for seat in (0,1)])
        from validation.probes.validate_hero_policy import hero_map_identity
        plan['map'] = hero_map_identity(plan['game_map'])
    sequence = getattr(args, 'sequence', None)
    if args.evaluate: sequence = state.progress.updates
    pilot.require(type(sequence) is int and sequence >= state.progress.updates, 'Choose an explicit unused batch sequence')
    if not args.evaluate:
        pilot.require(sequence >= 22, 'Cloud continuation starts at unused sequence 22 or later')
    planned_schedule = schedule(sequence, args.evaluate)
    if not args.evaluate:
        keys = {(r['engine_seed'], r['learner_player'], r['policy_seed']) for r in planned_schedule}
        prior_roots = {DURABLE_ROOT/'runs'}
        if migration:
            prior_roots.add(migration.new_root/'runs')
        for prior_path in (p for root in prior_roots for p in root.glob('*/plan.json')):
            prior = json.loads(prior_path.read_text())
            if prior.get('schema') == SCHEMA+'-plan':
                used = {(r['engine_seed'], r['learner_player'], r['policy_seed']) for r in prior['schedule']}
                pilot.require(not keys & used, 'Training seeds already scheduled; choose an unused sequence')
    if session:
        pilot.require(session['mode'] == ('gate' if args.evaluate and migration else 'campaign'), 'Session mode does not authorize this batch')
        if not args.evaluate:
            pilot.require(session['gate_verified'] is True, 'Both-seat local gate required')
        plan['local_session'] = resume.binding(args.session)
    plan.update(absolute_deadline_unix=hard_deadline, compute_deadline_unix=compute_deadline,
        prepared_check=source_check.stdout.strip(), batch_sequence=sequence, schema=(EVALUATION if args.evaluate else SCHEMA)+'-plan', configuration=configuration(session),
        schedule=planned_schedule, execution_output=str(output/'execution'),
        checkpoint=state.bindings['policy'], source=source, implementation=pilot.implementation(),
        prerequisites=pilot.prerequisites(), authorization=pilot.AUTHORIZATION, parent=state.bindings,
        mode='evaluation' if args.evaluate else 'training',
        wall_scope='3600 seconds including fresh collection, reconstruction/replay, at most one update, checkpoint and cleanup.',
        comparison_scope='Fixed paired held-out seeds and both seats; evaluation supplies no training samples.')
    if session:
        from .builtin_local365 import AUTHORIZATION
        plan['authorization'] = AUTHORIZATION
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
    from .builtin_local365 import session_for
    bound = plan.get('local_session')
    if bound:
        pilot.require(resume.binding(bound['path']) == bound, 'Local session binding changed')
    session = session_for(bound['path'] if bound else None)
    if session:
        pilot.require(session['mode'] == ('gate' if evaluation and plan.get('portable_migration') else 'campaign'), 'Session mode does not authorize plan')
        if plan.get('portable_migration'):
            pilot.require(plan['portable_migration'] == session['migration'], 'Session migration differs')
    hard = session['absolute_deadline_unix'] if session else HARD_DEADLINE
    compute = session['compute_deadline_unix'] if session else COMPUTE_DEADLINE
    pilot.require(plan.get('absolute_deadline_unix') == hard
                  and plan.get('compute_deadline_unix') == compute
                  and time.time() < compute, 'Absolute session deadline reached or changed')
    from .builtin_local365 import AUTHORIZATION
    authorization = AUTHORIZATION if session else pilot.AUTHORIZATION
    pilot.require(plan['schema'] == (EVALUATION if evaluation else SCHEMA)+'-plan'
        and plan['configuration'] == configuration(session) and plan['authorization'] == authorization
        and plan['execution_output'] == str(path.parent/'execution')
        and plan['schedule'] == schedule(plan['batch_sequence'], evaluation), 'Batch scope changed')
    return plan


def worker(args, *, plan_reader=None, state_reader=None, heldout_checker=None):
    plan_reader = plan_reader or read_plan
    state_reader = state_reader or parent_state
    heldout_checker = heldout_checker or require_heldout
    import torch
    from .builtin_gameplay import native_update, UpdateRolledBack
    from .checkpoints import TrainingProgress
    from .ppo import PPOConfig
    from validation.probes.validate_hero_policy import wall_limit
    torch.set_num_threads(1)
    plan = plan_reader(args.plan, args.sha256)
    cfg, output = plan['configuration'], Path(plan['execution_output'])
    report = json.loads((output/'report.json').read_text())
    report['parent'] = plan['parent']
    def save():
        report['execution_wall_seconds'] = time.monotonic()-args.started
        pilot.write(output/'worker-report.json', report)
    def budget(phase):
        offset = cfg['phase_deadlines_seconds'][phase]
        return pilot.Budget(args.started+offset, min(args.wall_started+offset, plan['compute_deadline_unix']))
    try:
        pilot.check_identity(plan)
        state = state_reader(plan)
        if plan['mode'] == 'training':
            heldout_checker(state, plan['runtime'], plan['native'])
        model = state.model
        receipts = []
        if 'parallel_collection' in cfg or plan.get('execution', {}).get('collection_workers', 1) == 4:
            from .parallel_collection import collect_batch
            from .builtin_gameplay import _parallel_collection_contract
            collection = _parallel_collection_contract(plan)
            batch_size = collection['batch_size']
            phase = 'evaluation' if plan['mode']=='evaluation' else 'training'
            limits = budget(phase)
            def save_parallel(index, value):
                report['attempts'][index] = value; save()
            collected = collect_batch(args.plan, args.sha256, model, state.checkpoint,
                workers=collection['workers'],
                verification_workers=plan.get('execution', {}).get('verification_workers', 1),
                pipeline_verification=plan.get('execution', {}).get('pipeline_verification', False),
                deadline_monotonic=limits.deadline, deadline_wall=limits.wall_deadline,
                save=save_parallel)
            pilot.require(len(collected.attempts) == len(plan['schedule']) == batch_size
                          and all(item['status'] == 'passed'
                                  and item.get('identity_verified') is True
                                  and item.get('cleanup_verified') is True
                                  for item in collected.attempts),
                          'Parallel collection did not verify the complete declared batch')
            pilot.require(len(collected.receipts) == (batch_size if phase == 'training' else 0),
                          'Parallel collection returned an incomplete receipt set')
            report['attempts'] = list(collected.attempts)
            report['parallel_collection'] = {
                'workers': collection['workers'],
                'games': len(collected.attempts),
                'pipeline_verification': collected.pipeline_verification,
            }
            receipts = list(collected.receipts)
            save()
        else:
            for index, row in enumerate(plan['schedule']):
                def save_attempt(value, index=index):
                    report['attempts'][index] = value; save()
                phase = 'evaluation' if plan['mode']=='evaluation' else 'training'
                # A passed attempt ended with a passed identity check; nothing runs between it and the next.
                attempt, receipt = pilot.run_attempt(plan, args.plan, args.sha256, index, model, state.checkpoint,
                    budget(phase).capped(cfg['attempt_wall_seconds']), save_attempt, identity_verified=index > 0)
                pilot.require(attempt['status']=='passed' and attempt.get('identity_verified') is True,
                              'Attempt failed; batch stopped without replacement')
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
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--evaluate',action='store_true');prep.add_argument('--runtime-bin',type=Path);prep.add_argument('--sequence',type=int);prep.add_argument('--migration',type=Path);prep.add_argument('--data',type=Path);prep.add_argument('--game-map',type=Path);prep.add_argument('--session',type=Path)
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
