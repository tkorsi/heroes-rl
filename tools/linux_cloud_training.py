"""One explicitly authorized Linux continuation, with private durable admission.

No model service is called. The existing collector, verifier and PPO execute the
games. Mac exclusion is the pinned handoff fence; cloud exclusion is one atomic
private Git claim plus a writable local flock, never an expiring takeover.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(Path(path).read_bytes())


def binding(path):
    from tools.linux_publication import binding as actual
    return actual(path)


def verify_lifetime(path):
    """Require an observed post-turn interval, never infer it from nohup."""
    receipt = read(path)
    require(receipt['schema'] == 'heroes-cloud-post-turn-probe-v1'
            and receipt['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'Post-turn lifetime evidence is for another VM')
    for value in (receipt['completion'], receipt['heartbeats']):
        require(binding(value['path']) == value, 'Lifetime evidence changed')
    completion = read(receipt['completion']['path'])
    from experiments.headless.contextual_evidence import strict_json
    rows = [strict_json(line) for line in Path(receipt['heartbeats']['path']).read_bytes().splitlines()]
    ended = receipt['agent_turn_completed_unix']
    require(type(ended) in (int,float) and completion['status'] == 'deadline_reached'
            and completion['gaps_observed'] == 0 and completion['started_unix'] < ended,
            'No complete background observation across agent turn completion')
    after = [row for row in rows if row['observed_unix'] > ended]
    require(len(after) >= 12 and after[-1]['observed_unix']-after[0]['observed_unix'] >= 60
            and all(row['pid'] == completion['pid'] and row['run_id'] == completion['run_id']
                    and row['gap_observed'] is False for row in rows),
            'Insufficient continuous post-turn background observations')
    return dict(post_turn_seconds_observed=after[-1]['observed_unix']-ended,
                run_id=completion['run_id'], ten_hour_service_guarantee=False)


def validate_native(folder, runtime, expected_count):
    from validation.probes.validate_garrison_dialogs import implementation_identity
    from tools.validate_garrison_regressions import implementation_identity as regression_identity
    folder = Path(folder)
    plan, report, shutdown = [read(folder/name) for name in
                             ('execution/plan.json', 'execution/report.json', 'shutdown.json')]
    require(plan['implementation'] == (implementation_identity() if expected_count == 16 else regression_identity())
            and plan['native_runtime'] == runtime and report['status'] == 'passed'
            and len(report['cases']) == len(plan['schedule']) == expected_count
            and all(row['status'] == 'passed' for row in report['cases'])
            and shutdown['returncode'] == 0 and shutdown['owned_cleanup_verified'] is True
            and shutdown['within_hard_deadline'] is True, 'Incomplete Linux native acceptance')
    return {name:binding(folder/file) for name,file in
            [('plan','execution/plan.json'),('report','execution/report.json'),('shutdown','shutdown.json')]}


def prepare(args):
    os.environ['HEROES_ARTIFACT_ROOT'] = str(args.artifact_root.resolve())
    from tools.linux_artifacts import Evidence, HANDOFF, FULL_STATE, MANIFEST_SHA256, FENCE_SHA256
    from tools.linux_runtime import check_freeze
    from tools import run_local_training as controller
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    from experiments.learner.inference import source_identity
    from experiments.headless.contextual_evidence import runtime_identity
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
            'Commit the complete Linux implementation before acceptance')
    access = Evidence(args.bundle, args.evidence)
    profile = read(ROOT/'configs/interrupted1229-recovery.json')
    audit = access.read(profile['immutable_evidence']['checkpoint_audit']['path'])
    old_config = access.read(audit['source_runtime']['machine']['path'])
    old_session = access.read(Path(profile['run_dir'])/'session.json')
    recovery = next(r['original_path'] for r in access.manifest['files'] if 'recovery_receipt' in r['categories'])
    with access.recovery_access():
        parent, remaining, continuation = controller.original_continuation(
            old_session['original_continuation']['origin_run'], old_config, recovery, artifacts=access)
    require(remaining == 2600 and continuation['next_sequence'] == 869
            and continuation['admitted_before_session'] == 2400, 'Fenced continuation counts differ')
    parent = access.rebind(parent)
    freeze = check_freeze(args.freeze)
    runtime = runtime_identity(Path(freeze['runtime_bin']))
    native = validate_native(args.garrison, runtime, 16)
    regression = validate_native(args.regression, runtime, 28)
    pp, pr, ps = [read(args.policy/name) for name in ('plan.json','execution/report.json','shutdown.json')]
    require(pp['source'] == source_identity(local.INTERFACE) and pp['runtime'] == runtime
            and pp['freeze'] == binding(args.freeze) and pp['mode'] == 'evaluation'
            and pr['plan_sha256'] == binding(args.policy/'plan.json')['sha256']
            and pr['status'] == 'passed' and pr['optimizer_updates'] == pr['optimizer_steps'] == pr['training_samples'] == 0
            and pr['full_parent_unchanged'] is True and pr['full_parent_restore'] is True
            and len(pr['attempts']) == len(pp['schedule']) == 8
            and ps['returncode'] == 0 and ps['owned_cleanup_verified'] is True,
            'Frozen Linux policy acceptance incomplete')
    require([r['learner_player'] for r in pp['schedule']] == [0]*4+[1]*4, 'Both frozen-policy seats required')
    for expected, row in zip(pp['schedule'], pr['attempts']):
        require(all(row[k] == v for k,v in expected.items()) and row['status'] == 'passed'
                and row['training_samples'] == 0 and row['guard_invariants_verified'] is True
                and all(row[k] is True for k in ('completion_receipt','cleanup_verified','identity_verified','replay_verified','weights_unchanged'))
                and row['reconstruction']['passed'] is True
                and row['reconstruction']['maximum_absolute_difference'] == 0, 'Unverified frozen policy game')
    execution = local.check_execution(old_session['execution'])
    require(execution['collection_workers'] == 4 and execution['experiments'] == [], 'Original execution choices differ')
    require(old_session['policy_support'] == 'visible-interaction-budget-v1', 'Policy support changed')
    root = args.artifact_root.resolve()
    require(not root.is_relative_to(ROOT), 'Private artifacts must stay outside the source checkout')
    lifetime = verify_lifetime(args.lifetime_receipt)
    config = dict(schema='heroes-linux1229-cloud-job-v1', default_profile='original',
        seed_namespace=local.SEED_NAMESPACES['original'], source=source_identity(local.INTERFACE),
        implementation=pilot.implementation(), prerequisites=pilot.prerequisites(), parent=parent,
        configuration=local.operational_configuration(parent, 10800), runtime=runtime,
        runtime_bin=freeze['runtime_bin'], data=str(args.data.resolve()), maps=[freeze['map']]*2,
        artifact_root=str(root), run_root=str(root/'official-cloud'), execution=execution,
        policy_support='visible-interaction-budget-v1', handoff_id=HANDOFF, full_state_sha256=FULL_STATE,
        remaining_games=remaining, continuation=continuation,
        evidence=dict(freeze=binding(args.freeze), manifest=binding(args.bundle/'manifest.json'),
                      lifetime=binding(args.lifetime_receipt),
                      fence=binding(args.bundle/'handoff-control.json'),
                      **{'native/'+k:v for k,v in native.items()},
                      **{'regression/'+k:v for k,v in regression.items()},
                      **{'policy/'+k:binding(args.policy/name) for k,name in
                         [('plan','plan.json'),('report','execution/report.json'),('shutdown','shutdown.json')]}),
        handoff_manifest_sha256=MANIFEST_SHA256, mac_fence_sha256=FENCE_SHA256,
        base_ref='refs/heads/codex/checkpoint1229-full-handoff',
        base_commit='cc674fc0f7f455c74d34f139255ab8aefb85c3b1',
        bundle=str(args.bundle.resolve()), extracted=str(args.evidence.resolve()),
        authorized_hours=10, platform_lifetime_observation=lifetime, automatic_restart=False, training_started=False)
    require(not args.output.exists(), 'Existing machine descriptor will not be overwritten')
    from tools.run_local_training import atomic
    atomic(args.output, config)
    print(json.dumps({'config':binding(args.output),'training_ready_checks_passed':True,'training_started':False}))


def check_config(path, expected):
    require(binding(path)['sha256'] == expected, 'Linux machine configuration changed')
    value = read(path)
    os.environ['HEROES_ARTIFACT_ROOT'] = value['artifact_root']
    from tools.linux_runtime import check_freeze
    from tools.linux_artifacts import HANDOFF, MANIFEST_SHA256, FENCE_SHA256
    from experiments.learner.inference import source_identity
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    require(value['schema'] == 'heroes-linux1229-cloud-job-v1' and value['authorized_hours'] == 10
            and value['handoff_id'] == HANDOFF and value['handoff_manifest_sha256'] == MANIFEST_SHA256
            and value['mac_fence_sha256'] == FENCE_SHA256 and value['automatic_restart'] is False,
            'Unsupported Linux continuation or handoff')
    require(source_identity(local.INTERFACE) == value['source'] and not value['source']['tracked_worktree_dirty']
            and pilot.implementation() == value['implementation'] and pilot.prerequisites() == value['prerequisites'],
            'Frozen Linux source or prerequisites changed')
    for item in value['evidence'].values(): require(binding(item['path']) == item, 'Linux acceptance evidence changed')
    verify_lifetime(value['evidence']['lifetime']['path'])
    check_freeze(value['evidence']['freeze']['path'])
    state = local.restore_parent(value['parent'])
    from tools.recover_interrupted_training import fingerprint
    import torch, random
    snapshot=dict(model=dict(state.model.state_dict()),optimizer=state.optimizer.state_dict(),
        generator=state.generator.get_state(),pool=state.pool.state_dict(),progress=vars(state.progress),
        sampler=state.sampler_state,torch_rng=torch.get_rng_state(),python_rng=random.getstate())
    require(fingerprint(snapshot) == value['full_state_sha256'] and state.progress.updates == 1229
            and state.optimizer_steps == 21238, 'Complete starting state differs')
    return value


def freeze_source(artifact_root):
    # A regular independent clone freezes the source; no additional Git worktree.
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
            'Commit the complete source before freezing')
    release = Path(artifact_root).resolve()/'release'/commit
    release.parent.mkdir(parents=True, exist_ok=True)
    if not release.exists():
        subprocess.run(['git','clone','--no-hardlinks','--local',str(ROOT),str(release)],check=True)
        subprocess.run(['git','-C',str(release),'checkout','--detach',commit],check=True)
        # Upstream has tracked symlinks; dereferencing them changes its recipe.
        shutil.copytree(ROOT/'vcmi',release/'vcmi',symlinks=True,
                        ignore=lambda folder,names: ['build'] if Path(folder)==ROOT/'vcmi' else [])
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=release,text=True).strip() == commit
            and not subprocess.check_output(['git','status','--porcelain'],cwd=release,text=True).strip(),
            'Existing frozen release differs; no overwrite or repair')
    subprocess.run([sys.executable,str(release/'tools/prepare_vcmi.py'),'--check'],cwd=release,check=True)
    return release


def launch(args):
    config = check_config(args.config, args.sha256)
    require(args.hours == 10, 'Exactly the authorized ten-hour maximum is supported')
    root = Path(config['run_root'])
    require(not root.exists(), 'Cloud run root already exists; never restart or reset a later state')
    require(shutil.disk_usage(config['artifact_root']).free >= 12*1024**3, 'At least 12 GiB private run headroom required')
    release = freeze_source(config['artifact_root'])
    root.mkdir(mode=0o700)
    wall, mono = time.time(), time.monotonic()
    command = [sys.executable,str(release/'tools/linux_cloud_training.py'),'_supervisor',
               '--config',str(args.config.resolve()),'--sha256',args.sha256,'--hours','10',
               '--launched-wall',repr(wall),'--launched-mono',repr(mono)]
    with (root/'supervisor.log').open('x') as log:
        process = subprocess.Popen(command,cwd=release,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
    from tools.run_local_training import atomic
    atomic(root/'launch-request.json',dict(pid=process.pid,process_group=os.getpgid(process.pid),command=command,
        started_unix=wall,hard_deadline_unix=wall+36000,training_started=False))
    print(json.dumps({'launch_requested':True,'pid':process.pid,'process_group':os.getpgid(process.pid),
                      'receipt':str(root/'launch-request.json'),'live_training_not_yet_confirmed':True}))


def arm_supervisor_deadline(wall, monotonic, evidence):
    """Arm the existing independent guardian around this detached leader."""
    from experiments.headless.process_deadline import _deadline, _remaining, _event, _guardian
    wall = _deadline(wall, 'wall'); monotonic = _deadline(monotonic, 'monotonic')
    require(os.getpid() == os.getpgrp() == os.getsid(0) and _remaining(monotonic, wall) > 0,
            'Independent supervisor deadline needs a live private session')
    evidence = Path(evidence)
    fd = os.open(evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    _event(evidence, 'standalone_supervisor_requested', leader_pid=os.getpid(),
           deadline_wall=wall, deadline_monotonic=monotonic)
    owner_read, owner_write = os.pipe()
    ready_read, ready_write = os.pipe()
    leader = os.getpid()
    child = os.fork()
    if child == 0:
        os.close(owner_write); os.close(ready_read)
        try:
            code = _guardian(leader, monotonic, wall, evidence, ready_write, owner_read)
        except BaseException:
            if os.getppid() == leader: os.killpg(leader, 9)
            code = 125
        os._exit(code)
    os.close(owner_read); os.close(ready_write)
    try:
        require(os.read(ready_read, 1) == b'1', 'Supervisor guardian failed to arm')
    finally:
        os.close(ready_read)
    # The leader owns this noninheritable descriptor until it exits. The child
    # independently enforces both clocks even during blocking final publication.
    return owner_write


def supervisor(args):
    guardian_owner = arm_supervisor_deadline(args.launched_wall+36000,
        args.launched_mono+36000, args.config.parent/'official-cloud'/'supervisor-deadline.jsonl')
    config = check_config(args.config,args.sha256)
    from tools import run_local_training as controller
    from tools.linux_artifacts import Evidence, HANDOFF
    from tools.linux_publication import publish_transaction, ref_sha
    from experiments.learner import local_script_training as local, builtin_first_run as pilot
    from experiments.learner.inference import source_identity
    root = Path(config['run_root']); run=root/'session'
    require(args.hours == 10 and os.getsid(0) == os.getpid() and os.getpgrp() == os.getpid(),
            'Supervisor must own its separate session and process group')
    hard=args.launched_wall+36000
    require(min(hard-time.time(),args.launched_mono+36000-time.monotonic())>300, 'Original launch deadline exhausted')
    with controller.exclusive_lock(root/'owner.lock'):
        require(not (root/'session.json').exists() and not (root/'pending-checkpoint.json').exists(), 'Consumed cloud launch')
        access=Evidence(config['bundle'],config['extracted']); access.fence()
        require(ref_sha(config['base_ref'],hard) == config['base_commit'], 'Private handoff ref changed before claim')
        run.mkdir()
        claim=dict(handoff_id=HANDOFF,pid=os.getpid(),process_group=os.getpgrp(),started_unix=args.launched_wall,
                   hard_deadline_unix=hard,source=config['source'],runtime=config['runtime'],config=binding(args.config))
        controller.atomic(run/'claim.json',claim)
        claim_ref='refs/heads/codex/claim-'+HANDOFF
        require(ref_sha(claim_ref,hard) is None, 'This handoff already has a cloud claim; no duplicate or restart')
        claimed=publish_transaction(base_ref=config['base_ref'],base_sha=config['base_commit'],destination_ref=claim_ref,
            prefix='claims/'+HANDOFF,files={'claim.json':run/'claim.json'},deadline=hard,
            temporary_root=Path(config['artifact_root'])/'publication-tmp')
        cloud=dict(handoff_id=HANDOFF,claim={'ref':claim_ref,'commit':claimed['commit']},
                   results_ref='refs/heads/codex/results-'+HANDOFF)
        session=dict(schema=local.SESSION_SCHEMA,scope=local.SCOPE,profile='original',
            source=source_identity(local.INTERFACE),implementation=pilot.implementation(),prerequisites=pilot.prerequisites(),
            configuration=config['configuration'],batch_wall_seconds=10800,execution=config['execution'],
            policy_support=config['policy_support'],asset_config=binding(args.config),
            runtime_bin=config['runtime_bin'],runtime=config['runtime'],data=config['data'],maps=config['maps'],
            artifact_root=config['artifact_root'],run_root=str(root),run_dir=str(run),initial_parent=config['parent'],
            next_sequence=869,supervisor_pid=os.getpid(),training_engine_seed_base=40000000,training_policy_seed_base=50000000,
            max_updates=650,target_games=2600,evaluation_interval_updates=100,
            original_continuation={**config['continuation'],'evidence':{'linux_acceptance':binding(args.config)}},
            frozen_files={name:binding(ROOT/name) for name in
                ('tools/linux_cloud_training.py','tools/linux_publication.py','tools/linux_runtime.py','tools/linux_artifacts.py',
                 'tools/run_local_training.py','tools/prepare_local_training.py','tools/recover_interrupted_training.py',
                 'configs/interrupted1229-recovery.json','train-local.sh')},
            started_unix=args.launched_wall,started_monotonic=args.launched_mono,
            compute_deadline_unix=hard-300,compute_deadline_monotonic=args.launched_mono+35700,
            absolute_deadline_unix=hard,hard_deadline_monotonic=args.launched_mono+36000,linux_cloud=cloud,
            authorization='One explicitly authorized ten-hour cloud continuation; no LLM calls or automatic restart')
        controller.atomic(root/'sequence-ledger.json',config['continuation']['seed_ledger_snapshot'])
        controller.atomic(root/'session.json',session)
        path=run/'session.json';controller.atomic(path,session)
        controller.atomic(root/'latest-run.json',dict(run_dir=str(run),session=binding(path)))
        initial=publish_transaction(base_ref=claim_ref,destination_ref=cloud['results_ref'],prefix='cloud/'+HANDOFF+'/start',
            files={'session.json':path,'sequence-ledger.json':root/'sequence-ledger.json','config.json':args.config},
            deadline=hard-300,temporary_root=Path(config['artifact_root'])/'publication-tmp')
        controller.atomic(root/'live-session.json',dict(pid=os.getpid(),process_group=os.getpgrp(),session=binding(path),
            starting_checkpoint=1229,next_sequence=869,remaining_games=2600,private_start_receipt=initial,
            started_unix=args.launched_wall,hard_deadline_unix=hard,training_supervisor_live=True))
        rc=controller.supervise(path)
        files={name:run/name for name in ('session.json','campaign-report.json','summary.json','shutdown.json')}
        files['sequence-ledger.json']=root/'sequence-ledger.json'
        if (root/'latest-checkpoint.json').exists(): files['latest-checkpoint.json']=root/'latest-checkpoint.json'
        if (root/'latest-checkpoint.json').exists():
            latest = read(root/'latest-checkpoint.json')['parent']
            for key,name in [('plan','plan.json'),('report','report.json'),('policy','after-policy.pt'),
                             ('resume','after-resume.pt'),('initial_checkpoint','initial-checkpoint.pt')]:
                files[name] = Path(latest[key]['path'])
        if (root/'pending-checkpoint.json').exists():
            files['pending-checkpoint.json']=root/'pending-checkpoint.json'
            pending = read(root/'pending-checkpoint.json')['parent']
            for key in ('plan','report','policy','resume','initial_checkpoint'):
                files['pending-'+key+Path(pending[key]['path']).suffix] = Path(pending[key]['path'])
        closure=publish_transaction(base_ref=claim_ref,destination_ref=cloud['results_ref'],prefix='cloud/'+HANDOFF+'/closure',
            files=files,deadline=hard,temporary_root=Path(config['artifact_root'])/'publication-tmp')
        controller.atomic(root/'private-closure.json',closure)
        live = read(root/'live-session.json')
        live.update(training_supervisor_live=False, private_closure=closure, supervisor_returncode=rc)
        controller.atomic(root/'live-session.json',live)
        return rc


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for name in ('bundle','evidence','freeze','garrison','regression','policy','artifact-root','data','output','lifetime-receipt'):
        p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('freeze-source');p.add_argument('--artifact-root',type=Path,required=True)
    for action in ('check','launch','_supervisor'):
        p=sub.add_parser(action);p.add_argument('--config',type=Path,required=True);p.add_argument('--sha256',required=True)
        p.add_argument('--hours',type=float,default=10)
        if action=='_supervisor':
            p.add_argument('--launched-wall',type=float,required=True);p.add_argument('--launched-mono',type=float,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='freeze-source':print(json.dumps({'release':str(freeze_source(args.artifact_root))}))
    elif args.action=='check':check_config(args.config,args.sha256);print(json.dumps({'checks_passed':True,'training_started':False}))
    elif args.action=='launch':launch(args)
    else:return supervisor(args)
    return 0


if __name__=='__main__':raise SystemExit(main())
