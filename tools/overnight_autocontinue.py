"""One cloud-owned final-handoff -> acceptance -> PPO chain with fixed cutoff.

Run from a committed bootstrap checkout. Author/status writes and the later
immutable executable release use separate checkouts. No human/browser wake is
needed. This process and the independent watchdog have distinct exclusive locks.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

BOOTSTRAP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BOOTSTRAP))
from tools.overnight_handoff import materialize,validate_marker,binding,put,COMPUTE,HARD,OLD
from experiments.headless.process_deadline import guarded_popen,close_owned_process,cleanup_from_evidence

BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
AUTHOR=Path('/workspace/heroes-overnight')
PRIVATE=Path('/workspace/heroes-overnight-private')
IMPORT_REPO=Path('/workspace/heroes-rl-training-state')
PUBLIC_BRANCH='codex/teacher-overnight-20261026'
ENV={**os.environ,'HEROES_ARTIFACT_ROOT':'/workspace/Heroes-artifacts',
    'HEROES_OVERNIGHT_OWNER':'teacher-20261026','HEROES_OVERNIGHT_ROLE':'preservation',
    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def utc():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def event(kind,**fields):
    row=dict(timestamp_utc=utc(),event=kind,**fields)
    for path in (BASE/'autocontinue-events.jsonl',AUTHOR/'validation/overnight-cloud-2026-10-26.jsonl'):
        with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    with (AUTHOR/'docs/research/overnight-cloud-2026-10-26.md').open('a') as f:
        f.write('\n- '+row['timestamp_utc']+' '+kind+': '+json.dumps(fields)+'\n')


def git(repo,*args):
    left=HARD-time.time()-10
    if left<=0:raise TimeoutError('Absolute Git publication deadline')
    return subprocess.check_output(['git','-C',str(repo),*args],env=ENV,timeout=min(75,left)).decode().strip()


def public_push(message,config=False):
    paths=['docs/research/overnight-cloud-2026-10-26.md','validation/overnight-cloud-2026-10-26.jsonl',
        'reports/cloud/teacher-overnight-status.json']
    if config:paths.append('configs/teacher-overnight-linux-v1.json')
    git(AUTHOR,'add','-f','--',*paths)
    if git(AUTHOR,'diff','--cached','--name-only'):
        git(AUTHOR,'commit','-m',message)
    git(AUTHOR,'push','origin','HEAD:refs/heads/'+PUBLIC_BRANCH)
    return git(AUTHOR,'rev-parse','HEAD')


def status(phase,**fields):
    old={}
    path=AUTHOR/'reports/cloud/teacher-overnight-status.json'
    if path.exists():old=json.loads(path.read_text())
    old.update(timestamp_utc=utc(),phase=phase,compute_deadline_utc='2026-10-26T09:55:00Z',
        hard_deadline_utc='2026-10-26T10:00:00Z',remaining_compute_seconds=max(0,COMPUTE-time.time()),
        autonomous_controller_pid=os.getpid(),**fields)
    atomic(path,old);atomic(BASE/'autocontinue-status.json',old)


def run(command,label,cwd,*,compute=True,maximum=None):
    wall=COMPUTE if compute else HARD-30
    if maximum is not None:wall=min(wall,time.time()+maximum)
    if wall-time.time()<15:raise TimeoutError('No safe owned job reserve: '+label)
    mono=time.monotonic()+wall-time.time()
    env={**ENV,'HEROES_OVERNIGHT_ROLE':'compute' if compute else 'preservation'}
    proc=None
    try:
        with (BASE/(label+'.log')).open('x') as log:
            proc=guarded_popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                deadline_wall=wall,deadline_monotonic=mono,evidence_path=BASE/(label+'-deadline.jsonl'))
            while proc.poll() is None:
                if time.time()>=wall-2:raise TimeoutError('Owned job absolute deadline: '+label)
                time.sleep(min(5,max(.05,wall-time.time()-2)))
            if proc.returncode:raise RuntimeError(label+' exited '+str(proc.returncode)+'; retained log '+str(BASE/(label+'.log')))
    finally:
        if proc is not None:close_owned_process(proc,timeout=1)
        cleanup_from_evidence(BASE/(label+'-deadline.jsonl'),timeout=1)


def private_publish(folder,label):
    dest=PRIVATE/'results/overnight-20261026'/label
    dest.mkdir(parents=True,exist_ok=False)
    archive=dest/'evidence.tar.gz'
    with tarfile.open(archive,'x:gz',dereference=False) as t:t.add(folder,arcname=folder.name)
    atomic(dest/'receipt.json',dict(archive=binding(archive),external_data_dereferenced=False))
    git(PRIVATE,'add','--',str(dest.relative_to(PRIVATE)))
    git(PRIVATE,'commit','-m','Preserve overnight '+label)
    git(PRIVATE,'push','origin','HEAD:refs/heads/'+PUBLIC_BRANCH)
    return dict(commit=git(PRIVATE,'rev-parse','HEAD'),archive=binding(archive))


def wait_for_handoff():
    last_publish=0
    while time.time()<COMPUTE-1800:
        candidate=BASE/'handoff-candidate.json'
        if candidate.exists():
            value=json.loads(candidate.read_text())
            validate_marker(value['marker'],time.time())
            return value['private_commit']
        if time.time()-last_publish>600:
            status('waiting_final_local_handoff',ownership='local',cloud_training_started=False,
                blocker='Final verified local ownership transfer not yet published',new_admitted_games=0,new_ppo_updates=0)
            public_push('Record cloud-owned wait for final local handoff')
            last_publish=time.time()
        time.sleep(5)
    raise TimeoutError('No final local handoff with safe Linux acceptance reserve')


def verify_runtime():
    path=BASE/'runtime-ready.json'
    while time.time()<COMPUTE-1800:
        if path.exists():
            r=json.loads(path.read_text())
            if r.get('passed') is not True:raise ValueError('Current canonical Linux runtime failed preparation')
            from experiments.headless.contextual_evidence import runtime_identity
            if runtime_identity(Path(r['runtime_bin']))!=r['runtime']:raise ValueError('Prepared runtime bytes changed')
            return r
        time.sleep(5)
    raise TimeoutError('No prepared canonical Linux runtime with safe reserve')


def prepare_context(receipt,runtime):
    import torch
    torch.set_num_threads(1)
    from experiments.learner.builtin_preservation import verify_pair
    from experiments.learner.builtin_training_resume import _completion
    m=receipt['marker'];parent=receipt['parent']
    plan=json.loads(Path(parent['plan']['path']).read_text());report=json.loads(Path(parent['report']['path']).read_text())
    progress,steps,_=_completion(plan,report)
    if vars(progress)!=m['progress'] or steps!=m['cumulative_adam_steps']:raise ValueError('Computed final progress differs')
    migration=SimpleNamespace(checkpoint_options={'historical_torch_version':plan['source']['torch']})
    proofs=[verify_pair(parent['resume']['path'],parent['policy']['path'],plan,report,migration=migration)]
    backups=[]
    for dest in (BASE/'checkpoints/imported-final-local',PRIVATE/'checkpoints/overnight-20261026/imported-final-local'):
        already=dest.exists()
        dest.mkdir(parents=True,exist_ok=True)
        for name,key in [('after-policy.pt','policy'),('after-resume.pt','resume'),('plan.json','plan'),('report.json','report')]:
            if already:
                observed=binding(dest/name)
                if any(observed[k]!=parent[key][k] for k in ('sha256','bytes')):raise ValueError('Existing imported backup differs; quarantine instead of overwriting')
            else:shutil.copy2(parent[key]['path'],dest/name)
        proof=verify_pair(dest/'after-resume.pt',dest/'after-policy.pt',plan,report,migration=migration)
        if already:
            if json.loads((dest/'restore-receipt.json').read_text())!=proof:raise ValueError('Existing restore receipt differs; preserve it for diagnosis')
        else:atomic(dest/'restore-receipt.json',proof)
        backups.append(str(dest));proofs.append(proof)
    git(PRIVATE,'add','--','checkpoints/overnight-20261026/imported-final-local')
    if git(PRIVATE,'diff','--cached','--name-only'):
        git(PRIVATE,'commit','-m','Preserve exact final local handoff before cloud acceptance')
    git(PRIVATE,'push','origin','HEAD:refs/heads/'+PUBLIC_BRANCH)
    # Identical historical inputs with their original behavior checkpoint, not
    # the changed final weights. This is offline portability evidence only.
    imports=Path(receipt['relocated_root'])
    def relocate(path):return imports/Path(path).relative_to(OLD)
    previous_plan=relocate(plan['parent']['plan']['path'])
    previous_report=relocate(plan['parent']['report']['path'])
    previous_policy=relocate(plan['checkpoint_metadata']['initial_checkpoint']['path'])
    episode=relocate(report['attempts'][0]['episode_directory'])
    drift=BASE/'platform-drift.json'
    from tools.measure_overnight_platform_drift import episode_record
    record,_=episode_record(episode)
    event('platform_drift_preregistered',absolute_tolerance=1e-5,relative_tolerance=1e-4,
        original_behavior_checkpoint=binding(previous_policy),episode=binding(record),training_samples=0)
    run([sys.executable,str(BOOTSTRAP/'tools/measure_overnight_platform_drift.py'),'--episode',str(episode),
        '--checkpoint',str(previous_policy),'--plan',str(previous_plan),'--report',str(previous_report),'--output',str(drift)],
        'platform-drift',BOOTSTRAP,maximum=300)
    drift_result=json.loads(drift.read_text())
    if not drift_result['passed']:raise ValueError('Platform drift exceeds preregistered tolerance')
    for seat in (0,1):
        source=BASE/'imports'/receipt['ready_commit']/'maps'/f'curriculum-3-{seat}.vmap'
        put(BASE/'maps'/source.name,source.read_bytes())
    glue='experiments/learner/builtin_first_run.py'
    current_glue=binding(AUTHOR/glue)['sha256']
    value=dict(schema='heroes-overnight-linux-continuation-v1',parent=parent,
        historical_source=m['source'],historical_runtime=m['runtime'],linux_runtime=runtime['runtime'],
        ownership_transfer_verified=True,handoff_verification=binding(BASE/'imports'/receipt['ready_commit']/'handoff/verification.json'),
        ready_commit=receipt['ready_commit'],archive_commit=receipt['archive_commit'],public_source_commit=m['public_commit'],
        relocation_roots={str(OLD):receipt['relocated_root']},historical_campaign_root=receipt['historical_campaign_root'],
        progress=m['progress'],optimizer_steps=steps,next_unused_training_sequence=m['next_unused_training_sequence'],
        map_sha256_by_seat=[m['maps_by_seat'][str(seat)]['sha256'] for seat in (0,1)],
        heldout_schedule_by_seat=m['heldout_schedule_by_seat'],restore_backup_directories=backups,
        initial_checkpoint_dependency=m['initial_checkpoint_dependency'],
        glue_changes={glue:{'from':m['source']['source_sha256'][glue],'to':current_glue}},
        platform_drift_report=binding(drift),hard_deadline_unix=HARD,compute_deadline_unix=COMPUTE,
        monotonic_clock_scope='Fresh Linux wall/monotonic anchor; original Mac monotonic clock is immutable provenance only',
        historical_training_admission=False)
    atomic(AUTHOR/'configs/teacher-overnight-linux-v1.json',value)
    event('final_handoff_exact_restore_verified',ready_commit=receipt['ready_commit'],progress=m['progress'],
        optimizer_steps=steps,three_restore_proofs=proofs,platform_drift=drift_result)
    status('handoff_restored_preparing_linux_acceptance',ownership='cloud',checkpoint=parent,
        cloud_training_started=False,blocker='Fresh Linux both-seat mixed-outcome acceptance and baseline required')
    commit=public_push('Pin final local state and explicit Linux continuation context',config=True)
    return commit


def create_release(commit):
    release=Path('/workspace/releases')/('teacher-overnight-'+commit[:12])
    release.parent.mkdir(exist_ok=True)
    git(AUTHOR,'worktree','add','--detach',str(release),commit)
    run(['git','clone','--shared','--no-checkout',str(AUTHOR/'vcmi'),str(release/'vcmi')],
        'release-engine-clone',BOOTSTRAP,maximum=120)
    revision=json.loads((release/'dependencies/vcmi.json').read_text())['commit']
    run(['git','-C',str(release/'vcmi'),'checkout','--detach',revision],'release-engine-checkout',BOOTSTRAP,maximum=120)
    run([sys.executable,str(release/'tools/prepare_vcmi.py')],'release-engine-prepare',release,maximum=300)
    run([sys.executable,str(release/'tools/prepare_vcmi.py'),'--check'],'release-engine-check',release,maximum=120)
    return release


def campaign(release):
    command=[sys.executable,str(release/'tools/overnight_curriculum_campaign.py')]
    run([*command,'freeze'],'freeze-final-release',release,maximum=120)
    event('linux_acceptance_intent',executable_release=str(release),policy_games=64,controls=4,training_samples=0)
    public_push('Preregister fresh Linux mixed-outcome acceptance')
    acceptance_error=None
    try:run([sys.executable,str(release/'tools/run_overnight_acceptance.py')],'linux-acceptance',release,maximum=1220)
    except Exception as exc:acceptance_error=exc
    proof=private_publish(BASE/'study4','linux-acceptance-001')
    event('linux_acceptance_retained',private_preservation=proof,error=str(acceptance_error) if acceptance_error else None)
    if acceptance_error:raise acceptance_error
    run([*command,'prepare-release'],'prepare-training-release',release,maximum=180)
    status('launching_verified_cloud_campaign',ownership='cloud',executable_release=str(release),
        cloud_training_started=False,blocker=None)
    event('automatic_campaign_launch_intent',release=str(release),compute_cutoff=COMPUTE,hard_stop=HARD)
    public_push('Publish accepted Linux release and automatic campaign launch')
    proc=None;last_publish=0
    wall=HARD-90;mono=time.monotonic()+wall-time.time()
    try:
        with (BASE/'automatic-campaign.log').open('x') as log:
            proc=guarded_popen([*command,'launch'],cwd=release,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                deadline_wall=wall,deadline_monotonic=mono,evidence_path=BASE/'automatic-campaign-deadline.jsonl')
            atomic(BASE/'automatic-campaign-command.json',dict(pid=proc.pid,command=[*command,'launch'],cwd=str(release),
                ownership='cloud',launch_utc=utc(),hard_deadline_unix=HARD,compute_deadline_unix=COMPUTE))
            while proc.poll() is None and time.time()<wall-2:
                if time.time()-last_publish>600:
                    public_push('Publish overnight cloud campaign progress');last_publish=time.time()
                time.sleep(5)
        if proc.poll() is None:raise TimeoutError('Final controller exceeded preservation reserve')
        if proc.returncode:raise RuntimeError('Automatic campaign controller exited '+str(proc.returncode))
    finally:
        if proc is not None:close_owned_process(proc,timeout=1)
        cleanup_from_evidence(BASE/'automatic-campaign-deadline.jsonl',timeout=1)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--repair-preparation',default=None);args=parser.parse_args(argv)
    repair=args.repair_preparation
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'autocontinue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise
    prior=BASE/'autocontinue-finished.json'
    if repair is None:
        if prior.exists():raise ValueError('Consumed autonomous chain cannot restart')
    else:
        if repair!='preparation-repair-002':raise ValueError('Explicit unique repair identity required')
        failed=json.loads(prior.read_text())
        if failed['status']!='stopped_before_training' or failed['training_started'] is not False or (BASE/'campaign-v1/session.json').exists() or (BASE/'study4').exists():raise ValueError('Preparation-only repair cannot reuse any gameplay attempt')
        if (BASE/(repair+'-finished.json')).exists():raise ValueError('Repair attempt already consumed')
    atomic(BASE/'autocontinue-command.json',dict(pid=os.getpid(),started_utc=utc(),bootstrap=str(BOOTSTRAP),
        command=[sys.executable,*sys.argv],duplicate_exclusion='exclusive flock autocontinue.lock plus immutable claim and session lock',
        compute_deadline_unix=COMPUTE,hard_deadline_unix=HARD))
    event('autonomous_chain_started',pid=os.getpid(),bootstrap=str(BOOTSTRAP))
    result=dict(status='stopped_before_training',training_started=False)
    try:
        ready_commit=wait_for_handoff();runtime=verify_runtime()
        if repair is None and (BASE/'handoff-consumed.json').exists():raise ValueError('Final handoff already consumed by a chain')
        if repair is not None:
            claimed=json.loads((BASE/'handoff-consumed.json').read_text())
            if claimed['ready_commit']!=ready_commit:raise ValueError('Preparation repair must retain the exact same verified handoff')
            event('preparation_repair_intent',attempt=repair,prior_failure=binding(prior),same_handoff=ready_commit,deadline_extended=False)
        git(AUTHOR,'fetch','origin','codex/teacher-curriculum-v1')
        receipt=materialize(IMPORT_REPO,ready_commit,BASE/'imports'/ready_commit,AUTHOR)
        if repair is None:
            with (BASE/'handoff-consumed.json').open('x') as f:json.dump(dict(ready_commit=ready_commit,pid=os.getpid(),at_utc=utc()),f)
        commit=prepare_context(receipt,runtime)
        release=create_release(commit)
        campaign(release)
        report_path=BASE/'campaign-v1/campaign-report.json'
        report=json.loads(report_path.read_text())
        result.update(status=report['status'],training_started=report['new_updates']>0,
            new_updates=report['new_updates'],progress=report['latest_progress'],checkpoint=report['latest_verified_parent'])
        event('automatic_campaign_finished',result=result)
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}')
        event('autonomous_chain_stopped',result=result)
    finally:
        result.update(stopped_utc=utc(),hard_deadline_unix=HARD,compute_deadline_unix=COMPUTE,restart_authorized=False)
        atomic(BASE/((repair+'-finished.json') if repair else 'autocontinue-finished.json'),result)
        status('stopped_'+result['status'],ownership='cloud' if (BASE/'handoff-consumed.json').exists() else 'local',
            blocker=result.get('error'),autonomous_result=result)
        if time.time()<HARD-20:public_push('Publish overnight automatic continuation result')
        lock.close()


if __name__=='__main__':main()
