#!/usr/bin/env python3
"""One authorized two-hour local continuation, after an explicit frozen gate.

No cloud restart, retries, seed replacement, partial updates or background
extension. Fixed durable session and exclusive controller lock forbid a second
campaign launch. All active batches retain a clean, pinned source checkout.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.learner.builtin_preservation import DURABLE_ROOT, verify_pair
from experiments.learner.builtin_training_resume import binding,require
from experiments.learner.builtin_local365 import create_session,session_for,LocalMigration


def invoke(command,log):
    with log.open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0,'Command failed; inspect '+str(log))


def batch(parent,session,output,sequence,migration=None):
    command=[sys.executable,'-m','experiments.learner.builtin_training','prepare','--parent',str(parent),
        '--output',str(output),'--session',str(session)]
    command+=['--evaluate'] if sequence is None else ['--sequence',str(sequence)]
    if migration:
        value=json.loads(migration.read_text());native=value['runtime_transition']['to_native']
        command+=['--migration',str(migration),'--runtime-bin',str(Path(native['binary']).parent),
            '--data',str(DURABLE_ROOT/'game-data'),'--game-map',str(DURABLE_ROOT/'inputs/experiments/explorer/maps/hero-management-v1.vmap')]
    logs=DURABLE_ROOT/'local365/logs';logs.mkdir(parents=True,exist_ok=True)
    invoke(command,logs/(output.name+'-prepare.log'))
    plan=output/'plan.json'
    invoke([sys.executable,'-m','experiments.learner.builtin_training','execute','--plan',str(plan),
            '--sha256',binding(plan)['sha256']],logs/(output.name+'-execute.log'))
    report=json.loads((output/'execution/report.json').read_text())
    require(report['status']=='finished' and report['worker_complete'] and report['supervisor_cleanup_verified']
            and report['within_shared_budget'],'Batch did not finish cleanly; no replacement')
    return plan,report


def publish(plan,repo,deadline):
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=repo,text=True,timeout=max(1,min(90,deadline-time.time()))).strip()
    require(git('remote','get-url','origin')=='https://github.com/tkorsi/heroes-rl-training-state.git','Wrong private repository')
    require(git('branch','--show-current')=='codex/local365-teacher-results','Wrong private branch')
    require(not git('status','--porcelain'),'Private checkout must be clean before publishing')
    raw=json.loads(plan.read_text());report=json.loads((Path(raw['execution_output'])/'report.json').read_text())
    backup=DURABLE_ROOT/'checkpoints'/plan.parent.name
    proof=verify_pair(backup/'after-resume.pt',backup/'after-policy.pt',raw,report)
    target=repo/'checkpoints/local365'/plan.parent.name
    require(not target.exists(),'Snapshot already exists; no overwrite')
    shutil.copytree(backup,target)
    copied=verify_pair(target/'after-resume.pt',target/'after-policy.pt',raw,report)
    require(copied == proof, 'Private copy restoration differs')
    for name in ('after-resume.pt','after-policy.pt','plan.json','report.json'):
        require(binding(target/name)['sha256'] == binding(backup/name)['sha256'], 'Private copy bytes differ')
    archive=repo/'results/local365'/(plan.parent.name+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'x:gz',dereference=False) as tf:tf.add(plan.parent,arcname=plan.parent.name)
    receipt={'plan':binding(plan),'restore':proof,'archive':binding(archive),'game_data_symlinks_followed':False}
    archive.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    git('add','--',str(target.relative_to(repo)),str(archive.relative_to(repo)),str(archive.with_suffix('.json').relative_to(repo)))
    git('commit','-m','Preserve local365 Teacher update '+str(proof['progress']['updates']))
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results')
    return git('rev-parse','HEAD')


def campaign(args):
    session=DURABLE_ROOT/'local365/campaign-session.json'
    value=session_for(session)
    parent=args.parent;sequence=32;durations=[300.0];records=[]
    result={'schema':'heroes-local365-campaign-v1','session':binding(session),'status':'running','batches':records}
    try:
        while True:
            session_for(session)
            parent_plan=json.loads(parent.read_text())
            migration=args.migration if parent_plan['source']['torch'].endswith('+cpu') else None
            report_path=(LocalMigration(migration,ROOT).resolve(parent_plan['execution_output']) if migration else Path(parent_plan['execution_output']))/'report.json'
            report=json.loads(report_path.read_text());update=report['cumulative_progress']['updates']
            evaluate=update%5==0 and not (records and records[-1]['mode']=='evaluation' and records[-1]['at_update']==update)
            reserve=max(300,2*max(durations[-5:])+120)
            if value['compute_deadline_unix']-time.time()<reserve:
                result['status']='stopped_for_deadline_reserve';break
            # Never reuse a previously prepared sequence, even if its run failed.
            while (DURABLE_ROOT/'runs'/f'local365-continuation-{sequence:03d}').exists():sequence+=1
            name=f'local365-heldout-update-{update:03d}' if evaluate else f'local365-continuation-{sequence:03d}'
            plan,new=batch(parent,session,DURABLE_ROOT/'runs'/name,None if evaluate else sequence,migration)
            row={'mode':'evaluation' if evaluate else 'training','at_update':update,'plan':binding(plan),
                 'report':binding(plan.parent/'execution/report.json'),'samples':new['training_samples'],
                 'optimizer_steps':new['optimizer_steps'],'wall_seconds':new['execution_wall_seconds'],
                 'outcomes':[x.get('outcomes') for x in new['attempts']]}
            records.append(row)
            if not evaluate:
                row['private_commit']=publish(plan,args.private_repo,value['absolute_deadline_unix'])
                parent=plan;sequence+=1;durations.append(new['execution_wall_seconds'])
            result['latest_parent']=binding(parent)
            (session.parent/'campaign-report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    except Exception as error:
        result.update(status='stopped_failure',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(stopped_unix=time.time(),latest_parent=binding(parent))
        result['within_two_hours']=result['stopped_unix']<=value['absolute_deadline_unix']
        (session.parent/'campaign-report.json').write_text(json.dumps(result,indent=2)+'\n')


def supervise(args):
    from experiments.headless.process_deadline import guarded_popen,close_owned_process
    require(args.private_repo is not None,'Private checkpoint publication required')
    session=DURABLE_ROOT/'local365/campaign-session.json'
    lock=session.with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    # An exclusive durable lock prevents a second authorized two-hour launch.
    with lock.open('x') as stream:stream.write(str(os.getpid())+'\n')
    value=create_session(session,'campaign',args.migration)
    command=[sys.executable,str(Path(__file__).resolve()),'campaign-worker','--parent',str(args.parent),
             '--migration',str(args.migration),'--private-repo',str(args.private_repo)]
    deadline=value['absolute_deadline_unix']-10
    process=None;result={'session':binding(session),'controller_cleanup_verified':False}
    try:
        with (session.parent/'controller.log').open('x') as log:
            process=guarded_popen(command,deadline_monotonic=time.monotonic()+deadline-time.time(),
                deadline_wall=deadline,evidence_path=session.parent/'controller-deadline.jsonl',
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            result['returncode']=process.wait(timeout=max(1,deadline-time.time()))
    finally:
        if process is not None:close_owned_process(process,timeout=2)
        result.update(controller_cleanup_verified=True,stopped_unix=time.time())
        result['within_two_hours']=result['stopped_unix']<=value['absolute_deadline_unix']
        (session.parent/'controller-result.json').write_text(json.dumps(result,indent=2)+'\n')
    require(result.get('returncode')==0,'Local controller stopped; inspect durable report')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['gate','campaign','campaign-worker'])
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--migration',type=Path,required=True)
    p.add_argument('--private-repo',type=Path);args=p.parse_args()
    require(os.environ.get('HEROES_ARTIFACT_ROOT')==str(DURABLE_ROOT),'Explicit durable local artifact root required')
    if args.mode=='gate':
        session=DURABLE_ROOT/'local365/gate-session.json';create_session(session,'gate',args.migration)
        plan,report=batch(args.parent,session,DURABLE_ROOT/'runs/local365-update31-gate',None,args.migration)
        print(json.dumps({'plan':binding(plan),'report':binding(plan.parent/'execution/report.json')}))
    elif args.mode == 'campaign-worker':
        require(args.private_repo is not None,'Private checkpoint publication required');campaign(args)
    else:
        supervise(args)

if __name__=='__main__':main()
