"""Resume unchanged frozen Teacher after verified transient Git publication faults.

Only complete checkpoint-verified batches with closed owned processes qualify.
The immutable executable release, acceptance, model and original cloud clock
remain unchanged. Every recovery gets a fresh session and unused game sequences.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import overnight_autocontinue as owner
from experiments.learner.builtin_preservation import verify_pair
from experiments.learner.builtin_training_resume import binding,require,_completion
from experiments.headless.process_deadline import guarded_popen,close_owned_process,cleanup_from_evidence

BASE=owner.BASE
PRIVATE=owner.PRIVATE
COMPUTE=owner.COMPUTE
HARD=owner.HARD
BRANCH=owner.PUBLIC_BRANCH


def transient(error):
    return bool(re.search(r'(?:HTTP (?:429|502|503|504)|returned error: (?:429|502|503|504))',error))


def git(*args):
    timeout=min(75,HARD-time.time()-20)
    require(timeout>0,'Absolute publication stop')
    return subprocess.run(['git','-C',str(PRIVATE),*args],env=owner.ENV,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,timeout=timeout)


def checked(*args):
    result=git(*args)
    require(result.returncode==0,'Git metadata command failed: '+args[0])
    return result.stdout


def push_ack():
    """Idempotent bounded retry; remote acknowledgement, never return-code guessing."""
    head=checked('rev-parse','HEAD').decode().strip()
    for attempt in range(3):
        pushed=git('push','origin','HEAD:refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH)
        if remote.returncode==0 and remote.stdout.decode().split()[0]==head:
            return dict(commit=head,remote_acknowledged=True,attempts=attempt+1)
        require(pushed.returncode==0 or transient(pushed.stderr.decode(errors='replace')),
            'Non-transient or divergent private publication; do not force or rewrite')
        require(time.time()<HARD-60,'No safe publication retry reserve')
        time.sleep(3)
    raise RuntimeError('Private publication unavailable after bounded retries')


def verify_stopped(report_path):
    v=json.loads(report_path.read_text())
    stopped=json.loads((report_path.parent/'controller-result.json').read_text())
    require(stopped['owned_cleanup_verified'] is True and stopped['within_budget'] is True,'Previous owned controller cleanup required')
    require(v['status']=='stopped_failure' and 'push' in v.get('error',''),'Only publication-only failure may recover automatically')
    last=v['batches'][-1]
    require(last.get('checkpoint_verified') is True and last.get('owned_cleanup_verified') is True
        and last.get('preservation_error') and not last.get('error') and not last.get('cleanup_error'),
        'No replay/gameplay/PPO/integrity failure may use transport recovery')
    log=(report_path.parent/'controller.log').read_text()
    require(transient(log),'A concrete transient HTTP publication error is required')
    plan_path=Path(v['latest_verified_parent']['path']);require(binding(plan_path)==v['latest_verified_parent'],'Verified parent drift')
    plan=json.loads(plan_path.read_text());report=json.loads((Path(plan['execution_output'])/'report.json').read_text())
    progress,steps,_=_completion(plan,report)
    require(vars(progress)==v['latest_progress'] and stopped['latest_restore']['progress']==v['latest_progress'],
        'Stopped progress/parent differs')
    proofs=[]
    for folder in [plan_path.parent/'execution',*[Path(p['directory']) for p in stopped['final_independent_restores']]]:
        proof=verify_pair(folder/'after-resume.pt',folder/'after-policy.pt',plan,report)
        require(proof['progress']==v['latest_progress'],'Restored copy progress differs')
        proofs.append(dict(directory=str(folder),restore=proof))
    private_folder=next(Path(p['directory']) for p in stopped['final_independent_restores'] if Path(p['directory']).is_relative_to(PRIVATE))
    for kind in ('policy','resume'):
        path=private_folder/f'after-{kind}.pt'
        blob=checked('show','HEAD:'+str(path.relative_to(PRIVATE)))
        require(hashlib.sha256(blob).hexdigest()==report['after_checkpoint'][kind]['sha256'],
            'Private Git checkpoint bytes differ from complete state')
    return v,stopped,plan_path,plan,report,proofs


def prepare_session(old_report,index,details):
    v,stopped,parent,plan,report,proofs=details
    target=BASE/'campaign-v1/recoveries'/f'http-transport-{index:03d}'
    target.mkdir(parents=True,exist_ok=False)
    for name in ('frozen.json','release.json'):
        (target/name).symlink_to(BASE/'campaign-v1'/name)
    # These are additional real copies for finalization even if recovery stops
    # before another update; the original copies and metadata remain untouched.
    backup=target/'checkpoints'/parent.parent.name
    private=PRIVATE/'checkpoints/overnight-20261026/campaign-v1'/target.name/parent.parent.name
    source=Path(stopped['final_independent_restores'][0]['directory'])
    for dest in (backup,private):
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,dest,symlinks=False)
        proof=verify_pair(dest/'after-resume.pt',dest/'after-policy.pt',plan,report)
        owner.atomic(dest/'transport-copy-restore.json',proof)
    checked('add','--',str(private.relative_to(PRIVATE)))
    checked('commit','-m','Preserve verified parent for '+target.name)
    acknowledgement=push_ack()
    record=dict(attempt=target.name,parent=binding(parent),progress=v['latest_progress'],
        optimizer_steps=stopped['latest_restore']['optimizer_steps'],copies=proofs,
        private_ack=acknowledgement,prior_report=binding(old_report),
        executable_release=json.loads((BASE/'campaign-v1/frozen.json').read_text())['checkout'],
        original_window=binding(BASE/'campaign-v1/window.json'),
        compute_deadline_unix=COMPUTE,hard_deadline_unix=HARD,clock_reset=False,
        learner_or_runtime_changed=False,failed_game_reused=False)
    owner.atomic(target/'transport-recovery-receipt.json',record)
    owner.event('transient_publication_recovery_preregistered',**record)
    return target,record


def main():
    p=argparse.ArgumentParser();p.add_argument('--from-report',type=Path,required=True);a=p.parse_args()
    lock=(BASE/'autocontinue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(time.time()<COMPUTE-600,'No safe original compute window for recovery')
    owner.atomic(BASE/'transport-watch-command.json',dict(pid=os.getpid(),command=[sys.executable,*sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(),compute_deadline_unix=COMPUTE,hard_deadline_unix=HARD))
    current=a.from_report.resolve();result={}
    try:
        for index in range(1,33):
            require(time.time()<COMPUTE-600,'No safe original compute window for another recovery')
            details=verify_stopped(current);ack=push_ack()
            owner.event('private_publication_acknowledged_after_transient_failure',ack=ack,
                verified_checkpoint=details[0]['latest_verified_parent'],three_exact_restores=details[-1])
            target,record=prepare_session(current,index,details)
            release=Path(record['executable_release']);session=target/'session.json'
            env={**owner.ENV,'HEROES_CURRICULUM_SESSION':str(session)}
            command=[sys.executable,str(release/'tools/overnight_curriculum_campaign.py'),'recover','--recover-from',str(current)]
            wall=HARD-90;mono=time.monotonic()+wall-time.time();proc=None;last_publish=0
            try:
                with (target/'transport-supervisor.log').open('x') as log:
                    proc=guarded_popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        deadline_wall=wall,deadline_monotonic=mono,evidence_path=target/'transport-supervisor-deadline.jsonl')
                    owner.atomic(target/'owned-command.json',dict(pid=proc.pid,command=command,session=str(session)))
                    while proc.poll() is None and time.time()<wall-2:
                        if time.time()-last_publish>600:
                            try:owner.public_push('Publish resumed overnight Teacher progress')
                            except Exception:pass # Preserve local facts; private admission still blocks on publication.
                            last_publish=time.time()
                        time.sleep(5)
            finally:
                if proc is not None:close_owned_process(proc,timeout=1)
                cleanup_from_evidence(target/'transport-supervisor-deadline.jsonl',timeout=1)
            current=target/'campaign-report.json';v=json.loads(current.read_text())
            result=dict(status=v['status'],new_updates=v['new_updates'],progress=v['latest_progress'],report=binding(current))
            if v['status']!='stopped_failure':break
            # The next loop rechecks the exact error and all copies before any restart.
        owner.event('transport_watch_stopped',result=result)
    except Exception as exc:
        result.update(status='stopped_transport_watch',error=f'{type(exc).__name__}: {exc}')
        owner.event('transport_watch_blocked',result=result,current_report=str(current))
    finally:
        result.update(stopped_utc=datetime.now(timezone.utc).isoformat(),compute_deadline_unix=COMPUTE,hard_deadline_unix=HARD)
        owner.atomic(BASE/'transport-watch-result.json',result)
        if time.time()<HARD-20:
            try:owner.public_push('Publish overnight recovery controller result')
            except Exception:pass
        lock.close()


if __name__=='__main__':main()
