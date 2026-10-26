"""Publish one Git-only local-to-cloud handoff after verified local shutdown.

This tool never launches games. It operates from the authoring checkout after
all local controllers have exited; the frozen executable release is unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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
from experiments.learner.builtin_training_resume import binding,require
from experiments.learner.builtin_preservation import verify_pair
from experiments.headless.process_deadline import cleanup_from_evidence

BASE=Path('/Users/orsi/Projects/Heroes-artifacts/local-curriculum-update146')
PRIVATE=Path('/Users/orsi/Projects/Heroes-artifacts/private-local365-state')
REL=Path('handoffs/2026-10-26-local-to-cloud')


def utc():return datetime.now(timezone.utc).isoformat()


def git(*args,binary=False):
    raw=subprocess.check_output(['git','-C',str(PRIVATE),*args],timeout=60)
    return raw if binary else raw.decode().strip()


def validate_final(session,controller,journal,now,mono):
    require(controller.get('owned_cleanup_verified') is True,'Controller cleanup unverified')
    require(journal['status'] in ('finished_budget_reserve','stopped_failure'),'Local campaign still active')
    require(journal['new_updates']>0,'No completed local PPO update to hand off')
    require(min(session['absolute_deadline_unix']-now,session['started_monotonic']+7200-mono)<=600,
            'Do not shorten the local window for handoff')
    require(session['absolute_deadline_unix']==session['started_unix']+7200,'Original window changed')
    require(session['compute_deadline_unix']==session['started_unix']+6900,'Finalization reserve changed')


def main(session_root,finalization_proof=None):
    session_root=Path(session_root).resolve();require(session_root.is_relative_to(BASE/'campaign-v3'),'Wrong campaign namespace')
    require(not (PRIVATE/REL/'ready.json').exists(),'Unique readiness manifest already published; never overwrite it')
    require(git('remote','get-url','origin')=='https://github.com/tkorsi/heroes-rl-training-state.git'
        and git('branch','--show-current')=='codex/local365-teacher-results' and not git('status','--porcelain'),'Private Git checkout not ready')
    session=json.loads((session_root/'session.json').read_text())
    controller=json.loads((session_root/'controller-result.json').read_text())
    journal=json.loads((session_root/'campaign-report.json').read_text())
    validate_final(session,controller,journal,time.time(),time.monotonic())
    # Check every recorded attempt, including failed/recovery attempts, without
    # signalling old numeric PIDs that might now belong to unrelated processes.
    proof=[]
    for path in (BASE/'campaign-v3').rglob('*deadline*.jsonl'):
        groups=cleanup_from_evidence(path,timeout=2)
        proof.append(dict(receipt=binding(path),closed_groups=list(groups)))
    controller_checks=[]
    for path in (BASE/'campaign-v3').rglob('session.json'):
        value=json.loads(path.read_text());pid=value['controller_pid']
        process=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=5)
        command=process.stdout.strip()
        require('curriculum_campaign.py' not in command,'An owned local controller is still running')
        controller_checks.append(dict(session=binding(path),pid=pid,owned_command_absent=True))
    checkpoint_plan=Path(journal['latest_verified_parent']['path'])
    require(binding(checkpoint_plan)==journal['latest_verified_parent'],'Final parent plan changed')
    plan=json.loads(checkpoint_plan.read_text());report_path=Path(plan['execution_output'])/'report.json'
    report=json.loads(report_path.read_text());saved=report['after_checkpoint']
    restoration=verify_pair(saved['resume']['path'],saved['policy']['path'],plan,report)
    require(restoration['progress']==journal['latest_progress'],'Final counters mismatch')
    initial=plan['checkpoint_metadata']['initial_checkpoint']
    require(binding(initial['path'])==initial,'Final checkpoint initial-policy dependency changed')
    require(binding(plan['parent']['plan']['path'])==plan['parent']['plan'],'Parent plan dependency changed')
    require(binding(plan['parent']['report']['path'])==plan['parent']['report'],'Parent report dependency changed')
    require(binding(plan['parent']['resume']['path'])==plan['parent']['resume'],'Parent resume dependency changed')
    public_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    public_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    require(public_branch=='codex/teacher-curriculum-v1','Publish authored work from the explicit Teacher branch')
    target=PRIVATE/REL;require(not target.exists(),'Preserve any partial handoff before a distinct recovery')
    target.mkdir(parents=True);bundle=target/'bundle';bundle.mkdir()
    sources={'after-policy.pt':Path(saved['policy']['path']),'after-resume.pt':Path(saved['resume']['path']),
        'plan.json':checkpoint_plan,'report.json':report_path,'dependencies/initial-policy.pt':Path(initial['path']),
        'dependencies/parent-plan.json':Path(plan['parent']['plan']['path']),
        'dependencies/parent-report.json':Path(plan['parent']['report']['path']),
        'dependencies/parent-resume.pt':Path(plan['parent']['resume']['path']),
        'maps/curriculum-3-0.vmap':BASE/'maps/curriculum-3-0.vmap',
        'maps/curriculum-3-1.vmap':BASE/'maps/curriculum-3-1.vmap',
        'window.json':BASE/'campaign-v3/window.json'}
    for name in ('session.json','release.json','frozen.json','campaign-report.json','controller-result.json','analysis.json','research-log.jsonl'):
        sources[name]=session_root/name
    if finalization_proof is not None:sources['automatic-finalization-proof.json']=Path(finalization_proof)
    index={}
    for name,source in sources.items():
        require(source.is_file() and not source.is_symlink(),'Handoff inputs must be ordinary files')
        out=bundle/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out)
        original=binding(source);require(binding(out)['sha256']==original['sha256'],'Handoff copy mismatch')
        index[name]=dict(original=original,committed_path=str((REL/'bundle'/name)),sha256=original['sha256'],bytes=original['bytes'])
    copy_restore=verify_pair(bundle/'after-resume.pt',bundle/'after-policy.pt',plan,report)
    (bundle/'restore-receipt.json').write_text(json.dumps(dict(canonical=restoration,copied=copy_restore),indent=2)+'\n')
    shutdown=dict(verified_utc=utc(),receipts=proof,controllers=controller_checks,
        local_auto_resume=False,reason='No automatic recovery process remains; finalization reserve leaves less than the controller minimum600seconds required for recovery. Ownership will transfer to cloud.')
    (bundle/'shutdown-proof.json').write_text(json.dumps(shutdown,indent=2)+'\n')
    archive=target/'bundle.tar.gz'
    with tarfile.open(archive,'x:gz',dereference=False) as t:t.add(bundle,arcname='bundle')
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if m.isfile():require(hashlib.sha256(t.extractfile(m).read()).hexdigest()==binding(target/m.name)['sha256'],'Archive mismatch')
    archive_bound=binding(archive)
    (target/'inventory.json').write_text(json.dumps(index,indent=2)+'\n')
    git('add','--',str(REL));git('commit','-m','Preserve final local curriculum state and complete cloud dependencies')
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results');archive_commit=git('rev-parse','HEAD')
    for name,item in index.items():require(hashlib.sha256(git('show',archive_commit+':'+item['committed_path'],binary=True)).hexdigest()==item['sha256'],'Committed dependency differs')
    require(hashlib.sha256(git('show',archive_commit+':'+str(REL/'bundle.tar.gz'),binary=True)).hexdigest()==archive_bound['sha256'],'Committed archive differs')
    next_sequence=1+max(b['sequence'] for b in journal['batches'] if b['mode']=='training')
    from experiments.learner.curriculum_training import schedule
    deadline=datetime(2026,10,26,10,tzinfo=timezone.utc).timestamp()
    require(time.time()<deadline,'Cloud hard deadline elapsed; do not publish readiness or extend it')
    seal=dict(timestamp_utc=utc(),campaign_session=binding(session_root/'session.json'),local_auto_resume=False,
        ownership='local -> cloud',no_local_training_after_readiness=True)
    with (BASE/'campaign-v3/local-resume-disabled.json').open('x') as f:json.dump(seal,f,indent=2);f.write('\n')
    manifest=dict(schema='heroes-local-to-cloud-ready-v1',ready_utc=utc(),ownership_transfer='local -> cloud',
        local_training_finished=True,local_auto_resume=False,no_local_training_after_readiness=True,
        cloud_hard_deadline_utc='2026-10-26T10:00:00+00:00',local_session=binding(session_root/'session.json'),
        clock={k:session[k] for k in ('started_unix','started_monotonic','compute_deadline_unix','absolute_deadline_unix')},
        local_stop=controller.get('stopped_utc'),local_finalization=controller.get('finalized_utc'),
        shutdown_proof=dict(commit=archive_commit,path=str(REL/'bundle/shutdown-proof.json'),sha256=binding(bundle/'shutdown-proof.json')['sha256']),
        final_update=restoration['progress']['updates'],progress=restoration['progress'],
        cumulative_adam_steps=report['parent']['optimizer_steps']+report['optimizer_steps'],
        admitted_local_games=sum(b.get('completed_games',0) for b in journal['batches'] if b.get('checkpoint_verified')),
        local_ppo_updates=journal['new_updates'],heldout_verified_games=sum(b.get('completed_games',0) for b in journal['batches'] if b['mode']!='training'),
        checkpoint=saved,initial_checkpoint_dependency=initial,restore_evidence=dict(commit=archive_commit,path=str(REL/'bundle/restore-receipt.json'),sha256=binding(bundle/'restore-receipt.json')['sha256']),
        repository='tkorsi/heroes-rl-training-state',branch='codex/local365-teacher-results',archive_commit=archive_commit,
        archive=dict(path=str(REL/'bundle.tar.gz'),sha256=archive_bound['sha256'],bytes=archive_bound['bytes']),files=index,
        public_repository='tkorsi/heroes-rl',public_branch=public_branch,public_commit=public_commit,
        executable_source_commit=plan['source']['git_commit'],source=plan['source'],runtime=plan['runtime'],configuration=plan['configuration'],
        maps_by_seat={str(seat):index[f'maps/curriculum-3-{seat}.vmap'] for seat in (0,1)},
        next_unused_training_sequence=next_sequence,next_training_schedule=schedule(next_sequence,next_sequence%2),
        heldout_schedule_by_seat={str(seat):schedule(restoration['progress']['updates'],seat,True) for seat in (0,1)},
        distribution='Four-game same-seat batches alternating Red/Blue; seat-specific open8x8 initial states. Never one map substituted for both seats.',
        cloud_relocation='Requires distinct validated Linux adapter/runtime relocation. Preserve exact tensors, Adam/RNG/progress/pool and scientific configuration; Macintosh runtime paths are historical identities, not Linux executables.',
        prior_batch_archives=[b['private_preservation']['commit'] for b in journal['batches'] if b.get('private_preservation')],
        final_controller_archive=controller.get('private_final_preservation'))
    with (target/'ready.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    git('add','--',str(REL/'ready.json'));git('commit','-m','Transfer completed local curriculum ownership to cloud')
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results')
    receipt=dict(ready_commit=git('rev-parse','HEAD'),ready_path=str(REL/'ready.json'),ready_sha256=binding(target/'ready.json')['sha256'],archive_commit=archive_commit,final_update=manifest['final_update'],ownership_transfer='local -> cloud')
    (BASE/'campaign-v3/handoff-publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session-root',type=Path,required=True);p.add_argument('--finalization-proof',type=Path);a=p.parse_args();main(a.session_root,a.finalization_proof)
