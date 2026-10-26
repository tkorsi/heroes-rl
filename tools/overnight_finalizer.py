"""Detached preservation and optional final fixed pair; never launches PPO.

Waits for the existing exclusive controller to finish. All gameplay uses the
unchanged released evaluator, original session and absolute compute deadline.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
AUTHOR=Path('/workspace/heroes-overnight')
RELEASE=Path('/workspace/releases/teacher-overnight-30629a84bc30')
COMPUTE=1793008500.;HARD=1793008800.


def require(condition,message):
    if not condition:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.finalizer-tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sufficient_final_reserve(journal,now):
    durations=[]
    for row in journal['batches']:
        if row['mode']!='training' and row.get('status')=='passed':
            durations.append((datetime.fromisoformat(row['finished_utc'])-datetime.fromisoformat(row['started_utc'])).total_seconds())
    return journal['status']=='finished_budget_reserve' and COMPUTE-now>max(120,2*max(durations,default=30)+60)


def finish(report_path,evaluate):
    """Executed with the original release first on sys.path, no mutable modules."""
    sys.path.insert(0,str(RELEASE))
    from tools import overnight_curriculum_campaign as c
    from experiments.headless.process_deadline import cleanup_from_evidence
    require(Path(c.__file__).resolve().is_relative_to(RELEASE),'Wrong executable evaluator')
    require(c.c.SESSION.parent==report_path.parent,'Original session required')
    journal=json.loads(report_path.read_text())
    old=json.loads((report_path.parent/'controller-result.json').read_text())
    require(old['owned_cleanup_verified'] is True,'Controller cleanup required')
    for path in (report_path.parent/'batches').rglob('*deadline*.jsonl'):cleanup_from_evidence(path,timeout=1)
    state=c.c.restore(journal['latest_verified_parent']['path'])
    proofs=c.restore_final_copies(journal,state)
    receipt=dict(started_utc=utc(),original_controller_result=c.binding(report_path.parent/'controller-result.json'),
        original_journal=c.binding(report_path),before_restore=proofs,final_evaluation_requested=evaluate,
        compute_deadline_unix=COMPUTE,hard_deadline_unix=HARD,clock_reset=False,training_samples_added=0,optimizer_steps_added=0)
    shutil.copy2(report_path,report_path.parent/'before-finalizer-campaign-report.json')
    if evaluate and sufficient_final_reserve(journal,time.time()):
        # The original evaluator allocates final-UPDATE-seatN paths, verifies the
        # exact source/runtime/parent/maps, and rejects all PPO in evaluation.
        try:
            for seat in (0,1):
                require(COMPUTE-time.time()>45,'Insufficient remaining final evaluation reserve')
                c.run(journal,Path(journal['latest_verified_parent']['path']),journal['latest_progress']['updates'],seat,'final')
            receipt['final_pair_completed']=True
        except Exception as exc:
            receipt.update(final_pair_completed=False,final_pair_error=f'{type(exc).__name__}: {exc}')
            journal['status']='stopped_final_evaluation_failure';c.save(journal)
    else:receipt.update(final_pair_completed=False,final_pair_omitted_reason='Insufficient measured safe reserve or campaign not cleanly finished')
    restored=c.c.restore(journal['latest_verified_parent']['path'])
    require(restored.bindings==state.bindings,'Final evaluation changed learned state')
    receipt['after_restore']=c.restore_final_copies(journal,restored)
    cleanup_errors=[]
    for path in (report_path.parent/'batches').rglob('*deadline*.jsonl'):
        try:cleanup_from_evidence(path,timeout=1)
        except Exception as exc:cleanup_errors.append(str(exc))
    analysis=c.analysis(journal)
    analysis.update(finalizer_utc=utc(),final_checkpoint=restored.bindings,
        absolute_compute_cutoff_utc='2026-10-26T09:55:00Z',absolute_hard_stop_utc='2026-10-26T10:00:00Z',
        original_window=json.loads(c.c.WINDOW.read_text()),no_resume_authorized_after_hard_stop=True)
    c.atomic(report_path.parent/'final-scientific-analysis.json',analysis)
    c.atomic(AUTHOR/'validation/overnight-cloud-final-2026-10-26.json',analysis)
    receipt.update(finished_utc=utc(),analysis=c.binding(report_path.parent/'final-scientific-analysis.json'),
        latest_restore=restored.bindings,owned_cleanup_verified=not cleanup_errors,cleanup_errors=cleanup_errors)
    c.atomic(report_path.parent/'finalizer-restore-receipt.json',receipt)
    folder=report_path.parent/'autonomous-final-preservation';folder.mkdir(exist_ok=False)
    for name in ('campaign-report.json','controller-result.json','before-finalizer-campaign-report.json',
                 'final-scientific-analysis.json','finalizer-restore-receipt.json','session.json','research-log.jsonl'):
        path=report_path.parent/name
        if path.exists():shutil.copy2(path,folder/name)
    # Small orchestration/failure receipts only; all game Data stays outside.
    orchestration=folder/'orchestration';orchestration.mkdir()
    for path in BASE.iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl','.log') and path.stat().st_size<20_000_000:
            shutil.copy2(path,orchestration/path.name)
    receipt['private_final_preservation']=c.publish(folder,'autonomous-final-'+report_path.parent.name)
    remote=c.git('ls-remote','origin','refs/heads/codex/teacher-overnight-20261026').split()[0]
    require(remote==receipt['private_final_preservation']['commit'],'Final private remote acknowledgement differs')
    receipt['private_remote_acknowledged']=True
    c.atomic(report_path.parent/'autonomous-final-result.json',receipt)
    require(not cleanup_errors,'Final owned cleanup failed; evidence preserved')
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finish-report',type=Path);parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.finish_report:
        require(args.finish_report.is_relative_to(BASE/'campaign-v1'),'Foreign final journal')
        return finish(args.finish_report,args.evaluate)
    final_lock=(BASE/'finalizer.lock').open('a');fcntl.flock(final_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(pid=os.getpid(),command=[sys.executable,*sys.argv],started_utc=utc(),phase='waiting_existing_controller',
        compute_deadline_utc='2026-10-26T09:55:00Z',hard_deadline_utc='2026-10-26T10:00:00Z',no_ppo_launch_path=True)
    atomic(BASE/'finalizer-status.json',status)
    main_lock=(BASE/'autocontinue.lock').open('a')
    while time.time()<HARD-90:
        try:fcntl.flock(main_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            time.sleep(min(10,max(.05,HARD-90-time.time())));continue
        live=json.loads((AUTHOR/'reports/cloud/teacher-overnight-status.json').read_text())
        report=Path(live['campaign_report_path'])
        require(report.is_relative_to(BASE/'campaign-v1'),'Foreign active journal')
        journal=json.loads(report.read_text())
        require(journal['status']!='running','Controller vanished without terminal journal; no final gameplay')
        status.update(phase='preserving_last_complete_state',report=str(report),timestamp_utc=utc())
        atomic(BASE/'finalizer-status.json',status)
        env={**os.environ,'HEROES_CURRICULUM_SESSION':str(report.parent/'session.json'),
             'HEROES_OVERNIGHT_OWNER':'teacher-20261026','HEROES_OVERNIGHT_ROLE':'preservation'}
        command=[sys.executable,str(Path(__file__).resolve()),'--finish-report',str(report)]
        if sufficient_final_reserve(journal,time.time()):command.append('--evaluate')
        with (BASE/'finalizer-work.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=RELEASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=max(.1,HARD-time.time()-65))
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=5);code=-15
        status.update(phase='stopped',returncode=code,stopped_utc=utc(),restart_authorized=False)
        result=report.parent/'autonomous-final-result.json'
        if result.exists():status['preservation']=json.loads(result.read_text())
        atomic(BASE/'finalizer-status.json',status)
        # Public log/status publication is separate from the immutable release.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from tools import overnight_autocontinue as owner
        owner.event('autonomous_finalizer_stopped',result=status)
        live=json.loads((AUTHOR/'reports/cloud/teacher-overnight-status.json').read_text())
        live.update(timestamp_utc=utc(),phase='stopped_preserved' if code==0 else 'stopped_finalizer_error',
            finalizer=status,restart_authorized=False)
        if result.exists():live['final_scientific_analysis']='validation/overnight-cloud-final-2026-10-26.json'
        owner.atomic(AUTHOR/'reports/cloud/teacher-overnight-status.json',live)
        final=AUTHOR/'validation/overnight-cloud-final-2026-10-26.json'
        if final.exists():owner.git(AUTHOR,'add','-f','--',str(final.relative_to(AUTHOR)))
        owner.public_push('Publish autonomous overnight final analysis and verified preservation')
        return
    status.update(phase='hard_stop_reserve_exhausted',stopped_utc=utc(),restart_authorized=False)
    atomic(BASE/'finalizer-status.json',status)


if __name__=='__main__':main()
