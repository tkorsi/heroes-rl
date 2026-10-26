"""Owned one-shot finalization watcher; never launches or resumes training."""
import argparse
from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.learner.builtin_training_resume import binding,require
from experiments.headless.process_deadline import guarded_popen,close_owned_process,cleanup_from_evidence
from tools.finalize_curriculum_handoff import validate_final, PRIVATE, REL
BASE=Path('/Users/orsi/Projects/Heroes-artifacts/local-curriculum-update146')
OUT=BASE/'automatic-finalization-v3'
WINDOW=BASE/'campaign-v3/window.json'
HELPER=ROOT/'tools/finalize_curriculum_handoff.py'


def utc():return datetime.now(timezone.utc).isoformat()

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def left(clock):return min(clock['absolute_deadline_unix']-time.time(),clock['started_monotonic']+7200-time.monotonic())

def command(args,clock):
    require(left(clock)>5,'Original finalization deadline exhausted')
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=min(90,left(clock)-3)).strip()

def code_check(manifest):
    for name,value in manifest['code'].items():require(binding(ROOT/name)==value,'Finalizer source drift')
    require(binding(WINDOW)==manifest['window'],'Original campaign clock drift')

def candidate(clock):
    status=json.loads((BASE/'campaign-status.json').read_text());path=status.get('actual_session_path')
    if not path:return None
    root=Path(path).parent
    require(root.is_relative_to(BASE/'campaign-v3'),'Foreign finalization session')
    result_path=root/'controller-result.json'
    if not result_path.exists():return None
    result=json.loads(result_path.read_text())
    if not result.get('finalized_utc') or not result.get('private_final_preservation'):return None
    session=json.loads((root/'session.json').read_text());journal=json.loads((root/'campaign-report.json').read_text())
    require(session['window']==binding(WINDOW),'Recovery uses another clock')
    process=subprocess.run(['ps','-p',str(session['controller_pid']),'-o','command='],capture_output=True,text=True,timeout=5)
    if 'curriculum_campaign.py' in process.stdout:return None
    if left(clock)>600:return None
    validate_final(session,result,journal,time.time(),time.monotonic())
    return root

def publish_results(root,clock):
    journal=json.loads((root/'campaign-report.json').read_text());result=json.loads((root/'controller-result.json').read_text())
    for path in (BASE/'campaign-v3').rglob('*deadline*.jsonl'):cleanup_from_evidence(path,timeout=2)
    analysis=json.loads((root/'analysis.json').read_text())
    analysis.update(session=binding(root/'session.json'),controller=binding(root/'controller-result.json'),
        private_final_preservation=result['private_final_preservation'],owned_local_training_shutdown_verified=True)
    out=ROOT/'validation/curriculum-v3-final-2026-10-26.json';write(out,analysis)
    message=(f"\n\n## Completed local campaign\n\nLocal campaign status: `{journal['status']}`. It completed {journal['new_updates']} PPO updates, "
             f"advancing update146 to{journal['latest_progress']['updates']}. "
             f"Cumulative progress is {journal['latest_progress']['rollouts']} admitted rollouts and "
             f"{journal['latest_progress']['decisions']} decisions. "
             "The final controller archive and exact checkpoint restore are verified; all owned local training processes are stopped. "
             "Per-seat training outcomes and baseline/final held-out results are recorded in "
             "[the final analysis](../../validation/curriculum-v3-final-2026-10-26.json). "
             "These are easier8x8 curriculum results, not original-map performance. "
             "The separate one-shot watcher now performs the Git-only cloud handoff.\n")
    with (ROOT/'docs/research/curriculum-v1.md').open('a') as f:f.write(message)
    with (ROOT/'README.md').open('a') as f:f.write(f"\n\nThe local curriculum campaign is complete at update{journal['latest_progress']['updates']} "
        f"after {journal['new_updates']} new PPO updates. Owned local training shutdown and final preservation are verified. "
        "See [final results](validation/curriculum-v3-final-2026-10-26.json) and the timestamped local handoff status.\n")
    event=dict(timestamp_utc=utc(),event='automatic_final_scientific_publication',analysis=analysis)
    with (ROOT/'validation/curriculum-v1-research-log.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    command(['git','add','README.md','docs/research/curriculum-v1.md','validation/curriculum-v1-research-log.jsonl',str(out.relative_to(ROOT))],clock)
    command(['git','add','-f','reports/local/teacher-curriculum-status.json'],clock)
    command(['git','commit','-m','Publish completed local curriculum results and verified shutdown'],clock)
    command(['git','push','origin','HEAD:refs/heads/codex/teacher-curriculum-v1'],clock)

def worker():
    manifest=json.loads((OUT/'launch.json').read_text());clock=json.loads(WINDOW.read_text())
    require(os.getppid()==manifest['supervisor_pid'],'Wrong finalizer supervisor')
    with (OUT/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    code_check(manifest);write(OUT/'execution.json',dict(phase='waiting_for_controller_exit',started_utc=utc()))
    root=None
    while left(clock)>5:
        code_check(manifest);root=candidate(clock)
        if root is not None:break
        time.sleep(min(5,max(.1,left(clock)-3)))
    require(root is not None,'No complete shutdown/restore evidence before original deadline')
    publish_results(root,clock);code_check(manifest)
    proof=dict(phase='handoff_preconditions_verified',timestamp_utc=utc(),launch=binding(OUT/'launch.json'),
        latest_session=binding(root/'session.json'),command=[sys.executable,str(HELPER),'--session-root',str(root)],
        duplicate_exclusion='Exclusive launch directory, worker.lock and unique ready.json; no training entry point',
        final_scientific_public_commit=command(['git','rev-parse','HEAD'],clock))
    write(OUT/'prepublication.json',proof)
    answer=command([sys.executable,str(HELPER),'--session-root',str(root),'--finalization-proof',str(OUT/'prepublication.json')],clock)
    receipt=json.loads(answer.splitlines()[-1]);write(OUT/'execution.json',dict(phase='ownership_handoff_completed',finished_utc=utc(),handoff=receipt,launch=binding(OUT/'launch.json')))
    # Publish the immutable execution receipt after readiness, without changing ready.json.
    import shutil
    for name in ('launch.json','execution.json','prepublication.json'):
        shutil.copy2(OUT/name,PRIVATE/REL/('finalizer-'+name))
    command(['git','-C',str(PRIVATE),'add','--',str(REL/'finalizer-launch.json'),str(REL/'finalizer-execution.json'),str(REL/'finalizer-prepublication.json')],clock)
    command(['git','-C',str(PRIVATE),'commit','-m','Record automatic finalization command and execution receipt'],clock)
    command(['git','-C',str(PRIVATE),'push','origin','HEAD:refs/heads/codex/local365-teacher-results'],clock)
    receipt['finalizer_receipt_commit']=command(['git','-C',str(PRIVATE),'rev-parse','HEAD'],clock)
    status=json.loads((BASE/'campaign-status.json').read_text());status.update(timestamp_utc=utc(),phase='local_complete_ownership_transferred_to_cloud',handoff=receipt,
        local_auto_resume=False,no_local_training_after_readiness=True,automatic_finalization_receipt=binding(OUT/'execution.json'))
    for path in (BASE/'campaign-status.json',ROOT/'reports/local/teacher-curriculum-status.json'):write(path,status)
    with (ROOT/'validation/curriculum-v1-research-log.jsonl').open('a') as f:f.write(json.dumps(dict(event='automatic_ownership_handoff_published',timestamp_utc=utc(),receipt=receipt))+'\n')
    command(['git','add','validation/curriculum-v1-research-log.jsonl'],clock);command(['git','add','-f','reports/local/teacher-curriculum-status.json'],clock)
    command(['git','commit','-m','Publish completed automatic local-to-cloud ownership handoff'],clock)
    command(['git','push','origin','HEAD:refs/heads/codex/teacher-curriculum-v1'],clock)

def launch():
    clock=json.loads(WINDOW.read_text());require(left(clock)>300,'Finalizer needs original-window reserve')
    OUT.mkdir(exist_ok=False)
    names=['tools/watch_curriculum_finalization.py','tools/finalize_curriculum_handoff.py',
        'experiments/learner/builtin_preservation.py','experiments/learner/builtin_training_resume.py',
        'experiments/learner/checkpoints.py','experiments/learner/contextual_model.py',
        'experiments/learner/contextual.py','experiments/headless/process_deadline.py']
    manifest=dict(created_utc=utc(),supervisor_pid=os.getpid(),window=binding(WINDOW),
        command=[sys.executable,str(Path(__file__).resolve()),'worker'],code={n:binding(ROOT/n) for n in names},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline_unix=clock['absolute_deadline_unix'],deadline_monotonic=clock['started_monotonic']+7200,
        duplicate_exclusion='Exclusive automatic-finalization-v3 directory and worker.lock',training_allowed=False)
    write(OUT/'launch.json',manifest);process=None;result={}
    try:
        with (OUT/'worker.log').open('x') as log:
            process=guarded_popen(manifest['command'],deadline_monotonic=manifest['deadline_monotonic']-2,
                deadline_wall=manifest['deadline_unix']-2,evidence_path=OUT/'watch-deadline.jsonl',cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            result['returncode']=process.wait(timeout=max(.1,left(clock)-1))
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
        if process is not None:close_owned_process(process,timeout=2)
        result.update(finished_utc=utc(),owned_finalizer_shutdown_verified=True,within_original_window=left(clock)>=0)
        write(OUT/'supervisor-result.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','worker']);a=p.parse_args()
    {'launch':launch,'worker':worker}[a.mode]()
