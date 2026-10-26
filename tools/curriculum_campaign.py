"""Single-use frozen local campaign; ordinary verified PPO, owned deadlines and journals."""
import argparse
from collections import Counter
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.learner import curriculum_training as c
from experiments.learner.builtin_training_resume import binding, require
from experiments.learner.builtin_preservation import verify_pair
from experiments.learner.inference import source_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence

REPO = Path('/Users/orsi/Projects/Heroes-artifacts/private-local365-state')
AUTHOR = Path('/Users/orsi/.codex/worktrees/teacher-local365/Heroes')
PUBLISHED_PARENT = 'ce61fb66c73adef250e780ec205b9196f288f9e5'


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, path)


def remaining(session, reserve=0):
    return min(session['absolute_deadline_unix'] - time.time(),
               session['started_monotonic'] + 7200 - time.monotonic()) - reserve


def command_timeout(maximum=60):
    if not c.SESSION.exists(): return maximum
    value = remaining(json.loads(c.SESSION.read_text()), 3)
    if value <= 0: raise TimeoutError('Campaign deadline reached')
    return min(maximum, value)


def git(*args, binary=False):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
        text=not binary, timeout=command_timeout()).strip() if not binary else subprocess.check_output(
        ['git', '-C', str(REPO), *args], timeout=command_timeout())


def private_preflight():
    require(git('remote','get-url','origin') == 'https://github.com/tkorsi/heroes-rl-training-state.git'
            and git('branch','--show-current') == 'codex/local365-teacher-results'
            and git('status','--porcelain') == '', 'Private repository not ready')


def event(kind, **values):
    row = dict(timestamp_utc=utc(), event=kind, **values)
    # Authoring checkout is separate from the executable release checkout.
    require(ROOT != AUTHOR, 'Research writing must not dirty the frozen checkout')
    for path in (c.BASE/'campaign-v2/research-log.jsonl', AUTHOR/'validation/curriculum-v1-research-log.jsonl'):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f: f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())


def save(v):
    atomic(c.SESSION.parent/'campaign-report.json', v)
    s = json.loads(c.SESSION.read_text())
    admitted = [b for b in v['batches'] if b['mode']=='training' and b.get('checkpoint_verified')]
    status = dict(timestamp_utc=utc(), phase=v['status'], campaign_launched=True,
        actual_session_path=str(c.SESSION), release_path=s['release']['path'],
        controller_report_path=str(c.SESSION.parent/'controller-result.json'),
        campaign_report_path=str(c.SESSION.parent/'campaign-report.json'), source=s['source'],
        runtime=s['runtime'], diagnostic_games=64, diagnostic_training_samples=0,
        heldout_games=sum(b.get('completed_games',0) for b in v['batches'] if b['mode']!='training'),
        admitted_training_games=sum(b['completed_games'] for b in admitted), ppo_updates=v['new_updates'],
        latest_verified_checkpoint=v['latest_verified_parent'], latest_preserved_checkpoint=v['latest_private_parent'],
        launch_utc=utc(s['started_unix']), compute_deadline_utc=utc(s['compute_deadline_unix']),
        hard_deadline_utc=utc(s['absolute_deadline_unix']), validation_outcomes=s['validation_outcomes'],
        error=v.get('error'), final_analysis=v.get('analysis'), owned_cleanup_verified=v.get('owned_cleanup_verified'))
    for path in (c.BASE/'campaign-status.json', AUTHOR/'reports/local/teacher-curriculum-status.json'): atomic(path, status)


def publish(folder, label, checkpoint=False):
    private_preflight(); dest=REPO/'results/local-curriculum-update146'/label
    require(not dest.exists(), 'Private attempt already exists'); dest.mkdir()
    archive=dest/'evidence.tar.gz'
    with tarfile.open(archive,'x:gz',dereference=False) as t: t.add(folder,arcname=folder.name)
    files=links=0
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            original=folder.parent/m.name
            if m.isfile():
                require(hashlib.sha256(t.extractfile(m).read()).hexdigest()==binding(original)['sha256'], 'Archive file mismatch'); files+=1
            elif m.issym(): require(m.linkname==os.readlink(original),'Archive symlink mismatch'); links+=1
    receipt=dict(archive=binding(archive), verified_files=files, verified_symlinks=links, external_data_dereferenced=False)
    if checkpoint:
        plan=json.loads((folder/'plan.json').read_text()); report=json.loads((folder/'execution/report.json').read_text())
        backup=c.SESSION.parent/'checkpoints'/folder.name
        target=REPO/'checkpoints/local-curriculum-update146/campaign-v2'/folder.name
        require(not target.exists(),'Private checkpoint exists'); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(backup,target)
        receipt['checkpoint_restore']=verify_pair(target/'after-resume.pt',target/'after-policy.pt',plan,report)
        git('add','--',str(target.relative_to(REPO)))
    c.write_new(dest/'receipt.json',receipt)
    git('add','--',str(dest.relative_to(REPO))); git('commit','-m','Preserve curriculum '+label)
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results')
    commit=git('rev-parse','HEAD')
    require(hashlib.sha256(git('show',commit+':'+str(archive.relative_to(REPO)),binary=True)).hexdigest()==receipt['archive']['sha256'],'Private Git blob differs')
    return dict(commit=commit, **receipt)


def verified_report(report, returncode):
    return (returncode==0 and report.get('status')=='finished'
        and all(report.get(k) is True for k in ('worker_complete','supervisor_cleanup_verified','within_shared_budget'))
        and len(report.get('attempts',[]))==4
        and all(a.get('status')=='passed' and all(a.get(k) is True for k in
            ('completion_receipt','cleanup_verified','identity_verified','replay_verified')) for a in report['attempts']))


def run(v,parent,seq,seat,mode):
    final=mode=='final'; evaluate=mode!='training'
    row=dict(sequence=seq,seat=seat,mode=mode,status='preparing',parent=binding(parent),started_utc=utc())
    v['batches'].append(row); save(v); plan_path=None; proc=None; error=None
    event('campaign_batch_intent', **row, schedule=c.schedule(seq,seat,evaluate))
    try:
        plan_path=c.prepare(parent,seq,seat,evaluate,final); row.update(plan=binding(plan_path),status='running');save(v)
        s=c.session(); wall=s['absolute_deadline_unix']-(90 if final else 300)
        mono=s['started_monotonic']+(7110 if final else 6900)
        with (plan_path.parent/'controller.log').open('x') as log:
            proc=guarded_popen([sys.executable,'-m','experiments.learner.curriculum_training','execute',
                '--plan',str(plan_path),'--sha256',row['plan']['sha256']],
                deadline_monotonic=mono,deadline_wall=wall,evidence_path=plan_path.parent/'supervisor-deadline.jsonl',
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            rc=proc.wait(timeout=max(.1,min(wall-time.time(),mono-time.monotonic())))
        report_path=plan_path.parent/'execution/report.json';report=json.loads(report_path.read_text())
        row.update(report=binding(report_path),completed_games=sum(a['status']=='passed' for a in report['attempts']),
            outcomes=[a['outcomes'][('red','blue')[seat]] for a in report['attempts']],
            optimizer_steps=report['optimizer_steps'],training_samples=report['training_samples'])
        require(verified_report(report,rc),'Material batch failure: '+str(report.get('stop_reason') or report.get('supervisor_error')))
        if not evaluate:
            require(report['optimizer_updates']==1 and report['training_performed'] is True,'Missing complete PPO update')
            backup=c.SESSION.parent/'checkpoints'/plan_path.parent.name; plan=json.loads(plan_path.read_text())
            proof=verify_pair(backup/'after-resume.pt',backup/'after-policy.pt',plan,report)
            row['checkpoint_verified']=True
            v.update(latest_verified_parent=binding(plan_path),latest_progress=proof['progress'],new_updates=v['new_updates']+1)
        else: require(report['optimizer_updates']==report['optimizer_steps']==report['training_samples']==0,'Evaluation admitted')
        row['status']='verified';save(v)
    except Exception as exc:
        row.update(status='failed',error=f'{type(exc).__name__}: {exc}');error=exc
    finally:
        try:
            if proc is not None: close_owned_process(proc,timeout=min(2,command_timeout()))
            if plan_path:
                for p in plan_path.parent.rglob('*deadline*.jsonl'): cleanup_from_evidence(p,timeout=min(2,command_timeout()))
            row['owned_cleanup_verified']=True
        except Exception as exc: row['cleanup_error']=str(exc);error=error or exc
        save(v)
        if plan_path:
            try:
                row['private_preservation']=publish(plan_path.parent,'campaign-v2-'+plan_path.parent.name,
                    checkpoint=row.get('checkpoint_verified',False))
                if row.get('checkpoint_verified'):v['latest_private_parent']=binding(plan_path)
            except Exception as exc: row['preservation_error']=str(exc);error=error or exc
        row['status']='passed' if error is None else 'failed';row['finished_utc']=utc();save(v)
        event('campaign_batch_result', **row)
    if error: raise RuntimeError(row.get('error') or row.get('preservation_error') or row.get('cleanup_error')) from error
    return parent if evaluate else plan_path


def enough_for_batch(s,durations):
    left=min(s['compute_deadline_unix']-time.time(),s['started_monotonic']+6900-time.monotonic())
    return left > max(180,2*max(durations[-5:])+90)


def worker():
    s=c.session();require(os.getppid()==s['controller_pid'],'Wrong supervisor')
    with (c.SESSION.parent/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    parent=c.PARENT;v=dict(status='running',batches=[],new_updates=0,latest_verified_parent=binding(parent),
        latest_private_parent=binding(parent),latest_progress=dict(updates=146,rollouts=584,decisions=40641))
    save(v);durations=[60.];seq=0
    try:
        for seat in (0,1):run(v,parent,146,seat,'baseline')
        while enough_for_batch(c.session(),durations):
            started=time.monotonic();parent=run(v,parent,seq,seq%2,'training');seq+=1;durations.append(time.monotonic()-started)
            if v['new_updates']%5==0 and enough_for_batch(c.session(),durations):
                for seat in (0,1):run(v,parent,v['latest_progress']['updates'],seat,'periodic')
        for seat in (0,1):run(v,parent,v['latest_progress']['updates'],seat,'final')
        v['status']='finished_budget_reserve'
    except Exception as exc:v.update(status='stopped_failure',error=f'{type(exc).__name__}: {exc}')
    finally:v['stopped_utc']=utc();save(v)


def freeze():
    private_preflight()
    require(ROOT != AUTHOR and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'Freeze only a separate clean checkout')
    state=c.restore(c.PARENT)
    require(vars(state.progress)==dict(updates=146,rollouts=584,decisions=40641) and state.optimizer_steps==1475,'Wrong initial state')
    from validation.probes.validate_capabilities import capability_implementation_identity
    from validation.probes.validate_curriculum_policy import schedule as diagnostic_schedule
    runtime_bin='/Users/orsi/Projects/Heroes-artifacts/runtime/local-v8-update92-artifact-fix2/bin'
    runtime=runtime_identity(Path(runtime_bin))
    require(runtime==json.loads(c.PARENT.read_text())['runtime'],'Stable runtime changed')
    names=['tools/curriculum_campaign.py','tools/run_curriculum_study3.py',
           'configs/teacher-curriculum-v2.json','experiments/explorer/create_curriculum_scenario.py']
    frozen=dict(checkout=str(ROOT),source=source_identity(c.INTERFACE),implementation=c.pilot.implementation(),
        capability_implementation=capability_implementation_identity(),bound_files={n:binding(ROOT/n) for n in names},
        parent=state.bindings,configuration=c.configuration(),runtime_bin=runtime_bin,runtime=runtime,
        data='/Users/orsi/Projects/Heroes-artifacts/game-data',
        maps=[c.map_identity(c.BASE/f'maps/curriculum-3-{seat}.vmap',seat) for seat in (0,1)],
        diagnostic_schedule=[r for block in range(8) for seat in (0,1) for r in diagnostic_schedule(seat,3,block,3)],
        control_seeds=[1030001,1030002],heldout={str(seat):c.schedule(146,seat,True) for seat in (0,1)},
        distribution='Alternating four-game same-seat batches on open8x8 stage3; no difficulty transition in this campaign',
        success_criteria='Clean64game acceptance with natural wins and losses in each seat; capture/wait controls separately; descriptive fixed heldout win-rate change after PPO',
        hypothesis='The unchanged policy can sample wins on the easier start; terminal feedback may increase fixed heldout wins.',
        exclusions='All diagnostics, controls, previous studies and scripted rush excluded from PPO',
        acceptance_seconds=1200,campaign_seconds=7200,finalization_reserve_seconds=300,
        reward={'win':1,'loss':-1,'ordinary':0},created_utc=utc())
    path=c.SESSION.parent/'frozen.json';c.write_new(path,frozen)
    event('acceptance_and_campaign_preregistered',frozen=binding(path),protocol=frozen)
    print(json.dumps(binding(path)))


def prepare_release():
    private_preflight();frozen=c.check_frozen();require(not c.SESSION.exists(),'Consumed campaign cannot restart')
    counts=c.check_gate(c.BASE/'study3/study-report.json');state=c.restore(c.PARENT)
    require(state.bindings==frozen['parent'],'Parent drift')
    # Both durable and private checkpoint copies must restore before the campaign clock starts.
    plan=json.loads(c.PARENT.read_text());report=json.loads(Path(state.bindings['report']['path']).read_text())
    proofs=[]
    for folder in (Path('/Users/orsi/Projects/Heroes-artifacts/checkpoints')/c.PARENT.parent.name,
                   REPO/'checkpoints/local-v8-update92'/c.PARENT.parent.name):
        for kind in ('policy','resume'):require(binding(folder/f'after-{kind}.pt')['sha256']==state.bindings[kind]['sha256'],'Backup changed')
        proofs.append(verify_pair(folder/'after-resume.pt',folder/'after-policy.pt',plan,report))
    release=dict(frozen,scope=c.SCOPE,authorization='Renewed user authorization: clean acceptance then one two-hour local curriculum PPO session; no automatic restart.',
        gate=binding(c.BASE/'study3/study-report.json'),validation_outcomes=counts,independent_parent_restores=proofs,
        whole_wall_seconds=7200,finalization_reserve_seconds=300,diagnostic_admission=False)
    path=c.SESSION.parent/'release.json';c.write_new(path,release)
    event('campaign_release_ready',release=binding(path),gate_counts=counts)
    print(json.dumps(binding(path)))


def analysis(v):
    result=dict(schema='heroes-curriculum-campaign-v2-analysis',timestamp_utc=utc(),status=v['status'],
        new_updates=v['new_updates'],progress=v['latest_progress'],training_outcomes_by_seat={},heldout_by_seat={},
        scope='Open8x8 easier-start distribution, not original full-map performance',
        interpretation='Four fixed games per seat are descriptive paired checks, not evidence of general strategic improvement.')
    for seat in (0,1):
        color=('red','blue')[seat]
        rows=[b for b in v['batches'] if b['seat']==seat]
        result['training_outcomes_by_seat'][color]=dict(Counter(o for b in rows if b['mode']=='training' and b.get('checkpoint_verified') for o in b['outcomes']))
        result['heldout_by_seat'][color]={mode:[dict(parent=b['parent'],outcomes=b.get('outcomes'),status=b['status']) for b in rows if b['mode']==mode] for mode in ('baseline','final')}
    return result


def launch():
    private_preflight();path=c.SESSION.parent/'release.json';r=json.loads(path.read_text());frozen=c.check_frozen()
    require(r['source']==frozen['source'] and r['parent']==c.restore(c.PARENT).bindings,'Release source/parent drift')
    require(binding(r['gate']['path'])==r['gate'] and json.loads(json.dumps(c.check_gate(r['gate']['path'])))==r['validation_outcomes'],'Gate drift')
    require(not c.SESSION.exists(),'One campaign only')
    with c.SESSION.with_suffix('.lock').open('x') as f:f.write(str(os.getpid()))
    now=time.time();mono=time.monotonic();s=dict(scope=c.SCOPE,source=r['source'],runtime=r['runtime'],release=binding(path),
        started_unix=now,started_monotonic=mono,compute_deadline_unix=now+6900,absolute_deadline_unix=now+7200,
        compute_deadline_monotonic=mono+6900,hard_deadline_monotonic=mono+7200,controller_pid=os.getpid(),validation_outcomes=r['validation_outcomes'])
    c.write_new(c.SESSION,s);event('campaign_launched',session=binding(c.SESSION),launch_utc=utc(now),compute_utc=utc(now+6900),hard_utc=utc(now+7200))
    proc=None;result=dict(session=binding(c.SESSION))
    try:
        with (c.SESSION.parent/'controller.log').open('x') as log:
            proc=guarded_popen([sys.executable,str(Path(__file__).resolve()),'worker'],deadline_monotonic=mono+7110,deadline_wall=now+7110,
                evidence_path=c.SESSION.parent/'controller-deadline.jsonl',cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            result['returncode']=proc.wait(timeout=max(.1,remaining(s,90)))
    except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        errors=[]
        try:
            if proc is not None:close_owned_process(proc,timeout=2)
            for p in (c.SESSION.parent/'batches').rglob('*deadline*.jsonl'):cleanup_from_evidence(p,timeout=1)
        except Exception as exc:errors.append(str(exc))
        result.update(cleanup_errors=errors,owned_cleanup_verified=not errors,stopped_utc=utc(),within_budget=remaining(s)>=0)
        v=json.loads((c.SESSION.parent/'campaign-report.json').read_text())
        if result.get('error') or errors:v.update(status='stopped_failure',error=result.get('error') or str(errors))
        result['latest_restore']=c.restore(v['latest_verified_parent']['path']).bindings
        v['owned_cleanup_verified']=not errors
        report=analysis(v);atomic(c.SESSION.parent/'analysis.json',report);v['analysis']=binding(c.SESSION.parent/'analysis.json');save(v)
        atomic(c.SESSION.parent/'controller-result.json',result)
        event('campaign_stopped',result=result,analysis=report)
        folder=c.BASE/'campaign-v2-final-evidence';folder.mkdir(exist_ok=False)
        for p in c.SESSION.parent.iterdir():
            if p.is_file():shutil.copy2(p,folder/p.name)
        try:result['private_final_preservation']=publish(folder,'campaign-v2-final-session')
        except Exception as exc:result['final_preservation_error']=str(exc)
        result.update(finalized_utc=utc(),within_budget=remaining(s)>=0)
        atomic(c.SESSION.parent/'controller-result.json',result)
        atomic(c.BASE/'campaign-v2-final-preservation.json',result)
        event('campaign_finalized',result=result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare-release','launch','worker']);a=p.parse_args()
    {'freeze':freeze,'prepare-release':prepare_release,'launch':launch,'worker':worker}[a.mode]()
