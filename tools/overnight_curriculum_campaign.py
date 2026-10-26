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
from experiments.learner import overnight_training as c
from experiments.learner.builtin_training_resume import binding, require
from experiments.learner.builtin_preservation import verify_pair
from experiments.learner.inference import source_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence

REPO = Path('/workspace/heroes-overnight-private')
AUTHOR = Path('/workspace/heroes-overnight')
PUBLISHED_PARENT = None


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, path)


def remaining(session, reserve=0):
    return min(session['absolute_deadline_unix'] - time.time(),
               session['started_monotonic'] + (session['absolute_deadline_unix']-session['started_unix']) - time.monotonic()) - reserve


def command_timeout(maximum=60):
    if not c.WINDOW.exists(): return maximum
    value = remaining(json.loads(c.WINDOW.read_text()), 3)
    if value <= 0: raise TimeoutError('Campaign deadline reached')
    return min(maximum, value)


def git(*args, binary=False):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
        text=not binary, timeout=command_timeout()).strip() if not binary else subprocess.check_output(
        ['git', '-C', str(REPO), *args], timeout=command_timeout())


def private_preflight():
    require(git('remote','get-url','origin') == 'https://github.com/tkorsi/heroes-rl-training-state.git'
            and git('branch','--show-current') == 'codex/teacher-overnight-20261026'
            and git('status','--porcelain') == '', 'Private repository not ready')


def event(kind, **values):
    row = dict(timestamp_utc=utc(), event=kind, **values)
    # Authoring checkout is separate from the executable release checkout.
    require(ROOT != AUTHOR, 'Research writing must not dirty the frozen checkout')
    if kind in ('campaign_batch_intent','campaign_batch_result','campaign_stopped','campaign_launched'):
        with (AUTHOR/'docs/research/overnight-cloud-2026-10-26.md').open('a') as f:
            f.write('\n- '+row['timestamp_utc']+' '+kind+': '+json.dumps({k:v for k,v in values.items() if k in ('sequence','seat','mode','status','error','result')})+'\n')
    for path in (c.SESSION.parent/'research-log.jsonl', AUTHOR/'validation/overnight-cloud-2026-10-26.jsonl'):
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
        latest_verified_checkpoint=v['latest_verified_parent'], latest_preserved_checkpoint=v['latest_private_parent'],ownership='cloud',remaining_compute_seconds=max(0,c.COMPUTE-time.time()),
        launch_utc=utc(s['started_unix']), compute_deadline_utc=utc(s['compute_deadline_unix']),
        hard_deadline_utc=utc(s['absolute_deadline_unix']), validation_outcomes=s['validation_outcomes'],
        error=v.get('error'), final_analysis=v.get('analysis'), owned_cleanup_verified=v.get('owned_cleanup_verified'))
    status['cloud_training_started'] = any(b['mode']=='training' and b.get('plan') for b in v['batches'])
    status['new_admitted_games'] = status['admitted_training_games']
    status['new_ppo_updates'] = v['new_updates']
    status['latest_complete_progress'] = v['latest_progress']
    status['next_periodic_update'] = c.latest_periodic_update(v['latest_progress']['updates'])+c.PERIODIC_EVERY
    status['current_activity'] = {k:v['batches'][-1].get(k) for k in ('mode','sequence','seat','status','started_utc','finished_utc')} if v['batches'] else {'mode':'initializing'}
    status['evaluation_natural_outcomes'] = dict(Counter(o for b in v['batches'] if b['mode']!='training' and b.get('status')=='passed' for o in b.get('outcomes',[]) if o in ('win','loss')))
    status['natural_outcomes'] = dict(Counter(o for b in admitted for o in b.get('outcomes',[]) if o in ('win','loss')))
    status['cutoffs'] = sum(o is None for b in admitted for o in b.get('outcomes',[]))
    status['failed_batches'] = sum(b.get('status')=='failed' for b in v['batches'])
    for path in (c.BASE/'campaign-status.json', AUTHOR/'reports/cloud/teacher-overnight-status.json'): atomic(path, status)


def publish(folder, label, checkpoint=False):
    private_preflight(); dest=REPO/'results/overnight-20261026'/label
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
        target=REPO/'checkpoints/overnight-20261026/campaign-v1'/c.SESSION.parent.name/folder.name
        require(not target.exists(),'Private checkpoint exists'); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(backup,target)
        receipt['checkpoint_restore']=verify_pair(target/'after-resume.pt',target/'after-policy.pt',plan,report)
        git('add','--',str(target.relative_to(REPO)))
    c.write_new(dest/'receipt.json',receipt)
    git('add','--',str(dest.relative_to(REPO))); git('commit','-m','Preserve curriculum '+label)
    git('push','origin','HEAD:refs/heads/codex/teacher-overnight-20261026')
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
        s=c.session(); wall=s['compute_deadline_unix']
        mono=s['compute_deadline_monotonic']
        with (plan_path.parent/'controller.log').open('x') as log:
            proc=guarded_popen([sys.executable,'-m','experiments.learner.overnight_training','execute',
                '--plan',str(plan_path),'--sha256',row['plan']['sha256']],
                deadline_monotonic=mono,deadline_wall=wall,evidence_path=plan_path.parent/'supervisor-deadline.jsonl',
                cwd=ROOT,env={**os.environ,'HEROES_OVERNIGHT_OWNER':'teacher-20261026','HEROES_OVERNIGHT_ROLE':'compute'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
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
                row['private_preservation']=publish(plan_path.parent,'campaign-v1-'+c.SESSION.parent.name+'-'+plan_path.parent.name,
                    checkpoint=row.get('checkpoint_verified',False))
                if row.get('checkpoint_verified'):v['latest_private_parent']=binding(plan_path)
            except Exception as exc: row['preservation_error']=str(exc);error=error or exc
        row['status']='passed' if error is None else 'failed';row['finished_utc']=utc();save(v)
        event('campaign_batch_result', **row)
    if error: raise RuntimeError(row.get('error') or row.get('preservation_error') or row.get('cleanup_error')) from error
    return parent if evaluate else plan_path


def enough_for_batch(s,durations):
    left=min(s['compute_deadline_unix']-time.time(),s['compute_deadline_monotonic']-time.monotonic())
    return left > max(180,2*max(durations[-5:])+90)


def worker():
    s=c.session();require(os.getppid()==s['controller_pid'],'Wrong supervisor')
    with (c.SESSION.parent/'worker.lock').open('x') as f:f.write(str(os.getpid()))
    parent=c.PARENT;v=dict(status='running',batches=[],new_updates=0,latest_verified_parent=binding(parent),
        latest_private_parent=binding(parent),latest_progress=c.context()['progress'])
    if s.get('recovery_from'):
        require(binding(s['recovery_from']['path'])==s['recovery_from'],'Recovery journal drift')
        v=json.loads(Path(s['recovery_from']['path']).read_text());v.update(status='running',error=None)
        parent=Path(v['latest_verified_parent']['path'])
        c.restore(parent)
    gate=json.loads((c.BASE/'study4/study-report.json').read_text())
    gate_durations=[r['finished_unix']-r['started_unix'] for r in gate['attempts'] if r['kind']=='policy']
    save(v);durations=[max(60.,max(gate_durations))];seq=1+max((b['sequence'] for b in v['batches'] if b['mode']=='training'),default=c.context()['next_unused_training_sequence']-1)
    try:
        for seat in (0,1):
            if not any(b['mode']=='baseline' and b['seat']==seat and b['status']=='passed' for b in v['batches']):
                require(enough_for_batch(c.session(),durations),'Insufficient measured reserve for frozen baseline')
                start=time.monotonic();run(v,c.PARENT,c.context()['progress']['updates'],seat,'baseline');durations.append(time.monotonic()-start)
        while enough_for_batch(c.session(),durations):
            started=time.monotonic();parent=run(v,parent,seq,seq%2,'training');seq+=1;durations.append(time.monotonic()-started)
            if (v['latest_progress']['updates']-c.PERIODIC_ORIGIN)%c.PERIODIC_EVERY==0:
                require(enough_for_batch(c.session(),durations),'Periodic evaluation due but insufficient safe time')
                for seat in (0,1):
                    require(enough_for_batch(c.session(),durations),'Insufficient measured reserve for due paired evaluation')
                    run(v,parent,v['latest_progress']['updates'],seat,'periodic')
        if enough_for_batch(c.session(),durations):
            for seat in (0,1):
                if not enough_for_batch(c.session(),durations):break
                run(v,parent,v['latest_progress']['updates'],seat,'final')
        v['status']='finished_budget_reserve'
    except Exception as exc:v.update(status='stopped_failure',error=f'{type(exc).__name__}: {exc}')
    finally:v['stopped_utc']=utc();save(v)


def check_engine_checkout():
    """Reject incomplete release preparation before any control or policy game."""
    lock=json.loads((ROOT/'dependencies/vcmi.json').read_text())
    repo=ROOT/lock['directory']
    require(repo.is_dir() and not repo.is_symlink(), 'Generated VCMI dependency missing: run tools/prepare_vcmi.py before freezing')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()
    require(head==lock['commit'], 'Generated VCMI HEAD differs from pinned engine')
    checked=subprocess.run([sys.executable,str(ROOT/'tools/prepare_vcmi.py'),'--check'],
        cwd=ROOT,capture_output=True,text=True,timeout=60,check=True)
    return dict(engine_revision=head,check=checked.stdout.strip())


def freeze():
    engine_preflight=check_engine_checkout()
    private_preflight()
    require(ROOT != AUTHOR and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'Freeze only a separate clean checkout')
    from tools.overnight_handoff import audit_materialized
    handoff=c.context()['handoff_verification'];require(binding(handoff['path'])==handoff,'Handoff verification bytes changed')
    handoff_audit=audit_materialized(handoff['path'],Path('/workspace/heroes-rl-training-state'))
    state=c.restore(c.PARENT)
    require(c.context()['ownership_transfer_verified'] is True,'Final local ownership transfer required')
    require([c.map_identity(c.BASE/f'maps/curriculum-3-{seat}.vmap',seat)['sha256'] for seat in (0,1)]==c.context()['map_sha256_by_seat'],'Final handed-off map bytes differ')
    require(vars(state.progress)==c.context()['progress'] and state.optimizer_steps==c.context()['optimizer_steps'],'Wrong final handoff state')
    from validation.probes.validate_capabilities import capability_implementation_identity
    from validation.probes.validate_overnight_policy import schedule as diagnostic_schedule
    runtime_bin='/workspace/Heroes-artifacts/runtime/overnight-20261026/bin'
    runtime=runtime_identity(Path(runtime_bin))
    ready=json.loads((c.BASE/'runtime-ready.json').read_text())
    require(binding(ready['check']['path'])==ready['check'],'Runtime build receipt differs')
    build=json.loads(Path(ready['check']['path']).read_text());require(build['passed'] is True,'Failed Linux runtime check')
    all_files=c.runtime_files(runtime_bin)
    for name,expected in build['artifacts'].items():
        require({k:all_files[name][k] for k in ('sha256','bytes')}==expected,'Executable differs from built/checked Linux runtime')
    require(runtime==c.context()['linux_runtime'],'Unpinned Linux runtime')
    require(json.loads(c.PARENT.read_text())['runtime']==c.context()['historical_runtime'],'Historical runtime changed')
    for module in ('Nullkiller','BattleAI','ExplorerRL'):
        require((Path(runtime_bin)/'AI'/f'lib{module}.so').is_file(),'Required AI module missing')
    for seat in (0,1):
        c.VcmiLauncher(Path(runtime_bin)/'vcmiclient',Path('/workspace/Heroes-artifacts/game-data'),c.BASE/f'maps/curriculum-3-{seat}.vmap').preflight()
    names=['tools/overnight_curriculum_campaign.py','tools/run_overnight_acceptance.py','tools/overnight_cloud_supervisor.py','configs/teacher-overnight-linux-v1.json',
           'configs/teacher-curriculum-v3.json','experiments/explorer/create_curriculum_scenario.py']
    frozen=dict(handoff_dependency_audit=handoff_audit,checkout=str(ROOT),engine_preflight=engine_preflight,dependency=c.dependency_identity(),source=source_identity(c.INTERFACE),implementation=c.pilot.implementation(),
        capability_implementation=capability_implementation_identity(),bound_files={n:binding(ROOT/n) for n in names},
        parent=state.bindings,configuration=c.configuration(),runtime_bin=runtime_bin,runtime=runtime,runtime_files=all_files,runtime_report=ready['check'],
        data='/workspace/Heroes-artifacts/game-data',
        maps=[c.map_identity(c.BASE/f'maps/curriculum-3-{seat}.vmap',seat) for seat in (0,1)],
        diagnostic_schedule=[r for block in range(8) for seat in (0,1) for r in diagnostic_schedule(seat,3,block,4)],
        periodic_origin_update=c.PERIODIC_ORIGIN,periodic_every_updates=c.PERIODIC_EVERY,next_periodic_update=c.latest_periodic_update(state.progress.updates)+c.PERIODIC_EVERY,
        control_seeds=[1130001,1130002],heldout={str(seat):c.schedule(c.context()['progress']['updates'],seat,True) for seat in (0,1)},
        distribution='Alternating four-game same-seat batches on open8x8 stage3; no difficulty transition in this campaign',
        success_criteria='Clean64game acceptance with natural wins and losses in each seat; capture/wait controls separately; descriptive fixed heldout win-rate change after PPO',
        hypothesis='The unchanged policy can sample wins on the easier start; terminal feedback may increase fixed heldout wins.',
        exclusions='All diagnostics, controls, previous studies and scripted rush excluded from PPO',
        acceptance_seconds=1200,absolute_compute_deadline=c.COMPUTE,absolute_hard_deadline=c.HARD,finalization_reserve_seconds=300,
        reward={'win':1,'loss':-1,'ordinary':0},created_utc=utc())
    path=c.SESSION.parent/'frozen.json';c.write_new(path,frozen)
    event('acceptance_and_campaign_preregistered',frozen=binding(path),protocol=frozen)
    print(json.dumps(binding(path)))


def prepare_release():
    private_preflight();frozen=c.check_frozen();require(not c.SESSION.exists(),'Consumed campaign cannot restart')
    counts=c.check_gate(c.BASE/'study4/study-report.json');state=c.restore(c.PARENT)
    drift=c.context()['platform_drift_report'];require(binding(drift['path'])==drift,'Platform drift receipt changed')
    drift_value=json.loads(Path(drift['path']).read_text());require(drift_value['passed'] is True and drift_value['absolute_tolerance']==1e-5 and drift_value['relative_tolerance']==1e-4,'Unaccepted platform drift')
    require(state.bindings==frozen['parent'],'Parent drift')
    # Both durable and private checkpoint copies must restore before the campaign clock starts.
    plan=json.loads(c.PARENT.read_text());report=json.loads(Path(state.bindings['report']['path']).read_text())
    proofs=[]
    from types import SimpleNamespace
    options=SimpleNamespace(checkpoint_options={'historical_torch_version':plan['source']['torch']})
    for folder in map(Path,c.context()['restore_backup_directories']):
        for kind in ('policy','resume'):require(binding(folder/f'after-{kind}.pt')['sha256']==state.bindings[kind]['sha256'],'Backup changed')
        proofs.append(verify_pair(folder/'after-resume.pt',folder/'after-policy.pt',plan,report,migration=options))
    release=dict(frozen,scope=c.SCOPE,authorization='Renewed overnight cloud authorization after final local ownership transfer and Linux acceptance; fixed compute cutoff2026-10-26T09:55Z and hard stop10:00Z; no deadline reset.',
        gate=binding(c.BASE/'study4/study-report.json'),validation_outcomes=counts,independent_parent_restores=proofs,
        absolute_compute_deadline=c.COMPUTE,absolute_hard_deadline=c.HARD,finalization_reserve_seconds=300,diagnostic_admission=False)
    path=c.SESSION.parent/'release.json';c.write_new(path,release)
    event('campaign_release_ready',release=binding(path),gate_counts=counts)
    print(json.dumps(binding(path)))


def analysis(v):
    admitted=[b for b in v['batches'] if b['mode']=='training' and b.get('checkpoint_verified')]
    evaluations=[b for b in v['batches'] if b['mode']!='training' and b.get('status')=='passed']
    result=dict(schema='heroes-overnight-curriculum-analysis-v1',timestamp_utc=utc(),status=v['status'],
        new_updates=v['new_updates'],progress=v['latest_progress'],training_outcomes_by_seat={},heldout_by_seat={},
        training_games=sum(b['completed_games'] for b in admitted),evaluation_games=sum(b['completed_games'] for b in evaluations),
        admitted_decisions=sum(b['training_samples'] for b in admitted),new_adam_steps=sum(b['optimizer_steps'] for b in admitted),
        failed_batches=[{k:b.get(k) for k in ('sequence','seat','mode','error','cleanup_error','preservation_error','plan','report')} for b in v['batches'] if b.get('status')=='failed'],
        scope='Seat-specific open8x8 easier-start curriculum, not original full-map performance',
        interpretation='Repeated four-game fixed checks per seat are descriptive paired evidence; curriculum wins or changed weights alone do not prove general strategy improvement.')
    behavior=Counter();battle_roles=Counter();reported_wall=0.;failed_attempts=unstarted=0
    for batch in v['batches']:
        if not batch.get('report'):continue
        bound=batch['report'];require(binding(bound['path'])==bound,'Analysis report changed')
        report=json.loads(Path(bound['path']).read_text())
        failed_attempts+=sum(a['status']=='failed' for a in report['attempts'])
        unstarted+=sum(a['status']=='not_started' for a in report['attempts'])
        if batch not in admitted and batch not in evaluations:continue
        if batch in admitted:reported_wall+=report['execution_wall_seconds']
        for attempt in report['attempts']:
            require(attempt['status']=='passed' and attempt['replay_verified'] and attempt['identity_verified'],'Analysis contains unverified game')
            bound=attempt['behavior'];require(binding(bound['path'])==bound,'Behavior evidence changed')
            metrics=json.loads(Path(bound['path']).read_text());require(metrics['status']=='complete' and metrics['coverage']['complete'],'Incomplete own-history behavior')
            agg=metrics['aggregate']
            for output,group,key in [('recruitment_completed_actions','recruitment','completed'),
                ('across_army_operations','army_operations','across_armies'),
                ('visible_net_army_redistribution','army_operations','observed_cross_army_net_count_redistribution'),
                ('movement_attempts','movement','attempts'),('movement_endpoint_changes','movement','endpoint_changes'),
                ('moves_toward_visible_opponent_town','movement','targets_visible_opponent_town')]:behavior[output]+=agg[group][key]
            behavior['town_capture_evidence']+=agg['town_capture_evidence']
            folder=Path(attempt['episode_directory']);manifest=json.loads((folder/'manifest.json').read_text())
            color=('red','blue')[attempt['learner_player']]
            # Select the history through its manifest rather than a guessed version.
            from experiments.headless.builtin_evidence import manifest_profile
            declaration=manifest['players'][color][manifest_profile(manifest).history_key]
            history_path=folder/declaration['path'];history_bytes=history_path.read_bytes()
            require(hashlib.sha256(history_bytes).hexdigest()==declaration['sha256'],'Analysis own history changed')
            for line in history_bytes.splitlines():
                frame=json.loads(line)
                if frame.get('kind')=='event' and frame.get('event',{}).get('kind')=='battle_start':battle_roles[frame['event']['side']]+=1
    result.update(failed_game_attempts=failed_attempts,unstarted_games=unstarted,behavior_totals=dict(behavior),
        battle_start_roles=dict(battle_roles),reported_training_batch_wall_seconds=reported_wall,
        admitted_decisions_per_reported_batch_second=result['admitted_decisions']/reported_wall if reported_wall else None,
        throughput_scope='Complete training batch capture, audit, native replay, PPO and checkpoint/supervisor wall; excludes private Git publication and acceptance.',
        behavior_scope='Verified training and heldout own player-visible histories only; controls, diagnostic and failed/partial games excluded.',
        behavior_limitations=['Recruitment counts actions, not unique units. Visible net transfer is not unique units or strategic concentration.',
            'Attacker role alone does not distinguish attacks on neutrals from attacks on the opponent.'])
    for seat in (0,1):
        color=('red','blue')[seat];rows=[b for b in v['batches'] if b['seat']==seat]
        outcomes=[o for b in admitted if b['seat']==seat for o in b['outcomes']]
        result['training_outcomes_by_seat'][color]=dict(natural_wins=outcomes.count('win'),natural_losses=outcomes.count('loss'),valid_cutoffs=outcomes.count(None))
        result['heldout_by_seat'][color]={mode:[dict(parent=b['parent'],outcomes=b.get('outcomes'),status=b['status']) for b in rows if b['mode']==mode] for mode in ('baseline','periodic','final')}
    result['valid_final_paired_comparison_available']=all(any(b['mode']=='final' and b['seat']==seat and b.get('status')=='passed'
        and b['parent']==v['latest_verified_parent'] for b in v['batches']) for seat in (0,1))
    return result


def restore_final_copies(v,state):
    from types import SimpleNamespace
    parent=Path(v['latest_verified_parent']['path'])
    plan=json.loads(parent.read_text());report=json.loads(Path(state.bindings['report']['path']).read_text())
    imported=parent==c.PARENT
    migration=SimpleNamespace(checkpoint_options={'historical_torch_version':plan['source']['torch']}) if imported else None
    folders=list(map(Path,c.context()['restore_backup_directories'])) if imported else [
        c.SESSION.parent/'checkpoints'/parent.parent.name,
        REPO/'checkpoints/overnight-20261026/campaign-v1'/c.SESSION.parent.name/parent.parent.name]
    proofs=[]
    for folder in folders:
        for kind in ('policy','resume'):
            actual=binding(folder/f'after-{kind}.pt');expected=state.bindings[kind]
            require(all(actual[k]==expected[k] for k in ('sha256','bytes')),'Final checkpoint copy differs')
        proof=verify_pair(folder/'after-resume.pt',folder/'after-policy.pt',plan,report,migration=migration)
        proofs.append(dict(directory=str(folder),restore=proof))
    return proofs


def launch(recover_from=None):
    private_preflight();path=c.SESSION.parent/'release.json';r=json.loads(path.read_text());frozen=c.check_frozen()
    require(r['source']==frozen['source'] and r['parent']==c.restore(c.PARENT).bindings,'Release source/parent drift')
    require(binding(r['gate']['path'])==r['gate'] and json.loads(json.dumps(c.check_gate(r['gate']['path'])))==r['validation_outcomes'],'Gate drift')
    require(not c.SESSION.exists(),'One campaign only')
    with c.SESSION.with_suffix('.lock').open('x') as f:f.write(str(os.getpid()))
    if recover_from is None:
        require(not c.WINDOW.exists(),'Original campaign window already consumed; use explicit recovery')
        now=time.time();mono=time.monotonic()
        require(now<c.COMPUTE-600,'Insufficient overnight compute reserve')
        clock=dict(started_unix=now,started_monotonic=mono,compute_deadline_unix=c.COMPUTE,absolute_deadline_unix=c.HARD)
        c.write_new(c.WINDOW,clock);previous=None
    else:
        require(c.SESSION!=c.BASE/'campaign-v1/session.json','Recovery requires a fresh subordinate session path')
        clock=json.loads(c.WINDOW.read_text());now=clock['started_unix'];mono=clock['started_monotonic']
        require(remaining(clock,600)>0,'Insufficient original window for safe recovery')
        old=Path(recover_from);require(old.is_relative_to(c.BASE/'campaign-v1'),'Foreign recovery journal')
        final=json.loads((old.parent/'controller-result.json').read_text())
        require(final['owned_cleanup_verified'],'Previous attempt cleanup required')
        for p in old.parent.rglob('*deadline*.jsonl'):cleanup_from_evidence(p,timeout=2)
        previous=binding(old)
    s=dict(scope=c.SCOPE,source=r['source'],runtime=r['runtime'],release=binding(path),window=binding(c.WINDOW),**clock,
        compute_deadline_monotonic=mono+c.COMPUTE-now,hard_deadline_monotonic=mono+c.HARD-now,controller_pid=os.getpid(),
        validation_outcomes=r['validation_outcomes'],recovery_from=previous,attempt_started_utc=utc())
    c.write_new(c.SESSION,s);event('campaign_launched',session=binding(c.SESSION),launch_utc=utc(now),compute_utc=utc(c.COMPUTE),hard_utc=utc(c.HARD))
    proc=None;result=dict(session=binding(c.SESSION))
    try:
        with (c.SESSION.parent/'controller.log').open('x') as log:
            proc=guarded_popen([sys.executable,str(Path(__file__).resolve()),'worker'],deadline_monotonic=mono+c.COMPUTE-now,deadline_wall=c.COMPUTE,
                evidence_path=c.SESSION.parent/'controller-deadline.jsonl',cwd=ROOT,env={**os.environ,'HEROES_OVERNIGHT_OWNER':'teacher-20261026','HEROES_OVERNIGHT_ROLE':'compute'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
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
        final_state=c.restore(v['latest_verified_parent']['path'])
        result['latest_restore']=final_state.bindings
        result['final_independent_restores']=restore_final_copies(v,final_state)
        v['owned_cleanup_verified']=not errors
        report=analysis(v);atomic(c.SESSION.parent/'analysis.json',report);v['analysis']=binding(c.SESSION.parent/'analysis.json');save(v)
        atomic(c.SESSION.parent/'controller-result.json',result)
        event('campaign_stopped',result=result,analysis=report)
        folder=c.SESSION.parent/'final-evidence';folder.mkdir(exist_ok=False)
        for p in c.SESSION.parent.iterdir():
            if p.is_file():shutil.copy2(p,folder/p.name)
        try:result['private_final_preservation']=publish(folder,'campaign-v1-'+c.SESSION.parent.name+'-final-session')
        except Exception as exc:result['final_preservation_error']=str(exc)
        result.update(finalized_utc=utc(),within_budget=remaining(s)>=0)
        atomic(c.SESSION.parent/'controller-result.json',result)
        atomic(c.SESSION.parent/'final-preservation.json',result)
        event('campaign_finalized',result=result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare-release','launch','worker','recover']);p.add_argument('--recover-from',type=Path);a=p.parse_args()
    if a.mode=='recover':
        require(a.recover_from is not None,'Explicit prior journal required');launch(a.recover_from)
    else:{'freeze':freeze,'prepare-release':prepare_release,'launch':launch,'worker':worker}[a.mode]()
