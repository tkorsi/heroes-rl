"""Explicit initial-state distribution continuation; unchanged PPO/admission worker."""
import argparse
from copy import deepcopy
import json,os,random,subprocess,sys,time
from pathlib import Path
import torch
from . import builtin_first_run as pilot,builtin_training as batches,builtin_training_resume as resume,checkpoints as cp
from .builtin_preservation import binding,verify_pair,preserve
from .inference import source_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import VcmiLauncher
from experiments.explorer.create_curriculum_scenario import verify

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
CONTEXT_PATH=ROOT/'configs/teacher-overnight-linux-v1.json'
HARD=1793008800.0
COMPUTE=1793008500.0

def context():
    value=json.loads(CONTEXT_PATH.read_text())
    relative=str(CONTEXT_PATH.relative_to(ROOT))
    require(subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)==CONTEXT_PATH.read_bytes(),'Migration context must be committed')
    require(value['hard_deadline_unix']==HARD and value['compute_deadline_unix']==COMPUTE,'Fixed overnight deadline changed')
    return value

def relocated(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'Unsafe historical path')
    if path.is_relative_to(BASE):return path
    for old,new in context()['relocation_roots'].items():
        if path.is_relative_to(old):
            target=(Path(new)/path.relative_to(old)).resolve()
            require(target.is_relative_to(Path(new).resolve()),'Relocation escape')
            return target
    raise ValueError('No explicit relocation for '+str(path))

def artifact(bound):
    actual=binding(relocated(bound['path']))
    require(all(actual[k]==bound[k] for k in ('sha256','bytes')),'Historical artifact bytes changed')
    return actual
WINDOW=BASE/'campaign-v1/window.json'
SESSION=Path(os.environ.get('HEROES_CURRICULUM_SESSION',str(BASE/'campaign-v1/session.json'))).resolve()
PARENT=Path(json.loads(CONTEXT_PATH.read_text())['parent']['plan']['path']) if CONTEXT_PATH.exists() else BASE/'not-yet-handed-off/plan.json'
INTERFACE='adventure-capabilities-v1'
SCOPE='heroes-overnight-curriculum-20261026'
require=resume.require
CORE=('contextual.py','contextual_model.py','contextual_rollout.py','checkpoints.py','ppo.py',
      'builtin_gameplay.py','builtin_rollout.py','trajectory.py','contract.py')


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def configuration():
    original=json.loads(PARENT.read_text())['configuration']
    return deepcopy(original)


def map_identity(path,seat):
    path=Path(path).resolve()
    require(path==BASE/f'maps/curriculum-3-{seat}.vmap','Only preregistered small-board state')
    verify(path,3,seat)
    return dict(binding(path),scenario='curriculum-v1-open8x8',stage=3,seat=seat)


def schedule(sequence,seat,evaluate=False):
    require(type(sequence) is int and sequence>=0 and seat in (0,1),'Invalid schedule')
    return [dict(episode=i+1,phase='evaluation' if evaluate else 'training',learner_player=seat,
        engine_seed=710001+i if evaluate else 910001+4*sequence+i,
        policy_seed=720001+i if evaluate else 920001+4*sequence+i) for i in range(4)]


def dependency_identity():
    import hashlib
    repo=ROOT/'vcmi'
    require(repo.is_dir() and not repo.is_symlink(),'Independent generated engine required')
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    head=git('rev-parse','HEAD').decode().strip()
    require(head==json.loads((ROOT/'dependencies/vcmi.json').read_text())['commit'],'Pinned engine HEAD changed')
    untracked=[p for p in git('ls-files','--others','--exclude-standard','-z').decode().split('\0') if p]
    return dict(head=head,diff_sha256=hashlib.sha256(git('diff','HEAD','--binary')).hexdigest(),
        untracked={p:binding(repo/p)['sha256'] for p in sorted(untracked)})


def runtime_files(directory):
    directory=Path(directory).resolve()
    files={}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(),'Frozen Linux runtime cannot retain mutable external links')
        if path.is_file():files[str(path.relative_to(directory))]=binding(path)
    require(all(name in files for name in ('vcmiclient','vcmiserver','vcmireplay','libvcmi.so','AI/libExplorerRL.so','AI/libNullkiller.so','AI/libBattleAI.so')),'Incomplete executable runtime')
    return files


def check_frozen():
    frozen=json.loads((SESSION.parent/'frozen.json').read_text())
    require(dependency_identity()==frozen['dependency'],'Generated dependency drift')
    require(str(ROOT)==frozen['checkout'], 'Run only from the frozen checkout')
    require(source_identity(INTERFACE)==frozen['source'] and not frozen['source']['tracked_worktree_dirty'], 'Frozen source drift')
    require(pilot.implementation()==frozen['implementation'], 'Frozen implementation drift')
    require(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()=='', 'Frozen checkout must remain clean')
    for name,value in frozen['bound_files'].items():require(binding(ROOT/name)==value,'Frozen tool/config drift')
    require(configuration()==frozen['configuration'], 'Frozen configuration drift')
    require(runtime_identity(Path(frozen['runtime_bin']))==frozen['runtime'], 'Frozen runtime drift')
    require(runtime_files(frozen['runtime_bin'])==frozen['runtime_files'],'Frozen client/server/module/resource tree drift')
    require(binding(frozen['runtime_report']['path'])==frozen['runtime_report'],'Native build/check receipt drift')
    require([map_identity(BASE/f'maps/curriculum-3-{seat}.vmap',seat) for seat in (0,1)]==frozen['maps'],'Frozen map drift')
    for key in ('plan','report','policy','resume'):require(binding(frozen['parent'][key]['path'])==frozen['parent'][key],'Frozen initial checkpoint drift')
    return frozen


def check_gate(path):
    frozen=check_frozen()
    gate=json.loads(Path(path).read_text());require(binding(SESSION.parent/'frozen.json')==gate['frozen'],'Gate belongs to another release')
    require(len(gate['attempts'])==20,'Complete64game/fourcontrol acceptance required')
    require(gate['status']=='finished' and gate['within_20_minutes']
        and gate['owned_cleanup_verified'],'Preparation study incomplete')
    counts={seat:{'win':0,'loss':0} for seat in (0,1)};controls={}
    for row in gate['attempts']:
        require(row['status']=='finished' and row['owned_cleanup_verified'],'Unfinished gate attempt')
        b=row['report'];require(binding(b['path'])==b,'Gate report changed');r=json.loads(Path(b['path']).read_text())
        if row['kind']=='control':
            require(r['replay']['verified'] and r['owned_processes_closed'] and r['termination_reason']=='natural_terminal','Control unverified')
            controls[(row['mode'],row['seat'])]=r['outcomes'][('red','blue')[row['seat']]]
        else:
            from validation.probes.validate_overnight_policy import schedule as diagnostic_schedule
            dp=json.loads(Path(r['plan']['path']).read_text())
            require(binding(r['plan']['path'])==r['plan'] and dp['study']==4 and dp['schedule']==diagnostic_schedule(row['seat'],3,row['block'],4),'Acceptance seeds/plan changed')
            require(dp['implementation']==frozen['capability_implementation'] and dp['runtime']==frozen['runtime'],'Acceptance source/runtime differ from frozen release')
            require(dp['checkpoint']==frozen['parent']['policy'] and len(r['attempts'])==4,'Acceptance checkpoint/game count changed')
            require(r['status']=='passed' and r['training_samples']==r['optimizer_steps']==0,'Diagnostic admitted or failed')
            for a in r['attempts']:
                require(a['status']=='passed' and a['replay_verified'] and a['weights_unchanged'] and a['finished'] and a['reconstruction']['passed'] and a['reconstruction']['weights_unchanged'] and a['termination_reason']=='natural_terminal', 'Policy gate unverified')
                counts[a['learner_player']][a['outcomes'][('red','blue')[a['learner_player']]]]+=1
    for seat in (0,1):
        require(sum(counts[seat].values())==32,'Incomplete seat acceptance')
        require(counts[seat]['win']>0 and counts[seat]['loss']>0,'Mixed outcomes required in each seat')
        require(controls[('capture',seat)]=='win' and controls[('wait',seat)]=='loss','Reachability/wait gate failed')
    return counts


def session():
    check_frozen()
    require(SESSION.is_relative_to(BASE/'campaign-v1'),'Foreign session namespace')
    v=json.loads(SESSION.read_text());clock=json.loads(WINDOW.read_text())
    require(binding(WINDOW)==v['window'],'Cloud clock binding changed')
    require(all(v[k]==clock[k] for k in ('started_unix','started_monotonic','compute_deadline_unix','absolute_deadline_unix')),'Recovery cannot reset cloud clock')
    require(v['scope']==SCOPE and v['source']==source_identity(INTERFACE),'Campaign source drift')
    require(v['absolute_deadline_unix']==HARD and v['compute_deadline_unix']==COMPUTE,'Overnight deadline extension forbidden')
    require(time.time()<COMPUTE and 0<=time.monotonic()-v['started_monotonic']<COMPUTE-v['started_unix'],'Campaign compute expired')
    require(binding(v['release']['path'])==v['release'],'Release drift')
    release=json.loads(Path(v['release']['path']).read_text())
    require(release['source']==v['source'] and release['configuration']==configuration(),'Release source/config drift')
    return v


def restore(path):
    """Exact learned-state restore with immutable historical metadata and explicit relocation."""
    path=Path(path).resolve();plan=json.loads(path.read_text())
    historical=path==PARENT
    report_path=relocated(Path(plan['execution_output'])/'report.json') if historical else Path(plan['execution_output'])/'report.json'
    report=json.loads(report_path.read_text());require(report['plan_sha256']==binding(path)['sha256'],'Parent report mismatch')
    if historical:
        expected=context()['parent']
        require(binding(path)==expected['plan'] and binding(report_path)==expected['report'],'Handoff parent differs')
        require(plan['source']==context()['historical_source'],'Historical source identity changed')
        resume._committed_source(plan['source'])
        current=source_identity(INTERFACE)
        for name,digest in plan['source']['source_sha256'].items():
            actual=current['source_sha256'].get(name)
            if actual!=digest:
                require(name=='experiments/learner/builtin_first_run.py' and context()['glue_changes'].get(name)=={'from':digest,'to':actual},'Unreviewed existing learner source change')
        require(plan['source']['interface']==INTERFACE,'Learned schema changed')
        saved={kind:artifact(report['after_checkpoint'][kind]) for kind in ('policy','resume')}
        for kind in saved:require(saved[kind]==expected[kind],'Checkpoint differs from final handoff')
        artifact(plan['checkpoint_metadata']['initial_checkpoint'])
        require(str(torch.__version__).removesuffix('+cpu')==plan['source']['torch'].removesuffix('+cpu'),'Torch public release changed')
        options={'historical_torch_version':plan['source']['torch']}
    else:
        require(path.is_relative_to(BASE/'campaign-v1') and plan['scope']==SCOPE,'Foreign cloud parent')
        require(plan['source']['torch']==str(torch.__version__) and plan['source']['python']==sys.version,'Unpinned cloud runtime change')
        saved={kind:report['after_checkpoint'][kind] for kind in ('policy','resume')};options={}
        for value in saved.values():require(binding(value['path'])==value,'Cloud checkpoint changed')
    for name in CORE:
        require(binding(ROOT/'experiments/learner'/name)['sha256']==plan['source']['source_sha256']['experiments/learner/'+name],'Policy/PPO implementation changed')
    metadata=resume._metadata(plan,report['plan_sha256']);progress,steps,sampler=resume._completion(plan,report)
    policy=cp.load_policy_checkpoint(saved['policy']['path'],expected_source_metadata=metadata,
        allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),interface=INTERFACE,**options)
    model=policy.model.requires_grad_(True);optimizer=torch.optim.Adam(model.parameters(),**plan['configuration']['adam']);generator=torch.Generator()
    state=cp.load_checkpoint(saved['resume']['path'],model,optimizer,expected_ppo_config=plan['configuration']['ppo'],
        expected_source_metadata=metadata,allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),interface=INTERFACE,
        generators={'minibatch':generator},at_rollout_boundary=True,**options)
    raw=torch.load(saved['resume']['path'],map_location='cpu',weights_only=True)
    for a,b in [(dict(model.state_dict()),raw['model_state']),(optimizer.state_dict(),raw['resume']['optimizer_state']),
                (torch.get_rng_state(),raw['rng']['torch_cpu']),(random.getstate(),raw['rng']['python']),
                (generator.get_state(),raw['rng']['torch_generators']['minibatch']),
                (state.opponent_pool.state_dict(),policy.opponent_pool.state_dict())]:resume._exact(a,b)
    require(state.progress==progress and state.sampler_state==sampler,'Parent progress drift')
    require(cp.model_weight_digest(model)==policy.weights_sha256==report['after_checkpoint']['weights_sha256'],'Learned weights differ')
    if historical:require(vars(progress)==context()['progress'] and steps==context()['optimizer_steps'],'Final handoff progress differs')
    links=dict(plan=binding(path),report=binding(report_path),policy=saved['policy'],resume=saved['resume'],
               progress=vars(progress),optimizer_steps=steps,source_proof={'explicit_platform_transition':historical,'policy_ppo_unchanged':True},
               runtime_proof={'portable_context':binding(CONTEXT_PATH),'historical_metadata_unchanged':True} if historical else {'runtime_unchanged':True})
    model.eval().requires_grad_(False)
    return resume.ParentState(model,optimizer,generator,state.opponent_pool,progress,sampler,metadata,policy.checkpoint,policy.weights_sha256,links,steps)


def check_identity(plan):
    v=session();require(plan['scope']==SCOPE and plan['source']==v['source'] and not v['source']['tracked_worktree_dirty'],'Unreleased source')
    require(plan['implementation']==pilot.implementation() and plan['prerequisites']==pilot.prerequisites(),'Implementation drift')
    require(plan['configuration']==configuration(),'PPO/configuration drift')
    seat=plan['curriculum']['seat'];require(plan['map']==map_identity(plan['game_map'],seat),'Map drift')
    require(plan['schedule']==schedule(plan['batch_sequence'],seat,plan['mode']=='evaluation'),'Schedule drift')
    require(plan['local_session']==binding(SESSION),'Wrong session')
    for k in ('checkpoint','replay_binary'):require(binding(plan[k]['path'])==plan[k], 'Artifact changed')
    launcher=VcmiLauncher(Path(plan['binary']),Path(plan['data']),Path(plan['game_map']))
    require(launcher.identity()==plan['native'] and runtime_identity(Path(plan['binary']).parent)==plan['runtime'],'Native drift')
    release=json.loads(Path(v['release']['path']).read_text());require(plan['runtime']==release['runtime'],'Unreleased runtime')
    require(plan['schema']==(batches.EVALUATION if plan['mode']=='evaluation' else batches.SCHEMA)+'-plan','Schema drift')
    require(plan['native_opponents']==[pilot.native_opponent(release['runtime'],s) for s in (0,1)],'Opponent drift')
    for key in ('plan','report','policy','resume'):require(binding(plan['parent'][key]['path'])==plan['parent'][key],'Parent artifact drift')
    require(plan['absolute_deadline_unix']==v['absolute_deadline_unix'],'Extended batch deadline')
    expected=v['compute_deadline_unix']
    require(plan['compute_deadline_unix']==expected,'Changed compute budget')


def prepare(parent,sequence,seat,evaluate=False,final=False):
    v=session()
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_vcmi.py'),'--check'],cwd=ROOT,check=True,capture_output=True,timeout=60)
    state=restore(parent);release=json.loads(Path(v['release']['path']).read_text())
    output=SESSION.parent/'batches'/f"{'final' if final else 'eval' if evaluate else 'train'}-{sequence:04d}-seat{seat}"
    require(not output.exists(),'No retry or overwrite')
    rows=schedule(sequence,seat,evaluate)
    if not evaluate:
        require(sequence>=context()['next_unused_training_sequence'],'Consumed local sequence')
        keys={(r['engine_seed'],r['policy_seed'],r['learner_player']) for r in rows}
        for f in [*(BASE/'campaign-v1').rglob('plan.json'),*(Path(context()['historical_campaign_root'])).rglob('plan.json')]:
            old=json.loads(f.read_text());require(not keys & {(r['engine_seed'],r['policy_seed'],r['learner_player']) for r in old['schedule'] if r['phase']=='training'},'Consumed seeds')
    game_map=BASE/f'maps/curriculum-3-{seat}.vmap';binary=Path(release['runtime_bin'])/'vcmiclient'
    native=VcmiLauncher(binary,Path(release['data']),game_map).identity()
    from .builtin_first_run import native_opponent
    plan=dict(schema=(batches.EVALUATION if evaluate else batches.SCHEMA)+'-plan',scope=SCOPE,mode='evaluation' if evaluate else 'training',
        source=v['source'],runtime=release['runtime'],native=native,map=map_identity(game_map,seat),binary=str(binary),data=release['data'],
        game_map=str(game_map),replay_binary=binding(binary.parent/'vcmireplay'),native_opponents=[native_opponent(release['runtime'],s) for s in (0,1)],
        configuration=configuration(),schedule=rows,batch_sequence=sequence,execution_output=str(output/'execution'),
        parent=state.bindings,checkpoint=state.bindings['policy'],local_session=binding(SESSION),authorization=release['authorization'],
        absolute_deadline_unix=v['absolute_deadline_unix'],compute_deadline_unix=v['compute_deadline_unix'],
        implementation=pilot.implementation(),prerequisites=pilot.prerequisites(),
        wall_scope='Fixed overnight compute cutoff2026-10-26T09:55:00Z; hard stop10:00Z; no clock reset',
        comparison_scope='Same8x8 initial-state distribution, separate seats, fixed heldout seeds; not original-start performance',
        curriculum={'stage':3,'seat':seat,'map_change_explicit':True},final_evaluation=final)
    plan['checkpoint_metadata']={**v['source'],'parent_plan_sha256':state.bindings['plan']['sha256'],
        'parent_report_sha256':state.bindings['report']['sha256'],'initial_checkpoint':state.bindings['policy'],
        'native_opponent':{k:x for k,x in plan['native_opponents'][0].items() if k!='player'},'archived_neural_pool_executed':False}
    check_identity(plan);write_new(output/'plan.json',plan);return output/'plan.json'


def read_plan(path,sha):
    path=Path(path);require(path.is_relative_to(SESSION.parent/'batches') and binding(path)['sha256']==sha,'Plan changed')
    plan=json.loads(path.read_text());require(plan['execution_output']==str(path.parent/'execution'),'Output drift');check_identity(plan);return plan


def heldout(state,runtime,native):
    # Every campaign starts with both frozen seats. Periodic/final checks are controller-owned.
    v=session();journal=json.loads((SESSION.parent/'campaign-report.json').read_text())
    for seat in (0,1):
        require(any(b.get('mode')=='baseline' and b.get('seat')==seat and b.get('status')=='passed'
                    for b in journal['batches']),'Both frozen baselines required')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['execute','worker']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--started',type=float);p.add_argument('--wall-started',type=float);a=p.parse_args()
    if a.command=='worker':result=batches.worker(a,plan_reader=read_plan,state_reader=lambda plan:restore(plan['parent']['plan']['path']),heldout_checker=heldout)
    else:
        result=pilot.execute(a,plan_reader=read_plan,worker_module=__spec__.name)
        if result['status']=='finished' and result['training_performed']:
            preserve(a.plan,a.plan.parent/'execution/report.json',SESSION.parent/'checkpoints'/a.plan.parent.name)
    print(json.dumps({k:result.get(k) for k in ('status','stop_reason','optimizer_updates','optimizer_steps','training_samples')}))

if __name__=='__main__':main()
