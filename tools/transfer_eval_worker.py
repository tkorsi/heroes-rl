"""Frozen diagnostics with original learner imports; never performs an update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def require(value,message):
    if not value:raise ValueError(message)


def binding(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def resources():
    fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    return dict(rss_bytes=int(fields['VmRSS'].split()[0])*1024,descriptors=len(list(Path('/proc/self/fd').iterdir())),threads=int(fields['Threads']))


def load_frozen(spec):
    import torch
    from experiments.learner.builtin_preservation import verify_pair
    from experiments.learner.builtin_training_resume import _metadata
    from experiments.learner import checkpoints as cp
    for bound in spec['files'].values():require(binding(bound['path'])==bound,'Frozen input changed')
    for value in spec['dependencies'].values():
        require(binding(value['relocated']['path'])==value['relocated'],'Dependency changed')
        require(all(value['original_binding'][k]==value['relocated'][k] for k in ('sha256','bytes')),'Relocation altered dependency')
    plan=json.loads(Path(spec['files']['plan.json']['path']).read_text());report=json.loads(Path(spec['files']['report.json']['path']).read_text())
    require(str(torch.__version__).removesuffix('+cpu')==plan['source']['torch'].removesuffix('+cpu'),'Public CPU Torch release changed')
    options=dict(historical_torch_version=plan['source']['torch'])
    proof=verify_pair(spec['files']['after-resume.pt']['path'],spec['files']['after-policy.pt']['path'],plan,report,
        migration=SimpleNamespace(checkpoint_options=options))
    require(proof['progress']==spec['progress'] and proof['exact_model_adam_rng_progress_pool'],'Full restoration failed')
    loaded=cp.load_policy_checkpoint(spec['files']['after-policy.pt']['path'],expected_source_metadata=_metadata(plan,report['plan_sha256']),
        allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),interface='adventure-capabilities-v1',**options)
    loaded.model.eval().requires_grad_(False)
    return loaded,proof


def execute(args):
    sys.path.insert(0,str(args.source))
    import torch
    torch.set_num_threads(1)
    from experiments.learner.inference import source_identity
    from experiments.learner.builtin_rollout import BuiltinCollector,audit_persisted_builtin_policy,retain_failed_builtin_evidence
    from experiments.learner.evaluate_builtin_opponent import native_opponent
    from experiments.learner.builtin_behavior import analyze_episode
    from experiments.headless.builtin_replay import verify_episode
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher,Limits
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from validation.probes.validate_builtin_boundaries import until
    from experiments.learner.checkpoints import model_weight_digest
    manifest=json.loads(args.manifest.read_text());spec=manifest['checkpoints'][str(args.checkpoint)]
    args.output.mkdir(parents=True,exist_ok=False)
    loaded,proof=load_frozen(spec);source=source_identity('adventure-capabilities-v1')
    require(not source['tracked_worktree_dirty'],'Uncommitted learner source')
    result=dict(status='prepared',mode=args.mode,restore_before=proof,source=source,rows=[],training_samples=0,ppo_updates=0,optimizer_steps=0,resources_start=resources())
    write(args.output/'report.json',result)
    if args.mode=='restore':
        backup=dict(spec,files={name:dict(bound,path=str(Path(spec['backup'])/name)) for name,bound in spec['files'].items()})
        _,result['independent_backup_restore']=load_frozen(backup)
        result.update(status='passed',restore_after=load_frozen(spec)[1]);write(args.output/'report.json',result);return
    if args.mode=='archived':
        for folder in args.archived:
            raw=json.loads((folder/'builtin-rollout.json').read_text())
            try:
                receipt=audit_persisted_builtin_policy(folder,loaded.model,loaded.checkpoint,raw['policy_seed'],raw['opponent_identity'])
                result['rows'].append(dict(folder=str(folder),status='passed',audit=receipt))
            except Exception as exc:result['rows'].append(dict(folder=str(folder),status='failed',error=f'{type(exc).__name__}: {exc}'))
        result['status']='passed' if all(x['status']=='passed' for x in result['rows']) else 'failed'
        result['restore_after']=load_frozen(spec)[1];write(args.output/'report.json',result);return
    window=json.loads(args.window.read_text());require(time.time()<window['compute_deadline_unix'],'Expired diagnostic window')
    require(source==manifest['sources'][args.runtime_key],'Source differs from prospective manifest')
    for path,expected in manifest['data_stat'].items():
        stat=Path(path).stat();actual={k:getattr(stat,k) for k in expected}
        require(actual==expected,'Private game Data changed since full hash verification')
    rows=json.loads(args.rows.read_text());require(1<=len(rows)<=4,'Embedded process game cap exceeded')
    require(all(r['checkpoint']==args.checkpoint for r in rows),'Mixed checkpoint process')
    map_bound=manifest['maps'][rows[0]['map_key']];require(all(r['map_key']==rows[0]['map_key'] and r['seat']==rows[0]['seat'] for r in rows),'One map/seat per worker')
    require(binding(map_bound['path'])=={k:map_bound[k] for k in ('path','sha256','bytes')},'Map bytes changed')
    runtime=runtime_identity(args.binary.parent);engine=None;env=None
    require(runtime==manifest['runtimes'][args.runtime_key]['files'],'Runtime differs from prospective manifest')
    for group in ('all_runtime_files','module_files'):
        for bound in manifest['runtimes'][args.runtime_key][group].values():
            require(binding(bound['path'])==bound,'Frozen executable/module/resource changed')
    launcher=VcmiLauncher(args.binary,Path(manifest['data']),Path(map_bound['path']))
    if args.backend=='embedded':
        from experiments.headless.embedded_env import EmbeddedEngine,EmbeddedLauncher,EmbeddedBuiltinEnv
        launcher=EmbeddedLauncher(args.binary,Path(manifest['data']),Path(map_bound['path']));engine=EmbeddedEngine.acquire(launcher,args.output/'embedded-worker')
    try:
        for index,row in enumerate(rows):
            item=dict(schedule=row,status='running',resources_before=resources(),training_samples=0,optimizer_steps=0)
            result['rows'].append(item);write(args.output/'report.json',result);start=time.monotonic();collector=None;env=None
            try:
                seat=row['seat'];opponent=native_opponent(runtime,seat)
                limits=Limits(max_days=365,max_actions=5120,startup_seconds=60,shutdown_seconds=20,watchdog_seconds=60,bridge_timeout=30)
                options=dict(learner_player=seat,limits=limits,agents={('red','blue')[seat]:dict(kind='frozen-transfer-evaluation',evaluation_only=True)},interface='adventure-capabilities-v1')
                env=EmbeddedBuiltinEnv(launcher,args.output/'games',engine=engine,**options) if engine else BuiltinHeadlessEnv(launcher,args.output/'games',**options)
                collector=BuiltinCollector(loaded.model,loaded.checkpoint,learner_player=seat,opponent_identity=opponent,policy_seed=row['policy_seed'],interface='adventure-capabilities-v1')
                wall=min(window['compute_deadline_unix'],time.time()+300);mono=min(window['compute_deadline_monotonic'],time.monotonic()+300)
                with until(mono,wall,0):
                    episode=collector.run_episode(env,episode=index+1,seed=row['engine_seed'],max_requests=4960)
                write(env.episode_dir/'builtin-rollout.json',episode.to_dict());item['collection_seconds']=time.monotonic()-start
                env.close();folder=env.episode_dir;item['episode_directory']=str(folder)
                begun=time.monotonic();replayed=verify_episode(folder,binary=args.replay_binary,data=Path(manifest['data']),timeout=min(300,max(1,window['compute_deadline_unix']-time.time())))
                item.update(replay_seconds=time.monotonic()-begun,replay_verified=replayed.get('verified') is True)
                require(item['replay_verified'],'Independent replay failed: '+str(replayed.get('error')))
                rebuilt,_=load_frozen(spec);begun=time.monotonic()
                audit=audit_persisted_builtin_policy(folder,rebuilt.model,rebuilt.checkpoint,row['policy_seed'],opponent)
                item.update(audit_seconds=time.monotonic()-begun,policy_reconstruction=audit)
                require(audit['passed'],'Strict policy reconstruction failed')
                behavior=analyze_episode(folder);require(behavior['coverage']['complete'],'Behavior evidence incomplete');write(folder/'behavior.json',behavior)
                native=json.loads((folder/'manifest.json').read_text());profile=__import__('experiments.headless.builtin_evidence',fromlist=['manifest_profile']).manifest_profile(native)
                declaration=native['players'][('red','blue')[seat]][profile.history_key]
                history=(folder/declaration['path']).read_bytes();roles={}
                require(hashlib.sha256(history).hexdigest()==declaration['sha256'],'Behavior history binding differs')
                for line in history.splitlines():
                    frame=json.loads(line)
                    if frame.get('kind')=='event' and frame.get('event',{}).get('kind')=='battle_start':
                        role=frame['event']['side'];roles[role]=roles.get(role,0)+1
                item.update(status='passed',outcome=native['outcomes_by_player'][('red','blue')[seat]],termination_reason=native['termination_reason'],
                    episode_status=native['status'],decisions=len(episode.decisions),history_sha256=declaration['sha256'],
                    decisions_sha256=hashlib.sha256(json.dumps(episode.to_dict()['decisions'],sort_keys=True).encode()).hexdigest(),
                    estimates_sha256=hashlib.sha256(json.dumps(episode.to_dict()['estimates'],sort_keys=True).encode()).hexdigest(),
                    behavior=behavior['aggregate'],battle_start_roles=roles,manifest=binding(folder/'manifest.json'),cleanup_verified=True)
            except Exception as exc:
                item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                if env is not None:retain_failed_builtin_evidence(env,collector,item,exc,cleanup_seconds=20)
            finally:
                if env is not None:env.close()
                item.update(resources_after=resources(),wall_seconds=time.monotonic()-start)
                require(model_weight_digest(loaded.model)==proof['weights_sha256'],'Frozen model changed')
                write(args.output/'report.json',result)
            if item['status']!='passed':break
    finally:
        if engine is not None and engine.active:engine.finish(natural=False,timeout=20)
        result.update(status='passed' if len(result['rows'])==len(rows) and all(x['status']=='passed' for x in result['rows']) else 'failed',
            restore_after=load_frozen(spec)[1],resources_end=resources(),runtime_unchanged=runtime_identity(args.binary.parent)==runtime,
            source_unchanged=source_identity('adventure-capabilities-v1')==source)
        require(result['runtime_unchanged'] and result['source_unchanged'],'Source/runtime drift')
        write(args.output/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('restore','archived','run'),required=True)
    for key in ('source','manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint',type=int,choices=(146,254,412),required=True)
    for key in ('binary','replay-binary','window','rows'):p.add_argument('--'+key,type=Path)
    p.add_argument('--backend',choices=('cold','embedded'),default='cold');p.add_argument('--archived',type=Path,nargs='*',default=[])
    p.add_argument('--runtime-key',choices=('cold','accelerated'))
    execute(p.parse_args())
