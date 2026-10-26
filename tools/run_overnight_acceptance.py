"""One prospective, at-most20-minute preparation study; never launches PPO."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.learner.builtin_training_resume import binding
from experiments.headless.process_deadline import guarded_popen,close_owned_process,cleanup_from_evidence
BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
RUNTIME=Path('/workspace/Heroes-artifacts/runtime/overnight-20261026/bin')
DATA=Path('/workspace/Heroes-artifacts/game-data')
CHECKPOINT=None


def main():
    from experiments.learner.overnight_training import check_frozen,COMPUTE
    frozen=check_frozen()
    checkpoint=Path(frozen['parent']['policy']['path'])
    output=BASE/'study4';output.mkdir(exist_ok=False)
    start=time.time();mono=time.monotonic();hard=min(start+1200,COMPUTE);limit=mono+hard-start
    schedule=[{'kind':'control','mode':mode,'seat':seat,'engine_seed':1130001+seat} for mode in ('capture','wait') for seat in (0,1)]
    schedule += [{'kind':'policy','block':block,'seat':seat,'games':4} for block in range(8) for seat in (0,1)]
    ledger={'schema':'heroes-curriculum-study4-v1','started_unix':start,'started_monotonic':mono,
            'hard_deadline_unix':hard,'hard_deadline_monotonic':limit,'ppo_launched':False,'status':'running','frozen':binding(BASE/'campaign-v1/frozen.json'),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'checkpoint':binding(checkpoint),'attempts':[dict(row,status='not_started') for row in schedule]}
    def save():
        p=output/'study-report.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,p)
    save()
    for i,row in enumerate(ledger['attempts']):
        if min(hard-time.time(),limit-time.monotonic())<40:ledger['status']='budget_stop';break
        check_frozen()
        folder=output/(f"control-{row['mode']}-{row['seat']}" if row['kind']=='control' else f"policy-block{row['block']}-seat{row['seat']}")
        row.update(status='running',output=str(folder),started_unix=time.time());save()
        common=['--data',str(DATA),'--map',str(BASE/f"maps/curriculum-3-{row['seat']}.vmap"),'--output',str(folder)]
        env={**os.environ,'HEROES_ARTIFACT_ROOT':'/workspace/Heroes-artifacts'}
        if row['kind']=='control':
            env['CURRICULUM_CONTROL']=row['mode']
            command=[sys.executable,'-m','validation.probes.validate_curriculum_control',*common,'--runtime',str(RUNTIME),
                     '--seat',str(row['seat']),'--seed',str(row['engine_seed']),'--seconds','120','--max-actions','5120','--hard-stop',str(hard-10)]
        else:
            command=[sys.executable,'-m','validation.probes.validate_overnight_policy',*common,'--binary',str(RUNTIME/'vcmiclient'),
                     '--replay-binary',str(RUNTIME/'vcmireplay'),'--checkpoint',str(checkpoint),
                     '--study','4','--seed-block',str(row['block']),'--deadline-unix',str(hard-10)]
        proc=None
        try:
            with (output/f'attempt-{i}.log').open('x') as log:
                proc=guarded_popen(command,deadline_monotonic=limit-5,deadline_wall=hard-5,
                    evidence_path=output/f'attempt-{i}-deadline.jsonl',cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                rc=proc.wait(timeout=max(1,min(hard-time.time(),limit-time.monotonic())-8))
            row.update(status='finished' if rc==0 else 'failed',returncode=rc)
            if (folder/'report.json').exists():row['report']=binding(folder/'report.json')
        except Exception as error:
            row.update(status='failed',error=f'{type(error).__name__}: {error}')
        finally:
            if proc is not None:close_owned_process(proc,timeout=1)
            for path in folder.glob('native-deadline*.jsonl'):cleanup_from_evidence(path,timeout=1)
            row.update(finished_unix=time.time(),owned_cleanup_verified=True);save()
        if row['status']=='failed':ledger['status']='technical_failure';break
    else:ledger['status']='finished'
    ledger.update(stopped_unix=time.time(),within_20_minutes=time.time()<=hard,owned_cleanup_verified=True)
    save();print(json.dumps({'status':ledger['status'],'path':str(output/'study-report.json')}))

if __name__=='__main__':main()
