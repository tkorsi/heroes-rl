"""One bounded frozen transfer study, with no training command or admission path."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.transfer_eval_watchdog import process,same,scan,stop
from tools.transfer_eval_worker import binding,write,require
from experiments.headless.process_deadline import guarded_popen,close_owned_process,cleanup_from_evidence
BASE=Path('/workspace/Heroes-artifacts/transfer-eval-20261026')
AUTHOR=Path('/workspace/heroes-transfer')
PRIVATE=Path('/workspace/heroes-transfer-private')
BRANCH='codex/teacher-transfer-eval-20261026'
CONFIG=AUTHOR/'configs/teacher-transfer-eval-20261026.json'


def utc():return datetime.now(timezone.utc).isoformat()


def event(kind,**fields):
    row=dict(timestamp_utc=utc(),event=kind,**fields)
    for path in (BASE/'events.jsonl',AUTHOR/'validation/transfer-eval-2026-10-26.jsonl'):
        with path.open('a') as out:out.write(json.dumps(row)+'\n');out.flush();os.fsync(out.fileno())
    with (AUTHOR/'docs/research/transfer-eval-2026-10-26.md').open('a') as out:out.write('\n- '+row['timestamp_utc']+' '+kind+': '+json.dumps(fields)+'\n')


def git(root,*args):
    if (BASE/'window.json').exists():require(time.time()<json.loads((BASE/'window.json').read_text())['hard_deadline_unix']-10,'Hard publication stop')
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT,timeout=45).decode().strip()


def push(root):
    head=git(root,'rev-parse','HEAD');errors=[]
    for attempt in range(3):
        try:git(root,'push','origin','HEAD:refs/heads/'+BRANCH)
        except Exception as exc:errors.append(str(exc))
        if git(root,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]==head:return dict(commit=head,remote_acknowledged=True,attempts=attempt+1,publication_errors=errors)
        time.sleep(1)
    raise ValueError('No exact remote acknowledgement after bounded publication')


def publish_public(message,final=False):
    paths=['docs/research/transfer-eval-2026-10-26.md','validation/transfer-eval-2026-10-26.jsonl','reports/cloud/teacher-transfer-status.json']
    if final:paths.append('validation/transfer-eval-2026-10-26.json')
    git(AUTHOR,'add','-f','--',*paths)
    if git(AUTHOR,'diff','--cached','--name-only'):git(AUTHOR,'commit','-m',message)
    return push(AUTHOR)


def preserve(label,folders):
    target=PRIVATE/'validations/teacher-transfer-eval-20261026'/label;target.mkdir(parents=True,exist_ok=False)
    archive=BASE/(label+'.tar.gz')
    with tarfile.open(archive,'x:gz',dereference=False) as tar:
        for folder in folders:tar.add(folder,arcname=Path(folder).name)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,'Unsafe evidence archive')
    pieces=[]
    with archive.open('rb') as stream:
        index=0
        while True:
            raw=stream.read(16*1024*1024)
            if not raw:break
            path=target/f'evidence.tar.gz.part{index:03d}';path.write_bytes(raw);pieces.append(binding(path));index+=1
    receipt=dict(archive=binding(archive),parts=pieces,game_data_symlinks_dereferenced=False,checkpoint_or_game_data_in_public_source=False)
    write(target/'receipt.json',receipt)
    git(PRIVATE,'add','--',str(target.relative_to(PRIVATE)));git(PRIVATE,'commit','-m','Preserve frozen transfer '+label)
    ack=push(PRIVATE)
    for piece in pieces:
        raw=subprocess.check_output(['git','-C',str(PRIVATE),'show',ack['commit']+':'+str(Path(piece['path']).relative_to(PRIVATE))],timeout=30)
        require(hashlib.sha256(raw).hexdigest()==piece['sha256'],'Private Git evidence blob differs')
    return dict(**ack,**receipt)


def status(value,phase,**fields):
    games=[r for cell in value['runs'] for r in cell.get('rows',[])]
    latest=dict(timestamp_utc=utc(),phase=phase,scope='Frozen diagnostics only; overnight PPO closed',ownership='cloud-transfer-only',
        training_samples=0,ppo_updates=0,optimizer_steps=0,gate_game_cap=8,transfer_game_cap=192,
        games_attempted=len(games),games_passed=sum(x['status']=='passed' for x in games),
        gate_status=value.get('gate_status'),selected_backend=value.get('selected_backend'),controller_pid=os.getpid(),
        runs_report=str(BASE/'result.json'),**fields)
    if (BASE/'window.json').exists():latest.update(window=json.loads((BASE/'window.json').read_text()),remaining_compute_seconds=max(0,json.loads((BASE/'window.json').read_text())['compute_deadline_unix']-time.time()))
    write(AUTHOR/'reports/cloud/teacher-transfer-status.json',latest);write(BASE/'status.json',latest)


def worker(manifest,schedule,label,runtime_key,backend,window):
    folder=BASE/'runs'/label;rows_path=BASE/'schedules'/(label+'.json');rows_path.parent.mkdir(exist_ok=True)
    require(not rows_path.exists(),'Attempt already consumed');write(rows_path,schedule)
    runtime=manifest['runtimes'][runtime_key]
    command=[sys.executable,str(ROOT/'tools/transfer_eval_worker.py'),'--mode','run','--source',runtime['source'],
        '--manifest',str(BASE/'manifest.json'),'--checkpoint',str(schedule[0]['checkpoint']),'--output',str(folder),
        '--binary',runtime['binary'],'--replay-binary',runtime['replay_binary'],'--window',str(BASE/'window.json'),
        '--rows',str(rows_path),'--backend',backend,'--runtime-key',runtime_key]
    evidence=BASE/'deadlines'/(label+'.jsonl');evidence.parent.mkdir(exist_ok=True);proc=None;started=time.monotonic()
    record=dict(label=label,schedule=schedule,command=command,started_utc=utc(),rows=[])
    event('game_cell_launch_intent',label=label,schedule=schedule,source=manifest['sources'][runtime_key]['commit'],backend=backend)
    env=dict(os.environ);env.pop('HEROES_VERIFY_NATIVE_CAPTURE',None)
    env['LD_LIBRARY_PATH']=str(Path(runtime['binary']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    if label.startswith('gate-accelerated'):env['HEROES_VERIFY_NATIVE_CAPTURE']='1'
    try:
        with (BASE/'logs'/(label+'.log')).open('x') as log:
            proc=guarded_popen(command,deadline_wall=window['compute_deadline_unix'],deadline_monotonic=window['compute_deadline_monotonic'],
                evidence_path=evidence,cwd=runtime['source'],env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            record['process_identity']=process(proc.pid)
            record['returncode']=proc.wait(timeout=max(.1,min(window['compute_deadline_unix']-time.time(),window['compute_deadline_monotonic']-time.monotonic())))
    except Exception as exc:record['supervisor_error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if proc is not None:close_owned_process(proc,timeout=2)
            cleanup_from_evidence(evidence,timeout=1)
            for path in folder.rglob('*deadline*.jsonl'):cleanup_from_evidence(path,timeout=1)
            record['owned_cleanup_verified']=True
        except Exception as exc:record['cleanup_error']=str(exc);record['owned_cleanup_verified']=False
    path=folder/'report.json'
    if path.exists():
        report=json.loads(path.read_text());record.update(rows=report['rows'],worker_report=binding(path),worker_status=report['status'],
            resources_start=report.get('resources_start'),resources_end=report.get('resources_end'))
    for row in record['rows']:
        if row['status']!='passed':row['status']='failed'
    if not record['rows']:record['rows']=[dict(schedule=schedule[0],status='failed',error=record.get('supervisor_error','Worker stopped before a complete row receipt'))]
    record.update(status='passed' if record.get('returncode')==0 and record.get('worker_status')=='passed' and record.get('owned_cleanup_verified') else 'failed',
        finished_utc=utc(),wrapper_seconds=time.monotonic()-started)
    event('game_cell_result',label=label,status=record['status'],report=record.get('worker_report'))
    return record


def gate_comparison(runs):
    baseline={};accelerated={}
    for cell in runs:
        for row in cell.get('rows',[]):
            key=(row['schedule']['seat'],row['schedule']['engine_seed'],row['schedule']['policy_seed'])
            (baseline if cell['label'].startswith('gate-cold') else accelerated)[key]=row
    comparisons=[]
    for key,cold in baseline.items():
        other=accelerated.get(key);fields=('history_sha256','decisions_sha256','estimates_sha256','outcome','termination_reason')
        equal=other is not None and cold['status']==other['status']=='passed' and all(cold.get(k)==other.get(k) for k in fields)
        comparisons.append(dict(seat=key[0],engine_seed=key[1],policy_seed=key[2],strict_equal=equal,
            differences=[] if equal else [k for k in fields if other is None or cold.get(k)!=other.get(k)]))
    return dict(passed=len(baseline)==len(accelerated)==4 and all(x['strict_equal'] for x in comparisons),comparisons=comparisons)


def analysis(value,manifest):
    allrows=[r for cell in value['runs'] if cell['label'].startswith('transfer-') for r in cell.get('rows',[])]
    by_id={r['schedule']['id']:r for r in allrows};cells={};paired={}
    for checkpoint in (146,254,412):
        for scenario in ('easy','original'):
            for seat in (0,1):
                intended=[r for r in manifest['transfer_schedule'] if r['checkpoint']==checkpoint and r['scenario']==scenario and r['seat']==seat]
                observed=[by_id[r['id']] for r in intended if r['id'] in by_id];passed=[r for r in observed if r['status']=='passed']
                wins=sum(r.get('outcome')=='win' for r in passed);losses=sum(r.get('outcome')=='loss' for r in passed);n=len(passed)
                interval=None
                if n:
                    z=1.959963984540054;p=wins/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;interval=[center-radius,center+radius]
                metrics=Counter();battles=Counter()
                for row in passed:
                    m=row['behavior']
                    for name,group,key in [('recruitment','recruitment','completed'),('army_transfers','army_operations','across_armies'),('visible_net_redistribution','army_operations','observed_cross_army_net_count_redistribution'),('movement_attempts','movement','attempts'),('movement_endpoints','movement','endpoint_changes'),('target_opponent_town','movement','targets_visible_opponent_town')]:metrics[name]+=m[group][key]
                    metrics['town_capture_evidence']+=m['town_capture_evidence'];battles.update(row['battle_start_roles'])
                cells[f'{checkpoint}-{scenario}-seat{seat}']=dict(checkpoint=checkpoint,scenario=scenario,seat=seat,planned=16,valid=n,natural_wins=wins,natural_losses=losses,
                    valid_cutoffs=n-wins-losses,technical_failures=len(observed)-n,unstarted=16-len(observed),win_fraction=wins/n if n else None,wilson95=interval,
                    observed_behavior=dict(metrics),battle_start_roles=dict(battles))
    for scenario in ('easy','original'):
        for earlier in (146,254):
            pairs=[]
            for row in allrows:
                s=row['schedule']
                if s['checkpoint']!=412 or s['scenario']!=scenario or row['status']!='passed':continue
                matches=[x for x in allrows if x['schedule']['checkpoint']==earlier and all(x['schedule'][k]==s[k] for k in ('scenario','seat','map_key','engine_seed','policy_seed')) and x['status']=='passed']
                if len(matches)==1:pairs.append((row,matches[0]))
            gain=sum(a.get('outcome')=='win' and b.get('outcome')!='win' for a,b in pairs);loss=sum(b.get('outcome')=='win' and a.get('outcome')!='win' for a,b in pairs)
            paired[f'412-vs-{earlier}-{scenario}']=dict(valid_matched_pairs=len(pairs),wins_only_412=gain,wins_only_earlier=loss,
                both_win=sum(a.get('outcome')==b.get('outcome')=='win' for a,b in pairs),neither_win=sum(a.get('outcome')!='win' and b.get('outcome')!='win' for a,b in pairs),
                scope='Descriptive exact-start pairing; neither-win includes any explicitly reported valid cutoff')
    return dict(schema='heroes-frozen-transfer-result-v1',timestamp_utc=utc(),status=value['status'],cells=cells,paired_comparisons=paired,
        gate_status=value.get('gate_status'),gate_comparison=value.get('gate_comparison'),selected_backend=value.get('selected_backend'),
        games_attempted=len(allrows),training_samples=0,ppo_updates=0,optimizer_steps=0,window=value.get('window'),private_preservation=value.get('private_preservation',[]),
        limitations=['16 games per checkpoint/scenario/seat; seed does not fix all native opponent scheduling.',
            'Repeated four fixed old gate starts are not transfer games. Varied easier coordinates and original scenario are separate.',
            'Embedded process reuse capped at two gate games and one transfer game; no memory plateau or PPO throughput claim.',
            'Behavior metrics are descriptive visible-history evidence, not rewards; attacker role alone need not identify opponent targets.'],
        next_curriculum_proposal='Use varied easy8x8 starts before increasing map size, contingent on the matched results; original-map economy/recruitment requires a separately authorized training budget.',
        no_training_authorized_after_diagnostics=True)


def main():
    lock=(BASE/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (BASE/'window.json').exists(),'This diagnostic clock is already consumed')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        p=process(int(path.name))
        if p:require(not any(x.startswith('/workspace/releases/teacher-overnight-') for x in p['command']),'Old Teacher process still live')
    manifest=json.loads(CONFIG.read_text());write(BASE/'manifest.json',manifest)
    for name,digest in manifest['harness_files'].items():require(binding(ROOT/name)['sha256']==digest,'Unfrozen harness')
    for folder in ('logs','runs','schedules'): (BASE/folder).mkdir(exist_ok=True)
    write(BASE/'controller-identity.json',process(os.getpid()))
    watchdog=subprocess.Popen([sys.executable,str(ROOT/'tools/transfer_eval_watchdog.py')],cwd=ROOT,stdin=subprocess.DEVNULL,
        stdout=(BASE/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(status='preparing',runs=[],private_preservation=[],training_samples=0,ppo_updates=0,optimizer_steps=0,watchdog_pid=watchdog.pid)
    result_path=BASE/'result.json';write(result_path,value)
    try:
        preparation=[BASE/f'restore-{u}' for u in (146,254,412)]+[BASE/'archived-cold-strict',BASE/'archived-accelerated-strict']
        preparation+=[p for p in BASE.iterdir() if p.is_file() and (p.suffix=='.log' or p.name.startswith('ldd-'))]
        value['private_preservation'].append(preserve('immutable-inputs',[BASE/'inputs',BASE/'independent-checkpoint-backup',BASE/'maps',BASE/'manifest.json',*preparation]))
        status(value,'inputs_preserved_before_first_game');publish_public('Publish frozen transfer preparation and private input acknowledgement')
        require(watchdog.poll() is None,'Independent watchdog is not running')
        now=time.time();mono=time.monotonic();window=dict(started_unix=now,started_monotonic=mono,compute_deadline_unix=now+5100,hard_deadline_unix=now+5400,
            compute_deadline_monotonic=mono+5100,hard_deadline_monotonic=mono+5400,scope='First new game launch intent; imports/launch overhead conservatively included; never reset')
        require(not (BASE/'window.json').exists(),'One clock only');write(BASE/'window.json',window);value['window']=window;event('first_game_clock_frozen',window=window)
        for _ in range(100):
            if (BASE/'watchdog-status.json').exists() and json.loads((BASE/'watchdog-status.json').read_text()).get('window')==window:break
            require(watchdog.poll() is None,'Watchdog stopped before first game');time.sleep(.05)
        else:raise ValueError('Watchdog did not acknowledge the immutable clock')
        gate=manifest['gate_schedule']
        for backend,runtime_key in (('cold-reference','cold'),('accelerated-embedded','accelerated')):
            for seat in (0,1):
                rows=[r for r in gate if r['backend']==backend and r['seat']==seat]
                cell=worker(manifest,rows,f'gate-{backend}-seat{seat}',runtime_key,'embedded' if runtime_key=='accelerated' else 'cold',window)
                value['runs'].append(cell);write(result_path,value);status(value,'linux_acceleration_gate')
                if runtime_key=='cold':require(cell['status']=='passed','Independent cold reference failed; no transfer may run')
        value['gate_comparison']=gate_comparison(value['runs'])
        strict=manifest['archived_strict']['cold']['passed'] and manifest['archived_strict']['accelerated']['passed']
        accepted=strict and value['gate_comparison']['passed'];value['gate_status']='passed' if accepted else 'failed_acceleration_use_accepted_cold'
        value['selected_backend']='accelerated-embedded' if accepted else 'original-cold'
        event('linux_acceleration_gate_decision',gate_status=value['gate_status'],comparison=value['gate_comparison'],archived_strict=strict)
        value['private_preservation'].append(preserve('linux-gate',[BASE/'runs',BASE/'window.json',BASE/'result.json',BASE/'schedules']))
        status(value,'transfer_running');publish_public('Publish Linux acceleration gate and selected frozen transfer runtime')
        completed_since_archive=[];last_public=time.time();durations=[]
        for row in manifest['transfer_schedule']:
            remaining=min(window['compute_deadline_unix']-time.time(),window['compute_deadline_monotonic']-time.monotonic())
            if remaining<max(90,2*max(durations[-12:],default=30)+30):break
            key='accelerated' if accepted else 'cold';label=row['id']
            status(value,'transfer_running',active_start=row)
            cell=worker(manifest,[row],label,key,'embedded' if accepted else 'cold',window);value['runs'].append(cell)
            durations.append(cell['wrapper_seconds']);completed_since_archive.append(BASE/'runs'/label);write(result_path,value)
            require(cell['status']=='passed','Transfer integrity/technical failure; preserve and stop, never admit or resample')
            if len(completed_since_archive)>=12:
                value['private_preservation'].append(preserve('transfer-through-'+label,completed_since_archive));completed_since_archive=[];write(result_path,value)
            if time.time()-last_public>300:publish_public('Publish bounded frozen transfer progress');last_public=time.time()
        value['status']='completed' if len([x for x in value['runs'] if x['label'].startswith('transfer-')])==192 else 'stopped_safe_budget_reserve'
    except BaseException as exc:value.update(status='stopped_failure',error=f'{type(exc).__name__}: {exc}');event('diagnostic_failure_retained',error=value['error'])
    finally:
        # All remaining work is preservation, with a distinct five-minute reserve.
        value['finished_utc']=utc();write(result_path,value);status(value,'preserving_and_restoring',error=value.get('error'))
        try:
            for bound in manifest['data_bindings'].values():require(binding(bound['path'])==bound,'Original private game Data changed')
            for update in (146,254,412):
                output=BASE/f'post-study-restore-{update}'
                subprocess.run([sys.executable,str(ROOT/'tools/transfer_eval_worker.py'),'--mode','restore','--source',manifest['runtimes']['cold']['source'],
                    '--manifest',str(BASE/'manifest.json'),'--checkpoint',str(update),'--output',str(output)],check=True,timeout=45,stdout=(BASE/f'post-study-restore-{update}.log').open('x'),stderr=subprocess.STDOUT)
            value['post_study_full_restore_verified']=True
            evidence=BASE/'final-evidence';evidence.mkdir(exist_ok=False)
            for name in ('window.json','result.json','events.jsonl','controller-identity.json','watchdog-status.json','manifest.json'):
                if (BASE/name).exists():shutil.copyfile(BASE/name,evidence/name)
            value['private_preservation'].append(preserve('final-all-captures',[BASE/'runs',BASE/'logs',BASE/'deadlines',BASE/'schedules',evidence,
                *[BASE/f'post-study-restore-{u}' for u in (146,254,412)]]))
        except Exception as exc:value['preservation_error']=f'{type(exc).__name__}: {exc}'
        final=analysis(value,manifest);final.update(error=value.get('error'),preservation_error=value.get('preservation_error'),post_study_full_restore_verified=value.get('post_study_full_restore_verified',False))
        write(BASE/'final-report.json',final);write(AUTHOR/'validation/transfer-eval-2026-10-26.json',final)
        write(result_path,value);status(value,'stopped_preserved' if value.get('post_study_full_restore_verified') and not value.get('preservation_error') else 'stopped_preservation_error',
            final_report='validation/transfer-eval-2026-10-26.json',restart_authorized=False,error=value.get('error'),preservation_error=value.get('preservation_error'))
        event('frozen_transfer_stopped',status=value['status'],report=binding(BASE/'final-report.json'),optimizer_steps=0,training_samples=0)
        publish_public('Publish frozen transfer results, failures and restoration',final=True)


if __name__=='__main__':main()
