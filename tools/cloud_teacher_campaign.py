#!/usr/bin/env python3
"""Sequential bounded Teacher continuation with private snapshots and fixed checks.

Only completed, restored, privately published batches advance the parent. This
controller never changes policy/PPO math, replaces a failed seed, or resumes an
interrupted update. Each subprocess inherits the independent absolute guardians.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.learner import builtin_training as runner
from experiments.learner.builtin_preservation import DURABLE_ROOT
from experiments.learner.builtin_training_resume import binding, require

STATUS=ROOT/'reports/cloud/teacher-status.json'
PROGRESS=ROOT/'validation/cloud-continuation-2026-10-25.json'


def run_json(command, log_prefix):
    log_prefix.parent.mkdir(parents=True,exist_ok=True)
    with log_prefix.with_suffix('.stderr.log').open('xb') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=stderr,text=True,check=True)
    log_prefix.with_suffix('.stdout.json').write_text(result.stdout)
    return json.loads(result.stdout)


def summarize():
    totals=Counter(); behavior=Counter(); batches=[]; evaluations=[]; latest=None
    for path in sorted((DURABLE_ROOT/'runs').glob('*/plan.json')):
        plan=json.loads(path.read_text());report_path=Path(plan['execution_output'])/'report.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text())
        if (report.get('status')!='finished' or report.get('supervisor_cleanup_verified') is not True
                or report.get('within_shared_budget') is not True):continue
        mode=plan.get('mode')
        training=mode=='training'
        if training and report.get('update_status')!='checkpoint_verified':continue
        rows=[]
        for row in report['attempts']:
            require(row['status']=='passed' and all(row.get(k) is True for k in
                ('completion_receipt','identity_verified','cleanup_verified','replay_verified')), 'Incomplete successful report')
            seat=row['learner_player'];color=('red','blue')[seat]
            outcome=row['outcomes'][color];category=outcome if outcome in ('win','loss') else 'cutoff'
            totals[category]+=1;totals['verified_games']+=1
            totals['training_games' if training else 'evaluation_games']+=1
            diag={}
            if row.get('behavior'):
                bound=row['behavior'];require(binding(bound['path'])==bound,'Behavior evidence changed')
                raw=json.loads(Path(bound['path']).read_text())['aggregate']
                diag={'recruitment_completed':raw['recruitment']['completed'],
                      'cross_army_operations':raw['army_operations']['across_armies'],
                      'cross_army_net_count_redistribution':raw['army_operations']['observed_cross_army_net_count_redistribution'],
                      'moves_toward_visible_opponent_town':raw['movement']['targets_visible_opponent_town'],
                      'town_capture_evidence':raw['town_capture_evidence'],
                      'visible_battle_starts':raw['player_visible_battle_starts']}
                behavior.update(diag)
            rows.append({'seat':seat,'outcome':category,'choices':row['sampled_choices'],'diagnostics':diag})
        entry={'run':path.parent.name,'plan_sha256':binding(path)['sha256'],'report_sha256':binding(report_path)['sha256'],
               'wall_seconds':report['execution_wall_seconds'],'rows':rows}
        if training:
            entry.update(progress=report['cumulative_progress'],optimizer_steps=report['optimizer_steps'],
                         training_samples=report['training_samples'])
            totals['training_samples']+=report['training_samples'];totals['adam_steps']+=report['optimizer_steps']
            totals['updates']+=1
            totals['training_wall_milliseconds']+=round(1000*report['execution_wall_seconds'])
            batches.append(entry)
            if latest is None or report['cumulative_progress']['updates']>latest['progress']['updates']:
                latest={'progress':report['cumulative_progress'],'adam_steps':plan['parent']['optimizer_steps']+report['optimizer_steps'],
                        'after_checkpoint':report['after_checkpoint'],'plan':binding(path),'report':binding(report_path)}
        else:
            entry['at_update']=plan.get('parent',{}).get('progress',{}).get('updates',21)
            evaluations.append(entry)
    seconds=totals.get('training_wall_milliseconds',0)/1000
    throughput={'reported_training_batch_wall_seconds':seconds,
        'admitted_decisions_per_reported_batch_second':totals.get('training_samples',0)/seconds if seconds else None,
        'verified_training_games_per_reported_batch_second':totals.get('training_games',0)/seconds if seconds else None,
        'scope':'Batch wall includes capture, audits, replay, PPO, checkpoint roundtrip and supervised cleanup; excludes preparation and private publication.'}
    return {'schema':'heroes-cloud-continuation-summary-v1','updated_utc':datetime.now(timezone.utc).isoformat(),
        'totals':dict(totals),'throughput':throughput,'behavior_diagnostics':dict(behavior),'batches':batches,'heldout_checks':evaluations,
        'latest_complete':latest,
        'limitations':['Repeated held-out seeds are paired regression checks, not independent win-rate samples.',
                      'Army redistribution counts are observed per-operation changes, not distinct units or strategic quality.',
                      'Visible battles do not establish that the learner initiated an attack.',
                      'Failed and diagnostic cohorts never enter this summary or PPO.']}


def publish_status(state,activity,*,snapshot=None,blocker=None):
    summary=summarize();PROGRESS.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    value=json.loads(STATUS.read_text());latest=summary['latest_complete']
    value.update(updated_utc=datetime.now(timezone.utc).isoformat(),state=state,activity=activity,blocker=blocker)
    if latest:
        p=latest['progress'];after=latest['after_checkpoint']
        value['latest_complete']={'updates':p['updates'],'rollouts':p['rollouts'],'decisions':p['decisions'],
            'adam_steps':latest['adam_steps'],'resume_sha256':after['resume']['sha256'],
            'policy_sha256':after['policy']['sha256'],'weights_sha256':after['weights_sha256'],
            'plan':latest['plan']['path']}
        value['since_update21']={'updates':p['updates']-21,'admitted_rollouts':p['rollouts']-84,
            'admitted_decisions':p['decisions']-3944,'adam_steps':latest['adam_steps']-149}
    value['validated_cloud_outcomes']=summary['totals'];value['throughput']=summary['throughput'];value['behavior_diagnostics']=summary['behavior_diagnostics']
    value['training_performed_in_cloud']=bool(summary['totals'].get('updates'))
    value['heldout_checks']=[{'at_update':x['at_update'],'run':x['run'],'rows':x['rows']} for x in summary['heldout_checks']]
    if snapshot:value['latest_private_snapshot']=snapshot
    value['next_step']='Continue fresh complete batches and required fixed held-out pairs only while sufficient absolute deadline reserve remains.'
    STATUS.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    subprocess.run(['git','add','--',str(PROGRESS.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','add','-f','--',str(STATUS.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',f'Cloud Teacher status: {state}; latest complete update {value["latest_complete"]["updates"]}'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','push','origin','HEAD:refs/heads/codex/cloud-teacher'],cwd=ROOT,check=True,timeout=60)
    return summary


def campaign(parent,sequence):
    logs=DURABLE_ROOT/'session/campaign-logs';durations=[145.0]
    while True:
        report_path=Path(json.loads(parent.read_text())['execution_output'])/'report.json'
        parent_report=json.loads(report_path.read_text());update=parent_report['cumulative_progress']['updates']
        evaluate=False
        if update%5==0:
            parent_plan=json.loads(parent.read_text())
            state=SimpleNamespace(progress=SimpleNamespace(updates=update),bindings={'plan':binding(parent)})
            try:
                runner.require_heldout(state,parent_plan['runtime'],parent_plan['native'])
            except ValueError:
                evaluate=True
        needed=180 if evaluate else max(300,2*max(durations[-5:])+120)
        remaining=runner.COMPUTE_DEADLINE-time.time()
        if remaining<needed:
            publish_status('stopped_for_deadline_reserve',f'No new batch launched: {remaining:.1f}s remains before compute cutoff; safe estimate is{needed:.1f}s.')
            print(json.dumps({'event':'stopped_for_deadline_reserve','latest_update':update,'remaining':remaining}),flush=True)
            return
        name=f'cloud-heldout-update-{update:03d}' if evaluate else f'cloud-continuation-2026-10-25-{sequence:03d}'
        output=DURABLE_ROOT/'runs'/name
        require(not output.exists(),'A planned run already exists; never overwrite or reuse its seeds')
        command=[sys.executable,'-m','experiments.learner.builtin_training','prepare','--parent',str(parent),'--output',str(output)]
        command+=['--evaluate'] if evaluate else ['--sequence',str(sequence)]
        prepared=run_json(command,logs/(name+'-prepare'))
        print(json.dumps({'event':'launched','mode':'evaluation' if evaluate else 'training','run':name,
                         'parent_update':update,'absolute_compute_deadline':runner.COMPUTE_DEADLINE}),flush=True)
        result=run_json([sys.executable,'-m','experiments.learner.builtin_training','execute',
            '--plan',prepared['plan'],'--sha256',prepared['sha256']],logs/(name+'-execute'))
        if result['status']!='finished':
            publish_status('stopped_correctness_or_resource_failure',f'{name} stopped; parent checkpoint retained.',blocker=result.get('stop_reason'))
            print(json.dumps({'event':'stopped_failure','run':name,'result':result}),flush=True)
            return
        snapshot=None
        if not evaluate:
            snapshot=run_json([sys.executable,str(ROOT/'tools/publish_teacher_snapshot.py'),'--plan',prepared['plan']],logs/(name+'-snapshot'))
            parent=Path(prepared['plan']);sequence+=1;durations.append(result['execution_wall_seconds'])
        summary=publish_status('continuing_within_deadline',f'{name} completed and verified; independent backup and private snapshot complete.' if not evaluate else f'{name} fixed held-out pair completed; zero training samples.',snapshot=snapshot)
        print(json.dumps({'event':'completed','run':name,'result':result,'latest_complete':summary['latest_complete']['progress']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--sequence',type=int)
    parser.add_argument('--publish-status',action='store_true')
    args=parser.parse_args()
    if args.publish_status:
        print(json.dumps(publish_status('ready_to_continue','Update22 complete, restore-tested and privately snapshotted; preparing sequence23.'),indent=2))
    else:
        require(args.parent is not None and args.sequence is not None,'Explicit completed parent and unused sequence required')
        campaign(args.parent.resolve(),args.sequence)
