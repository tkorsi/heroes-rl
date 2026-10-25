#!/usr/bin/env python3
"""Offline local365 campaign accounting, including failures and unstarted games."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.learner.builtin_training_resume import binding,require


def summarize(root):
    session=json.loads((root/'local365/campaign-session.json').read_text())
    controller=json.loads((root/'local365/controller-result.json').read_text())
    campaign=json.loads((root/'local365/campaign-report.json').read_text())
    totals=Counter();by_mode={};behavior={};runs=[];heldout=[];latest=None;days=[]
    for p in sorted((root/'runs').glob('local365-*/plan.json')):
        plan=json.loads(p.read_text());rp=Path(plan['execution_output'])/'report.json'
        require(rp.exists(),'Planned attempt lacks final report')
        report=json.loads(rp.read_text());require(report['status']!='running','Campaign still active')
        require(report['plan_sha256']==binding(p)['sha256'],'Plan/report binding changed')
        mode='gate' if p.parent.name=='local365-update31-gate' else plan['mode']
        counts=by_mode.setdefault(mode,Counter());aggregate=behavior.setdefault(mode,Counter());rows=[]
        for a in report['attempts']:
            row={k:a.get(k) for k in ('episode','learner_player','status','sampled_choices','training_samples','episode_status','termination_reason','error','engine_seed','policy_seed')}
            category=a['status'];diag=None
            if a['status']=='passed':
                require(all(a.get(k) is True for k in ('completion_receipt','cleanup_verified','identity_verified','replay_verified')),'Passed attempt lacks proof')
                color=('red','blue')[a['learner_player']];outcome=a['outcomes'][color]
                category='natural_'+outcome if a.get('episode_status')=='completed' else 'cutoff'
                require(category in ('natural_win','natural_loss','cutoff'),'Unknown passed outcome')
                b=a['behavior'];require(binding(b['path'])==b,'Behavior bytes changed')
                raw=json.loads(Path(b['path']).read_text())['aggregate']
                diag={'last_public_day':raw['last_public_day_max'],
                    'recruitment_completed_actions':raw['recruitment']['completed'],
                    'movement_attempts':raw['movement']['attempts'],'movement_endpoint_changes':raw['movement']['endpoint_changes'],
                    'moves_toward_visible_opponent_town':raw['movement']['targets_visible_opponent_town'],
                    'town_capture_evidence':raw['town_capture_evidence'],
                    'cross_army_operations':raw['army_operations']['across_armies'],
                    'cross_army_net_count_redistribution':raw['army_operations']['observed_cross_army_net_count_redistribution'],
                    'visible_battle_starts':raw['player_visible_battle_starts']}
                aggregate.update({k:v for k,v in diag.items() if k!='last_public_day'})
                if mode=='training':days.append(diag['last_public_day'])
                row['behavior']=b
            row.update(classification=category,diagnostics_only=diag);rows.append(row)
            totals[category]+=1;counts[category]+=1
        entry={'run':p.parent.name,'mode':mode,'status':report['status'],'stop_reason':report.get('stop_reason'),
            'plan':binding(p),'report':binding(rp),'source_commit':plan['source']['git_commit'],
            'wall_seconds':report['execution_wall_seconds'],'optimizer_updates':report['optimizer_updates'],
            'optimizer_steps':report['optimizer_steps'],'training_samples':report['training_samples'],
            'cleanup_verified':report['supervisor_cleanup_verified'],'rows':rows,'ppo':report.get('ppo')}
        runs.append(entry)
        if mode!='training':heldout.append({'update':plan['parent']['progress']['updates'],'run':p.parent.name,'rows':rows})
        if mode=='training' and report['status']=='finished':
            require(report['update_status']=='checkpoint_verified','Completed update lacks checkpoint')
            latest={'run':p.parent.name,'progress':report['cumulative_progress'],
                    'optimizer_steps':plan['parent']['optimizer_steps']+report['optimizer_steps'],
                    'after_checkpoint':report['after_checkpoint'],'plan':binding(p),'report':binding(rp)}
    seconds=controller['stopped_unix']-session['started_unix']
    training=[r for r in runs if r['mode']=='training' and r['status']=='finished']
    since={'updates':sum(r['optimizer_updates'] for r in training),'admitted_training_games':sum(len(r['rows']) for r in training),
        'admitted_decisions':sum(r['training_samples'] for r in training),'adam_steps':sum(r['optimizer_steps'] for r in training)}
    require(latest['progress']=={'updates':31+since['updates'],'rollouts':124+since['admitted_training_games'],'decisions':6776+since['admitted_decisions']},'Cumulative accounting differs')
    require(latest['optimizer_steps']==253+since['adam_steps'],'Adam accounting differs')
    result={'schema':'heroes-local365-final-analysis-v1','session':binding(root/'local365/campaign-session.json'),
        'controller':binding(root/'local365/controller-result.json'),'campaign_report':binding(root/'local365/campaign-report.json'),
        'started_utc':datetime.fromtimestamp(session['started_unix'],timezone.utc).isoformat(),
        'stopped_utc':datetime.fromtimestamp(controller['stopped_unix'],timezone.utc).isoformat(),
        'hard_deadline_utc':datetime.fromtimestamp(session['absolute_deadline_unix'],timezone.utc).isoformat(),
        'elapsed_seconds':seconds,'within_two_hours':controller['within_two_hours'],'controller_cleanup_verified':controller['controller_cleanup_verified'],
        'status':campaign['status'],'restart_authorized':False,'latest_complete':latest,'since_update31':since,
        'totals':dict(totals),'by_mode':{k:dict(v) for k,v in by_mode.items()},'heldout_checks':heldout,
        'behavior_diagnostics':{k:dict(v) for k,v in behavior.items()},
        'training_public_days':{'median':statistics.median(days),'maximum':max(days),'beyond_previous_30_day_limit':sum(d>30 for d in days)},
        'throughput':{'completed_training_batch_seconds':sum(r['wall_seconds'] for r in training),
            'median_batch_seconds':statistics.median(r['wall_seconds'] for r in training),
            'longest_batches':sorted([{'run':r['run'],'seconds':r['wall_seconds'],'samples':r['training_samples']} for r in training],key=lambda r:r['seconds'],reverse=True)[:5]},
        'runs':runs,'limitations':['Fixed held-out seeds are paired regression checks, not independent win-rate samples.',
            'Latest held-out update90 precedes final checkpoint92 by two updates; no games were launched after the technical stop.',
            'Training and evaluation behavior counts have different sample sizes and trajectory lengths.',
            'Recruitment counts actions; redistribution counts per-operation visible net changes, not distinct units.',
            'Failed episode93 is unscored, unverified by independent replay and excluded from all admitted-game behavior statistics.']}
    result['campaign_source_commits']=sorted({r['source_commit'] for r in runs if r['mode']!='gate'})
    result['gate_source_commits']=sorted({r['source_commit'] for r in runs if r['mode']=='gate'})
    result['checkpoint_restoration']=json.loads((root/'local365/final-restoration.json').read_text())
    result['private_evidence']=json.loads((root/'local365/private-final-publication.json').read_text())
    result['interpretation']=[
        'No learner game victory or town capture was verified. All 244 admitted training games and all 26 gate/evaluation games were natural losses.',
        'On the fixed Red check, survival rose from day2 at update31 to days8-9 at updates70-90, with recruitment; Blue remained a day2 loss except its transient day8 check at update50.',
        'These paired fixed-seed behavior changes do not establish general strategic improvement or an independent win-rate estimate.',
        'Three admitted training games lasted beyond the old30-day horizon (days65,66,107), supplying natural-loss outcomes; long trajectories substantially increased PPO processing time.',
        'Failure93 occurred on day62 in the structured informational-notice handler. A preceding player-visible battle_complete won=true event is not a verified game victory. The failed capture supplies zero samples and was not retried.',
        'The latest evaluated checkpoint is update90; final update92 is preserved but was not evaluated after the correctness stop. No automatic restart is authorized.']
    result['failure_diagnostic']={'run':'local365-continuation-093','episode':1,'day':62,
        'reason':'unsupported_interface','native_error':'unsupported_interaction: informational notice',
        'handler':'engine/ExplorerRL/Contextual.cpp:showStructuredInfoDialog',
        'exact_notice_grammar_not_identified':True,'sampled_choices_excluded':1337,
        'training_samples':0,'optimizer_updates':0,'unstarted_games':3,
        'retained_manifest':binding(root/'runs/local365-continuation-093/execution/episode-000001/manifest.json')}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=summarize(a.root);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:r[k] for k in ['status','elapsed_seconds','since_update31','totals','by_mode','training_public_days','behavior_diagnostics']},indent=2))
