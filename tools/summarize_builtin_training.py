#!/usr/bin/env python3
"""Compact offline training evidence. Never forwards metrics to the learner."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.learner.builtin_training_metrics import per_episode
from experiments.learner.builtin_training_resume import binding


def summarize(root):
    runs=[]; totals=Counter(); phases={}
    paths=[root/'recovery-2026-10-25-002', *sorted(root.glob('continuation-2026-10-25-*')),
           *sorted(root.glob('heldout-2026-10-25-*'))]
    for folder in paths:
        report_path=folder/'execution/report.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text())
        if report['status']=='running':continue
        plan=json.loads((folder/'plan.json').read_text())
        rows=[]
        for a in report['attempts']:
            seat=('red','blue')[a['learner_player']]
            category=('natural_'+a['outcomes'][seat] if a.get('episode_status')=='completed' else 'cutoff') if a['status']=='passed' else a['status']
            totals[category]+=1
            phase=phases.setdefault(a['phase'],Counter());phase[category]+=1
            row={k:a.get(k) for k in ('episode','phase','learner_player','engine_seed','policy_seed','status','sampled_choices','training_samples','error','wall_seconds')}
            row['classification']=category
            if a.get('episode_directory'):
                directory=Path(a['episode_directory'])
                metrics_path=directory/'training-metrics.json'
                if not metrics_path.exists():
                    metrics_path.write_text(json.dumps(per_episode(directory),indent=2)+'\n')
                metrics=json.loads(metrics_path.read_text());d=metrics['details']
                if d:
                    actions=d['armies']['actions']
                    row['diagnostics_only']={
                        'recruit_attempts':d['recruitment']['attempts'],
                        'recruit_units':d['recruitment']['accepted_quantity'],
                        'accepted_army_operations':dict(Counter(x['kind'] for x in actions if x['accepted'])),
                        'observed_cross_army_units':sum(v['count'] for x in actions for v in (x.get('effect',{}).get('observed_cross_army_net_redistribution') or [])),
                        'town_loss_days':[x['day'] for x in d['towns']['ownership_losses']],
                        'corroborated_town_captures':len(d['towns']['corroborated_captures']),
                        'terminal_day':None if not d['terminal'] else d['terminal']['day'],
                        'max_observed_army_units':max((x['maximum_observed']['creature_count'] for x in d['armies']['totals_by_subject']),default=None),
                    }
                row['metrics']=binding(metrics_path)
            rows.append(row)
        after=report.get('after_checkpoint')
        runs.append({'run':folder.name,'source_commit':plan['source']['git_commit'],'status':report['status'],
            'stop_reason':report.get('stop_reason'),'plan':binding(folder/'plan.json'),'report':binding(report_path),
            'wall_seconds':report['execution_wall_seconds'],'optimizer_updates':report['optimizer_updates'],
            'optimizer_steps':report['optimizer_steps'],'training_samples':report['training_samples'],
            'cumulative_progress':report.get('cumulative_progress'), 'ppo':report.get('ppo'),
            'after_checkpoint':after,'cleanup_verified':report['supervisor_cleanup_verified'],'attempts':rows})
    return {'schema':'heroes-builtin-continuation-summary-v1','root':str(root),
            'scope':'New recovery-002 lineage only; failed recovery-001 remains separately recorded and unscored.',
            'totals':dict(totals),'phases':{k:dict(v) for k,v in phases.items()},
            'optimizer_updates':sum(x['optimizer_updates'] or 0 for x in runs),
            'optimizer_steps':sum(x['optimizer_steps'] or 0 for x in runs),
            'training_samples':sum(x['training_samples'] for x in runs),
            'execution_wall_seconds':sum(x['wall_seconds'] for x in runs),'runs':runs,
            'limitations':['Fixed repeated held-out seeds are not independent fresh evaluations.',
                'Recruitment, troop counts, transfer and town timing are diagnostics only, never rewards.',
                'Raw artifacts and exact resumable checkpoints are retained outside managed worktrees.',
                'Observed army counts cover displayed own rows only; they do not measure combat strength.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.root.resolve())
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('totals','optimizer_updates','optimizer_steps','training_samples','execution_wall_seconds')},indent=2))
