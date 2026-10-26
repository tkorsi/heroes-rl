"""Read-only independent timing/resource/accounting summary of persisted diagnostics."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path('/workspace/Heroes-artifacts/transfer-eval-20261026')


def binding(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def summarize():
    value=json.loads((BASE/'result.json').read_text())
    if value['status'] not in ('completed','stopped_failure','stopped_safe_budget_reserve'):
        raise ValueError('Do not interpret an unfinished study as a final result')
    manifest=json.loads((BASE/'manifest.json').read_text());rows=[];gate=[]
    for cell in value['runs']:
        bound=cell.get('worker_report')
        if bound:
            if binding(bound['path'])!=bound:raise ValueError('Immutable worker report changed')
            saved=json.loads(Path(bound['path']).read_text())
            if saved['rows']!=cell['rows']:raise ValueError('Controller rows differ from worker evidence')
        target=gate if cell['label'].startswith('gate-') else rows
        for row in cell.get('rows',[]):target.append(dict(**row,wrapper_seconds=cell['wrapper_seconds']/max(1,len(cell['rows']))))
    groups={}
    for checkpoint in (146,254,412):
        for scenario in ('easy','original'):
            accepted=[r for r in rows if r['status']=='passed' and r['schedule']['checkpoint']==checkpoint and r['schedule']['scenario']==scenario]
            groups[f'{checkpoint}-{scenario}']=timings(accepted)
    gate_groups={}
    for backend in ('cold-reference','accelerated-embedded'):
        accepted=[r for r in gate if r['status']=='passed' and r['schedule']['backend']==backend]
        gate_groups[backend]=timings(accepted)
    ratios={}
    for field in ('collection_seconds','replay_seconds','audit_seconds','wall_seconds','wrapper_seconds'):
        cold=gate_groups['cold-reference'].get('totals',{}).get(field)
        accelerated=gate_groups['accelerated-embedded'].get('totals',{}).get(field)
        ratios[field]=cold/accelerated if cold and accelerated else None
    frozen={}
    for update in (146,254,412):
        path=BASE/f'post-study-restore-{update}/report.json';proof=json.loads(path.read_text())
        if proof['status']!='passed' or proof['restore_before']!=proof['restore_after']:
            raise ValueError('Final checkpoint restoration proof differs')
        frozen[str(update)]=dict(report=binding(path),restore=proof['restore_after'],independent_backup=proof['independent_backup_restore'])
    verification={}
    for cell in value['runs']:
        if not cell['label'].startswith('gate-accelerated'):continue
        path=BASE/'logs'/(cell['label']+'.log');text=path.read_text()
        verification[cell['label']]=dict(log=binding(path),capture_recomputation_observed='unchanged-view captures confirmed' in text)
    guard=json.loads((BASE/'watchdog-status.json').read_text())
    final=json.loads((BASE/'final-report.json').read_text())
    easy412=sum(c['natural_wins'] for c in final['cells'].values() if c['checkpoint']==412 and c['scenario']=='easy')
    original412=sum(c['natural_wins'] for c in final['cells'].values() if c['checkpoint']==412 and c['scenario']=='original')
    if easy412>=24 and original412<16:
        proposal='The varied easy cohort transfers substantially better than original-map play. A next separately budgeted curriculum should bridge to an owned-town economy and visible opponent farther away, and mix original-start retention checks. Recruitment and transfers remain explicit measured objectives, never shaped rewards.'
    elif easy412<24:
        proposal='Broaden the easy8x8 coordinate/distance distribution before increasing map size or economic complexity; retain both-seat original-start evaluation to monitor forgetting. Any PPO needs a separately authorized budget.'
    else:
        proposal='Retain mixed varied easy and original starts, then preregister new original-map layouts or longer distances before claiming broad generalization. A further training budget must be separately authorized.'
    return dict(schema='heroes-transfer-independent-summary-v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        original_final_report=binding(BASE/'final-report.json'),controller_report=binding(BASE/'result.json'),
        transfer_timings_by_checkpoint_scenario=groups,gate_timings=gate_groups,gate_speed_ratios_cold_over_accelerated=ratios,
        gate_timing_limit='Four games per backend; includes only the matched current-host gate. Wrapper startup/load is separately reported. No PPO throughput claim.',
        capture_recomputation=verification,full_final_restore=frozen,watchdog=guard,
        window=json.loads((BASE/'window.json').read_text()),
        unstarted_transfer_ids=[r['id'] for r in manifest['transfer_schedule'] if r['id'] not in {x['schedule']['id'] for x in rows}],
        natural_outcomes=dict(Counter(r.get('outcome') for r in rows if r['status']=='passed')),
        new_training_samples=0,new_ppo_updates=0,new_adam_steps=0,next_curriculum_proposal=proposal,
        resource_limit='RSS/descriptors/threads measured within at most two embedded gate games and one transfer game per process. Long-run plateau is unproven.',
        scientific_limit='Matched frozen starts are descriptive heldout evidence. Original scenario uses one fixed board; seed variation alone is not new layout generalization.')


def timings(rows):
    totals={field:sum(r.get(field,0) for r in rows) for field in ('collection_seconds','replay_seconds','audit_seconds','wall_seconds','wrapper_seconds')}
    observed=[r[key] for r in rows for key in ('resources_before','resources_after') if key in r]
    return dict(games=len(rows),decisions=sum(r.get('decisions',0) for r in rows),totals=totals,
        means={k:v/len(rows) if rows else None for k,v in totals.items()},
        medians={k:statistics.median([r.get(k,0) for r in rows]) if rows else None for k in totals},
        maximum_observed_resources={k:max((r[k] for r in observed),default=None) for k in ('rss_bytes','descriptors','threads')})


if __name__=='__main__':print(json.dumps(summarize(),indent=2))
