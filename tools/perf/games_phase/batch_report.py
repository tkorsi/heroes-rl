"""What one timed batch was, from its state folder and its run log:   batch_report.py LABEL [LABEL ...]

For each label (claude-g10/states/LABEL and logs/LABEL.log): the collection time (start of collection until the
100th usable game was in), the PPO time, the wall time of the command, decisions, decisions per second, usable
games, failed games (replaced), games stopped at day 365 and at the decision limit, samples, optimizer steps, the
weights before and after, when the games ended (the 50th, 90th, last), the longest game, and the processor use of
actors and engines over the batch. A second label is compared game by game with the first: the same seeds must give
the same games (day, choices, samples, outcome).
"""
import datetime
import json
import os
import re
import sys

G = os.path.expanduser('~/Projects/Heroes-artifacts/claude-g10')


def load(label):
    rows = [json.loads(line) for line in open(f'{G}/states/{label}/log.jsonl')]
    start = next(r for r in rows if r['event'] == 'start')
    update = next((r for r in rows if r['event'] == 'update'), None)
    games = [r for r in rows if r['event'] == 'game']
    log = open(f'{G}/logs/{label}.log', errors='replace').read()
    real = re.search(r'^\s*([0-9.]+) real', log, re.M)
    t0 = datetime.datetime.fromisoformat(start['time'])
    ends = sorted((datetime.datetime.fromisoformat(g['time']) - t0).total_seconds() for g in games if g['status'] != 'failed')
    cpu = []
    try:
        for line in open(f'{G}/logs/{label}.cpu'):
            parts = line.split()
            cpu.append((float(parts[2]), float(parts[5])))
    except OSError:
        pass
    usable = [g for g in games if g['status'] != 'failed']
    report = {
        'label': label, 'workers': start.get('workers'), 'seed': start.get('seed'),
        'weights_before': (update or {}).get('weights_before', start.get('weights'))[:12] if update or start.get('weights') else None,
        'weights_after': (update or {}).get('weights_after', '')[:12],
        'collect_seconds': (update or {}).get('collect_seconds'), 'ppo_seconds': (update or {}).get('update_seconds'),
        'command_seconds': float(real.group(1)) if real else None,
        'decisions': sum(sum(g['choices']) for g in usable),
        'usable_games': len(usable), 'failed_games_replaced': len(games) - len(usable),
        'stopped_at_day_365': sum(g['status'] == 'truncated' and g['reason'] == 'game_day_limit' for g in usable),
        'stopped_at_decision_limit': sum(g['status'] == 'truncated' and g['reason'] == 'agent_action_limit' for g in usable),
        'red_wins': (update or {}).get('red'), 'blue_wins': (update or {}).get('blue'),
        'samples': (update or {}).get('samples'), 'optimizer_steps': (update or {}).get('optimizer_steps'),
        'mean_day': round(sum(g['day'] for g in usable) / max(1, len(usable)), 1),
        'longest_game_seconds': max((g['seconds'] for g in usable), default=None),
        'longest_game_decisions': max((sum(g['choices']) for g in usable), default=None),
        'game_ended_at': {'50th': ends[49] if len(ends) >= 50 else None, '90th': ends[89] if len(ends) >= 90 else None,
                          'last': ends[-1] if ends else None},
        'actor_cpu_percent_mean': round(sum(p for _, p in cpu) / max(1, len(cpu))) if cpu else None,
        'engine_cpu_percent_mean': round(sum(e for e, _ in cpu) / max(1, len(cpu))) if cpu else None,
    }
    if report['collect_seconds']:
        report['decisions_per_second'] = round(report['decisions'] / report['collect_seconds'], 1)
    return report, {(g['engine_seed'], tuple(g['policy_seeds'])): g for g in usable}


reports = [load(label) for label in sys.argv[1:]]
for report, _ in reports:
    print(json.dumps(report))
if len(reports) > 1:
    base = reports[0][1]
    for report, games in reports[1:]:
        same = sum(1 for key, g in games.items() if key in base and (base[key]['day'], base[key]['choices'], base[key]['samples'],
                                                                        base[key]['outcomes']) == (g['day'], g['choices'], g['samples'], g['outcomes']))
        shared = sum(1 for key in games if key in base)
        differ = [(base[key]['number'], g['number'], base[key]['day'], g['day'], base[key]['choices'], g['choices'])
                  for key, g in games.items() if key in base and (base[key]['day'], base[key]['choices']) != (g['day'], g['choices'])]
        print(f"{report['label']} against {reports[0][0]['label']}: {shared} games with the same seeds, {same} the same in day, "
              f"choices, samples and outcome; differing: {differ[:6]}")
