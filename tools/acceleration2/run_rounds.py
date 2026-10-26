"""Run a fixed, counterbalanced list of benchmark campaigns one after another.

    run_rounds.py SCHEDULE.json [--runner tools/run_campaign.py] [--only NAME...]

SCHEDULE.json is a list of campaigns, each {name, variant, checkout, runtime, workload,
first_sequence, seats, push_delay?, hours?, experiments?, prefix_helpers?}. Campaigns run
strictly in order, one at a time, each through ``run_campaign.py``; a failed campaign is recorded
and kept, and the next one still runs. Before each campaign the measured checkout must be clean and at the
commit the schedule names (``head``), so nothing changes under a measurement set.
``rounds.jsonl`` next to the schedule records start, end, exit status and the host load.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load():
    with open('/proc/loadavg') as f:
        return f.read().split()[:3]


def main():
    p = argparse.ArgumentParser(); p.add_argument('schedule', type=Path)
    p.add_argument('--runner', type=Path, default=Path(__file__).resolve().with_name('run_campaign.py'))
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    schedule = json.loads(a.schedule.read_text()); log = a.schedule.with_name('rounds.jsonl')
    for item in schedule:
        if a.only and item['name'] not in a.only:
            continue
        checkout = Path(item['checkout'])
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True)
        row = dict(name=item['name'], variant=item['variant'], head=head, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   started=time.monotonic(), load_before=load())
        if dirty or head != item['head']:
            row.update(returncode=None, error='measured checkout is dirty or not at the scheduled commit')
        else:
            command = [sys.executable, str(a.runner), '--checkout', str(checkout), '--runtime', item['runtime'],
                       '--variant', item['variant'], '--workload', item['workload'], '--name', item['name'],
                       '--first-sequence', str(item['first_sequence']), '--seats', item['seats'],
                       '--push-delay', str(item.get('push_delay', 0.0)), '--hours', str(item.get('hours', 2.0)),
                       *(x for e in item.get('experiments', []) for x in ('--experiment', e)),
                       *(('--prefix-helpers', str(item['prefix_helpers'])) if item.get('prefix_helpers') else ())]
            with open(a.schedule.with_name(item['name'] + '.out'), 'w') as out:
                row['returncode'] = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=os.environ).returncode
        row.update(seconds=time.monotonic() - row['started'], load_after=load())
        with open(log, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: row[k] for k in ('name', 'returncode', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
