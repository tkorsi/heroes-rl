#!/usr/bin/env python3
"""Run a fixed list of Linux benchmark campaigns strictly one at a time, on the same CPUs.

    run_schedule.py SCHEDULE.json

SCHEDULE.json: {"cpus": "1-3", "updates": 2, "hours": 3.5, "not_after_utc": "...",
                "campaigns": [{"name": ..., "variant": ..., "checkout": ..., "start_args": [...]}, ...]}

Every campaign runs ``run_local_bench.py`` (the manual runner from 629 copies, benchmark namespace,
own run root, first sequence 266, stop requested during the K-th training batch) confined to the
same CPU set with ``taskset``, so every variant gets identical hardware; nothing else heavy is
started on those CPUs. After a campaign the schedule writes its summary (with checkpoint state
digests) and compresses its raw episodes on the remaining CPU. A campaign is not started after
``not_after_utc``; the deadline watchdog independently stops everything at its own deadline.
Refuses a checkout that is dirty or whose HEAD differs from the one recorded for it.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = os.environ.get('HEROES_BENCH_PYTHON', '/home/claude/heroes-rl/.venv/bin/python')
BENCH = Path(os.environ.get('HEROES_LOCAL_BENCH_ROOT', '/home/claude/bench3')).resolve()
WATCHDOG = os.environ.get('HEROES_BENCH_WATCHDOG', '/home/claude/overnight/watchdog.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def head(checkout):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip():
        raise SystemExit(f'Checkout {checkout} is dirty')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()


def main():
    path = Path(sys.argv[1]).resolve()
    schedule = json.loads(path.read_text())
    subprocess.run(['python3', WATCHDOG, 'own', str(os.getpid()), 'schedule-' + path.stem], check=False,
                   stdout=subprocess.DEVNULL)
    heads = {c['checkout']: head(c['checkout']) for c in schedule['campaigns']}
    log_path = path.with_suffix('.log.jsonl')
    not_after = datetime.fromisoformat(schedule['not_after_utc'])

    def log(**row):
        with log_path.open('a') as stream:
            stream.write(json.dumps(dict(utc=utc(), **row)) + '\n')

    log(event='schedule_started', heads=heads, schedule=schedule)
    for campaign in schedule['campaigns']:
        if datetime.now(timezone.utc) >= not_after:
            log(event='not_started_after_cutoff', name=campaign['name']); continue
        if head(campaign['checkout']) != heads[campaign['checkout']]:
            log(event='checkout_moved', name=campaign['name']); break
        command = ['taskset', '-c', schedule['cpus'], PYTHON, str(HERE / 'run_local_bench.py'),
                   '--name', campaign['name'], '--checkout', campaign['checkout'], '--variant', campaign['variant'],
                   '--max-updates', str(schedule['updates']), '--stop-after-updates', str(schedule['updates']),
                   '--hours', str(schedule['hours']), '--', *campaign.get('start_args', [])]
        log(event='campaign_started', name=campaign['name'], command=command)
        started = time.monotonic()
        result = subprocess.run(command, cwd=BENCH, capture_output=True, text=True)
        log(event='campaign_finished', name=campaign['name'], returncode=result.returncode,
            seconds=time.monotonic() - started, stdout=result.stdout[-2000:], stderr=result.stderr[-2000:])
        run_root = BENCH / 'art' / 'runs' / campaign['name']
        out = BENCH / 'art' / 'campaigns' / campaign['name'] / 'summary.json'
        summary = subprocess.run(['taskset', '-c', '0', PYTHON, str(HERE / 'campaign_summary.py'), str(run_root),
                                  '--out', str(out)], capture_output=True, text=True)
        compress = subprocess.run(['taskset', '-c', '0', PYTHON, str(HERE / 'compress_episodes.py'), str(run_root)],
                                  capture_output=True, text=True)
        log(event='campaign_archived', name=campaign['name'], summary_returncode=summary.returncode,
            summary_stderr=summary.stderr[-1000:], compress=compress.stdout[-500:], compress_stderr=compress.stderr[-500:])
    log(event='schedule_finished')


if __name__ == '__main__':
    main()
