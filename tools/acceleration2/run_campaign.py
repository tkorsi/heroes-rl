"""Run one benchmark campaign end to end and record its resources.

    run_campaign.py --checkout DIR --runtime NAME --variant B|C --workload easy|original \
        --name CAMPAIGN --first-sequence S --seats 0,1,0,1 [--push-delay SECONDS]

Runs ``bench_campaign.py`` init, freeze, release and launch from the given checkout with its
own interpreter, samples every descendant of the launch through /proc (resident memory,
descriptors, threads, CPU) and writes ``bench-run.json`` into the campaign directory.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

BENCH = Path(os.environ.get('HEROES_BENCH_ROOT', '/home/claude/bench2')).resolve()
DATA = Path(os.environ.get('HEROES_BENCH_DATA', '/home/claude/heroes-private/game-data')).resolve()
TICK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


def children(pid):
    found, frontier = set(), [pid]
    while frontier:
        current = frontier.pop()
        try:
            for task in os.listdir(f'/proc/{current}/task'):
                with open(f'/proc/{current}/task/{task}/children') as f:
                    for child in f.read().split():
                        if int(child) not in found:
                            found.add(int(child)); frontier.append(int(child))
        except OSError:
            continue
    return found


def sample(pid):
    try:
        with open(f'/proc/{pid}/stat') as f: stat = f.read().rsplit(')', 1)[1].split()
        with open(f'/proc/{pid}/cmdline', 'rb') as f: command = f.read().split(b'\0')[:4]
        return dict(pid=pid, rss=int(stat[21]) * PAGE, threads=int(stat[17]), cpu=(int(stat[11]) + int(stat[12])) / TICK,
                    fds=len(os.listdir(f'/proc/{pid}/fd')), command=' '.join(c.decode(errors='replace') for c in command))
    except OSError:
        return None


class Sampler(threading.Thread):
    def __init__(self, root, path, period=0.25):
        super().__init__(daemon=True); self.root, self.path, self.period = root, path, period
        self.stop = threading.Event(); self.peak = {}

    def run(self):
        with open(self.path, 'w') as out:
            while not self.stop.is_set():
                now = time.monotonic(); rows = [r for r in map(sample, [self.root, *children(self.root)]) if r]
                out.write(json.dumps(dict(t=now, total_rss=sum(r['rss'] for r in rows), processes=rows)) + '\n')
                for r in rows:
                    key = r['command'].split(' ')[-1] if 'python' in r['command'] else r['command'].split(' ')[0]
                    peak = self.peak.setdefault(key, dict(rss=0, fds=0, threads=0))
                    for k in peak: peak[k] = max(peak[k], r[k])
                self.stop.wait(self.period)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkout', type=Path, required=True); p.add_argument('--runtime', required=True)
    p.add_argument('--variant', required=True); p.add_argument('--workload', choices=['easy', 'original'], required=True)
    p.add_argument('--name', required=True); p.add_argument('--first-sequence', type=int, required=True)
    p.add_argument('--seats', required=True); p.add_argument('--push-delay', type=float, default=0.0)
    p.add_argument('--hours', type=float, default=2.0)
    p.add_argument('--experiment', action='append', default=[], help='numerical experiment, recorded in the plan')
    a = p.parse_args()
    checkout = a.checkout.resolve(); python = str(checkout/'.venv/bin/python')
    campaign = BENCH/'campaigns'/a.name
    env = {**os.environ, 'HEROES_ARTIFACT_ROOT': str(BENCH), 'HEROES_BENCH_CAMPAIGN': str(campaign),
           'HEROES_BENCH_PUSH_DELAY': str(a.push_delay), 'PYTHONDONTWRITEBYTECODE': '0'}
    tool = str(checkout/'tools/acceleration2/bench_campaign.py')
    steps = {}
    def step(name, *args):
        started = time.monotonic()
        subprocess.run([python, tool, name, *args], cwd=checkout, env=env, check=True)
        steps[name] = time.monotonic() - started
    step('init', '--campaign', str(campaign), '--relocated', str(BENCH/'inputs/relocated-412'),
         '--runtime-bin', str(BENCH/'runtime'/a.runtime/'bin'), '--data', str(DATA), '--maps', str(BENCH/'inputs/maps'),
         '--historical-plans', str(BENCH/'inputs/historical-plans'), '--durable', str(BENCH/'durable'),
         '--variant', a.variant, '--workload', a.workload, '--hours', str(a.hours), *(x for e in a.experiment for x in ('--experiment', e)))
    step('freeze'); step('release')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    process = subprocess.Popen([python, tool, 'launch', '--seats', a.seats, '--first-sequence', str(a.first_sequence)],
                               cwd=checkout, env=env)
    sampler = Sampler(process.pid, campaign/'resources.jsonl'); sampler.start()
    returncode = process.wait(); sampler.stop.set(); sampler.join()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = dict(variant=a.variant, workload=a.workload, experiments=a.experiment, checkout=str(checkout), runtime=a.runtime, seats=a.seats,
                  first_sequence=a.first_sequence, push_delay=a.push_delay, setup_seconds=steps,
                  launch_seconds=time.monotonic()-started, returncode=returncode,
                  child_cpu_seconds=(after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime),
                  child_user_seconds=after.ru_utime-before.ru_utime, child_system_seconds=after.ru_stime-before.ru_stime,
                  child_max_rss_kib=after.ru_maxrss, peaks=sampler.peak,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip())
    (campaign/'bench-run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('variant', 'workload', 'launch_seconds', 'returncode', 'child_cpu_seconds')}))


if __name__ == '__main__':
    main()
