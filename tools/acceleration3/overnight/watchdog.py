"""Independent owned deadline watchdog for the overnight optimization assignment.

    watchdog.py run            # start (refuses to change recorded deadlines)
    watchdog.py own PID LABEL  # register a benchmark root process by verified identity
    watchdog.py status         # print the recorded identity, deadlines and heartbeat

Deadlines are fixed: heavy work stops at 2026-10-27T10:50:00Z and the assignment ends at
2026-10-27T11:00:00Z. They are recorded once, with the equivalent CLOCK_MONOTONIC values, and
a restart may never move them. At the heavy-work deadline the watchdog terminates only
registered roots whose pid and kernel start time still match, together with every descendant
and their process groups (SIGTERM, then SIGKILL after 10 s), verifies that none remain and writes
a stop receipt. It never signals itself, its parent session or any unregistered process.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone

HOME = Path(__file__).resolve().parent
STATE, BEAT, OWNED = HOME/'watchdog.json', HOME/'watchdog-heartbeat.json', HOME/'owned'
HEAVY_UTC, HARD_UTC = '2026-10-27T10:50:00Z', '2026-10-27T11:00:00Z'


def unix(text):
    return datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=1) + '\n'); os.replace(tmp, path)


def identity(pid):
    try:
        with open(f'/proc/{pid}/stat') as f: fields = f.read().rsplit(')', 1)[1].split()
        with open(f'/proc/{pid}/cmdline', 'rb') as f: command = [c.decode(errors='replace') for c in f.read().split(b'\0') if c]
        return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), starttime=int(fields[19]), state=fields[0], command=command)
    except (OSError, IndexError, ValueError):
        return None


def descendants(pid):
    found, frontier = [], [pid]
    while frontier:
        current = frontier.pop()
        try:
            for task in os.listdir(f'/proc/{current}/task'):
                with open(f'/proc/{current}/task/{task}/children') as f:
                    for child in map(int, f.read().split()):
                        if child not in found: found.append(child); frontier.append(child)
        except OSError:
            continue
    return found


def owned_roots():
    roots = []
    for path in sorted(OWNED.glob('*.json')):
        entry = json.loads(path.read_text()); now = identity(entry['pid'])
        entry['alive'] = bool(now and now['starttime'] == entry['starttime'])
        roots.append(entry)
    return roots


def terminate(reason):
    me = os.getpid(); protected = {me, os.getppid(), 1}
    targets = {}
    for root in owned_roots():
        if not root['alive']: continue
        for pid in [root['pid'], *descendants(root['pid'])]:
            info = identity(pid)
            if info and pid not in protected: targets[pid] = dict(info, root=root['label'])
    groups = {t['pgid'] for t in targets.values()} - {os.getpgid(me)}
    sent = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, t in targets.items():
            now = identity(pid)
            if now and now['starttime'] == t['starttime']:
                try: os.kill(pid, sig); sent.append([pid, sig.name])
                except ProcessLookupError: pass
        for g in groups:
            members = [p for p, t in targets.items() if t['pgid'] == g and identity(p)]
            if members:
                try: os.killpg(g, sig)
                except ProcessLookupError: pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any((i := identity(p)) and i['starttime'] == t['starttime'] and i['state'] != 'Z' for p, t in targets.items()):
            time.sleep(0.2)
    remaining = [p for p, t in targets.items() if (i := identity(p)) and i['starttime'] == t['starttime'] and i['state'] != 'Z']
    return dict(reason=reason, utc=utc(), monotonic=time.monotonic(), targets=list(targets.values()), signals=sent,
                remaining=remaining, all_stopped=not remaining)


def run():
    me = identity(os.getpid())
    with open('/proc/sys/kernel/random/boot_id') as f: boot = f.read().strip()
    now_unix, now_mono = time.time(), time.monotonic()
    record = dict(schema='claude-overnight-watchdog-v1', heavy_stop_utc=HEAVY_UTC, hard_stop_utc=HARD_UTC,
                  heavy_stop_unix=unix(HEAVY_UTC), hard_stop_unix=unix(HARD_UTC))
    if STATE.exists():
        old = json.loads(STATE.read_text())
        if any(old[k] != record[k] for k in ('heavy_stop_utc', 'hard_stop_utc')) or old['boot_id'] != boot:
            sys.exit('Recorded deadlines differ or the host rebooted; refusing to start')
        record = old; record.setdefault('restarts', []).append(dict(utc=utc(), pid=me['pid'], starttime=me['starttime']))
    else:
        record.update(boot_id=boot, created_utc=utc(now_unix), created_monotonic=now_mono,
                      heavy_stop_monotonic=now_mono + (unix(HEAVY_UTC) - now_unix),
                      hard_stop_monotonic=now_mono + (unix(HARD_UTC) - now_unix))
    record.update(pid=me['pid'], starttime=me['starttime'], pgid=me['pgid'], command=me['command'], started_utc=utc())
    write(STATE, record)
    stopped = record.get('heavy_stop_receipt') is not None
    while True:
        wall_left = record['hard_stop_unix'] - time.time(); mono_left = record['hard_stop_monotonic'] - time.monotonic()
        heavy_left = min(record['heavy_stop_unix'] - time.time(), record['heavy_stop_monotonic'] - time.monotonic())
        if heavy_left <= 0 and not stopped:
            record['heavy_stop_receipt'] = terminate('heavy-work deadline'); stopped = True; write(STATE, record)
        if min(wall_left, mono_left) <= 0:
            record['hard_stop_receipt'] = terminate('assignment hard stop'); write(STATE, record)
            write(BEAT, dict(utc=utc(), monotonic=time.monotonic(), pid=me['pid'], phase='exited at hard stop'))
            return
        write(BEAT, dict(utc=utc(), monotonic=time.monotonic(), pid=me['pid'], starttime=me['starttime'],
                         seconds_to_heavy_stop=round(heavy_left, 1), seconds_to_hard_stop=round(min(wall_left, mono_left), 1),
                         owned=[{k: r[k] for k in ('label', 'pid', 'alive')} for r in owned_roots()]))
        time.sleep(max(0.5, min(30, heavy_left if heavy_left > 0 else min(wall_left, mono_left))))


def own(pid, label):
    info = identity(int(pid))
    if info is None: sys.exit('No such process')
    entry = dict(label=label, pid=info['pid'], pgid=info['pgid'], starttime=info['starttime'], command=info['command'], registered_utc=utc())
    write(OWNED/f"{label}-{info['pid']}.json", entry); print(json.dumps(entry))


if __name__ == '__main__':
    {'run': lambda: run(), 'own': lambda: own(*sys.argv[2:4]),
     'status': lambda: print(STATE.read_text(), BEAT.read_text() if BEAT.exists() else '')}[sys.argv[1]]()
