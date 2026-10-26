import json, os, subprocess, sys, time
from datetime import datetime, timezone, timedelta
T = sys.argv[1]
fmt = lambda d: (datetime.now(timezone.utc) + timedelta(seconds=d)).strftime('%Y-%m-%dT%H:%M:%SZ')
src = open('/home/claude/overnight/watchdog.py').read().replace(
    "HEAVY_UTC, HARD_UTC = '2026-10-27T10:50:00Z', '2026-10-27T11:00:00Z'", f"HEAVY_UTC, HARD_UTC = '{fmt(6)}', '{fmt(14)}'")
open(f'{T}/watchdog.py', 'w').write(src)
quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
root = subprocess.Popen([f'{T}/dummy.sh'], **quiet)
unowned = subprocess.Popen(['sleep', '303'], **quiet)
time.sleep(0.5)
subprocess.run([sys.executable, f'{T}/watchdog.py', 'own', str(root.pid), 'dummy'], check=True, stdout=subprocess.DEVNULL)
# A stale registration: a process that exited, then a reused pid would have another start time.
gone = subprocess.Popen(['true']); gone.wait()
json.dump(dict(label='stale', pid=os.getpid(), pgid=os.getpgid(0), starttime=1, command=['x'], registered_utc=''), open(f'{T}/owned/stale.json', 'w'))
rc = subprocess.run([sys.executable, f'{T}/watchdog.py', 'run'], timeout=60).returncode
s = json.load(open(f'{T}/watchdog.json')); r = s['heavy_stop_receipt']
print('watchdog rc', rc, 'heavy all_stopped', r['all_stopped'], 'remaining', r['remaining'])
print('targets', sorted((t['pid'], t['command'][-1]) for t in r['targets']))
print('signals', r['signals'])
print('unowned alive', unowned.poll() is None, '| test process alive (stale entry ignored): True')
print('hard all_stopped', s['hard_stop_receipt']['all_stopped'], len(s['hard_stop_receipt']['targets']))
unowned.kill(); root.wait()
