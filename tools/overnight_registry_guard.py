"""Fixed cutoff for registered owned process trees without /proc/environ access.

PID plus kernel start time and exact command prevent signalling reused PIDs.
Only explicitly registered roots and their observed descendants qualify.
"""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import time

BASE=Path('/workspace/Heroes-artifacts/overnight-20261026')
COMPUTE=1793008500.;HARD=1793008800.


def snapshot(pid):
    root=Path('/proc')/str(pid)
    try:
        fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        command=[p.decode() for p in (root/'cmdline').read_bytes().split(b'\0') if p]
        return dict(pid=int(pid),parent_pid=int(fields[1]),start_ticks=int(fields[19]),command=command)
    except (OSError,ValueError):return None


def same(expected,actual):
    return actual is not None and all(expected[k]==actual[k] for k in ('pid','start_ticks','command'))


def terminate(expected,sig):
    if not same(expected,snapshot(expected['pid'])):return False
    try:os.kill(expected['pid'],sig);return True
    except ProcessLookupError:return False


def atomic(path,value):
    tmp=path.with_suffix('.registry-tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def compute_command(command):
    if len(command)>2 and command[1]=='-m' and command[2]=='experiments.learner.overnight_training':return True
    if len(command)>2 and command[1].endswith('/tools/overnight_curriculum_campaign.py') and command[2]=='worker':return True
    if len(command)>2 and command[1].endswith('/tools/overnight_cloud_supervisor.py') and '--poller' in command:return True
    return False


def scan(ledger):
    live={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row=snapshot(int(path.name))
            if row is not None:live[row['pid']]=row
    # Remove exited identities; never accept a reused PID through old ancestry.
    result={pid:row for pid,row in ledger.items() if same(row,live.get(pid))}
    changed=True
    while changed:
        changed=False
        for pid,row in live.items():
            if pid in result:continue
            parent=result.get(row['parent_pid'])
            if parent is not None:
                role='compute' if parent['role']=='compute' or compute_command(row['command']) else 'preservation'
                result[pid]=dict(**row,role=role,registered_by_parent=parent['pid']);changed=True
    return result


def main():
    lock=(BASE/'registry-guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=json.loads((BASE/'registry-guard-roots.json').read_text());ledger={}
    for row in receipt['roots']:
        if not same(row,snapshot(row['pid'])):raise ValueError('Registered root is no longer the exact live process')
        ledger[row['pid']]=row
    now=time.time();mono=time.monotonic();compute_mono=mono+max(0,COMPUTE-now);hard_mono=mono+max(0,HARD-now)
    status=dict(pid=os.getpid(),command=[os.sys.executable,*os.sys.argv],phase='armed',
        compute_deadline_utc='2026-10-26T09:55:00Z',hard_deadline_utc='2026-10-26T10:00:00Z',
        identity='kernel PID+start_ticks+exact argv and registered ancestry; no environment reads')
    signalled=set()
    while True:
        now=time.time();mono=time.monotonic();ledger=scan(ledger)
        compute=now>=COMPUTE or mono>=compute_mono;hard=now>=HARD or mono>=hard_mono
        for pid,row in sorted(ledger.items(),reverse=True):
            if hard or compute and row['role']=='compute':
                sig=signal.SIGKILL if hard or now>=COMPUTE+10 or mono>=compute_mono+10 else signal.SIGTERM
                key=(pid,row['start_ticks'],int(sig))
                if key not in signalled and terminate(row,sig):
                    with (BASE/'registry-guard-signals.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),signal=int(sig),process=row))+'\n')
                    signalled.add(key)
        status.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase='hard_stop' if hard else 'preservation_shutdown' if compute else 'armed',
            live_owned=list(ledger.values()),signals_sent=len(signalled))
        atomic(BASE/'registry-guard-status.json',status)
        if hard:
            # Recheck once after signals; zombies are stopped, not live compute.
            time.sleep(.2);status.update(phase='stopped_absolute_deadline',live_owned=list(scan(ledger).values()),restart_authorized=False)
            atomic(BASE/'registry-guard-status.json',status);return
        time.sleep(min(2,max(.05,HARD-time.time())))


if __name__=='__main__':main()
