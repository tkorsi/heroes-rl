"""Cloud-owned handoff poller and fixed absolute owned-process stop guard.

The ready marker is evidence to validate, never permission to bypass acceptance.
Only processes explicitly tagged for this campaign are signalled.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

COMPUTE = datetime(2026, 10, 26, 9, 55, tzinfo=timezone.utc).timestamp()
HARD = datetime(2026, 10, 26, 10, tzinfo=timezone.utc).timestamp()
TAG = b'HEROES_OVERNIGHT_OWNER=teacher-20261026'
BRANCH = 'codex/local365-teacher-results'
MARKER = 'handoffs/2026-10-26-local-to-cloud/ready.json'


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def owned():
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and int(p.name) != os.getpid():
            try:
                if TAG in (p/'environ').read_bytes().split(b'\0'):
                    result.append(int(p.name))
            except (OSError, PermissionError):
                pass
    return result


def stop_owned(sig):
    for pid in owned():
        try: os.kill(pid, sig)
        except ProcessLookupError: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--poller', action='store_true')
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    if not args.poller:
        return watchdog(args)
    lock = (args.state/'poller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time(); mono = time.monotonic()
    # A forward wall-clock jump or the initial monotonic budget ends work.
    compute_mono = mono + max(0, COMPUTE-start)
    hard_mono = mono + max(0, HARD-start)
    next_poll = 0; stopped = False
    status = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        compute_deadline_utc='2026-10-26T09:55:00Z', hard_deadline_utc='2026-10-26T10:00:00Z',
        ownership='local', phase='waiting_final_local_handoff', training_started=False)
    atomic(args.state/'poller-status.json',status)
    while time.time() < HARD and time.monotonic() < hard_mono:
        now = time.time()
        if now >= COMPUTE or time.monotonic() >= compute_mono:
            if not stopped:
                atomic(args.state/'stop-launching.json', dict(at_unix=now, reason='absolute_compute_cutoff'))
                stop_owned(signal.SIGTERM); stopped = True
            if now >= COMPUTE+10: stop_owned(signal.SIGKILL)
            status.update(phase='preservation_shutdown', owned_pids=owned())
        elif now >= next_poll:
            next_poll = now + 120
            try:
                subprocess.run(['git','-C',str(args.repository),'fetch','--no-tags','origin',BRANCH],
                    check=True, capture_output=True, timeout=min(75, COMPUTE-now))
                commit = subprocess.check_output(['git','-C',str(args.repository),'rev-parse','FETCH_HEAD'], text=True, timeout=10).strip()
                raw = subprocess.run(['git','-C',str(args.repository),'show',commit+':'+MARKER],capture_output=True,timeout=10)
                status.update(last_poll_unix=now, private_commit=commit, access=True)
                if raw.returncode == 0:
                    candidate = json.loads(raw.stdout)
                    atomic(args.state/'handoff-candidate.json',dict(private_commit=commit,marker=candidate))
                    status.update(phase='handoff_candidate_requires_verification', candidate_available=True)
                else: status.update(marker_available=False)
            except Exception as exc:
                status.update(last_poll_unix=now, poll_error=str(exc))
        status.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), remaining_compute_seconds=max(0,COMPUTE-time.time()))
        atomic(args.state/'poller-status.json',status)
        time.sleep(min(5,max(.05,HARD-time.time())))
    stop_owned(signal.SIGKILL)
    status.update(phase='stopped_absolute_deadline', owned_pids=owned(), restart_authorized=False)
    atomic(args.state/'poller-status.json',status)


def watchdog(args):
    lock = (args.state/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start=time.time(); mono=time.monotonic()
    compute_mono=mono+max(0,COMPUTE-start); hard_mono=mono+max(0,HARD-start)
    status=dict(pid=os.getpid(), phase='armed', compute_deadline_utc='2026-10-26T09:55:00Z',
        hard_deadline_utc='2026-10-26T10:00:00Z', owned_process_marker=TAG.decode())
    atomic(args.state/'supervisor-status.json',status)
    with (args.state/'poller.log').open('a') as log:
        child=subprocess.Popen([os.sys.executable,__file__,'--poller','--state',str(args.state),
            '--repository',str(args.repository)], env={**os.environ,'HEROES_OVERNIGHT_OWNER':'teacher-20261026'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while time.time()<HARD and time.monotonic()<hard_mono:
        if time.time()>=COMPUTE or time.monotonic()>=compute_mono:
            atomic(args.state/'stop-launching.json',dict(reason='absolute_compute_cutoff',at_unix=time.time()))
            stop_owned(signal.SIGTERM)
            if time.time()>=COMPUTE+10: stop_owned(signal.SIGKILL)
            status['phase']='preservation_shutdown'
        status.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),poller_pid=child.pid)
        atomic(args.state/'supervisor-status.json',status)
        time.sleep(min(5,max(.05,HARD-time.time())))
    stop_owned(signal.SIGKILL)
    status.update(phase='stopped_absolute_deadline',owned_pids=owned(),restart_authorized=False)
    atomic(args.state/'supervisor-status.json',status)


if __name__ == '__main__': main()
