"""Independent fixed diagnostic stop using kernel identities and owned ancestry."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE=Path('/workspace/Heroes-artifacts/transfer-eval-20261026')


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=int(pid),parent_pid=int(fields[1]),start_ticks=int(fields[19]),command=[s.decode() for s in (path/'cmdline').read_bytes().split(b'\0') if s])
    except (OSError,ValueError):return None


def same(left,right):return right is not None and all(left[k]==right[k] for k in ('pid','start_ticks','command'))


def stop(record,sig):
    if same(record,process(record['pid'])):
        try:os.kill(record['pid'],sig);return True
        except ProcessLookupError:pass
    return False


def compute(command):
    return '--mode' in command and len(command)>command.index('--mode')+1 and command[command.index('--mode')+1]=='run'


def scan(ledger):
    current={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            p=process(int(path.name))
            if p is not None and p['pid']!=os.getpid():current[p['pid']]=p
    owned={pid:p for pid,p in ledger.items() if same(p,current.get(pid))}
    changed=True
    while changed:
        changed=False
        for pid,p in current.items():
            if pid not in owned and p['parent_pid'] in owned:
                parent=owned[p['parent_pid']]
                role='compute' if parent['role']=='compute' or compute(p['command']) else 'preservation'
                owned[pid]=dict(**p,role=role);changed=True
    return owned


def write(path,value):
    temp=path.with_suffix('.guard-tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    lock=(BASE/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=json.loads((BASE/'controller-identity.json').read_text());ledger={root['pid']:dict(root,role='preservation')}
    if not same(root,process(root['pid'])):raise ValueError('Wrong owned controller identity')
    status=dict(pid=os.getpid(),phase='armed_before_first_game',root=root,window=None,signals_sent=0)
    window_raw=None;window=None;seen=set();start=time.monotonic()
    while True:
        if (BASE/'window.json').exists():
            raw=(BASE/'window.json').read_bytes()
            if window_raw is None:
                window_raw=raw;window=json.loads(raw)
                if window['compute_deadline_unix']-window['started_unix']!=5100 or window['hard_deadline_unix']-window['started_unix']!=5400:
                    raise ValueError('Diagnostic window duration changed')
                if window['compute_deadline_monotonic']-window['started_monotonic']!=5100 or window['hard_deadline_monotonic']-window['started_monotonic']!=5400:
                    raise ValueError('Diagnostic monotonic duration changed')
            if raw!=window_raw:raise ValueError('Immutable diagnostic clock was changed')
        ledger=scan(ledger);now=time.time();mono=time.monotonic()
        cutoff=window is not None and (now>=window['compute_deadline_unix'] or mono>=window['compute_deadline_monotonic'])
        hard=window is not None and (now>=window['hard_deadline_unix'] or mono>=window['hard_deadline_monotonic'])
        if window is None and mono-start>3600:hard=True
        for pid,p in sorted(ledger.items(),reverse=True):
            if hard or cutoff and p['role']=='compute':
                force=hard or now>=window['compute_deadline_unix']+5 or mono>=window['compute_deadline_monotonic']+5
                sig=signal.SIGKILL if force else signal.SIGTERM;key=(pid,p['start_ticks'],int(sig))
                if key not in seen and stop(p,sig):
                    with (BASE/'watchdog-signals.jsonl').open('a') as out:out.write(json.dumps(dict(at_unix=now,signal=int(sig),process=p))+'\n')
                    seen.add(key)
        status.update(phase='hard_stop' if hard else 'preservation_shutdown' if cutoff else 'armed' if window else 'armed_before_first_game',
            at_unix=now,window=window,live_owned=list(ledger.values()),signals_sent=len(seen))
        write(BASE/'watchdog-status.json',status)
        if hard or not ledger:
            status.update(phase='stopped_absolute_deadline' if hard else 'completed_early_no_live_owned',live_owned=list(scan(ledger).values()),restart_authorized=False)
            write(BASE/'watchdog-status.json',status);return
        time.sleep(.5)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        owned={}
        try:
            root=json.loads((BASE/'controller-identity.json').read_text())
            if same(root,process(root['pid'])):owned=scan({root['pid']:dict(root,role='preservation')})
            for record in reversed(list(owned.values())):stop(record,signal.SIGKILL)
        finally:
            write(BASE/'watchdog-failure.json',dict(error=f'{type(exc).__name__}: {exc}',owned=list(owned.values()),restart_authorized=False))
        raise
