#!/usr/bin/env python3
"""Run the preregistered, sequential, frozen checkpoint-412 diagnostic only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def identity(root):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError(f'Tracked source must be committed before execution: {root}')
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
              for p in files if p and (root / p).is_file()}
    return {'commit': head, 'tracked_file_sha256': hashes}


def process_table():
    text = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,stat=,lstart=,command='], text=True)
    result = {}
    for line in text.splitlines():
        parts = line.split(None, 9)
        if len(parts) == 10 and 'Z' not in parts[3]:
            result[int(parts[0])] = {'pid': int(parts[0]), 'ppid': int(parts[1]),
                'group': int(parts[2]), 'start': ' '.join(parts[4:9]), 'command': parts[9]}
    return result


def owned_processes(child_pid, root, output, wrapper_identity=None):
    table = process_table()
    current = table.get(child_pid)
    found = ({child_pid} if wrapper_identity and current and
             (current['start'], current['command']) ==
             (wrapper_identity['start'], wrapper_identity['command']) else set())
    # Cold games create their own sessions; their unique log/evidence argument
    # also identifies them after an abruptly terminated parent is reparented.
    binaries = [str(root / 'vcmi/build/bin' / name) + ' ' for name in ('vcmiclient', 'vcmireplay', 'vcmiserver')]
    found.update(pid for pid, p in table.items() if str(output) + '/' in p['command']
                 and any(p['command'].startswith(binary) for binary in binaries))
    while True:
        children = {pid for pid, p in table.items() if p['ppid'] in found}
        if children <= found:
            break
        found.update(children)
    return [table[pid] for pid in sorted(found)]


def stop_verified(receipts, sig):
    current = process_table()
    for old in reversed(receipts):
        live = current.get(old['pid'])
        if live and (live['start'], live['command']) == (old['start'], old['command']):
            try:
                os.kill(old['pid'], sig)
            except ProcessLookupError:
                pass


def execute(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    variants = {'base-cold': (args.base.resolve(), 'cold'),
                'accelerated-cold': (args.accelerated.resolve(), 'cold'),
                'accelerated-embedded': (args.accelerated.resolve(), 'embedded')}
    sources = {str(root): identity(root) for root, _ in variants.values()}
    started = time.monotonic()
    deadline = started + args.wall_seconds
    if args.deadline_unix is not None:
        deadline = min(deadline, started + args.deadline_unix - time.time())
    rows = []
    names = list(variants)
    schedule = [{'round': r + 1, 'variant': name, 'seat': seat}
                for r in range(3) for name in names[r:] + names[:r] for seat in (0, 1)]
    plan = {'schema': 'heroes-checkpoint412-acceleration-study-v1', 'started_utc': utc(),
            'wall_seconds': args.wall_seconds, 'compute_deadline_monotonic': deadline,
            'absolute_deadline_unix': args.deadline_unix,
            'schedule': schedule, 'source': sources, 'input': str(args.input.resolve()),
            'training_samples': 0, 'ppo_updates': 0, 'optimizer_steps': 0}
    write(out / 'plan.json', plan)
    result = {'started_utc': plan['started_utc'], 'status': 'running', 'runs': rows,
              'training_samples': 0, 'ppo_updates': 0, 'optimizer_steps': 0}
    write(out / 'result.json', result)
    for entry in schedule:
        name = f"round{entry['round']}-{entry['variant']}-seat{entry['seat']}"
        root, backend = variants[entry['variant']]
        row = dict(entry, name=name, started_utc=utc())
        rows.append(row)
        write(out / 'result.json', result)
        if identity(root) != sources[str(root)]:
            row.update(status='source_changed')
            break
        remaining = deadline - time.monotonic()
        if remaining < 30:
            row.update(status='unstarted_budget')
            break
        # Resolving a venv's Python symlink loses its environment and wheel pins.
        command = [str(args.python.expanduser().absolute()), str(root / 'tools/validate_checkpoint_acceleration.py'),
                   '--input', str(args.input.resolve()), '--backend', backend,
                   '--binary', str(root / 'vcmi/build/bin/vcmiclient'),
                   '--replay-binary', str(root / 'vcmi/build/bin/vcmireplay'),
                   '--data', str(args.data.resolve()), '--seat', str(entry['seat']),
                   '--output', str(out / name), '--episodes', '4',
                   '--engine-seed', '710001', '--policy-seed', '720001',
                   '--row-seconds', '300', '--watchdog', '60']
        env = dict(os.environ)
        env.pop('HEROES_VERIFY_NATIVE_CAPTURE', None)
        verify_capture = entry['round'] == 1 and entry['variant'] != 'base-cold'
        if verify_capture:
            env['HEROES_VERIFY_NATIVE_CAPTURE'] = '1'
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        row.update(command=command, capture_verification=verify_capture)
        wall_start = time.monotonic()
        with (out / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            wrapper_identity = process_table().get(child.pid)
            row.update(pid=child.pid, process_group=child.pid, wrapper_identity=wrapper_identity)
            write(out / 'result.json', result)
            try:
                code = child.wait(timeout=min(remaining, 1350))
            except subprocess.TimeoutExpired:
                owned = owned_processes(child.pid, root, out / name, wrapper_identity)
                row['timeout_owned_processes'] = owned
                stop_verified(owned, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_verified(owned, signal.SIGKILL)
                    child.wait()
                stop_verified(owned, signal.SIGKILL)
                code = child.returncode
                row['timeout'] = True
        row.update(returncode=code, finished_utc=utc(), wall_seconds=time.monotonic() - wall_start)
        leftovers = owned_processes(child.pid, root, out / name, wrapper_identity)
        if leftovers:
            row['unexpected_owned_processes'] = leftovers
            stop_verified(leftovers, signal.SIGTERM)
            time.sleep(1)
            stop_verified(leftovers, signal.SIGKILL)
            code = code or 1
        row['owned_processes_after'] = owned_processes(child.pid, root, out / name, wrapper_identity)
        if row['owned_processes_after']:
            code = code or 1
        receipt = out / name / 'validation-receipt.json'
        report = out / name / 'games' / 'report.json'
        if code or not receipt.exists() or not report.exists():
            row['status'] = 'failed'
            break
        value = json.loads(report.read_text())
        row.update(verdict=value.get('verdict'), summary=value.get('summary'), receipt_sha256=
                   hashlib.sha256(receipt.read_bytes()).hexdigest())
        row['status'] = 'passed' if value.get('verdict') == 'passed' else 'failed'
        if identity(root) != sources[str(root)]:
            row['status'] = 'source_changed'
        write(out / 'result.json', result)
        print(json.dumps({k: row[k] for k in ('name', 'status', 'wall_seconds')}), flush=True)
        if row['status'] != 'passed':
            break
    result.update(finished_utc=utc(), wall_seconds=time.monotonic() - started,
                  status='passed' if len(rows) == len(schedule) and all(r.get('status') == 'passed' for r in rows)
                  else 'stopped_incomplete', unstarted=schedule[len(rows):])
    write(out / 'result.json', result)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'accelerated', 'python', 'input', 'data', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=2700)
    parser.add_argument('--deadline-unix', type=float,
                        help='Optional original absolute deadline, retained for a fresh setup-repair attempt')
    args = parser.parse_args()
    if not 60 <= args.wall_seconds <= 2700:
        parser.error('The diagnostic wall limit must be between 60 and 2700 seconds')
    raise SystemExit(execute(args))
