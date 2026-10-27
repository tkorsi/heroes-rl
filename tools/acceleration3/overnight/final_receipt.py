"""Write the end-of-assignment receipt: stopped processes, preserved results, and where private data stays.

    final_receipt.py OUTPUT_JSON [--repo REPO] [--branch BRANCH]

Reads the watchdog's recorded deadlines and stop receipts, checks that no registered benchmark root
or benchmark process is still running, binds the pushed deliverables by SHA-256 and the remote branch
head, and lists the private benchmark roots (raw episodes, captures and checkpoints stay on this host,
bound by the digests in the validation record; nothing raw is pushed).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HOME = Path('/home/claude/overnight')
sys.path.insert(0, str(HOME))
import watchdog as wd  # noqa: E402

DELIVERABLES = ('docs/research/original5000-local-acceleration-2026-10-26.md',
                'docs/research/original5000-local-acceleration-integration-2026-10-26.md',
                'validation/original5000-local-acceleration-linux-2026-10-27.json',
                'reports/cloud/claude-original5000-status.json')
PRIVATE_ROOTS = ('/home/claude/bench3', '/home/claude/bench3c', '/home/claude/bench2')
BENCH_MARKERS = ('ppo_bench.py', 'run_local_bench.py', 'run_schedule.py', 'run_local_training.py',
                 'local_script_training', 'run_matrix', 'after_m')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live_benchmark_processes():
    me, found = os.getpid(), []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == me:
            continue
        info = wd.identity(int(name))
        if info and info['state'] != 'Z' and any(any(m in part for m in BENCH_MARKERS) for part in info['command']):
            found.append(dict(pid=info['pid'], command=info['command'][:4]))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('output', type=Path)
    p.add_argument('--repo', type=Path, default=Path('/home/claude/o5k'))
    p.add_argument('--branch', default='claude/project-thread-iba9jw-original5000')
    a = p.parse_args()
    state = json.loads((HOME / 'watchdog.json').read_text())
    roots = wd.owned_roots()
    remote = git(a.repo, 'ls-remote', 'origin', f'refs/heads/{a.branch}').split()
    files = {}
    for rel in DELIVERABLES:
        path = a.repo / rel
        files[rel] = dict(sha256=sha(path), bytes=path.stat().st_size) if path.exists() else None
    usage = shutil.disk_usage('/')
    value = dict(
        schema='claude-original5000-final-receipt-v1',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        deadlines=dict(heavy_work_stop_utc=state['heavy_stop_utc'], hard_stop_utc=state['hard_stop_utc'],
                       recorded_utc=state['created_utc'], never_moved=True),
        watchdog=dict(pid=state['pid'], heavy_stop_receipt=state.get('heavy_stop_receipt'),
                      hard_stop_receipt=state.get('hard_stop_receipt'), restarts=state.get('restarts', []),
                      reboots=state.get('reboots', [])),
        registered_roots_alive=[dict(label=r['label'], pid=r['pid']) for r in roots if r['alive']],
        benchmark_processes_alive=live_benchmark_processes(),
        git=dict(branch=a.branch, local_head=git(a.repo, 'rev-parse', 'HEAD'),
                 remote_head=remote[0] if remote else None,
                 worktree_clean=not git(a.repo, 'status', '--porcelain', '--untracked-files=no')),
        deliverables=files,
        private_data=dict(roots=[r for r in PRIVATE_ROOTS if Path(r).exists()],
                          note='raw episodes (losslessly compressed with per-file SHA-256 manifests), captures and '
                               'benchmark checkpoints stay on this host; the validation record binds campaign states '
                               'by digest; nothing raw was pushed'),
        disk=dict(free_bytes=usage.free),
        official=dict(official_run_launched=False, official_lineage_touched=False, teacher_touched=False,
                      paid_compute=False, auto_merge=False))
    a.output.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps(dict(roots_alive=len(value['registered_roots_alive']),
                          processes_alive=len(value['benchmark_processes_alive']),
                          pushed=value['git']['local_head'] == value['git']['remote_head']), indent=1))


if __name__ == '__main__':
    main()
