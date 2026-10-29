#!/usr/bin/env python3
"""Bounded optimizer diagnostics on privately copied, completed training evidence.

The prepare step audits a completed four-game batch before copying its immutable
inputs. Run reads only those copied learned-state/capture inputs. It has a local
snapshot lease and independent process deadline, never an official campaign lock.
No capability is minted and no result can enter official training or its lineage.
The existing live-input ppo_bench exclusion policy remains unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import cProfile
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path.home()/'Projects/Heroes-artifacts'
SCHEMA = 'heroes-copied-training-diagnostic-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def no_links(path, *, regular=False):
    path = Path(path).expanduser()
    require('..' not in path.parts, 'Parent traversal is forbidden')
    path = path.absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            require(not item.is_symlink(), 'Symlink input or parent is forbidden: '+str(item))
    if regular:
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                'Input must be a private regular file without hard links: '+str(path))
    return path


def binding(path):
    path = no_links(path, regular=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def relative_file(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/')),
            'Unsafe snapshot member name')
    path = no_links(Path(root)/name)
    require(path.is_relative_to(Path(root)), 'Snapshot member escapes its root')
    return path


def private_path(path, *, fresh=False, artifacts=None):
    path = no_links(path)
    artifacts = Path(ARTIFACTS if artifacts is None else artifacts).resolve()
    require(path.is_relative_to(artifacts) and path != artifacts
            and not path.is_relative_to(artifacts/'local-script-training')
            and not path.is_relative_to(artifacts/'releases')
            and not path.is_relative_to(ROOT), 'Diagnostic data must stay outside source and official roots')
    ancestor = path if path.is_dir() else path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    probe = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                           capture_output=True, text=True, timeout=10)
    require(probe.returncode != 0, 'Diagnostic data cannot be inside a Git checkout')
    if fresh:
        require(not path.exists(), 'Diagnostic output already exists')
    return path


def disjoint_output(output, snapshot):
    output, snapshot = no_links(output), no_links(snapshot)
    require(not output.is_relative_to(snapshot) and not snapshot.is_relative_to(output),
            'Diagnostic output and input snapshot must be separate directory trees')


def checkout_identity(path):
    path = no_links(path)
    require(path.is_dir(), 'Selected checkout is missing')
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                     text=True, timeout=10).strip()
    require(not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'],
                                         text=True, timeout=10).strip(),
            'Selected checkout must be clean and committed')
    return dict(path=str(path), commit=commit)


def load_helpers(checkout):
    sys.path.insert(0, str(checkout))
    from tools.acceleration3 import ppo_bench as bench
    from experiments.headless import process_deadline
    require(Path(bench.__file__).resolve().is_relative_to(checkout)
            and Path(bench.ppo.__file__).resolve().is_relative_to(checkout),
            'Benchmark helpers differ from the selected checkout')
    return bench, process_deadline


def copy_verified(source, target, expected=None):
    source = no_links(source, regular=True)
    identity = binding(source)
    if expected is not None:
        require(all(identity[key] == expected[key] for key in ('sha256', 'bytes')),
                'Original capture binding changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        while block := incoming.read(1024*1024):
            outgoing.write(block)
    copied = binding(target)
    require(all(copied[key] == identity[key] for key in ('sha256', 'bytes'))
            and binding(source) == identity, 'Input changed while making the private copy')
    return {'source': identity, 'sha256': copied['sha256'], 'bytes': copied['bytes']}


def prepare_snapshot(args, bench):
    batch = no_links(args.batch)
    plan_path, report_path = batch.parent/'plan.json', batch/'report.json'
    plan = json.loads(no_links(plan_path, regular=True).read_text())
    report = json.loads(no_links(report_path, regular=True).read_text())
    require(report.get('status') == 'finished' and report.get('update_status') == 'checkpoint_verified'
            and report.get('training_performed') is True and report.get('optimizer_updates') == 1
            and report.get('supervisor_cleanup_verified') is True,
            'Snapshot requires a completed, verified four-game update')
    parent = plan['parent_descriptor']
    audit_args = SimpleNamespace(checkout=args.checkout, batch=batch, capture_plan=plan_path,
        capture_report=report_path, parent_plan=Path(parent['plan']['path']),
        parent_report=Path(parent['report']['path']), policy=Path(parent['policy']['path']),
        resume=Path(parent['resume']['path']), require_rolled_back=False)
    # Reject aliases before the historical audit opens receipt members.
    sources = {'payload/capture/plan.json': (plan_path, None),
               'payload/capture/execution/report.json': (report_path, None)}
    for role in ('plan', 'report', 'policy', 'resume'):
        filename = {'plan': 'plan.json', 'report': 'report.json', 'policy': 'policy.pt', 'resume': 'resume.pt'}[role]
        sources['payload/parent/'+filename] = (Path(parent[role]['path']), parent[role])
    require(len(report.get('attempts', [])) == 4, 'Exactly four completed attempts required')
    folders = []
    for attempt in report['attempts']:
        name = Path(attempt['episode_directory']).name
        require(re.fullmatch(r'episode-[0-9]{6}', name) is not None and name not in folders,
                'Invalid or duplicated captured episode')
        folders.append(name)
        folder = no_links(batch/name)
        receipt_path = no_links(folder/'training-receipt.json', regular=True)
        receipt = json.loads(receipt_path.read_text())
        sources['payload/capture/execution/'+name+'/training-receipt.json'] = (receipt_path, None)
        for relative, expected in receipt['files'].items():
            source = relative_file(folder, relative)
            sources['payload/capture/execution/'+name+'/'+relative] = (source, expected)
        require({'manifest.json', 'builtin-training-rollout.json'} <= set(receipt['files']),
                'Receipt must bind the rollout and manifest')
    for source, _ in sources.values():
        no_links(source, regular=True)
    audit = bench.audit_capture(audit_args)
    snapshot_root = args.output_dir/'snapshot'
    snapshot_root.mkdir(mode=0o700)
    inventory = {name: copy_verified(source, snapshot_root/name, expected)
                 for name, (source, expected) in sorted(sources.items())}
    require(bench.audit_capture(audit_args) == audit, 'Completed original capture changed while copying')
    for value in inventory.values():
        require(binding(value['source']['path']) == value['source'], 'Original input changed after copying')
    (snapshot_root/'lease.lock').touch(exist_ok=False)
    snapshot = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_driver=binding(__file__), preparation_checkout=checkout_identity(args.checkout),
        inventory=inventory, audit=audit, files_copied=len(inventory),
        copied_bytes=sum(value['bytes'] for value in inventory.values()),
        original_capture_modified=False, original_inputs_unchanged=True,
        diagnostic_only=True, official_samples_admitted=0, official_updates=0,
        admission_capability_created=False, game_data_copied=False)
    write_new(snapshot_root/'snapshot.json', snapshot)
    verify_snapshot(snapshot_root, binding(snapshot_root/'snapshot.json')['sha256'])
    return dict(snapshot=binding(snapshot_root/'snapshot.json'), **{k: snapshot[k]
        for k in ('files_copied', 'copied_bytes', 'original_inputs_unchanged', 'diagnostic_only')})


def verify_snapshot(root, expected_sha256):
    root = private_path(root)
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or '') is not None, 'Snapshot SHA256 is required')
    require(binding(root/'snapshot.json')['sha256'] == expected_sha256, 'Snapshot manifest changed')
    value = json.loads((root/'snapshot.json').read_text())
    require(value.get('schema') == SCHEMA and value.get('diagnostic_only') is True
            and value.get('official_samples_admitted') == value.get('official_updates') == 0
            and value.get('admission_capability_created') is False and value.get('original_inputs_unchanged') is True,
            'Snapshot is not an audited excluded diagnostic copy')
    files = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            no_links(Path(directory)/name)
        for name in names:
            files.add(no_links(Path(directory)/name, regular=True).relative_to(root).as_posix())
    require(files == set(value['inventory']) | {'snapshot.json', 'lease.lock'},
            'Snapshot contains unbound or missing files')
    for name, expected in value['inventory'].items():
        actual = binding(relative_file(root, name))
        require(all(actual[key] == expected[key] for key in ('sha256', 'bytes')), 'Copied input changed: '+name)
    require((root/'lease.lock').stat().st_size == 0, 'Snapshot lease file changed')
    return value


@contextmanager
def snapshot_lease(root, *, worker=False):
    path = no_links(Path(root)/'lease.lock', regular=True)
    with path.open('rb') as stream:
        identity = os.fstat(stream.fileno())
        fcntl.flock(stream, (fcntl.LOCK_SH if worker else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        if not worker:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            current = path.stat()
            require((current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino),
                    'Snapshot lease identity changed')


def run_copied(args, bench):
    os.nice(max(0, 15-os.getpriority(os.PRIO_PROCESS, 0)))
    snapshot = verify_snapshot(args.snapshot, args.snapshot_sha256)
    capture = args.snapshot/'payload/capture/execution'
    parent = args.snapshot/'payload/parent'
    inputs = SimpleNamespace(parent_plan=parent/'plan.json', parent_report=parent/'report.json',
                             policy=parent/'policy.pt', resume=parent/'resume.pt')
    bench.ppo.EXPERIMENTS = frozenset()
    bench.ppo.PREFIX_HELPERS = 0
    bench.ppo.REPLAY_TEXT_LEAVES = False
    bench.ppo.TEXT_REPLAY_WORKERS = 0
    bench.ppo.NATIVE_EXECUTION = None
    native = None
    if args.native_manifest is not None:
        private_path(args.native_manifest)
        native = binding(args.native_manifest)
        require(native['sha256'] == args.native_sha256, 'Native manifest changed')
        bench.ppo.NATIVE_EXECUTION = dict(manifest=str(args.native_manifest),
            sha256=args.native_sha256, prefix_threads=args.cpu_threads)
    items = bench.rollouts(capture)
    require([len(item.samples) for item in items] == [game['samples'] for game in snapshot['audit']['games']],
            'Copied rollout sample count differs from audited evidence')
    profile = cProfile.Profile() if args.profile else None
    result = bench.run_once(inputs, items, profile)
    state = result.pop('diagnostic_state')
    bench.torch.save(state, args.output_dir/'diagnostic-state.pt')
    for key in ('model', 'optimizer', 'generator'):
        result.pop(key)
    if profile is not None:
        profile.dump_stats(args.output_dir/'profile.prof')
    if args.compare_state is not None:
        private_path(args.compare_state)
        reference_binding = binding(args.compare_state)
        reference = bench.torch.load(args.compare_state, map_location='cpu', weights_only=True)
        result['difference'] = bench.difference(reference, state)
        result['full_state_exact'] = bench.tensor_digest(reference) == bench.tensor_digest(state)
        require(binding(args.compare_state) == reference_binding, 'Reference diagnostic state changed')
        result['reference_state'] = reference_binding
    require(verify_snapshot(args.snapshot, args.snapshot_sha256) == snapshot, 'Snapshot changed during diagnostic')
    if native is not None:
        require(binding(args.native_manifest) == native, 'Native manifest changed during diagnostic')
    return dict(schema=SCHEMA, diagnostic_only=True, official_samples_admitted=0, official_updates=0,
        admission_capability_created=False, no_native_games_launched=True,
        snapshot=binding(args.snapshot/'snapshot.json'), samples=sum(len(item.samples) for item in items),
        native_execution=bench.ppo.NATIVE_EXECUTION, native_manifest=native,
        cpu_budget=args.cpu_threads, nice=os.getpriority(os.PRIO_PROCESS, 0),
        result=result, saved_state=binding(args.output_dir/'diagnostic-state.pt'))


def worker(args):
    identity = checkout_identity(args.checkout)
    driver = binding(__file__)
    copy_verified(__file__, args.output_dir/'diagnostic-driver.py', driver)
    bench, _ = load_helpers(args.checkout)
    bench.torch.set_num_threads(1)
    os.nice(max(0, 15-os.getpriority(os.PRIO_PROCESS, 0)))
    require(min(args.deadline_wall-time.time(), args.deadline_monotonic-time.monotonic()) > 10,
            'Diagnostic worker has no compute budget')
    if args.command == 'prepare':
        result = prepare_snapshot(args, bench)
    else:
        with snapshot_lease(args.snapshot, worker=True):
            result = run_copied(args, bench)
    require(checkout_identity(args.checkout) == identity and binding(__file__) == driver,
            'Selected source or diagnostic driver changed during execution')
    result.update(checkout=identity, driver=driver, official_owner_lock_acquired=False)
    write_new(args.output_dir/'result.json', result)
    return int(result.get('result', {}).get('full_state_exact') is False)


def supervise(args):
    started_wall, started_mono = time.time(), time.monotonic()
    hard_wall, hard_mono = started_wall+args.wall_seconds, started_mono+args.wall_seconds
    identity = checkout_identity(args.checkout)
    driver = binding(__file__)
    args.output_dir = private_path(args.output_dir, fresh=True)
    args.output_dir.mkdir(parents=True, mode=0o700)
    _, deadline = load_helpers(args.checkout)
    command = [sys.executable, '-B', str(Path(__file__).absolute()), *sys.argv[1:], '--worker',
               '--deadline-wall', repr(hard_wall), '--deadline-monotonic', repr(hard_mono)]
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    with (args.output_dir/'worker.log').open('x') as log:
        process = deadline.guarded_popen(command, deadline_wall=hard_wall-10,
            deadline_monotonic=hard_mono-10, evidence_path=args.output_dir/'deadline.jsonl',
            stdout=log, stderr=subprocess.STDOUT, env=environment)
        status, cleanup_error, code = 'failed', None, None
        try:
            code = process.wait(timeout=max(0, min(hard_wall-time.time(), hard_mono-time.monotonic())-10))
            status = 'finished' if code == 0 else 'failed'
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            status = 'timed_out_or_interrupted'
        finally:
            try:
                deadline.close_owned_process(process, timeout=10, deadline_wall=hard_wall,
                                             deadline_monotonic=hard_mono)
            except Exception as error:
                cleanup_error = type(error).__name__+': '+str(error)
        unchanged = checkout_identity(args.checkout) == identity and binding(__file__) == driver
        receipt = dict(status=status, returncode=process.returncode, cleanup_verified=cleanup_error is None,
            cleanup_error=cleanup_error, source_and_driver_unchanged=unchanged, checkout=identity, driver=driver,
            started_unix=started_wall, finished_unix=time.time(), deadline_unix=hard_wall,
            deadline_monotonic=hard_mono, guardian_armed_before_exec=True,
            official_owner_lock_acquired=False, official_samples_admitted=0, official_updates=0)
        write_new(args.output_dir/'shutdown.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if status == 'finished' and cleanup_error is None and unchanged else (code or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--snapshot-sha256')
    parser.add_argument('--native-manifest', type=Path)
    parser.add_argument('--native-sha256')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--wall-seconds', type=float, default=300)
    parser.add_argument('--compare-state', type=Path)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--deadline-wall', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--deadline-monotonic', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(math.isfinite(args.wall_seconds) and 30 <= args.wall_seconds <= 1800,
            'A fixed 30..1800 second deadline is required')
    require(1 <= args.cpu_threads <= min(16, os.cpu_count() or 1), 'CPU budget must fit 1..16 available CPUs')
    require((args.native_manifest is None) == (args.native_sha256 is None), 'Bind the native manifest and its SHA256 together')
    if args.native_sha256 is not None:
        require(re.fullmatch(r'[0-9a-f]{64}', args.native_sha256) is not None, 'Invalid native manifest SHA256')
    args.checkout = no_links(args.checkout)
    args.output_dir = private_path(args.output_dir)
    if args.command == 'prepare':
        require(args.batch is not None and args.snapshot is None and args.snapshot_sha256 is None
                and args.native_manifest is None and args.compare_state is None and not args.profile,
                'Prepare only audits and copies one completed batch')
    else:
        require(args.batch is None and args.snapshot is not None and args.snapshot_sha256 is not None,
                'Run requires the bound private snapshot, never a live batch')
        args.snapshot = private_path(args.snapshot)
        disjoint_output(args.output_dir, args.snapshot)
        verify_snapshot(args.snapshot, args.snapshot_sha256)
        if args.compare_state is not None:
            args.compare_state = private_path(args.compare_state)
            require(not args.compare_state.is_relative_to(args.output_dir),
                    'Reference state cannot be inside a new diagnostic output')
            binding(args.compare_state)
    if args.worker:
        require(all(type(v) is float and math.isfinite(v) and v > 0
                    for v in (args.deadline_wall, args.deadline_monotonic)), 'Worker requires the supervisor deadlines')
        return worker(args)
    require(args.deadline_wall is None and args.deadline_monotonic is None,
            'Only the supervisor selects immutable diagnostic deadlines')
    if args.command == 'run':
        with snapshot_lease(args.snapshot):
            return supervise(args)
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
