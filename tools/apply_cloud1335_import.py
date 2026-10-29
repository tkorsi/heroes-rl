#!/usr/bin/env python3
"""Check, or explicitly install, the reviewed cloud1335 import while Mac is fenced.

The default is read-only. --apply journals old/new bytes before changing the
official selection; failed partial transactions stay fenced for manual review.
No training, signals, seed reservation, historical rewrite or rollback occurs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import cloud_state_import as cloud
from tools import recover_interrupted_training as recovery
from tools import run_local_training as runner

SCHEMA = 'heroes-cloud1335-import-application-v1'
FENCE_SHA256 = 'fa47da0413e261c0118aaf5711519897cbb01caceb987a74c3c35031c0e30322'
FENCE_RELATIVE = 'run-investigations/2026-10-29-1229-sleep/mac-cloud-handoff-fence-20261029T023358Z-9acbf8b0.json'
ORIGIN = 'run-20261026T181133Z-062cd75a'
MOVING = ('latest-run.json', 'latest-checkpoint.json', 'sequence-ledger.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def content_binding(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def safe_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    require(path.resolve() == path and not path.is_symlink(), 'Path contains a symlink: ' + str(path))
    return path


def reviewed_source():
    """Both the operator entry point and the import profile must be committed."""
    cloud.reviewed_profile()
    path = Path(__file__).resolve()
    committed = subprocess.run(['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()],
        cwd=ROOT, capture_output=True, check=True, timeout=10,
        env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_ALLOW_PROTOCOL': '', 'GIT_TERMINAL_PROMPT': '0'}).stdout
    require(committed == path.read_bytes(), 'Apply helper must be exactly committed before use')


def local_process_closure(config):
    """Reject current official launchers/workers without trusting old numeric PIDs."""
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,uid=,command='],
                            capture_output=True, text=True, check=True, timeout=10)
    matches = []
    patterns = (r'(?:^|[/\s])run_local_training\.py(?:\s|$)',
                r'experiments\.learner\.(?:local_script_training|builtin_training)(?:\s|$)',
                r'(?:^|[/\s])(?:continue-original-4720|train-1000-games)\.sh(?:\s|$)',
                r'(?:^|[/\s])linux_cloud_training\.py(?:\s|$)')
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 4)
        require(len(fields) == 5 and all(value.isdigit() for value in fields[:4]), 'Malformed current process inventory')
        pid, ppid, pgid, uid = map(int, fields[:4])
        command = fields[4]
        if pid == os.getpid() or uid != os.getuid():
            continue
        official_path = str(config['artifact_root']) + '/local-script-training/' in command
        native = bool(re.search(r'(?:^|[/\s])vcmi(?:client|server)(?:\s|$)', command) and official_path)
        collector = 'experiments.learner.parallel_collection' in command and official_path
        if native or collector or any(re.search(pattern, command) for pattern in patterns):
            matches.append({'pid': pid, 'parent_pid': ppid, 'process_group': pgid, 'command': command})
    require(not matches, 'Current official training processes remain: ' + json.dumps(matches))
    return {'observed_utc': utc(), 'matching_official_processes': [], 'signals_sent': 0}


def expected_locks(config, fence):
    paths = [str(safe_path(Path(config['artifact_root']) / 'local-script-training/owner.lock')),
             str(safe_path(Path(config['run_root']) / 'owner.lock'))]
    require(fence.get('schema') == 'heroes-mac-cloud-handoff-fence-v1'
            and fence.get('handoff_id') == cloud.HANDOFF and fence.get('uid') == os.getuid()
            and fence.get('current_mode') == '0444' and fence.get('locks_held_through_fence') is True
            and [item['binding']['path'] for item in fence.get('locks', [])] == paths
            and all(item.get('original_mode') == '0644' for item in fence['locks']),
            'Pinned handoff fence identity or original lock modes differ')
    return fence['locks']


def lock_evidence(item, fd=None, mode=0o444):
    path = safe_path(item['binding']['path'])
    current = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino, current.st_uid) == (item['device'], item['inode'], item['uid'])
            and stat.S_IMODE(current.st_mode) == mode and cloud.binding(path) == item['binding'],
            'Handoff lock identity, bytes or mode changed: ' + str(path))
    if fd is not None:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_uid) == (current.st_dev, current.st_ino, current.st_uid),
                'Open lock descriptor was replaced')
    return {'path': str(path), 'device': current.st_dev, 'inode': current.st_ino, 'uid': current.st_uid,
            'mode': format(stat.S_IMODE(current.st_mode), '04o'), 'binding': item['binding']}


def expect_contents(path, previous):
    path = safe_path(path)
    if previous is None:
        require(not path.exists(), 'An import destination already exists: ' + str(path))
    else:
        require(path.is_file() and path.read_bytes() == previous,
                'Official state changed since the reviewed handoff: ' + str(path))


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_install(path, payload, previous):
    """Replace one expected file, or install an absent file without overwriting it."""
    path = safe_path(path)
    expect_contents(path, previous)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-import-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        expect_contents(path, previous)
        if previous is None:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_journal(path, value):
    previous = path.read_bytes() if path.exists() else None
    atomic_install(path, encoded(value), previous)


def restore_lock_mode(fd, mode):
    os.fchmod(fd, mode)
    os.fsync(fd)


def re_fence(opened):
    errors = []
    for item, fd in opened:
        try:
            os.fchmod(fd, 0o444)
            os.fsync(fd)
            lock_evidence(item, fd, 0o444)
        except BaseException as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))
    require(not errors, 'Could not re-establish both Mac fences: ' + json.dumps(errors))


def apply_import(config_path, receipt_path, *, apply=False, output=None):
    require(type(apply) is bool, 'Apply must be explicit')
    reviewed_source()
    config_path, receipt_path = safe_path(config_path), safe_path(receipt_path)
    config_bound, receipt_bound = cloud.binding(config_path), cloud.binding(receipt_path)
    config, receipt = cloud.read(config_path), cloud.read(receipt_path)
    require(config.get('default_profile') == 'original', 'Only the original campaign can be imported')
    root = safe_path(config['run_root'])
    anchor = root / ORIGIN
    marker = root / 'cloud-import.json'
    fence_path = safe_path(Path(config['artifact_root']) / FENCE_RELATIVE)
    fence_bound = cloud.binding(fence_path)
    require(fence_bound['sha256'] == FENCE_SHA256, 'The pinned Mac handoff fence changed')
    fence = cloud.read(fence_path)
    locks = expected_locks(config, fence)
    base = cloud.read(cloud.bound(receipt['base_recovery_receipt']))
    before = base['verification']['original_moving_snapshots']
    require(type(before) is dict and set(before) == set(MOVING)
            and all(type(value) is str for value in before.values()), 'Exact original moving snapshots are required')
    before = {name: value.encode() for name, value in before.items()}
    output_path = journal_dir = None
    if apply:
        require(output is not None, '--apply requires a fresh private output receipt')
        output_path = safe_path(output)
        private = safe_path(Path(config['artifact_root']) / 'run-investigations')
        require(output_path.is_relative_to(private) and output_path.parent.is_dir(),
                'Apply journal and receipt must remain in the private investigation directory')
        expect_contents(output_path, None)
        journal_dir = output_path.with_name(output_path.name + '.journal')
        require(not journal_dir.exists(), 'Apply journal already exists; inspect the prior transaction manually')
    with recovery.existing_campaign_locks(config) as check_locks, ExitStack() as stack:
        opened = []
        for item in locks:
            fd = os.open(item['binding']['path'], os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            lock_evidence(item, fd)
            opened.append((item, fd))
        initial_locks = [lock_evidence(item, fd) for item, fd in opened]
        expect_contents(marker, None)
        for name, contents in before.items():
            expect_contents(root / name, contents)
        process_proof = local_process_closure(config)
        audit = cloud.audit_import(receipt_path, config, anchor,
                                   local_auditor=runner._local_original_continuation, preview=True)
        require(audit['selected_update'] == 1335 and audit['progress']['updates'] == 1335
                and audit['ledger']['next_sequence'] == 975, 'This application is pinned to checkpoint1335 / sequence975')
        require(audit['receipt'] == {**receipt_bound, 'path': str(marker)}, 'Import marker bytes or official path changed')
        import_run = safe_path(audit['import_run'])
        require(import_run == root / 'cloud-import-1335' and not import_run.exists(), 'Import-only directory is not fresh')
        pointers = cloud.moving_pointers(audit)
        require(set(pointers) == set(MOVING), 'Unexpected official moving files')
        summary = {'schema': SCHEMA + '-summary', 'status': 'imported_verified_cloud_state', 'profile': 'original',
                   'imported_checkpoint': 1335, 'latest_progress': audit['progress'],
                   'original_campaign': {'target_games': 5000, 'admitted_games': 2824, 'remaining_games': 2176},
                   'next_sequence': 975, 'cloud_import': audit['receipt'], 'new_local_games': 0,
                   'new_local_optimizer_updates': 0, 'training_launched': False,
                   'historical_cloud_status': 'stopped_failure',
                   'note': 'Imported a separately certified complete cloud checkpoint. This directory is not a training run.'}
        after = {name: encoded(value) for name, value in pointers.items()}
        check_locks()
        for item, fd in opened:
            lock_evidence(item, fd)
        for value in (config_bound, receipt_bound, fence_bound):
            cloud.bound(value)
        for name, contents in before.items():
            expect_contents(root / name, contents)
        expect_contents(marker, None)
        if not apply:
            return {'schema': SCHEMA, 'status': 'check_passed', 'official_state_written': False,
                    'locks_changed': False, 'training_launched': False, 'checkpoint': 1335,
                    'remaining_original_games': 2176, 'next_sequence': 975, 'process_closure': process_proof,
                    'marker': audit['receipt'], 'lock_evidence': initial_locks}
        journal_dir.mkdir(mode=0o700)
        # Persist the journal directory entry before any official pointer can move.
        fsync_directory(journal_dir.parent)
        journal_path = journal_dir / 'transaction.json'
        journal = {'schema': SCHEMA + '-journal', 'created_utc': utc(), 'status': 'prepared',
                   'config': config_bound, 'private_import_receipt': receipt_bound, 'fence': fence_bound,
                   'locks_before': initial_locks, 'process_closure': process_proof,
                   'before': {str(root / name): {'binding': content_binding(value), 'utf8': value.decode()}
                              for name, value in before.items()},
                   'after': {str(root / name): {'binding': content_binding(value), 'utf8': value.decode()}
                             for name, value in after.items()}, 'completed_writes': [],
                   'manual_recovery_rule': 'Partial import remains fenced. Inspect this journal; no automatic rollback or retry.'}
        marker_bytes, summary_bytes = receipt_path.read_bytes(), encoded(summary)
        for path, payload in ((marker, marker_bytes), (import_run / 'summary.json', summary_bytes)):
            journal['before'][str(path)] = None
            journal['after'][str(path)] = {'binding': content_binding(payload), 'utf8': payload.decode()}
        write_journal(journal_path, journal)
        try:
            import_run.mkdir(mode=0o700)
            fsync_directory(import_run.parent)
            # The marker is installed while both locks deny official launches.
            writes = [(marker, marker_bytes, None), (import_run / 'summary.json', summary_bytes, None)]
            writes.extend((root / name, after[name], before[name]) for name in MOVING)
            for path, payload, previous in writes:
                check_locks()
                for item, fd in opened:
                    lock_evidence(item, fd)
                atomic_install(path, payload, previous)
                journal['completed_writes'].append(str(path))
                journal['status'] = 'installing_while_fenced'
                write_journal(journal_path, journal)
            parent, remaining, continuation = runner.original_continuation(anchor, config)
            require(parent == audit['parent'] and remaining == 2176
                    and continuation['admitted_before_session'] == 2824
                    and continuation['remaining_before_session'] == 2176
                    and continuation['next_sequence'] == 975
                    and continuation['seed_ledger_snapshot'] == audit['ledger'],
                    'Installed official continuation differs from verified checkpoint or accounting')
            process_proof = local_process_closure(config)
            for path, payload, _ in writes:
                expect_contents(path, payload)
            for value in (config_bound, receipt_bound, fence_bound):
                cloud.bound(value)
            check_locks()
            for item, fd in opened:
                lock_evidence(item, fd)
            journal.update(status='verified_before_unlock', continuation_remaining=remaining,
                           continuation_next_sequence=975, final_process_closure=process_proof)
            write_journal(journal_path, journal)
            # Restoring launch permission is the final official mutation.
            for item, fd in opened:
                restore_lock_mode(fd, int(item['original_mode'], 8))
            check_locks()
            final_locks = [lock_evidence(item, fd, 0o644) for item, fd in opened]
            journal.update(status='applied', locks_after=final_locks, completed_utc=utc())
            write_journal(journal_path, journal)
            result = {'schema': SCHEMA, 'status': 'applied', 'official_state_written': True,
                      'training_launched': False, 'games_launched': 0, 'optimizer_updates_applied': 0,
                      'historical_evidence_rewritten': False, 'checkpoint': 1335,
                      'remaining_original_games': 2176, 'next_sequence': 975,
                      'journal': cloud.binding(journal_path), 'marker': cloud.binding(marker),
                      'moving_pointers': {name: cloud.binding(root / name) for name in MOVING},
                      'summary': cloud.binding(import_run / 'summary.json'),
                      'locks_before': initial_locks, 'locks_after': final_locks,
                      'process_closure': process_proof, 'verified_before_unlock': True}
            atomic_install(output_path, encoded(result), None)
            return result
        except BaseException as exc:
            # This also covers an exception after restoring only the first mode.
            re_fence(opened)
            journal.update(status='failed_fenced_manual_review_required', error=type(exc).__name__ + ': ' + str(exc),
                           locks_after=[lock_evidence(item, fd) for item, fd in opened], failed_utc=utc())
            try:
                write_journal(journal_path, journal)
            except BaseException:
                pass  # The durable pre-write journal still contains every old/new byte.
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True, help='The exact private import receipt pinned by the committed profile')
    parser.add_argument('--config', type=Path, default=runner.ORIGINAL_CONFIG)
    parser.add_argument('--apply', action='store_true', help='Install verified state and restore recorded Mac lock modes')
    parser.add_argument('--output', type=Path, help='Fresh private application receipt; required with --apply')
    args = parser.parse_args()
    if args.apply and args.output is None:
        parser.error('--apply requires --output')
    result = apply_import(args.config, args.receipt, apply=args.apply, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
