#!/usr/bin/env python3
"""Compare eight sequential and eight concurrent native diagnostic games.

Run only with an explicit benchmark request. The fixed policy, original map,
schedule, collection bounds and complete verification are identical in both
arms. Fresh training captures exercise the ordinary admission gates; their
in-memory receipts are discarded without calling PPO. No official campaign
write occurs. All subprocesses have
independent absolute deadlines; failed arms are retained and never retried.

Example (from this checkout, after committing the implementation):
    /Users/orsi/Projects/Heroes/.venv/bin/python tools/benchmark_parallel_collection.py

The default study uses eight Red games, because checkpoint 699's Blue games
usually finish too quickly to exercise eight workers. --seat blue selects a
separate eight-game Blue study; it does not replace a failed Red attempt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH_ROOT = Path.home() / 'Projects/Heroes-bench/parallel-eight'
ARTIFACTS = Path.home() / 'Projects/Heroes-artifacts'
PARENT = ARTIFACTS / 'local-script-training/original/run-20261026T181133Z-062cd75a'
PARENT_BATCH = PARENT / 'batches/train-000335-seat1'
POLICY = PARENT_BATCH / 'execution/after-policy.pt'
RESUME = PARENT_BATCH / 'execution/after-resume.pt'
RUNTIME = ARTIFACTS / 'runtime/local-v8-update92-artifact-fix2'
GAME_MAP = RUNTIME / 'maps/hero-management-v2.vmap'
DATA = Path.home() / 'Library/Application Support/vcmi'
POLICY_SHA = '045d48215be54bd2bd3f438a12f2626ffe83b2a53445a6a8623254feeb0ac046'
RESUME_SHA = 'c5f8e601040e738bbda9fee987f996829107dda3737e2954e1cc4ba371c0afcf'
MAP_SHA = 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3'
INTERFACE = 'adventure-capabilities-v1'
TOTAL_SECONDS = 45 * 60
FINAL_RESERVE = 45
ATTEMPT_SECONDS = 600
SCHEMA = 'heroes-parallel-collection-benchmark-v1'


def utc(stamp=None):
    return datetime.fromtimestamp(time.time() if stamp is None else stamp, timezone.utc).isoformat()


def binding(path):
    path = Path(path).expanduser().resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(path, parent):
    """Resolve symlinks and account for case-insensitive macOS paths."""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if parent.exists() and any(p.exists() and os.path.samefile(p, parent)
                               for p in (path, *path.parents)):
        return True
    return (str(path).casefold().rstrip('/') + '/').startswith(
        str(parent).casefold().rstrip('/') + '/')


def reserve_schedule(output, seat):
    """A separate benchmark ledger; official training sequences are untouched."""
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = BENCH_ROOT / 'seed-ledger.json'
    with (BENCH_ROOT / 'seed-ledger.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(ledger.read_text()) if ledger.exists() else {
            'schema': SCHEMA + '-seeds', 'next_study': 0, 'reserved': []}
        require(value.get('schema') == SCHEMA + '-seeds', 'Unknown benchmark seed ledger')
        study = value['next_study']
        require(type(study) is int and 0 <= study < 100_000, 'Benchmark seed namespace exhausted')
        # Dedicated range, disjoint from the 20M/30M easy, 40M/50M original,
        # 60M/70M old benchmark and 80M mac699 diagnostic namespaces.
        engine_base, policy_base = 1_500_000_000 + study * 16, 1_600_000_000 + study * 16
        schedule = [dict(episode=i + 1, phase='training', learner_player=seat,
                         engine_seed=engine_base + i, policy_seed=policy_base + i)
                    for i in range(8)]
        reservation = {'study': study, 'output': str(output), 'reserved_utc': utc(),
                       'schedule': schedule}
        value['next_study'] = study + 1
        value['reserved'].append(reservation)
        replace_json(ledger, value)
        directory = os.open(BENCH_ROOT, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return reservation


def checked_copy(source, target):
    before = binding(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    after = binding(target)
    require((before['sha256'], before['bytes']) == (after['sha256'], after['bytes']),
            'Input copy changed bytes: ' + str(source))
    return after


def pinned_inputs():
    inputs = {'official_policy': binding(POLICY), 'official_resume': binding(RESUME),
              'official_plan': binding(PARENT_BATCH / 'plan.json'),
              'official_report': binding(PARENT_BATCH / 'execution/report.json'),
              'map': binding(GAME_MAP), 'binary': binding(RUNTIME / 'bin/vcmiclient'),
              'replay_binary': binding(RUNTIME / 'bin/vcmireplay')}
    require(inputs['official_policy']['sha256'] == POLICY_SHA
            and inputs['official_resume']['sha256'] == RESUME_SHA
            and inputs['map']['sha256'] == MAP_SHA, 'Pinned checkpoint 699 or original map changed')
    return inputs


def official_state_snapshot():
    """Observe mutable official files without locking or writing any of them."""
    paths = [ARTIFACTS/'local-script-training/machine-original.json']
    paths.extend(ARTIFACTS/'local-script-training/original'/name for name in
                 ('latest-checkpoint.json', 'latest-run.json', 'sequence-ledger.json'))
    return {str(path): binding(path) if path.exists() else None for path in paths}


def clean_source():
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit the complete implementation before running a benchmark')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def check_prepared_engine_sources():
    """Fail before any guarded/native launch if generated engine sources differ."""
    command = [sys.executable, str(ROOT / 'tools/prepare_vcmi.py'), '--check']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=60, check=True)
    return {'command': command, 'returncode': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip(),
            'checked_utc': utc(), 'read_only': True}


def hardware():
    result = {'cpu_count': os.cpu_count(), 'platform': platform.uname()._asdict()}
    if sys.platform == 'darwin':
        for name, key in (('machdep.cpu.brand_string', 'cpu_model'), ('hw.memsize', 'memory_bytes')):
            try:
                value = subprocess.check_output(['sysctl', '-n', name], text=True, timeout=5).strip()
                result[key] = int(value) if key == 'memory_bytes' else value
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                result[key + '_error'] = str(error)
    return result


def parent_rng_fingerprints(torch):
    return {'torch_cpu_sha256': hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
            'python_sha256': hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}


def study_deadlines(total_seconds, not_after_unix=None):
    now_wall, now_mono = time.time(), time.monotonic()
    wall = now_wall + total_seconds
    if not_after_unix is not None:
        require(math.isfinite(not_after_unix), 'Absolute study cutoff must be finite')
        wall = min(wall, not_after_unix)
    require(wall - now_wall > 15, 'Inherited study cutoff leaves no time for launch and cleanup')
    return now_mono + (wall - now_wall), wall


def comparison(serial, parallel):
    from tools.acceleration3.campaign_summary import game_digest
    a, b = serial.get('attempts', []), parallel.get('attempts', [])
    pairs = []
    for index in range(8):
        left = a[index] if index < len(a) else {}
        right = b[index] if index < len(b) else {}
        row = {'schedule_index': index, 'serial_status': left.get('status', 'not_started'),
               'parallel_status': right.get('status', 'not_started')}
        if left.get('status') == right.get('status') == 'passed':
            row['serial_game_digest'] = game_digest(left['episode_directory'])
            row['parallel_game_digest'] = game_digest(right['episode_directory'])
            row['semantic_match'] = row['serial_game_digest'] == row['parallel_game_digest']
        else:
            row['semantic_match'] = None
        pairs.append(row)
    matched = all(row['semantic_match'] is True for row in pairs)
    complete = all(arm.get('status') == 'passed' for arm in (serial, parallel))
    speedups = {}
    if complete:
        for metric in ('collection_seconds', 'verification_seconds', 'verified_batch_seconds'):
            if parallel[metric] > 0:
                speedups[metric] = serial[metric] / parallel[metric]
    return {'pairs': pairs, 'all_semantic_matches': matched, 'both_arms_verified': complete,
            'observed_speedup': speedups,
            'identical_workload_speedup': speedups if complete and matched else None}


def run_study(output, seat, hard_mono, hard_wall):
    """Runs only inside the separately supervised study process."""
    import torch
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.learner import builtin_first_run as pilot
    from experiments.learner.checkpoints import load_policy_checkpoint
    from experiments.learner.contextual_model import weights_sha256
    from experiments.learner.inference import source_identity
    from experiments.learner.parallel_collection import collect_batch

    started = time.monotonic()
    summary = {'schema': SCHEMA, 'started_utc': utc(), 'status': 'running',
               'arms': [dict(name=name, workers=workers, status='not_started', attempts=[
                   dict(episode=i+1, status='not_started') for i in range(8)])
                   for name, workers in (('serial-1', 1), ('parallel-8', 8))],
               'optimizer_updates': 0, 'training_samples_admitted': 0, 'official_writes': False,
               'hard_deadline_utc': utc(hard_wall), 'hard_deadline_unix': hard_wall,
               'limitations': [
                   'One serial arm followed by one concurrent arm; no repetitions or counterbalancing.',
                   'Collection timing includes child startup, checkpoint loading and owned collection cleanup.',
                   'Verification runs sequentially after all collection and is included in complete batch timing.',
                   'Eight games from one seat measure this fixed workload only; no general training-throughput claim.',
                   'Fresh training receipts exercise collection/admission but are discarded without a PPO call.',
                   'Identical seeds do not guarantee identical Nullkiller choices; semantic histories are compared.',
                   'Other workloads on this Mac can affect both timings; no existing process is controlled.']}
    before = None
    try:
        source_commit = clean_source()
        before = pinned_inputs()
        summary['original_inputs_before'] = before
        summary['prepared_engine_source_check'] = json.loads(
            (output / 'launch.json').read_text())['prepared_engine_source_check']
        summary['official_state_before'] = official_state_snapshot()
        policy = checked_copy(POLICY, output / 'input/after-policy.pt')
        resume = checked_copy(RESUME, output / 'input/after-resume.pt')
        game_map = checked_copy(GAME_MAP, output / 'input/hero-management-v2.vmap')
        reservation = reserve_schedule(output, seat)
        for arm in summary['arms']:
            arm['attempts'] = [dict(row, status='not_started') for row in reservation['schedule']]
        torch.set_num_threads(1)
        raw = torch.load(policy['path'], map_location='cpu', weights_only=True)
        loaded = load_policy_checkpoint(policy['path'], expected_source_metadata=raw['source_metadata'],
            allowed_provenance=('approved_builtin_gameplay_v1',), interface=INTERFACE)
        loaded.model.eval().requires_grad_(False)
        expected_weights = loaded.weights_sha256
        source = source_identity(INTERFACE)
        implementation = pilot.implementation()
        for name in ('tools/benchmark_parallel_collection.py', 'tools/acceleration3/campaign_summary.py'):
            implementation[name] = binding(ROOT / name)['sha256']
        runtime = runtime_identity(RUNTIME / 'bin')
        launcher = VcmiLauncher(RUNTIME / 'bin/vcmiclient', DATA, Path(game_map['path']))
        launcher.preflight()
        native = launcher.identity()
        cfg = json.loads((PARENT_BATCH / 'plan.json').read_text())['configuration']
        cfg.update(attempt_wall_seconds=ATTEMPT_SECONDS, max_days=365, ordinary_choice_limit=4960,
                   max_total_choices=5120, reserved_dialog_choices=160, cleanup_reserve_seconds=30,
                   allow_verified_training_prefix=False)
        summary.update(source_commit=source_commit, checkpoint=policy, checkpoint_resume_copy=resume,
                       weights_sha256=expected_weights, native=native, runtime=runtime,
                       machine=hardware(), python=sys.executable,
                       torch=str(torch.__version__), seat=seat, namespace_reservation=reservation)
        preregistered = {**summary, 'schedule': reservation['schedule'], 'configuration': cfg,
                        'order': ['serial-1', 'parallel-8'], 'source': source,
                        'implementation': implementation, 'checkpoint_source': raw['source_metadata'],
                        'compute_deadline_unix': hard_wall - FINAL_RESERVE}
        write_new(output / 'preregistered.json', preregistered)
        replace_json(output / 'summary.json', summary)
        for arm in summary['arms']:
            name, workers = arm['name'], arm['workers']
            require(clean_source() == source_commit, 'Benchmark source changed between arms')
            require(pinned_inputs() == before, 'Pinned input changed between benchmark arms')
            require(min(hard_mono-time.monotonic(), hard_wall-time.time()) > FINAL_RESERVE + 30,
                    'Insufficient remaining study budget to start another arm')
            folder = output / name
            folder.mkdir()
            plan = {'schema': 'heroes-builtin-training-v1-plan', 'mode': 'training',
                    'configuration': {**cfg, 'parallel_collection': {'batch_size': 8, 'workers': workers}},
                    'schedule': reservation['schedule'], 'source': source, 'implementation': implementation,
                    'execution_output': str(folder / 'execution'), 'checkpoint': policy,
                    'checkpoint_source': raw['source_metadata'], 'native': native, 'runtime': runtime,
                    'binary': str(RUNTIME / 'bin/vcmiclient'), 'replay_binary': binding(RUNTIME / 'bin/vcmireplay'),
                    'game_map': game_map['path'], 'data': str(DATA),
                    'compute_deadline_unix': hard_wall - FINAL_RESERVE,
                    'purpose': 'isolated eight-game collection diagnostic; discard fresh receipts without PPO'}
            plan_path = folder / 'plan.json'
            write_new(plan_path, plan)
            arm.update(status='running', started_utc=utc(), plan=binding(plan_path))
            arm_started = time.monotonic()
            batch = None

            def save(index, attempt):
                arm['attempts'][index] = dict(attempt)
                replace_json(folder / 'progress.json', arm)

            try:
                arm['parent_rng_before'] = parent_rng_fingerprints(torch)
                batch = collect_batch(plan_path, arm['plan']['sha256'], loaded.model, loaded.checkpoint,
                    workers=workers, deadline_monotonic=hard_mono-FINAL_RESERVE,
                    deadline_wall=hard_wall-FINAL_RESERVE, save=save)
                arm.update(attempts=list(batch.attempts), collection_seconds=batch.collection_seconds,
                           verification_seconds=batch.verification_seconds,
                           verified_batch_seconds=time.monotonic()-arm_started)
                require(len(batch.attempts) == 8 and all(row['status'] == 'passed' for row in batch.attempts),
                        'An arm did not verify all eight declared games')
                require(len(batch.receipts) == 8 and all(receipt is not None for receipt in batch.receipts),
                        'The collection diagnostic did not verify all eight fresh training receipts')
                require(weights_sha256(loaded.model) == expected_weights,
                        'Frozen model changed during collection or verification')
                arm['parent_rng_after'] = parent_rng_fingerprints(torch)
                arm['parent_rng_unchanged'] = arm['parent_rng_before'] == arm['parent_rng_after']
                require(arm['parent_rng_unchanged'], 'Parallel collection changed coordinator global RNG state')
                arm.update(status='passed', weights_unchanged=True,
                           fresh_training_receipts_discarded=len(batch.receipts),
                           training_samples_admitted=0, optimizer_updates=0)
            except BaseException as error:
                arm.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                           verified_batch_seconds=time.monotonic()-arm_started)
                raise
            finally:
                try:
                    if batch is not None:
                        # Revoke only this arm's process-local capabilities, then
                        # release its potentially large in-memory histories.
                        batch.discard()
                        del batch
                except Exception as error:
                    arm.update(status='failed', discard_error=type(error).__name__ + ': ' + str(error))
                    raise
                finally:
                    arm['finished_utc'] = utc()
                    write_new(folder / 'result.json', arm)
                    replace_json(output / 'summary.json', summary)
        summary['comparison'] = comparison(*summary['arms'])
        summary['status'] = ('passed' if summary['comparison']['all_semantic_matches']
                             else 'verified_nonidentical_games')
    except BaseException as error:
        summary.update(status='failed', error=type(error).__name__ + ': ' + str(error))
    finally:
        try:
            summary['original_inputs_after'] = pinned_inputs()
            summary['original_inputs_unchanged'] = before == summary['original_inputs_after']
            if not summary['original_inputs_unchanged']:
                summary['status'] = 'failed'
        except Exception as error:
            summary.update(status='failed', input_recheck_error=type(error).__name__ + ': ' + str(error))
        try:
            summary['official_state_after'] = official_state_snapshot()
            summary['official_state_unchanged'] = summary.get('official_state_before') == summary['official_state_after']
            if not summary['official_state_unchanged']:
                summary['external_activity_note'] = (
                    'Official state changed while the isolated benchmark ran. A user-launched process '
                    'may have progressed; the benchmark never writes or locks these files. '
                    'Do not claim that official state remained unchanged.')
        except Exception as error:
            summary['official_state_recheck_error'] = type(error).__name__ + ': ' + str(error)
        for arm in summary['arms']:
            if arm['status'] == 'not_started':
                arm['reason'] = 'study_stopped_before_arm'
                for attempt in arm['attempts']:
                    attempt['reason'] = 'study_stopped_before_arm'
        summary.update(finished_utc=utc(), total_seconds=time.monotonic()-started)
        replace_json(output / 'summary.json', summary)
    print(json.dumps({'output': str(output), 'status': summary['status'],
                      'comparison': summary.get('comparison')}, sort_keys=True), flush=True)
    return 0 if summary['status'] == 'passed' else 1


def launch(args):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output = args.output.expanduser().resolve()
    require(inside(output, BENCH_ROOT) and output != BENCH_ROOT.resolve(),
            'Use a fresh child directory of ' + str(BENCH_ROOT))
    require(not inside(output, ARTIFACTS), 'Benchmark output cannot be inside official artifacts')
    require(not output.exists(), 'Never overwrite or resume a previous benchmark attempt')
    require(60 <= args.total_seconds <= TOTAL_SECONDS, 'Study budget must be 60 to 2700 seconds')
    clean_source()
    pinned_inputs()
    prepared_engine_source_check = check_prepared_engine_sources()
    hard_mono, hard_wall = study_deadlines(args.total_seconds, getattr(args, 'not_after_unix', None))
    output.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--study-child', '--output', str(output),
               '--seat', args.seat, '--hard-monotonic', repr(hard_mono), '--hard-wall', repr(hard_wall)]
    write_new(output / 'launch.json', {'schema': SCHEMA+'-launch', 'command': command,
              'created_utc': utc(), 'hard_deadline_unix': hard_wall, 'hard_deadline_monotonic': hard_mono,
              'prepared_engine_source_check': prepared_engine_source_check,
              'total_seconds': args.total_seconds, 'not_after_unix': getattr(args, 'not_after_unix', None),
              'training_samples_admitted': 0, 'optimizer_updates': 0})
    process = None
    status = None
    cleanup = {'verified': False}
    try:
        with (output / 'study.log').open('x') as log:
            process = guarded_popen(command, deadline_monotonic=hard_mono, deadline_wall=hard_wall,
                evidence_path=output/'study-guard.jsonl', cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            status = process.wait(timeout=max(0.01, min(hard_mono-time.monotonic(), hard_wall-time.time())-10))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    finally:
        if process is not None:
            try:
                close_owned_process(process, timeout=3, deadline_monotonic=hard_mono, deadline_wall=hard_wall)
                proofs = []
                for evidence in sorted(output.rglob('*deadline*.jsonl')):
                    groups = cleanup_from_evidence(evidence, timeout=2,
                        deadline_monotonic=hard_mono, deadline_wall=hard_wall)
                    proofs.append({'path': str(evidence), 'groups': groups})
                cleanup = {'verified': True, 'nested_groups': proofs}
            except Exception as error:
                cleanup = {'verified': False, 'error': type(error).__name__ + ': ' + str(error)}
        result = {'schema': SCHEMA+'-guard-result', 'returncode': status, 'cleanup': cleanup,
                  'finished_utc': utc(), 'hard_deadline_unix': hard_wall,
                  'within_hard_deadline': time.time() <= hard_wall}
        write_new(output/'guard-result.json', result)
    print(json.dumps({'output': str(output), **result}, indent=2), flush=True)
    return status if status is not None and cleanup['verified'] and result['within_hard_deadline'] else 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = BENCH_ROOT / (datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    parser.add_argument('--output', type=Path, default=default)
    parser.add_argument('--seat', choices=('red', 'blue'), default='red')
    parser.add_argument('--total-seconds', type=int, default=TOTAL_SECONDS)
    parser.add_argument('--not-after-unix', type=float,
                        help='Optional inherited absolute cutoff; never extends the relative budget')
    parser.add_argument('--study-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--hard-monotonic', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--hard-wall', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.study_child:
        require(args.hard_monotonic is not None and args.hard_wall is not None, 'Study deadlines required')
        return run_study(args.output.resolve(), 0 if args.seat == 'red' else 1,
                         args.hard_monotonic, args.hard_wall)
    require(args.hard_monotonic is None and args.hard_wall is None, 'Only the supervisor declares deadlines')
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
