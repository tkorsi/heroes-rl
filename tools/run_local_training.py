#!/usr/bin/env python3
"""User-invoked, bounded local PPO. No language model or cloud agent is called."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = Path.home() / 'Projects/Heroes-artifacts/local-script-training/machine.json'
ORIGINAL_CONFIG = DEFAULT_CONFIG.with_name('machine-original.json')
# The default path releases this accepted implementation unchanged. Only an explicit --fast-path
# releases the current commit, whose reviewed exact fast path the session then records.
ACCEPTED_IMPLEMENTATION = 'a8ac411d32658387598376c792136bc586edc5f9'


class StopRequested(Exception):
    pass


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def event(run, kind, **fields):
    with (Path(run) / 'research-log.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp_utc=utc(), event=kind, **fields)) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another local training script owns this artifact root') from exc
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def budget(hours):
    if not math.isfinite(hours) or not 0.25 <= hours <= 48:
        raise ValueError('--hours must be between 0.25 and 48')
    seconds = int(hours * 3600)
    wall, mono = time.time(), time.monotonic()
    return dict(started_unix=wall, started_monotonic=mono,
                compute_deadline_unix=wall + seconds - 300,
                compute_deadline_monotonic=mono + seconds - 300,
                absolute_deadline_unix=wall + seconds,
                hard_deadline_monotonic=mono + seconds)


def left(session, hard=False):
    return min(session['absolute_deadline_unix' if hard else 'compute_deadline_unix'] - time.time(),
               session['hard_deadline_monotonic' if hard else 'compute_deadline_monotonic'] - time.monotonic())


def cleanup_owned(session, process=None, evidence_root=None, awake=None):
    """One aggregate hard budget; failed checks never certify shutdown."""
    from experiments.headless.process_deadline import close_owned_process, cleanup_from_evidence
    limits = dict(deadline_monotonic=session['hard_deadline_monotonic'],
                  deadline_wall=session['absolute_deadline_unix'])
    errors, checked = [], []

    def attempt(label, operation):
        try:
            operation()
            checked.append(label)
        except Exception as error:
            errors.append(dict(check=label, error=f'{type(error).__name__}: {error}'))

    if process is not None:
        # Even an exhausted timeout releases the ownership pipe without blocking.
        attempt('owned_worker', lambda: close_owned_process(process,
            timeout=max(0, min(2, left(session, hard=True))), **limits))
    if awake is not None:
        def stop_awake():
            if awake.poll() is None:
                awake.terminate()
                try:
                    awake.wait(timeout=max(0, min(3, left(session, hard=True))))
                except subprocess.TimeoutExpired:
                    awake.kill()
                    awake.wait(timeout=max(0, min(1, left(session, hard=True))))
        attempt('owned_caffeinate', stop_awake)
    if evidence_root is not None:
        try:
            for path in Path(evidence_root).rglob('*deadline*.jsonl'):
                if left(session, hard=True) <= 0:
                    errors.append(dict(check='remaining_deadline_evidence',
                        error='Hard deadline exhausted; remaining process closure is unverified'))
                    break
                attempt(str(path), lambda path=path: cleanup_from_evidence(path,
                    timeout=max(0, min(2, left(session, hard=True))), **limits))
        except Exception as error:
            errors.append(dict(check='deadline_evidence_inventory', error=f'{type(error).__name__}: {error}'))
    return dict(owned_cleanup_verified=not errors, checked=checked, errors=errors,
                finished_utc=utc(), hard_deadline_exhausted=left(session, hard=True) <= 0)


def enough(session, durations, gamesets=1):
    # Do not begin work likely to be killed; both final seats share one checkpoint.
    return left(session) > gamesets * max(90, 2 * max(durations[-5:], default=60)) + 60


def reserve_sequence(root, floor=266, engine_base=20_000_000, policy_base=30_000_000):
    """Consume before launch, including failed or never-started reserved attempts."""
    path = Path(root) / 'sequence-ledger.json'
    old = read(path) if path.exists() else {'next_sequence': floor, 'consumed': []}
    sequence = max(floor, old['next_sequence'])
    if max(engine_base, policy_base) + 4 * sequence + 3 >= 2**31:
        raise ValueError('Training seed namespace exhausted')
    old['next_sequence'] = sequence + 1
    old['consumed'].append({'sequence': sequence, 'reserved_utc': utc()})
    atomic(path, old)
    return sequence


@contextmanager
def campaign_locks(config):
    # The first easy runner already holds this path. Share it across profiles.
    shared = Path(config['artifact_root']) / 'local-script-training' / 'owner.lock'
    paths = [shared, Path(config['run_root']) / 'owner.lock']
    with ExitStack() as stack:
        for path in dict.fromkeys(p.resolve() for p in paths):
            stack.enter_context(exclusive_lock(path))
        yield


def final_parent(run):
    """Select the final verified state, never the moving latest pointer."""
    run = Path(run).expanduser().resolve()
    session, report, shutdown = [read(run / name) for name in
        ('session.json', 'campaign-report.json', 'shutdown.json')]
    if Path(session['run_dir']).resolve() != run:
        raise ValueError('Parent campaign path differs from its session')
    if (report.get('status') not in ('finished', 'stopped_requested')
            or report.get('owned_cleanup_verified') is not True
            or report.get('initial_state_unchanged') is not True
            or report.get('full_final_restore', {}).get('exact') is not True
            or shutdown.get('owned_cleanup_verified') is not True
            or shutdown.get('worker_returncode') != 0
            or shutdown.get('within_hard_deadline') is not True):
        raise ValueError('Parent campaign is not cleanly finalized with a verified full checkpoint')
    return report['latest_parent']


def release_checkout(artifact_root, commit=None):
    """Freeze code in a separate checkout so editing main cannot change a run.

    ``commit`` selects an accepted ancestor of the current commit; otherwise the current commit.
    """
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('Commit the trainer source before launch; the source checkout is dirty')
    head = git('rev-parse', 'HEAD')
    if commit is None:
        commit = head
    elif subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', commit, head]).returncode:
        raise ValueError('The accepted implementation is not part of this checkout\'s history')
    destination = Path(artifact_root) / 'releases' / ('local-script-' + commit)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(destination), commit], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip()
    if actual != commit or dirty:
        raise ValueError('Existing frozen release differs; no overwrite or repair was attempted')
    # Native evidence requires a real canonical dependency checkout, even when using an existing frozen binary.
    subprocess.run([sys.executable, str(destination/'tools/prepare_vcmi.py')], cwd=destination, check=True, timeout=600)
    subprocess.run([sys.executable, str(destination/'tools/prepare_vcmi.py'), '--check'], cwd=destination, check=True, timeout=120)
    return destination, commit


def verified_batch(report, returncode):
    return (returncode == 0 and report.get('status') == 'finished'
            and all(report.get(k) is True for k in ('worker_complete', 'supervisor_cleanup_verified', 'within_shared_budget'))
            and len(report.get('attempts', [])) == 4
            and all(a.get('status') == 'passed' and all(a.get(k) is True for k in
                    ('completion_receipt', 'cleanup_verified', 'identity_verified', 'replay_verified'))
                    for a in report['attempts']))


def summarize(value):
    training = [b for b in value['batches'] if b.get('checkpoint_verified')]
    evaluation = [b for b in value['batches'] if b['mode'] != 'training' and b.get('status') == 'passed']
    target = value.get('target_games')
    profile = value.get('profile', 'easier8x8')
    return dict(schema='heroes-local-script-summary-v1', timestamp_utc=utc(), status=value['status'],
        profile=profile, target_games=target, target_games_met=None if target is None else 4*len(training) == target,
        remaining_training_games=None if target is None else max(0, target-4*len(training)),
        stop_reason=value.get('stop_reason'),
        new_ppo_updates=len(training), admitted_training_games=4 * len(training),
        training_outcomes=dict(Counter(o for b in training for o in b.get('outcomes', []))),
        evaluation_games=4 * len(evaluation),
        evaluation_outcomes=dict(Counter(o for b in evaluation for o in b.get('outcomes', []))),
        baseline=[b for b in evaluation if b['mode'] == 'baseline'],
        final=[b for b in evaluation if b['mode'] == 'final'],
        failed_batches=[b for b in value['batches'] if b['status'] == 'failed'],
        latest_parent=value['latest_parent'], latest_progress=value.get('latest_progress'),
        error=value.get('error'), owned_cleanup_verified=value.get('owned_cleanup_verified', False),
        initial_state_unchanged=value.get('initial_state_unchanged', False),
        full_final_restore=value.get('full_final_restore'),
        limitations=[('Original hero-management-v2 map; results apply to this authored scenario.' if profile == 'original'
                     else 'Same accepted easier8x8 distribution; this is not an original-map experiment.'),
                     'Fixed held-out starts are repeated; evaluations never supply PPO samples.',
                     'Local checkpoint durability is separate from any subsequent private Git publication.'])


def save(run, value):
    atomic(Path(run) / 'campaign-report.json', value)
    atomic(Path(run) / 'summary.json', summarize(value))


def write_markdown(run, summary):
    text = ('# Local training result\n\n'
            f"Status: {summary['status']}\n\n"
            f"New PPO updates: {summary['new_ppo_updates']}. "
            f"Admitted training games: {summary['admitted_training_games']}.\n\n"
            f"Profile: {summary['profile']}. Target: {summary['target_games']}. "
            f"Target met: {summary['target_games_met']}. Stop reason: {summary['stop_reason']}.\n\n"
            f"Training outcomes: {summary['training_outcomes']}. "
            f"Evaluation outcomes: {summary['evaluation_outcomes']}.\n\n"
            f"Last progress: {summary['latest_progress']}.\n\n"
            f"Error: {summary['error'] or 'none'}.\n\n"
            'See summary.json, campaign-report.json, research-log.jsonl and batches/ for the full record. '
            'Only complete verified four-game batches were admitted. Fixed evaluation games are excluded. '
            + summary['limitations'][0] + '\n')
    (Path(run) / 'RESULTS.md').write_text(text)


def execute_batch(session_path, parent, sequence, seat, mode, value):
    from experiments.learner import local_script_training as adapter
    from experiments.learner.builtin_training_resume import binding, require
    from experiments.learner.builtin_preservation import preserve, verify_pair
    from experiments.headless.process_deadline import guarded_popen
    session = read(session_path); run = Path(session['run_dir'])
    if (run / 'stop-requested').exists():
        raise StopRequested('Manual stop requested before the next batch')
    row = dict(sequence=sequence, seat=seat, mode=mode, parent=parent, status='preparing', started_utc=utc())
    value['batches'].append(row); save(run, value)
    event(run, 'batch_intent', **row)
    process = None; plan_path = None
    try:
        plan_path = adapter.prepare_batch(session_path, parent, sequence, seat, evaluate=mode != 'training', label='train' if mode == 'training' else mode)
        row['plan'] = binding(plan_path); row['status'] = 'running'; save(run, value)
        with (plan_path.parent / 'controller.log').open('x') as log:
            process = guarded_popen([sys.executable, '-m', 'experiments.learner.local_script_training',
                'execute', '--plan', str(plan_path), '--sha256', row['plan']['sha256']],
                deadline_monotonic=session['compute_deadline_monotonic'], deadline_wall=session['compute_deadline_unix'],
                evidence_path=plan_path.parent / 'controller-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            rc = process.wait(timeout=max(.1, left(session)))
        report_path = plan_path.parent / 'execution/report.json'; report = read(report_path)
        row.update(report=binding(report_path), completed_games=sum(a['status'] == 'passed' for a in report['attempts']),
                   outcomes=[a.get('outcomes', {}).get(('red', 'blue')[seat]) for a in report['attempts']],
                   optimizer_steps=report.get('optimizer_steps'), training_samples=report.get('training_samples'))
        require(verified_batch(report, rc), 'Batch failed: ' + str(report.get('stop_reason') or report.get('supervisor_error')))
        row['cleanup'] = cleanup_owned(session, process, plan_path.parent)
        row['owned_cleanup_verified'] = row['cleanup']['owned_cleanup_verified']
        require(row['owned_cleanup_verified'], 'Batch owned cleanup unverified: ' + str(row['cleanup']['errors']))
        if mode == 'training':
            require(report['optimizer_updates'] == 1 and report['update_status'] == 'checkpoint_verified', 'Incomplete PPO update')
            backup = run / 'checkpoints' / plan_path.parent.name
            proof = preserve(plan_path, report_path, backup, root=Path(session['artifact_root']))
            plan = read(plan_path)
            verify_pair(backup / 'after-resume.pt', backup / 'after-policy.pt', plan, report)
            parent = adapter.binding_descriptor(plan_path, report_path,
                report['after_checkpoint']['policy']['path'], report['after_checkpoint']['resume']['path'],
                plan['checkpoint_metadata']['initial_checkpoint']['path'])
            row.update(checkpoint_verified=True, preservation=proof)
            value.update(latest_parent=parent, latest_progress=proof['restore']['progress'])
            # Write only after a complete batch and independent copy restore; no interrupted update becomes latest.
            atomic(Path(session['run_root']) / 'latest-checkpoint.json', dict(parent=parent, run_dir=str(run),
                   backup=str(backup), progress=value['latest_progress'], timestamp_utc=utc()))
        else:
            require(report['optimizer_updates'] == report['optimizer_steps'] == report['training_samples'] == 0,
                    'Evaluation supplied training samples')
        row['status'] = 'passed'
    except BaseException as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if 'cleanup' not in row:
            row['cleanup'] = cleanup_owned(session, process, plan_path.parent if plan_path else None)
            row['owned_cleanup_verified'] = row['cleanup']['owned_cleanup_verified']
        if not row['owned_cleanup_verified']:
            row.update(status='failed', cleanup_error=str(row['cleanup']['errors']))
        row['finished_utc'] = utc(); save(run, value); event(run, 'batch_result', **row)
    return parent


def worker(session_path):
    from experiments.learner import local_script_training as adapter
    from experiments.learner.builtin_training_resume import binding
    session = read(session_path); run = Path(session['run_dir']); parent = session['initial_parent']
    if os.getppid() != session['supervisor_pid']:
        raise RuntimeError('Worker must be launched by its live owned supervisor')
    state = adapter.restore_parent(parent)
    value = dict(status='running', batches=[], latest_parent=parent, latest_progress=vars(state.progress),
                 profile=session.get('profile', 'easier8x8'), target_games=session.get('target_games'))
    save(run, value)
    durations = [60.0]
    try:
        for seat in (0, 1):
            execute_batch(session_path, parent, state.progress.updates, seat, 'baseline', value)
        while True:
            updates = sum(b.get('checkpoint_verified', False) for b in value['batches'])
            if (run / 'stop-requested').exists():
                value['stop_reason'] = 'manual_stop'; break
            if session.get('target_games') is not None and 4*updates >= session['target_games']:
                value['stop_reason'] = 'training_game_target_reached'; break
            if session.get('max_updates') is not None and updates >= session['max_updates']:
                value['stop_reason'] = 'update_limit_reached'; break
            if not enough(session, durations, gamesets=3):
                value['stop_reason'] = 'compute_budget_reserve'; break
            sequence = reserve_sequence(session['run_root'], session['next_sequence'],
                session.get('training_engine_seed_base', 20_000_000), session.get('training_policy_seed_base', 30_000_000))
            started = time.monotonic()
            parent = execute_batch(session_path, parent, sequence, sequence % 2, 'training', value)
            durations.append(time.monotonic() - started)
            updates += 1
            print(f"PPO +{updates} | games {4*updates} | checkpoint {value['latest_progress']['updates']} | "
                  f"remaining {max(0, int(left(session)))}s", flush=True)
            reached = session.get('target_games') is not None and 4*updates >= session['target_games']
            if not reached and updates % session.get('evaluation_interval_updates', 5) == 0 and enough(session, durations, gamesets=2):
                for seat in (0, 1):
                    execute_batch(session_path, parent, value['latest_progress']['updates'], seat, 'periodic', value)
        if enough(session, durations, gamesets=2):
            for seat in (0, 1):
                execute_batch(session_path, parent, value['latest_progress']['updates'], seat, 'final', value)
        value['status'] = ('stopped_requested' if (run / 'stop-requested').exists() else
            'stopped_budget' if value.get('stop_reason') == 'compute_budget_reserve' else 'finished')
    except StopRequested:
        value.update(status='stopped_requested', stop_reason='manual_stop')
    except BaseException as error:
        value.update(status='stopped_failure', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            final = adapter.restore_parent(value['latest_parent'])
            value['full_final_restore'] = dict(progress=vars(final.progress), optimizer_steps=final.optimizer_steps, exact=True)
            value['initial_state_unchanged'] = all(binding(v['path']) == v for v in session['initial_parent'].values())
            value['owned_cleanup_verified'] = all(b.get('owned_cleanup_verified') is True for b in value['batches'])
        except Exception as error:
            value.update(status='stopped_failure', error=f'Final preservation: {error}')
        value['stopped_utc'] = utc(); save(run, value)
        summary = summarize(value); write_markdown(run, summary); event(run, 'campaign_finished', summary=summary)
    return 0 if value['status'] in ('finished', 'stopped_requested', 'stopped_budget') else 1


def start(args):
    config = read(args.config)
    os.environ['HEROES_ARTIFACT_ROOT'] = config['artifact_root']
    from prepare_local_training import validate_config
    validate_config(config)
    run_root = Path(config['run_root'])
    with campaign_locks(config), exclusive_lock(run_root / 'preparation.lock'):
        if args.parent_final is not None: final_parent(args.parent_final)
        release, commit = release_checkout(config['artifact_root'], None if args.fast_path else ACCEPTED_IMPLEMENTATION)
        # Re-enter the exact frozen source before creating any session or game.
        command = [sys.executable, str(release / 'tools/run_local_training.py'), '_launch',
                   '--config', str(args.config.resolve()), '--hours', str(args.hours),
                   '--profile', args.profile, '--eval-every', str(args.eval_every)]
        if args.fast_path:
            command += ['--fast-path', '--prefix-helpers', str(args.prefix_helpers),
                        '--overlap-verification' if args.overlap_verification else '--serial-verification',
                        *(x for name in args.experiment for x in ('--experiment', name))]
        if args.max_updates is not None: command += ['--max-updates', str(args.max_updates)]
        if args.games is not None: command += ['--games', str(args.games)]
        if args.parent is not None: command += ['--parent', str(args.parent.resolve())]
        if args.parent_final is not None: command += ['--parent-final', str(args.parent_final.resolve())]
    os.chdir(release)
    os.execv(sys.executable, command)
    raise AssertionError("exec returned")


def launch(args):
    with campaign_locks(read(args.config)):
        return launch_owned(args)


def launch_owned(args):
    from experiments.learner import local_script_training as adapter
    from experiments.learner import builtin_first_run as pilot
    from experiments.learner.inference import source_identity
    from experiments.learner.builtin_training_resume import binding, require
    from prepare_local_training import validate_config
    config = read(args.config); validate_config(config)
    # This release is the opt-in fast path; the default path releases ACCEPTED_IMPLEMENTATION instead.
    require(args.fast_path, 'This release runs only as the explicit fast path; launch with start')
    execution = adapter.execution_choice(overlapped_verification=args.overlap_verification,
                                         prefix_helpers=args.prefix_helpers, experiments=args.experiment)
    run_root = Path(config['run_root']); run_root.mkdir(parents=True, exist_ok=True)
    latest = run_root / 'latest-checkpoint.json'
    parent = (final_parent(args.parent_final) if args.parent_final else read(args.parent)['parent'] if args.parent
              else read(latest)['parent'] if latest.exists() else config['parent'])
    initial = adapter.restore_parent(parent)
    adapter.check_lineage(adapter.read_descriptor(parent)[0], execution)  # before any session or game
    parent_final_evidence = None
    if args.parent_final is not None:
        prior = read(args.parent_final / 'campaign-report.json')
        require(vars(initial.progress) == prior['full_final_restore']['progress'],
                'Final parent counters differ from the restored checkpoint')
        parent_final_evidence = {name: binding(args.parent_final / name) for name in
                                ('session.json', 'campaign-report.json', 'shutdown.json')}
    run = run_root / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    session = dict(schema='heroes-local-script-session-v1', scope='heroes-local-script-training-v1',
        source=source_identity('adventure-capabilities-v1'), implementation=pilot.implementation(), prerequisites=pilot.prerequisites(),
        configuration=adapter.configuration(parent), execution=execution, asset_config=binding(args.config),
        runtime_bin=config['runtime_bin'], runtime=config['runtime'], data=config['data'], maps=config['maps'],
        artifact_root=config['artifact_root'], run_root=str(run_root), run_dir=str(run), initial_parent=parent,
        next_sequence=config.get('next_sequence', 266), supervisor_pid=os.getpid(),
        training_engine_seed_base=config.get('seed_namespace', {}).get('engine_base', 20_000_000),
        training_policy_seed_base=config.get('seed_namespace', {}).get('policy_base', 30_000_000),
        max_updates=args.max_updates, target_games=args.games, profile=config['default_profile'],
        evaluation_interval_updates=args.eval_every,
        parent_final_run=str(args.parent_final.resolve()) if args.parent_final else None,
        parent_final_evidence=parent_final_evidence,
        frozen_files={p: binding(ROOT / p) for p in ('tools/run_local_training.py', 'tools/prepare_local_training.py', 'train-local.sh')},
        authorization='Manual user invocation; no agent/model calls, no automatic retry or post-deadline restart',
        **budget(args.hours))
    require(not session['source']['tracked_worktree_dirty'], 'Frozen source is dirty')
    session_path = run / 'session.json'; atomic(session_path, session)
    atomic(run_root / 'latest-run.json', dict(run_dir=str(run), session=binding(session_path)))
    hypothesis = ('Continue the final easy-trained state on original hero-management-v2; test whether stochastic on-policy PPO discovers wins within the declared game budget. A first win is not proof of reliable strategy.'
                  if session['profile'] == 'original' else
                  'Continue the accepted easier8x8 distribution; measure fixed paired outcomes without claiming original-map transfer.')
    event(run, 'preregistered', session=session, hypothesis=hypothesis)
    print(f"Implementation: opt-in exact fast path ({'overlapped' if execution['overlapped_verification'] else 'serial'} "
          f"verification, {execution['prefix_helpers']} prefix helpers"
          + (f"; NUMERICAL EXPERIMENT {', '.join(execution['experiments'])}" if execution['experiments'] else '') + ')', flush=True)
    print(f"Run: {run}\nParent: update {initial.progress.updates}\nCompute cutoff: {utc(session['compute_deadline_unix'])}\n"
          f"Hard stop: {utc(session['absolute_deadline_unix'])}\nNo LLM calls. Ctrl-C requests a stop after the current batch.", flush=True)
    return supervise(session_path)


def supervise(session_path):
    """Own one worker, preserving startup/cleanup failures in final receipts."""
    from experiments.headless.process_deadline import guarded_popen
    session = read(session_path); run = Path(session['run_dir'])
    run_root = Path(session['run_root']); parent = session['initial_parent']
    process = None; awake = None; rc = None; supervisor_error = None; previous = {}
    def stop_requested(signum, frame):
        (run / 'stop-requested').touch(exist_ok=True)
        print('Stop requested; finishing the current bounded batch.', flush=True)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop_requested)
        process = guarded_popen([sys.executable, str(ROOT / 'tools/run_local_training.py'), '_worker', '--session', str(session_path)],
            deadline_monotonic=session['hard_deadline_monotonic'] - 5,
            deadline_wall=session['absolute_deadline_unix'] - 5, evidence_path=run / 'campaign-deadline.jsonl', cwd=ROOT)
        if sys.platform == 'darwin':
            awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)])
        rc = process.wait(timeout=max(0, left(session, hard=True)))
    except BaseException as error:
        supervisor_error = f'{type(error).__name__}: {error}'
    finally:
        try:
            cleanup = cleanup_owned(session, process, run, awake)
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
    report_error = None
    try:
        value = read(run / 'campaign-report.json')
        if not isinstance(value.get('batches'), list) or 'latest_parent' not in value:
            raise ValueError('Incomplete campaign report')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        value = dict(batches=[], latest_parent=parent)
        report_error = f'{type(error).__name__}: {error}'
    errors = [e for e in (value.get('error'), supervisor_error, report_error) if e]
    within_hard_deadline = left(session, hard=True) >= 0
    if not cleanup['owned_cleanup_verified']:
        errors.append('Owned shutdown unverified: ' + str(cleanup['errors']))
    if not within_hard_deadline:
        errors.append('Supervisor completion exceeded the original hard deadline')
    if not (run / 'RESULTS.md').exists() or value.get('status') == 'running':
        value['status'] = 'interrupted'
        errors.append('Worker exited without a final report; only previously verified checkpoints are usable')
    if supervisor_error or report_error or not cleanup['owned_cleanup_verified'] or not within_hard_deadline or rc != 0:
        value['status'] = 'stopped_failure'
        if rc not in (None, 0): errors.append(f'Worker return code: {rc}')
    value.update(error='; '.join(errors) or None, supervisor_error=supervisor_error,
                 supervisor_cleanup=cleanup, owned_cleanup_verified=cleanup['owned_cleanup_verified'],
                 supervisor_within_hard_deadline=within_hard_deadline)
    shutdown = dict(timestamp_utc=utc(), worker_returncode=rc, supervisor_error=supervisor_error,
        owned_cleanup_verified=cleanup['owned_cleanup_verified'], cleanup=cleanup,
        within_hard_deadline=within_hard_deadline)
    # Write closure evidence even if later result rendering fails.
    atomic(run / 'shutdown.json', shutdown)
    save(run, value); write_markdown(run, summarize(value))
    print(f"Run status: {value['status']}. Read {run / 'RESULTS.md'}", flush=True)
    if (run_root / 'latest-checkpoint.json').exists():
        print(f"Latest verified state: {run_root / 'latest-checkpoint.json'}", flush=True)
    return 0 if rc == 0 and value['status'] in ('finished', 'stopped_requested', 'stopped_budget') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('start', 'check', 'status', '_launch', '_worker'), default='start')
    parser.add_argument('--profile', choices=('easier8x8', 'original'), default='easier8x8')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--hours', type=float, help='Safety cap: default 2 hours, or 48 hours with --games')
    parser.add_argument('--games', type=int, help='Target newly admitted training games, a positive multiple of four; evaluations excluded')
    parser.add_argument('--eval-every', type=int, help='Paired evaluation interval in updates: default 5, or 100 with --games')
    parser.add_argument('--max-updates', type=int)
    parents = parser.add_mutually_exclusive_group()
    parents.add_argument('--parent', type=Path, help='Explicit latest-checkpoint.json; otherwise continue the last verified profile state')
    parents.add_argument('--parent-final', type=Path, help='Restore only the fully finalized state of this prior run')
    parser.add_argument('--session', type=Path)
    fast = parser.add_argument_group('opt-in exact fast path (default: the accepted implementation)')
    fast.add_argument('--fast-path', action='store_true', help='Run the reviewed exact fast path of this commit')
    overlap = fast.add_mutually_exclusive_group()
    overlap.add_argument('--overlap-verification', dest='overlap_verification', action='store_true', default=None,
                         help='Overlap per-game verification in forked children (default except on macOS)')
    overlap.add_argument('--serial-verification', dest='overlap_verification', action='store_false',
                         help='Run every verification step serially in the worker (default on macOS)')
    fast.add_argument('--prefix-helpers', type=int, default=0, help='Exact forked PPO prefix helpers (0 to 8; default 0)')
    fast.add_argument('--experiment', action='append', default=[], choices=('shared_graph_encodings',),
                      help='Separately reported NUMERICAL experiment (reordered gradient summation); never a default')
    args = parser.parse_args()
    if not args.fast_path and (args.overlap_verification is not None or args.prefix_helpers or args.experiment):
        parser.error('Fast-path options require --fast-path')
    if not 0 <= args.prefix_helpers <= 8: parser.error('--prefix-helpers must be between 0 and 8')
    if args.overlap_verification is None: args.overlap_verification = sys.platform != 'darwin'
    args.config = args.config or (ORIGINAL_CONFIG if args.profile == 'original' else DEFAULT_CONFIG)
    args.hours = args.hours if args.hours is not None else (48 if args.games is not None else 2)
    args.eval_every = args.eval_every if args.eval_every is not None else (100 if args.games is not None else 5)
    if args.games is not None and (args.games <= 0 or args.games % 4): parser.error('--games must be a positive multiple of four')
    if args.games is not None and args.max_updates is not None: parser.error('Choose --games or --max-updates, not both')
    if args.eval_every < 1: parser.error('--eval-every must be positive')
    if args.max_updates is not None and args.max_updates < 1: parser.error('--max-updates must be positive')
    if args.command == '_worker': return worker(args.session)
    if args.command == 'status':
        root = Path(read(args.config)['run_root']); latest = read(root / 'latest-run.json')
        print(json.dumps(read(Path(latest['run_dir']) / 'summary.json'), indent=2)); return 0
    if not args.config.exists():
        raise FileNotFoundError(f'Prepare local assets first: {sys.executable} tools/prepare_local_training.py --output {args.config}')
    config = read(args.config); os.environ['HEROES_ARTIFACT_ROOT'] = config['artifact_root']
    if config['default_profile'] != args.profile: parser.error('--profile differs from the selected machine configuration')
    if args.experiment:
        official = {Path(read(c)['run_root']).resolve() for c in (DEFAULT_CONFIG, ORIGINAL_CONFIG) if c.exists()}
        if args.config.resolve() in (DEFAULT_CONFIG.resolve(), ORIGINAL_CONFIG.resolve()) \
                or Path(config['run_root']).resolve() in official:
            parser.error('A numerical experiment needs its own --config and run root, never the official ones')
    if args.command == 'check':
        from prepare_local_training import validate_config
        checked = validate_config(config)
        print(json.dumps(dict(status='offline_checks_passed', checkpoint=checked['checkpoint_restore']['progress'],
                              exact_restore=True, backend=checked['backend'], gameplay_performed=False, ppo_updates=0), indent=2))
        return 0
    budget(args.hours)  # Validate before filesystem/session mutation.
    return launch(args) if args.command == '_launch' else start(args)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'Cannot start/continue: {error}', file=sys.stderr); raise SystemExit(2)
