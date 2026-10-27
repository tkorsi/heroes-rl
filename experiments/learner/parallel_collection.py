"""Fresh eight-game collection in isolated, deadline-owned interpreters.

An immutable plan opts into one or eight collector processes. Every game uses
the same frozen policy; the parent retains schedule ordering and independently
verifies every returned capture. Workers transfer fresh capture provenance over
private live pipes, never training receipts. Only the existing parent verifier
can mint receipts. This module never updates a model or launches a campaign.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from . import builtin_first_run as pilot
from . import builtin_gameplay as gameplay
from experiments.headless import contextual_evidence as evidence
from experiments.headless.process_deadline import (
    guarded_popen, close_owned_process, cleanup_from_evidence, _release_owned_process,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'heroes-parallel-collection-v1'
require = gameplay.require


@dataclass(frozen=True)
class BatchCollection:
    attempts: tuple
    captures: tuple
    bindings: tuple
    receipts: tuple
    collection_seconds: float
    verification_seconds: float

    def discard(self):
        """Revoke this batch's in-memory admission capabilities after a probe.

        Files remain immutable evidence. Drop this result object afterwards to
        release its potentially large histories; they cannot be readmitted.
        """
        _discard_capabilities(self.bindings, self.captures, self.receipts)


def _discard_capabilities(bindings, captures, receipts):
    for objects, registry in ((receipts, gameplay._RECEIPTS),
                              (captures, gameplay._CAPTURES), (bindings, gameplay._BINDINGS)):
        for value in objects:
            if value is not None and registry.get(id(value)) is value:
                registry.pop(id(value))


def discard_batch(result):
    """Revoke only the given completed collection's admission capabilities."""
    require(type(result) is BatchCollection, 'Expected a parallel collection result')
    result.discard()


@dataclass
class _Job:
    index: int
    token: str
    lease: object = None
    process: object = None
    log: object = None


class CollectionFailed(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = tuple(attempts)


def _read_plan(path, sha):
    path = Path(path).resolve()
    require(pilot.digest(path) == sha, 'Immutable parallel collection plan changed')
    plan = evidence.strict_json(path.read_bytes())
    require(plan.get('schema') in ('heroes-builtin-training-v1-plan',
                                   'heroes-builtin-checkpoint-evaluation-v1-plan'),
            'Unsupported parallel collection plan schema')
    cfg = plan['configuration']
    execution = cfg.get('parallel_collection')
    require(type(execution) is dict and set(execution) == {'batch_size', 'workers'}
            and type(execution['batch_size']) is int and execution['batch_size'] == 8
            and type(execution['workers']) is int and execution['workers'] in (1, 8),
            'Declare exactly eight games and one or eight collection workers')
    rows = plan['schedule']
    require(type(rows) is list and len(rows) == 8
            and len({row['episode'] for row in rows}) == 8
            and len({(row['engine_seed'], row['policy_seed'], row['learner_player']) for row in rows}) == 8,
            'Parallel collection needs eight unique scheduled slots')
    phase = 'training' if plan['schema'] == 'heroes-builtin-training-v1-plan' else 'evaluation'
    require(plan.get('mode', phase) == phase, 'Plan mode differs from its schema')
    require(all(row.get('phase') == phase for row in rows), 'Mixed training and evaluation slots are unsupported')
    for row in rows:
        for name, low, high in (('episode', 1, 2**31-1), ('engine_seed', 1, 2**31-1),
                                ('learner_player', 0, 1), ('policy_seed', 0, 2**63-1)):
            require(type(row.get(name)) is int and low <= row[name] <= high,
                    'Invalid collection schedule field: ' + name)
    require(cfg.get('allow_verified_training_prefix') is False,
            'Parallel collection requires the complete batch, never a prefix')
    require(Path(plan['execution_output']).is_absolute(), 'Use an absolute collection output directory')
    return plan


def _check_identity(plan):
    from .inference import source_identity
    from experiments.headless.builtin_evidence import profile_for
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    interface = profile_for(plan['source']['interface']).environment
    require(source_identity(interface) == plan['source'], 'Parallel collector source changed')
    implementation = plan.get('implementation')
    require(type(implementation) is dict and bool(implementation)
            and 'experiments/learner/parallel_collection.py' in implementation,
            'Plan must bind the parallel collector implementation')
    for name, sha in implementation.items():
        require(pilot.digest(evidence.inside(ROOT, name)) == sha, 'Prepared implementation changed: ' + name)
    for name in ('checkpoint', 'replay_binary'):
        require(pilot.binding(plan[name]['path']) == plan[name], 'Frozen artifact changed: ' + name)
    launcher = VcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']))
    require(launcher.identity() == plan['native']
            and runtime_identity(Path(plan['binary']).parent) == plan['runtime'],
            'Native runtime or scenario changed')


def _worker_command(plan_path, sha, index, deadline, wall):
    return [sys.executable, '-m', 'experiments.learner.parallel_collection', 'worker',
            '--plan', str(plan_path), '--sha256', sha, '--index', str(index),
            '--deadline-monotonic', str(deadline), '--deadline-wall', str(wall)]


def _supervise(jobs, *, workers, budget, launch, finish, cleanup_reserve=0):
    """Own every child from launch through cleanup, retaining input order.

    ``launch(job)`` assigns its guarded Popen to ``job.process`` immediately.
    ``finish(job)`` reads that child's private result pipe after a successful
    exit. Both callbacks run only on this coordinator's main thread.
    """
    pending, active, results = list(jobs), [], {}
    require(type(workers) is int and 1 <= workers <= 8, 'Expected one to eight workers')
    primary = None
    try:
        while pending or active:
            budget.remaining(cleanup_reserve)
            while pending and len(active) < workers:
                job = pending.pop(0)
                active.append(job)  # cleanup owns even a partially failed launch
                launch(job)
            finished = [job for job in active if job.process.poll() is not None]
            for job in finished:
                require(job.process.returncode == 0,
                        f'Collector slot {job.index} exited with status {job.process.returncode}')
                results[job.index] = finish(job)
                active.remove(job)
            if not finished and active:
                time.sleep(min(0.02, budget.remaining(cleanup_reserve)))
        return tuple(results[job.index] for job in jobs)
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        # Release all ownership pipes first: cancellation of eight workers must
        # not wait serially for each worker's grace period before stopping the next.
        for job in jobs:
            if job.process is not None:
                try:
                    _release_owned_process(job.process)
                except BaseException as error:
                    errors.append(str(error))
        cleanup_end = min(time.monotonic() + 20, budget.deadline)
        cleanup_wall = min(time.time() + 20, budget.wall_deadline)
        for job in jobs:
            try:
                if job.process is not None:
                    close_owned_process(job.process, timeout=max(0, cleanup_end-time.monotonic()),
                        deadline_monotonic=cleanup_end, deadline_wall=cleanup_wall)
            except BaseException as error:
                errors.append(str(error))
            finally:
                if job.process is not None:
                    for stream in (job.process.stdin, job.process.stdout):
                        if stream is not None:
                            stream.close()
                if job.log is not None:
                    job.log.close()
        if errors:
            if primary is not None:
                primary.add_note('Collector cleanup failures: ' + '; '.join(errors))
            else:
                raise RuntimeError('Collector cleanup failures: ' + '; '.join(errors))


def collect_batch(plan_path, plan_sha256, model, checkpoint, *, workers=8,
                  deadline_monotonic, deadline_wall, save=None):
    """Collect eight fresh games, then verify all and return live parent receipts.

    A failure cancels remaining children and admits no partial batch. ``save``
    receives ``(schedule_index, attempt_dict)``. Diagnostic plans yield eight
    no receipts. No global RNG, optimizer, model weights or campaign state
    is changed. The caller must keep the model frozen until this call returns.
    """
    from .contextual_model import weights_sha256
    from .builtin_rollout import load_builtin_episode, audit_persisted_builtin_policy
    from experiments.headless.builtin_replay import verify_episode
    from .builtin_behavior import analyze_episode
    from validation.probes.validate_hero_policy import wall_limit

    plan_path = Path(plan_path).resolve()
    plan = _read_plan(plan_path, plan_sha256)
    cfg, output = plan['configuration'], Path(plan['execution_output'])
    require(type(workers) is int and workers == cfg['parallel_collection']['workers'],
            'Worker count differs from the immutable plan')
    _check_identity(plan)
    before = weights_sha256(model)
    budget = pilot.Budget(deadline_monotonic, deadline_wall)
    budget.remaining(cfg['cleanup_reserve_seconds'])
    control = output / 'parallel-collection'
    control.mkdir(parents=True, exist_ok=False)
    attempts = [pilot.empty_attempt(row) for row in plan['schedule']]
    bindings, captures, receipts = [None]*8, [None]*8, [None]*8
    jobs = []
    training = plan['schema'] == 'heroes-builtin-training-v1-plan'

    def progress(index, **fields):
        attempts[index].update(fields)
        if save is not None:
            save(index, dict(attempts[index]))

    def launch(job):
        row = plan['schedule'][job.index]
        allowance = cfg['attempt_wall_seconds']
        deadline = min(deadline_monotonic-2, time.monotonic()+allowance)
        wall = min(deadline_wall-2, time.time()+allowance)
        job.log = (control / f'worker-{job.index:02d}.log').open('xb')
        job.process = guarded_popen(_worker_command(plan_path, plan_sha256, job.index, deadline, wall),
            deadline_monotonic=deadline, deadline_wall=wall,
            evidence_path=control / f'worker-{job.index:02d}-deadline.jsonl', cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=job.log,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'})
        if job.lease is not None:
            gameplay._attach_parallel_process(job.lease, job.process)
        progress(job.index, status='running', reason=None, stage='collection', worker_pid=job.process.pid)
        payload = {'token': job.token, 'checkpoint': checkpoint, 'weights_sha256': before}
        job.process.stdin.write((json.dumps(payload)+'\n').encode())
        job.process.stdin.close()

    def finish(job):
        index, row = job.index, plan['schedule'][job.index]
        folder = output / f"episode-{row['episode']:06d}"
        if training:
            capture, message = gameplay._complete_parallel_capture(job.lease)
        else:
            message = evidence.strict_json(job.process.stdout.read(65537))
            require(message.get('token') == job.token and message.get('schedule_index') == index
                    and message.get('plan_sha256') == plan_sha256 and message.get('status') == 'passed',
                    'Diagnostic completion differs from its live launch')
            path = folder / 'builtin-rollout.json'
            require(message.get('capture') == evidence.file_identity(path),
                    'Diagnostic capture changed after child completion')
            capture = load_builtin_episode(path)
        captures[index] = capture
        progress(index, **message['attempt'], stage='collected', status='pending_verification')
        return message

    started = time.monotonic()
    try:
        for index, row in enumerate(plan['schedule']):
            folder = output / f"episode-{row['episode']:06d}"
            require(not folder.exists(), 'Never overwrite or retry an existing episode')
            lease = None
            if training:
                bindings[index] = gameplay.TrainingBinding.create(plan_path=plan_path,
                    plan_sha256=plan_sha256, schedule_index=index, folder=folder,
                    checkpoint=checkpoint, checkpoint_file=plan['checkpoint']['path'], weights_sha256=before,
                    opponent_identity=pilot.native_opponent(plan['runtime'], row['learner_player']),
                    source_identity=plan['source'], native_identity=plan['native'], max_days=cfg['max_days'],
                    nominal_choices=cfg['ordinary_choice_limit'], max_choices=cfg['max_total_choices'])
                lease = gameplay._begin_parallel_capture(bindings[index])
            jobs.append(_Job(index, lease.token if lease is not None else secrets.token_hex(32), lease))
        _supervise(jobs, workers=workers, budget=budget, launch=launch, finish=finish,
                   cleanup_reserve=cfg['cleanup_reserve_seconds'])
        for path in sorted(control.glob('*native-deadline.jsonl')):
            cleanup_from_evidence(path, timeout=min(10, budget.remaining()),
                                  deadline_monotonic=budget.deadline, deadline_wall=budget.wall_deadline)
        collection_seconds = time.monotonic()-started
        verified_started = time.monotonic()
        for index, row in enumerate(plan['schedule']):
            folder = output / f"episode-{row['episode']:06d}"
            progress(index, stage='verification')
            checked_at = time.monotonic()
            with wall_limit(budget.remaining(cfg['cleanup_reserve_seconds'])):
                if training:
                    receipts[index] = gameplay.verify_training_capture(folder, captures[index], bindings[index], model,
                        replay_binary=Path(plan['replay_binary']['path']), data=Path(plan['data']),
                        remaining=lambda: budget.remaining(cfg['cleanup_reserve_seconds']))
                else:
                    reconstructed = audit_persisted_builtin_policy(folder, model, checkpoint, row['policy_seed'],
                        pilot.native_opponent(plan['runtime'], row['learner_player']))
                    pilot.write(folder / 'policy-reconstruction.json', reconstructed)
                    replay = verify_episode(folder, binary=Path(plan['replay_binary']['path']), data=Path(plan['data']),
                        timeout=budget.remaining(cfg['cleanup_reserve_seconds']))
                    require(replay.get('verified') is True, 'Independent native replay failed')
                    progress(index, reconstruction=reconstructed)
                pilot.write(folder / 'behavior.json', analyze_episode(folder))
            require(weights_sha256(model) == before, 'Frozen learner changed during parallel collection')
            progress(index, status='passed', stage='finished', reason=None, replay_verified=True,
                     completion_receipt=True, weights_unchanged=True,
                     verification_seconds=time.monotonic()-checked_at)
        _check_identity(_read_plan(plan_path, plan_sha256))
        verification_seconds = time.monotonic()-verified_started
        result = BatchCollection(tuple(attempts), tuple(captures), tuple(bindings), tuple(receipts) if training else (),
                                 collection_seconds, verification_seconds)
        pilot.write(control / 'result.json', {'schema': SCHEMA, 'status': 'passed', 'workers': workers,
            'batch_size': 8, 'collection_seconds': collection_seconds, 'verification_seconds': verification_seconds,
            'attempts': attempts, 'optimizer_updates': 0, 'training_samples_admitted': 0})
        return result
    except BaseException as error:
        _discard_capabilities(bindings, captures, receipts)
        for path in sorted(control.glob('*native-deadline.jsonl')):
            try:
                cleanup_from_evidence(path, timeout=min(10, budget.remaining()),
                                      deadline_monotonic=budget.deadline, deadline_wall=budget.wall_deadline)
            except Exception as cleanup_error:
                error.add_note(f'Native cleanup not verified for {path.name}: {cleanup_error}')
        for index, attempt in enumerate(attempts):
            if attempt['status'] != 'passed':
                progress(index, status='failed' if attempt['status'] != 'not_started' else 'not_started',
                         reason='batch_aborted', error=f'{type(error).__name__}: {error}', completion_receipt=False)
        pilot.write(control / 'result.json', {'schema': SCHEMA, 'status': 'failed', 'workers': workers,
            'batch_size': 8, 'wall_seconds': time.monotonic()-started, 'attempts': attempts,
            'optimizer_updates': 0, 'training_samples_admitted': 0, 'error': f'{type(error).__name__}: {error}'})
        if isinstance(error, Exception):
            raise CollectionFailed(str(error), attempts) from error
        raise
    finally:
        for job in jobs:
            if job.lease is not None and not job.lease.consumed:
                gameplay._cancel_parallel_capture(job.lease)


def _collect_worker(args):
    import torch
    from . import checkpoints
    from .contextual_model import weights_sha256
    from .builtin_rollout import BuiltinCollector, retain_failed_builtin_evidence
    from experiments.headless.env import Limits
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.headless.builtin_evidence import profile_for
    from validation.probes.validate_hero_policy import wall_limit

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = evidence.strict_json(sys.stdin.buffer.readline(65537))
    require(type(payload) is dict and set(payload) == {'token', 'checkpoint', 'weights_sha256'}
            and type(payload['token']) is str and len(payload['token']) == 64, 'Malformed private collection request')
    plan = _read_plan(args.plan, args.sha256)
    require(0 <= args.index < len(plan['schedule']), 'Unknown collection slot')
    cfg, row = plan['configuration'], plan['schedule'][args.index]
    output = Path(plan['execution_output'])
    control = output / 'parallel-collection'
    folder = output / f"episode-{row['episode']:06d}"
    interface = profile_for(plan['source']['interface']).environment
    budget = pilot.Budget(args.deadline_monotonic, args.deadline_wall)
    env, launcher, collector = None, None, None
    started = time.monotonic()
    result = None
    try:
        with wall_limit(budget.remaining(15)):
            _check_identity(plan)
            loaded = checkpoints.load_policy_checkpoint(plan['checkpoint']['path'],
                expected_source_metadata=plan['checkpoint_source'],
                allowed_provenance=('random_initialization', checkpoints.BUILTIN_GAMEPLAY_PROVENANCE),
                interface=interface)
            model = loaded.model.eval().requires_grad_(False)
            require(loaded.checkpoint == payload['checkpoint'] and loaded.weights_sha256 == payload['weights_sha256'],
                    'Worker frozen policy differs from the coordinator')
            opponent = pilot.native_opponent(plan['runtime'], row['learner_player'])
            if row['phase'] == 'training':
                bound = gameplay.TrainingBinding.create(plan_path=args.plan, plan_sha256=args.sha256,
                    schedule_index=args.index, folder=folder, checkpoint=loaded.checkpoint,
                    checkpoint_file=plan['checkpoint']['path'], weights_sha256=loaded.weights_sha256,
                    opponent_identity=opponent, source_identity=plan['source'], native_identity=plan['native'],
                    max_days=cfg['max_days'], nominal_choices=cfg['ordinary_choice_limit'],
                    max_choices=cfg['max_total_choices'])
                collector = gameplay.TrainingCollector(model, bound, interface=interface)
            else:
                class BoundedCollector(BuiltinCollector):
                    sampled = 0
                    def act(self, request):
                        pilot.require_choice_budget(self.sampled, request.record['day'], cfg)
                        action = super().act(request)
                        self.sampled += 1
                        return action
                collector = BoundedCollector(model, loaded.checkpoint, learner_player=row['learner_player'],
                    opponent_identity=opponent, policy_seed=row['policy_seed'], interface=interface)
            launcher = DeadlineVcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']),
                deadline_monotonic=args.deadline_monotonic-2, deadline_wall=args.deadline_wall-2,
                evidence_path=control / f'worker-{args.index:02d}-native-deadline.jsonl')
            limits = Limits(max_days=cfg['max_days'], max_actions=cfg['max_total_choices'],
                watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30)
            seat = row['learner_player']
            env = BuiltinHeadlessEnv(launcher, output, learner_player=seat, limits=limits,
                ledger=True, replay=True, interface=interface, agents={pilot.COLORS[seat]: {
                    'checkpoint': loaded.checkpoint, 'kind': 'recurrent-policy',
                    'evaluation_only': row['phase'] != 'training'}})
            collection_started = time.monotonic()
            collected = collector.run_episode(env, episode=row['episode'], seed=row['engine_seed'],
                                              max_requests=cfg['ordinary_choice_limit'])
            collection_seconds = time.monotonic()-collection_started
            experience = collected.experience if row['phase'] == 'training' else collected
            manifest = experience.manifest
            require(manifest['status'] in ('completed', 'truncated'), 'Technical episode failure is unscored')
            require(weights_sha256(model) == loaded.weights_sha256, 'Worker policy changed during collection')
            capture_path = folder / ('builtin-training-rollout.json' if row['phase'] == 'training' else 'builtin-rollout.json')
            if row['phase'] != 'training':
                pilot.write(capture_path, collected.to_dict())
            result = {'status': 'passed', 'token': payload['token'], 'schedule_index': args.index,
                'plan_sha256': args.sha256, 'capture': evidence.file_identity(capture_path),
                'attempt': {'episode_directory': str(folder), 'episode_status': manifest['status'],
                    'termination_reason': manifest['termination_reason'], 'sampled_choices': len(experience.decisions),
                    'checkpoint': loaded.checkpoint, 'weights_sha256': loaded.weights_sha256,
                    'opponent_identity': opponent, 'environment_limits': asdict(limits),
                    'outcomes': manifest['outcomes_by_player'], 'rewards': manifest['rewards_by_player'],
                    'collection_seconds': collection_seconds, 'external_player_histories': 1}}
    except BaseException as error:
        try:
            retain_failed_builtin_evidence(env, collector, {'schedule_index': args.index}, error,
                                           cleanup_seconds=min(5, budget.remaining()))
        except Exception:
            pass
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if launcher is not None:
                launcher.close_owned_groups(timeout=8)
    with wall_limit(budget.remaining()):
        _check_identity(_read_plan(args.plan, args.sha256))
    result['attempt'].update(cleanup_verified=True, identity_verified=True, wall_seconds=time.monotonic()-started)
    print(json.dumps(result, sort_keys=True, separators=(',', ':')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('worker', help='Private coordinator-owned collection child')
    worker.add_argument('--plan', type=Path, required=True)
    worker.add_argument('--sha256', required=True)
    worker.add_argument('--index', type=int, required=True)
    worker.add_argument('--deadline-monotonic', type=float, required=True)
    worker.add_argument('--deadline-wall', type=float, required=True)
    args = parser.parse_args()
    _collect_worker(args)


if __name__ == '__main__':
    main()
