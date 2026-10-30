#!/usr/bin/env python3
"""Manually evaluate one frozen self-play game with doubled per-player choices.

Default check mode performs no gameplay and consumes no seeds. Evaluation
captures are diagnostic-only and never create training admission capabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import benchmark_copied_training as copies
from tools import run_selfplay_training as campaign

BASE_COMMIT = 'b47c1b9aed1bbe8d2c52d87a02bcdac3e4931617'
BASE = campaign.ARTIFACTS / 'selfplay-one-game-2x'
SCHEMA = 'heroes-selfplay-one-game-2x-v1'
BUDGET = dict(ordinary_choices_per_player=9920, reserved_dialog_choices_per_player=320,
              hard_choices_per_player=10240, max_days=365, capture_seconds=1200)
ENGINE_SEED, POLICY_SEED = 1_960_000_000, 1_962_000_000
require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound


def source_identity():
    """The evaluator may add its wrapper; every reused runtime file stays exact."""
    from validation.probes.validate_capabilities import capability_implementation_identity
    from experiments.learner.inference import source_identity as learner_identity
    implementation = capability_implementation_identity()
    learner = learner_identity(campaign.INTERFACE)
    implementation.update(learner['source_sha256'])
    for name in ('tools/run_selfplay_training.py', 'tools/benchmark_copied_training.py'):
        implementation[name] = binding(ROOT / name)['sha256']
    archive = subprocess.check_output(['git', 'archive', BASE_COMMIT, *sorted(implementation)], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as source:
        original = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                    for member in source.getmembers() if member.isfile()}
    require(original == implementation, 'Evaluation changed an existing learner, adapter, engine or replay source')
    return dict(checkout=copies.checkout_identity(ROOT), source=learner,
        unchanged_runtime_base=BASE_COMMIT, implementation=implementation,
        driver=binding(__file__), launcher=binding(ROOT / 'play-one-selfplay-2x.sh'))


def seed_row(offset):
    require(type(offset) is int and 0 <= offset < 100000, 'Evaluation seed namespace exhausted')
    return dict(episode=1, engine_seed=ENGINE_SEED + offset,
                policy_seeds=[POLICY_SEED + 2 * offset, POLICY_SEED + 2 * offset + 1])


def seed_ledger():
    path = BASE / 'seed-reservations.json'
    if not path.exists():
        require(not BASE.exists() or not any(p.name != 'namespace.lock' for p in BASE.iterdir()),
                'Evaluation seed ledger is missing while prior artifacts remain; refusing seed reuse')
        return dict(schema=SCHEMA + '-seeds', reservations=[])
    ledger = read(path)
    require(ledger.get('schema') == SCHEMA + '-seeds' and type(ledger.get('reservations')) is list,
            'Invalid evaluation seed ledger')
    outputs = set()
    for index, row in enumerate(ledger['reservations']):
        require(row.get('offset') == index and row.get('schedule') == seed_row(index)
                and row.get('consumed_including_unstarted') is True and row['output'] not in outputs,
                'Evaluation seed reservation changed')
        require(read(BASE / f'reserved-{index:06d}.json') == row, 'Evaluation reservation receipt changed or is missing')
        outputs.add(row['output'])
    require({p.name for p in BASE.glob('reserved-*.json')}
            == {f'reserved-{i:06d}.json' for i in range(len(outputs))}, 'Unmatched evaluation reservation history')
    return ledger


def reserve(output):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'namespace.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = seed_ledger()
        require(all(row['output'] != str(output) for row in ledger['reservations']), 'Evaluation output already consumed seeds')
        index = len(ledger['reservations'])
        reservation = dict(offset=index, output=str(output), schedule=seed_row(index), consumed_including_unstarted=True)
        ledger['reservations'].append(reservation)
        # Consumption precedes the receipt. An interrupted write fails closed.
        write(BASE / 'seed-reservations.json', ledger)
        receipt = BASE / f'reserved-{index:06d}.json'
        write(receipt, reservation, new=True)
        return binding(receipt)


def inspect_parent(lineage_path):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.learner import selfplay_training as training
    from experiments.learner.native_execution import native_execution_choice
    identity = source_identity()
    lineage = campaign.inspect_lineage(lineage_path)
    selected = lineage['selected']; pair = selected['checkpoint']
    launch = read(bound(selected['launch']))
    require(pair['progress']['updates'] == 1753 and lineage['admitted_games'] == 100
            and launch['source']['checkout']['commit'] == BASE_COMMIT,
            'This evaluation requires the completed published checkpoint 1753 lineage')
    require(launch['policy_support'] == campaign.GUARD
            and runtime_identity(Path(launch['binary']).parent) == launch['runtime'], 'Accepted engine or policy support changed')
    bound(launch['replay_binary']); bound(launch['native_manifest'])
    execution = native_execution_choice(campaign.NATIVE, campaign.NATIVE_SHA, cpu_cap=16, require_acceptance=False)
    require(execution == launch['execution'], 'Accepted native learner execution profile changed')
    game_map = binding(launch['game_map'])
    require(game_map['sha256'] == launch['native']['map']['sha256'], 'Original map changed')
    state = training.restore_checkpoint_pair(pair, expected_metadata=pair['metadata'], expected_ppo_config=campaign.PPO,
        expected_optimizer_steps=lineage['optimizer_steps'])
    require(state.progress.updates == 1753 and state.weights_sha256 == pair['weights_sha256'], 'Restored evaluation policy changed')
    protected = dict(lineage['protected'])
    for value in (lineage['lineage'], lineage['seed_ledger'], game_map, launch['native_manifest']):
        protected[value['path']] = value
    campaign.verify_protected(protected)
    return dict(source=identity, parent=selected, protected=protected, checkpoint=1753,
        model_checkpoint=state.checkpoint, weights_sha256=state.weights_sha256,
        optimizer_steps=lineage['optimizer_steps'], runtime=launch['runtime'], native=launch['native'],
        binary=launch['binary'], data=launch['data'], game_map=launch['game_map'], replay_binary=launch['replay_binary'],
        execution=execution, native_manifest=launch['native_manifest'])


def prepare(args):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_parent(args.lineage)
    output = copies.private_path(args.output or BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    reservation = reserve(output)
    output.mkdir(parents=True, mode=0o700)
    now, wall = time.monotonic(), time.time()
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=campaign.GUARD,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=now + args.seconds, hard_deadline_unix=wall + args.seconds)
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    require(plan['schema'] == SCHEMA + '-plan' and plan['source'] == source_identity(), 'Frozen evaluation source changed')
    require(plan['budget'] == BUDGET and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and plan['policy_support'] == campaign.GUARD, 'Evaluation contract changed')
    require(runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Evaluation engine runtime changed')
    bound(plan['replay_binary'])
    bound(plan['native_manifest'])
    reservation = read(bound(plan['reservation']))
    require(reservation in seed_ledger()['reservations'] and plan['schedule'] == reservation['schedule'],
            'Evaluation seed binding differs')
    campaign.verify_protected(plan['protected'])


def frozen_models(plan):
    from experiments.learner import selfplay_training as training
    pair = plan['parent']['checkpoint']
    loaded = tuple(training.load_policy(bound(pair['policy']), pair['metadata']) for _ in range(2))
    require(all(item.checkpoint == plan['model_checkpoint'] and item.weights_sha256 == plan['weights_sha256']
                for item in loaded), 'Frozen model identity differs from the published parent')
    models = tuple(item.model.eval().requires_grad_(False) for item in loaded)
    require(not ({p.data_ptr() for p in models[0].parameters()} & {p.data_ptr() for p in models[1].parameters()}),
            'Both frozen players must own separate model tensors')
    return models


def verify_capture(folder, plan, models, remaining):
    from experiments.learner.contextual_rollout import CollectedEpisode, behavior_guard_version
    from experiments.learner.contextual_model import weights_sha256
    from experiments.learner import selfplay_training as training
    from experiments.headless import contextual_replay
    raw = read(folder / 'neural-rollout.json')
    collected = CollectedEpisode.from_dict(raw)
    manifest = read(folder / 'manifest.json')
    require(collected.to_dict() == raw and json.loads(collected.manifest_json) == manifest,
            'Persisted paired evaluation capture changed')
    require(manifest['episode'] == '1' and manifest['seed'] == plan['schedule']['engine_seed']
            and manifest['environment'] == campaign.INTERFACE and manifest['status'] in ('completed', 'truncated')
            and manifest['identity']['native_runtime_sha256'] == plan['runtime']
            and all(manifest['identity'].get(key) == value for key, value in plan['native'].items()),
            'Evaluation manifest runtime, seed or terminal boundary differs')
    for player, (rollout, model, config) in enumerate(zip(collected.rollouts, models, collected.model_configs_json)):
        require(rollout.player == player and rollout.purpose == 'diagnostic' and rollout.origin == 'native'
                and rollout.role == ('learner' if player == 0 else 'frozen_opponent')
                and rollout.optimization_eligible is False and rollout.opponent_pool == campaign.POOL
                and rollout.behavior_checkpoint == rollout.opponent_checkpoint == plan['model_checkpoint']
                and rollout.weights_sha256 == rollout.opponent_weights_sha256 == plan['weights_sha256'] == weights_sha256(model)
                and json.loads(config) == model.config.to_dict(), 'Frozen evaluation policy identity or exclusion differs')
        require((behavior_guard_version(rollout.decisions) == campaign.GUARD or not rollout.decisions and not rollout.samples)
                and len(rollout.decisions) <= BUDGET['hard_choices_per_player']
                and all(rollout.history[decision.decision_seq].day <= 365 for decision in rollout.decisions),
                'Evaluation guard, choices or sampled-day boundary differs')
    originals = training._files(folder, manifest)
    policies = training.reconstruct_policies(collected, models, plan['schedule']['policy_seeds'], remaining)
    replay = contextual_replay.verify_episode(folder, binary=bound(plan['replay_binary']), data=Path(plan['data']),
        output=folder / 'evaluation-verification', timeout=remaining())
    require(replay.get('schema') == contextual_replay.REPORT_SCHEMA and replay.get('verified') is True
            and replay.get('histories', {}).get('verified') is True
            and set(replay['histories']['players']) == {'red', 'blue'}, 'Exact two-player native replay failed')
    require(read(folder / 'evaluation-verification/contextual-replay-verification.json') == replay
            and training._files(folder, manifest) == originals, 'Evidence changed during evaluation verification')
    remaining(); check_identity(plan)
    proof = dict(policies=policies, replay=replay, training_input=False, optimizer_updates=0,
                 choices_by_player=[len(rollout.decisions) for rollout in collected.rollouts])
    write(folder / 'evaluation-policy-reconstruction.json', proof, new=True)
    return proof, manifest


def collect(plan, models):
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from validation.probes.validate_hero_policy import wall_limit, retain_failed_evidence
    output = Path(plan['output'])
    deadline = min(time.monotonic() + BUDGET['capture_seconds'], plan['hard_deadline_monotonic'] - 30)
    wall = min(time.time() + BUDGET['capture_seconds'], plan['hard_deadline_unix'] - 30)
    launcher = DeadlineVcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']),
        deadline_monotonic=deadline, deadline_wall=wall, evidence_path=output / 'native-process-deadline.jsonl')
    class DayBoundedCollector(ContextualCollector):
        def act(self, request):
            require(request.record['day'] <= 365, 'A sampled evaluation action exceeds the preserved day365 bound')
            return super().act(request)
    collector = DayBoundedCollector(models, (plan['model_checkpoint'],) * 2, learner_player=0,
        opponent_pool=campaign.POOL, interface=campaign.INTERFACE, purpose='diagnostic', origin='native',
        policy_seeds=tuple(plan['schedule']['policy_seeds']), guard_version=campaign.GUARD)
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=365, max_actions=10240,
        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30),
        ledger=True, replay=True, interface=campaign.INTERFACE,
        agents={color: dict(checkpoint=plan['model_checkpoint'], kind='frozen-neural', purpose='diagnostic')
                for color in ('red', 'blue')})
    try:
        with wall_limit(campaign.remaining(dict(hard_deadline_monotonic=deadline, hard_deadline_unix=wall))):
            collected = collector.run_episode(env, episode=1, seed=plan['schedule']['engine_seed'],
                max_requests_per_player=9920, max_dialog_requests_per_player=320)
            write(env.episode_dir / 'neural-rollout.json', collected.to_dict(), new=True)
            return env.episode_dir
    except Exception as error:
        attempt = dict(status='failed', error=f'{type(error).__name__}: {error}', training_input=False)
        left = min(plan['hard_deadline_monotonic'] - time.monotonic(), plan['hard_deadline_unix'] - time.time())
        if left > 1:
            retain_failed_evidence(env, collector, attempt, error, cleanup_seconds=min(20, left))
        write(output / 'capture-failure.json', attempt, new=True)
        raise
    finally:
        try:
            env.close()
        finally:
            launcher.close_owned_groups(timeout=10); launcher.assert_closed()


def worker(path, sha):
    from validation.probes.validate_hero_policy import wall_limit
    require(binding(path)['sha256'] == sha, 'Evaluation plan changed')
    plan = read(path); output = Path(plan['output'])
    require(os.getppid() == plan['supervisor_pid'], 'Only the lease-owning supervisor may start evaluation')
    write(output / 'worker-claim.json', dict(pid=os.getpid(), plan_sha256=sha), new=True)
    report = dict(schema=SCHEMA + '-result', status='running', plan=binding(path), checkpoint=1753,
        physical_games=1, purpose='diagnostic', training_input=False, training_samples_admitted=0, optimizer_updates=0,
        budget=BUDGET, policy_support=campaign.GUARD, owned_cleanup_verified=False)
    write(output / 'report.json', report, new=True)
    try:
        with wall_limit(campaign.remaining(plan, 30)):
            check_identity(plan); campaign.configure_native(plan['execution'])
            models = frozen_models(plan)
        # The capture and verification timers are sequential: wall_limit rejects
        # nesting, while the process guardian bounds the entire worker.
        folder = collect(plan, models)
        with wall_limit(campaign.remaining(plan, 30)):
            proof, manifest = verify_capture(folder, plan, frozen_models(plan), lambda: campaign.remaining(plan, 30))
            require(binding(path)['sha256'] == sha, 'Evaluation plan changed during execution')
            report.update(status='passed', episode_directory=str(folder), episode_status=manifest['status'],
                termination_reason=manifest['termination_reason'], outcomes=manifest['outcomes_by_player'],
                rewards=manifest['rewards_by_player'], choices_by_player=proof['choices_by_player'],
                native_replay_verified=True, both_policies_reconstructed=True,
                proof=binding(folder / 'evaluation-policy-reconstruction.json'), owned_cleanup_verified=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            report['protected_inputs_unchanged'] = campaign.verify_protected(plan['protected'])
        except Exception as error:
            report.update(status='failed', protection_error=str(error))
        write(output / 'report.json', report)
    return report


def supervise(plan):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output, process, errors = Path(plan['output']), None, []
    plan_binding = binding(output / 'plan.json')
    try:
        with (output / 'worker.log').open('xb') as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_worker', '--plan', str(output / 'plan.json'),
                '--sha256', plan_binding['sha256']],
                deadline_monotonic=plan['hard_deadline_monotonic'], deadline_wall=plan['hard_deadline_unix'],
                evidence_path=output / 'worker-process-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            process.wait(timeout=campaign.remaining(plan) + 1)
    finally:
        if process is not None:
            try:
                close_owned_process(process, timeout=10)
            except Exception as error:
                errors.append(str(error))
        for evidence in output.glob('*native-process-deadline.jsonl'):
            try:
                cleanup_from_evidence(evidence, timeout=10)
            except Exception as error:
                errors.append(str(error))
        protected, protection_error = False, None
        try:
            protected = campaign.verify_protected(plan['protected'])
        except Exception as error:
            protection_error = f'{type(error).__name__}: {error}'
        plan_unchanged, plan_error = False, None
        try:
            plan_unchanged = binding(output / 'plan.json') == plan_binding
        except Exception as error:
            plan_error = f'{type(error).__name__}: {error}'
        write(output / 'shutdown.json', dict(owned_cleanup_verified=not errors, errors=errors,
            protected_inputs_unchanged=protected, protection_error=protection_error,
            plan_unchanged=plan_unchanged, plan_error=plan_error,
            returncode=process.returncode if process else None), new=True)
    report = read(output / 'report.json') if (output / 'report.json').exists() else {}
    require(process is not None and process.returncode == 0 and not errors and protected and plan_unchanged
            and report.get('plan') == plan_binding and report.get('status') == 'passed',
            'Evaluation stopped; preserve this evidence and use fresh seeds for another manual attempt')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--lineage', type=Path, default=campaign.LINEAGE)
    parser.add_argument('command', nargs='?', choices=('_worker',))
    parser.add_argument('--plan', type=Path); parser.add_argument('--sha256')
    args = parser.parse_args(argv)
    if args.command == '_worker':
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check, 'Worker requires a bound plan')
        return int(worker(args.plan, args.sha256)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    started = time.monotonic()
    try:
        if not args.run:
            checked = inspect_parent(args.lineage)
            result = dict(status='passed', checkpoint=checked['checkpoint'], parent=checked['parent'],
                source=checked['source'], budget=BUDGET, total_seconds=args.seconds,
                next_seed=seed_row(len(seed_ledger()['reservations'])), gameplay_performed=False,
                seeds_reserved=False, optimizer_updates=0, exact_model_adam_rng_progress_sampler_pool=True)
        else:
            try:
                with campaign.activity_lease():
                    result = supervise(prepare(args))
            except ValueError as error:
                if str(error) == 'Another local campaign owns the machine':
                    raise ValueError('Another campaign or its post-run report still owns the machine; '
                                     'wait for it to finish; no game or seeds started') from error
                raise
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
