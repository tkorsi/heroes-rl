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
CONTROLLER_COMMIT = '04cfe4d38a88d87c71a133a18dcbd895260624f6'
CONTROLLER_PATH = 'tools/run_selfplay_training.py'
STORAGE_PROGRESS_COMMIT = '3bd810060f0d1a211c985a723df4f67a10665a0e'
STORAGE_PROGRESS_PATHS = frozenset({
    CONTROLLER_PATH, 'experiments/learner/contextual_rollout.py',
    'experiments/learner/contextual_storage.py', 'experiments/learner/game_progress.py',
    'experiments/learner/selfplay_training.py', 'experiments/learner/local_script_training.py',
})
STORAGE_PROGRESS_NEW = frozenset({'experiments/learner/contextual_storage.py',
                                  'experiments/learner/game_progress.py'})
# Exact reviewed additions to policy support; legacy support retains its rules.
# These digests bind the reviewed opt-in observation and support release.
RECRUIT_ONLY_SOURCE_SHA256 = {
    'experiments/learner/local_script_training.py': 'ea50cf114414f73f119494aaa06af19ba26f4bf8ff54613bc8b94924f4031acd',
    'engine/ExperimentReplay/ContextualReplay.cpp': 'bebe4b98fc48abe048e32240767baacd8f443d71631dc3ca666178c2aada43e4',
    'engine/ExplorerRL/Contextual.cpp': '09ddb160a8c70e060f3e2b2093a2879e24aabab078a0b5f4770bbc0ec7a2db63',
    'engine/ExplorerRL/ContextualEvidence.cpp': '4e31338745b9cfbe867d99949112fb63340a3a389fe82458dff839ac51574154',
    'engine/ExplorerRL/ContextualEvidence.h': 'f43e34373fd36a52f5b3d377ed876af513b47db0ef5334b78a0adb5f0017631b',
    'engine/ExplorerRL/ExplorerRL.cpp': '97ca85b5363236143fa78fe3cce7b734acdff0a2b7fa4aae6b978fcb7c226d94',
    'engine/ExplorerRL/ExplorerRL.h': 'cac4746edd8d9f4443024202ca968db620d127d321fc52364691c5a494fc0b9d',
    'experiments/headless/contextual_env.py': 'a1ca9947b7c21262fa2b71520868da537c690fbaf3da8c55bacdea8419e640cd',
    'experiments/headless/contextual_protocol.py': '62dc02cb177181b7b51e5972d06c6cf73426ca09188324e89b255344eb78eebc',
    'experiments/headless/contextual_replay.py': '4a5173bd3d87ee1729f763daa067b5a2fad60e2cf319fbde852b3e43378355ab',
    'experiments/headless/contextual_replay_history.py': '5cf37e415f90c435c5fa910de10f9c35ede6feaad4012873c46d021cc5fca529',
    'experiments/learner/contextual.py': '69893b70595307a50cc936927eae7fd624ebe0355a040464864f798cfe676883',
    'experiments/learner/contextual_model.py': '14f505e78974cb1fd8c2e26afa956996f7cd23caef49a671a376fc00e44f4cfe',
    'experiments/learner/interaction_guard.py': 'd5b295bcdca6c37da6aa050007526b7b11e4da574ccae0693d61038a093418c4',
}
# Exact display-only delta after the native army-information acceptance. These
# files do not alter policy inputs, sampling, learning or replay verification.
DAY_PROGRESS_SOURCES = {
    'experiments/learner/contextual_rollout.py': {'before': '6c9439d52c304c504bc7a17227112f23b2cb7c47796fd404dcd8ccf1a9ca1b71',
        'after': '66cb89c252208f8de6468fa82b6fdc38af1eb25415ab04da73dacdb8d8c426ab'},
    'experiments/learner/game_progress.py': {'before': 'e5c1290de42be830392684a861e7508f4b34dc7469e4ac078d95566632ea057b',
        'after': '5d79ad8b24314b2096a092cd51b825cf58a47797ec3315e34abc6617857ad447'},
    'experiments/learner/local_script_training.py': {'before': 'ea50cf114414f73f119494aaa06af19ba26f4bf8ff54613bc8b94924f4031acd',
        'after': '0925bf57440f910e8d1241d7d883c5326e0336d6ff9cf0c928724f36b457a9f8'},
}
BASE = campaign.ARTIFACTS / 'selfplay-one-game-2x'
SCHEMA = 'heroes-selfplay-one-game-2x-v1'
BUDGET = dict(ordinary_choices_per_player=9920, reserved_dialog_choices_per_player=320,
              hard_choices_per_player=10240, max_days=365, capture_seconds=1200)
ENGINE_SEED, POLICY_SEED = 1_960_000_000, 1_962_000_000
require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound


def verify_runtime_sources(implementation):
    """Bind the base plus exact reviewed storage/UI and support additions."""
    require(STORAGE_PROGRESS_PATHS <= implementation.keys(),
            'Evaluation source identity omitted a reviewed storage/progress dependency')
    require(RECRUIT_ONLY_SOURCE_SHA256.keys() <= implementation.keys(),
            'Evaluation source identity omitted a reviewed recruit-only dependency')
    def committed(commit, names):
        archive = subprocess.check_output(['git', 'archive', commit, *sorted(names)], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as source:
            files = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                     for member in source.getmembers() if member.isfile()}
        require(files.keys() == names, 'Committed evaluation source inventory differs')
        return files
    original = committed(BASE_COMMIT, implementation.keys() - STORAGE_PROGRESS_NEW)
    original.update(committed(STORAGE_PROGRESS_COMMIT, STORAGE_PROGRESS_PATHS))
    original.update(RECRUIT_ONLY_SOURCE_SHA256)
    original.update({name: pins['after'] for name, pins in DAY_PROGRESS_SOURCES.items()})
    require(original == implementation, 'Evaluation changed an existing learner, adapter, engine or replay source')


def verify_native_validation_sources(accepted, current):
    """Retain the original native receipt, allowing only a reviewed UI delta."""
    require(type(accepted) is dict and type(current) is dict,
            'Passive-army runtime lacks complete two-seat native validation sources')
    if accepted == current:
        return dict(kind='exact_native_validation_sources', changed=[])
    expected = dict(accepted)
    for name, pins in DAY_PROGRESS_SOURCES.items():
        require(accepted.get(name) == pins['before'] and current.get(name) == pins['after'],
                'Passive-army runtime lacks complete two-seat native validation source transition')
        expected[name] = pins['after']
    require(expected == current, 'Passive-army runtime lacks complete two-seat native validation source agreement')
    return dict(kind='reviewed_display_only_day_progress', changed=DAY_PROGRESS_SOURCES,
                native_capture_repeated=False)


def source_identity():
    """Every reused source matches an explicit reviewed content identity."""
    from validation.probes.validate_capabilities import capability_implementation_identity
    from experiments.learner.inference import source_identity as learner_identity
    implementation = capability_implementation_identity()
    learner = learner_identity(campaign.INTERFACE)
    implementation.update(learner['source_sha256'])
    for name in (CONTROLLER_PATH, 'tools/benchmark_copied_training.py'):
        implementation[name] = binding(ROOT / name)['sha256']
    verify_runtime_sources(implementation)
    return dict(checkout=copies.checkout_identity(ROOT), source=learner,
        unchanged_runtime_base=BASE_COMMIT, implementation=implementation,
        reporting_controller_base=CONTROLLER_COMMIT,
        storage_progress_base=STORAGE_PROGRESS_COMMIT,
        recruit_only_sources=RECRUIT_ONLY_SOURCE_SHA256,
        day_progress_sources=DAY_PROGRESS_SOURCES,
        driver=binding(__file__), launcher=binding(ROOT / 'play-one-selfplay-2x.sh'))


def policy_support(plan):
    from experiments.learner.interaction_guard import RECRUIT_ONLY_VERSION
    version = plan.get('policy_support')
    require(version in (campaign.GUARD, RECRUIT_ONLY_VERSION),
            'Unknown or missing one-game policy support')
    return version


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


def verify_capture(folder, plan, models, remaining, *, identity_check=None):
    from experiments.headless.contextual_protocol import declared_army_information
    from experiments.learner.contextual_rollout import behavior_guard_version
    from experiments.learner.contextual_storage import load_collected
    from experiments.learner.contextual_model import weights_sha256
    from experiments.learner import selfplay_training as training
    from experiments.headless import contextual_replay
    guard = policy_support(plan)
    collected = load_collected(folder / 'neural-rollout.json')
    manifest = read(folder / 'manifest.json')
    require(declared_army_information(manifest, campaign.INTERFACE) == plan.get('army_information'),
            'Evaluation army information extension differs from its plan')
    require(json.loads(collected.manifest_json) == manifest,
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
        require((behavior_guard_version(rollout.decisions) == guard or not rollout.decisions and not rollout.samples)
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
    remaining(); (check_identity if identity_check is None else identity_check)(plan)
    proof = dict(policies=policies, replay=replay, training_input=False, optimizer_updates=0, policy_support=guard,
                 choices_by_player=[len(rollout.decisions) for rollout in collected.rollouts])
    write(folder / 'evaluation-policy-reconstruction.json', proof, new=True)
    return proof, manifest


def collect(plan, models):
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.contextual_storage import save_collected
    from experiments.learner.game_progress import GameProgress, progress_path
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from validation.probes.validate_hero_policy import wall_limit, retain_failed_evidence
    guard = policy_support(plan)
    output = Path(plan['output'])
    progress = GameProgress(progress_path(output, 1), episode=1,
        ordinary=BUDGET['ordinary_choices_per_player'], reserved=BUDGET['reserved_dialog_choices_per_player'])
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
        policy_seeds=tuple(plan['schedule']['policy_seeds']), guard_version=guard)
    observation_options = ({'army_information': plan['army_information']} if 'army_information' in plan else {})
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=365, max_actions=10240,
        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30),
        ledger=True, replay=True, interface=campaign.INTERFACE,
        agents={color: dict(checkpoint=plan['model_checkpoint'], kind='frozen-neural', purpose='diagnostic')
                for color in ('red', 'blue')}, **observation_options)
    try:
        with wall_limit(campaign.remaining(dict(hard_deadline_monotonic=deadline, hard_deadline_unix=wall))):
            collected = collector.run_episode(env, episode=1, seed=plan['schedule']['engine_seed'],
                max_requests_per_player=9920, max_dialog_requests_per_player=320, progress=progress)
            progress.phase('storing')
            save_collected(env.episode_dir / 'neural-rollout.json', collected)
            return env.episode_dir
    except Exception as error:
        progress.phase('failed')
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


def worker(path, sha, *, identity_check=None, model_loader=None, result_schema=SCHEMA, checkpoint=1753):
    from validation.probes.validate_hero_policy import wall_limit
    from experiments.learner.game_progress import progress_path, set_phase
    identity_check = check_identity if identity_check is None else identity_check
    model_loader = frozen_models if model_loader is None else model_loader
    require(binding(path)['sha256'] == sha, 'Evaluation plan changed')
    plan = read(path); output = Path(plan['output'])
    require(os.getppid() == plan['supervisor_pid'], 'Only the lease-owning supervisor may start evaluation')
    write(output / 'worker-claim.json', dict(pid=os.getpid(), plan_sha256=sha), new=True)
    report = dict(schema=result_schema + '-result', status='running', plan=binding(path), checkpoint=checkpoint,
        physical_games=1, purpose='diagnostic', training_input=False, training_samples_admitted=0, optimizer_updates=0,
        budget=BUDGET, policy_support=policy_support(plan), owned_cleanup_verified=False)
    if 'army_information' in plan:
        report['army_information'] = plan['army_information']
    write(output / 'report.json', report, new=True)
    try:
        with wall_limit(campaign.remaining(plan, 30)):
            identity_check(plan); campaign.configure_native(plan['execution'])
            models = model_loader(plan)
        # The capture and verification timers are sequential: wall_limit rejects
        # nesting, while the process guardian bounds the entire worker.
        folder = collect(plan, models)
        set_phase(progress_path(output, 1), 'verifying')
        with wall_limit(campaign.remaining(plan, 30)):
            proof, manifest = verify_capture(folder, plan, model_loader(plan), lambda: campaign.remaining(plan, 30),
                                             identity_check=identity_check)
            require(binding(path)['sha256'] == sha, 'Evaluation plan changed during execution')
            report.update(status='passed', episode_directory=str(folder), episode_status=manifest['status'],
                termination_reason=manifest['termination_reason'], outcomes=manifest['outcomes_by_player'],
                rewards=manifest['rewards_by_player'], choices_by_player=proof['choices_by_player'],
                native_replay_verified=True, both_policies_reconstructed=True,
                proof=binding(folder / 'evaluation-policy-reconstruction.json'), owned_cleanup_verified=True)
            set_phase(progress_path(output, 1), 'verified')
    except Exception as error:
        set_phase(progress_path(output, 1), 'failed')
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            report['protected_inputs_unchanged'] = campaign.verify_protected(plan['protected'])
        except Exception as error:
            report.update(status='failed', protection_error=str(error))
        write(output / 'report.json', report)
    return report


def supervise(plan, *, worker_script=None, result_schema=SCHEMA, checkpoint=1753):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from experiments.learner.game_progress import ProgressDisplay
    output, process, errors = Path(plan['output']), None, []
    display = ProgressDisplay(output / 'progress')
    plan_binding = binding(output / 'plan.json')
    try:
        with (output / 'worker.log').open('xb') as log:
            script = Path(__file__).resolve() if worker_script is None else Path(worker_script).resolve()
            process = guarded_popen([sys.executable, str(script), '_worker', '--plan', str(output / 'plan.json'),
                '--sha256', plan_binding['sha256']],
                deadline_monotonic=plan['hard_deadline_monotonic'], deadline_wall=plan['hard_deadline_unix'],
                evidence_path=output / 'worker-process-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            display.wait(process, timeout=campaign.remaining(plan) + 1)
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
        display.finish(failed=bool(errors) or process is None or process.returncode != 0)
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
            and report.get('plan') == plan_binding and report.get('status') == 'passed'
            and report.get('schema') == result_schema + '-result' and report.get('checkpoint') == checkpoint
            and report.get('policy_support') == policy_support(plan)
            and report.get('army_information') == plan.get('army_information'),
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
