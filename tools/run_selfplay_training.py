#!/usr/bin/env python3
"""Manual self-play continuation with bounded PPO batches and verified diagnostics.

The original Nullkiller campaign is read-only. Diagnostics, consumed seeds and
failed batches cannot become training inputs. Manual continuations use new run
directories, fresh reservations and the latest published lineage checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import queue
import threading
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = Path('/Users/orsi/Projects/Heroes-artifacts')
ORIGINAL = ARTIFACTS / 'local-script-training/original'
BASE = ARTIFACTS / 'selfplay-training'
NATIVE = ARTIFACTS / 'native-learner-builds/remove-telemetry-v1-attempt1/manifest.json'
NATIVE_SHA = '53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573'
CONTINUATION_BASE = '28c57a26416715d28a3b0250b980f81a93960cd5'
BASE_ACCEPTANCE_SHA = '37bdf290216778c565997aab35517d387826684881563e69975751f63150d992'
LINEAGE = BASE / 'selfplay100-lineage.json'
CONTINUATION_ACCEPTANCE = BASE / 'continuation-acceptance-v1.json'
INTERFACE = 'adventure-capabilities-v1'
GUARD = 'visible-interaction-budget-revealed-v1'
POOL = 'selfplay-batch-start-v1'
SCHEMA = 'heroes-selfplay-campaign-v1'
PPO = dict(gamma=1.0, gae_lambda=1.0, clip_ratio=0.2, value_clip=0.2,
           value_coefficient=0.5, entropy_coefficient=0.0, max_grad_norm=0.5,
           epochs=1, decisions_per_window=16, windows_per_minibatch=2,
           normalize_advantages=True)
# Separate from original 40M/50M, easier-map 20M/30M and native diagnostic seeds.
NAMESPACES = {'training': (1_950_000_000, 1_952_000_000),
              'diagnostic': (1_940_000_000, 1_942_000_000)}
TEN_BASE = ARTIFACTS / 'selfplay-batch10-training'
TEN_LAUNCH = 'heroes-selfplay-batch-ten-campaign-v1'
TEN_DISK_RESERVE_GIB = 30
CYCLE_LAUNCH = "heroes-selfplay-ten-then-check-campaign-v1"
CYCLE_MODE = False


def select_cycle_mode():
    global CYCLE_MODE, TEN_BASE, TEN_LAUNCH, TEN_NAMESPACES
    CYCLE_MODE = True
    TEN_BASE = ARTIFACTS / "selfplay-ten-then-check"
    TEN_LAUNCH = CYCLE_LAUNCH
    TEN_NAMESPACES = {"training": (1_980_000_000, 1_982_000_000)}


def cycle_games():
    from experiments.learner import selfplay_training as training
    return training.CYCLE_GAMEPLAY["training_games"] if CYCLE_MODE else 10


def cycle_batch_sizes(games):
    """Balance complete PPO batches while bounding capture memory."""
    require(type(games) is int and 10 <= games <= 1000, "Use --games between 10 and 1000")
    count = (games + 99) // 100
    size, extra = divmod(games, count)
    return [size + (index < extra) for index in range(count)]


def cycle_width():
    return cycle_games() + 1 if CYCLE_MODE else 10


TEN_NAMESPACES = {'training': (1_970_000_000, 1_972_000_000)}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(Path(path).read_bytes())


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def bound(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'}
            and binding(value['path']) == value, 'Bound artifact changed')
    return Path(value['path'])


def write(path, value, *, new=False):
    """Atomic publication; a latest pointer is written only after its pair verifies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if new:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_name('.' + path.name + '-' + secrets.token_hex(6))
        try:
            write_bytes(temporary, payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def private_output(path):
    from tools.benchmark_copied_training import private_path
    return private_path(path, fresh=True)


@contextmanager
def activity_lease():
    """Use existing official lock inodes without rewriting their contents."""
    paths = (ARTIFACTS / 'local-script-training/owner.lock', ORIGINAL / 'owner.lock')
    with ExitStack() as stack:
        for path in paths:
            require(path.is_file() and not path.is_symlink(), 'Missing original campaign owner lock')
            stream = stack.enter_context(path.open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another local campaign owns the machine') from error
        yield


def protected_inputs(pointer):
    paths = {ORIGINAL / name for name in ('latest-checkpoint.json', 'latest-run.json', 'sequence-ledger.json')}
    paths.update(Path(item['path']) for item in pointer['parent'].values())
    run = Path(pointer['run_dir'])
    paths.update(run / name for name in ('session.json', 'summary.json', 'shutdown.json', 'campaign-report.json'))
    backup = Path(pointer['backup'])
    paths.update(backup / name for name in ('after-policy.pt', 'after-resume.pt', 'preservation-receipt.json'))
    return {str(path): binding(path) for path in sorted(paths)}


def verify_protected(values):
    for value in values.values():
        bound(value)
    return True


def source_identity():
    from tools.benchmark_copied_training import checkout_identity
    from experiments.learner.inference import source_identity as learner_identity
    from validation.probes.validate_capabilities import capability_implementation_identity
    checkout = checkout_identity(ROOT)
    implementation = capability_implementation_identity()
    for path in (Path(__file__).resolve(), ROOT / 'tools/benchmark_copied_training.py',
                 ROOT / 'tools/validate_selfplay_cutoffs.py'):
        implementation[str(path.relative_to(ROOT))] = binding(path)['sha256']
    return dict(checkout=checkout, source=learner_identity(INTERFACE), implementation=implementation)


def validate_configuration(cfg):
    expected = dict(interface=INTERFACE, adam={'lr': 0.0003}, ppo=PPO, max_days=365,
                    ordinary_choice_limit=4960, reserved_dialog_choices=160,
                    max_total_choices=5120, max_optimizer_steps=640, attempt_wall_seconds=600,
                    allow_verified_training_prefix=False)
    for key, value in expected.items():
        require(cfg.get(key) == value, 'Parent learning or safety setting differs: ' + key)
    result = deepcopy(cfg)
    result['training_games'] = 4
    return result


def inspect_parent():
    from experiments.learner import local_script_training as local
    from experiments.learner.native_execution import native_execution_choice
    from experiments.headless.contextual_evidence import runtime_identity
    pointer = read(ORIGINAL / 'latest-checkpoint.json')
    require(pointer['progress']['updates'] == 1728, 'This experiment starts only from verified checkpoint 1728')
    protected = protected_inputs(pointer)
    state = local.restore_parent(pointer['parent'])
    require(state.progress.updates == 1728, 'Restored checkpoint differs')
    plan = read(bound(pointer['parent']['plan']))
    require(plan.get('policy_support') == GUARD, 'Parent revealed interaction guard differs')
    cfg = validate_configuration(plan['configuration'])
    # This is a distinct experiment gate, not an override of official acceptance.
    execution = native_execution_choice(NATIVE, NATIVE_SHA, cpu_cap=16, require_acceptance=False)
    require(execution['native_learner']['prefix_threads'] == 16
            and execution['collection_workers'] == 4, 'This release requires the validated 16-CPU Mac budget')
    require(runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Actual engine runtime changed')
    verify_protected(protected)
    return pointer, state, cfg, execution, protected, plan


def reserve_schedule(output, mode, count, *, expected_ledger=None):
    """Reserve the entire run before launch, including games never reached."""
    require(mode in NAMESPACES and type(count) is int
            and (count == 8 if mode == 'diagnostic' else 4 <= count <= 100 and count % 4 == 0),
            'Wrong experiment size')
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'namespace.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path = BASE / 'seed-reservations.json'
        if expected_ledger is not None:
            require(bound(expected_ledger) == ledger_path.resolve(), 'Continuation seed ledger changed after inspection')
        ledger = read(ledger_path) if ledger_path.exists() else {'schema': 'heroes-selfplay-seeds-v1', 'reservations': []}
        require(ledger['schema'] == 'heroes-selfplay-seeds-v1', 'Unknown seed ledger')
        offset = sum(item['count'] for item in ledger['reservations'] if item['mode'] == mode)
        engine, policy = NAMESPACES[mode]
        require(offset + count < 1_000_000, 'Self-play namespace exhausted')
        reservation = dict(mode=mode, output=str(output), offset=offset, count=count,
                           engine_base=engine + offset, policy_base=policy + 2 * offset,
                           consumed_including_unstarted=True)
        require(all(item['output'] != str(output) for item in ledger['reservations']), 'Output already reserved')
        ledger['reservations'].append(reservation)
        write(ledger_path, ledger)
    return reservation


def seed_preview(mode, count):
    """Inspect the durable namespace without reserving or changing any seed."""
    ledger = read(BASE / 'seed-reservations.json')
    require(ledger.get('schema') == 'heroes-selfplay-seeds-v1', 'Unknown seed ledger')
    offsets = {name: 0 for name in NAMESPACES}
    outputs = set()
    for row in ledger['reservations']:
        namespace = row['mode']
        require(namespace in NAMESPACES and type(row['count']) is int and row['count'] > 0
                and row['output'] not in outputs and row['offset'] == offsets[namespace]
                and row['consumed_including_unstarted'] is True, 'Consumed seed ledger is inconsistent')
        engine, policy = NAMESPACES[namespace]
        require(row['engine_base'] == engine + row['offset']
                and row['policy_base'] == policy + 2 * row['offset'], 'Consumed seed namespace differs')
        offsets[namespace] += row['count']; outputs.add(row['output'])
    engine, policy = NAMESPACES[mode]
    require(offsets[mode] + count < 1_000_000, 'Self-play namespace exhausted')
    return dict(count=count, consumed_before=offsets[mode], engine_base=engine + offsets[mode],
                policy_base=policy + 2 * offsets[mode], reserved=False)


def schedule(reservation, start, count):
    require(type(start) is int and type(count) is int and 0 <= start < start + count <= reservation['count'],
            'Schedule exceeds its consumed reservation')
    return [dict(episode=i + 1, phase=reservation['mode'], learner_player=(start + i) % 2,
                 engine_seed=reservation['engine_base'] + start + i,
                 policy_seeds=[reservation['policy_base'] + 2 * (start + i),
                               reservation['policy_base'] + 2 * (start + i) + 1]) for i in range(count)]


def stopped_run_proof(output, launch, report, shutdown):
    require(shutdown.get('owned_cleanup_verified') is True
            and shutdown.get('protected_inputs_unchanged') is True, 'Previous self-play ownership is not closed')
    if report.get('status') == 'completed' and shutdown.get('returncode') == 0:
        return None
    path = Path(output) / 'stop-finalization.json'
    proof = read(path)
    require(proof.get('schema') == 'heroes-selfplay-stop-finalization-v1'
            and proof.get('status') in ('stopped_by_user', 'stopped')
            and proof.get('owned_cleanup_verified') is True
            and proof.get('protected_inputs_unchanged') is True
            and proof.get('no_unpublished_experience_admitted') is True,
            'Previous interrupted run has no verified stop finalization')
    for name in ('launch', 'report', 'shutdown'):
        require(bound(proof[name]) == Path(output) / (name + '.json'), 'Stop finalization binding differs')
    latest = Path(output) / 'latest-checkpoint.json'
    require(proof.get('latest') == (binding(latest) if latest.exists() else None),
            'Stop finalization latest publication changed')
    return binding(path)


def initial_checkpoint_state(original):
    from experiments.learner import builtin_training_resume as resume, checkpoints as cp
    plan, report = read(bound(original['plan'])), read(bound(original['report']))
    progress, steps, sampler = resume._completion(plan, report)
    policy = cp._read(bound(original['policy']), 'policy', resume._metadata(plan, original['plan']['sha256']),
                      (cp.BUILTIN_GAMEPLAY_PROVENANCE,), INTERFACE)
    require(policy['weights_sha256'] == report['after_checkpoint']['weights_sha256'],
            'Initial checkpoint weights differ from the original verified report')
    return dict(progress=vars(progress), optimizer_steps=steps, sampler=sampler,
                checkpoint=policy['checkpoint'], weights_sha256=policy['weights_sha256'])


def check_relocated_pair(pair, published):
    expected = deepcopy(published)
    for name in ('policy', 'resume'):
        bound(pair[name]); bound(published[name])
        require(all(pair[name][key] == published[name][key] for key in ('sha256', 'bytes')),
                'Relocated continuation checkpoint bytes differ from the published pair')
        expected[name] = pair[name]
    require(pair == expected, 'Relocated continuation progress, Adam, RNG or metadata differs')


def inspect_lineage(path):
    """Follow only registered launches and their published, complete batch pairs."""
    path = Path(path).resolve()
    lineage = read(path)
    require(lineage.get('schema') == 'heroes-selfplay-lineage-v1' and lineage.get('target_games') == 100
            and type(lineage.get('runs')) is list and lineage['runs'], 'Unknown self-play lineage')
    initial = read(bound(lineage['runs'][0]))
    require(initial['source']['checkout']['commit'] == CONTINUATION_BASE
            and initial['mode'] == 'training' and initial['reservation']['count'] == 100,
            'Lineage must begin at the accepted 100-game self-play launch')
    original = initial['original_parent']
    ledger_binding = binding(BASE / 'seed-reservations.json')
    seed_preview('training', 0)
    reservations = read(BASE / 'seed-reservations.json')['reservations']
    initial_state = initial_checkpoint_state(original)
    initial_progress = initial_state['progress']
    require(initial_progress['updates'] == 1728, 'Wrong initial self-play checkpoint')
    decisions, steps = initial_progress['decisions'], initial_state['optimizer_steps']
    previous_policy, previous_checkpoint, previous_weights = original['policy'], initial_state['checkpoint'], initial_state['weights_sha256']
    total, latest, selected, seen = 0, None, None, set()
    protected = dict(initial['protected'])
    for number, launch_binding in enumerate(lineage['runs']):
        launch_path = bound(launch_binding)
        launch = read(launch_path); output = launch_path.parent
        require(str(output) not in seen and launch['output'] == str(output) and launch['mode'] == 'training'
                and launch['original_parent'] == original and launch['configuration'] == initial['configuration']
                and launch['native_manifest'] == initial['native_manifest']
                and launch['runtime'] == initial['runtime'] and launch['native'] == initial['native']
                and launch.get('admitted_before', 0) == total,
                'Registered self-play launch is not the next unchanged lineage window')
        require(launch['reservation'] in reservations, 'Registered launch has no durable consumed seed reservation')
        seen.add(str(output))
        if number:
            require(launch.get('resume_parent', {}).get('latest') == latest,
                    'Continuation does not start from the preceding published checkpoint')
            require(launch['resume_parent']['checkpoint'] == selected['checkpoint'], 'Continuation selected another checkpoint pair')
            check_relocated_pair(launch['resume_pair'], selected['checkpoint'])
            previous_policy = launch['resume_pair']['policy']
        verify_protected(launch['protected'])
        report_path, shutdown_path = output / 'report.json', output / 'shutdown.json'
        report, shutdown = read(report_path), read(shutdown_path)
        require(report.get('source') == launch['source'] and report.get('mode') == 'training',
                'Previous report identity differs')
        finalization = stopped_run_proof(output, launch, report, shutdown)
        for value in (launch_binding, binding(report_path), binding(shutdown_path), finalization):
            if value is not None:
                protected[value['path']] = value
        pointer_path = output / 'latest-checkpoint.json'
        if not pointer_path.exists():
            require(report.get('admitted_games') == 0, 'Missing published checkpoint for admitted games')
            continue
        pointer = read(pointer_path)
        require(pointer.get('schema') == 'heroes-selfplay-latest-v1' and pointer.get('purpose') == 'training'
                and pointer.get('campaign_id') == launch['campaign_id']
                and type(pointer.get('batch')) is int and 0 <= pointer['batch'] < 25,
                'Invalid latest self-play publication')
        window_games = (pointer['batch'] + 1) * 4
        require(window_games <= launch['reservation']['count']
                and pointer['admitted_games'] == total + window_games
                and report.get('admitted_games') == window_games,
                'Published batch and admission counters differ')
        for index in range(pointer['batch'] + 1):
            batch = report['batches'][index]
            require(batch.get('index') == index and batch.get('status') == 'checkpoint_verified'
                    and len(batch.get('attempts', [])) == 4
                    and all(row.get('status') == 'passed' and row.get('replay_verified') is True
                            and row.get('both_policies_reconstructed') is True for row in batch['attempts']),
                    'Published parent contains an incomplete batch')
            plan_path = bound(batch['plan']); plan = read(plan_path)
            pair = batch['after_checkpoint']; progress = pair['progress']
            ppo = batch['ppo']
            require(type(ppo.get('samples')) is int and ppo['samples'] > 0
                    and type(ppo.get('optimizer_steps')) is int and 0 < ppo['optimizer_steps'] <= 640,
                    'Published batch lacks exact sample and optimizer-step counters')
            decisions += ppo['samples']; steps += ppo['optimizer_steps']
            require(plan_path == output / f'batch-{index:03d}' / 'plan.json'
                    and plan['launch'] == launch_binding and plan['configuration'] == initial['configuration']
                    and plan['source'] == launch['source']['source']
                    and plan['implementation'] == launch['source']['implementation']
                    and pair['metadata']['purpose'] == 'training'
                    and pair['metadata']['initial_checkpoint'] == original['policy']
                    and pair['metadata']['campaign_id'] == launch['campaign_id']
                    and pair['metadata']['plan_sha256'] == batch['plan']['sha256']
                    and pair['metadata']['opponent_checkpoint'] == plan['opponent_checkpoint']
                    and pair['exact_model_adam_rng_progress_pool'] is True
                    and plan['learner_checkpoint'] == previous_checkpoint
                    and plan['learner_weights_sha256'] == plan['opponent_weights_sha256'] == previous_weights
                    and plan['checkpoint_source'] == plan['opponent_checkpoint_source']
                    and plan['checkpoint_source']['opponent_checkpoint'] == previous_policy
                    and progress['updates'] == 1728 + (total + (index + 1) * 4) // 4
                    and progress['rollouts'] == initial_progress['rollouts'] + total + (index + 1) * 4
                    and progress['decisions'] == decisions
                    and pair['sampler_state']['next_batch_index'] == (total + (index + 1) * 4) // 4
                    and pair['sampler_state']['consumed_schedule_offset'] == total + (index + 1) * 4
                    and pair['sampler_state']['optimizer_steps'] == steps
                    and pair['sampler_state']['batch_optimizer_steps'] == ppo['optimizer_steps']
                    and pair['sampler_state']['original_parent_sampler'] == initial_state['sampler'],
                    'Published self-play checkpoint ancestry differs')
            for value in (batch['plan'], pair['policy'], pair['resume'], pair['metadata']['opponent_checkpoint']):
                bound(value); protected[value['path']] = value
            previous_policy, previous_checkpoint, previous_weights = pair['policy'], pair['checkpoint'], pair['weights_sha256']
        require(pointer['checkpoint'] == pair and pointer['plan'] == batch['plan']
                and pointer['cumulative_progress'] == pair['progress'], 'Latest pointer differs from its verified batch')
        total += window_games
        latest = binding(pointer_path); protected[latest['path']] = latest
        selected = dict(latest=latest, checkpoint=pair, plan=batch['plan'], launch=launch_binding)
    require(selected is not None and 0 < total <= 100 and total % 4 == 0, 'No eligible published self-play parent')
    verify_protected(protected)
    bound(ledger_binding)
    return dict(lineage=binding(path), lineage_value=lineage, initial_launch=initial,
                selected=selected, admitted_games=total, remaining_games=100 - total, protected=protected,
                optimizer_steps=steps, seed_ledger=ledger_binding)


def validate_continuation_acceptance(path, identity, lineage):
    """Reuse frozen native evidence only for the reviewed resume-only transition."""
    gate = read(path)
    require(gate.get('schema') == 'heroes-selfplay-continuation-acceptance-v1'
            and gate.get('accepted_for_manual_continuation') is True and gate.get('source') == identity,
            'Manual continuation requires its exact reviewed source acceptance')
    initial = lineage['initial_launch']
    require(all(read(bound(value))['source'] in (initial['source'], identity)
                for value in lineage['lineage_value']['runs']), 'Lineage contains an unreviewed source transition')
    base = read(bound(gate['base_acceptance']))
    require(gate['base_acceptance']['sha256'] == BASE_ACCEPTANCE_SHA
            and initial['acceptance'] == gate['base_acceptance']
            and base['source']['checkout']['commit'] == CONTINUATION_BASE
            and base['source'] == initial['source'] and base['accepted_for_training'] is True
            and gate['initial_launch'] == lineage['lineage_value']['runs'][0]
            and gate['native_manifest'] == base['native_manifest'] == binding(NATIVE),
            'Continuation does not descend from the accepted native release')
    for key in ('diagnostic_report', 'copied_update_report', 'offline_tests', 'prepare_vcmi_check', 'cutoff_report'):
        bound(base[key])
    diagnostic = read(base['diagnostic_report']['path'])
    require(diagnostic['source'] == base['source'] and diagnostic['status'] == 'completed'
            and len(diagnostic['batches']) == 2 and diagnostic['optimizer_updates'] == 0,
            'Original frozen native proof differs')
    for batch in diagnostic['batches']:
        bound(batch['plan'])
        require(len(batch['attempts']) == 4, 'Original frozen cohort differs')
        for attempt in batch['attempts']:
            require(attempt['status'] == 'passed' and attempt['replay_verified'] is True
                    and attempt['both_policies_reconstructed'] is True, 'Original frozen policy proof failed')
            for value in attempt['evidence'].values():
                bound(value)
    validate_resume_source_transition(base['source'], identity, gate.get('reviewed_files'))
    tests = read(bound(gate['offline_tests']))
    restored = read(bound(gate['restore_check']))
    require(tests.get('status') == 'passed' and tests.get('source') == identity
            and restored.get('status') == 'passed' and restored.get('source') == identity
            and restored.get('checkpoint') == 1730
            and restored.get('published_latest') == binding(Path(initial['output']) / 'latest-checkpoint.json')
            and restored.get('exact_model_adam_rng_progress_sampler_pool') is True,
            'Current continuation lacks offline and full-state restore validation')
    require(gate['launcher'] == binding(ROOT / 'continue-selfplay.sh'), 'Manual launcher differs from reviewed source')
    return binding(path)


def validate_resume_source_transition(base_source, identity, reviewed_files):
    import ast
    old = dict(base_source['implementation']); old.update(base_source['source']['source_sha256'])
    current = dict(identity['implementation']); current.update(identity['source']['source_sha256'])
    require(old.keys() == current.keys(), 'Continuation cannot add an unreviewed runtime module')
    changed = {name: dict(before=old[name], after=current[name]) for name in old if old[name] != current[name]}
    allowed = {'tools/run_selfplay_training.py', 'experiments/learner/selfplay_training.py',
               'experiments/learner/local_script_training.py'}
    require(set(changed) <= allowed and reviewed_files == changed,
            'Continuation changes unreviewed runtime behavior')
    def historical(name):
        data = subprocess.check_output(['git', 'show', CONTINUATION_BASE + ':' + name], cwd=ROOT)
        require(hashlib.sha256(data).hexdigest() == old[name], 'Original source hash is not committed')
        return ast.parse(data)
    helper = 'experiments/learner/selfplay_training.py'
    before, after = historical(helper), ast.parse((ROOT / helper).read_bytes())
    added = [node for node in after.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == 'restore_checkpoint_pair']
    require(len(added) == 1 and ast.dump(before) == ast.dump(ast.Module(
        body=[node for node in after.body if node not in added], type_ignores=[])),
        'Resume helper changed a preexisting learner/admission definition')
    adapter = 'experiments/learner/local_script_training.py'
    pins = {'FAST_PATH_SOURCES', 'FAST_PATH_TREE', 'LOCAL_ADAPTER_TRANSITION_AST'}
    def without_pins(tree):
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in pins)]
        return ast.dump(tree)
    require(without_pins(historical(adapter)) == without_pins(ast.parse((ROOT / adapter).read_bytes())),
            'Original adapter transition must change reviewed pins only')
    controller = 'tools/run_selfplay_training.py'
    old_functions = {node.name: ast.dump(node) for node in historical(controller).body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node) for node in ast.parse((ROOT / controller).read_bytes()).body
                     if isinstance(node, ast.FunctionDef)}
    unchanged = ('capture_worker', 'collect_batch', 'close_collection_jobs', 'batch_plan',
                 'batch_phase_deadlines', 'phase_remaining', 'configure_native', 'validate_configuration', 'schedule')
    require(all(old_functions[name] == new_functions[name] for name in unchanged),
            'Manual continuation changed capture, verification, policy support, or phase behavior')
    return changed


def inspect_continuation(args):
    from experiments.learner import selfplay_training as training
    from experiments.learner.native_execution import native_execution_choice
    from experiments.headless.contextual_evidence import runtime_identity
    identity = source_identity()
    lineage = inspect_lineage(args.lineage)
    acceptance = validate_continuation_acceptance(args.acceptance, identity, lineage)
    initial = lineage['initial_launch']
    cfg = validate_configuration(initial['configuration'])
    require(binding(NATIVE) == initial['native_manifest']
            and runtime_identity(Path(initial['binary']).parent) == initial['runtime'],
            'Accepted engine or native learner changed')
    execution = native_execution_choice(NATIVE, NATIVE_SHA, cpu_cap=16, require_acceptance=False)
    require(execution == initial['execution'], 'Manual continuation resource contract changed')
    pair = lineage['selected']['checkpoint']
    state = training.restore_checkpoint_pair(pair, expected_metadata=pair['metadata'], expected_ppo_config=PPO,
        expected_optimizer_steps=lineage['optimizer_steps'])
    require(state.progress.updates == 1728 + lineage['admitted_games'] // 4,
            'Restored checkpoint differs from published lineage')
    verify_protected(lineage['protected'])
    return identity, lineage, state, cfg, execution, acceptance


def prepare_continuation(args):
    from tools.benchmark_copied_training import copy_verified
    identity, lineage, _, cfg, execution, acceptance = inspect_continuation(args)
    count = lineage['remaining_games']
    require(count > 0, 'The 100-game self-play target is already complete')
    seconds = args.hours * 3600
    require(math.isfinite(seconds) and 120 <= seconds <= 36000, 'Use a finite budget from 120 seconds to ten hours')
    output = private_output(args.output or BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)))
    output.mkdir(parents=True, mode=0o700)
    selected, initial = lineage['selected'], lineage['initial_launch']
    pair = deepcopy(selected['checkpoint'])
    for name in ('policy', 'resume'):
        target = output / 'input' / ('parent-' + name + '.pt')
        copy_verified(bound(pair[name]), target, pair[name])
        pair[name] = binding(target)
        lineage['protected'][pair[name]['path']] = pair[name]
    check_relocated_pair(pair, selected['checkpoint'])
    reservation = reserve_schedule(output, 'training', count, expected_ledger=lineage['seed_ledger'])
    now, wall = time.monotonic(), time.time()
    launch = {key: deepcopy(initial[key]) for key in ('original_parent', 'parent_descriptor',
        'parent_policy_sha256', 'policy_support', 'binary', 'data', 'game_map', 'replay_binary', 'native', 'runtime')}
    launch.update(schema=SCHEMA, campaign_id=output.name, mode='training', output=str(output),
        supervisor_pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(), source=identity,
        protected=lineage['protected'], reservation=reservation, configuration=cfg, execution=execution,
        native_manifest=binding(NATIVE), acceptance=acceptance, maximum_seconds=seconds,
        hard_deadline_monotonic=now + seconds, hard_deadline_unix=wall + seconds,
        admitted_before=lineage['admitted_games'], target_games=100, resume_parent=selected, resume_pair=pair,
        parent_optimizer_steps=lineage['optimizer_steps'],
        lineage_path=str(Path(args.lineage).resolve()))
    write(output / 'launch.json', launch, new=True)
    bound(lineage['lineage'])
    registered = deepcopy(lineage['lineage_value'])
    registered['runs'].append(binding(output / 'launch.json'))
    write(args.lineage, registered)
    return launch


def validate_acceptance(path, identity, pointer):
    gate = read(path)
    require(gate.get('schema') == 'heroes-selfplay-acceptance-v1' and gate.get('accepted_for_training') is True,
            'Self-play requires its separate reviewed acceptance receipt')
    require(gate['source'] == identity and gate['parent_policy_sha256'] == pointer['parent']['policy']['sha256']
            and gate['native_manifest'] == binding(NATIVE), 'Self-play acceptance source, parent or native build differs')
    diagnostic = read(bound(gate['diagnostic_report']))
    require(diagnostic.get('status') == 'completed' and diagnostic.get('mode') == 'diagnostic'
            and diagnostic.get('optimizer_updates') == 0 and diagnostic.get('training_samples_admitted') == 0
            and diagnostic.get('owned_cleanup_verified') is True and diagnostic.get('protected_inputs_unchanged') is True
            and diagnostic.get('source') == identity and diagnostic.get('parent_policy_sha256') == gate['parent_policy_sha256']
            and diagnostic.get('native_manifest') == gate['native_manifest'] and diagnostic.get('policy_support') == GUARD,
            'Frozen diagnostic cohort is incomplete or belongs to another release')
    validate_configuration(diagnostic['configuration'])
    from experiments.learner import selfplay_training as training
    parent_plan = read(bound(pointer['parent']['plan']))
    validate_cutoff_controls(gate['cutoff_report'], identity, pointer, parent_plan)
    require(diagnostic.get('runtime') == parent_plan['runtime']
            and diagnostic.get('native') == parent_plan['native'], 'Diagnostic engine or map identity differs')
    for batch in diagnostic['batches']:
        plan_file = bound(batch['plan'])
        plan = training._plan(plan_file, batch['plan']['sha256'])
        require(plan['mode'] == 'diagnostic' and plan['source'] == identity['source']
                and plan['implementation'] == identity['implementation']
                and plan['runtime'] == parent_plan['runtime'] and plan['native'] == parent_plan['native']
                and plan['configuration'] == diagnostic['configuration']
                and plan['checkpoint_source']['purpose'] == 'diagnostic'
                and plan['checkpoint_source']['initial_checkpoint']['sha256'] == gate['parent_policy_sha256']
                and len(batch['attempts']) == 4, 'Diagnostic batch binding differs')
        for row, attempt in zip(plan['schedule'], batch['attempts'], strict=True):
            require(all(attempt.get(key) == value for key, value in row.items()), 'Diagnostic schedule differs')
    attempts = [attempt for batch in diagnostic['batches'] for attempt in batch['attempts']]
    require(len(attempts) == 8, 'Exactly eight diagnostic games are required')
    for attempt in attempts:
        require(attempt.get('status') == 'passed' and attempt.get('cleanup_verified') is True
                and attempt.get('replay_verified') is True and attempt.get('both_policies_reconstructed') is True
                and attempt.get('episode_status') in ('completed', 'truncated'), 'Unverified diagnostic game')
        for evidence in attempt['evidence'].values():
            bound(evidence)
        evidence = attempt['evidence']
        proof = read(evidence['training-policy-reconstruction.json']['path'])
        replay = read(evidence['training-verification/contextual-replay-verification.json']['path'])
        require(proof.get('replay_verified') is True and replay.get('verified') is True
                and proof.get('histories', {}).get('verified') is True
                and len(proof.get('policies', [])) == 2, 'Diagnostic lacks persisted two-policy replay proof')
        for player, policy in enumerate(proof['policies']):
            require(policy.get('player') == player and policy.get('maximum_absolute_difference') == 0
                    and policy.get('private_memory_reset') is True and policy.get('weights_unchanged') is True,
                    'Diagnostic policy reconstruction was not exact and private')
    copied = read(bound(gate['copied_update_report']))
    require(copied.get('status') == 'passed' and copied.get('diagnostic_optimizer_updates') == 1
            and copied.get('official_updates') == 0 and copied.get('exact_model_adam_rng_progress_pool') is True
            and copied.get('source') == identity and copied.get('parent_policy_sha256') == gate['parent_policy_sha256']
            and copied.get('native_manifest') == gate['native_manifest'],
            'Missing copied optimizer and checkpoint roundtrip control')
    tests = read(bound(gate['offline_tests']))
    require(tests.get('status') == 'passed' and tests.get('source') == identity, 'Offline tests belong to another source')
    engine = read(bound(gate['prepare_vcmi_check']))
    engine_files = {name: sha for name, sha in identity['implementation'].items() if name.startswith('engine/')}
    require(engine.get('status') == 'passed' and engine.get('engine_inventory') == engine_files,
            'Engine source check belongs to another recipe')
    return binding(path)


def validate_cutoff_controls(value, identity, pointer, parent_plan):
    """Require real two-seat capability cutoff coverage before self-play training."""
    from tools import validate_selfplay_cutoffs as controls
    report = read(bound(value))
    plan = read(bound(report['plan']))
    require(report.get('schema') == controls.SCHEMA and report.get('status') == 'passed'
            and report.get('not_started') == 0 and report.get('training_input') is False
            and report.get('training_samples') == report.get('optimizer_updates') == 0
            and report.get('owned_cleanup_verified') is True and report.get('protected_inputs_unchanged') is True,
            'Native cutoff controls are incomplete or not excluded from training')
    cutoff_source = plan.get('source', {})
    require(report.get('source') == cutoff_source
            and cutoff_source.get('checkout') == identity['checkout'] and cutoff_source.get('source') == identity['source']
            and all(cutoff_source.get('implementation', {}).get(name) == digest
                    for name, digest in identity['implementation'].items())
            and report.get('native_runtime') == plan.get('native_runtime') == parent_plan['runtime']
            and plan.get('policy_support') == GUARD
            and plan.get('full_model_adam_rng_progress_sampler_pool_restored') is True
            and report['parent_descriptor'] == plan['parent_descriptor']
            and plan['parent_descriptor']['policy']['sha256'] == pointer['parent']['policy']['sha256'],
            'Native cutoff controls belong to another source, runtime or checkpoint')
    controls.check_identity(plan)
    wanted = [(case, player) for case in ('optional-capability', 'mandatory-level-up') for player in ('red', 'blue')]
    attempts = report.get('attempts', [])
    require(len(attempts) == len(plan['schedule']) == 4
            and [(row.get('case'), row.get('player')) for row in attempts] == wanted,
            'All four native cutoff cases and seats are required')
    for planned, attempt in zip(plan['schedule'], attempts, strict=True):
        require(all(attempt.get(key) == item for key, item in planned.items())
                and attempt.get('status') == 'passed' and attempt.get('cleanup_verified') is True
                and attempt.get('history_audit_verified') is True and attempt.get('replay_verified') is True
                and attempt.get('episode_status') == 'truncated' and attempt.get('training_input') is False,
                'A native cutoff control is incomplete')
        for evidence in attempt['evidence'].values():
            bound(evidence)
        require(read(attempt['evidence']['replay']['path']).get('verified') is True,
                'Native cutoff replay evidence did not pass')
        if attempt['case'] == 'mandatory-level-up':
            proof = attempt.get('mixed_policy_reconstruction', {})
            require(proof.get('passed') is True and proof.get('maximum_absolute_difference') == 0
                    and proof.get('private_memory_reset') is True and proof.get('weights_unchanged') is True
                    and proof.get('training_input') is False and proof.get('actions_resampled', 0) >= 1,
                    'Mandatory cutoff response was not reconstructed from the actual policy')
    shutdown = read(Path(value['path']).parent / 'shutdown.json')
    require(shutdown.get('owned_cleanup_verified') is True and shutdown.get('returncode') == 0
            and shutdown.get('protected_inputs_unchanged') is True, 'Native cutoff supervisor did not close cleanly')


def prepare(args, mode):
    from tools.benchmark_copied_training import copy_verified
    output = private_output(args.output)
    identity = source_identity()
    pointer, _, cfg, execution, protected, parent_plan = inspect_parent()
    acceptance = validate_acceptance(args.acceptance, identity, pointer) if mode == 'training' else None
    seconds = args.hours * 3600 if mode == 'training' else args.seconds
    require(math.isfinite(seconds) and 120 <= seconds <= 36000, 'Use a finite budget from 120 seconds to ten hours')
    output.mkdir(parents=True, mode=0o700)
    descriptor = {}
    for name, original in pointer['parent'].items():
        target = output / 'input' / (name + Path(original['path']).suffix)
        copy_verified(Path(original['path']), target, original)
        descriptor[name] = binding(target)
    reservation = reserve_schedule(output, mode, 100 if mode == 'training' else 8)
    now, wall = time.monotonic(), time.time()
    launch = dict(schema=SCHEMA, campaign_id=output.name, mode=mode, output=str(output),
                  supervisor_pid=os.getpid(),
                  created_utc=datetime.now(timezone.utc).isoformat(), source=identity,
                  parent_descriptor=descriptor, original_parent=pointer['parent'], protected=protected,
                  parent_policy_sha256=pointer['parent']['policy']['sha256'], reservation=reservation,
                  configuration=cfg, execution=execution, policy_support=GUARD,
                  binary=parent_plan['binary'], data=parent_plan['data'], game_map=parent_plan['game_map'],
                  replay_binary=(parent_plan['replay_binary'] if isinstance(parent_plan['replay_binary'], dict)
                                 else binding(parent_plan['replay_binary'])),
                  native=parent_plan['native'], runtime=parent_plan['runtime'],
                  native_manifest=binding(NATIVE), acceptance=acceptance, maximum_seconds=seconds,
                  hard_deadline_monotonic=now + seconds, hard_deadline_unix=wall + seconds)
    write(output / 'launch.json', launch, new=True)
    return launch


def load_launch(path, sha):
    require(binding(path)['sha256'] == sha, 'Launch plan hash changed')
    launch = read(path)
    if launch.get('schema') == TEN_LAUNCH:
        return load_ten_launch(path, sha)
    require(launch['schema'] == SCHEMA and source_identity() == launch['source'], 'Frozen experiment source changed')
    require(launch['native_manifest'] == binding(NATIVE), 'Native learner changed')
    from experiments.headless.contextual_evidence import runtime_identity
    require(runtime_identity(Path(launch['binary']).parent) == launch['runtime'], 'Engine runtime changed')
    bound(launch['replay_binary'])
    verify_protected(launch['protected'])
    if 'resume_parent' in launch:
        parent = read(bound(launch['resume_parent']['latest']))
        require(parent['checkpoint'] == launch['resume_parent']['checkpoint'], 'Published continuation parent changed')
        check_relocated_pair(launch['resume_pair'], parent['checkpoint'])
    return launch


def remaining(launch, reserve=0):
    if launch['hard_deadline_monotonic'] is None and launch['hard_deadline_unix'] is None:
        return float('inf')
    left = min(launch['hard_deadline_monotonic'] - time.monotonic(), launch['hard_deadline_unix'] - time.time()) - reserve
    if left <= 0:
        raise TimeoutError('Self-play experiment deadline exhausted')
    return left


def batch_phase_deadlines(launch):
    """Preserve the parent's phase offsets separately from each capture timeout."""
    now, wall = time.monotonic(), time.time()
    cfg = launch['configuration']
    if launch['hard_deadline_monotonic'] is None and launch['hard_deadline_unix'] is None:
        return {phase: dict(phase=phase, configured_seconds=None, cleanup_reserve_seconds=0,
                            hard_deadline_monotonic=None, hard_deadline_unix=None)
                for phase in ('training', 'update')}
    return {phase: dict(phase=phase, configured_seconds=seconds,
                       cleanup_reserve_seconds=cfg['cleanup_reserve_seconds'],
                       hard_deadline_monotonic=min(now + seconds, launch['hard_deadline_monotonic']),
                       hard_deadline_unix=min(wall + seconds, launch['hard_deadline_unix']))
            for phase, seconds in cfg['phase_deadlines_seconds'].items() if phase in ('training', 'update')}


def phase_remaining(budget, reserve=None):
    try:
        return remaining(budget, budget['cleanup_reserve_seconds'] if reserve is None else reserve)
    except TimeoutError as error:
        raise TimeoutError(f"Self-play {budget['phase']} phase deadline exhausted") from error


def configure_native(execution, *, policy_backend='cpu'):
    import torch
    from experiments.learner import ppo
    from experiments.learner.native_execution import check_native_execution
    from experiments.learner.native_backend import verified_manifest
    from experiments.learner.mlx_policy import set_policy_backend
    check_native_execution(execution)
    native = execution['native_learner']
    verified_manifest(native['manifest'], native['sha256'])
    torch.set_num_threads(1)
    ppo.NATIVE_EXECUTION = native
    ppo.PREFIX_HELPERS = 0
    ppo.REPLAY_TEXT_LEAVES = False
    ppo.TEXT_REPLAY_WORKERS = 0
    ppo.EXPERIMENTS = frozenset()
    set_policy_backend(policy_backend)


def batch_plan(launch, state, index, start, count, previous_policy):
    from experiments.learner import checkpoints as cp, selfplay_training as training
    folder = Path(launch['output']) / f'batch-{index:03d}'
    folder.mkdir()
    inputs = folder / 'input'; inputs.mkdir()
    metadata = training.metadata(binding(Path(launch['output']) / 'launch.json')['sha256'],
                                 launch['original_parent']['policy'], previous_policy,
                                 campaign_id=launch['campaign_id'], purpose=launch['mode'])
    opponent = deepcopy(state.model).eval().requires_grad_(False)
    opponent_id = f"{launch['campaign_id']}-batch-{index:03d}-frozen"
    for filename, model, checkpoint in (('learner.pt', state.model, state.checkpoint),
                                         ('opponent.pt', opponent, opponent_id)):
        cp.save_policy_checkpoint(inputs / filename, model, checkpoint=checkpoint,
            provenance=cp.SELFPLAY_GAMEPLAY_PROVENANCE, source_metadata=metadata,
            opponent_pool=state.pool, generators={'minibatch': state.generator}, interface=INTERFACE)
    plan = dict(schema='heroes-selfplay-batch-v1', campaign_id=launch['campaign_id'], mode=launch['mode'],
                output=str(folder), execution_output=str(folder / 'execution'),
                launch=binding(Path(launch['output']) / 'launch.json'),
                source=launch['source']['source'], implementation=launch['source']['implementation'],
                native=launch['native'], runtime=launch['runtime'], binary=launch['binary'],
                data=launch['data'], game_map=launch['game_map'], replay_binary=launch['replay_binary'],
                configuration=launch['configuration'], policy_support=GUARD, execution=launch['execution'],
                checkpoint=binding(inputs / 'learner.pt'), opponent_checkpoint=binding(inputs / 'opponent.pt'),
                checkpoint_source=metadata, opponent_checkpoint_source=metadata,
                learner_checkpoint=state.checkpoint, opponent_checkpoint_id=opponent_id,
                learner_weights_sha256=cp.model_weight_digest(state.model), opponent_weights_sha256=cp.model_weight_digest(opponent),
                opponent_pool=POOL, schedule=schedule(launch['reservation'], start, count))
    write(folder / 'plan.json', plan, new=True)
    return folder / 'plan.json', plan, opponent


def capture_worker(args):
    # Registration rechecks source and input bindings. A short game must not
    # finish while its coordinator is still registering this live child.
    require(sys.stdin.buffer.readline(66) == (args.token + '\n').encode('ascii'),
            'Capture requires its coordinator registration release')
    from experiments.learner import selfplay_training as training
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.contextual_storage import save_collected, save_live_collected
    from experiments.learner.game_progress import GameProgress, progress_path
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    require(binding(args.plan)['sha256'] == args.sha256, 'Batch plan hash changed')
    plan = read(args.plan); launch = load_launch(plan['launch']['path'], plan['launch']['sha256'])
    require((args.deadline is None and args.wall_deadline is None
             and launch['hard_deadline_monotonic'] is None and launch['hard_deadline_unix'] is None)
            or (args.deadline is not None and args.wall_deadline is not None
                and launch['hard_deadline_monotonic'] is not None and launch['hard_deadline_unix'] is not None
                and args.deadline <= launch['hard_deadline_monotonic']
                and args.wall_deadline <= launch['hard_deadline_unix']),
            'Worker cannot extend the campaign deadline')
    configure_native(plan['execution'], policy_backend=plan['configuration'].get('policy_backend', 'cpu'))
    row = plan['schedule'][args.index]
    ten = plan.get('schema') in (training.TEN_SCHEMA, training.CYCLE_SCHEMA)
    cfg = plan['configuration']
    ordinary = cfg['ordinary_choice_limit'] if ten else 4960
    reserved = cfg['reserved_dialog_choices'] if ten else 160
    days = cfg['max_days'] if ten else 365
    hard_choices = cfg['max_total_choices'] if ten else 5120
    progress = GameProgress(progress_path(launch['output'], row['episode']), episode=row['episode'],
                            ordinary=ordinary, reserved=reserved)
    learner = training.load_policy(bound(plan['checkpoint']), plan['checkpoint_source'])
    opponent = training.load_policy(bound(plan['opponent_checkpoint']), plan['opponent_checkpoint_source'])
    seat = row['learner_player']
    models = (learner.model, opponent.model) if seat == 0 else (opponent.model, learner.model)
    checkpoints = (learner.checkpoint, opponent.checkpoint) if seat == 0 else (opponent.checkpoint, learner.checkpoint)
    output = Path(plan['execution_output']); output.mkdir(exist_ok=True)
    launcher = DeadlineVcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']),
        deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
        evidence_path=output / f"native-{row['episode']:06d}-deadline.jsonl",
        **({'equipment_policy': plan['equipment_policy']} if ten else {}))
    class DayBoundedCollector(ContextualCollector):
        def act(self, request):
            require(request.record['day'] <= days, 'A sampled policy action exceeds its declared day bound')
            return super().act(request)

    capture_only = plan.get('schema') == training.CYCLE_SCHEMA and plan['mode'] == 'training'
    collector = DayBoundedCollector(models, checkpoints, learner_player=seat, opponent_pool=POOL,
        interface=INTERFACE, purpose=plan['mode'], origin='native', policy_seeds=tuple(row['policy_seeds']),
        guard_version=plan['policy_support'] if ten else GUARD,
        capture_only=capture_only)
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=days, max_actions=hard_choices,
        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20,
        bridge_timeout=0 if args.deadline is None else 30, unlimited_wait=args.deadline is None),
        ledger=not capture_only, replay=True, interface=INTERFACE, capture_only=capture_only,
        agents={color: {'checkpoint': checkpoint, 'kind': 'frozen-neural', 'purpose': plan['mode']}
                for color, checkpoint in zip(('red', 'blue'), checkpoints)},
        **({'army_information': plan['army_information']} if ten else {}))
    cleanup = False
    try:
        collected = collector.run_episode(env, episode=row['episode'], seed=row['engine_seed'],
                                         max_requests_per_player=ordinary, max_dialog_requests_per_player=reserved,
                                         progress=progress)
        folder = env.episode_dir
        progress.phase('storing')
        (save_live_collected if capture_only else save_collected)(folder / 'neural-rollout.json', collected)
        require(read(folder / 'manifest.json')['status'] in ('completed', 'truncated'), 'Technical capture failure')
        progress.phase('captured' if capture_only else 'awaiting verification')
    except Exception as error:
        progress.phase('failed')
        from validation.probes.validate_hero_policy import retain_failed_evidence
        failure = dict(status='failed', error=f'{type(error).__name__}: {error}',
                       outcomes={'red': None, 'blue': None}, rewards={'red': None, 'blue': None})
        seconds = (20 if args.deadline is None else
                   min(args.deadline - time.monotonic(), args.wall_deadline - time.time()))
        if seconds > 1:
            retain_failed_evidence(env, collector, failure, error, cleanup_seconds=min(20, seconds))
        write(output / f"failure-{row['episode']:06d}.json", failure, new=True)
        raise
    finally:
        try:
            env.close()
        finally:
            launcher.close_owned_groups()
            cleanup = True
    require(cleanup, 'Capture cleanup was not verified')
    load_launch(plan['launch']['path'], plan['launch']['sha256'])
    message = dict(schema='heroes-selfplay-capture-complete-v1', token=args.token, episode=row['episode'],
                   files={name: {k: v for k, v in binding(folder / name).items() if k != 'path'}
                          for name in training.capture_artifact_names(plan)})
    print(json.dumps(message), flush=True)


def discard_training_streams(folder):
    """Fresh training streams are temporary once native PPO programs are prepared."""
    names = {'contextual-replay.jsonl', 'bridge.jsonl', 'history-v8.jsonl',
             'live-history-v8.jsonl', 'replay.jsonl'}
    removed = []
    for path in Path(folder).rglob('*.jsonl'):
        if path.name in names:
            require(not path.is_symlink(), 'Training stream must be a regular owned file')
            removed.append(dict(path=str(path.relative_to(folder)), bytes=path.stat().st_size))
            path.unlink()
    write(Path(folder) / 'temporary-streams-removed.json',
          dict(training_verification='not_performed', files=removed), new=True)


def require_cycle_memory_budget():
    """Leave substantial headroom for macOS and temporary history compilation."""
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    processes = [tuple(map(int, line.split())) for line in rows.splitlines()]
    owned = {os.getpid()}
    while True:
        descendants = owned | {pid for pid, parent, rss in processes if parent in owned}
        if descendants == owned:
            break
        owned = descendants
    resident = sum(rss * 1024 for pid, parent, rss in processes if pid in owned)
    physical = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
    require(resident <= max(8 * 1024**3, physical - 48 * 1024**3),
            'Cycle memory budget reached; stopping owned processes and preserving the checkpoint')
    pressure = subprocess.check_output(['memory_pressure', '-Q'], text=True)
    free_percent = int(pressure.rsplit(':', 1)[-1].strip().removesuffix('%'))
    require(free_percent >= 20,
            'System memory pressure reached the cycle reserve; preserving the checkpoint')


class CaptureDispatch:
    """Replenish actor processes while main performs deadline-bound preparation.

    The dispatcher never loads model tensors, admits data or runs PPO. The caller owns
    all launched groups and joins this dispatcher before process cleanup/update.
    """

    def __init__(self, count, *, workers, launch, remaining, ready=None):
        require(type(count) is int and count > 0 and type(workers) is int and workers > 0,
                'Invalid capture dispatcher bounds')
        self.count, self.workers = count, workers
        self.launch, self.remaining, self.ready = launch, remaining, ready
        self.stop, self.results = threading.Event(), queue.Queue()
        self.thread = threading.Thread(target=self._run, name='capture-dispatch', daemon=True)

    def _run(self):
        active, next_index, current_index = [], 0, 0
        def start_next():
            nonlocal next_index, current_index
            self.remaining()
            if next_index < self.count and not self.stop.is_set():
                index = current_index = next_index
                next_index += 1
                try:
                    active.append(self.launch(index))
                except BaseException as error:
                    self.results.put(dict(index=index, error=error))
                    self.stop.set()
        try:
            while len(active) < self.workers and next_index < self.count and not self.stop.is_set():
                start_next()
            while active and not self.stop.is_set():
                self.remaining()
                finished = [job for job in active if job['process'].poll() is not None]
                if not finished:
                    self.stop.wait(.05)
                    continue
                for job in finished:
                    current_index = job['index']
                    if self.stop.is_set():
                        return
                    job['stdout'] = job['process'].communicate(timeout=min(5, self.remaining()))[0]
                    active.remove(job)
                    if job['process'].returncode != 0:
                        self.results.put(job)
                        self.stop.set()
                        return
                    if self.ready is not None:
                        self.ready(job)
                    # Refill before handing heavy preparation to the main thread.
                    start_next()
                    self.results.put(job)
            if not self.stop.is_set():
                self.results.put(None)
        except BaseException as error:
            self.results.put(dict(index=current_index, error=error))
            self.stop.set()

    def __iter__(self):
        self.thread.start()
        while True:
            self.remaining()
            try:
                job = self.results.get(timeout=.2)
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError('Capture dispatcher exited without its completion marker')
                continue
            if job is None:
                return
            yield job

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
        return 'Capture dispatcher failed to stop' if self.thread.is_alive() else None


def collect_batch(plan_path, plan, learner, opponent, launch, save_attempt, *, phase_deadline):
    from experiments.learner import selfplay_training as training
    from experiments.learner.game_progress import progress_path, set_phase
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from validation.probes.validate_hero_policy import wall_limit
    output = Path(plan['execution_output']); output.mkdir()
    jobs, receipts, attempts = [], [], []
    current_index, current_stage, dispatch = None, 'capture', None
    live = plan.get('schema') == training.CYCLE_SCHEMA and plan['mode'] == 'training'
    bindings = []
    worker_env = dict(os.environ)
    if live:
        # Cap dense-math pools per actor; the existing parent/native PPO thread
        # configuration remains unchanged.
        worker_env.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})

    def launch_capture(index):
        phase_remaining(phase_deadline)
        row = plan['schedule'][index]
        folder = output / f"episode-{row['episode']:06d}"
        capability = bindings[index] if live else training.create_capture_binding(
            plan_path, binding(plan_path)['sha256'], index, folder)
        token = secrets.token_hex(32)
        allowance = plan['configuration']['attempt_wall_seconds']
        deadline = (min(time.monotonic() + allowance, phase_deadline['hard_deadline_monotonic'] - 2)
                    if allowance is not None else None)
        wall = (min(time.time() + allowance, phase_deadline['hard_deadline_unix'] - 2)
                if allowance is not None else None)
        log = (output / f'worker-{index:02d}.log').open('xb')
        command = [sys.executable, str(Path(__file__).resolve()), '_capture', '--plan', str(plan_path),
                   '--sha256', binding(plan_path)['sha256'], '--index', str(index), '--token', token,
                   '--deadline', str(deadline), '--wall-deadline', str(wall)]
        process = guarded_popen(command, deadline_monotonic=deadline, deadline_wall=wall,
            evidence_path=output / f'worker-{index:02d}-deadline.jsonl', cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            **({"env": worker_env} if live else {}))
        job = dict(process=process, log=log, binding=capability, index=index, folder=folder,
                   deadline=deadline, wall_deadline=wall,
                   deadlines=dict(capture_seconds=allowance, capture_deadline_monotonic=deadline,
                                  capture_deadline_unix=wall, collection_and_verification=phase_deadline))
        jobs.append(job)
        training.register_capture_launch(capability, process, token)
        process.stdin.write((token + '\n').encode('ascii'))
        process.stdin.close()
        process.stdin = None
        save_attempt(index, dict(**row, status='running', outcomes={'red': None, 'blue': None},
                                rewards={'red': None, 'blue': None}, deadlines=job['deadlines']))
        return job

    def queued(job):
        row = plan['schedule'][job['index']]
        save_attempt(job['index'], dict(**row, status='preparing',
            outcomes={'red': None, 'blue': None}, rewards={'red': None, 'blue': None},
            deadlines=job['deadlines']))

    def historical_waves():
        nonlocal current_index
        width = 10 if plan.get('schema') == training.TEN_SCHEMA else 4
        for first in range(0, len(plan['schedule']), width):
            wave = []
            for index in range(first, min(first + width, len(plan['schedule']))):
                current_index = index
                wave.append(launch_capture(index))
            yield from wave

    try:
        if live:
            require(plan['configuration'].get('collection_workers') == 10,
                    'Fifty-game collection requires ten independent actors')
            # Policy loading/binding stays on main before any dispatcher thread:
            # no concurrent model construction or global RNG contexts.
            for index, row in enumerate(plan['schedule']):
                current_index = index
                phase_remaining(phase_deadline)
                bindings.append(training.create_capture_binding(plan_path, binding(plan_path)['sha256'], index,
                    output / f"episode-{row['episode']:06d}"))
            dispatch = CaptureDispatch(len(plan['schedule']), workers=10, launch=launch_capture,
                remaining=lambda: phase_remaining(phase_deadline), ready=queued)
            captures = dispatch
        else:
            captures = historical_waves()
        for job in captures:
            current_index = job['index']
            current_stage = 'capture'
            if 'error' in job:
                raise job['error']
            process = job['process']
            wait_seconds = phase_remaining(phase_deadline)
            if plan.get('schema') not in (training.TEN_SCHEMA, training.CYCLE_SCHEMA):
                wait_seconds = min(605, wait_seconds)
            stdout = job['stdout'] if live else process.communicate(
                timeout=wait_seconds if math.isfinite(wait_seconds) else None)[0]
            require(process.returncode == 0, 'Self-play capture failed; no retries or partial batch admission')
            close_owned_process(process, timeout=min(5, phase_remaining(phase_deadline)))
            cleanup_from_evidence(output / f"native-{plan['schedule'][job['index']]['episode']:06d}-deadline.jsonl",
                                  timeout=min(5, phase_remaining(phase_deadline)))
            require(len(stdout) <= 65536, 'Oversized capture completion')
            training.accept_capture_completion(job['binding'], json.loads(stdout))
            def verification_remaining():
                return phase_remaining(phase_deadline)
            if live:
                current_stage = 'preparation'
                set_phase(progress_path(launch['output'], plan['schedule'][job['index']]['episode']), 'preparing PPO')
                preparation_started = time.monotonic()
                with (wall_limit(verification_remaining()) if math.isfinite(verification_remaining()) else nullcontext()):
                    receipt = training.prepare_live_capture(job['binding'], learner, opponent,
                                                            remaining=verification_remaining)
                manifest = read(job['folder'] / 'manifest.json')
                row = plan['schedule'][job['index']]
                attempt = dict(**row, status='captured', verification_status='not_performed',
                    preparation_seconds=time.monotonic() - preparation_started,
                    cleanup_verified=True, replay_verified=False, both_policies_reconstructed=False,
                    episode_status=manifest['status'], termination_reason=manifest['termination_reason'],
                    outcomes=manifest['outcomes_by_player'], rewards=manifest['rewards_by_player'],
                    episode_directory=str(job['folder']), sampled_by_player=list(receipt.choices),
                    natural_terminal_after_horizon=manifest.get('natural_terminal_after_horizon'),
                    evidence={name: binding(job['folder'] / name) for name in
                        ('manifest.json', 'neural-rollout.json', 'training-capture-receipt.json')},
                    verification=receipt.to_dict(), deadlines=job['deadlines'])
                discard_training_streams(job['folder'])
                set_phase(progress_path(launch['output'], row['episode']), 'captured')
            else:
                set_phase(progress_path(launch['output'], plan['schedule'][job['index']]['episode']), 'verifying')
                current_stage = 'verification'
                with (wall_limit(verification_remaining()) if math.isfinite(verification_remaining()) else nullcontext()):
                    receipt = training.verify_training_capture(job['binding'], learner, opponent,
                        replay_binary=Path(plan['replay_binary']['path']), data=Path(plan['data']),
                        remaining=verification_remaining)
                set_phase(progress_path(launch['output'], plan['schedule'][job['index']]['episode']), 'verified')
                manifest = read(job['folder'] / 'manifest.json')
                row = plan['schedule'][job['index']]
                attempt = dict(**row, status='passed', cleanup_verified=True, replay_verified=True,
                    both_policies_reconstructed=True, episode_status=manifest['status'],
                    termination_reason=manifest['termination_reason'], outcomes=manifest['outcomes_by_player'],
                    rewards=manifest['rewards_by_player'], episode_directory=str(job['folder']),
                    sampled_by_player=[len(item.decisions) for item in receipt.collected.rollouts],
                    evidence={name: binding(job['folder'] / name) for name in ('manifest.json', 'neural-rollout.json',
                        'training-policy-reconstruction.json', 'training-verification/contextual-replay-verification.json',
                        'training-verification/native-report.json', 'training-receipt.json')},
                    verification=receipt.to_dict(), deadlines=job['deadlines'])
                if plan.get('schema') in (training.TEN_SCHEMA, training.CYCLE_SCHEMA):
                    attempt['natural_terminal_after_horizon'] = manifest.get('natural_terminal_after_horizon')
            attempts.append(attempt); receipts.append(receipt)
            save_attempt(job['index'], attempt)
        require(len(receipts) == len(plan['schedule']), 'Incomplete self-play batch')
        if live:
            receipts.sort(key=lambda item: item.binding.schedule_index)
            attempts.sort(key=lambda item: item["episode"])
        return tuple(receipts), attempts
    except Exception as error:
        if dispatch is not None:
            dispatch.stop.set()
        if current_index is not None:
            set_phase(progress_path(launch['output'], plan['schedule'][current_index]['episode']), 'failed')
            details = {}
            if plan.get('schema') in (training.TEN_SCHEMA, training.CYCLE_SCHEMA):
                details = ten_capture_fields(output / f"episode-{plan['schedule'][current_index]['episode']:06d}")
                details.update(failure_stage=current_stage,
                               verification_status=('not_performed' if plan.get('schema') == training.CYCLE_SCHEMA
                                   and plan['mode'] == 'training' else
                                   'failed' if current_stage == 'verification' else 'not_started'))
            failure = dict(**plan['schedule'][current_index], status='failed',
                error=f'{type(error).__name__}: {error}', outcomes={'red': None, 'blue': None},
                rewards={'red': None, 'blue': None}, training_samples_admitted=0,
                deadlines=next((job['deadlines'] for job in jobs if job['index'] == current_index),
                               {'collection_and_verification': phase_deadline}))
            failure.update(details)
            save_attempt(current_index, failure)
        raise
    finally:
        dispatch_error = dispatch.close() if dispatch is not None else None
        errors = close_collection_jobs(jobs, output, launch)
        if dispatch_error:
            errors.append(dispatch_error)
        write(output / 'shutdown.json', dict(owned_cleanup_verified=not errors, errors=errors), new=True)
        require(not errors, 'Self-play owned-process cleanup failed: ' + '; '.join(errors))


def close_collection_jobs(jobs, output, launch):
    """Release every worker before checking native groups that they may still own."""
    from experiments.headless.process_deadline import close_owned_process, cleanup_from_evidence
    errors = []
    for job in jobs:
        try:
            close_owned_process(job['process'], timeout=3,
                deadline_monotonic=launch['hard_deadline_monotonic'], deadline_wall=launch['hard_deadline_unix'])
        except Exception as error:
            errors.append(f"worker {job['index']}: {error}")
        finally:
            for stream in (getattr(job['process'], 'stdin', None), getattr(job['process'], 'stdout', None)):
                if stream is not None:
                    stream.close()
            job['log'].close()
    for evidence in Path(output).glob('native-*-deadline.jsonl'):
        try:
            cleanup_from_evidence(evidence, timeout=3,
                deadline_monotonic=launch['hard_deadline_monotonic'], deadline_wall=launch['hard_deadline_unix'])
        except Exception as error:
            errors.append(f'native evidence {evidence.name}: {error}')
    return errors


def coordinator(args):
    from experiments.learner import local_script_training as local, selfplay_training as training
    from experiments.learner.checkpoints import TrainingProgress
    from experiments.learner.ppo import PPOConfig
    from validation.probes.validate_hero_policy import wall_limit
    launch = load_launch(args.plan, args.sha256)
    if launch.get('schema') == TEN_LAUNCH:
        return ten_coordinator(args)
    require(os.getppid() == launch['supervisor_pid'], 'Only the lease-owning supervisor may launch this campaign')
    configure_native(launch['execution'], policy_backend=launch['configuration'].get('policy_backend', 'cpu'))
    output = Path(launch['output']); started = time.monotonic()
    write(output / 'coordinator-claim.json', dict(pid=os.getpid(), launch_sha256=args.sha256,
          reservation=launch['reservation']), new=True)
    report = dict(schema=SCHEMA + '-result', mode=launch['mode'], status='running', batches=[],
        source=launch['source'], parent_policy_sha256=launch['parent_policy_sha256'], native_manifest=launch['native_manifest'],
        runtime=launch['runtime'], native=launch['native'],
        policy_support=GUARD, configuration=launch['configuration'], optimizer_updates=0,
        admitted_games=0, admitted_before=launch.get('admitted_before', 0),
        training_samples_admitted=0, original_campaign_updates=0,
        owned_cleanup_verified=False, protected_inputs_unchanged=False)
    write(output / 'report.json', report, new=True)
    try:
        if 'resume_parent' in launch:
            pair = launch['resume_pair']
            state = training.restore_checkpoint_pair(pair, expected_metadata=pair['metadata'], expected_ppo_config=PPO,
                expected_optimizer_steps=launch['parent_optimizer_steps'])
            require(state.progress.updates == 1728 + launch['admitted_before'] // 4,
                    'Continuation restore differs from the admitted starting checkpoint')
            previous_policy = pair['policy']
        else:
            state = local.restore_parent(launch['parent_descriptor'])
            previous_policy = launch['original_parent']['policy']
        total = launch['reservation']['count']
        size = 4
        for index, start in enumerate(range(0, total, size)):
            remaining(launch, 60)
            deadlines = batch_phase_deadlines(launch)
            plan_path, plan, opponent = batch_plan(launch, state, index, start, size, previous_policy)
            batch = dict(index=index, plan=binding(plan_path), status='collecting', deadlines=deadlines,
                         attempts=[dict(**row, status='not_started') for row in plan['schedule']])
            report['batches'].append(batch); write(output / 'report.json', report)
            def save_attempt(position, attempt):
                batch['attempts'][position] = attempt
                write(plan_path.parent / 'attempts.json', batch['attempts'])
                write(output / 'report.json', report)
            receipts, attempts = collect_batch(plan_path, plan, state.model, opponent, launch, save_attempt,
                                               phase_deadline=deadlines['training'])
            batch.update(status='verified', attempts=attempts); write(output / 'report.json', report)
            if launch['mode'] == 'training':
                state.model.train().requires_grad_(True)
                batch['status'] = 'updating'; write(output / 'report.json', report)
                with (wall_limit(phase_remaining(deadlines['update']))
                  if math.isfinite(phase_remaining(deadlines['update'])) else nullcontext()):
                    result, rollouts = training.native_update(state.model, state.optimizer, receipts,
                        plan_sha256=binding(plan_path)['sha256'], config=PPOConfig(**PPO), generator=state.generator,
                        remaining=lambda: phase_remaining(deadlines['update']), max_optimizer_steps=640)
                # Adam serialization validates every trainable named parameter.
                # Freeze again only after the exact policy/resume roundtrip.
                state.model.eval()
                progress = TrainingProgress(state.progress.updates + 1, state.progress.rollouts + 4,
                                            state.progress.decisions + result.samples)
                consumed = launch.get('admitted_before', 0) + start + 4
                sampler = dict(next_batch_index=consumed // 4, consumed_schedule_offset=consumed,
                    optimizer_steps=state.optimizer_steps + result.optimizer_steps,
                    batch_optimizer_steps=result.optimizer_steps,
                    original_parent_sampler=(state.sampler_state.get('original_parent_sampler', state.sampler_state)))
                metadata = training.metadata(binding(plan_path)['sha256'], launch['original_parent']['policy'],
                    plan['opponent_checkpoint'], campaign_id=launch['campaign_id'], purpose='training')
                checkpoint = f"selfplay-{launch['campaign_id']}-update-{progress.updates}"
                with (wall_limit(phase_remaining(deadlines['update'], 0))
                  if math.isfinite(phase_remaining(deadlines['update'], 0)) else nullcontext()):
                    pair = training.save_checkpoint_pair(plan_path.parent / 'checkpoint', state.model, state.optimizer,
                        checkpoint=checkpoint, ppo_config=PPO, progress=progress, metadata=metadata,
                        opponent_pool=state.pool, sampler_state=sampler, generator=state.generator)
                    require(pair['exact_model_adam_rng_progress_pool'] is True, 'Checkpoint roundtrip incomplete')
                    state.model.requires_grad_(False)
                    # A source/runtime change during PPO leaves the candidate pair as
                    # evidence, without publishing it as an admitted continuation.
                    load_launch(args.plan, args.sha256)
                    state = replace(state, progress=progress, sampler_state=sampler, checkpoint=checkpoint,
                                    optimizer_steps=state.optimizer_steps + result.optimizer_steps,
                                    metadata=metadata, weights_sha256=pair['weights_sha256'])
                    previous_policy = pair['policy']
                    batch.update(status='checkpoint_verified', ppo=asdict(result), after_checkpoint=pair)
                    report['optimizer_updates'] += 1; report['admitted_games'] += 4
                    report['training_samples_admitted'] += result.samples
                    report['cumulative_progress'] = vars(progress)
                    report['cumulative_admitted_games'] = report['admitted_before'] + report['admitted_games']
                    write(output / 'report.json', report)
                    write(output / 'latest-checkpoint.json', dict(schema='heroes-selfplay-latest-v1', purpose='training',
                        campaign_id=launch['campaign_id'], batch=index, plan=binding(plan_path), checkpoint=pair,
                        cumulative_progress=vars(progress), admitted_games=report['cumulative_admitted_games'],
                        window_admitted_games=report['admitted_games']))
                del rollouts
            else:
                training.release_receipts(receipts)
            del receipts
            verify_protected(launch['protected'])
        load_launch(args.plan, args.sha256)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        for batch in report['batches']:
            if batch['status'] in ('collecting', 'updating'):
                batch['status'] = 'failed'
            for attempt in batch['attempts']:
                if attempt['status'] == 'running':
                    attempt.update(status='interrupted', outcomes={'red': None, 'blue': None},
                                   rewards={'red': None, 'blue': None}, training_samples_admitted=0)
    finally:
        shutdowns = [read(path) for path in output.glob('batch-*/execution/shutdown.json')]
        report['owned_cleanup_verified'] = bool(shutdowns) and all(row['owned_cleanup_verified'] for row in shutdowns)
        try:
            report['protected_inputs_unchanged'] = verify_protected(launch['protected'])
        except Exception as error:
            report.update(status='stopped', protection_error=str(error))
        report['elapsed_seconds'] = time.monotonic() - started
        write(output / 'report.json', report)
    return report


def analyze_completed(report, output):
    """Post hoc descriptions cannot change admission, rewards or checkpoint status."""
    from experiments.learner.selfplay_behavior import analyze_episode, aggregate_episodes
    rows, errors = [], []
    attempts = [attempt for batch in report.get('batches', []) for attempt in batch['attempts']
                if attempt.get('status') == 'passed']
    for index, attempt in enumerate(attempts, 1):
        print(f"Reporting game {index}/{len(attempts)}", flush=True)
        try:
            rows.append(analyze_episode(attempt['episode_directory'], attempt['learner_player']))
        except Exception as error:
            errors.append(dict(episode_directory=attempt['episode_directory'], error=str(error)))
        write(Path(output) / 'behavior-progress.json', dict(processed_games=index, total_games=len(attempts),
              successful_games=len(rows), errors=errors, training_input=False))
    try:
        summary = aggregate_episodes(rows)
        write(Path(output) / 'behavior-analysis.json', dict(summary=summary, errors=errors,
              complete=not errors, policy_input=False, reward_input=False, admission_input=False), new=True)
        return not errors
    except Exception as error:
        write(Path(output) / 'behavior-analysis-error.json', dict(error=str(error), errors=errors,
              admission_unchanged=True), new=True)
        return False


def report_inputs(run_dir):
    """Accept only an already completed run; never restore or modify training."""
    run_dir = Path(run_dir).resolve()
    inputs = {name: binding(run_dir / name) for name in ('launch.json', 'report.json', 'shutdown.json')}
    launch, report, shutdown = (read(bound(inputs[name])) for name in inputs)
    require(Path(launch['output']).resolve() == run_dir and report.get('schema') == SCHEMA + '-result'
            and report.get('source') == launch['source'] and report.get('status') == 'completed'
            and report.get('owned_cleanup_verified') is True and report.get('protected_inputs_unchanged') is True
            and shutdown.get('returncode') == 0 and shutdown.get('owned_cleanup_verified') is True
            and shutdown.get('protected_inputs_unchanged') is True and shutdown.get('errors') == [],
            'Reporting requires completed training and verified successful shutdown')
    attempts = [attempt for batch in report['batches'] for attempt in batch['attempts']]
    folders = [Path(attempt['episode_directory']).resolve() for attempt in attempts]
    require(attempts and all(attempt.get('status') == 'passed' for attempt in attempts)
            and len(set(folders)) == len(folders) and all(folder.is_relative_to(run_dir) for folder in folders),
            'Reporting requires unique passed episodes inside the completed run')
    return inputs, report


def report_worker(args):
    require(binding(args.plan)['sha256'] == args.sha256, 'Reporting plan changed')
    plan = read(args.plan)
    require(plan.get('schema') == SCHEMA + '-offline-report' and plan.get('training_input') is False,
            'Unexpected reporting plan')
    inputs, report = report_inputs(plan['run_dir'])
    require(inputs == plan['inputs'], 'Completed run changed before reporting')
    os.nice(10)
    complete = analyze_completed(report, plan['output'])
    for value in inputs.values():
        bound(value)
    require(binding(args.plan)['sha256'] == args.sha256, 'Reporting plan changed during analysis')
    return 0 if complete else 1


def report_completed(args):
    """Run optional statistics separately with no campaign lease or training calls."""
    from experiments.headless.process_deadline import guarded_popen, close_owned_process
    require(math.isfinite(args.seconds) and 1 <= args.seconds <= 7200,
            'Use a finite report budget from 1 to 7200 seconds')
    inputs, _ = report_inputs(args.run_dir)
    run_dir = Path(args.run_dir).resolve()
    output = private_output(args.output or ARTIFACTS / 'selfplay-reports' / (
        run_dir.name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4))).resolve()
    require(not output.is_relative_to(run_dir), 'Write optional reports outside the completed training run')
    output.mkdir(parents=True, mode=0o700)
    plan = dict(schema=SCHEMA + '-offline-report', run_dir=str(run_dir), output=str(output), inputs=inputs,
                training_input=False, maximum_seconds=args.seconds)
    path = output / 'report-input.json'; write(path, plan, new=True)
    plan_binding = binding(path)
    print(f'Optional report output: {output}', flush=True)
    process, errors, cleanup, interruption = None, [], False, None
    try:
        process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_report',
            '--plan', str(path), '--sha256', plan_binding['sha256']],
            deadline_monotonic=time.monotonic() + args.seconds, deadline_wall=time.time() + args.seconds,
            evidence_path=output / 'report-deadline.jsonl', cwd=ROOT, stdin=subprocess.DEVNULL)
        process.wait(timeout=args.seconds + 1)
        if process.returncode != 0:
            errors.append(f'Report worker exited with status {process.returncode}; training remains completed')
    except (KeyboardInterrupt, SystemExit) as error:
        interruption = error
        errors.append(f'{type(error).__name__}: optional reporting interrupted; training remains completed')
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=5)
            cleanup = True
        except Exception as error:
            errors.append(f'Report cleanup: {error}')
        try:
            bound(plan_binding)
            for value in inputs.values():
                bound(value)
        except Exception as error:
            errors.append(f'Report input changed: {error}')
        completed = process is not None and process.returncode == 0 and cleanup and not errors
        result = dict(status='completed' if completed else 'incomplete', output=str(output), errors=errors,
            owned_cleanup_verified=cleanup, returncode=process.returncode if process else None,
            training_input=False, optimizer_updates=0, gameplay_performed=False)
        write(output / 'shutdown.json', result, new=True)
    if interruption is not None:
        raise interruption
    return result


def supervise(launch):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from experiments.learner.game_progress import ProgressDisplay, BatchProgressDisplay
    output = Path(launch['output']); path = output / 'launch.json'; process = None; errors = []
    cleanup_errors = []
    display = (BatchProgressDisplay(output / 'progress', launch=launch) if launch.get('schema') == TEN_LAUNCH
               else ProgressDisplay(output / 'progress'))
    try:
        coordinator_environment = None
        if launch.get('configuration', {}).get('learner_backend', 'cpu') == 'metal':
            require(launch['configuration'].get('metal_matmul_preference') == '1'
                    and launch['configuration'].get('metal_fast_math') == '0',
                    'Metal PPO requires the reviewed matrix-kernel and precision configuration')
            # The child receives these before Python/Torch starts; PyTorch latches them on first use.
            coordinator_environment = {**os.environ, 'PYTORCH_MPS_PREFER_METAL': '1',
                                       'PYTORCH_MPS_FAST_MATH': '0'}
        with (output / 'coordinator.log').open('xb') as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_coordinator',
                    '--plan', str(path), '--sha256', binding(path)['sha256']],
                deadline_monotonic=launch['hard_deadline_monotonic'], deadline_wall=launch['hard_deadline_unix'],
                evidence_path=output / 'coordinator-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                **({'env': coordinator_environment} if coordinator_environment is not None else {}))
            if launch.get('schema') == TEN_LAUNCH:
                emitted, last_report = set(), 0.0
                while process.poll() is None:
                    display.poll()
                    if time.monotonic() - last_report >= 2:
                        if CYCLE_MODE:
                            require_cycle_memory_budget()
                        require(shutil.disk_usage(output).free >=
                                launch['configuration']['disk_reserve_gib'] * 1024**3,
                                'Ten-game disk reserve reached; stopping owned workers')
                        try:
                            current = read(output / 'report.json')
                            display.update_report(current)
                            for phase in current.get('batches', []):
                                if phase.get('status') == 'checkpoint_verified' and phase['name'] not in emitted:
                                    if not display.dashboard:
                                        display.lines = 0
                                        ten_print_phase(phase)
                                    emitted.add(phase['name'])
                            for phase in current.get('evaluations', []):
                                if phase.get('status') == 'evaluation_verified' and phase['name'] not in emitted:
                                    if not display.dashboard:
                                        display.lines = 0
                                        ten_print_phase(phase)
                                    emitted.add(phase['name'])
                        except (OSError, ValueError):
                            pass
                        last_report = time.monotonic()
                    try:
                        process.wait(timeout=min(0.5, remaining(launch)))
                    except subprocess.TimeoutExpired:
                        pass
            else:
                display.wait(process, timeout=remaining(launch) + 1)
    except BaseException as error:
        if launch.get('schema') != TEN_LAUNCH:
            raise
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=5)
            for evidence in output.glob('*/execution/*-deadline.jsonl' if CYCLE_MODE else 'batch-*/execution/*-deadline.jsonl'):
                cleanup_from_evidence(evidence, timeout=5)
        except Exception as error:
            errors.append(str(error))
            cleanup_errors.append(str(error))
        if launch.get('schema') == TEN_LAUNCH:
            try:
                display.update_report(read(output / 'report.json'))
            except (OSError, ValueError):
                pass
        display.finish(failed=bool(errors) or process is None or process.returncode != 0)
        if launch.get('schema') == TEN_LAUNCH:
            try:
                protected = verify_protected(launch['protected'])
            except Exception as error:
                protected = False; errors.append(str(error))
        else:
            protected = verify_protected(launch['protected'])
        write(output / 'shutdown.json', dict(
              owned_cleanup_verified=not cleanup_errors if launch.get('schema') == TEN_LAUNCH else not errors,
              errors=errors,
              returncode=process.returncode if process else None, protected_inputs_unchanged=protected), new=True)
        if launch['mode'] == 'training' and launch.get('schema') != TEN_LAUNCH and not errors and protected:
            finalize_stopped_run(output, launch)
    report = read(output / 'report.json') if (output / 'report.json').exists() else {'status': 'stopped'}
    if launch.get('schema') == TEN_LAUNCH:
        shutdown = read(output / 'shutdown.json')
        report.update(owned_cleanup_verified=shutdown['owned_cleanup_verified'],
                      protected_inputs_unchanged=shutdown['protected_inputs_unchanged'])
        if process is None or process.returncode != 0 or errors or report['status'] != 'completed':
            report.update(status='stopped', supervisor_errors=errors)
        ten_finalize_report(output, report)
        return report
    require(process is not None and process.returncode == 0 and not errors and report['status'] == 'completed',
            'Self-play stopped; preserve this run and do not replay its reservation')
    return report


def finalize_stopped_run(output, launch):
    """Seal stop evidence without promoting an unpublished in-flight update."""
    output = Path(output)
    report_path = output / 'report.json'
    if not report_path.exists():
        write(report_path, dict(status='stopped', source=launch['source'], mode='training', batches=[], admitted_games=0), new=True)
    report = read(report_path)
    if report.get('status') == 'completed' and read(output / 'shutdown.json').get('returncode') == 0:
        return
    pointer_path = output / 'latest-checkpoint.json'
    pointer = read(pointer_path) if pointer_path.exists() else None
    last = pointer['batch'] if pointer else -1
    write(output / 'stop-finalization.json', dict(schema='heroes-selfplay-stop-finalization-v1', status='stopped',
        launch=binding(output / 'launch.json'), report=binding(report_path), shutdown=binding(output / 'shutdown.json'),
        latest=binding(pointer_path) if pointer else None, owned_cleanup_verified=True, protected_inputs_unchanged=True,
        admitted_games=pointer['admitted_games'] if pointer else launch.get('admitted_before', 0),
        ignored_unpublished_batches=[row['index'] for row in report.get('batches', []) if row['index'] > last],
        no_unpublished_experience_admitted=True), new=True)


def mlx_runtime_identity():
    from importlib.metadata import version
    import mlx.core as mx
    require(mx.metal.is_available(), 'MLX requires an Apple GPU')
    versions = {name: version(name) for name in ('mlx', 'mlx-metal')}
    require(versions == {'mlx': '0.31.1', 'mlx-metal': '0.31.1'},
            'MLX runtime differs from dependencies/learner-mlx.lock')
    return versions


def ten_configuration(checked, seconds, *, games=None, use_metal=False, use_mlx=False):
    from experiments.learner import selfplay_training as training
    from experiments.learner.ppo import PPOConfig
    require(type(use_metal) is bool and (not use_metal or CYCLE_MODE),
            'Metal PPO is an explicit cycle-only experiment')
    require(type(use_mlx) is bool and (not use_mlx or CYCLE_MODE) and not (use_metal and use_mlx),
            'Select one cycle learner backend')
    mlx_runtime = mlx_runtime_identity() if use_mlx else None
    if use_metal:
        import torch
        require(torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader'),
                'Metal PPO requires MPS and torch.mps.compile_shader')
    inherited = validate_configuration(checked['inherited_configuration'] if 'inherited_configuration' in checked
        else read(bound(checked['parent']['launch']))['configuration'])
    cfg = {key: deepcopy(inherited[key]) for key in
           ('interface', 'adam', 'ppo', 'torch_threads', 'allow_verified_training_prefix')}
    cfg.update(schema='heroes-selfplay-batch-ten-config-v1', maximum_seconds=seconds)
    cfg.update(training.CYCLE_GAMEPLAY if CYCLE_MODE else training.TEN_GAMEPLAY)
    if games is not None:
        require(CYCLE_MODE and type(games) is int and 10 <= games <= 100, 'Invalid PPO batch size')
        cfg['training_games'] = games
    cfg.update(max_optimizer_steps=training.optimizer_step_cap(PPOConfig(**PPO), cfg['training_games'], cfg['max_total_choices']),
               attempt_wall_seconds=seconds, cleanup_reserve_seconds=30,
               disk_reserve_gib=TEN_DISK_RESERVE_GIB,
               phase_deadlines_seconds={'training': seconds, 'update': seconds})
    if CYCLE_MODE:
        cfg.update(schema='heroes-selfplay-ten-then-check-config-v1',
                   learner_backend='mlx' if use_mlx else ('metal' if use_metal else 'cpu'),
                   training_verification='not_performed', evaluation_games_per_cycle=1,
                   collection_workers=10, collection_worker_threads=1, preparation_workers=1,
                   collection_scheduling='rolling-v1', memory_reserve_gib=48,
                   training_storage='owned-binary-v1', training_history_audit='not_performed',
                   training_bridge_ledger=False, training_history_jsonl=False)
    if use_metal:
        cfg['metal_matmul_preference'] = '1'
        cfg['metal_fast_math'] = '0'
    if use_mlx:
        cfg.update(policy_backend='cpu', mlx_runtime=mlx_runtime)
    return cfg


def ten_reservations(requests=None, *, width=None):
    """Use separate durable namespaces; every reserved attempt stays consumed."""
    path = TEN_BASE / 'seed-reservations.json'
    require(path.exists() or not TEN_BASE.exists()
            or all(item.name == 'namespace.lock' for item in TEN_BASE.iterdir()),
            'Ten-game seed ledger is missing while prior artifacts remain; refusing seed reuse')
    value = read(path) if path.exists() else dict(schema=TEN_LAUNCH + '-seeds', reservations=[])
    require(value.get('schema') == TEN_LAUNCH + '-seeds' and type(value.get('reservations')) is list,
            'Unknown ten-game seed ledger')
    offsets, outputs = dict.fromkeys(TEN_NAMESPACES, 0), set()
    for row in value['reservations']:
        mode = row['mode']
        row_width = row.get('cycle_width', 11 if CYCLE_MODE else 10)
        require(type(row_width) is int and (11 <= row_width <= 1001 if CYCLE_MODE else row_width == 10)
                and mode in offsets and type(row['count']) is int and row['count'] > 0
                and row['count'] % row_width == 0 and row['offset'] == offsets[mode]
                and row['consumed_including_unstarted'] is True
                and (row['output'], mode) not in outputs, 'Invalid ten-game seed reservation')
        engine, policy = TEN_NAMESPACES[mode]
        require(row['engine_base'] == engine + row['offset']
                and row['policy_base'] == policy + 2 * row['offset'], 'Ten-game seed namespace changed')
        offsets[mode] += row['count']; outputs.add((row['output'], mode))
    if requests is None:
        return value, offsets
    width = cycle_width() if width is None else width
    require(type(width) is int and (11 <= width <= 1001 if CYCLE_MODE else width == 10), "Invalid reservation width")
    added = {}
    for mode, count, output in requests:
        require(mode in offsets and type(count) is int and count > 0 and count % width == 0
                and offsets[mode] + count < 1_000_000 and (str(output), mode) not in outputs,
                'Invalid or reused ten-game seed request')
        engine, policy = TEN_NAMESPACES[mode]
        row = dict(mode=mode, output=str(output), offset=offsets[mode], count=count,
                   engine_base=engine + offsets[mode], policy_base=policy + 2 * offsets[mode],
                   consumed_including_unstarted=True)
        if CYCLE_MODE:
            row['cycle_width'] = width
        value['reservations'].append(row); offsets[mode] += count
        outputs.add((str(output), mode)); added[mode] = row
    write(path, value)
    return added


def ten_seed_preview(count, *, width=None):
    """Describe the next fresh reservation without allocating seeds or files."""
    _, offsets = ten_reservations()
    width = cycle_width() if width is None else width
    offset = offsets['training']
    require(type(count) is int and count > 0 and count % width == 0 and offset + count < 1_000_000,
            'Invalid or exhausted ten-game seed preview')
    engine, policy = TEN_NAMESPACES['training']
    return dict(count=count, consumed_before=offset, engine_base=engine + offset,
                policy_base=policy + 2 * offset, policy_seed_count=count * 2, reserved=False)


def cycle_retained():
    """An audited checkpoint replaces retired raw history, never current-run checks."""
    path = TEN_BASE / 'retained/state.json'
    if not CYCLE_MODE or not path.exists():
        return None
    value = read(path)
    require(value.get('schema') == CYCLE_LAUNCH + '-retained-v1', 'Unknown retained cycle state')
    # A consumed seed prefix cannot disappear when old captures are removed.
    ledger, _ = ten_reservations()
    prefix = value['seed_reservations']
    require(ledger['reservations'][:len(prefix)] == prefix, 'Retained seed consumption changed')
    for pair in (value['initial'], value['checkpoint']):
        for name in ('policy', 'resume'):
            path = bound(pair[name])
            require(path.is_relative_to((TEN_BASE / 'retained').resolve())
                    and not Path(pair[name]['path']).is_symlink(),
                    'Retained checkpoint is outside its private directory')
    return value


def cycle_retained_candidate(value):
    from tools import run_recruit_only_one_game as one
    from experiments.learner.native_execution import native_execution_choice
    checked = deepcopy(value['checked'])
    checked.update(source=one.source_identity(), parent={'checkpoint': value['initial']},
                   protected={}, experiment=one.experiment(one.HERO_RELATIVE_VERSION))
    require(checked['native_manifest'] == binding(NATIVE) and binding(NATIVE)['sha256'] == NATIVE_SHA,
            'Retained native learner changed')
    require(native_execution_choice(NATIVE, NATIVE_SHA, cpu_cap=16, require_acceptance=False)
            == checked['execution'], 'Retained native execution profile changed')
    validate_configuration(checked['inherited_configuration'])
    checked.update(one.runtime_validation.inspect_runtime(checked))
    checked['protected'].update(checked.pop('runtime_protected'))
    for pin in (binding(TEN_BASE / 'retained/state.json'), checked['native_manifest']):
        checked['protected'][pin['path']] = pin
    one.frozen_models(checked)
    return one.runtime_validation.inspect_cycle_scenario(checked)


def cycle_finish_retirement(value):
    """Retry deletion after atomic checkpoint publication, using bounded owned paths."""
    for name in value['retired_runs']:
        path = TEN_BASE / name
        require(Path(name).name == name and name.startswith('run-') and not path.is_symlink()
                and path.resolve().parent == TEN_BASE.resolve(), 'Unsafe retired run directory')
        if path.exists():
            shutil.rmtree(path)
    for name in ('lineage', 'latest'):
        pin = value['retired_' + name]
        path = TEN_BASE / ('lineage.json' if name == 'lineage' else 'latest-checkpoint.json')
        if pin is not None and path.exists():
            require(binding(path) == pin, 'New cycle metadata appeared during retirement')
            path.unlink()


def cycle_retire(inspected):
    """Keep exact continuation state, then remove the analyzed previous run(s)."""
    from experiments.learner import selfplay_training as training
    require(CYCLE_MODE, 'Artifact retirement is restricted to the manual cycle')
    processes = subprocess.check_output(['ps', '-axo', 'command='], text=True).splitlines()
    require(not any(line.split() and (Path(line.split()[0]).name in ('vcmiclient', 'vcmireplay')
                or ('run_selfplay_training.py' in line and any(' ' + mode + ' ' in line
                    for mode in ('_capture', '_coordinator')))) for line in processes),
            'Game or learner workers are still running; keep their artifacts')
    lineage = inspected['lineage']
    if lineage is None:
        retained = cycle_retained()
        if retained is not None:
            cycle_finish_retirement(retained)
        return ten_inspect()
    outputs = [Path(read(bound(pin))['output']) for pin in lineage['runs']]
    require(outputs, 'Cannot retire an empty cycle lineage')
    require(all(path.parent == TEN_BASE and path.name.startswith('run-') and not path.is_symlink()
                and path.resolve().parent == TEN_BASE.resolve() for path in outputs),
            'Only owned cycle run directories can be retired')
    journal = ROOT / 'docs/research-log.md'
    require(f'Retention-reviewed run: `{outputs[-1].name}`.' in journal.read_text(),
            'Analyze the latest run and record its Retention-reviewed run marker in docs/research-log.md first')
    verify_protected(inspected['protected'])
    copied = []
    for pair, steps in ((inspected['initial'], inspected['checked']['optimizer_steps']),
                        (inspected['parent'], inspected['optimizer_steps'])):
        replacement = deepcopy(pair)
        folder = TEN_BASE / 'retained' / ('checkpoint-' + pair['policy']['sha256'])
        require(not folder.is_symlink() and folder.resolve().parent == TEN_BASE.resolve() / 'retained',
                'Unsafe retained checkpoint directory')
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('policy', 'resume'):
            source = bound(pair[name]); target = folder / (name + '.pt')
            require(not target.is_symlink(), 'Retained checkpoint cannot be a symlink')
            if not target.exists():
                temporary = target.with_name('.' + name + '-' + secrets.token_hex(6))
                try:
                    write_bytes(temporary, source.read_bytes())
                    require(binding(temporary)['sha256'] == pair[name]['sha256'], 'Checkpoint copy changed')
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            replacement[name] = binding(target)
            require(all(replacement[name][key] == pair[name][key] for key in ('sha256', 'bytes')),
                    'Retained checkpoint copy differs')
        training.restore_checkpoint_pair(replacement, expected_metadata=pair['metadata'],
            expected_ppo_config=PPO, expected_optimizer_steps=steps)
        descriptor = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        copied.append(replacement)
    checked = inspected['checked']
    inherited = checked.get('inherited_configuration')
    if inherited is None:
        inherited = read(bound(checked['parent']['launch']))['configuration']
    # Runtime acceptance is rechecked from the preserved frozen runtime, not old captures.
    baseline = {key: checked[key] for key in ('data', 'execution', 'native_manifest', 'optimizer_steps',
        'checkpoint', 'model_checkpoint', 'weights_sha256', 'model_config', 'parameters')}
    baseline.update(inherited_configuration=validate_configuration(inherited),
        game_map=read(bound(checked['runtime_origin']['freeze']))['map']['path'])
    value = dict(schema=CYCLE_LAUNCH + '-retained-v1', initial=copied[0], checkpoint=copied[1],
        optimizer_steps=inspected['optimizer_steps'], pending_evaluation=inspected['pending_evaluation'],
        checked=baseline, seed_reservations=ten_reservations()[0]['reservations'],
        retired_runs=[path.name for path in outputs], retired_lineage=binding(TEN_BASE / 'lineage.json'),
        retired_latest=binding(TEN_BASE / 'latest-checkpoint.json') if (TEN_BASE / 'latest-checkpoint.json').exists() else None,
        audited_utc=datetime.now(timezone.utc).isoformat(), source=source_identity(), analysis_journal=binding(journal))
    # Publication happens before deletion. Readers recognize the retired pointers
    # even if interruption leaves any old metadata or partially removed directory.
    write(TEN_BASE / 'retained/state.json', value)
    ten_inspect()
    cycle_finish_retirement(value)
    keep = {Path(pair['policy']['path']).parent for pair in copied}
    for path in (TEN_BASE / 'retained').glob('checkpoint-*'):
        require(path.is_dir() and not path.is_symlink(), 'Unsafe obsolete checkpoint directory')
        if path.resolve() not in keep:
            shutil.rmtree(path)
    return ten_inspect()


def ten_inspect():
    """Audit the new publication chain, then restore only its complete last pair."""
    from tools import run_recruit_only_one_game as one
    from experiments.learner import selfplay_training as training
    require(CYCLE_MODE or shutil.disk_usage(ARTIFACTS).free >= TEN_DISK_RESERVE_GIB * 1024**3,
            'Ten-game launch requires the existing 30 GiB disk reserve')
    retained = cycle_retained()
    checked = cycle_retained_candidate(retained) if retained else one.inspect_candidate(one.HERO_RELATIVE_VERSION)
    if CYCLE_MODE and retained is None:
        checked = one.runtime_validation.inspect_cycle_scenario(checked)
    initial = checked['parent']['checkpoint']
    selected, steps = initial, checked['optimizer_steps']
    lineage_path, latest_path = TEN_BASE / 'lineage.json', TEN_BASE / 'latest-checkpoint.json'
    lineage = read(lineage_path) if lineage_path.exists() else None
    if retained and lineage_path.exists() and binding(lineage_path) == retained['retired_lineage']:
        lineage = None
    latest_exists = latest_path.exists() and not (retained and binding(latest_path) == retained['retired_latest'])
    protected = dict(checked['protected'])
    publication = None
    completed_evaluations = {}
    published_games = published_updates = 0
    if retained:
        selected, steps = retained['checkpoint'], retained['optimizer_steps']
        published_updates = selected['progress']['updates'] - initial['progress']['updates']
        published_games = selected['progress']['rollouts'] - initial['progress']['rollouts']
        if not retained['pending_evaluation']:
            completed_evaluations[selected['progress']['updates']] = selected['weights_sha256']
    seed_ledger, _ = ten_reservations()
    reported_publications, all_publications = set(), set()
    if lineage is not None:
        require(lineage.get('schema') == TEN_LAUNCH and lineage.get('initial_checkpoint') == initial
                and lineage.get('initial_optimizer_steps') == checked['optimizer_steps'] and type(lineage.get('runs')) is list,
                'Ten-game lineage initial state changed')
        outputs = set()
        for launch_pin in lineage['runs']:
            launch = read(bound(launch_pin)); output = Path(launch['output'])
            require(launch.get('schema') == TEN_LAUNCH and str(output) not in outputs,
                    'Duplicate or foreign ten-game launch')
            outputs.add(str(output))
            require(all(row in seed_ledger['reservations'] for row in launch['reservations'].values())
                    and launch['reservations']['training']['count'] == launch.get('reserved_cycles', launch['batches']) * (launch.get('requested_games', launch['configuration']['training_games']) + int(CYCLE_MODE)),
                    'Registered ten-game launch has lost its consumed reservation')
            shutdown = read(output / 'shutdown.json')
            require(shutdown.get('owned_cleanup_verified') is True
                    and shutdown.get('protected_inputs_unchanged') is True,
                    'Previous ten-game run has not closed its owned processes')
            report = read(output / 'report.json')
            if CYCLE_MODE:
                for evaluation in report.get('evaluations', []):
                    if evaluation.get('status') != 'evaluation_verified':
                        continue
                    require(len(evaluation.get('attempts', [])) == 1, 'Cycle diagnostic result is incomplete')
                    evaluation_plan = read(bound(evaluation['plan']))
                    require(evaluation_plan.get('schema') == training.CYCLE_SCHEMA
                            and evaluation_plan.get('mode') == 'diagnostic'
                            and evaluation_plan['learner_weights_sha256']
                                == evaluation_plan['opponent_weights_sha256'] == evaluation['weights_before'],
                            'Cycle diagnostic must use the updated policy in both seats')
                    attempt = evaluation['attempts'][0]
                    require(attempt.get('replay_verified') is True
                            and attempt.get('both_policies_reconstructed') is True,
                            'Cycle diagnostic lacks complete validation')
                    for pin in attempt['evidence'].values():
                        bound(pin)
                    proof = read(bound(attempt['evidence']['training-receipt.json']))
                    require(proof == attempt['verification']
                            and proof['verification'].get('replay_verified') is True
                            and len(proof['verification'].get('policies', [])) == 2
                            and all(policy.get('maximum_absolute_difference') == 0
                                    for policy in proof['verification']['policies']),
                            'Cycle diagnostic lacks persisted exact policy reconstruction')
                    completed_evaluations[evaluation['checkpoint_before']] = evaluation['weights_before']
            all_publications.update(binding(path)['sha256'] for path in output.glob('batch-*/publication.json'))
            for phase in report.get('batches', []):
                if phase.get('status') == 'checkpoint_verified':
                    path = Path(bound(phase['plan'])).parent / 'publication.json'
                    item = read(path)
                    require(item['checkpoint'] == phase['after_checkpoint'], 'Reported checkpoint publication changed')
                    reported_publications.add(binding(path)['sha256'])
        if latest_exists:
            latest = read(latest_path)
            require(latest.get('schema') == TEN_LAUNCH + '-latest', 'Unknown ten-game latest pointer')
            publication = latest['publication']
            cursor, reverse, seen = publication, [], set()
            while cursor is not None:
                require(cursor['sha256'] not in seen, 'Cyclic ten-game checkpoint chain')
                seen.add(cursor['sha256'])
                row = read(bound(cursor))
                require(row.get('schema') == TEN_LAUNCH + '-publication'
                        and row['launch'] in lineage['runs'], 'Unregistered ten-game checkpoint publication')
                reverse.append(row); cursor = row['previous_publication']
            for row in reversed(reverse):
                plan = read(bound(row['plan']))
                require(plan.get('schema') == (training.CYCLE_SCHEMA if CYCLE_MODE else training.TEN_SCHEMA) and plan['mode'] == 'training'
                        and row['parent_checkpoint'] == selected and row['parent_optimizer_steps'] == steps
                        and row['checkpoint']['metadata']['plan_sha256'] == row['plan']['sha256']
                        and row['checkpoint']['metadata']['training_contract'] == (training.CYCLE_SCHEMA if CYCLE_MODE else training.TEN_SCHEMA),
                        'Ten-game checkpoint ancestry changed')
                count = plan['configuration']['training_games']
                require(type(count) is int and (10 <= count <= 100 if CYCLE_MODE else count == 10)
                        and len(plan['schedule']) == count
                        and row['checkpoint']['metadata'].get('training_games') == count
                        and len(row['receipts']) == count and len({item['path'] for item in row['receipts']}) == count,
                        'Ten-game publication lacks its complete receipt set')
                for index, receipt in enumerate(row['receipts']):
                    proof = read(bound(receipt))
                    folder = Path(plan['execution_output']) / f'episode-{index + 1:06d}'
                    require(proof.get('schema') == (training.CAPTURE_RECEIPT_SCHEMA if CYCLE_MODE else training.RECEIPT_SCHEMA)
                            and proof.get('plan_sha256') == row['plan']['sha256']
                            and proof.get('schedule_index') == index and proof.get('learner_player') == index % 2
                            and Path(proof['folder']).resolve() == folder.resolve()
                            and proof.get('admission_from_file') is False and proof.get('learner_only_ppo') is True
                            and proof['verification'].get('replay_verified') is (not CYCLE_MODE),
                            'Published training receipt identity changed')
                    from experiments.headless import contextual_evidence as evidence
                    for name, expected in proof['files'].items():
                        require(evidence.file_identity(evidence.inside(folder, name)) == expected,
                                'Published ten-game evidence changed: ' + name)
                before, after = selected['progress'], row['checkpoint']['progress']
                require(after['updates'] == before['updates'] + 1
                        and after['rollouts'] == before['rollouts'] + count
                        and after['decisions'] == before['decisions'] + row['ppo']['samples']
                        and row['optimizer_steps'] == steps + row['ppo']['optimizer_steps']
                        and row['checkpoint']['sampler_state']['optimizer_steps'] == row['optimizer_steps'],
                        'Ten-game progress or optimizer count changed')
                published_games += count
                published_updates += 1
                selected, steps = row['checkpoint'], row['optimizer_steps']
            require(latest['checkpoint'] == selected and latest['optimizer_steps'] == steps,
                    'Latest pointer differs from its complete publication chain')
            require(reported_publications <= seen, 'Latest checkpoint pointer was rolled back')
            require(all_publications <= seen,
                    'Unpublished checkpoint candidate or rolled-back pointer remains; recover explicitly before continuing')
        else:
            require(not all_publications, 'Ten-game latest pointer is missing while checkpoint candidates remain; recover explicitly')
    else:
        require(not latest_exists, 'Ten-game latest pointer has no lineage')
    state = training.restore_checkpoint_pair(selected, expected_metadata=selected['metadata'],
        expected_ppo_config=PPO, expected_optimizer_steps=steps)
    require(state.progress.updates == initial['progress']['updates'] + published_updates
            and state.progress.rollouts == initial['progress']['rollouts'] + published_games,
            'Restored progress differs from complete published batches')
    for pair in (initial, selected):
        for name in ('policy', 'resume'):
            protected[pair[name]['path']] = pair[name]
    verify_protected(protected)
    ten_reservations()
    return dict(checked=checked, initial=initial, parent=selected, optimizer_steps=steps,
                publication=publication, lineage=lineage, protected=protected, state=state,
                pending_evaluation=CYCLE_MODE and state.progress.updates > 1753
                    and completed_evaluations.get(state.progress.updates) != state.weights_sha256)


def prepare_ten(args, inspected):
    from experiments.learner import interaction_guard
    require(shutil.disk_usage(ARTIFACTS).free >= TEN_DISK_RESERVE_GIB * 1024**3,
            'Ten-game launch requires the existing 30 GiB disk reserve after artifact retirement')
    require(type(args.batches) is int and 1 <= args.batches <= 100,
            'Use between one and 100 complete batches')
    seconds = None if CYCLE_MODE and args.hours is None else args.hours * 3600
    maximum = 36000 * ((getattr(args, 'games', None) or 100) / 100) if CYCLE_MODE else 36000
    require(seconds is None or (math.isfinite(seconds) and 120 <= seconds <= maximum),
            'Time budget exceeds the requested game allowance')
    checked, lineage = inspected['checked'], inspected['lineage']
    require((lineage is None) == (args.command == 'batch10-run'),
            'Use batch10-run for a new lineage, or batch10-continue for its latest complete checkpoint')
    if args.command == 'batch10-continue':
        require(lineage['runs'] and Path(read(bound(lineage['runs'][-1]))['output']).resolve()
                == args.run_dir.resolve(), 'Continue only from the latest registered ten-game run')
    requested = getattr(args, 'games', None) if CYCLE_MODE else None
    sizes = cycle_batch_sizes(requested) if requested is not None else None
    configuration = ten_configuration(checked, seconds, games=sizes[0] if sizes else None,
        use_metal=getattr(args, 'metal', False), use_mlx=getattr(args, 'mlx', False))
    reserved_cycles = (1 if sizes else args.batches) + int(inspected.get('pending_evaluation', False))
    width = requested + 1 if sizes else cycle_width()
    output = private_output(args.output or TEN_BASE / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                                    + '-' + secrets.token_hex(4)))
    TEN_BASE.mkdir(parents=True, exist_ok=True)
    with (TEN_BASE / 'namespace.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reservations = ten_reservations([('training', reserved_cycles * width, output)], width=width)
    output.mkdir(parents=True, mode=0o700)
    now, wall = time.monotonic(), time.time()
    launch = dict(schema=TEN_LAUNCH, campaign_id=output.name, mode='training', output=str(output),
        supervisor_pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(), source=source_identity(),
        parent_pair=inspected['parent'], parent_optimizer_steps=inspected['optimizer_steps'],
        parent_publication=inspected['publication'], original_parent=inspected['initial'],
        parent_policy_sha256=inspected['parent']['policy']['sha256'], protected=inspected['protected'],
        reservations=reservations, batches=args.batches, reserved_cycles=reserved_cycles,
        pending_evaluation=inspected.get('pending_evaluation', False),
        configuration=configuration, execution=checked['execution'],
        policy_support=(interaction_guard.STACK_MANAGEMENT_VERSION
                        if CYCLE_MODE else checked['experiment']['policy_support']), equipment_policy=checked['equipment_policy'],
        army_information=checked['army_information'], maximum_seconds=seconds,
        hard_deadline_monotonic=None if seconds is None else now + seconds,
        hard_deadline_unix=None if seconds is None else wall + seconds,
        **{key: checked[key] for key in ('binary', 'data', 'game_map', 'replay_binary', 'native', 'runtime', 'native_manifest')})
    if sizes:
        launch.update(requested_games=requested, batch_sizes=sizes)
    write(output / 'launch.json', launch, new=True)
    if lineage is None:
        lineage = dict(schema=TEN_LAUNCH, initial_checkpoint=inspected['initial'],
                       initial_optimizer_steps=checked['optimizer_steps'], runs=[])
    lineage['runs'].append(binding(output / 'launch.json'))
    write(TEN_BASE / 'lineage.json', lineage)
    return launch


def load_ten_launch(path, sha):
    require(binding(path)['sha256'] == sha, 'Ten-game launch changed')
    launch = read(path)
    require(launch.get('schema') == TEN_LAUNCH and launch['source'] == source_identity(),
            'Ten-game launch source changed')
    if launch['configuration'].get('learner_backend', 'cpu') == 'mlx':
        require(launch['configuration'].get('mlx_runtime') == mlx_runtime_identity()
                and launch['configuration'].get('policy_backend') in ('cpu', 'mlx'),
                'MLX runtime or policy backend differs from its launch')
    if launch['configuration'].get('learner_backend', 'cpu') == 'metal':
        require(launch['configuration'].get('metal_matmul_preference') == '1'
                and launch['configuration'].get('metal_fast_math') == '0'
                and os.environ.get('PYTORCH_MPS_PREFER_METAL') == '1'
                and os.environ.get('PYTORCH_MPS_FAST_MATH') == '0',
                'Metal PPO matrix-kernel or precision environment differs from its launch')
    if 'requested_games' in launch:
        require(launch.get('batch_sizes') == cycle_batch_sizes(launch['requested_games'])
                and launch['batches'] == len(launch['batch_sizes']), 'Requested batch partition changed')
    require(launch['configuration'].get('disk_reserve_gib') == TEN_DISK_RESERVE_GIB,
            'Ten-game disk reserve changed')
    require(launch['native_manifest'] == binding(NATIVE), 'Native learner changed')
    from experiments.headless.contextual_evidence import runtime_identity
    require(runtime_identity(Path(launch['binary']).parent) == launch['runtime'], 'Ten-game engine runtime changed')
    bound(launch['replay_binary']); verify_protected(launch['protected'])
    ledger, _ = ten_reservations()
    require(all(row in ledger['reservations'] for row in launch['reservations'].values()),
            'Ten-game consumed seed reservation changed')
    lineage = read(TEN_BASE / 'lineage.json')
    require(binding(path) in lineage['runs'], 'Ten-game launch is not registered')
    return launch


def ten_batch_plan(launch, state, *, name, mode, start, opponent_model, opponent_policy, count=None):
    from experiments.learner import checkpoints as cp, selfplay_training as training
    folder = Path(launch['output']) / name
    folder.mkdir(); inputs = folder / 'input'; inputs.mkdir()
    count = cycle_games() if count is None else count
    configuration = dict(launch['configuration'])
    if CYCLE_MODE:
        from experiments.learner.ppo import PPOConfig
        configuration.update(training_games=count, max_optimizer_steps=training.optimizer_step_cap(
            PPOConfig(**PPO), count, configuration['max_total_choices']))
    metadata = training.metadata(binding(Path(launch['output']) / 'launch.json')['sha256'],
        launch['original_parent']['policy'], opponent_policy, campaign_id=launch['campaign_id'],
        purpose=mode, training_contract=training.CYCLE_SCHEMA if CYCLE_MODE else training.TEN_SCHEMA,
        training_games=count, policy_support=launch["policy_support"])
    opponent = deepcopy(opponent_model).eval().requires_grad_(False)
    opponent_id = f"{launch['campaign_id']}-{name}-frozen"
    for filename, model, checkpoint in (('learner.pt', state.model, state.checkpoint),
                                         ('opponent.pt', opponent, opponent_id)):
        cp.save_policy_checkpoint(inputs / filename, model, checkpoint=checkpoint,
            provenance=cp.SELFPLAY_GAMEPLAY_PROVENANCE, source_metadata=metadata,
            opponent_pool=state.pool, generators={'minibatch': state.generator}, interface=INTERFACE)
    plan = dict(schema=training.CYCLE_SCHEMA if CYCLE_MODE else training.TEN_SCHEMA, campaign_id=launch['campaign_id'], mode=mode,
        output=str(folder), execution_output=str(folder / 'execution'), launch=binding(Path(launch['output']) / 'launch.json'),
        source=launch['source']['source'], implementation=launch['source']['implementation'],
        configuration=configuration, checkpoint=binding(inputs / 'learner.pt'),
        opponent_checkpoint=binding(inputs / 'opponent.pt'), checkpoint_source=metadata, opponent_checkpoint_source=metadata,
        learner_checkpoint=state.checkpoint, opponent_checkpoint_id=opponent_id,
        learner_weights_sha256=cp.model_weight_digest(state.model), opponent_weights_sha256=cp.model_weight_digest(opponent),
        opponent_pool=POOL, schedule=schedule(launch['reservations']['training' if CYCLE_MODE else mode], start,
            1 if CYCLE_MODE and mode == 'diagnostic' else count),
        **{key: launch[key] for key in ('native', 'runtime', 'binary', 'data', 'game_map', 'replay_binary',
                                      'policy_support', 'execution', 'army_information', 'equipment_policy')})
    if CYCLE_MODE:
        for index, row in enumerate(plan['schedule']):
            row.update(phase=mode, learner_player=index % 2)
    write(folder / 'plan.json', plan, new=True)
    return folder / 'plan.json', plan, opponent


def ten_capture_fields(folder):
    """Preserve game facts even when later verification fails or is interrupted."""
    try:
        manifest = read(Path(folder) / 'manifest.json')
    except (OSError, ValueError):
        return {}
    return dict(episode_status=manifest.get('status'), outcomes=manifest.get('outcomes_by_player', {}),
                rewards=manifest.get('rewards_by_player', {}), termination_reason=manifest.get('termination_reason'),
                natural_terminal_after_horizon=manifest.get('natural_terminal_after_horizon'))


def ten_finalize_report(output, report):
    # The pointer is the admission boundary. Reconcile a crash after that atomic
    # write without granting admission to an orphan candidate checkpoint.
    latest_path = TEN_BASE / 'latest-checkpoint.json'
    committed = []
    if latest_path.exists():
        latest = read(latest_path)
        cursor, seen = latest['publication'], set()
        while cursor is not None:
            require(cursor['sha256'] not in seen, 'Cyclic committed checkpoint chain')
            seen.add(cursor['sha256'])
            row = read(bound(cursor))
            if not committed and cursor == latest['publication']:
                require(latest['checkpoint'] == row['checkpoint'], 'Latest checkpoint and publication differ')
            if Path(row['launch']['path']).resolve() == (Path(output) / 'launch.json').resolve():
                committed.append(row)
            cursor = row['previous_publication']
    by_plan = {row['plan']['sha256']: row for row in committed}
    for phase in report.get('batches', []) + report.get('evaluations', []):
        row = by_plan.get(phase['plan']['sha256'])
        if row is not None:
            pair = row['checkpoint']
            phase.update(status='checkpoint_verified', ppo=row['ppo'], after_checkpoint=pair,
                checkpoint_before=row['parent_checkpoint']['progress']['updates'], checkpoint_after=pair['progress']['updates'],
                weights_before=row['parent_checkpoint']['weights_sha256'], weights_after=pair['weights_sha256'],
                weights_changed=row['parent_checkpoint']['weights_sha256'] != pair['weights_sha256'],
                admitted_games=pair['progress']['rollouts'] - row['parent_checkpoint']['progress']['rollouts'])
        elif phase.get('status') in ('collecting', 'updating'):
            phase['status'] = 'failed'
        folder = Path(phase['plan']['path']).parent / 'execution'
        for attempt in phase.get('attempts', []):
            if attempt.get('status') != 'passed':
                attempt.update(ten_capture_fields(folder / f"episode-{attempt['episode']:06d}"))
            if attempt.get('status') in ('running', 'preparing'):
                attempt['status'] = 'interrupted'
                attempt['verification_status'] = ('interrupted'
                    if attempt.get('episode_status') in ('completed', 'truncated') else 'not_started')
        phase['counts'] = ten_outcomes(phase.get('attempts', []))
    report['optimizer_updates'] = len(committed)
    report['admitted_games'] = sum(row['checkpoint']['progress']['rollouts']
                                   - row['parent_checkpoint']['progress']['rollouts'] for row in committed)
    report['training_samples_admitted'] = sum(row['ppo']['samples'] for row in committed)
    if committed:
        report['latest_checkpoint'] = max(row['checkpoint']['progress']['updates'] for row in committed)
    write(Path(output) / 'report.json', report)


def ten_print_phase(phase):
    counts = phase.get('counts', {})
    if CYCLE_MODE:
        if phase.get('mode') == 'diagnostic':
            print(f"Validated self-play: Red wins {counts.get('red_wins', 0)}, "
                  f"Blue wins {counts.get('blue_wins', 0)}, cutoffs {counts.get('game_cutoffs', 0)}", flush=True)
        else:
            print(f"{phase['name']}: engine Red wins {counts.get('engine_red_wins', 0)}, "
                  f"Blue wins {counts.get('engine_blue_wins', 0)}, cutoffs {counts.get('game_cutoffs', 0)}; "
                  f"PPO updates {int(phase.get('status') == 'checkpoint_verified')}, "
                  f"weights changed {phase.get('weights_changed', False)}; "
                  f"checkpoint {phase.get('checkpoint_before')} -> {phase.get('checkpoint_after', 'not updated')}", flush=True)
        return
    print(f"{phase['name']}: Red {counts.get('red_wins', 0)}, Blue {counts.get('blue_wins', 0)}, "
          f"cutoffs {counts.get('game_cutoffs', 0)}, verified {counts.get('verified', 0)}/10, "
          f"verification failures {counts.get('verification_failures', 0)}; "
          f"checkpoint {phase.get('checkpoint_before')} -> {phase.get('checkpoint_after', 'not updated')}", flush=True)


def ten_outcomes(attempts):
    from tools.run_recruit_only_one_game import captured_game_result, summarize_game_results
    rows = []
    learner_wins = learner_losses = 0
    for item in attempts:
        passed = item.get('status') == 'passed'
        outcomes = item.get('outcomes', {})
        row = dict(status='verified' if passed else item.get('status', 'not_started'),
                   verification_status='passed' if passed else item.get('verification_status', 'not_started'),
                   failure_stage=item.get('failure_stage'))
        row.update(captured_game_result(dict(status=item.get('episode_status'), outcomes_by_player=outcomes,
            rewards_by_player=item.get('rewards', {}), termination_reason=item.get('termination_reason'),
            natural_terminal_after_horizon=item.get('natural_terminal_after_horizon'))))
        rows.append(row)
        if passed:
            outcome = outcomes.get(('red', 'blue')[item['learner_player']])
            learner_wins += outcome == 'win'; learner_losses += outcome == 'loss'
    return dict(**summarize_game_results(rows),
                engine_red_wins=sum(row.get('game_outcome') == 'red_win' for row in rows),
                engine_blue_wins=sum(row.get('game_outcome') == 'blue_win' for row in rows),
                learner_wins=learner_wins, learner_losses=learner_losses,
                learner_win_fraction_all_verified=(learner_wins / sum(r['verification_status'] == 'passed' for r in rows)
                    if any(r['verification_status'] == 'passed' for r in rows) else None))


def ten_coordinator(args):
    from experiments.learner import selfplay_training as training
    from experiments.learner.checkpoints import TrainingProgress
    from experiments.learner.ppo import PPOConfig
    from experiments.learner.game_progress import PPOProgress, ppo_progress_path
    from validation.probes.validate_hero_policy import wall_limit
    launch = load_ten_launch(args.plan, args.sha256)
    require(os.getppid() == launch['supervisor_pid'], 'Only the lease owner may launch ten-game training')
    configure_native(launch['execution'], policy_backend=launch['configuration'].get('policy_backend', 'cpu'))
    output = Path(launch['output']); started = time.monotonic(); started_wall = time.time()
    state = training.restore_checkpoint_pair(launch['parent_pair'], expected_metadata=launch['parent_pair']['metadata'],
        expected_ppo_config=PPO, expected_optimizer_steps=launch['parent_optimizer_steps'])
    report = dict(schema=TEN_LAUNCH + '-result', status='running', mode='training', batches=[],
        learner_backend=launch['configuration'].get('learner_backend', 'cpu'),
        inference_backend=launch['configuration'].get('policy_backend', 'cpu'),
        optimizer_updates=0, admitted_games=0, training_samples_admitted=0,
        parent_checkpoint=state.progress.updates, latest_checkpoint=state.progress.updates,
        owned_cleanup_verified=False, protected_inputs_unchanged=False)
    if report['learner_backend'] == 'metal':
        report['metal_matmul_preference'] = os.environ['PYTORCH_MPS_PREFER_METAL']
        report['metal_fast_math'] = os.environ['PYTORCH_MPS_FAST_MATH']
    if report['learner_backend'] == 'mlx':
        report['mlx_runtime'] = mlx_runtime_identity()
    write(output / 'report.json', report, new=True)
    previous_pair, publication = launch['parent_pair'], launch['parent_publication']
    update_progress = None

    def collect_phase(name, mode, start, opponent_model, opponent_policy, count=None):
        load_ten_launch(args.plan, args.sha256)
        path, plan, opponent = ten_batch_plan(launch, state, name=name, mode=mode, start=start,
                                             opponent_model=opponent_model, opponent_policy=opponent_policy, count=count)
        phase = dict(name=name, mode=mode, plan=binding(path), status='collecting',
                     learner_backend=launch['configuration'].get('learner_backend', 'cpu') if mode == 'training' else None,
                     collection_started_utc=datetime.now(timezone.utc).isoformat(),
                     comparison_checkpoint='batch_start',
                     checkpoint_before=state.progress.updates, weights_before=state.weights_sha256,
                     planned_games=len(plan['schedule']),
                     attempts=[dict(**row, status='not_started') for row in plan['schedule']])
        if mode == 'training' and report['learner_backend'] == 'metal':
            phase['metal_matmul_preference'] = report['metal_matmul_preference']
            phase['metal_fast_math'] = report['metal_fast_math']
        target = 'evaluations' if CYCLE_MODE and mode == 'diagnostic' else 'batches'
        report.setdefault(target, []).append(phase); write(output / 'report.json', report)
        record_lock = threading.Lock()
        def record(index, value):
            with record_lock:
                phase['attempts'][index] = value
                phase['counts'] = ten_outcomes(phase['attempts'])
                write(path.parent / 'attempts.json', phase['attempts'])
                write(output / 'report.json', report)
        deadlines = batch_phase_deadlines(launch)
        collection_started = time.monotonic(); collection_wall = time.time()
        receipts, attempts = collect_batch(path, plan, state.model, opponent, launch, record,
                                           phase_deadline=deadlines['training'])
        phase.update(status='verified', attempts=attempts, counts=ten_outcomes(attempts),
                     collection_and_preparation_seconds=time.monotonic() - collection_started,
                     collection_and_preparation_wall_seconds=time.time() - collection_wall)
        write(output / 'report.json', report)
        return path, plan, phase, receipts, deadlines

    try:
        seed_offset = 0
        if CYCLE_MODE and launch.get('pending_evaluation'):
            _, _, recovery_phase, recovery_receipts, _ = collect_phase(
                'evaluation-recovery', 'diagnostic', 0, state.model, previous_pair['policy'])
            recovery_phase['status'] = 'evaluation_verified'
            training.release_receipts(recovery_receipts)
            del recovery_receipts
            write(output / 'report.json', report)
            seed_offset = launch.get('requested_games', cycle_games()) + 1
        batch_sizes = launch.get('batch_sizes', [cycle_games()] * launch['batches'])
        for index in range(launch['batches']):
            remaining(launch, 60)
            path, plan, phase, receipts, deadlines = collect_phase(f'batch-train-{index:03d}', 'training', seed_offset + (sum(batch_sizes[:index]) if 'batch_sizes' in launch else index * cycle_width()),
                state.model, previous_pair['policy'], count=batch_sizes[index])
            phase['status'] = 'updating'; write(output / 'report.json', report)
            update_progress = PPOProgress(ppo_progress_path(output, phase['name'])) if CYCLE_MODE else None
            state.model.train().requires_grad_(True)
            update_started = time.monotonic(); update_wall = time.time()
            with (wall_limit(phase_remaining(deadlines['update']))
                  if math.isfinite(phase_remaining(deadlines['update'])) else nullcontext()):
                result, rollouts = (training.live_native_update if CYCLE_MODE else training.native_update)(state.model, state.optimizer, receipts,
                    plan_sha256=binding(path)['sha256'], config=PPOConfig(**PPO), generator=state.generator,
                    remaining=lambda: phase_remaining(deadlines['update']),
                    max_optimizer_steps=plan['configuration']['max_optimizer_steps'],
                    **({'progress': update_progress} if CYCLE_MODE else {}))
            phase['update_seconds'] = time.monotonic() - update_started
            phase['update_wall_seconds'] = time.time() - update_wall
            state.model.eval()
            next_progress = TrainingProgress(state.progress.updates + 1, state.progress.rollouts + batch_sizes[index],
                                             state.progress.decisions + result.samples)
            steps = state.optimizer_steps + result.optimizer_steps
            sampler = dict(next_batch_index=next_progress.updates - 1753,
                consumed_schedule_offset=next_progress.rollouts - launch['original_parent']['progress']['rollouts'],
                optimizer_steps=steps, batch_optimizer_steps=result.optimizer_steps,
                original_parent_sampler=(state.sampler_state['original_parent_sampler']
                    if state.metadata.get('training_contract') in (training.TEN_SCHEMA, training.CYCLE_SCHEMA) else state.sampler_state))
            metadata = training.metadata(binding(path)['sha256'], launch['original_parent']['policy'],
                plan['opponent_checkpoint'], campaign_id=launch['campaign_id'], purpose='training',
                training_contract=training.CYCLE_SCHEMA if CYCLE_MODE else training.TEN_SCHEMA,
                training_games=batch_sizes[index], policy_support=plan["policy_support"])
            if update_progress is not None:
                update_progress.phase('saving')
            checkpoint_started = time.monotonic()
            with (wall_limit(phase_remaining(deadlines['update'], 0))
                  if math.isfinite(phase_remaining(deadlines['update'], 0)) else nullcontext()):
                pair = training.save_checkpoint_pair(path.parent / 'checkpoint', state.model, state.optimizer,
                    checkpoint=f"selfplay-ten-{launch['campaign_id']}-update-{next_progress.updates}",
                    ppo_config=PPO, progress=next_progress, metadata=metadata, opponent_pool=state.pool,
                    sampler_state=sampler, generator=state.generator)
                require(pair['exact_model_adam_rng_progress_pool'] is True, 'Ten-game checkpoint roundtrip incomplete')
                load_ten_launch(args.plan, args.sha256)
                entry = dict(schema=TEN_LAUNCH + '-publication', launch=binding(args.plan), plan=binding(path),
                    previous_publication=publication, parent_checkpoint=previous_pair,
                    parent_optimizer_steps=state.optimizer_steps, checkpoint=pair, optimizer_steps=steps,
                    ppo=asdict(result), receipts=[binding(Path(item['episode_directory']) / ('training-capture-receipt.json' if CYCLE_MODE else 'training-receipt.json'))
                                                for item in phase['attempts']])
                write(path.parent / 'publication.json', entry, new=True)
                publication = binding(path.parent / 'publication.json')
                write(TEN_BASE / 'latest-checkpoint.json', dict(schema=TEN_LAUNCH + '-latest',
                    publication=publication, checkpoint=pair, optimizer_steps=steps))
            phase['checkpoint_seconds'] = time.monotonic() - checkpoint_started
            state.model.requires_grad_(False)
            state = replace(state, progress=next_progress, sampler_state=sampler, checkpoint=pair['checkpoint'],
                optimizer_steps=steps, metadata=metadata, weights_sha256=pair['weights_sha256'])
            previous_pair = pair
            phase.update(status='checkpoint_verified', ppo=asdict(result), after_checkpoint=pair,
                checkpoint_after=next_progress.updates, weights_after=pair['weights_sha256'],
                weights_changed=phase['weights_before'] != pair['weights_sha256'], admitted_games=batch_sizes[index])
            report['optimizer_updates'] += 1; report['admitted_games'] += batch_sizes[index]
            report['training_samples_admitted'] += result.samples
            report['latest_checkpoint'] = next_progress.updates
            write(output / 'report.json', report)
            if update_progress is not None:
                update_progress.phase('saved')
            del receipts, rollouts
            if CYCLE_MODE:
                import gc
                gc.collect()
            if CYCLE_MODE and ('batch_sizes' not in launch or index == len(batch_sizes) - 1):
                path, evaluation_plan, evaluation_phase, evaluation_receipts, _ = collect_phase(
                    f'evaluation-{index:03d}', 'diagnostic', seed_offset + (sum(batch_sizes) if 'batch_sizes' in launch else index * cycle_width() + cycle_games()),
                    state.model, previous_pair['policy'], count=batch_sizes[index])
                evaluation_phase['status'] = 'evaluation_verified'
                training.release_receipts(evaluation_receipts)
                del evaluation_receipts
                gc.collect()
                write(output / 'report.json', report)
                print(f"Validated self-play {index + 1}: {evaluation_phase['counts']}", flush=True)
        load_ten_launch(args.plan, args.sha256)
        report['status'] = 'completed'
    except Exception as error:
        if update_progress is not None and update_progress.value['phase'] not in ('saved', 'rolled_back'):
            update_progress.phase('failed')
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        ten_finalize_report(output, report)
    finally:
        shutdowns = [read(path) for path in output.glob('*/execution/shutdown.json' if CYCLE_MODE else 'batch-*/execution/shutdown.json')]
        report['owned_cleanup_verified'] = bool(shutdowns) and all(row['owned_cleanup_verified'] for row in shutdowns)
        try:
            report['protected_inputs_unchanged'] = verify_protected(launch['protected'])
        except Exception as error:
            report.update(status='stopped', protection_error=str(error))
        report['elapsed_seconds'] = time.monotonic() - started
        report['elapsed_wall_seconds'] = time.time() - started_wall
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Restore and verify checkpoint1728 without gameplay or updates')
    diagnostic = sub.add_parser('diagnostic', help='Eight frozen games; never admitted to PPO')
    diagnostic.add_argument('--output', type=Path, required=True)
    diagnostic.add_argument('--seconds', type=float, default=7200)
    run = sub.add_parser('run', help='Exactly100 self-play games in25 complete four-game updates')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--acceptance', type=Path, required=True)
    run.add_argument('--hours', type=float, default=10)
    report = sub.add_parser('report', help='Optional bounded statistics for a completed run; no training lock')
    report.add_argument('--run-dir', type=Path, required=True)
    report.add_argument('--output', type=Path)
    report.add_argument('--seconds', type=float, default=600)
    for name in ('continuation-check', 'continue'):
        continuation = sub.add_parser(name, help='Check or manually continue the registered self-play lineage')
        continuation.add_argument('--lineage', type=Path, default=LINEAGE)
        continuation.add_argument('--acceptance', type=Path, default=CONTINUATION_ACCEPTANCE)
        if name == 'continue':
            continuation.add_argument('--output', type=Path)
            continuation.add_argument('--hours', type=float, required=True)
    for name in ('batch10-check', 'batch10-run', 'batch10-continue',
                 'cycle-check', 'cycle-run', 'cycle-continue', 'cycle-cleanup'):
        ten = sub.add_parser(name, help=('Configurable training games, bounded PPO batches and one final validated game'
            if name.startswith('cycle-') else 'Ten-game learning batches from verified1753; no extra evaluation games'))
        ten.add_argument('--batches', '--cycles', dest='batches', type=int, default=1)
        ten.add_argument('--hours', type=float, default=None)
        if name.startswith('cycle-'):
            ten.add_argument('--games', type=int, help='10–1000 training games; PPO batches up to 100, one final diagnostic')
            backend = ten.add_mutually_exclusive_group()
            backend.add_argument('--metal', action='store_true', help='PyTorch/Metal PPO; CPU gameplay and diagnostics')
            backend.add_argument('--mlx', action='store_true', help='MLX PPO with CPU gameplay and diagnostics; compatible checkpoint format')
        ten.add_argument('--output', type=Path)
        ten.add_argument('--run-dir', type=Path, required=name in ('batch10-continue', 'cycle-continue'))
    for name in ('_coordinator', '_capture', '_report'):
        worker = sub.add_parser(name)
        worker.add_argument('--plan', type=Path, required=True); worker.add_argument('--sha256', required=True)
        if name == '_capture':
            worker.add_argument('--index', type=int, required=True); worker.add_argument('--token', required=True)
            worker.add_argument('--deadline', type=lambda value: None if value == 'None' else float(value), required=True); worker.add_argument('--wall-deadline', type=lambda value: None if value == 'None' else float(value), required=True)
    args = parser.parse_args(argv)
    if args.command.startswith('cycle-'):
        if args.games is not None:
            try:
                sizes = cycle_batch_sizes(args.games)
            except ValueError as error:
                parser.error(str(error))
            if args.batches != 1:
                parser.error('--games cannot be combined with multiple --cycles')
            args.batches = len(sizes)
        select_cycle_mode()
        args.command = args.command.replace('cycle-', 'batch10-', 1)
    elif args.command in ('_capture', '_coordinator'):
        worker_plan = read(args.plan)
        if worker_plan.get('schema') in (CYCLE_LAUNCH, 'heroes-selfplay-ten-then-check-v1'):
            select_cycle_mode()
    if args.command == '_capture':
        capture_worker(args); return 0
    if args.command == '_coordinator':
        return int(coordinator(args)['status'] != 'completed')
    if args.command == '_report':
        return report_worker(args)
    started = time.monotonic()
    try:
        if args.command.startswith('batch10-'):
            with (nullcontext() if args.command == 'batch10-check' else activity_lease()):
                requested = getattr(args, 'games', None) if CYCLE_MODE else None
                if args.hours is None and not CYCLE_MODE:
                    args.hours = 10
                max_hours = (requested or 100) / 10
                inspected = ten_inspect()
                require(type(args.batches) is int and 1 <= args.batches <= 100
                        and ((CYCLE_MODE and args.hours is None)
                             or (args.hours is not None and math.isfinite(args.hours)
                                 and 120 <= args.hours * 3600 <= max_hours * 3600)),
                        f'Use 1–100 batches and a finite budget up to {max_hours:g} hours')
                if getattr(args, 'metal', False) or getattr(args, 'mlx', False):
                    # Reject unsupported selection before retirement or seed reservation.
                    ten_configuration(inspected['checked'], None if args.hours is None else args.hours * 3600,
                        games=cycle_batch_sizes(requested)[0] if requested is not None else None,
                        use_metal=getattr(args, 'metal', False), use_mlx=getattr(args, 'mlx', False))
                if args.run_dir is not None:
                    require(inspected['lineage'] is not None and inspected['lineage']['runs']
                        and Path(read(bound(inspected['lineage']['runs'][-1]))['output']).resolve() == args.run_dir.resolve(),
                        'The supplied run is not the latest ten-game launch')
                if args.command == 'batch10-cleanup':
                    inspected = cycle_retire(inspected)
                    result = dict(status='passed', gameplay_performed=False, optimizer_updates=0,
                        checkpoint=inspected['state'].progress.updates,
                        exact_model_adam_rng_progress_sampler_pool=True,
                        artifact_retention='previous analyzed runs deleted',
                        free_gib=shutil.disk_usage(ARTIFACTS).free / 1024**3)
                elif args.command == 'batch10-check':
                    result = dict(status='passed', gameplay_performed=False, optimizer_updates=0,
                        checkpoint=inspected['state'].progress.updates,
                        exact_model_adam_rng_progress_sampler_pool=True,
                        configuration=ten_configuration(inspected['checked'], None if args.hours is None else args.hours * 3600,
                            games=cycle_batch_sizes(requested)[0] if requested is not None else None,
                            use_metal=getattr(args, 'metal', False), use_mlx=getattr(args, 'mlx', False)),
                        training_games=args.batches * cycle_games(), held_out_games=(args.batches + int(inspected.get('pending_evaluation', False))) if CYCLE_MODE else 0,
                        total_physical_games=args.batches * cycle_width() + int(inspected.get('pending_evaluation', False)),
                        expected_optimizer_updates=args.batches,
                        next_fresh_seeds=(ten_seed_preview((1 + int(inspected.get('pending_evaluation', False))) * (requested + 1), width=requested + 1)
                            if requested is not None else ten_seed_preview((args.batches + int(inspected.get('pending_evaluation', False))) * cycle_width())),
                        artifact_retention='analyze, then delete before next launch' if CYCLE_MODE else 'unchanged',
                        disk_reserve_available=shutil.disk_usage(ARTIFACTS).free >= TEN_DISK_RESERVE_GIB * 1024**3,
                        next_command='cycle-run' if CYCLE_MODE else
                            ('batch10-run' if inspected['lineage'] is None else 'batch10-continue'))
                    if CYCLE_MODE:
                        from experiments.learner.interaction_guard import STACK_MANAGEMENT_VERSION
                        result['policy_support'] = STACK_MANAGEMENT_VERSION
                    if requested is not None:
                        held_out = 1 + int(inspected.get('pending_evaluation', False))
                        result.update(training_games=requested, held_out_games=held_out,
                            batch_sizes=cycle_batch_sizes(requested), total_physical_games=requested + held_out)
                else:
                    if CYCLE_MODE:
                        inspected = cycle_retire(inspected)
                        args.command = 'batch10-run'
                    launch = prepare_ten(args, inspected)
                    print(f"Private run: {launch['output']}", flush=True)
                    result = supervise(launch)
            for phase in result.get('batches', []) + result.get('evaluations', []):
                ten_print_phase(phase)
            if CYCLE_MODE and args.command not in ('batch10-check', 'batch10-cleanup'):
                print(f"Run {result['status']}; checkpoint {result.get('latest_checkpoint')}; "
                      f"PPO updates {result.get('optimizer_updates', 0)}.", flush=True)
                if result.get('error') or result.get('supervisor_errors'):
                    print(result.get('error') or result['supervisor_errors'], flush=True)
                print(f"Private report: {Path(launch['output']) / 'report.json'}", flush=True)
            else:
                print(json.dumps(result, indent=2))
            return int(result['status'] not in ('passed', 'completed'))
        if args.command == 'report':
            result = report_completed(args)
            print(json.dumps(result, indent=2))
            return int(result['status'] != 'completed')
        launch = None
        with activity_lease():
            if args.command == 'check':
                identity = source_identity()
                pointer, state, cfg, execution, protected, _ = inspect_parent()
                result = dict(status='passed', checkpoint=state.progress.updates, source=identity,
                              parent=pointer['parent'], configuration=cfg, execution=execution,
                              protected_files=len(protected), gameplay_performed=False, optimizer_updates=0)
            elif args.command == 'continuation-check':
                identity, lineage, state, cfg, execution, acceptance = inspect_continuation(args)
                result = dict(status='passed', checkpoint=state.progress.updates, source=identity,
                    admitted_games=lineage['admitted_games'], remaining_games=lineage['remaining_games'],
                    target_games=100, parent=lineage['selected'], configuration=cfg, execution=execution,
                    acceptance=acceptance, gameplay_performed=False, optimizer_updates=0,
                    exact_model_adam_rng_progress_sampler_pool=True,
                    next_fresh_seeds=seed_preview('training', lineage['remaining_games']))
            else:
                launch = (prepare_continuation(args) if args.command == 'continue'
                          else prepare(args, 'training' if args.command == 'run' else 'diagnostic'))
                result = supervise(launch)
        if launch is not None:
            print('Run completed and training lock released. Optional behavior statistics were not started.', flush=True)
            print('Generate them separately: ' + shlex.join([sys.executable, str(Path(__file__).resolve()),
                  'report', '--run-dir', launch['output'], '--seconds', '600']), flush=True)
        print(json.dumps(result, indent=2)); return 0
    finally:
        from experiments.learner.game_progress import format_duration
        print(f'Total running time: {format_duration(time.monotonic() - started)}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
