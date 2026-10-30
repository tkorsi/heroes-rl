#!/usr/bin/env python3
"""A separate, explicitly invoked 100-game self-play experiment from update 1728.

The original Nullkiller campaign is read-only. Diagnostics, consumed seeds and
failed batches cannot become training inputs. Manual continuations use new run
directories, fresh reservations and the latest published lineage checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
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
    left = min(launch['hard_deadline_monotonic'] - time.monotonic(), launch['hard_deadline_unix'] - time.time()) - reserve
    if left <= 0:
        raise TimeoutError('Self-play experiment deadline exhausted')
    return left


def batch_phase_deadlines(launch):
    """Preserve the parent's phase offsets separately from each capture timeout."""
    now, wall = time.monotonic(), time.time()
    cfg = launch['configuration']
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


def configure_native(execution):
    import torch
    from experiments.learner import ppo
    from experiments.learner.native_execution import check_native_execution
    from experiments.learner.native_backend import verified_manifest
    check_native_execution(execution)
    native = execution['native_learner']
    verified_manifest(native['manifest'], native['sha256'])
    torch.set_num_threads(1)
    ppo.NATIVE_EXECUTION = native
    ppo.PREFIX_HELPERS = 0
    ppo.REPLAY_TEXT_LEAVES = False
    ppo.TEXT_REPLAY_WORKERS = 0
    ppo.EXPERIMENTS = frozenset()


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
    from experiments.learner import selfplay_training as training
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    require(binding(args.plan)['sha256'] == args.sha256, 'Batch plan hash changed')
    plan = read(args.plan); launch = load_launch(plan['launch']['path'], plan['launch']['sha256'])
    require(args.deadline <= launch['hard_deadline_monotonic'] and args.wall_deadline <= launch['hard_deadline_unix'],
            'Worker cannot extend the campaign deadline')
    configure_native(plan['execution'])
    row = plan['schedule'][args.index]
    learner = training.load_policy(bound(plan['checkpoint']), plan['checkpoint_source'])
    opponent = training.load_policy(bound(plan['opponent_checkpoint']), plan['opponent_checkpoint_source'])
    seat = row['learner_player']
    models = (learner.model, opponent.model) if seat == 0 else (opponent.model, learner.model)
    checkpoints = (learner.checkpoint, opponent.checkpoint) if seat == 0 else (opponent.checkpoint, learner.checkpoint)
    output = Path(plan['execution_output']); output.mkdir(exist_ok=True)
    launcher = DeadlineVcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']),
        deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
        evidence_path=output / f"native-{row['episode']:06d}-deadline.jsonl")
    class DayBoundedCollector(ContextualCollector):
        def act(self, request):
            require(request.record['day'] <= 365, 'A sampled policy action exceeds the preserved day365 bound')
            return super().act(request)

    collector = DayBoundedCollector(models, checkpoints, learner_player=seat, opponent_pool=POOL,
        interface=INTERFACE, purpose=plan['mode'], origin='native', policy_seeds=tuple(row['policy_seeds']), guard_version=GUARD)
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=365, max_actions=5120,
        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30),
        ledger=True, replay=True, interface=INTERFACE,
        agents={color: {'checkpoint': checkpoint, 'kind': 'frozen-neural', 'purpose': plan['mode']}
                for color, checkpoint in zip(('red', 'blue'), checkpoints)})
    cleanup = False
    try:
        collected = collector.run_episode(env, episode=row['episode'], seed=row['engine_seed'],
                                         max_requests_per_player=4960, max_dialog_requests_per_player=160)
        folder = env.episode_dir
        write(folder / 'neural-rollout.json', collected.to_dict(), new=True)
        require(read(folder / 'manifest.json')['status'] in ('completed', 'truncated'), 'Technical capture failure')
    except Exception as error:
        from validation.probes.validate_hero_policy import retain_failed_evidence
        failure = dict(status='failed', error=f'{type(error).__name__}: {error}',
                       outcomes={'red': None, 'blue': None}, rewards={'red': None, 'blue': None})
        seconds = min(args.deadline - time.monotonic(), args.wall_deadline - time.time())
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
                          for name in ('manifest.json', 'neural-rollout.json')})
    print(json.dumps(message), flush=True)


def collect_batch(plan_path, plan, learner, opponent, launch, save_attempt, *, phase_deadline):
    from experiments.learner import selfplay_training as training
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from validation.probes.validate_hero_policy import wall_limit
    output = Path(plan['execution_output']); output.mkdir()
    jobs, receipts, attempts = [], [], []
    current_index = None
    try:
        for first in range(0, len(plan['schedule']), 4):
            wave = []
            for index in range(first, min(first + 4, len(plan['schedule']))):
                current_index = index
                phase_remaining(phase_deadline)
                row = plan['schedule'][index]
                folder = output / f"episode-{row['episode']:06d}"
                capability = training.create_capture_binding(plan_path, binding(plan_path)['sha256'], index, folder)
                token = secrets.token_hex(32)
                allowance = plan['configuration']['attempt_wall_seconds']
                deadline = min(time.monotonic() + allowance, phase_deadline['hard_deadline_monotonic'] - 2)
                wall = min(time.time() + allowance, phase_deadline['hard_deadline_unix'] - 2)
                log = (output / f'worker-{index:02d}.log').open('xb')
                command = [sys.executable, str(Path(__file__).resolve()), '_capture', '--plan', str(plan_path),
                           '--sha256', binding(plan_path)['sha256'], '--index', str(index), '--token', token,
                           '--deadline', str(deadline), '--wall-deadline', str(wall)]
                process = guarded_popen(command, deadline_monotonic=deadline, deadline_wall=wall,
                    evidence_path=output / f'worker-{index:02d}-deadline.jsonl', cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
                job = dict(process=process, log=log, binding=capability, index=index, folder=folder,
                           deadline=deadline, wall_deadline=wall,
                           deadlines=dict(capture_seconds=allowance, capture_deadline_monotonic=deadline,
                                          capture_deadline_unix=wall, collection_and_verification=phase_deadline))
                jobs.append(job); wave.append(job)
                training.register_capture_launch(capability, process, token)
                save_attempt(index, dict(**row, status='running', outcomes={'red': None, 'blue': None},
                                        rewards={'red': None, 'blue': None}, deadlines=job['deadlines']))
            for job in wave:
                current_index = job['index']
                process = job['process']
                stdout, _ = process.communicate(timeout=min(605, phase_remaining(phase_deadline)))
                require(process.returncode == 0, 'Self-play capture failed; no retries or partial batch admission')
                close_owned_process(process, timeout=min(5, phase_remaining(phase_deadline)))
                cleanup_from_evidence(output / f"native-{plan['schedule'][job['index']]['episode']:06d}-deadline.jsonl",
                                      timeout=min(5, phase_remaining(phase_deadline)))
                require(len(stdout) <= 65536, 'Oversized capture completion')
                training.accept_capture_completion(job['binding'], json.loads(stdout))
                def verification_remaining():
                    return phase_remaining(phase_deadline)
                with wall_limit(verification_remaining()):
                    receipt = training.verify_training_capture(job['binding'], learner, opponent,
                        replay_binary=Path(plan['replay_binary']['path']), data=Path(plan['data']),
                        remaining=verification_remaining)
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
                attempts.append(attempt); receipts.append(receipt)
                save_attempt(job['index'], attempt)
        require(len(receipts) == len(plan['schedule']), 'Incomplete self-play batch')
        return tuple(receipts), attempts
    except Exception as error:
        if current_index is not None:
            save_attempt(current_index, dict(**plan['schedule'][current_index], status='failed',
                error=f'{type(error).__name__}: {error}', outcomes={'red': None, 'blue': None},
                rewards={'red': None, 'blue': None}, training_samples_admitted=0,
                deadlines=next((job['deadlines'] for job in jobs if job['index'] == current_index),
                               {'collection_and_verification': phase_deadline})))
        raise
    finally:
        errors = close_collection_jobs(jobs, output, launch)
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
    require(os.getppid() == launch['supervisor_pid'], 'Only the lease-owning supervisor may launch this campaign')
    configure_native(launch['execution'])
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
                with wall_limit(phase_remaining(deadlines['update'])):
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
                with wall_limit(phase_remaining(deadlines['update'], 0)):
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
    for batch in report.get('batches', []):
        for attempt in batch['attempts']:
            if attempt.get('status') != 'passed':
                continue
            try:
                rows.append(analyze_episode(attempt['episode_directory'], attempt['learner_player']))
            except Exception as error:
                errors.append(dict(episode_directory=attempt['episode_directory'], error=str(error)))
    try:
        summary = aggregate_episodes(rows)
        write(Path(output) / 'behavior-analysis.json', dict(summary=summary, errors=errors,
              complete=not errors, policy_input=False, reward_input=False, admission_input=False), new=True)
    except Exception as error:
        write(Path(output) / 'behavior-analysis-error.json', dict(error=str(error), errors=errors,
              admission_unchanged=True), new=True)


def supervise(launch):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output = Path(launch['output']); path = output / 'launch.json'; process = None; errors = []
    try:
        with (output / 'coordinator.log').open('xb') as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_coordinator',
                    '--plan', str(path), '--sha256', binding(path)['sha256']],
                deadline_monotonic=launch['hard_deadline_monotonic'], deadline_wall=launch['hard_deadline_unix'],
                evidence_path=output / 'coordinator-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            process.wait(timeout=remaining(launch) + 1)
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=5)
            for evidence in output.glob('batch-*/execution/*-deadline.jsonl'):
                cleanup_from_evidence(evidence, timeout=5)
        except Exception as error:
            errors.append(str(error))
        protected = verify_protected(launch['protected'])
        write(output / 'shutdown.json', dict(owned_cleanup_verified=not errors, errors=errors,
              returncode=process.returncode if process else None, protected_inputs_unchanged=protected), new=True)
        if launch['mode'] == 'training' and not errors and protected:
            finalize_stopped_run(output, launch)
    report = read(output / 'report.json') if (output / 'report.json').exists() else {'status': 'stopped'}
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
    for name in ('continuation-check', 'continue'):
        continuation = sub.add_parser(name, help='Check or manually continue the registered self-play lineage')
        continuation.add_argument('--lineage', type=Path, default=LINEAGE)
        continuation.add_argument('--acceptance', type=Path, default=CONTINUATION_ACCEPTANCE)
        if name == 'continue':
            continuation.add_argument('--output', type=Path)
            continuation.add_argument('--hours', type=float, required=True)
    for name in ('_coordinator', '_capture'):
        worker = sub.add_parser(name)
        worker.add_argument('--plan', type=Path, required=True); worker.add_argument('--sha256', required=True)
        if name == '_capture':
            worker.add_argument('--index', type=int, required=True); worker.add_argument('--token', required=True)
            worker.add_argument('--deadline', type=float, required=True); worker.add_argument('--wall-deadline', type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == '_capture':
        capture_worker(args); return 0
    if args.command == '_coordinator':
        return int(coordinator(args)['status'] != 'completed')
    started = time.monotonic()
    try:
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
                analyze_completed(result, launch['output'])
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
