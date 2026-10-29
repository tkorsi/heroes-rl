"""Optimizer-only PPO benchmark on disposable exact copies.

Restores a parent checkpoint pair (model, Adam, minibatch generator, global RNG) with the strict
loaders, audits the four individually verified games of an archived local-runner attempt, and
runs the PPO kernel (``ppo._update``) of the selected checkout exactly once. The attempt may be
an entirely rolled-back update: its files are diagnostic inputs, never new admission receipts.
The result records model, gradients, Adam, all RNG, progress, sampler and pool state plus the
minibatch order. Progress remains unchanged. Nothing is written into any checkpoint lineage.

    ppo_bench.py --checkout DIR --parent-plan P --parent-report R --policy F --resume F --batch DIR
                 --output JSON [--reference-resume F] [--save-state F] [--compare-state F]
                 [--experiment shared_graph_encodings] [--prefix-helpers N] [--profile OUT] [--output JSON]
                 [--capture-plan P] [--capture-report R] [--require-rolled-back] [--audit-only]
                 [--wall-seconds 10800] [--deadline-unix T --deadline-monotonic T]

The checkout's own code is imported (the accepted implementation or the fast path); a historical
checkpoint written by another Torch build is read through the loader's explicit historical-version
argument (on this Linux host together with ``bench_site/sitecustomize.py``).
An independent guardian is armed before child execution, holds a fresh process group for at
most 10800 seconds, and stops it if the supervisor disappears. Parent and worker retain shared
official owner locks after an exclusive admission check. A supplied original study deadline
always wins; ten seconds within that deadline are reserved for cleanup. The bound includes
audit/restore time, not just the timed optimizer kernel. Guard evidence stays beside the result.
"""
import argparse
from contextlib import contextmanager
import cProfile
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import random
import re
import resource
import subprocess
import sys
import time

if '--checkout' in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index('--checkout') + 1]).resolve()))

import torch

from experiments.learner import checkpoints as cp
from experiments.learner.builtin_training_resume import _completion, _metadata, _exact, require
from experiments.learner.builtin_rollout import load_builtin_episode
from experiments.learner.builtin_gameplay import _samples
from experiments.learner import ppo
from experiments.learner.ppo import PPOConfig, _update
from experiments.headless.contextual_evidence import strict_json
from experiments.headless.process_deadline import guarded_popen, close_owned_process


def file_binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}


def same_file_bytes(path, expected):
    actual = file_binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'Bound input changed: ' + str(path))
    return actual


def audit_capture(args):
    """Read-only audit of an archived update attempt; never mint fresh admission receipts.

    The source/runtime and four persisted verification receipts are checked against their
    historical bindings. These files may diagnose a rolled-back update, but cannot re-enter
    native admission. The benchmark writes no campaign checkpoint or seed ledger.
    """
    from experiments.learner.builtin_gameplay import _capture_file_bindings
    from experiments.headless import builtin_evidence, builtin_replay
    batch = Path(args.batch).resolve()
    plan_path = Path(args.capture_plan or batch.parent / 'plan.json')
    report_path = Path(args.capture_report or batch / 'report.json')
    plan = strict_json(plan_path.read_bytes()); report = strict_json(report_path.read_bytes())
    plan_binding = file_binding(plan_path)
    require(report['plan_sha256'] == plan_binding['sha256'], 'Captured plan/report mismatch')
    require(plan['mode'] == 'training' and len(plan['schedule']) == 4
            and all(row['phase'] == 'training' for row in plan['schedule']), 'Expected one complete four-game capture')
    for role in ('plan', 'report', 'policy', 'resume'):
        path = getattr(args, 'parent_' + role) if role in ('plan', 'report') else getattr(args, role)
        same_file_bytes(path, plan['parent_descriptor'][role])
    if args.require_rolled_back:
        require(report.get('update_status') == 'rolled_back' and report.get('training_performed') is False
                and report.get('training_samples') == report.get('optimizer_steps') == report.get('optimizer_updates') == 0
                and not report.get('after_checkpoint') and not list(batch.glob('after-*.pt')),
                'The historical failed update must remain entirely rolled back')
    # Historical recipe, not candidate verifier bytes: no compatibility claim is inferred from a path.
    for name, sha in plan['source']['source_sha256'].items():
        raw = subprocess.check_output(['git', '-C', str(args.checkout), 'show', plan['source']['git_commit'] + ':' + name])
        require(hashlib.sha256(raw).hexdigest() == sha, 'Historical source binding differs: ' + name)
    for name, sha in plan['runtime'].items():
        require(file_binding(Path(plan['binary']).parent / name)['sha256'] == sha, 'Historical runtime differs: ' + name)
    same_file_bytes(plan['game_map'], plan['map'])
    attempts = report.get('attempts', [])
    require(len(attempts) == 4, 'Expected all four historical attempts')
    rows = []
    for index, (attempt, row) in enumerate(zip(attempts, plan['schedule'], strict=True)):
        require(attempt['status'] == 'passed' and all(attempt.get(k) is True for k in
                ('cleanup_verified', 'identity_verified', 'completion_receipt', 'replay_verified', 'weights_unchanged')),
                'Historical game was not completely verified')
        folder = batch / Path(attempt['episode_directory']).name
        receipt = strict_json((folder / 'training-receipt.json').read_bytes())
        bound = receipt['binding']
        require(receipt['schema'] == 'heroes-builtin-training-receipt-v1' and receipt['admission_from_file'] is False
                and bound['schedule_index'] == index and bound['plan_sha256'] == plan_binding['sha256']
                and json.loads(bound['row_json']) == row and json.loads(bound['source_json']) == plan['source']
                and bound['weights_sha256'] == attempt['weights_sha256'], 'Historical receipt binding differs')
        same_file_bytes(args.policy, json.loads(bound['checkpoint_file_json']))
        manifest = strict_json((folder / 'manifest.json').read_bytes())
        expected = set(_capture_file_bindings(folder, manifest)) | {
            'training-history-reconstruction.json', 'training-policy-reconstruction.json',
            'training-verification/builtin-replay-verification.json', 'training-verification/native-report.json'}
        require(set(receipt['files']) == expected, 'Historical receipt omits bound evidence')
        for name, identity in receipt['files'].items():
            target = (folder / name).resolve()
            require(target.is_relative_to(folder.resolve()), 'Receipt path escapes its capture')
            same_file_bytes(target, identity)
        history = strict_json((folder / 'training-history-reconstruction.json').read_bytes())
        policy = strict_json((folder / 'training-policy-reconstruction.json').read_bytes())
        replay = strict_json((folder / 'training-verification/builtin-replay-verification.json').read_bytes())
        final = builtin_evidence.read_finalization(folder, manifest)
        require(history == receipt['history_reconstruction'] and history.get('verified') is True
                and history.get('opponent_history') is None and policy == receipt['policy_reconstruction']
                and policy['maximum_absolute_difference'] == 0 and policy['weights_unchanged'] is True
                and replay.get('verified') is True and replay['identity_check']['mode'] == 'exact_capture'
                and replay.get('history') == history and replay['training_samples'] == 0
                and replay['training_performed'] is False, 'Historical replay or policy verification is incomplete')
        builtin_replay.validate_native_report(replay['native'], replay['wire'], final, cutoff=manifest['status'] == 'truncated')
        require(receipt['samples'] == attempt['sampled_choices'] == policy['actions_resampled'], 'Capture sample count differs')
        rows.append({'receipt': file_binding(folder / 'training-receipt.json'), 'samples': receipt['samples'],
                     'files': receipt['files'], 'historical_status': attempt['status']})
    return {'plan': plan_binding, 'report': file_binding(report_path), 'source': plan['source'], 'runtime': plan['runtime'],
            'map': plan['map'], 'games': rows, 'historical_update_status': report.get('update_status'),
            'diagnostic_only': True, 'official_samples_admitted': 0, 'official_updates': 0}


class Rollout:
    """The fields of a verified training rollout that the PPO kernel reads."""
    def __init__(self, experience, samples):
        self.experience, self.samples = experience, samples
        self.history, self.decisions = experience.history, experience.decisions


def tensor_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().contiguous(); digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(ctypes.string_at(data.data_ptr(), data.numel() * data.element_size()) if data.numel() else b'')
        elif isinstance(item, dict):
            for key in sorted(item, key=str): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item: visit(element)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def difference(reference, state):
    """Largest absolute and relative element differences per part, and how many tensors differ at all."""
    def tensors(value):
        if isinstance(value, torch.Tensor): yield value
        elif isinstance(value, dict):
            for key in sorted(value, key=str): yield from tensors(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: yield from tensors(item)
    result = {}
    for part in ('model', 'gradients', 'adam'):
        absolute = relative = 0.0; differing = total = 0
        for left, right in zip(tensors(reference[part]), tensors(state[part]), strict=True):
            if not left.is_floating_point(): continue
            total += 1
            if torch.equal(left, right): continue
            differing += 1
            delta = (left.double() - right.double()).abs()
            absolute = max(absolute, delta.max().item())
            relative = max(relative, (delta / left.double().abs().clamp_min(1e-30)).max().item())
        result[part] = dict(tensors=total, differing=differing, max_absolute=absolute, max_relative=relative)
    return result


def restore(plan_path, report_path, policy_path, resume_path):
    plan, report = json.loads(Path(plan_path).read_text()), json.loads(Path(report_path).read_text())
    require(report['plan_sha256'] == file_binding(plan_path)['sha256'], 'Parent plan/report bytes differ')
    for kind, path in (('policy', policy_path), ('resume', resume_path)):
        same_file_bytes(path, report['after_checkpoint'][kind])
    metadata = _metadata(plan, report['plan_sha256'])
    allowed = (cp.BUILTIN_GAMEPLAY_PROVENANCE,)
    interface = plan['source']['interface']
    options = {} if plan['source']['torch'] == str(torch.__version__) else {'historical_torch_version': plan['source']['torch']}
    policy = cp.load_policy_checkpoint(policy_path, expected_source_metadata=metadata, allowed_provenance=allowed, interface=interface,
                                       **options)
    model = policy.model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
    generator = torch.Generator()
    state = cp.load_checkpoint(resume_path, model, optimizer, expected_ppo_config=plan['configuration']['ppo'],
                               expected_source_metadata=metadata, generators={'minibatch': generator}, interface=interface,
                               allowed_provenance=allowed, at_rollout_boundary=True, **options)
    progress, steps, sampler = _completion(plan, report)
    require(state.progress == progress and state.sampler_state == sampler, 'Parent progress differs')
    raw = torch.load(resume_path, map_location='cpu', weights_only=True)
    require(state.checkpoint == policy.checkpoint and cp.model_weight_digest(model) == policy.weights_sha256,
            'Parent policy/resume weights differ')
    for actual, expected in ((dict(model.state_dict()), raw['model_state']),
            (optimizer.state_dict(), raw['resume']['optimizer_state']),
            (state.opponent_pool.state_dict(), policy.opponent_pool.state_dict()),
            (torch.get_rng_state(), raw['rng']['torch_cpu']), (random.getstate(), raw['rng']['python']),
            (generator.get_state(), raw['rng']['torch_generators']['minibatch'])):
        _exact(actual, expected)
    return model, optimizer, generator, plan, state


def rollouts(batch):
    result = []
    for folder in sorted(Path(batch).glob('episode-*')):
        raw = strict_json((folder/'builtin-training-rollout.json').read_bytes())
        experience = load_builtin_episode(raw['experience'])
        receipt = strict_json((folder/'training-receipt.json').read_bytes())
        reference = ('training-verification/builtin-replay-verification.json#sha256='
                     + receipt['files']['training-verification/builtin-replay-verification.json']['sha256'])
        samples = _samples(experience, reference)
        require(len(samples) == receipt['samples'], 'Rebuilt samples differ from the receipt')
        result.append(Rollout(experience, samples))
    require(len(result) == 4, 'Expected the four verified games of one batch')
    return tuple(result)


def run_once(args, items, profile=None):
    model, optimizer, generator, plan, parent = restore(args.parent_plan, args.parent_report, args.policy, args.resume)
    torch.set_num_threads(1)
    config = PPOConfig(**plan['configuration']['ppo'])
    order_state = generator.get_state().clone()
    before = full_state(model, optimizer, generator, parent)
    model.train().requires_grad_(True)
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    if profile is not None: profile.enable()
    report = _update(model, optimizer, items, config=config, generator=generator,
                     max_optimizer_steps=plan['configuration']['max_optimizer_steps'])
    if profile is not None: profile.disable()
    seconds = time.perf_counter() - started
    cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = ((cpu_finished.ru_utime + cpu_finished.ru_stime)
                   - (cpu_started.ru_utime + cpu_started.ru_stime))
    order_generator = torch.Generator(); order_generator.set_state(order_state)
    windows = sum((len(item.samples) + config.decisions_per_window - 1) // config.decisions_per_window for item in items)
    order = torch.randperm(windows, generator=order_generator).tolist()
    model.eval().requires_grad_(False)
    after = full_state(model, optimizer, generator, parent)
    for name in ('progress', 'sampler', 'pool'):
        _exact(before[name], after[name])  # this diagnostic advances no official bookkeeping
    return dict(seconds=seconds, cpu_seconds=cpu_seconds, average_cpu_cores=cpu_seconds / seconds,
                report={k: v for k, v in vars(report).items() if k != 'timings_seconds'},
                timings=report.timings_seconds, sampler_order=order,
                model_sha256=tensor_digest(dict(model.state_dict())), weights_sha256=cp.model_weight_digest(model),
                gradient_sha256=tensor_digest([p.grad for p in model.parameters()]),
                adam_sha256=tensor_digest(optimizer.state_dict()), generator_sha256=tensor_digest(generator.get_state()),
                before_full_state_sha256=tensor_digest(before), full_state_sha256=tensor_digest(after),
                state_parts_sha256={k: tensor_digest(v) for k, v in after.items()},
                diagnostic_state={**after, 'gradients': [p.grad for p in model.parameters()]},
                model=model, optimizer=optimizer, generator=generator)


def full_state(model, optimizer, generator, parent):
    """Every learned/resume component, with progress deliberately unchanged in a diagnostic."""
    from copy import deepcopy
    return deepcopy(dict(model=dict(model.state_dict()), adam=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), python_rng=random.getstate(), minibatch_rng=generator.get_state(),
        progress=vars(parent.progress), sampler=parent.sampler_state, pool=parent.opponent_pool.state_dict()))


@contextmanager
def official_exclusion(*, shared=False):
    """Every diagnostic starts exclusively; its guarded worker then holds a shared lease.

    Official training requires an exclusive lock. Shared parent/worker leases avoid passing
    arbitrary file descriptors through the deadline guard. If the parent dies before worker
    acquisition and an official runner wins the lock, the worker fails before any PPO work.
    """
    lock = Path.home() / 'Projects/Heroes-artifacts/local-script-training/owner.lock'
    descriptor = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield descriptor
    finally:
        os.close(descriptor)


def supervise(arguments, wall_seconds, evidence_path, *, deadline_unix=None, deadline_monotonic=None):
    with official_exclusion() as descriptor:
        # guarded_popen forbids pass_fds. Both holders independently block the official EX lock.
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return _supervise_locked(arguments, wall_seconds, evidence_path,
                                deadline_unix=deadline_unix, deadline_monotonic=deadline_monotonic)


def effective_deadlines(wall_seconds, *, deadline_unix=None, deadline_monotonic=None):
    require(math.isfinite(wall_seconds) and 10 < wall_seconds <= 10800, 'Wall budget must include ten seconds for cleanup')
    require((deadline_unix is None) == (deadline_monotonic is None), 'Supply both original study deadlines')
    wall, monotonic = time.time(), time.monotonic()
    hard_wall, hard_monotonic = wall + wall_seconds, monotonic + wall_seconds
    if deadline_unix is not None:
        require(all(math.isfinite(value) and value > 0 for value in (deadline_unix, deadline_monotonic)),
                'Original study deadlines must be finite positive timestamps')
        hard_wall, hard_monotonic = min(hard_wall, deadline_unix), min(hard_monotonic, deadline_monotonic)
    require(min(hard_wall - wall, hard_monotonic - monotonic) > 10, 'No original study time remains for update and cleanup')
    return hard_wall, hard_monotonic


def _supervise_locked(arguments, wall_seconds, evidence_path, *, deadline_unix=None, deadline_monotonic=None):
    hard_wall, hard_monotonic = effective_deadlines(wall_seconds, deadline_unix=deadline_unix,
                                                 deadline_monotonic=deadline_monotonic)
    command = [sys.executable, str(Path(__file__).resolve()), *arguments, '--worker',
               '--deadline-unix', repr(hard_wall), '--deadline-monotonic', repr(hard_monotonic)]
    started = time.time()
    process = guarded_popen(command, deadline_monotonic=hard_monotonic - 10,
                            deadline_wall=hard_wall - 10, evidence_path=evidence_path)
    cleanup_verified, cleanup_error = False, None
    code = None
    try:
        code = process.wait(timeout=max(0, min(hard_wall - time.time(), hard_monotonic - time.monotonic()) - 10))
        status = 'finished' if code == 0 else 'failed'
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        status = 'timed_out_or_interrupted'
    finally:
        try:
            close_owned_process(process, timeout=10, deadline_monotonic=hard_monotonic, deadline_wall=hard_wall)
            cleanup_verified = True
        except Exception as error:
            cleanup_error = type(error).__name__ + ': ' + str(error)
        code = process.returncode
    result_code = code if code is not None else 1
    if not cleanup_verified or status != 'finished': result_code = result_code or 1
    return result_code, {'pid': process.pid, 'command': command, 'started_unix': started,
                  'deadline_unix': hard_wall, 'deadline_monotonic': hard_monotonic,
                  'compute_deadline_unix': hard_wall - 10, 'compute_deadline_monotonic': hard_monotonic - 10,
                  'finished_unix': time.time(), 'guard_evidence': str(evidence_path),
                  'guardian_armed_before_exec': True, 'official_owner_lock': 'parent and worker shared after exclusive admission',
                  'status': status, 'returncode': code, 'child_reaped': code is not None,
                  'cleanup_verified': cleanup_verified, 'cleanup_error': cleanup_error}


def _main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkout', type=Path, required=True)
    for name in ('parent-plan', 'parent-report', 'policy', 'resume', 'batch'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1); p.add_argument('--reference-resume', type=Path)
    p.add_argument('--profile', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--save-state', type=Path, help='write the last repetition\'s model, gradients and Adam state')
    p.add_argument('--compare-state', type=Path, help='report differences from a state saved by --save-state')
    p.add_argument('--experiment', action='append', default=[], choices=['shared_graph_encodings'],
                   help='enable a separate numerical experiment (never a default)')
    p.add_argument('--prefix-helpers', type=int, default=0, help='exact forked prefix-encoding helpers (0: serial)')
    p.add_argument('--no-text-replay', dest='text_leaf_replay', action='store_false',
                   help='compute every text leaf occurrence separately (the --no-text-replay fallback)')
    p.add_argument('--text-replay-workers', type=int, help='replay threads (default: as many as prefix helpers)')
    p.add_argument('--native-manifest', type=Path, help='explicit verified native learner manifest; no default promotion')
    p.add_argument('--native-manifest-sha256', help='expected SHA256 of the native learner manifest')
    p.add_argument('--native-prefix-threads', type=int, default=1, help='native independent-prefix thread budget')
    p.add_argument('--capture-plan', type=Path, help='historical plan bound to --batch; default: its parent/plan.json')
    p.add_argument('--capture-report', type=Path, help='historical report; default: --batch/report.json')
    p.add_argument('--require-rolled-back', action='store_true', help='require zero admission and no after checkpoint in the original attempt')
    p.add_argument('--audit-only', action='store_true', help='verify persisted capture/parent bindings; run no PPO')
    p.add_argument('--wall-seconds', type=float, default=10800, help='fixed independent diagnostic wall limit (maximum 10800 seconds)')
    p.add_argument('--deadline-unix', type=float, help='original enclosing study hard deadline; never extended')
    p.add_argument('--deadline-monotonic', type=float, help='original same-host monotonic hard deadline; paired with --deadline-unix')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    require((a.native_manifest is None) == (a.native_manifest_sha256 is None),
            'Native manifest and its explicit SHA256 must be supplied together')
    require(a.native_prefix_threads >= 1, 'Native prefix thread count must be positive')
    require(a.native_manifest is not None or a.native_prefix_threads == 1,
            'Native prefix threads require an explicit native manifest')
    if a.native_manifest is not None:
        require(re.fullmatch(r'[0-9a-f]{64}', a.native_manifest_sha256) is not None,
                'Native manifest SHA256 must be a lowercase hexadecimal digest')
    require(a.repeat == 1, 'One bounded update only')
    effective_deadlines(a.wall_seconds, deadline_unix=a.deadline_unix, deadline_monotonic=a.deadline_monotonic)
    require(a.output is not None, 'A fresh diagnostic output path is required')
    for path in (a.output, a.save_state, a.profile):
        if path is not None:
            require(not path.exists(), 'Diagnostic output already exists: ' + str(path))
            official = Path.home() / 'Projects/Heroes-artifacts/local-script-training'
            require(not path.resolve().is_relative_to(official.resolve()), 'Never write diagnostic output into official lineage')
    if not a.worker:
        guard_path = a.output.with_suffix(a.output.suffix + '.guard.json')
        evidence_path = a.output.with_suffix(a.output.suffix + '.guard.jsonl')
        require(not guard_path.exists() and not evidence_path.exists(), 'Diagnostic guard receipt already exists')
        code, guard = supervise(sys.argv[1:], a.wall_seconds, evidence_path,
                                deadline_unix=a.deadline_unix, deadline_monotonic=a.deadline_monotonic)
        with guard_path.open('x') as stream:
            json.dump(guard, stream, indent=2, sort_keys=True); stream.write('\n')
        return code
    require(Path(ppo.__file__).resolve().is_relative_to(a.checkout.resolve()), 'Imported PPO differs from selected checkout')
    if a.require_rolled_back:
        require(not a.experiment and a.prefix_helpers == 0 and a.text_replay_workers in (None, 0, 2),
                'Captured rollback stress allows only exact serial or two-thread replay, without fork helpers')
    checkout_head = subprocess.check_output(['git', '-C', str(a.checkout), 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', '-C', str(a.checkout), 'status', '--porcelain'], text=True).strip(),
            'Selected implementation checkout must be immutable and clean')
    driver = file_binding(__file__)
    audit = audit_capture(a)
    input_bindings = {k: file_binding(getattr(a, k)) for k in ('parent_plan', 'parent_report', 'policy', 'resume')}
    if a.audit_only:
        model, optimizer, generator, _, parent = restore(a.parent_plan, a.parent_report, a.policy, a.resume)
        a.output.write_text(json.dumps(dict(audit=audit, parent=input_bindings,
            restored_full_state_sha256=tensor_digest(full_state(model, optimizer, generator, parent)),
            audit_only=True, official_samples_admitted=0, official_updates=0), indent=2) + '\n')
        return 0
    if a.experiment or a.prefix_helpers:
        require(hasattr(ppo, 'EXPERIMENTS') and hasattr(ppo, 'PREFIX_HELPERS'), 'This checkout has no such execution choice')
        ppo.EXPERIMENTS = frozenset(a.experiment)
        ppo.PREFIX_HELPERS = a.prefix_helpers
    if not a.text_leaf_replay or a.text_replay_workers is not None:
        require(hasattr(ppo, 'REPLAY_TEXT_LEAVES') and hasattr(ppo, 'TEXT_REPLAY_WORKERS'), 'This checkout has no text-leaf replay')
        ppo.REPLAY_TEXT_LEAVES = a.text_leaf_replay
        ppo.TEXT_REPLAY_WORKERS = a.text_replay_workers
    native_binding = None
    if a.native_manifest is not None:
        require(hasattr(ppo, 'NATIVE_EXECUTION'), 'Selected checkout has no native PPO integration')
        native_binding = file_binding(a.native_manifest)
        require(native_binding['sha256'] == a.native_manifest_sha256, 'Native manifest differs from expected SHA256')
        ppo.NATIVE_EXECUTION = dict(manifest=str(a.native_manifest.resolve()), sha256=a.native_manifest_sha256,
                                    prefix_threads=a.native_prefix_threads)
    items = rollouts(a.batch)
    require([len(item.samples) for item in items] == [row['samples'] for row in audit['games']], 'Rebuilt sample count differs')
    results = []
    for index in range(a.repeat):
        profile = cProfile.Profile() if a.profile and index == a.repeat-1 else None
        result = run_once(a, items, profile)
        if profile is not None:
            profile.dump_stats(str(a.profile))
            pstats.Stats(str(a.profile)).sort_stats('cumulative').print_stats(40)
        if a.reference_resume:
            raw = torch.load(a.reference_resume, map_location='cpu', weights_only=True)
            checks = {}
            for name, left, right in (('model', dict(result['model'].state_dict()), raw['model_state']),
                                      ('adam', result['optimizer'].state_dict(), raw['resume']['optimizer_state']),
                                      ('minibatch_generator', result['generator'].get_state(), raw['rng']['torch_generators']['minibatch'])):
                try: _exact(left, right); checks[name] = True
                except ValueError: checks[name] = False
            result['reference_equal'] = checks
        state = result.pop('diagnostic_state')
        if a.save_state and index == a.repeat-1: torch.save(state, a.save_state)
        if a.compare_state:
            reference = torch.load(a.compare_state, weights_only=True)
            result['difference'] = difference(reference, state)
            result['full_state_exact'] = tensor_digest(reference) == tensor_digest(state)
        for key in ('model', 'optimizer', 'generator'): result.pop(key)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('seconds', 'weights_sha256', 'model_sha256', 'adam_sha256', 'gradient_sha256')}
                         | {'reference_equal': result.get('reference_equal'), 'difference': result.get('difference'), 'timings': result['timings']}))
    for name, identity in input_bindings.items(): same_file_bytes(getattr(a, name), identity)
    same_file_bytes(__file__, driver)
    if native_binding is not None:
        same_file_bytes(a.native_manifest, native_binding)
    require(audit_capture(a) == audit, 'Capture evidence changed during diagnostic update')
    require(subprocess.check_output(['git', '-C', str(a.checkout), 'rev-parse', 'HEAD'], text=True).strip() == checkout_head
            and not subprocess.check_output(['git', '-C', str(a.checkout), 'status', '--porcelain'], text=True).strip(),
            'Selected implementation changed during diagnostic update')
    record = dict(checkout_head=checkout_head, driver=driver, audit=audit, parent=input_bindings,
                  diagnostic_only=True, official_samples_admitted=0, official_updates=0,
                  experiments=sorted(getattr(ppo, 'EXPERIMENTS', ())), prefix_helpers=getattr(ppo, 'PREFIX_HELPERS', 0),
                  text_leaf_replay=getattr(ppo, 'REPLAY_TEXT_LEAVES', None), text_replay_workers=getattr(ppo, 'TEXT_REPLAY_WORKERS', None),
                  native_execution=getattr(ppo, 'NATIVE_EXECUTION', None), native_manifest=native_binding,
                  samples=sum(len(i.samples) for i in items), frames=[len(i.history) for i in items],
                  decisions=[len(i.decisions) for i in items], torch=str(torch.__version__), repetitions=results,
                  identical_across_repetitions=len({r['model_sha256'] + r['adam_sha256'] + r['gradient_sha256'] for r in results}) == 1)
    if a.output: a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('samples', 'frames', 'decisions', 'identical_across_repetitions')}))
    return 1 if any(r.get('full_state_exact') is False for r in results) else 0


def main():
    if '--worker' in sys.argv:
        with official_exclusion(shared=True):
            return _main()
    return _main()


if __name__ == '__main__':
    raise SystemExit(main())
