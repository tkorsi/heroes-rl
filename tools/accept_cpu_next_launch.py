#!/usr/bin/env python3
"""Accept excluded training-style pipeline captures without relabeling or admission.

This read-only gate retains the original capture phase and revoked receipt
evidence. Its builder writes one new descriptor exclusively; official state,
historical descriptors and source acceptance pins are never modified.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_copied_training as copies
from tools.training_environment import NEXT_LAUNCH_ENVIRONMENT, verify_environment

SCHEMA = 'heroes-pipeline-collection-diagnostic-v1'
RELEASE = 'cpu-next-launch-v1'
OLD_RELEASE = 'terminal-replay-1392-v1'
ORIGINAL_MAP_SHA256 = 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3'
HELPERS = ('tools/benchmark_pipeline_collection.py', 'tools/benchmark_copied_training.py',
           'tools/benchmark_parallel_collection.py', 'tools/isolated_diagnostic.py',
           'tools/acceleration3/campaign_summary.py')
PIN_ASSIGNMENTS = {'FAST_PATH_SOURCES', 'FAST_PATH_TREE', 'LOCAL_ADAPTER_TRANSITION_AST'}
require, binding = copies.require, copies.binding


def checked(identity):
    require(type(identity) is dict and set(identity) == {'path', 'sha256', 'bytes'}
            and binding(identity['path']) == identity, 'Acceptance evidence binding changed')
    return Path(identity['path'])


def read(identity):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(checked(identity).read_bytes())


def adapter_execution_ast(contents):
    """Exclude only recursive acceptance pins; every executable AST node remains."""
    tree = ast.parse(contents)
    retained = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in PIN_ASSIGNMENTS):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                raise ValueError('Source acceptance pins must remain literal metadata') from None
        else:
            retained.append(node)
    tree.body = retained
    return ast.dump(tree, include_attributes=False)


def committed_blobs(commit, identities):
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'Invalid execution commit')
    names = sorted(identities)
    require(all(type(name) is str and name and '\n' not in name and ':' not in name
                and not Path(name).is_absolute() and '..' not in Path(name).parts for name in names),
            'Unsafe implementation source path')
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT, input=''.join(
        commit+':'+name+'\n' for name in names).encode(), capture_output=True, check=True, timeout=30)
    offset, blobs = 0, {}
    for name in names:
        end = result.stdout.index(b'\n', offset)
        header = result.stdout[offset:end].split()
        require(len(header) == 3 and header[1] == b'blob', 'Execution source is not committed: '+name)
        size = int(header[2]); start = end+1
        contents = result.stdout[start:start+size]
        require(hashlib.sha256(contents).hexdigest() == identities[name],
                'Recorded execution source differs from its commit: '+name)
        blobs[name] = contents; offset = start+size+1
    require(offset == len(result.stdout), 'Unexpected committed source inventory')
    return blobs


def verify_execution_source(report, launch):
    from experiments.learner import builtin_first_run as pilot, local_script_training as local
    from experiments.learner.inference import source_identity
    source = report['source']
    current = source_identity(local.INTERFACE)
    require(source.get('tracked_worktree_dirty') is False
            and source.get('git_commit') == launch['checkout'].get('commit')
            and all(source.get(key) == current[key] for key in ('torch', 'python', 'interface'))
            and set(source['source_sha256']) == set(current['source_sha256']),
            'Pipeline execution source/runtime differs')
    expected = pilot.implementation()
    expected.update({name: binding(ROOT/name)['sha256'] for name in HELPERS})
    recorded = report['implementation']
    # Tests were recorded but never executed by the gameplay worker. Their
    # historical bytes still must exist exactly in the recorded execution commit.
    active = lambda values: {name for name in values if not name.startswith('tests/')}
    require(active(recorded) == active(expected), 'Pipeline execution dependency inventory differs')
    identities = dict(recorded)
    for name, digest in source['source_sha256'].items():
        require(name not in identities or identities[name] == digest, 'Source identity declarations disagree')
        identities[name] = digest
    blobs = committed_blobs(source['git_commit'], identities)
    for name, digest in identities.items():
        if name.startswith('tests/'):
            continue
        path = copies.no_links(ROOT/name, regular=True)
        actual = path.read_bytes()
        if name == 'experiments/learner/local_script_training.py':
            require(adapter_execution_ast(blobs[name]) == adapter_execution_ast(actual),
                    'Pipeline local adapter execution changed')
        else:
            require(hashlib.sha256(actual).hexdigest() == digest, 'Pipeline execution code changed: '+name)
    harness = launch['harness']
    checked(harness)
    require(harness['sha256'] == recorded[HELPERS[0]], 'Pipeline driver differs from recorded execution')


def restore_fingerprints(descriptor, progress):
    import torch
    from experiments.learner import local_script_training as local
    from tools.validate_interaction_guard import state_snapshot
    from tools.acceleration3.ppo_bench import tensor_digest
    before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
    try:
        state = local.restore_parent(descriptor)
        require(vars(state.progress) == progress, 'Pipeline parent progress changed')
        snapshot = state_snapshot(state)
        require(set(snapshot) == {'model', 'optimizer', 'generator', 'pool', 'torch_rng',
                                  'python_rng', 'sampler', 'progress'}, 'Incomplete restored state inventory')
        return {'weights_sha256': state.weights_sha256,
                'fingerprints': {name: tensor_digest(value) for name, value in snapshot.items()}}
    finally:
        torch.set_rng_state(before_torch); random.setstate(before_python)


def verify_choice_counts(episode, attempt, policy, receipt):
    from experiments.learner.builtin_gameplay import _samples
    reference = 'training-verification/builtin-replay-verification.json#sha256=' + receipt['files'][
        'training-verification/builtin-replay-verification.json']['sha256']
    # A held or superseded choice is reconstructed but is not a consumed PPO
    # target. Retain both denominators and independently derive their counts.
    require(policy['actions_resampled'] == attempt['sampled_choices']
                == attempt['guarded_decisions'] == len(episode.decisions)
            and policy['values_recomputed'] == len(episode.estimates)
            and receipt['samples'] == len(_samples(episode, reference)),
            'Candidate reconstructed choices or consumed targets differ')


def audit_attempt(attempt, row, index, batch_plan, plan_binding, launch, weights):
    from experiments.headless import builtin_evidence, builtin_replay
    from experiments.learner import builtin_gameplay as gameplay, interaction_guard
    from experiments.learner.builtin_rollout import load_builtin_episode
    from experiments.learner.contextual_rollout import behavior_guard_version
    require(all(attempt.get(key) == value for key, value in row.items())
            and row['phase'] == 'training' and attempt.get('status') == 'passed'
            and all(attempt.get(key) is True for key in ('cleanup_verified', 'identity_verified',
                'completion_receipt', 'replay_verified', 'weights_unchanged', 'guard_invariants_verified',
                'fresh_receipt_discarded'))
            and attempt.get('optimization_eligible') is False
            and attempt.get('training_samples') == attempt.get('training_samples_admitted') == 0,
            'Candidate capture lacks excluded exact verification')
    folder = copies.no_links(attempt['episode_directory'])
    require(folder.parent == Path(batch_plan['execution_output']), 'Capture escaped its diagnostic batch')
    receipt_binding = binding(folder/'training-receipt.json')
    receipt = read(receipt_binding); bound = receipt['binding']
    require(receipt.get('schema') == gameplay.RECEIPT_SCHEMA and receipt.get('admission_from_file') is False
            and receipt.get('opponent_history') is None and receipt.get('opponent_policy') is None
            and receipt.get('rollout_schema') == gameplay.ROLLOUT_SCHEMA
            and receipt.get('replay_binary') == str(checked(launch['replay_binary']))
            and bound['plan_path'] == plan_binding['path'] and bound['plan_sha256'] == plan_binding['sha256']
            and bound['schedule_index'] == index and bound['folder'] == str(folder)
            and json.loads(bound['row_json']) == row and json.loads(bound['source_json']) == batch_plan['source']
            and json.loads(bound['native_identity_json']) == batch_plan['native']
            and json.loads(bound['checkpoint_file_json']) == launch['parent_descriptor']['policy']
            and bound['weights_sha256'] == attempt['weights_sha256'] == weights,
            'Candidate persisted receipt belongs to another source, checkpoint or schedule')
    manifest = read(binding(folder/'manifest.json'))
    builtin_replay.verify_identity(manifest, checked(launch['replay_binary']))
    expected = set(gameplay._capture_file_bindings(folder, manifest)) | {
        'training-history-reconstruction.json', 'training-policy-reconstruction.json',
        'training-verification/builtin-replay-verification.json', 'training-verification/native-report.json'}
    require(set(receipt['files']) == expected, 'Candidate receipt omitted capture evidence')
    for name, identity in receipt['files'].items():
        actual = binding(copies.relative_file(folder, name))
        require(identity == {key: actual[key] for key in ('sha256', 'bytes')}, 'Candidate capture bytes changed')
    history = read(binding(folder/'training-history-reconstruction.json'))
    policy = read(binding(folder/'training-policy-reconstruction.json'))
    replay = read(binding(folder/'training-verification/builtin-replay-verification.json'))
    native = read(binding(folder/'training-verification/native-report.json'))
    raw = read(binding(folder/'builtin-training-rollout.json'))
    require(raw.get('schema') == gameplay.CAPTURE_SCHEMA and raw.get('purpose') == 'training'
            and raw.get('optimization_eligible') is False and raw.get('binding') == bound,
            'Candidate capture was relabeled or changed')
    episode = load_builtin_episode(raw['experience'])
    require(episode.manifest == manifest and episode.weights_sha256 == weights
            and behavior_guard_version(episode.decisions) == interaction_guard.VERSION
            and episode.player == row['learner_player'], 'Candidate visible-policy support changed')
    require(history == receipt['history_reconstruction'] and history.get('verified') is True
            and history.get('opponent_history') is None and policy == receipt['policy_reconstruction']
            and policy.get('maximum_absolute_difference') == 0
            and policy.get('private_memory_reset') is True and policy.get('weights_unchanged') is True
            and attempt['reconstruction'] == dict(policy, passed=True)
            and replay.get('verified') is True and replay.get('training_samples') == 0
            and replay.get('training_performed') is False and 'diagnostic_verified' not in replay
            and replay.get('identity_check', {}).get('mode') == 'exact_capture'
            and replay.get('history') == history and native == {key: value for key, value in replay['native'].items()
                if key not in ('returncode', 'seconds')}, 'Candidate reconstruction or replay is incomplete')
    verify_choice_counts(episode, attempt, policy, receipt)
    final = builtin_evidence.read_finalization(folder, manifest)
    builtin_replay.validate_native_report(replay['native'], replay['wire'], final, cutoff=manifest['status'] == 'truncated')
    require(binding(folder/'training-receipt.json') == receipt_binding, 'Receipt changed during acceptance')
    return {'receipt': receipt_binding, 'choices': len(episode.decisions)}


def pipeline_policy_acceptance(descriptor, runtime):
    from experiments.learner import interaction_guard, parallel_collection
    require(descriptor.get('release_id') == RELEASE and descriptor.get('python_environment') == NEXT_LAUNCH_ENVIRONMENT,
            'Pipeline policy evidence is exclusive to the reviewed next-launch release')
    evidence = descriptor.get('pipeline_policy')
    require(type(evidence) is dict and set(evidence) == {'launch', 'report', 'shutdown'},
            'Pipeline acceptance requires actual launch, report and shutdown bindings')
    launch, report, shutdown = [read(evidence[name]) for name in ('launch', 'report', 'shutdown')]
    output = copies.private_path(launch['output'])
    require(all(Path(evidence[name]['path']) == output/(name+'.json') for name in evidence),
            'Pipeline evidence must be the original diagnostic documents')
    require(launch.get('schema') == SCHEMA+'-launch' and report.get('schema') == SCHEMA
            and launch.get('diagnostic_only') is True and launch.get('optimization_eligible') is False
            and report.get('status') == 'passed' and report.get('launch') == evidence['launch']
            and report.get('plan_sha256') == evidence['launch']['sha256']
            and report.get('parent_descriptor') == launch.get('parent_descriptor')
            and report.get('parent_progress', {}).get('updates') == descriptor.get('checkpoint_update') == 1392
            and report.get('runtime') == launch.get('runtime') == runtime
            and report.get('policy_support') == launch.get('policy_support') == interaction_guard.VERSION,
            'Pipeline diagnostic identity is incomplete')
    require(all(report.get(key) is True for key in ('full_parent_restore', 'full_parent_unchanged',
                'sampler_state_exact', 'original_inputs_unchanged', 'cleanup_verified', 'within_hard_deadline'))
            and report.get('training_performed') is False and report.get('optimization_eligible') is False
            and all(item.get(key) == 0 for item, keys in ((launch, ('optimizer_updates', 'training_samples_admitted')),
                (report, ('optimizer_updates', 'optimizer_steps', 'training_samples_admitted')),
                (shutdown, ('optimizer_updates', 'training_samples_admitted'))) for key in keys)
            and shutdown.get('returncode') == 0 and shutdown.get('owned_cleanup_verified') is True
            and shutdown.get('within_hard_deadline') is True
            and report.get('hard_deadline_unix') == launch['hard_deadline_unix']
            and 120 <= launch['maximum_seconds'] <= 1800
            and all(datetime.fromisoformat(item['finished_utc']).timestamp() <= launch['hard_deadline_unix']
                    for item in (report, shutdown)), 'Pipeline restoration, exclusion or deadline cleanup failed')
    require(launch['prepared_engine_source_check'].get('returncode') == 0
            and Path(launch['binary']) == Path(descriptor['runtime_bin'])/'vcmiclient'
            and checked(launch['replay_binary']) == Path(descriptor['runtime_bin'])/'vcmireplay',
            'Pipeline used another engine runtime')
    for item in launch['inputs'].values(): checked(item)
    for name, item in launch['original_inputs'].items():
        actual = binding(item['path']); copied = launch['inputs'][name]
        require(actual == item and all(item[key] == copied[key] for key in ('sha256', 'bytes')),
                'Pipeline input copy or immutable source changed')
        require(Path(copied['path']).is_relative_to(output/'input'), 'Learned input was not privately copied')
    require(set(launch['parent_descriptor']) == {'plan', 'report', 'policy', 'resume', 'initial_checkpoint'}
            and all(launch['inputs'][name] == value for name, value in launch['parent_descriptor'].items()),
            'Pipeline omitted complete copied parent state')
    completed = read(launch['inputs']['completed-plan'])
    require(launch['configuration'] == completed['configuration']
            and launch['inputs']['map']['sha256'] == ORIGINAL_MAP_SHA256,
            'Pipeline changed learning settings or the original map')
    verify_execution_source(report, launch)
    restored = restore_fingerprints(launch['parent_descriptor'], report['parent_progress'])
    rows = launch['reservation']['schedule']
    require(len(rows) == 8 and [row['learner_player'] for row in rows] == [0]*4+[1]*4
            and [row['episode'] for row in rows] == list(range(1, 9))
            and all(row['phase'] == 'training' for row in rows)
            and len({row['engine_seed'] for row in rows}) == len({row['policy_seed'] for row in rows}) == 8
            and all(1_730_000_000 <= row['engine_seed'] < 1_730_800_000
                    and 1_830_000_000 <= row['policy_seed'] < 1_830_800_000 for row in rows)
            and launch['reservation'].get('retries_permitted') is False,
            'Pipeline requires eight distinct fresh diagnostic training slots split across both seats')
    arms = report['arms']
    require(len(arms) == 2 and [arm['name'] for arm in arms] == ['sequential-verification', 'pipelined-verification']
            and all(arm.get('status') == 'passed' and arm.get('frozen_model_unchanged') is True
                    and arm.get('parent_rng_unchanged') is True for arm in arms), 'Pipeline paired study did not finish')
    arm = arms[1]
    require(arm.get('pipeline_verification') is True and len(arm['attempts']) == 8
            and len(arm['batches']) == 2, 'Candidate arm is incomplete')
    captures = []
    for seat, batch in enumerate(arm['batches']):
        plan, result = read(batch['plan']), read(batch['report'])
        require(batch.get('seat') == seat and batch.get('status') == 'passed'
                and plan.get('schema') == 'heroes-builtin-training-v1-plan' and plan.get('mode') == 'training'
                and plan.get('diagnostic_scope') == SCHEMA and plan.get('optimization_eligible') is False
                and plan.get('diagnostic_output') == str(output)
                and plan.get('diagnostic_launch_sha256') == evidence['launch']['sha256']
                and plan.get('schedule') == rows[4*seat:4*seat+4]
                and plan.get('execution') == dict(collection_workers=4, verification_workers=4, pipeline_verification=True)
                and plan.get('source') == report['source'] and plan.get('implementation') == report['implementation']
                and plan.get('configuration') == launch['configuration'] and plan.get('runtime') == runtime
                and plan.get('checkpoint') == launch['parent_descriptor']['policy']
                and plan.get('policy_support') == interaction_guard.VERSION
                and plan.get('native') == report['native']
                and all(plan.get(key) == launch[key] for key in ('binary', 'replay_binary', 'data'))
                and plan.get('game_map') == launch['inputs']['map']['path']
                and Path(plan['execution_output']) == output/'pipelined-verification'/f'seat-{seat}'/'execution'
                and Path(batch['plan']['path']) == Path(plan['execution_output']).parent/'plan.json'
                and Path(batch['report']['path']) == Path(plan['execution_output'])/'parallel-collection/result.json',
                'Candidate batch plan changed its actual diagnostic contract')
        require(result.get('schema') == parallel_collection.SCHEMA and result.get('status') == 'passed'
                and result.get('workers') == result.get('verification_workers') == result.get('batch_size') == 4
                and result.get('pipeline_verification') is True
                and result.get('optimizer_updates') == result.get('training_samples_admitted') == 0
                and len(result['attempts']) == 4, 'Candidate collection result is incomplete')
        for index, actual in enumerate(result['attempts']):
            attempt = arm['attempts'][4*seat+index]
            require(all(attempt.get(key) == value for key, value in actual.items()),
                    'Candidate summary changed an original collection attempt')
            captures.append(audit_attempt(attempt, rows[4*seat+index], index, plan,
                                          batch['plan'], launch, restored['weights_sha256']))
    choices = sum(item['choices'] for item in captures)
    require(choices > 0 and descriptor.get('policy_acceptance_games', 8) == 8
            and descriptor.get('policy_acceptance_choices', choices) == choices,
            'Pipeline reconstructed choice summary differs')
    return dict(evidence=evidence, candidate_games=8, choices=choices, captures=captures,
                restored_state=restored, phase='training', receipts_revoked=True,
                execution_commit=report['source']['git_commit'], training_samples_admitted=0, optimizer_updates=0)


def build(native_output, regression_output, pipeline_output, output):
    from tools import prepare_local_training as prep
    output = copies.private_path(output, fresh=True)
    old, old_binding = prep.accepted_runtime_descriptor(OLD_RELEASE)
    value = deepcopy(old)
    value.pop('policy_reconstruction', None)
    value.pop('policy_acceptance_games', None)
    value.pop('policy_acceptance_choices', None)
    proof = lambda folder: {name: binding(Path(folder)/path) for name, path in (
        ('plan', 'execution/plan.json'), ('report', 'execution/report.json'), ('shutdown', 'shutdown.json'))}
    value.update(release_id=RELEASE, native=proof(native_output), regression=proof(regression_output),
        prior_acceptance=old_binding['descriptor'], python_environment=NEXT_LAUNCH_ENVIRONMENT,
        pipeline_policy={name: binding(Path(pipeline_output)/(name+'.json')) for name in ('launch', 'report', 'shutdown')},
        acceptance_refresh_scope='Current CPU execution and isolated NumPy environment; eight fresh training-style diagnostic captures with revoked receipts, exact replay/reconstruction and no PPO admission.',
        timestamp_utc=datetime.now(timezone.utc).isoformat())
    verify_environment(NEXT_LAUNCH_ENVIRONMENT)
    runtime = {name: digest for name, digest in value['runtime_files'].items() if name != 'vcmiclient'}
    # Full native, terminal and pipeline gates run before any output is written.
    accepted = prep.accepted_runtime_origin(Path(value['runtime_bin']), runtime, value)
    value['policy_acceptance_games'] = accepted['pipeline_policy']['candidate_games']
    value['policy_acceptance_choices'] = accepted['pipeline_policy']['choices']
    value['source_refresh_commit'] = accepted['pipeline_policy']['execution_commit']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return dict(descriptor=binding(output), candidate_games=8, choices=value['policy_acceptance_choices'],
                official_state_modified=False, acceptance_pins_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-output', type=Path, required=True)
    parser.add_argument('--regression-output', type=Path, required=True)
    parser.add_argument('--pipeline-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    copies.checkout_identity(ROOT)
    print(json.dumps(build(args.native_output, args.regression_output, args.pipeline_output, args.output), indent=2))


if __name__ == '__main__':
    main()
