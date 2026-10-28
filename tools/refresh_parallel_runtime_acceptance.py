#!/usr/bin/env python3
"""Build immutable source-refresh evidence through the existing strict native gates.

The fixed default paths describe the completed checkpoint-1095 validation.
HEROES_ARTIFACT_ROOT relocates the private artifact root; historical bindings
must still match their recorded absolute locations. No gameplay is launched.

Run ``python tools/refresh_parallel_runtime_acceptance.py candidate`` after the
44 current-source native controls, then run the existing frozen-policy probe.
Only its completed eight-game report permits the ``acceptance`` phase. Existing
candidate and acceptance files are never overwritten. Official machine files,
checkpoint pointers and historical acceptance descriptors remain untouched.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('HEROES_ARTIFACT_ROOT', str(Path.home()/'Projects/Heroes-artifacts'))).expanduser().resolve()
BASE = ARTIFACTS/'parallel-continuation-validation-20261028'
RUNTIME = ARTIFACTS/'runtime/local-v8-terminal-replay-repair1'
CANDIDATE = BASE/'parallel-runtime-candidate-v1.json'
POLICY = BASE/'policy1095-runtime-refresh1'
ACCEPTANCE = RUNTIME/'parallel-collection-acceptance-v1.json'
POLICY_ACCEPTANCE = RUNTIME/'parallel-collection-policy-acceptance-v1.json'
os.environ['HEROES_ARTIFACT_ROOT'] = str(ARTIFACTS)
sys.path.insert(0, str(ROOT))
from tools.prepare_local_training import candidate_runtime_origin, accepted_runtime_origin, terminal_replay_acceptance
from experiments.learner.builtin_training_resume import binding, require
from validation.probes.validate_garrison_dialogs import implementation_identity


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def proof(folder):
    return {name:binding(folder/path) for name,path in
            [('plan','execution/plan.json'),('report','execution/report.json'),('shutdown','shutdown.json')]}


def candidate():
    old = RUNTIME/'accepted-runtime.json'
    require(binding(old)['sha256'] == 'fffce8b1ee74c010f235b97a88547adc6ada3dce65251ac959a4b6dbe6e032b8',
            'Historical acceptance changed')
    value = deepcopy(read(old))
    value.pop('policy_reconstruction')
    value.update(schema='heroes-local-training-runtime-candidate-v1', status='candidate_native_validated',
        release_id='terminal-replay-parallel-v1', checkpoint_update=1095,
        native=proof(BASE/'native-refresh/garrison-16'),
        regression=proof(BASE/'native-refresh/regression-28'),
        prior_acceptance=binding(old), source_refresh_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        acceptance_refresh_scope='Current Python collection/admission source on the unchanged accepted native binaries and engine recipe. Original terminal-replay support classification and proof are retained; no new native defect or game-rule change.',
        timestamp_utc=datetime.now(timezone.utc).isoformat())
    runtime = {name:sha for name,sha in value['runtime_files'].items() if name != 'vcmiclient'}
    candidate_runtime_origin(Path(value['runtime_bin']), runtime, value)
    terminal_replay_acceptance(value, runtime)
    write_new(CANDIDATE,value)
    print(json.dumps({'candidate':binding(CANDIDATE),'native_controls':44,'native_gate_passed':True}))


def acceptance():
    value = read(CANDIDATE)
    runtime = {name:sha for name,sha in value['runtime_files'].items() if name != 'vcmiclient'}
    candidate_runtime_origin(Path(value['runtime_bin']),runtime,value)
    plan, report, shutdown = [read(POLICY/name) for name in ['plan.json','execution/report.json','shutdown.json']]
    require(plan['runtime_candidate'] == binding(CANDIDATE) and plan['mode'] == 'evaluation'
            and plan['parent_update'] == 1095 and plan['runtime'] == runtime
            and plan['source']['tracked_worktree_dirty'] is False, 'Policy plan is not the frozen candidate evaluation')
    require(report.get('status') == 'passed' and report.get('plan_sha256') == binding(POLICY/'plan.json')['sha256']
            and report.get('training_performed') is False and report.get('optimizer_steps') == 0
            and report.get('training_samples') == 0 and report.get('full_parent_unchanged') is True
            and report.get('full_parent_restore') is True and report.get('cleanup_verified') is True
            and report.get('parent_progress',{}).get('updates') == 1095,
            'Frozen policy report is incomplete')
    require(shutdown.get('returncode') == 0 and shutdown.get('owned_cleanup_verified') is True
            and shutdown.get('hard_deadline_exhausted') is False and shutdown.get('errors') == []
            and datetime.fromisoformat(shutdown['finished_utc']).timestamp() <= plan['absolute_deadline_unix'],
            'Frozen policy owned cleanup failed or exceeded its immutable deadline')
    attempts = deepcopy(report['attempts'])
    require(len(attempts) == 8 and len(plan['schedule']) == 8, 'Exactly eight evaluations required')
    for intended, actual in zip(plan['schedule'],attempts):
        require(all(actual.get(k) == v for k,v in intended.items())
                and actual.get('status') == 'passed' and actual.get('training_samples') == 0
                and actual.get('reconstruction',{}).get('passed') is True
                and actual.get('reconstruction',{}).get('maximum_absolute_difference') == 0
                and actual.get('guard_invariants_verified') is True
                and all(actual.get(k) is True for k in ['identity_verified','cleanup_verified','completion_receipt','replay_verified','weights_unchanged']),
                'Policy attempt lacks exact replay/reconstruction/guard proof')
        actual['optimizer_steps'] = report['optimizer_steps']
    require(sorted(r['learner_player'] for r in attempts) == [0]*4+[1]*4
            and len({r['engine_seed'] for r in attempts}) == len({r['policy_seed'] for r in attempts}) == 8,
            'Policy coverage or fresh seeds missing')
    decisions = sum(row['reconstruction']['actions_resampled'] for row in attempts)
    require(decisions == sum(row['sampled_choices'] for row in attempts) and decisions > 0,
            'Not every actual policy choice was independently reconstructed')
    outcomes = {'win':0,'loss':0,'cutoff':0}
    for row in attempts:
        outcome = row['outcomes'][('red','blue')[row['learner_player']]]
        outcomes[outcome if outcome in ('win','loss') else 'cutoff'] += 1
    policy = dict(schema='heroes-garrison-policy-reconstruction-v1', status='passed',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), checkpoint_update=1095,
        scope='Eight fresh frozen copied checkpoint1095 evaluations on the original map; no PPO input or update.',
        implementation=implementation_identity(), runtime=runtime,
        native_report_sha256=value['native']['report']['sha256'], checkpoint=plan['parent_descriptor'],
        evidence=[binding(POLICY/name) for name in ['plan.json','execution/report.json','shutdown.json']],
        decisions_checked=decisions, mismatches=0, parameters_unchanged=True,
        exact_model_adam_rng_progress_pool=True, sampler_state_exact=True,
        training_performed=False, optimizer_steps=0, attempts=attempts, outcomes=outcomes,
        policy_execution_source_commit=plan['source']['git_commit'])
    write_new(POLICY_ACCEPTANCE,policy)
    value.update(schema='heroes-local-training-runtime-acceptance-v1',status='accepted',
                 policy_reconstruction=binding(POLICY_ACCEPTANCE), policy_acceptance_games=8,
                 policy_acceptance_choices=decisions, checkpoint_update=1095)
    accepted_runtime_origin(Path(value['runtime_bin']), runtime, value)
    write_new(ACCEPTANCE,value)
    print(json.dumps({'accepted_descriptor':binding(ACCEPTANCE),'policy_acceptance':binding(POLICY_ACCEPTANCE),
                      'native_controls':44,'frozen_policy_games':8,'choices':decisions,'outcomes':outcomes,
                      'official_machine_modified':False},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['candidate','acceptance'])
    args = parser.parse_args()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
            'Current source must be clean and committed')
    (candidate if args.phase == 'candidate' else acceptance)()
