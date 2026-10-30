"""Next-launch acceptance preserves actual diagnostic labels and all exclusion gates."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import accept_cpu_next_launch as accept


class PipelineAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root/'diagnostic'; self.output.mkdir()
        self.runtime = {'vcmireplay': 'a'*64}
        self.configuration = {'unchanged': 'complete learning settings'}
        self.addCleanup(patch.stopall)
        patch.object(accept.copies, 'ARTIFACTS', self.root).start()
        self.source_gate = patch.object(accept, 'verify_execution_source').start()
        self.restore = patch.object(accept, 'restore_fingerprints', return_value={
            'weights_sha256': 'b'*64, 'fingerprints': {'model': 'c'*64}}).start()
        self.capture = patch.object(accept, 'audit_attempt', return_value={'receipt': {}, 'choices': 3}).start()
        copied, original = {}, {}
        values = dict(plan={'parent': True}, report={'status': 'finished'}, policy={'tensor': 1},
                      resume={'full': 'state'}, initial_checkpoint={'initial': True}, map={'original': True},
                      **{'completed-plan': {'configuration': self.configuration}, 'completed-report': {'verified': True}})
        for name, value in values.items():
            original[name] = self.write(self.root/'original'/(name+'.json'), value)
            copied[name] = self.write(self.output/'input'/(name+'.json'), value)
        patch.object(accept, 'ORIGINAL_MAP_SHA256', copied['map']['sha256']).start()
        parent = {name: copied[name] for name in ('plan', 'report', 'policy', 'resume', 'initial_checkpoint')}
        rows = [dict(episode=i+1, phase='training', learner_player=i//4,
                     engine_seed=1_730_000_000+i, policy_seed=1_830_000_000+i) for i in range(8)]
        self.launch = dict(schema=accept.SCHEMA+'-launch', diagnostic_only=True, optimization_eligible=False,
            output=str(self.output), inputs=copied, original_inputs=original, parent_descriptor=parent,
            runtime=self.runtime, policy_support='visible-interaction-budget-v1',
            hard_deadline_unix=time.time()+120, maximum_seconds=120,
            optimizer_updates=0, training_samples_admitted=0, prepared_engine_source_check={'returncode': 0},
            binary=str(self.root/'runtime/vcmiclient'), replay_binary=self.write(self.root/'runtime/vcmireplay', {'runtime': True}),
            data=str(self.root/'data'), configuration=self.configuration,
            reservation={'schedule': rows, 'retries_permitted': False})
        launch_binding = self.write(self.output/'launch.json', self.launch)
        attempts = [{**row, 'status': 'passed'} for row in rows]
        candidate = dict(name='pipelined-verification', pipeline_verification=True, status='passed',
                         frozen_model_unchanged=True, parent_rng_unchanged=True, attempts=attempts, batches=[])
        self.report = dict(schema=accept.SCHEMA, status='passed', launch=launch_binding,
            plan_sha256=launch_binding['sha256'], parent_descriptor=parent,
            runtime=self.runtime, policy_support=self.launch['policy_support'],
            full_parent_restore=True, full_parent_unchanged=True, sampler_state_exact=True,
            original_inputs_unchanged=True, cleanup_verified=True, within_hard_deadline=True,
            training_performed=False, optimization_eligible=False,
            optimizer_updates=0, optimizer_steps=0, training_samples_admitted=0,
            hard_deadline_unix=self.launch['hard_deadline_unix'], finished_utc=datetime.now(timezone.utc).isoformat(),
            source={'git_commit': 'd'*40}, implementation={'exact': 'implementation'}, native={'exact': 'native'},
            parent_progress={'updates': 1392}, arms=[dict(name='sequential-verification', status='passed',
                frozen_model_unchanged=True, parent_rng_unchanged=True), candidate])
        for seat in (0, 1):
            folder = self.output/'pipelined-verification'/f'seat-{seat}'
            plan = dict(schema='heroes-builtin-training-v1-plan', mode='training', diagnostic_scope=accept.SCHEMA,
                optimization_eligible=False, diagnostic_output=str(self.output),
                diagnostic_launch_sha256=launch_binding['sha256'], schedule=rows[4*seat:4*seat+4],
                execution=dict(collection_workers=4, verification_workers=4, pipeline_verification=True),
                source=self.report['source'], implementation=self.report['implementation'],
                configuration=self.configuration, runtime=self.runtime, checkpoint=parent['policy'],
                policy_support=self.launch['policy_support'], native=self.report['native'],
                binary=self.launch['binary'], replay_binary=self.launch['replay_binary'], data=self.launch['data'],
                game_map=copied['map']['path'], execution_output=str(folder/'execution'))
            result = dict(schema='heroes-parallel-collection-v1', status='passed', workers=4,
                verification_workers=4, batch_size=4, pipeline_verification=True,
                optimizer_updates=0, training_samples_admitted=0, attempts=attempts[4*seat:4*seat+4])
            candidate['batches'].append(dict(seat=seat, status='passed', plan=self.write(folder/'plan.json', plan),
                report=self.write(folder/'execution/parallel-collection/result.json', result)))
        shutdown = dict(returncode=0, owned_cleanup_verified=True, within_hard_deadline=True,
                        optimizer_updates=0, training_samples_admitted=0, finished_utc=self.report['finished_utc'])
        self.descriptor = dict(release_id=accept.RELEASE, checkpoint_update=1392, python_environment=accept.NEXT_LAUNCH_ENVIRONMENT,
            runtime_bin=str(self.root/'runtime'), pipeline_policy=dict(launch=launch_binding,
                report=self.write(self.output/'report.json', self.report),
                shutdown=self.write(self.output/'shutdown.json', shutdown)))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return accept.binding(path)

    def run_gate(self):
        return accept.pipeline_policy_acceptance(self.descriptor, self.runtime)

    def change_document(self, name, mutate):
        identity = self.descriptor['pipeline_policy'][name]
        value = json.loads(Path(identity['path']).read_text()); mutate(value)
        self.descriptor['pipeline_policy'][name] = self.write(Path(identity['path']), value)

    def test_actual_training_phase_is_retained_without_admission(self):
        value = self.run_gate()
        self.assertEqual((value['candidate_games'], value['choices'], value['phase']), (8, 24, 'training'))
        self.assertTrue(value['receipts_revoked'])
        self.assertEqual((value['training_samples_admitted'], value['optimizer_updates']), (0, 0))
        self.assertEqual(self.capture.call_count, 8)
        self.restore.assert_called_once_with(self.launch['parent_descriptor'], self.report['parent_progress'])

    def test_missing_full_state_exclusion_or_cleanup_flags_fail(self):
        for key in ('full_parent_restore', 'full_parent_unchanged', 'sampler_state_exact',
                    'original_inputs_unchanged', 'cleanup_verified', 'within_hard_deadline'):
            self.change_document('report', lambda row, key=key: row.update({key: False}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'restoration'):
                self.run_gate()
            self.change_document('report', lambda row, key=key: row.update({key: True}))
        for key in ('optimizer_updates', 'optimizer_steps', 'training_samples_admitted'):
            self.change_document('report', lambda row, key=key: row.update({key: 1}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'restoration'):
                self.run_gate()
            self.change_document('report', lambda row, key=key: row.update({key: 0}))

    def test_failed_shutdown_and_false_deadline_claim_fail(self):
        for field, wrong in (('returncode', 1), ('owned_cleanup_verified', False),
                             ('training_samples_admitted', 1), ('finished_utc', '2099-01-01T00:00:00+00:00')):
            original = json.loads(Path(self.descriptor['pipeline_policy']['shutdown']['path']).read_text())
            self.change_document('shutdown', lambda row: row.update({field: wrong}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'deadline cleanup'):
                self.run_gate()
            self.descriptor['pipeline_policy']['shutdown'] = self.write(self.output/'shutdown.json', original)

    def test_relabeling_candidate_plan_or_summary_is_rejected(self):
        for mode in ('plan', 'summary'):
            saved = deepcopy(self.report)
            changed_plan = None
            if mode == 'summary':
                self.report['arms'][1]['attempts'][0]['phase'] = 'evaluation'
            else:
                batch = self.report['arms'][1]['batches'][0]
                changed_plan = Path(batch['plan']['path'])
                original_plan = json.loads(changed_plan.read_text())
                plan = deepcopy(original_plan); plan['mode'] = 'evaluation'
                batch['plan'] = self.write(Path(batch['plan']['path']), plan)
            self.descriptor['pipeline_policy']['report'] = self.write(self.output/'report.json', self.report)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_gate()
            self.report = saved
            if changed_plan is not None:
                self.write(changed_plan, original_plan)

    def test_runtime_settings_input_and_raw_binding_tampering_fail(self):
        self.descriptor['release_id'] = 'historical'
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            self.run_gate()
        self.descriptor['release_id'] = accept.RELEASE
        with self.assertRaisesRegex(ValueError, 'identity'):
            accept.pipeline_policy_acceptance(self.descriptor, {'different': 'runtime'})
        path = Path(self.launch['inputs']['resume']['path']); path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            self.run_gate()

    def test_receipt_audit_or_restore_failure_cannot_be_accepted(self):
        self.capture.side_effect = ValueError('changed receipt bytes')
        with self.assertRaisesRegex(ValueError, 'changed receipt'):
            self.run_gate()
        self.capture.side_effect = None
        self.restore.side_effect = ValueError('full Adam state differs')
        with self.assertRaisesRegex(ValueError, 'Adam state'):
            self.run_gate()


class ExecutionClosureTests(unittest.TestCase):
    def test_unconsumed_choices_are_reconstructed_but_not_counted_as_targets(self):
        from experiments.learner import builtin_gameplay as gameplay
        episode = SimpleNamespace(decisions=[1, 2, 3], estimates=[1, 2, 3])
        attempt = dict(sampled_choices=3, guarded_decisions=3)
        policy = dict(actions_resampled=3, values_recomputed=3)
        receipt = dict(samples=2, files={'training-verification/builtin-replay-verification.json': {'sha256': 'a'*64}})
        with patch.object(gameplay, '_samples', return_value=(1, 2)) as samples:
            accept.verify_choice_counts(episode, attempt, policy, receipt)
            samples.assert_called_once_with(episode, 'training-verification/builtin-replay-verification.json#sha256='+'a'*64)
            with self.assertRaisesRegex(ValueError, 'consumed targets'):
                accept.verify_choice_counts(episode, attempt, policy, dict(receipt, samples=3))
            with self.assertRaisesRegex(ValueError, 'reconstructed choices'):
                accept.verify_choice_counts(episode, attempt, dict(policy, actions_resampled=2), receipt)

    def test_only_three_nonexecuted_pin_assignments_are_excluded(self):
        source = 'FAST_PATH_SOURCES = {"x": "old"}\nFAST_PATH_TREE = {}\nLOCAL_ADAPTER_TRANSITION_AST = "old"\ndef execute(x):\n    return x+1\n'
        changed = source.replace('"old"', '"new"')
        self.assertEqual(accept.adapter_execution_ast(source), accept.adapter_execution_ast(changed))
        self.assertNotEqual(accept.adapter_execution_ast(source), accept.adapter_execution_ast(source.replace('x+1', 'x+2')))
        self.assertNotEqual(accept.adapter_execution_ast(source), accept.adapter_execution_ast(source+'REVIEWED_TREE = "changed"\n'))
        with self.assertRaisesRegex(ValueError, 'literal metadata'):
            accept.adapter_execution_ast(source.replace('FAST_PATH_TREE = {}', 'FAST_PATH_TREE = execute(1)'))

    def test_recorded_source_is_verified_against_exact_execution_commit(self):
        raw = b'example source\n'; digest = hashlib.sha256(raw).hexdigest()
        response = type('Result', (), {'stdout': b'0'*40+b' blob '+str(len(raw)).encode()+b'\n'+raw+b'\n'})()
        with patch.object(accept.subprocess, 'run', return_value=response):
            self.assertEqual(accept.committed_blobs('a'*40, {'file.py': digest}), {'file.py': raw})
            with self.assertRaisesRegex(ValueError, 'differs from its commit'):
                accept.committed_blobs('a'*40, {'file.py': 'b'*64})
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                accept.committed_blobs('a'*40, {'../file.py': digest})


class RevealedAcceptanceTests(unittest.TestCase):
    write = PipelineAcceptanceTests.write
    run_gate = PipelineAcceptanceTests.run_gate
    change_document = PipelineAcceptanceTests.change_document

    def setUp(self):
        from experiments.learner import interaction_guard
        PipelineAcceptanceTests.setUp(self)
        self.parent_gate = patch.object(accept, 'verify_revealed_parent').start()
        self.capture.return_value = {'receipt': {}, 'choices': 3,
                                    'revealed_movement': dict(movement_choices=2, revealed_targets=2, unrevealed_targets=0)}
        details = dict(study_release=accept.REVEALED_RELEASE, checkpoint_selection='completed_after',
                       policy_checkpoint_update=1585, policy_support=interaction_guard.REVEALED_VERSION)
        self.launch.update(details, schema=accept.REVEALED_SCHEMA+'-launch')
        launch_binding = self.write(self.output/'launch.json', self.launch)
        template = deepcopy(self.report['arms'][1])
        self.report.update(details, schema=accept.REVEALED_SCHEMA, launch=launch_binding,
            plan_sha256=launch_binding['sha256'], parent_progress={'updates': 1585},
            revealed_movement=dict(movement_choices=32, revealed_targets=32, unrevealed_targets=0), arms=[])
        for name, pipelined in (('sequential-verification', False), ('pipelined-verification', True)):
            arm = deepcopy(template); arm.update(name=name, pipeline_verification=pipelined, batches=[])
            for seat, original_batch in enumerate(template['batches']):
                plan = json.loads(Path(original_batch['plan']['path']).read_text())
                result = json.loads(Path(original_batch['report']['path']).read_text())
                folder = self.output/name/f'seat-{seat}'
                plan.update(diagnostic_scope=accept.REVEALED_SCHEMA,
                    diagnostic_launch_sha256=launch_binding['sha256'], policy_support=details['policy_support'],
                    execution_output=str(folder/'execution'),
                    execution=dict(collection_workers=4, verification_workers=4, pipeline_verification=pipelined))
                result['pipeline_verification'] = pipelined
                arm['batches'].append(dict(seat=seat, status='passed', plan=self.write(folder/'plan.json', plan),
                    report=self.write(folder/'execution/parallel-collection/result.json', result)))
            self.report['arms'].append(arm)
        self.descriptor.update(release_id=accept.REVEALED_RELEASE, policy_checkpoint_update=1585,
                               policy_support=details['policy_support'])
        self.descriptor['pipeline_policy'].update(launch=launch_binding,
            report=self.write(self.output/'report.json', self.report))

    def test_new_gate_checks_both_arms_and_keeps_native_and_policy_checkpoints_distinct(self):
        value = self.run_gate()
        self.assertEqual(self.descriptor['checkpoint_update'], 1392)
        self.assertEqual((value['policy_checkpoint_update'], value['candidate_games'], value['diagnostic_games']),
                         (1585, 8, 16))
        self.assertEqual((value['choices'], self.capture.call_count), (24, 16))
        self.assertEqual(value['revealed_movement']['unrevealed_targets'], 0)
        self.parent_gate.assert_called_once()

    def test_before_checkpoint_or_old_guard_cannot_satisfy_new_release(self):
        for field, wrong in (('policy_checkpoint_update', 1584), ('checkpoint_update', 1585),
                             ('policy_support', 'visible-interaction-budget-v1')):
            original = self.descriptor[field]; self.descriptor[field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'engine proof 1392 and policy proof 1585'):
                self.run_gate()
            self.descriptor[field] = original
        self.change_document('report', lambda row: row.update(parent_progress={'updates': 1584}))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_gate()

    def test_all_actual_targets_must_be_revealed_and_counts_must_match(self):
        for wrong in (dict(movement_choices=32, revealed_targets=31, unrevealed_targets=1),
                      dict(movement_choices=31, revealed_targets=31, unrevealed_targets=0)):
            self.change_document('report', lambda row, wrong=wrong: row.update(revealed_movement=wrong))
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, 'zero-unrevealed-target'):
                self.run_gate()
        empty = dict(movement_choices=0, revealed_targets=0, unrevealed_targets=0)
        self.capture.return_value['revealed_movement'] = empty
        self.change_document('report', lambda row: row.update(revealed_movement=empty))
        with self.assertRaisesRegex(ValueError, 'lacks actual movement'):
            self.run_gate()

    def test_sequential_arm_cannot_be_accepted_only_from_its_summary(self):
        batch = self.report['arms'][0]['batches'][0]
        result = json.loads(Path(batch['report']['path']).read_text()); result['status'] = 'failed'
        batch['report'] = self.write(Path(batch['report']['path']), result)
        self.descriptor['pipeline_policy']['report'] = self.write(self.output/'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'collection result'):
            self.run_gate()


class RevealedEvidenceTests(unittest.TestCase):
    def test_actual_sampled_target_check_does_not_trust_summary(self):
        from tools import benchmark_pipeline_collection as diagnostic
        observation = SimpleNamespace(heroes=[SimpleNamespace(handle='own', position=(0, 0, 0))],
                                      tiles=[SimpleNamespace(position=(1, 2, 0))])
        record = SimpleNamespace(decision_seq=1, action=SimpleNamespace(kind='move_to', subject='own', x=1, y=2))
        episode = SimpleNamespace(history=[SimpleNamespace(seq=1, kind='decision', observation=observation)],
                                  decisions=[record])
        actual = diagnostic.revealed_movement_audit(episode)
        self.assertEqual(accept.verify_revealed_choices(episode, {'revealed_movement': actual}), actual)
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            accept.verify_revealed_choices(episode, {'revealed_movement': dict(actual, movement_choices=2)})
        record.action.x = 9
        with self.assertRaisesRegex(ValueError, 'unrevealed movement target'):
            accept.verify_revealed_choices(episode, {'revealed_movement': actual})

    def test_parent_selection_binds_after_progress_and_all_five_original_files(self):
        from tools import benchmark_pipeline_collection as diagnostic
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def document(name, value):
                path = root/name; path.write_text(json.dumps(value)); return accept.binding(path)
            selected = {name: document(name+'.json', {'name': name}) for name in
                        ('plan', 'report', 'policy', 'resume', 'initial_checkpoint')}
            completed = {'configuration': 'unchanged'}
            plan = document('completed-plan.json', completed)
            report = document('completed-report.json', {'cumulative_progress': {'updates': 1585}})
            launch = dict(study_release=accept.REVEALED_RELEASE, checkpoint_selection='completed_after',
                policy_checkpoint_update=1585, inputs={'completed-report': report},
                original_inputs={**selected, 'completed-plan': plan})
            result = dict(study_release=accept.REVEALED_RELEASE, checkpoint_selection='completed_after',
                          policy_checkpoint_update=1585, parent_progress={'updates': 1585})
            with patch.object(diagnostic, 'selected_parent', return_value=selected) as select:
                accept.verify_revealed_parent(launch, result, completed)
                select.assert_called_once()
                launch['original_inputs']['initial_checkpoint'] = selected['policy']
                with self.assertRaisesRegex(ValueError, 'another parent or initial dependency'):
                    accept.verify_revealed_parent(launch, result, completed)
                launch['original_inputs']['initial_checkpoint'] = selected['initial_checkpoint']
                result['parent_progress'] = {'updates': 1584}
                with self.assertRaisesRegex(ValueError, 'another parent or initial dependency'):
                    accept.verify_revealed_parent(launch, result, completed)
                launch['checkpoint_selection'] = 'completed_before'
                with self.assertRaisesRegex(ValueError, 'completed after-state'):
                    accept.verify_revealed_parent(launch, result, completed)


if __name__ == '__main__':
    unittest.main()
