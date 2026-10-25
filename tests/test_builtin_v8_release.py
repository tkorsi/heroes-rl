"""The release must bind actual final mechanics, runtime and interface sources."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import builtin_v8_session as scope
from experiments.learner import builtin_v8_migration as migration
from validation.probes.summarize_capabilities import compact_report, summarize
from validation.probes.validate_capabilities import CASES


class NativeAcceptanceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.patch_root = patch.object(scope, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        for target, value in (
            ('experiments.learner.first_run.ROOT', self.root),
            ('experiments.learner.first_run.CONFIG', self.root/'configs/first-training-v1.json'),
            ('validation.probes.validate_capabilities.ROOT', self.root)):
            patched = patch(target, value)
            patched.start()
            self.addCleanup(patched.stop)
        for name in (
            'engine/ExplorerRL/Capabilities.cpp', 'engine/patches/0028-visions.patch',
            'dependencies/vcmi.json', 'tools/prepare_vcmi.py',
            'experiments/headless/player_history.py', 'experiments/headless/bridge_protocol.py',
            'experiments/explorer/create_capability_scenarios.py',
            'validation/probes/validate_capabilities.py', 'validation/probes/summarize_capabilities.py',
            'validation/probes/validate_contextual_native.py',
            'validation/probes/audit_contextual_histories.py',
            'experiments/learner/contextual.py', 'experiments/learner/contextual_model.py',
            'experiments/headless/contextual_protocol.py', 'README.md', 'tools/v8_teacher_campaign.py',
            'experiments/learner/builtin_v8_session.py', 'tests/test_controller.py'):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original fixture source\n')
        self.binary = self.root/'vcmiclient'
        self.binary.write_bytes(b'synthetic runtime')
        self.runtime = {'vcmireplay': 'replay-hash', 'AI/libExplorerRL.dylib': 'explorer-hash'}
        self.directory = self.root/'acceptance'
        self.directory.mkdir()
        schedule = [dict(case=case, fixture=fixture, player=player, seed=17, repeat=0)
                    for case, fixture in CASES.items() for player in ('red', 'blue')]
        self.plan = {'interface': scope.INTERFACE, 'source_commit': 'synthetic-accepted-source',
                     'implementation': scope.native_codec_identity(), 'native_runtime': self.runtime,
                     'binary_sha256': scope.binding(self.binary)['sha256'],
                     'replay_binary_sha256': self.runtime['vcmireplay'], 'schedule': schedule}
        self.report = {'schema': 'heroes-capability-native-validation-v1', 'status': 'passed',
                       'not_started': 0, 'training_input': False, 'training_performed': False,
                       'optimizer_steps': 0, 'cases': []}
        for scheduled in schedule:
            checks = [{'check': 'native_mechanic', 'status': 'passed'}]
            if scheduled['case'].startswith('dimension-door-'):
                blocked = scheduled['case'] == 'dimension-door-blocked'
                checks = [{'check': 'native_dimension_door_fog_attempt', 'status': 'passed',
                           'blocked_fixture': blocked, 'target': [4, 18],
                           'player_view_before_sha256': 'same-view', 'offered_spell_targets_sha256': 'same-targets',
                           'offered_spell_target_count': 224,
                           'result_status': 'rejected' if blocked else 'capability_applied'}]
            self.report['cases'].append({**scheduled, 'status': 'passed', 'replay_verified': True,
                'mechanics': {'history_audit_passed': True, 'probe_status': 'passed',
                              'episode_status': 'truncated', 'checks': checks}})
        self.migration_plan = {'runtime': self.runtime, 'binary': str(self.binary)}
        self.save_evidence()

    def write_json(self, path, value):
        path.write_text(json.dumps(value))
        return scope.binding(path)

    def save_evidence(self):
        self.probe_bound = self.write_json(self.directory/'plan.json', self.plan)
        self.report['plan_sha256'] = self.probe_bound['sha256']
        self.report_bound = self.write_json(self.directory/'report.json', self.report)
        self.summary = summarize([compact_report(self.directory/'report.json')])
        # Final retained summaries add independent replay counters to each row.
        self.summary['reports'][0]['rows'][0]['native_replay'] = {'unchecked': 0}
        self.value = {'native_acceptance': self.write_json(self.root/'summary.json', self.summary)}

    def check(self):
        return scope.validate_native_acceptance(self.value, self.migration_plan)

    def test_actual_complete_evidence_allows_only_its_bound_report(self):
        self.assertEqual(self.check(), {(self.report_bound['sha256'], self.probe_bound['sha256'])})

    def test_docs_and_controller_changes_do_not_invalidate_mechanics(self):
        for name in ('README.md', 'tools/v8_teacher_campaign.py',
                     'experiments/learner/builtin_v8_session.py', 'tests/test_controller.py'):
            (self.root/name).write_text('unrelated later change')
        self.check()

    def test_native_or_codec_drift_cannot_reuse_an_unchanged_runtime_pass(self):
        for name in ('engine/ExplorerRL/Capabilities.cpp', 'engine/patches/0028-visions.patch',
                     'experiments/learner/contextual_model.py', 'experiments/headless/contextual_protocol.py'):
            with self.subTest(name=name):
                path = self.root/name
                original = path.read_bytes()
                path.write_bytes(b'changed behavior with same binary')
                with self.assertRaisesRegex(ValueError, 'codec source changed'):
                    self.check()
                path.write_bytes(original)

    def test_added_or_removed_interface_source_is_detected(self):
        path = self.root/'experiments/learner/contextual_extra.py'
        path.write_text('new codec source')
        with self.assertRaisesRegex(ValueError, 'codec source changed'):
            self.check()
        path.unlink()
        (self.root/'experiments/learner/contextual_model.py').unlink()
        with self.assertRaisesRegex(ValueError, 'codec source changed'):
            self.check()

    def test_runtime_and_main_binary_must_match_final_acceptance(self):
        self.migration_plan['runtime'] = {**self.runtime, 'AI/libExplorerRL.dylib': 'changed'}
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.check()
        self.migration_plan['runtime'] = self.runtime
        self.binary.write_bytes(b'changed runtime')
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.check()

    def test_summary_cannot_mask_missing_mechanics_or_hidden_information_leak(self):
        row = next(row for row in self.report['cases']
                   if row['case'] == 'dimension-door-blocked' and row['player'] == 'red')
        row['mechanics']['checks'][0]['offered_spell_targets_sha256'] = 'hidden-layout-leak'
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, '28-case native acceptance is incomplete'):
            self.check()
        self.summary['native_coverage_complete'] = True
        self.value['native_acceptance'] = self.write_json(self.root/'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            self.check()

    def test_missing_or_failed_case_cannot_be_omitted_from_final_acceptance(self):
        self.report['cases'].pop()
        self.report['not_started'] = 1
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, '28-case native acceptance is incomplete'):
            self.check()

    def test_rebound_summary_cannot_change_frozen_implementation_hash(self):
        self.summary['reports'][0]['implementation_sha256'] = 'other-source'
        self.value['native_acceptance'] = self.write_json(self.root/'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'report binding or mechanics changed'):
            self.check()

    def release(self):
        source = {'tracked_worktree_dirty': False}
        game_map = self.root/'game-map'; game_map.write_bytes(b'map')
        plan = {**self.migration_plan, 'source': source, 'map': {}, 'game_map': str(game_map),
                'native': {'map': scope.binding(game_map)},
                'parent': {key: {'sha256': migration.PINS[key]} for key in ('policy', 'resume')}}
        receipt = {'schema': migration.SCHEMA+'-receipt',
                   'plan': self.write_json(self.root/'migration.json', plan),
                   'report': self.write_json(self.root/'migration-report.json', {})}
        receipt_bound = self.write_json(self.root/'migration-receipt.json', receipt)
        private_bound = self.write_json(self.root/'private-receipt.json', {'migration_receipt': receipt_bound})
        value = {**self.value, 'schema': scope.SCOPE+'-release', 'coordinator_released': True,
                 'release_message': 'Synthetic explicit release', 'authorization': scope.AUTHORIZATION,
                 'configuration': {}, 'source': source, 'migration_receipt': receipt_bound,
                 'private_migration_receipt': private_bound, 'runtime': self.runtime, 'map': {},
                 'feature_cases': [{'feature': feature, 'player': seat, 'passed': True, 'replay_verified': True,
                                    'evidence': self.report_bound, 'plan': self.probe_bound}
                                   for feature in scope.FEATURES for seat in (0, 1)]}
        return value, source

    def test_release_rejects_feature_mapping_from_another_passing_report(self):
        value, source = self.release()
        path = self.root/'release.json'
        with patch.object(scope, 'configuration', return_value={}), \
             patch.object(scope, 'source_identity', return_value=source), \
             patch.object(migration, 'verify_private'), \
             patch('experiments.headless.contextual_evidence.runtime_identity', return_value=self.runtime):
            self.write_json(path, value)
            scope.validate_release(path)
            old_plan = deepcopy(self.plan)
            old_plan['source_commit'] = 'another-otherwise-passing-run'
            old_bound = self.write_json(self.root/'older-plan.json', old_plan)
            old_report = deepcopy(self.report)
            old_report['plan_sha256'] = old_bound['sha256']
            value['feature_cases'][0].update(plan=old_bound,
                evidence=self.write_json(self.root/'older-report.json', old_report))
            self.write_json(path, value)
            with self.assertRaisesRegex(ValueError, 'outside the final native acceptance'):
                scope.validate_release(path)


if __name__ == '__main__':
    unittest.main()
