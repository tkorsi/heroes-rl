"""Offline input gates for manual local training; no native subprocess is run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_local_training as prep


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is needed for the local runner")
class MachineConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.inputs = self.root / "input"
        self.runtime = self.root / "runtime/bin"
        self.data = self.root / "data"; (self.data / "Data").mkdir(parents=True)
        self.origin = self.root / "accepted"
        self.output = self.root / "outputs"
        self.proof = {"checkpoint": "synthetic-412", "weights_sha256": "a" * 64,
                      "progress": {"updates": 412, "rollouts": 1648, "decisions": 95836},
                      "exact_model_adam_rng_progress_pool": True}
        inputs = {}
        for name, (relative, _) in prep.INPUT_PINS.items():
            inputs[name] = (relative, self.file(self.inputs / relative, "synthetic " + name)["sha256"])
        maps = {seat: self.file(self.inputs / f"maps/curriculum-3-{seat}.vmap", "map " + seat)["sha256"]
                for seat in ("0", "1")}
        runtime = {name: self.file(self.runtime / name, "binary " + name)["sha256"]
                   for name in prep.RUNTIME_PINS}
        prerequisites = {name: self.file(self.root / "inputs/validation" / name, "historical " + name)["sha256"]
                         for name in prep.PREREQUISITE_PINS}
        runtime_identity = {name: sha for name, sha in runtime.items() if name != "vcmiclient"}
        gate = self.file(self.origin / "gate.json", json.dumps({"status": "finished",
                         "owned_cleanup_verified": True, "ppo_launched": False}))
        recipe = {"engine/example": "synthetic-recipe-sha"}
        release = self.file(self.origin / "release.json", json.dumps({"runtime": runtime_identity,
                 "runtime_bin": str(self.runtime), "gate": gate, "implementation": recipe,
                 "validation_outcomes": {seat: {"win": 1, "loss": 3} for seat in ("0", "1")}}))
        external = self.file(self.root / "external-library", "shared library")
        resource = self.file(self.runtime / "config/settings.json", "fixed engine settings")
        resource["path"] = "config/settings.json"
        freeze = self.file(self.origin / "freeze.json", json.dumps({"frozen_bin": str(self.runtime),
                 "artifacts": [{"path": name, "frozen_sha256": sha} for name, sha in runtime.items()],
                 "external_dependencies": [external], "resources": [resource]}))
        changes = {"DEFAULT_RUNTIME": self.runtime,
                   "INPUT_PINS": inputs, "MAP_PINS": maps, "RUNTIME_PINS": runtime,
                   "PREREQUISITE_PINS": prerequisites,
                   "ORIGIN_FILES": {name: self.origin / (name + ".json") for name in ("freeze", "release", "gate")},
                   "ORIGIN_PINS": {"freeze": freeze["sha256"], "release": release["sha256"], "gate": gate["sha256"]}}
        self.addCleanup(patch.stopall)
        for name, value in changes.items(): patch.object(prep, name, value).start()
        patch.object(prep, "verify_parent", return_value=self.proof).start()
        patch.object(prep, "current_engine_recipe", return_value=recipe).start()
        self.config = prep.build_configuration(self.inputs, self.runtime, self.data, self.root, self.output)

    def file(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        return prep.binding(path)

    def test_offline_roundtrip_has_explicit_dependencies_and_no_training(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(prep.validate_config(self.config), self.config)
        self.assertEqual(len(self.config["maps"]), 2)
        self.assertEqual(self.config["next_sequence"], 266)
        self.assertEqual(self.config["seed_namespace"], {"engine_base": 20_000_000, "policy_base": 30_000_000})
        self.assertIn("initial_checkpoint", self.config["parent"])
        self.assertFalse(self.config["gameplay_performed"])
        self.assertEqual(self.config["ppo_updates"], 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_existing_easy_descriptor_keeps_its_original_namespace(self):
        old = deepcopy(self.config); old.pop('seed_namespace')
        self.assertEqual(prep.validate_config(old), old)

    def test_original_profile_has_same_exact_map_both_seats_and_separate_seeds(self):
        path = self.root/'original-map.vmap'
        expected = self.file(path, 'original map fixture')
        with patch.object(prep, 'DEFAULT_ORIGINAL_MAP', path), \
             patch.object(prep, 'original_map_identity', side_effect=lambda p: prep.pinned(p, expected['sha256'])):
            value = prep.build_configuration(self.inputs, self.runtime, self.data, self.root,
                                             self.output/'original', 'original')
            self.assertEqual(value['maps'], [expected, expected])
            self.assertEqual(value['default_profile'], 'original')
            self.assertEqual(value['seed_namespace'], {'engine_base': 40_000_000, 'policy_base': 50_000_000})
            self.assertEqual(prep.validate_config(value), value)
            bad = deepcopy(value); bad.pop('seed_namespace')
            with self.assertRaisesRegex(ValueError, 'fields changed'): prep.validate_config(bad)
            bad = deepcopy(value); bad['maps'][1] = self.config['maps'][1]
            with self.assertRaisesRegex(ValueError, 'same exact map'): prep.validate_config(bad)
            bad = deepcopy(value); bad['seed_namespace'] = self.config['seed_namespace']
            with self.assertRaisesRegex(ValueError, 'configuration changed'): prep.validate_config(bad)
            path.write_text('changed original map')
            with self.assertRaisesRegex(ValueError, 'accepted identity'): prep.validate_config(value)

    def test_input_dependency_map_and_historical_evidence_changes_are_rejected(self):
        paths = [self.inputs / relative for relative, _ in prep.INPUT_PINS.values()]
        paths += [self.inputs / "maps/curriculum-3-0.vmap",
                  self.root / "inputs/validation" / next(iter(prep.PREREQUISITE_PINS))]
        for path in paths:
            original = path.read_bytes(); path.write_bytes(original + b"changed")
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, "accepted identity"):
                prep.validate_config(self.config)
            path.write_bytes(original)

    def test_runtime_changes_and_extra_modules_are_rejected(self):
        binary = self.runtime / "vcmiclient"; original = binary.read_bytes()
        binary.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "accepted identity"): prep.validate_config(self.config)
        binary.write_bytes(original)
        self.file(self.runtime / "AI/libUnexpected.dylib", "new unvalidated module")
        with self.assertRaisesRegex(ValueError, "inventory"): prep.validate_config(self.config)

    def test_runtime_relocation_and_unrelated_output_are_rejected(self):
        value = deepcopy(self.config); value["runtime_bin"] = str(self.root / "relocated")
        with self.assertRaises(ValueError): prep.validate_config(value)
        with self.assertRaisesRegex(ValueError, "child"):
            prep.build_configuration(self.inputs, self.runtime, self.data, self.root, self.root)
        with self.assertRaisesRegex(ValueError, "managed worktrees"):
            prep.durable("/Users/example/.codex/worktrees/task/output")

    def test_descriptor_cannot_silently_change_interpreter_seeds_or_proof(self):
        for key, replacement in (("python", {}), ("next_sequence", 1), ("checkpoint_restore", {}),
                                 ("training_authorized_by_preparation", True), ("runtime_origin", {})):
            value = deepcopy(self.config); value[key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "configuration changed"):
                prep.validate_config(value)
        value = deepcopy(self.config); value["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "fields changed"): prep.validate_config(value)

    def test_origin_must_be_passed_cold_replay_evidence(self):
        release = json.loads((self.origin / "release.json").read_text())
        release["validation_outcomes"]["1"]["loss"] = 0
        bound = self.file(self.origin / "release.json", json.dumps(release))
        prep.ORIGIN_PINS["release"] = bound["sha256"]
        with self.assertRaisesRegex(ValueError, "incomplete"): prep.validate_config(self.config)

    def test_linked_libraries_engine_resources_and_recipe_are_bound(self):
        for path in (self.root / "external-library", self.runtime / "config/settings.json"):
            original = path.read_bytes(); path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "accepted identity"):
                prep.validate_config(self.config)
            path.write_bytes(original)
        with patch.object(prep, "current_engine_recipe", return_value={"engine/extra": "new"}):
            with self.assertRaisesRegex(ValueError, "engine recipe"): prep.validate_config(self.config)


class AcceptedRuntimeTests(MachineConfigurationTests):
    def setUp(self):
        super().setUp()
        from experiments.explorer.create_garrison_scenarios import CASES, COLORS
        self.candidate = self.root / "garrison/bin"
        pins = {name: self.file(self.candidate / name, "garrison binary " + name)["sha256"] for name in prep.RUNTIME_PINS}
        runtime = {name: sha for name, sha in pins.items() if name != "vcmiclient"}
        recipe = prep.current_engine_recipe()
        implementation = {**recipe, "experiments/learner/contextual.py": "synthetic-contract-sha"}
        patch('validation.probes.validate_garrison_dialogs.implementation_identity', return_value=implementation).start()
        frozen = json.loads((self.origin / 'freeze.json').read_text())
        frozen.update(schema='heroes-local-runtime-freeze-v2', frozen_bin=str(self.candidate),
            source_recipe_check=True, donor_load_references_removed=True, old_runtime_modified=False,
            training_started=False, gameplay_started=False,
            source_inputs={name: {'sha256': sha} for name, sha in recipe.items()},
            artifacts=[dict(path=name, frozen_sha256=sha) for name, sha in pins.items()])
        server = self.file(self.candidate/'vcmiserver', 'frozen native server')
        frozen['artifacts'].append(dict(path='vcmiserver', frozen_sha256=server['sha256'], bytes=server['bytes']))
        dependency = self.file(self.origin/'dependency-check.stdout', 'exact canonical dependency recipe')
        frozen['dependency_check'] = {key: dependency[key] for key in ('sha256', 'bytes')}
        game_map = self.file(self.root/'frozen-original.vmap', 'unchanged preflight map')
        frozen['map'] = game_map
        for item in frozen['resources']:
            self.file(self.candidate / item['path'], 'fixed engine settings')
        (self.candidate/'Mods').mkdir()
        self.freeze = self.file(self.origin/'candidate-freeze.json', json.dumps(frozen))
        preflight = dict(schema='heroes-local-runtime-preflight-v2', launcher_preflight_passed=True,
            client_help_exit_code=0, donor_load_detected=False, gameplay_started=False, training_started=False,
            runtime=runtime, map=game_map, actual_client_libvcmi_load=[str(self.candidate/'libvcmi.dylib')],
            native=dict(binary=str(self.candidate/'vcmiclient'), binary_sha256=pins['vcmiclient'],
                        explorer_rl_sha256=pins['AI/libExplorerRL.dylib'], map={'sha256': game_map['sha256']}))
        self.preflight = self.file(self.origin/'preflight.json', json.dumps(preflight))
        schedule = [dict(case=case, player=color, seed=83010000+i*2+j)
                    for i, case in enumerate(CASES) for j, color in enumerate(COLORS)]
        plan = dict(schema='heroes-garrison-diagnostic-plan-v1', implementation=implementation,
            native_runtime=runtime, binary_sha256=pins['vcmiclient'], replay_binary_sha256=pins['vcmireplay'],
            schedule=schedule, training_performed=False, training_input=False, optimizer_steps=0)
        self.native_plan = self.file(self.origin/'native-plan.json', json.dumps(plan))
        report = dict(schema='heroes-garrison-native-validation-v1', plan_sha256=self.native_plan['sha256'],
            status='passed', not_started=0, training_performed=False, training_input=False, optimizer_steps=0,
            cases=[dict(**row, status='passed', replay_verified=True,
                   mechanics=dict(history_audit_passed=True, probe_status='passed')) for row in schedule])
        self.native_report = self.file(self.origin/'native-report.json', json.dumps(report))
        shutdown = dict(training_performed=False, optimizer_steps=0, owned_cleanup_verified=True,
                        within_hard_deadline=True, returncode=0, errors=[])
        self.shutdown = self.file(self.origin/'native-shutdown.json', json.dumps(shutdown))
        from validation.probes import validate_capabilities as capability
        regression_implementation = {**implementation, 'tools/validate_garrison_regressions.py': 'synthetic-wrapper-sha'}
        patch('tools.validate_garrison_regressions.implementation_identity', return_value=regression_implementation).start()
        legacy_schedule = capability.schedule(tuple(capability.CASES), capability.fixtures.COLORS, 83210000, 1)
        legacy_plan = dict(schema='heroes-capability-diagnostic-plan-v1', interface=capability.INTERFACE,
            implementation=regression_implementation, native_runtime=runtime, binary_sha256=pins['vcmiclient'],
            replay_binary_sha256=pins['vcmireplay'], schedule=legacy_schedule,
            training_performed=False, training_input=False, optimizer_steps=0)
        legacy_plan_bound = self.file(self.origin/'regression-plan.json', json.dumps(legacy_plan))
        legacy_report = dict(schema='heroes-capability-native-validation-v1', plan_sha256=legacy_plan_bound['sha256'],
            status='passed', not_started=0, training_performed=False, training_input=False, optimizer_steps=0,
            cases=[dict(**row, status='passed', replay_verified=True,
                mechanics=dict(history_audit_passed=True, probe_status='passed', episode_status='truncated',
                    training_performed=False, checks=[dict(check='independent_native_replay', status='passed')]))
                   for row in legacy_schedule])
        self.regression = dict(plan=legacy_plan_bound,
            report=self.file(self.origin/'regression-report.json', json.dumps(legacy_report)),
            shutdown=self.file(self.origin/'regression-shutdown.json',
                               json.dumps(dict(shutdown, training_input=False, cleanup_errors=[]))))
        policy = dict(schema='heroes-garrison-policy-reconstruction-v1', status='passed', implementation=implementation,
            runtime=runtime, native_report_sha256=self.native_report['sha256'], decisions_checked=128,
            mismatches=0, parameters_unchanged=True, exact_model_adam_rng_progress_pool=True,
            training_performed=False, optimizer_steps=0,
            checkpoint={name: self.file(self.origin/(name+'.pt'), 'frozen770-'+name)
                        for name in ('plan', 'report', 'policy', 'resume', 'initial_checkpoint')},
            evidence=[self.file(self.origin/'policy-report.json', 'synthetic policy evidence')],
            attempts=[dict(phase='evaluation', learner_player=i//4, engine_seed=83410000+i, policy_seed=83450000+i,
                status='passed', replay_verified=True, identity_verified=True, cleanup_verified=True,
                completion_receipt=True, weights_unchanged=True, training_samples=0, optimizer_steps=0,
                reconstruction={'passed': True}) for i in range(8)])
        self.policy = self.file(self.origin/'policy-reconstruction.json', json.dumps(policy))
        self.descriptor = dict(schema=prep.RUNTIME_ACCEPTANCE_SCHEMA, release_id='garrison-dialogs-v1',
            status='accepted', change_classification='adapter-support-extension', runtime_bin=str(self.candidate),
            runtime_files=pins, engine_recipe=recipe, freeze=self.freeze, preflight=self.preflight,
            native=dict(plan=self.native_plan, report=self.native_report, shutdown=self.shutdown),
            regression=self.regression, policy_reconstruction=self.policy)
        self.pin_descriptor()

    def pin_descriptor(self):
        receipt = self.file(self.origin/'accepted-runtime.json', json.dumps(self.descriptor))
        patch.object(prep, 'ACCEPTED_RUNTIME_RELEASES', {'garrison-dialogs-v1':
            {'descriptor': receipt['path'], 'sha256': receipt['sha256']}}).start()

    def accepted(self):
        return prep.build_configuration(self.inputs, self.candidate, self.data, self.root, self.output,
                                        runtime_release='garrison-dialogs-v1')

    def test_explicit_reviewed_release_roundtrips_without_changing_legacy_defaults(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        value = self.accepted()
        self.assertEqual(value['runtime_bin'], str(self.candidate))
        self.assertEqual(value['runtime_release']['id'], 'garrison-dialogs-v1')
        self.assertEqual(prep.validate_config(value), value)
        self.assertEqual(prep.validate_config(self.config), self.config)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_unreviewed_missing_or_changed_descriptor_fails_closed(self):
        with patch.object(prep, 'ACCEPTED_RUNTIME_RELEASES', {'garrison-dialogs-v1': {'sha256': None}}):
            with self.assertRaisesRegex(ValueError, 'reviewed acceptance pin'): self.accepted()
        with self.assertRaisesRegex(ValueError, 'reviewed acceptance pin'):
            prep.accepted_runtime_descriptor('arbitrary-release')
        path = self.origin/'accepted-runtime.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'accepted identity'): self.accepted()

    def test_native_candidate_can_be_checked_without_becoming_machine_acceptance(self):
        from tools.validate_interaction_guard import load_runtime_candidate
        candidate = dict(self.descriptor, schema='heroes-local-training-runtime-candidate-v1')
        candidate.pop('policy_reconstruction')
        bound = self.file(self.origin/'candidate.json', json.dumps(candidate))
        with patch.object(prep, 'ACCEPTED_RUNTIME_RELEASES', {}):
            value, actual = load_runtime_candidate(bound['path'], bound)
            self.assertEqual(value, candidate)
            self.assertEqual(actual, bound)
            with self.assertRaisesRegex(ValueError, 'reviewed acceptance pin'): self.accepted()
        with self.assertRaisesRegex(ValueError, 'descriptor changed'):
            load_runtime_candidate(bound['path'], {**bound, 'sha256': '0'*64})

    def test_legacy_regression_is_required_and_keeps_paired_fog_seeds(self):
        value = self.accepted()
        self.assertEqual(value['runtime_origin']['regression'], self.regression)
        schedule = json.loads(Path(self.regression['plan']['path']).read_text())['schedule']
        self.assertEqual(len(schedule), 28)
        self.assertEqual(len({row['seed'] for row in schedule}), 26)
        self.assertEqual([row['seed'] for row in schedule[-4:-2]], [row['seed'] for row in schedule[-2:]])
        self.descriptor.pop('regression')
        self.pin_descriptor()
        with self.assertRaisesRegex(ValueError, 'regression bindings'): self.accepted()

    def test_failed_missing_or_learning_legacy_case_fails_closed(self):
        for mutation in ('missing', 'failed', 'replay', 'history', 'learning', 'unverified_check', 'cleanup'):
            legacy = deepcopy(self.regression)
            report = json.loads(Path(legacy['report']['path']).read_text())
            if mutation == 'cleanup':
                shutdown = json.loads(Path(legacy['shutdown']['path']).read_text())
                shutdown['cleanup_errors'] = ['still alive']
                legacy['shutdown'] = self.file(self.root/('regression-'+mutation+'.json'), json.dumps(shutdown))
            else:
                row = report['cases'][0]
                if mutation == 'missing': report['cases'].pop()
                elif mutation == 'failed': row['status'] = 'failed'
                elif mutation == 'replay': row['replay_verified'] = False
                elif mutation == 'history': row['mechanics']['history_audit_passed'] = False
                elif mutation == 'learning': row['mechanics']['training_performed'] = True
                else: row['mechanics']['checks'][0]['status'] = 'unverified'
                legacy['report'] = self.file(self.root/('regression-'+mutation+'.json'), json.dumps(report))
            self.descriptor['regression'] = legacy
            self.pin_descriptor()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'regression'): self.accepted()
        self.descriptor['regression'] = self.regression

    def test_legacy_regression_binds_source_runtime_and_exact_schedule(self):
        for mutation in ('source', 'runtime', 'seed', 'fixture', 'repeat'):
            legacy = deepcopy(self.regression)
            plan = json.loads(Path(legacy['plan']['path']).read_text())
            if mutation == 'source': plan['implementation'] = {}
            elif mutation == 'runtime': plan['binary_sha256'] = '0'*64
            elif mutation == 'seed': plan['schedule'][1]['seed'] = plan['schedule'][0]['seed']
            elif mutation == 'fixture': plan['schedule'][0]['fixture'] = 'artifacts'
            else: plan['schedule'][0]['repeat'] = 1
            legacy['plan'] = self.file(self.root/('regression-'+mutation+'.json'), json.dumps(plan))
            report = json.loads(Path(legacy['report']['path']).read_text())
            report['plan_sha256'] = legacy['plan']['sha256']
            legacy['report'] = self.file(self.root/('regression-'+mutation+'-report.json'), json.dumps(report))
            self.descriptor['regression'] = legacy
            self.pin_descriptor()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'regression'): self.accepted()
        self.descriptor['regression'] = self.regression

    def test_incomplete_native_case_and_cleanup_cannot_be_accepted(self):
        for field in ('replay_verified', 'history_audit_passed', 'owned_cleanup_verified'):
            report = json.loads(Path(self.native_report['path']).read_text())
            shutdown = json.loads(Path(self.shutdown['path']).read_text())
            if field == 'owned_cleanup_verified':
                shutdown[field] = False
                self.descriptor['native']['shutdown'] = self.file(self.root/(field+'.json'), json.dumps(shutdown))
            else:
                row = report['cases'][0]
                (row['mechanics'] if field == 'history_audit_passed' else row)[field] = False
                self.descriptor['native']['report'] = self.file(self.root/(field+'.json'), json.dumps(report))
            self.pin_descriptor()
            with self.subTest(field=field), self.assertRaises(ValueError): self.accepted()
            self.descriptor['native'] = dict(plan=self.native_plan, report=self.native_report, shutdown=self.shutdown)

    def test_policy_reconstruction_cannot_be_omitted_or_admit_learning(self):
        for field, bad in (('mismatches', 1), ('parameters_unchanged', False), ('decisions_checked', 0),
                           ('optimizer_steps', 1), ('attempts', [])):
            policy = json.loads(Path(self.policy['path']).read_text()); policy[field] = bad
            self.descriptor['policy_reconstruction'] = self.file(self.root/(field+'.json'), json.dumps(policy))
            self.pin_descriptor()
            with self.subTest(field=field), self.assertRaises(ValueError): self.accepted()

    def test_replay_repair_classification_requires_its_additional_terminal_receipt(self):
        self.descriptor['change_classification'] = 'replay-verifier-support-extension'
        self.pin_descriptor()
        with self.assertRaisesRegex(ValueError, "original failure's diagnostic evidence"):
            self.accepted()

    def test_native_identity_and_freeze_must_match_current_source_and_inventory(self):
        with patch.object(prep, 'current_engine_recipe', return_value={'engine/changed': 'different'}):
            with self.assertRaisesRegex(ValueError, 'engine recipe'): self.accepted()
        with patch('validation.probes.validate_garrison_dialogs.implementation_identity', return_value={}):
            with self.assertRaisesRegex(ValueError, 'source/runtime'): self.accepted()
        self.file(self.candidate/'AI/libUnexpected.dylib', 'not accepted')
        with self.assertRaisesRegex(ValueError, 'inventory'): self.accepted()

    def test_actual_nine_artifact_freeze_binds_server_separately(self):
        value = self.accepted()
        frozen = json.loads(Path(self.freeze['path']).read_text())
        self.assertEqual(len(frozen['artifacts']), 9)
        self.assertNotIn('vcmiserver', value['runtime'])
        self.assertNotIn('vcmiserver', value['runtime_files'])
        self.assertEqual(value['runtime_origin']['server'], prep.binding(self.candidate/'vcmiserver'))
        self.file(self.candidate/'vcmiserver', 'changed server outside runtime_identity')
        with self.assertRaisesRegex(ValueError, 'accepted identity'): self.accepted()

    def test_preflight_cannot_hide_loader_failure_and_dependency_check_is_bound(self):
        for field, bad in (('launcher_preflight_passed', False), ('donor_load_detected', True),
                           ('runtime', {}), ('actual_client_libvcmi_load', [])):
            preflight = json.loads(Path(self.preflight['path']).read_text()); preflight[field] = bad
            self.descriptor['preflight'] = self.file(self.root/(field+'.json'), json.dumps(preflight))
            self.pin_descriptor()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'preflight'):
                self.accepted()
        self.descriptor['preflight'] = self.preflight; self.pin_descriptor()
        self.file(self.origin/'dependency-check.stdout', 'changed dependency audit')
        with self.assertRaisesRegex(ValueError, 'accepted identity'): self.accepted()

    def test_extra_resource_and_replaced_resource_symlink_are_rejected(self):
        added = self.candidate/'Mods/unreviewed/mod.json'
        self.file(added, 'new mod changes loaded rules')
        with self.assertRaisesRegex(ValueError, 'resource inventory'): self.accepted()
        added.unlink()
        target = self.candidate/'config/settings.json'
        original = target.read_bytes(); target.unlink()
        donor = self.root/'moved-settings.json'; donor.write_bytes(original)
        target.symlink_to(donor)
        with self.assertRaisesRegex(ValueError, 'resource symlink'): self.accepted()


class TerminalReplayAcceptanceTests(unittest.TestCase):
    """Synthetic proof admission checks; these do not claim native replay."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'diagnostic.json'
        self.runtime = {'vcmireplay': 'a' * 64}
        from tests.test_builtin_terminal_replay import TerminalReportTests
        native = TerminalReportTests(); native.setUp(); native.add_accounting(190)
        native.wire['request_order_verified'] = True
        self.proof = dict(schema='heroes-contextual-builtin-replay-diagnostic-v1',
            diagnostic_verified=True, verified=False, original_cohort_result_unchanged=True,
            original_cohort_admission=False, optimization_eligible=False, training_performed=False,
            training_samples=0,
            identity_check=dict(mode='diagnostic_verifier_only', client_unchanged=True,
                adapter_sources_unchanged=True, nonverifier_runtime_unchanged=True, nonverifier_recipe_unchanged=True,
                runtime_changes={'vcmireplay': {'captured': 'b' * 64, 'current': 'a' * 64}}),
            wire=native.wire, native=native.native,
            history=dict(schema='heroes-builtin-history-reconstruction-v1', verified=True,
                         boundary='natural', opponent_history=None),
            effective_engine={'verified': True},
            prefix=dict(server_partial_bytes=0, server_unpaired_inputs=0, contextual_partial_bytes=0))

    def descriptor(self, proof=None):
        self.path.write_text(json.dumps(self.proof if proof is None else proof))
        return {'terminal_replay': prep.binding(self.path)}

    def test_exact_excluded_diagnostic_is_read_only_and_bound_to_the_new_verifier(self):
        descriptor = self.descriptor()
        before = self.path.read_bytes()
        self.assertEqual(prep.terminal_replay_acceptance(descriptor, self.runtime), descriptor['terminal_replay'])
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_or_mutated_diagnostic_binding_fails_closed(self):
        for bound in (None, {}, {'path': str(self.path)}, {'path': str(self.path), 'sha256': 'a'*64, 'bytes': 1, 'extra': 1}):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                prep.terminal_replay_acceptance({'terminal_replay': bound}, self.runtime)
        descriptor = self.descriptor()
        self.path.write_text(self.path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'accepted identity'):
            prep.terminal_replay_acceptance(descriptor, self.runtime)
        descriptor = self.descriptor(); descriptor['terminal_replay']['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            prep.terminal_replay_acceptance(descriptor, self.runtime)

    def test_failed_rescored_changed_original_or_learning_diagnostic_is_rejected(self):
        changes = [('schema', 'heroes-contextual-builtin-replay-verification-v1'), ('diagnostic_verified', False),
            ('verified', True), ('original_cohort_result_unchanged', False), ('original_cohort_admission', True),
            ('optimization_eligible', True), ('training_performed', True), ('training_samples', 1)]
        for key, bad in changes:
            for remove in (False, True):
                proof = deepcopy(self.proof)
                if remove: proof.pop(key)
                else: proof[key] = bad
                with self.subTest(key=key, remove=remove), self.assertRaises(ValueError):
                    prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)

    def test_identity_changes_cannot_cross_the_verifier_boundary(self):
        for key in ('client_unchanged', 'adapter_sources_unchanged', 'nonverifier_runtime_unchanged',
                    'nonverifier_recipe_unchanged'):
            proof = deepcopy(self.proof); proof['identity_check'][key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)
        for changes in ({}, {'vcmiclient': {'current': 'a'*64}},
                        {'vcmireplay': {'current': 'c'*64}},
                        {'vcmireplay': {'current': 'a'*64}, 'libvcmi.dylib': {'current': 'd'*64}}):
            proof = deepcopy(self.proof); proof['identity_check']['runtime_changes'] = changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)
        proof = deepcopy(self.proof); proof['identity_check']['mode'] = 'exact_capture'
        with self.assertRaises(ValueError):
            prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)

    def test_only_the_single_exchange_artifacts_tail_satisfies_this_repair_gate(self):
        for tail in ([], [{'pack_type': 193}], [{'pack_type': 198}],
                     [{'pack_type': 190}, {'pack_type': 190}], [{'pack_type': '190'}]):
            proof = deepcopy(self.proof); proof['wire']['request_accounting']['terminal_unapplied'] = tail
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, 'ExchangeArtifacts tail'):
                prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)

    def test_native_history_and_effect_proofs_cannot_be_missing_or_incomplete(self):
        changes = [lambda p: p.pop('native'), lambda p: p['native'].update(returncode=1),
            lambda p: p['native'].update(verified=False), lambda p: p['native'].update(schema='wrong'),
            lambda p: p['native']['summary']['checked'].update(requests=4),
            lambda p: p['native']['summary']['unchecked'].update(requests=1),
            lambda p: p['native']['summary']['request_accounting']['terminal_unapplied'][0].update(pack='aabb'),
            lambda p: p['native']['summary']['server'].update(inputs=5),
            lambda p: p['native']['summary']['terminal_barrier']['request'].update(id=41),
            lambda p: p.pop('history'), lambda p: p['history'].update(verified=False),
            lambda p: p['history'].update(boundary='cutoff'), lambda p: p['history'].update(opponent_history={}),
            lambda p: p['effective_engine'].update(verified=False),
            lambda p: p['wire'].update(request_order_verified=False),
            lambda p: p['prefix'].update(server_unpaired_inputs=1)]
        for change in changes:
            proof = deepcopy(self.proof); change(proof)
            with self.subTest(change=change), self.assertRaises(ValueError):
                prep.terminal_replay_acceptance(self.descriptor(proof), self.runtime)


if __name__ == "__main__":
    unittest.main()
