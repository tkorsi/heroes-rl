"""Fail-closed Linux plan/portability checks using synthetic private fixtures only."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import linux_validation_plan as plan
from tools import port_fixed_policy_linux as portability


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


class PlanDocumentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "plan.json"
        self.document = dict.fromkeys(plan.PLAN_FIELDS)
        self.document.update(schema=plan.SCHEMA, configuration=deepcopy(plan.CONFIGURATION),
            schedule=deepcopy(plan.SCHEDULE), verification=list(plan.VERIFICATION),
            execution_output=str(self.root / "execution"), training_performed=False, optimizer_steps=0,
            training_samples=0, maps={"fixture": {}, "ordinary": {}}, data=str(self.root / "game"),
            binary=str(self.root / "bin/vcmiclient"),
            replay_binary={"path": str(self.root / "bin/vcmireplay"), "sha256": "a" * 64},
            runtime={"vcmireplay": "a" * 64})
        for name in ("original_checkpoint", "checkpoint", "portability", "input_manifest"):
            self.document[name] = {"path": str(self.root / name)}

    def read(self, document=None):
        write_json(self.path, self.document if document is None else document)
        return plan.read_plan(self.path, plan.binding(self.path)["sha256"])

    def test_fixed_four_rows_share_one_budget_and_preserve_scopes(self):
        loaded = self.read()
        self.assertEqual(loaded["configuration"]["shared_wall_seconds"], 3600)
        self.assertEqual([(row["map_kind"], row["learner_player"]) for row in loaded["schedule"]],
                         [("fixture", 0), ("fixture", 1), ("ordinary", 0), ("ordinary", 1)])
        self.assertEqual([row["max_days"] for row in loaded["schedule"]], [12, 12, 200, 200])

    def test_rehashed_plan_cannot_expand_limits_reseed_reorder_or_enable_training(self):
        changes = [lambda p: p["configuration"].update(shared_wall_seconds=3601),
            lambda p: p["schedule"][0].update(engine_seed=1), lambda p: p["schedule"].reverse(),
            lambda p: p["schedule"][0].update(learner_player=False),
            lambda p: p["schedule"][2].update(max_total_choices=2049),
            lambda p: p["schedule"][2].update(max_days=201),
            lambda p: p["schedule"][2].update(case_wall_seconds=1201),
            lambda p: p.update(training_performed=True), lambda p: p.update(optimizer_steps=False),
            lambda p: p["verification"].pop(), lambda p: p.update(resume=True)]
        for change in changes:
            with self.subTest(change=change):
                document = deepcopy(self.document)
                change(document)
                with self.assertRaises(ValueError):
                    self.read(document)

    def test_changed_plan_bytes_and_duplicate_fields_rejected(self):
        self.read()
        sha = plan.binding(self.path)["sha256"]
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            plan.read_plan(self.path, sha)
        self.path.write_text('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            plan.read_plan(self.path, plan.binding(self.path)["sha256"])

    def test_private_paths_output_and_separate_verifier_are_constrained(self):
        changes = [lambda p: p.update(execution_output=str(self.root / "old-execution")),
            lambda p: p.update(data=str(plan.ROOT / "runs/private")),
            lambda p: p["checkpoint"].update(path=str(plan.ROOT / "ignored.pt")),
            lambda p: p["replay_binary"].update(path=str(self.root / "other/vcmireplay")),
            lambda p: p["replay_binary"].update(sha256="b" * 64)]
        for change in changes:
            with self.subTest(change=change):
                document = deepcopy(self.document)
                change(document)
                with self.assertRaises(ValueError):
                    self.read(document)


class PrivateInventoryBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "game/Data").mkdir(parents=True)
        (self.root / "game/Data/test.bin").write_bytes(b"Synthetic bytes, not game data")
        self.original = {"bytes": 4, "sha256": "0" * 64}
        self.path = self.root / "inputs.json"
        self.document = {"schema": "heroes-cloud-private-inputs-v1", "provenance": "Synthetic test",
            "data_files": plan.inputs.inventory_data(self.root / "game"),
            "checkpoint": {"filename": "fixed-policy.pt", **self.original},
            "game_data_validation": "byte_inventory_only", "gameplay_performed": False}
        write_json(self.path, self.document)

    def validate(self, **kwargs):
        return plan.validate_inventory(self.root / "game", self.original, self.path, **kwargs)

    def test_byte_inventory_is_not_gameplay_evidence_and_detects_changes(self):
        result = self.validate()
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["validation"], "byte_inventory_only")
        (self.root / "game/Data/test.bin").write_bytes(b"Edited data")
        with self.assertRaisesRegex(ValueError, "Data differs"):
            self.validate()

    def test_rebound_inventory_rejects_missing_duplicate_unsafe_or_malformed_rows(self):
        changes = [lambda d: d.update(data_files=[]), lambda d: d["data_files"].append(d["data_files"][0]),
            lambda d: d["data_files"][0].update(path="../private"),
            lambda d: d["data_files"][0].update(path="/private"),
            lambda d: d["data_files"][0].update(bytes=True),
            lambda d: d["checkpoint"].update(sha256="1" * 64),
            lambda d: d.update(game_data_validation="gameplay_verified"),
            lambda d: d.update(gameplay_performed=True)]
        for change in changes:
            with self.subTest(change=change):
                document = deepcopy(self.document)
                change(document)
                write_json(self.path, document)
                with self.assertRaises(ValueError):
                    self.validate(full_data=False)


class RuntimeResourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "vcmi/build/bin/vcmiclient"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"synthetic")
        for name in ("config", "Mods"):
            directory = self.root / "vcmi" / name
            directory.mkdir()
            (directory / "test.json").write_text('{"synthetic":true}')
            (self.binary.parent / name).symlink_to(directory, target_is_directory=True)

    def test_rule_file_changes_and_symlink_retargets_change_or_fail_binding(self):
        with patch.object(plan, "ROOT", self.root):
            before = plan.native_resources(self.binary)
            (self.root / "vcmi/config/test.json").write_text('{"synthetic":false}')
            self.assertNotEqual(plan.native_resources(self.binary), before)
            (self.binary.parent / "config").unlink()
            (self.binary.parent / "config").symlink_to(self.root / "vcmi/Mods")
            with self.assertRaisesRegex(ValueError, "resource root"):
                plan.native_resources(self.binary)

    def test_shadow_assets_and_nested_resource_symlinks_rejected(self):
        with patch.object(plan, "ROOT", self.root):
            (self.binary.parent / "data").mkdir()
            with self.assertRaisesRegex(ValueError, "resource directory"):
                plan.native_resources(self.binary)
            (self.binary.parent / "data").rmdir()
            (self.root / "vcmi/config/link").symlink_to(self.root / "vcmi/Mods/test.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                plan.native_resources(self.binary)

    def test_case_variant_rule_roots_cannot_shadow_bound_resources(self):
        with patch.object(plan, "ROOT", self.root):
            (self.binary.parent / "Config").mkdir()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                plan.native_resources(self.binary)

    def test_missing_dynamic_library_is_never_ignored(self):
        checked = SimpleNamespace(stdout="libmissing.so => not found\n", stderr="", returncode=0)
        with patch.object(plan.subprocess, "run", return_value=checked):
            with self.assertRaisesRegex(ValueError, "resolution failed"):
                plan.native_dependencies(self.binary, {})


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None and str(torch.__version__) == "2.9.1+cpu"
                     and platform.system() == "Linux" and platform.machine() == "x86_64"
                     and plan.sys.version_info[:2] == (3, 12), "Requires validated Linux CPU portability runtime")
class LinuxCheckpointBindingTests(unittest.TestCase):
    def setUp(self):
        from experiments.learner.checkpoints import FrozenOpponentPool, save_policy_checkpoint
        from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, DIALOGS_MODEL_SCHEMA
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "synthetic-original.pt"
        model = ContextualActorCritic(ModelConfig(schema=DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        pool = FrozenOpponentPool()
        pool.add("synthetic-opponent", model)
        save_policy_checkpoint(self.original, model, checkpoint="synthetic-policy", source_metadata=plan.source_identity(),
                               opponent_pool=pool, interface=plan.INTERFACE)
        payload = torch.load(self.original, weights_only=True)
        payload["torch_version"] = payload["source_metadata"]["torch"] = "2.9.1"
        payload["source_metadata"]["python"] = "3.13.5 (synthetic source fixture)"
        torch.save(payload, self.original)
        self.expected = {"original_sha256": portability.digest(self.original), "original_bytes": self.original.stat().st_size,
                         "learner": payload["weights_sha256"], "opponents": [list(x) for x in pool.identities]}
        directory = self.root / "converted"
        portability._convert(self.original, directory, expected_sha256=self.expected["original_sha256"],
            expected_bytes=self.expected["original_bytes"], expected_learner=self.expected["learner"],
            expected_opponents=list(pool.identities))
        self.converted = directory / "fixed-policy-linux.pt"
        self.report = directory / "portability.json"
        self.original_payload = payload
        self.converted_payload = torch.load(self.converted, weights_only=True)
        self.report_document = json.loads(self.report.read_text())

    def load(self):
        return plan._load_fixed_inputs(self.original, self.converted, self.report, expected=self.expected)

    def replace_converted(self, payload):
        torch.save(payload, self.converted)
        report = deepcopy(self.report_document)
        report["converted"].update(sha256=portability.digest(self.converted), bytes=self.converted.stat().st_size)
        write_json(self.report, report)

    def test_exact_reloaded_models_preserve_original_rng_and_inference_only_identity(self):
        before = self.original.read_bytes()
        rng = torch.get_rng_state().clone()
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer")):
            loaded, fixed = self.load()
        self.assertEqual(before, self.original.read_bytes())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(loaded.model.training)
        self.assertTrue(all(not value.requires_grad for value in loaded.model.parameters()))
        self.assertEqual(fixed["opponents"], self.expected["opponents"])
        portability.assert_exact_state(self.original_payload["model_state"], dict(loaded.model.state_dict()))
        portability.assert_exact_state(self.original_payload["opponent_pool"][0]["model_state"],
                                      dict(loaded.opponent_pool.get("synthetic-opponent").model.state_dict()))

    def test_public_loader_refuses_synthetic_original_even_with_self_consistent_report(self):
        with self.assertRaisesRegex(ValueError, "Original checkpoint size or SHA-256"):
            plan._load_fixed_inputs(self.original, self.converted, self.report)

    def test_rebound_report_cannot_change_either_model_or_provenance(self):
        def change_tensor(payload, opponent=False, dtype=False):
            state = payload["opponent_pool"][0]["model_state"] if opponent else payload["model_state"]
            key = next(iter(state))
            if dtype:
                state[key] = state[key].double()
            else:
                state[key].reshape(-1)[0] += 1
        changes = [lambda p: change_tensor(p), lambda p: change_tensor(p, opponent=True),
            lambda p: change_tensor(p, dtype=True), lambda p: p.update(resume={}),
            lambda p: p.update(checkpoint="invented-policy"),
            lambda p: p["source_metadata"]["portability"].update(optimizer_updates=1),
            lambda p: p["source_metadata"].update(git_commit="not-a-commit"),
            lambda p: p["source_metadata"].update(git_commit="0" * 40),
            lambda p: p["source_metadata"].update(tracked_worktree_dirty="not-a-bool"),
            lambda p: p["source_metadata"]["source_sha256"].pop(next(iter(p["source_metadata"]["source_sha256"]))),
            lambda p: p["source_metadata"]["portability"]["original_source_metadata"].update(torch="wrong")]
        for change in changes:
            with self.subTest(change=change):
                payload = deepcopy(self.converted_payload)
                change(payload)
                self.replace_converted(payload)
                with self.assertRaises(ValueError):
                    self.load()

    def test_original_learner_source_guard_remains_strict(self):
        changed = deepcopy(plan.source_identity())
        name = next(iter(self.original_payload["source_metadata"]["source_sha256"]))
        changed["source_sha256"][name] = "0" * 64
        with patch.object(plan, "source_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "Original learner source changed"):
                self.load()

    def test_report_claims_cannot_substitute_for_actual_verification(self):
        changes = [lambda r: r["original"].update(sha256="0" * 64),
            lambda r: r["converted"].update(sha256="0" * 64),
            lambda r: r.update(exact_learner_and_opponent_tensors=False),
            lambda r: r.update(optimizer_updates=False), lambda r: r.update(gameplay_performed=True),
            lambda r: r["target_runtime"].update(torch="2.9.1"),
            lambda r: r.update(cross_platform_behavior_equivalence="verified")]
        for change in changes:
            with self.subTest(change=change):
                document = deepcopy(self.report_document)
                change(document)
                write_json(self.report, document)
                with self.assertRaises(ValueError):
                    self.load()

    def test_prepare_read_identity_roundtrip_and_bound_artifact_drift(self):
        """Exercise the whole plan API; synthetic stand-ins never execute VCMI.

        Only unavailable private identities, clean-source admission and native
        recipe/map/ELF preflights are substituted. Real plan serialization,
        checkpoint loading, file hashes, launcher identity, Data inventory and
        all check_identity comparisons remain active.
        """
        binary = self.root / "bin/vcmiclient"
        binary.parent.mkdir()
        binary.write_bytes(b"synthetic VCMI_EXPERIMENT_DIR EXPLORER_RL_PLAYERS VCMI_EXPERIMENT_REPLAY")
        binary.chmod(0o700)
        (binary.parent / "AI").mkdir()
        for name in ("libvcmi.so", "AI/libBattleAI.so", "AI/libNullkiller.so", "vcmireplay"):
            (binary.parent / name).write_bytes(b"synthetic runtime only")
        (binary.parent / "AI/libExplorerRL.so").write_bytes(b"visible_update-v2 last_action_id observations-p")
        game = self.root / "game"
        (game / "Data").mkdir(parents=True)
        data_file = game / "Data/synthetic.bin"
        data_file.write_bytes(b"synthetic private input, not game data")
        ordinary, fixture = self.root / "ordinary.vmap", self.root / "fixture.vmap"
        ordinary.write_bytes(b"synthetic ordinary map")
        fixture.write_bytes(b"synthetic townless fixture")
        rules = self.root / "rules.json"
        rules.write_text('{"synthetic":true}')
        manifest = self.root / "private-inputs.json"
        write_json(manifest, {"schema": "heroes-cloud-private-inputs-v1", "provenance": "Synthetic integration fixture",
            "data_files": plan.inputs.inventory_data(game),
            "checkpoint": {"filename": "fixed-policy.pt", "bytes": self.expected["original_bytes"],
                           "sha256": self.expected["original_sha256"]},
            "game_data_validation": "byte_inventory_only", "gameplay_performed": False})
        args = SimpleNamespace(output=self.root / "prepared", original_checkpoint=self.original,
            checkpoint=self.converted, portability=self.report, input_manifest=manifest, data=game,
            binary=binary, replay_binary=binary.parent / "vcmireplay", map=ordinary, fixture_map=fixture)
        actual_load = plan._load_fixed_inputs
        fixed_runtime = plan.runtime()
        def synthetic_load(original, converted, report):
            return actual_load(original, converted, report, expected=self.expected)
        def synthetic_map(path, kind):
            return {**plan.binding(path), "fixture": kind == "fixture", "kind": kind}
        with ExitStack() as stack:
            stack.enter_context(patch.object(plan, "_load_fixed_inputs", side_effect=synthetic_load))
            stack.enter_context(patch.object(plan, "require_clean_source", return_value=None))
            stack.enter_context(patch.object(plan, "map_identity", side_effect=synthetic_map))
            stack.enter_context(patch.object(plan, "prepared_check", return_value="synthetic recipe preflight"))
            stack.enter_context(patch.object(plan, "runtime", side_effect=lambda: deepcopy(fixed_runtime)))
            stack.enter_context(patch.object(plan, "native_resources", side_effect=lambda _: {"rules": plan.binding(rules)}))
            stack.enter_context(patch.object(plan, "native_dependencies",
                side_effect=lambda client, _: {"synthetic_client": plan.binding(client)}))
            prepared = plan.prepare(args)
            loaded = plan.read_plan(prepared["plan"], prepared["sha256"])
            self.assertEqual(loaded["execution_output"], str(args.output / "execution"))
            self.assertFalse(Path(loaded["execution_output"]).exists())
            plan.check_identity(loaded)
            frozen, fingerprints = plan.load_fixed(loaded)
            self.assertEqual(fingerprints, loaded["frozen"])
            self.assertEqual(frozen.source_metadata, loaded["checkpoint_source"])
            for path in (self.converted, self.report, binary, ordinary, data_file, rules):
                with self.subTest(changed=path.name):
                    before = path.read_bytes()
                    path.write_bytes(before + b" ")
                    try:
                        with self.assertRaises(ValueError):
                            plan.check_identity(loaded)
                    finally:
                        path.write_bytes(before)
            changed_source = deepcopy(loaded["source"])
            changed_source["source_sha256"]["synthetic-changed-source.py"] = "0" * 64
            with patch.object(plan, "source_identity", return_value=changed_source):
                with self.assertRaisesRegex(ValueError, "Authored source changed"):
                    plan.check_identity(loaded)
            fixed_runtime["torch_float32_matmul_precision"] = "changed"
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                plan.check_identity(loaded)
            before = Path(prepared["plan"]).read_bytes()
            with self.assertRaisesRegex(ValueError, "output already exists"):
                plan.prepare(args)
            self.assertEqual(Path(prepared["plan"]).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
