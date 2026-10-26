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


if __name__ == "__main__":
    unittest.main()
