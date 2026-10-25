"""Preparation-only checks: synthetic evidence and real random tensor serialization.

These tests never launch the engine or call an optimizer.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import first_run

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


class FirstRunConfigurationTests(unittest.TestCase):
    def config(self):
        return json.loads(first_run.CONFIG.read_text())

    def test_reviewed_budget_and_nontraining_schedule_are_exact(self):
        config = first_run.validate_config(self.config())
        rows = first_run.schedule(config)
        self.assertEqual([row["phase"] for row in rows], ["before"] * 4 + ["training"] * 2 + ["after"] * 4)
        self.assertEqual([(r["engine_seed"], r["learner_player"], r["policy_seeds"]) for r in rows[:4]],
                         [(r["engine_seed"], r["learner_player"], r["policy_seeds"]) for r in rows[-4:]])
        self.assertEqual(sum(r["optimizer_eligible"] for r in rows), 2)
        self.assertTrue(set(config["training_engine_seeds"]).isdisjoint(config["evaluation_engine_seeds"]))
        self.assertEqual((config["max_updates"], config["ppo"]["epochs"], config["max_optimizer_steps"]), (1, 1, 16))
        config["ppo"]["epochs"] = 9
        self.assertEqual(self.config()["ppo"]["epochs"], 1)

    def test_budget_seed_overlap_unknown_field_and_numeric_aliases_rejected(self):
        changes = [(("max_requests",), 257), (("torch_threads",), True), (("max_days",), 30.0),
                   (("training_engine_seeds",), [99201]), (("evaluation_engine_seeds",), (99201, 99202)),
                   (("ppo", "entropy_coefficient"), 0.01), (("ppo", "gamma"), 0.99),
                   (("ppo", "normalize_advantages"), 1), (("max_updates",), 2), (("unknown",), 1)]
        for keys, value in changes:
            with self.subTest(keys=keys, value=value):
                config = self.config(); target = config
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError): first_run.validate_config(config)
        for value in (None, [], 3, "config"):
            with self.assertRaises(ValueError): first_run.validate_config(value)

    def test_preparation_has_no_execution_import_or_call(self):
        import ast
        tree = ast.parse(Path(first_run.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, ("gameplay", "ppo", "experiments.learner.gameplay", "experiments.learner.ppo"))
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                self.assertNotIn(name, ("run_episode", "execute", "native_update", "ppo_update", "step", "launch"))

    def test_readiness_is_false_without_bound_evidence(self):
        result = first_run.check_readiness(None, source={}, implementation={}, native_identity={}, native_runtime={}, map_identity={})
        self.assertIs(result["ready_to_request_launch"], False)

    def test_readiness_rejects_changed_artifact_source_and_missing_check(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(first_run, "ROOT", Path(temporary).resolve()):
            root = Path(temporary)
            artifact = root / "synthetic-check.txt"; artifact.write_text("Synthetic evidence, no native validation.\n")
            item = {"path": artifact.name, **first_run.file_identity(artifact)}
            identities = {"source": {"source_sha256": {"source.py": "a" * 64}},
                          "implementation": {"source.py": "a" * 64}, "native_identity": {"client": "b" * 64},
                          "native_runtime": {"verifier": "c" * 64}, "map_identity": {"sha256": "d" * 64}}
            block = {"interface": first_run.INTERFACE, "source_sha256": identities["source"]["source_sha256"],
                     "implementation_sha256": identities["implementation"],
                     **{key: identities[key] for key in ("native_identity", "native_runtime", "map_identity")},
                     "checks": {name: {"status": "passed", "artifacts": [item]} for name in first_run.READINESS_CHECKS},
                     "remaining_scope": ["Synthetic verifier-input fixture only."]}
            original = {"schema": "heroes-town-transfer-validation-v1", "first_training_readiness": block}
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps(original))
            self.assertTrue(first_run.check_readiness(evidence, **identities)["ready_to_request_launch"])
            for mutation in ("source", "missing", "escape", "artifact_bool_size"):
                changed = deepcopy(original); section = changed["first_training_readiness"]
                if mutation == "source": section["implementation_sha256"]["source.py"] = "e" * 64
                if mutation == "missing": del section["checks"]["natural_blue"]
                if mutation == "escape": section["checks"]["full_tests"]["artifacts"][0]["path"] = "../escape"
                if mutation == "artifact_bool_size": section["checks"]["full_tests"]["artifacts"][0]["bytes"] = True
                evidence.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): first_run.check_readiness(evidence, **identities)
            evidence.write_text(json.dumps(original)); artifact.write_text("Changed after review\n")
            with self.assertRaises(ValueError): first_run.check_readiness(evidence, **identities)


@unittest.skipIf(torch is None, "Optional PyTorch not installed")
class FirstRunCheckpointTests(unittest.TestCase):
    def test_random_v4_checkpoint_has_frozen_independent_opponent_and_no_optimizer(self):
        from experiments.learner.checkpoints import load_policy_checkpoint
        from experiments.learner.contextual_model import weights_sha256
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "random.pt"
                config = json.loads(first_run.CONFIG.read_text())
                rng = torch.get_rng_state().clone()
                result = first_run.prepare_checkpoint(path, config, {"test": "serialization-only"})
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                policy = load_policy_checkpoint(path, interface=first_run.INTERFACE)
                self.assertEqual(weights_sha256(policy.model), result["weights_sha256"])
                self.assertEqual(policy.provenance, "random_initialization")
                self.assertEqual(len(policy.opponent_pool.identities), 1)
                entry = policy.opponent_pool.get(policy.opponent_pool.identities[0][0])
                self.assertNotEqual(entry.weights_sha256, result["weights_sha256"])
                self.assertFalse(any(p.requires_grad for p in entry.model.parameters()))
                data = torch.load(path, weights_only=True)
                self.assertIsNone(data["resume"])
                self.assertEqual(data["contracts"]["history_schema"], "heroes-player-history-v4")
                self.assertEqual(data["contracts"]["action_schema"], "heroes-context-actions-v4")
                self.assertEqual(result["sha256"], first_run.file_identity(path)["sha256"])
                second = first_run.prepare_checkpoint(Path(temporary) / "second.pt", config, {"test": "serialization-only"})
                self.assertEqual(result["weights_sha256"], second["weights_sha256"])
                self.assertEqual(result["opponents"], second["opponents"])
                with self.assertRaises(FileExistsError): first_run.prepare_checkpoint(path, config, {})
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__": unittest.main()
