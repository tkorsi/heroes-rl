"""Explicit profile isolation and synthetic socket capture; no native gameplay."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from experiments.headless import builtin_evidence as e
from experiments.learner import contextual as c
from tests import test_builtin_rollout as sockets
from tests.test_builtin_env import initial, manifest_for


class BuiltinProfileTests(unittest.TestCase):
    def test_legacy_profile_is_exact_and_older_interfaces_are_rejected(self):
        self.assertEqual(e.profile_for(c.DIALOGS_ENVIRONMENT), c.DIALOGS_PROFILE)
        for value in (c.ENVIRONMENT, c.TOWN_SCHEMA, c.MANAGEMENT_ENVIRONMENT, None, "latest"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                e.profile_for(value)

    def test_mixed_manifest_versions_fail_before_history_admission(self):
        history = initial()
        history.emit("cutoff", decision_seq=history.decision.seq, reason="collection_window")
        good = manifest_for(history)
        self.assertTrue(e.validate_manifest_history(history, good)["consistent"])
        for field, value in (("environment", "adventure-capabilities-v1"),
                             ("history_schema", "heroes-player-history-v8"),
                             ("action_schema", "heroes-context-actions-v8")):
            bad = deepcopy(good)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                e.validate_manifest_history(history, bad)

    @unittest.skipUnless(hasattr(c, "EXPANDED_PROFILE"), "Requires coordinator's explicit v8 contract")
    def test_v8_never_reads_a_v7_history_key(self):
        profile = c.EXPANDED_PROFILE
        history = initial(profile=profile)
        history.emit("cutoff", decision_seq=history.decision.seq, reason="collection_window")
        manifest = manifest_for(history)
        manifest.update(environment=profile.environment, history_schema=profile.history_schema,
                        action_schema=profile.action_schema)
        with self.assertRaises(ValueError):
            e.validate_manifest_history(history, manifest)
        manifest["players"]["red"][profile.history_key] = manifest["players"]["red"].pop("history_v7")
        self.assertTrue(e.validate_manifest_history(history, manifest)["consistent"])


    @unittest.skipUnless(hasattr(c, "EXPANDED_PROFILE"), "Requires coordinator's explicit v8 contract")
    def test_v8_named_tensors_match_but_checkpoint_contract_remains_distinct(self):
        from experiments.learner import checkpoints as cp, contextual_model as m
        from experiments.learner.gameplay import assert_same_state
        old = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        new = m.ContextualActorCritic(replace(old.config, schema=m.model_schema_for(c.EXPANDED_SCHEMA)))
        self.assertEqual([(n, tuple(p.shape), p.dtype) for n, p in old.named_parameters()],
                         [(n, tuple(p.shape), p.dtype) for n, p in new.named_parameters()])
        new.load_state_dict(old.state_dict(), strict=True)
        assert_same_state(new.state_dict(), old.state_dict())
        self.assertNotEqual(m.weights_sha256(old), m.weights_sha256(new))
        metadata = {"plan_sha256": "a" * 64,
            "initial_checkpoint": {"path": "/synthetic/parent.pt", "sha256": "b" * 64, "bytes": 1},
            "native_opponent": {"kind": "builtin", "module": "Nullkiller", "difficulty": 0, "binary_sha256": "c" * 64},
            "archived_neural_pool_executed": False}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v8.pt"
            cp.save_policy_checkpoint(path, new, checkpoint="synthetic-v8", source_metadata=metadata,
                provenance=cp.BUILTIN_GAMEPLAY_PROVENANCE, interface=c.EXPANDED_ENVIRONMENT)
            options = dict(expected_source_metadata=metadata, allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,))
            loaded = cp.load_policy_checkpoint(path, interface=c.EXPANDED_ENVIRONMENT, **options)
            assert_same_state(loaded.model.state_dict(), old.state_dict())
            with self.assertRaises(ValueError):
                cp.load_policy_checkpoint(path, interface=c.DIALOGS_ENVIRONMENT, **options)


@unittest.skipUnless(hasattr(c, "EXPANDED_PROFILE"), "Requires coordinator's explicit v8 contract")
class ExpandedCollectorIntegrationTests(sockets.BuiltinCollectorIntegrationTests):
    interface = "adventure-capabilities-v1"


if __name__ == "__main__":
    unittest.main()
