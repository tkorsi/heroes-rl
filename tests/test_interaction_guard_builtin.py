"""Guarded builtin record reconstruction on synthetic socket transport only."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.learner import interaction_guard as g
from experiments.learner.builtin_rollout import load_builtin_episode, audit_persisted_builtin_policy
from tests import test_builtin_rollout as builtin_fixture


class GuardedBuiltinRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, root, version):
        with patch.object(g, "ACTIVE_VERSION", version):
            return builtin_fixture.BuiltinCollectorIntegrationTests().run_case(root, 0, "cutoff")

    def test_guarded_builtin_roundtrip_reconstructs_without_active_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            got, folder, model, opponent, _, _ = self.fixture(Path(tmp), g.VERSION)
            self.assertTrue(all(item.support.guard is not None for item in got.decisions))
            with patch.object(g, "ACTIVE_VERSION", None):
                self.assertEqual(load_builtin_episode(got.to_dict()), got)
                audit = audit_persisted_builtin_policy(folder, model, got.checkpoint, 17, opponent)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["actions_resampled"], 1)
            self.assertFalse(got.optimization_eligible)
            self.assertEqual(got.training_samples, 0)

    def test_guarded_builtin_rejects_serialized_and_typed_support_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            got, _, _, _, _, _ = self.fixture(Path(tmp), g.VERSION)
            for mutate in (
                lambda support: support["interaction_guard"].update(movable_heroes=["invented"]),
                lambda support: support["interaction_guard"].update(blocked_navigation=[["open_hero", "invented"]]),
                lambda support: support["interaction_guard"].update(version="unknown"),
                lambda support: support.pop("interaction_guard"),
                lambda support: support.update(distribution_schema="heroes-context-policy-v5"),
            ):
                raw = got.to_dict()
                mutate(raw["decisions"][0]["support"])
                with self.assertRaises(ValueError): load_builtin_episode(raw)
            saved = got.decisions[0]
            changed = replace(saved.support.guard, movable_heroes=("invented",))
            forged = replace(saved, support=replace(saved.support, guard=changed))
            with self.assertRaises(ValueError): replace(got, decisions=(forged,))

    def test_historical_builtin_roundtrip_ignores_new_collection_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            got, folder, model, opponent, _, _ = self.fixture(Path(tmp), None)
            raw = got.to_dict()
            self.assertTrue(all("interaction_guard" not in item["support"] for item in raw["decisions"]))
            with patch.object(g, "ACTIVE_VERSION", g.VERSION):
                restored = load_builtin_episode(raw)
                audit = audit_persisted_builtin_policy(folder, model, got.checkpoint, 17, opponent)
            self.assertEqual(restored, got)
            self.assertEqual(restored.to_dict(), raw)
            self.assertTrue(audit["passed"])


if __name__ == "__main__":
    unittest.main()
