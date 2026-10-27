"""Artifact-bound guard support and synthetic-only PPO; no native games."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import torch

from experiments.learner import interaction_guard as guard
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig
from experiments.learner.contextual_rollout import ContextualRollout, behavior_guard_version
from experiments.learner.ppo import PPOConfig, ppo_update
from tests.test_contextual_ppo import synthetic_rollout


class InteractionGuardRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(851)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))

    def collect(self, version):
        with patch.object(guard, 'ACTIVE_VERSION', version):
            return synthetic_rollout(self.model)

    def test_saved_version_controls_decode_even_if_default_changes(self):
        for version, other in ((None, guard.VERSION), (guard.VERSION, None)):
            with self.subTest(version=version):
                rollout = self.collect(version)
                with patch.object(guard, 'ACTIVE_VERSION', other):
                    restored = ContextualRollout.from_dict(rollout.to_dict())
                self.assertEqual(restored, rollout)
                self.assertEqual(behavior_guard_version(restored.decisions), version)
                self.assertEqual([s.reward for s in restored.samples], [0, 1])

    def test_support_is_rederived_not_trusted_from_serialization(self):
        raw = self.collect(guard.VERSION).to_dict()
        forged = deepcopy(raw)
        forged['decisions'][0]['support']['interaction_guard']['movable_heroes'] = []
        with self.assertRaisesRegex(ValueError, 'support disagrees'):
            ContextualRollout.from_dict(forged)
        mixed = deepcopy(raw)
        del mixed['decisions'][0]['support']['interaction_guard']
        with self.assertRaisesRegex(ValueError, 'Mixed serialized'):
            ContextualRollout.from_dict(mixed)
        unknown = deepcopy(raw)
        unknown['decisions'][0]['support']['interaction_guard']['version'] = 'unknown'
        with self.assertRaises(ValueError): ContextualRollout.from_dict(unknown)

    def test_ppo_reconstructs_artifact_bound_support_not_active_default(self):
        for version, other in ((None, guard.VERSION), (guard.VERSION, None)):
            with self.subTest(version=version):
                rollout = self.collect(version)
                a, b = deepcopy(self.model), deepcopy(self.model)
                oa, ob = torch.optim.Adam(a.parameters(), lr=0.0003), torch.optim.Adam(b.parameters(), lr=0.0003)
                config = PPOConfig(epochs=1, decisions_per_window=2, windows_per_minibatch=1)
                with patch.object(guard, 'ACTIVE_VERSION', version):
                    ppo_update(a, oa, (rollout,), config=config, generator=torch.Generator().manual_seed(391))
                with patch.object(guard, 'ACTIVE_VERSION', other):
                    ppo_update(b, ob, (rollout,), config=config, generator=torch.Generator().manual_seed(391))
                for name, value in a.state_dict().items():
                    self.assertTrue(torch.equal(value, b.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
