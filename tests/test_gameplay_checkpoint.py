"""Synthetic serialization/admission checks; no game or gameplay update."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.checkpoints import (
    FrozenOpponentPool, TrainingProgress, load_checkpoint, save_checkpoint,
)
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, ARMIES_MODEL_SCHEMA, weights_sha256
from experiments.learner.ppo import PPOConfig, _update
from test_contextual_ppo import synthetic_rollout


class GameplayCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_provenance_requires_plan_identity_and_explicit_load_opt_in(self):
        model = ContextualActorCritic(ModelConfig(schema=ARMIES_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        pool = FrozenOpponentPool()
        pool.add("synthetic-initial-opponent", model)
        config = PPOConfig().to_dict()
        metadata = {"plan_sha256": "a" * 64, "purpose": "synthetic-format-test-no-gameplay"}
        kwargs = dict(checkpoint="synthetic-format", ppo_config=config, progress=TrainingProgress(),
                      source_metadata=metadata, opponent_pool=pool, sampler_state={},
                      at_rollout_boundary=True, interface=c.ARMIES_ENVIRONMENT, provenance="approved_gameplay")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "format.pt"
            with self.assertRaisesRegex(ValueError, "optimizer provenance"):
                save_checkpoint(path, model, optimizer, **{**kwargs, "provenance": "random_initialization"})
            self.assertFalse(path.exists())
            for invalid in ({"purpose": "synthetic"}, {"plan_sha256": True}, {"plan_sha256": "z" * 64}):
                with self.assertRaises(ValueError):
                    save_checkpoint(path, model, optimizer, **{**kwargs, "source_metadata": invalid})
                self.assertFalse(path.exists())
            save_checkpoint(path, model, optimizer, **kwargs)
            clone = ContextualActorCritic(model.config)
            clone_optimizer = torch.optim.Adam(clone.parameters(), lr=0.0003)
            load_args = dict(expected_ppo_config=config, expected_source_metadata=metadata,
                             at_rollout_boundary=True, interface=c.ARMIES_ENVIRONMENT)
            before = weights_sha256(clone)
            with self.assertRaises(ValueError): load_checkpoint(path, clone, clone_optimizer, **load_args)
            self.assertEqual(before, weights_sha256(clone))
            restored = load_checkpoint(path, clone, clone_optimizer, **load_args,
                                       allowed_provenance=("approved_gameplay",))
            self.assertEqual(restored.progress, TrainingProgress())
            self.assertEqual(weights_sha256(clone), weights_sha256(model))
            from experiments.learner.gameplay import assert_same_state
            assert_same_state(model.state_dict(), clone.state_dict())
            assert_same_state(optimizer.state_dict(), clone_optimizer.state_dict())
            self.assertFalse(optimizer.state)
            invalid = torch.load(path, weights_only=True)
            invalid["provenance"] = "random_initialization"
            invalid_path = Path(temp) / "invalid-purpose.pt"
            torch.save(invalid, invalid_path)
            with self.assertRaisesRegex(ValueError, "optimizer provenance"):
                load_checkpoint(invalid_path, clone, clone_optimizer, **load_args,
                                allowed_provenance=("random_initialization",))

    def test_resume_comparison_recurses_into_ordered_tensor_mappings(self):
        from experiments.learner.gameplay import assert_same_state
        state = {"model": OrderedDict(weight=torch.tensor([1.0, 2.0])),
                 "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.1, 0.2])}}},
                 "rng": [torch.tensor([7, 8]), (1, None)]}
        assert_same_state(state, deepcopy(state))
        changed = deepcopy(state)
        changed["model"]["weight"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "Resume tensor differs"):
            assert_same_state(state, changed)
        for changed_model in (OrderedDict(other=torch.tensor([1.0, 2.0])),
                              {"weight": torch.tensor([1.0, 2.0])}):
            with self.subTest(container=type(changed_model).__name__):
                changed = deepcopy(state)
                changed["model"] = changed_model
                with self.assertRaisesRegex(ValueError, "Resume mapping differs"):
                    assert_same_state(state, changed)

    def test_deadline_after_one_synthetic_step_restores_complete_state(self):
        from experiments.learner.gameplay import BudgetExpired, UpdateRolledBack, _transactional_update
        model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        generator = torch.Generator().manual_seed(91)
        before = weights_sha256(model)
        global_rng, sampler_rng = torch.get_rng_state().clone(), generator.get_state().clone()
        calls = 0

        def remaining():
            nonlocal calls
            calls += 1
            if calls == 3: raise BudgetExpired("synthetic deadline")
            return 1.0

        with self.assertRaises(UpdateRolledBack) as caught:
            _transactional_update(model, optimizer, (rollout,),
                config=PPOConfig(epochs=1, decisions_per_window=1, windows_per_minibatch=1),
                generator=generator, remaining=remaining, max_optimizer_steps=2)
        self.assertEqual(caught.exception.completed_steps, 1)
        self.assertEqual(weights_sha256(model), before)
        self.assertFalse(optimizer.state)
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
        self.assertTrue(torch.equal(generator.get_state(), sampler_rng))

    def test_optimizer_step_cap_refuses_a_partial_update_before_changing_weights(self):
        model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        before = weights_sha256(model)
        with self.assertRaisesRegex(ValueError, "optimizer-step cap"):
            _update(model, optimizer, (rollout,),
                    config=PPOConfig(epochs=1, decisions_per_window=1, windows_per_minibatch=1),
                    generator=torch.Generator().manual_seed(4), max_optimizer_steps=1)
        self.assertEqual(weights_sha256(model), before)
        self.assertFalse(optimizer.state)


if __name__ == "__main__":
    unittest.main()
