"""Synthetic CPU checkpoint/RNG continuity; no game process or training campaign."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
else:
    from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, weights_sha256
    from experiments.learner.checkpoints import (
        FrozenOpponentPool, TrainingProgress, load_checkpoint, load_policy_checkpoint,
        save_checkpoint, save_policy_checkpoint,
    )
    from experiments.learner.ppo import PPOConfig, ppo_update
    from test_contextual_ppo import synthetic_rollout


@unittest.skipIf(torch is None, "Optional PyTorch not installed")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(71); random.seed(13)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.pool = FrozenOpponentPool(); self.pool.add("frozen-initial", self.model)
        self.config = PPOConfig(epochs=1, normalize_advantages=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.generator = torch.Generator().manual_seed(14)
        self.python_rng = random.Random(15)
        self.sources = {"purpose": "synthetic", "source_revision": "test-fixture"}

    def save(self, path):
        return save_checkpoint(path, self.model, self.optimizer, checkpoint="synthetic-after-update",
            ppo_config=self.config.to_dict(), progress=TrainingProgress(1, 1, 2), source_metadata=self.sources,
            opponent_pool=self.pool, sampler_state={"next_episode": 2, "next_learner_seat": 1},
            generators={"minibatch": self.generator}, python_generators={"opponent": self.python_rng},
            at_rollout_boundary=True)

    def load(self, path, model, optimizer):
        return load_checkpoint(path, model, optimizer, expected_ppo_config=self.config.to_dict(),
            expected_source_metadata=self.sources, generators={"minibatch": self.generator},
            python_generators={"opponent": self.python_rng}, at_rollout_boundary=True)

    def test_inference_roundtrip_requires_versions_and_preserves_rng_without_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "initial.pt"
            save_policy_checkpoint(path, self.model, checkpoint="initial", source_metadata=self.sources,
                                   opponent_pool=self.pool, generators={"policy": self.generator})
            before_rng = torch.get_rng_state().clone()
            restored = load_policy_checkpoint(path, expected_source_metadata=self.sources)
            self.assertEqual(weights_sha256(restored.model), weights_sha256(self.model))
            self.assertEqual(restored.provenance, "random_initialization")
            self.assertEqual(restored.opponent_pool.identities, self.pool.identities)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            payload = torch.load(path, weights_only=True)
            self.assertIsNone(payload["resume"])
            self.assertEqual(set(payload["contracts"]), {"environment", "history_schema", "action_schema", "encoder_schema", "distribution_schema"})
            with self.assertRaises(FileExistsError):
                save_policy_checkpoint(path, self.model, checkpoint="initial", source_metadata=self.sources)
            with self.assertRaises(ValueError): load_policy_checkpoint(path, interface="adventure-windows-v1")
            for key in payload["contracts"]:
                altered = deepcopy(payload); altered["contracts"][key] = "unreviewed-version"
                changed = Path(temp) / (key + ".pt"); torch.save(altered, changed)
                with self.assertRaises(ValueError): load_policy_checkpoint(changed)

    def test_frozen_pool_is_independent_and_detects_mutation(self):
        old = self.pool.identities
        with torch.no_grad(): self.model.value.bias.add_(1)
        self.assertEqual(self.pool.identities, old)
        entry = self.pool.sample(self.python_rng)
        self.assertFalse(any(parameter.requires_grad for parameter in entry.model.parameters()))
        self.assertNotEqual(next(entry.model.parameters()).data_ptr(), next(self.model.parameters()).data_ptr())
        with torch.no_grad(): entry.model.value.bias.add_(1)
        with self.assertRaises(ValueError): self.pool.sample(self.python_rng)

    def test_optimizer_resume_reproduces_rng_next_update_and_frozen_opponent(self):
        first = synthetic_rollout(self.model)
        ppo_update(self.model, self.optimizer, (first,), config=self.config, generator=self.generator)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"; self.save(path)
            expected_draws = (random.random(), torch.rand(1).item(), self.python_rng.random(),
                              torch.rand(1, generator=self.generator).item())
            next_batch = synthetic_rollout(self.model, episode="synthetic-next")
            ppo_update(self.model, self.optimizer, (next_batch,), config=self.config, generator=self.generator)
            expected = weights_sha256(self.model)
            clone = ContextualActorCritic(self.model.config)
            optimizer = torch.optim.Adam(clone.parameters(), lr=0.9)
            resumed = self.load(path, clone, optimizer)
            actual_draws = (random.random(), torch.rand(1).item(), self.python_rng.random(),
                            torch.rand(1, generator=self.generator).item())
            self.assertEqual(actual_draws, expected_draws)
            self.assertEqual(resumed.progress, TrainingProgress(1, 1, 2))
            self.assertEqual(resumed.sampler_state["next_learner_seat"], 1)
            self.assertEqual(resumed.opponent_pool.identities, self.pool.identities)
            ppo_update(clone, optimizer, (next_batch,), config=self.config, generator=self.generator)
            self.assertEqual(weights_sha256(clone), expected)

    def test_foreign_reordered_and_malformed_optimizer_states_reject_transactionally(self):
        ppo_update(self.model, self.optimizer, (synthetic_rollout(self.model),), config=self.config, generator=self.generator)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"; self.save(path)
            clone = ContextualActorCritic(self.model.config)
            before = weights_sha256(clone)
            for optimizer in (self.optimizer, torch.optim.Adam(reversed(list(clone.parameters())))):
                with self.assertRaises(ValueError): self.load(path, clone, optimizer)
                self.assertEqual(weights_sha256(clone), before)
            original = torch.load(path, weights_only=True)
            for mode in ("shape", "nan", "step"):
                payload = deepcopy(original)
                state = next(iter(payload["resume"]["optimizer_state"]["state"].values()))
                if mode == "shape": state["exp_avg"] = torch.zeros(3, 2, 7)
                elif mode == "nan": state["exp_avg_sq"].fill_(float("nan"))
                else: state["step"] = torch.tensor([1.0])
                changed = Path(temp) / (mode + ".pt"); torch.save(payload, changed)
                optimizer = torch.optim.Adam(clone.parameters())
                rng_before = torch.get_rng_state().clone()
                with self.assertRaises(ValueError): self.load(changed, clone, optimizer)
                self.assertEqual(weights_sha256(clone), before)
                self.assertFalse(optimizer.state)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            with self.assertRaises(ValueError):
                save_checkpoint(Path(temp) / "foreign.pt", clone, self.optimizer, checkpoint="wrong-owner",
                    ppo_config=self.config.to_dict(), progress=TrainingProgress(), source_metadata=self.sources,
                    opponent_pool=self.pool, sampler_state={}, at_rollout_boundary=True)


if __name__ == "__main__": unittest.main()
