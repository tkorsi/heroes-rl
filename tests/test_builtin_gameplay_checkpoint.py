"""Synthetic v7 checkpoint/preflight checks; no native game or gameplay data."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import checkpoints as cp
from experiments.learner.ppo import PPOConfig, ppo_update
from tests.test_contextual_dialogs import history_at, panel


def equal_state(test, left, right):
    """Exact tensor/container comparison, including dtype rather than casting."""
    test.assertIs(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.device, right.device)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_state(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_state(test, a, b)
    else:
        test.assertEqual(left, right)


def synthetic_adam_step(model, optimizer, policy_rng, minibatch_rng):
    """Populate actual v7 actor/value Adam state from a labelled synthetic loss.

    This is an optimizer serialization fixture, not a PPO update or reward rule.
    The dedicated PPO numerical suite separately checks trajectory targets.
    """
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index in torch.randperm(2, generator=minibatch_rng).tolist():
        frames = history_at(panel(("chest", "school_magic")[index])).frames
        state = model.initial_state()
        for frame in m.encode_history(frames):
            output = model.forward_step(frame, state)
            state = output.state
        distribution = output.decision.distribution
        action = distribution.sample(policy_rng)
        target = torch.tensor(float(index + 1))
        losses.append(-distribution.log_prob(action) + (output.decision.value - target).square())
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


class BuiltinGameplayCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        torch.manual_seed(81)
        random.seed(82)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.pool = cp.FrozenOpponentPool()
        self.pool.add("archived-neural-opponent-not-executed", self.model)
        self.initial = self.root / "random-v7.pt"
        cp.save_policy_checkpoint(self.initial, self.model, checkpoint="synthetic-random-v7",
            source_metadata={"purpose": "synthetic-checkpoint-fixture"}, opponent_pool=self.pool,
            interface=c.DIALOGS_ENVIRONMENT)
        data = self.initial.read_bytes()
        self.initial_hash = hashlib.sha256(data).hexdigest()
        self.sources = {"purpose": "synthetic-format-test-no-gameplay",
            "plan_sha256": "a" * 64,
            "initial_checkpoint": {"path": str(self.initial), "sha256": self.initial_hash, "bytes": len(data)},
            "native_opponent": {"kind": "builtin", "module": "Nullkiller", "difficulty": 0,
                                "binary_sha256": "b" * 64},
            "archived_neural_pool_executed": False}
        self.generators = {"policy": torch.Generator().manual_seed(83),
                           "minibatch": torch.Generator().manual_seed(84)}
        self.python_generators = {"schedule": random.Random(85)}
        self.config = PPOConfig(epochs=1, normalize_advantages=False).to_dict()
        self.progress = cp.TrainingProgress(1, 2, 4)
        self.sampler_state = {"next_episode": 3, "next_learner_seat": 0}

    def save(self, path, **changes):
        options = dict(checkpoint="synthetic-v7-resume", ppo_config=self.config, progress=self.progress,
            source_metadata=self.sources, opponent_pool=self.pool, sampler_state=self.sampler_state,
            generators=self.generators, python_generators=self.python_generators,
            at_rollout_boundary=True, interface=c.DIALOGS_ENVIRONMENT, provenance=cp.BUILTIN_GAMEPLAY_PROVENANCE)
        options.update(changes)
        return cp.save_checkpoint(path, self.model, self.optimizer, **options)

    def load(self, path, model, optimizer, **changes):
        options = dict(expected_ppo_config=self.config, expected_source_metadata=self.sources,
            generators=self.generators, python_generators=self.python_generators, at_rollout_boundary=True,
            interface=c.DIALOGS_ENVIRONMENT, allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,))
        options.update(changes)
        return cp.load_checkpoint(path, model, optimizer, **options)

    def snapshot(self, model, optimizer):
        return deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "torch": torch.get_rng_state(), "python": random.getstate(),
            "generators": {k: v.get_state() for k, v in self.generators.items()},
            "python_generators": {k: v.getstate() for k, v in self.python_generators.items()}})

    def populated_checkpoint(self):
        synthetic_adam_step(self.model, self.optimizer, self.generators["policy"], self.generators["minibatch"])
        path = self.root / "resume.pt"
        self.save(path)
        self.assertTrue(self.optimizer.state)
        return path

    def test_exact_adam_rng_progress_and_next_v7_step_resume(self):
        path = self.populated_checkpoint()
        original = self.snapshot(self.model, self.optimizer)
        archived = deepcopy(self.pool.state_dict())

        def draw():
            return (random.random(), torch.rand(3), self.python_generators["schedule"].random(),
                torch.rand(3, generator=self.generators["policy"]),
                torch.rand(3, generator=self.generators["minibatch"]))

        expected_draws = draw()
        expected_loss = synthetic_adam_step(self.model, self.optimizer,
            self.generators["policy"], self.generators["minibatch"])
        expected_next = self.snapshot(self.model, self.optimizer)
        clone = m.ContextualActorCritic(self.model.config)
        optimizer = torch.optim.Adam(clone.parameters(), lr=0.9)
        resumed = self.load(path, clone, optimizer)
        equal_state(self, self.snapshot(clone, optimizer), original)
        self.assertEqual(resumed.progress, self.progress)
        self.assertEqual(resumed.sampler_state, self.sampler_state)
        equal_state(self, resumed.opponent_pool.state_dict(), archived)
        equal_state(self, draw(), expected_draws)
        self.assertEqual(synthetic_adam_step(clone, optimizer, self.generators["policy"],
            self.generators["minibatch"]), expected_loss)
        equal_state(self, self.snapshot(clone, optimizer), expected_next)
        equal_state(self, resumed.opponent_pool.state_dict(), archived)
        self.assertEqual(hashlib.sha256(self.initial.read_bytes()).hexdigest(), self.initial_hash)
        initial = cp.load_policy_checkpoint(self.initial, interface=c.DIALOGS_ENVIRONMENT)
        self.assertEqual(initial.provenance, "random_initialization")
        self.assertEqual(initial.weights_sha256, self.pool.identities[0][1])

    def test_new_provenance_needs_v7_plan_initial_binding_and_real_opponent_identity(self):
        mutations = (
            lambda x: x.pop("plan_sha256"), lambda x: x.update(plan_sha256="z" * 64),
            lambda x: x.update(plan_sha256=True), lambda x: x.pop("initial_checkpoint"),
            lambda x: x["initial_checkpoint"].update(path="relative.pt"),
            lambda x: x["initial_checkpoint"].update(bytes=True),
            lambda x: x["initial_checkpoint"].update(sha256="x" * 64),
            lambda x: x.pop("native_opponent"), lambda x: x["native_opponent"].update(module="archived-neural-model"),
            lambda x: x["native_opponent"].update(difficulty=False), lambda x: x["native_opponent"].update(difficulty=1),
            lambda x: x["native_opponent"].update(player=0), lambda x: x["native_opponent"].update(binary_sha256="bad"),
            lambda x: x.pop("archived_neural_pool_executed"), lambda x: x.update(archived_neural_pool_executed=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                metadata = deepcopy(self.sources); mutate(metadata)
                path = self.root / f"invalid-{index}.pt"
                with self.assertRaises(ValueError):
                    self.save(path, source_metadata=metadata)
                self.assertFalse(path.exists())
        for interface in (c.ARMIES_ENVIRONMENT, c.MANAGEMENT_ENVIRONMENT):
            with self.assertRaises(ValueError):
                self.save(self.root / "wrong-interface.pt", interface=interface)
        with self.assertRaises(ValueError):
            self.save(self.root / "old-provenance.pt", provenance="approved_gameplay")

    def test_exact_next_synthetic_v7_ppo_update_after_resume(self):
        from tests.test_builtin_ppo_numerics import synthetic_dialog_rollout

        config = PPOConfig(epochs=1, decisions_per_window=1, windows_per_minibatch=1,
                           normalize_advantages=False)
        self.config = config.to_dict()
        first = (synthetic_dialog_rollout(self.model, player=0, boundary="natural"),
                 synthetic_dialog_rollout(self.model, player=1, boundary="cutoff"))
        report = ppo_update(self.model, self.optimizer, first, config=config,
                            generator=self.generators["minibatch"])
        self.assertEqual((report.samples, report.optimizer_steps), (4, 4))
        self.assertEqual(report.purpose, "synthetic_optimizer_test")
        path = self.root / "synthetic-ppo-resume.pt"
        self.save(path)
        saved = self.snapshot(self.model, self.optimizer)
        archive = deepcopy(self.pool.state_dict())
        following = (synthetic_dialog_rollout(self.model, player=1, boundary="natural", family="arena"),
                     synthetic_dialog_rollout(self.model, player=0, boundary="cutoff", family="school_war"))
        expected = ppo_update(self.model, self.optimizer, following, config=config,
                               generator=self.generators["minibatch"])
        expected_state = self.snapshot(self.model, self.optimizer)
        clone = m.ContextualActorCritic(self.model.config)
        optimizer = torch.optim.Adam(clone.parameters(), lr=0.5)
        resumed = self.load(path, clone, optimizer)
        equal_state(self, self.snapshot(clone, optimizer), saved)
        observed = ppo_update(clone, optimizer, following, config=config,
                               generator=self.generators["minibatch"])
        self.assertEqual(observed, expected)
        equal_state(self, self.snapshot(clone, optimizer), expected_state)
        equal_state(self, resumed.opponent_pool.state_dict(), archive)
        self.assertEqual(hashlib.sha256(self.initial.read_bytes()).hexdigest(), self.initial_hash)

    def test_explicit_plan_opponent_provenance_and_rng_names_gate_load_before_mutation(self):
        path = self.populated_checkpoint()
        clone = m.ContextualActorCritic(self.model.config)
        optimizer = torch.optim.Adam(clone.parameters())
        alternate_plan = deepcopy(self.sources); alternate_plan["plan_sha256"] = "c" * 64
        alternate_opponent = deepcopy(self.sources); alternate_opponent["native_opponent"]["binary_sha256"] = "d" * 64
        alternate_initial = deepcopy(self.sources); alternate_initial["initial_checkpoint"]["sha256"] = "e" * 64
        for changes in ({"allowed_provenance": ("approved_gameplay",)}, {"expected_source_metadata": None},
                {"expected_source_metadata": alternate_plan}, {"expected_source_metadata": alternate_opponent},
                {"expected_source_metadata": alternate_initial}, {"interface": c.MANAGEMENT_ENVIRONMENT},
                {"generators": {"policy": self.generators["policy"]}}, {"python_generators": {}},
                {"at_rollout_boundary": False}):
            with self.subTest(changes=tuple(changes)):
                before = self.snapshot(clone, optimizer)
                with self.assertRaises(ValueError):
                    self.load(path, clone, optimizer, **changes)
                equal_state(self, self.snapshot(clone, optimizer), before)

    def test_serialized_dtype_corruption_cannot_be_hidden_by_adam_casting(self):
        path = self.populated_checkpoint()
        original = torch.load(path, weights_only=True)
        clone = m.ContextualActorCritic(self.model.config)
        optimizer = torch.optim.Adam(clone.parameters())
        for mode in ("model_dtype", "moment_dtype", "second_moment_dtype", "step_dtype", "negative_step", "nan"):
            with self.subTest(mode=mode):
                value = deepcopy(original)
                state = next(iter(value["resume"]["optimizer_state"]["state"].values()))
                if mode == "model_dtype":
                    key = next(iter(value["model_state"])); value["model_state"][key] = value["model_state"][key].double()
                elif mode == "moment_dtype": state["exp_avg"] = state["exp_avg"].double()
                elif mode == "second_moment_dtype": state["exp_avg_sq"] = state["exp_avg_sq"].double()
                elif mode == "step_dtype": state["step"] = state["step"].double()
                elif mode == "negative_step": state["step"].fill_(-1)
                else: state["exp_avg_sq"].fill_(float("nan"))
                altered = self.root / (mode + ".pt"); torch.save(value, altered)
                before = self.snapshot(clone, optimizer)
                with self.assertRaises(ValueError):
                    self.load(altered, clone, optimizer)
                equal_state(self, self.snapshot(clone, optimizer), before)

    def test_policy_export_is_explicit_and_preserves_sampler_rng(self):
        self.populated_checkpoint()
        path = self.root / "policy.pt"
        cp.save_policy_checkpoint(path, self.model, checkpoint="synthetic-export",
            provenance=cp.BUILTIN_GAMEPLAY_PROVENANCE, source_metadata=self.sources,
            opponent_pool=self.pool, interface=c.DIALOGS_ENVIRONMENT)
        before = self.snapshot(self.model, self.optimizer)
        with self.assertRaises(ValueError):
            cp.load_policy_checkpoint(path, interface=c.DIALOGS_ENVIRONMENT)
        with self.assertRaises(ValueError):
            cp.load_policy_checkpoint(path, interface=c.DIALOGS_ENVIRONMENT,
                allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,))
        loaded = cp.load_policy_checkpoint(path, interface=c.DIALOGS_ENVIRONMENT,
            expected_source_metadata=self.sources, allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,))
        self.assertEqual(loaded.weights_sha256, m.weights_sha256(self.model))
        self.assertEqual(loaded.source_metadata["native_opponent"], self.sources["native_opponent"])
        self.assertFalse(loaded.source_metadata["archived_neural_pool_executed"])
        equal_state(self, self.snapshot(self.model, self.optimizer), before)

    def test_double_model_cannot_create_new_provenance_checkpoint(self):
        self.model.double()
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            self.save(self.root / "double.pt")
        self.assertFalse((self.root / "double.pt").exists())


if __name__ == "__main__":
    unittest.main()
