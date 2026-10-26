"""Phase-2 PPO fast paths: exact equivalence, mode restoration, poisoning and the separate experiment."""
from copy import deepcopy
from dataclasses import replace
import pickle
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import ppo
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history
from experiments.learner.ppo import PPOConfig, _evaluate_from_state, _verify_behavior, _verify_behavior_frames, ppo_update
from tests.test_contextual_ppo import synthetic_rollout
from tests.test_ppo_prefix_performance import uncached_prefixes


def tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tensors_equal(a, b) for a, b in zip(left, right))
    return left == right


class PhaseTwoAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2610)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.rollouts = (synthetic_rollout(self.model, turns=7, episode="phase2-a"),
                         synthetic_rollout(self.model, turns=4, episode="phase2-b", boundary="cutoff"))
        self.inputs = tuple(encode_history(item.history) for item in self.rollouts)
        self.addCleanup(setattr, ppo, "EXPERIMENTS", ppo.EXPERIMENTS)

    def modes(self, model=None):
        return [m.training for m in (model or self.model).modules()]

    def poisoned(self, factor):
        """Lower one stored probability (and its first conditional) by ``factor`` times the check's tolerance."""
        item = self.rollouts[0]
        record = item.decisions[1]
        delta = factor * max(2e-5 * abs(record.log_prob), 2e-5)
        first = replace(record.conditionals[0], log_prob=record.conditionals[0].log_prob - delta)
        bad = replace(record, log_prob=record.log_prob - delta, conditionals=(first, *record.conditionals[1:]))
        decisions = tuple(bad if r is record else r for r in item.decisions)
        samples = tuple(replace(s, decision=bad) if s.decision is record else s for s in item.samples)
        return (replace(item, decisions=decisions, samples=samples), self.rollouts[1])

    # Exact behavior verification in evaluation mode (E1).

    def test_behavior_check_uses_eval_memos_and_restores_mixed_modes(self):
        self.model.train()
        self.model.text_gru.eval()  # a mixed configuration must come back exactly
        before = self.modes()
        _verify_behavior(self.model, self.rollouts, self.inputs)
        self.assertEqual(before, self.modes())
        self.assertIn("_leaf_memo_state", self.model.__dict__)  # the exact memo served the check
        # The reference algorithm (train mode, no memo) accepts the same records.
        _verify_behavior_frames(self.model, self.rollouts, self.inputs)

    def test_poisoned_behavior_probability_is_rejected_by_both_paths_and_modes_survive(self):
        self.model.train()
        before = self.modes()
        for factor in (1.5, 50):
            rollouts = self.poisoned(factor)
            with self.assertRaisesRegex(ValueError, "Stored behavior probability differs"):
                _verify_behavior(self.model, rollouts, self.inputs)
            self.assertEqual(before, self.modes())
            with self.assertRaisesRegex(ValueError, "Stored behavior probability differs"):
                _verify_behavior_frames(self.model, rollouts, self.inputs)
        # A warm memo from an accepted pass cannot launder a later poisoned record.
        _verify_behavior(self.model, self.rollouts, self.inputs)
        with self.assertRaisesRegex(ValueError, "Stored behavior probability differs"):
            _verify_behavior(self.model, self.poisoned(1.5), self.inputs)

    def test_stale_memo_after_weight_change_cannot_accept_old_records(self):
        self.model.train()
        _verify_behavior(self.model, self.rollouts, self.inputs)
        with torch.no_grad():
            self.model.policy_query.weight.data.add_(0.25)  # evades version counters
            self.model.text_gru.bias_hh_l0.data.add_(0.25)
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            _verify_behavior(self.model, self.rollouts, self.inputs)

    def test_hooks_keep_the_reference_train_mode_path(self):
        seen = []
        hook = self.model.text_gru.register_forward_hook(lambda module, *args: seen.append(module.training))
        try:
            self.model.train()
            _verify_behavior(self.model, self.rollouts, self.inputs)
            self.assertTrue(seen and all(seen))
            self.assertNotIn("_leaf_memo_state", self.model.__dict__)
        finally:
            hook.remove()

    # Exact prefix reconstruction with the call-local node memo.

    def test_prefix_node_memo_matches_uncached_states_after_weight_changes(self):
        stops = tuple(range(len(self.inputs[0]) + 1))
        for _ in range(2):
            for train in (True, False):
                self.model.train(train)
                self.assertTrue(tensors_equal(self.model.prefix_states(self.inputs[0], stops),
                                              uncached_prefixes(self.model, self.inputs[0], stops)))
            with torch.no_grad():
                self.model.tree[0].bias.data.add_(0.1)

    def test_exact_update_is_unchanged_when_the_experiment_is_not_selected(self):
        self.assertEqual(ppo.EXPERIMENTS, frozenset())
        optimized, reference = deepcopy(self.model), deepcopy(self.model)
        reference.prefix_states = lambda inputs, stops: uncached_prefixes(reference, inputs, stops)
        import experiments.learner.ppo as kernel
        config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2)
        results = []
        for model in (optimized, reference):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            generator = torch.Generator().manual_seed(44)
            rollouts = tuple(replace(item) for item in self.rollouts)
            if model is reference:
                # Reference behavior check: the original train-mode loop without memos.
                original = kernel._verify_behavior
                kernel._verify_behavior = _verify_behavior_frames
                try:
                    report = ppo_update(model, optimizer, rollouts, config=config, generator=generator)
                finally:
                    kernel._verify_behavior = original
            else:
                report = ppo_update(model, optimizer, rollouts, config=config, generator=generator)
            results.append((replace(report, timings_seconds=None), model.state_dict(),
                            [p.grad for p in model.parameters()], optimizer.state_dict(), generator.get_state()))
            self.assertNotIn("experiment_shared_graph_encodings", report.timings_seconds)
        self.assertEqual(results[0][0], results[1][0])
        for left, right in zip(results[0][1:], results[1][1:]):
            self.assertTrue(tensors_equal(left, right))

    # The separate numerical experiment (never a default).

    def test_shared_graph_forward_values_are_bitwise_equal_and_gradients_close(self):
        self.model.train()
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        outputs, grads = [], []
        for shared in (False, True):
            self.model.zero_grad(set_to_none=True)
            context = self.model.shared_graph_encodings() if shared else None
            if context is not None:
                self.assertTrue(context.__enter__())
            evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            loss = sum(e.log_prob + e.value for e in evaluated)
            loss.backward()
            if context is not None:
                context.__exit__(None, None, None)
            self.assertNotIn("_graph_memo_state", self.model.__dict__)
            outputs.append([(e.log_prob.detach(), e.value.detach()) for e in evaluated])
            grads.append([None if p.grad is None else p.grad.clone() for p in self.model.parameters()])
        self.assertTrue(tensors_equal(outputs[0], outputs[1]))
        self.assertTrue(any(g is not None for g in grads[0]))
        for left, right in zip(*grads):
            self.assertEqual(left is None, right is None)
            if left is not None:
                torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)

    def test_experiment_selection_is_explicit_recorded_and_cleared_on_failure(self):
        ppo.EXPERIMENTS = frozenset({"shared_graph_encodings"})
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        generator = torch.Generator().manual_seed(3)
        report = ppo_update(self.model, optimizer, self.rollouts, config=PPOConfig(), generator=generator)
        self.assertEqual(report.timings_seconds["experiment_shared_graph_encodings"], 1.0)
        self.assertNotIn("_graph_memo_state", self.model.__dict__)
        # A failure inside a minibatch never leaves shared encodings behind.
        def fail():
            self.assertIn("_graph_memo_state", self.model.__dict__)
            raise RuntimeError("injected")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            ppo._update(self.model, optimizer, (synthetic_rollout(self.model, episode="fresh"),), config=PPOConfig(),
                        generator=generator, before_step=fail)
        self.assertNotIn("_graph_memo_state", self.model.__dict__)

    def test_experiment_is_inert_with_hooks_and_never_pickled(self):
        hook = self.model.tree.register_forward_hook(lambda *args: None)
        try:
            with self.model.shared_graph_encodings() as active:
                self.assertFalse(active)
                self.assertNotIn("_graph_memo_state", self.model.__dict__)
        finally:
            hook.remove()
        with self.model.shared_graph_encodings() as active:
            self.assertTrue(active)
            copy = pickle.loads(pickle.dumps(self.model))
            self.assertNotIn("_graph_memo_state", copy.__dict__)
            self.assertNotIn("_graph_memo_state", deepcopy(self.model).__dict__)
        self.assertNotIn("_graph_memo_state", self.model.__dict__)

    def test_environment_cannot_select_the_experiment(self):
        import inspect
        source = inspect.getsource(ppo)
        self.assertNotIn("os.environ", source)
        self.assertNotIn("getenv", source)


if __name__ == "__main__":
    unittest.main()
