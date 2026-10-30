"""Exact optimization equivalence on disposable synthetic models, no gameplay."""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history
from experiments.learner.ppo import PPOConfig, ppo_update
from tests.test_contextual_ppo import synthetic_rollout


def uncached_prefixes(model, inputs, stops):
    """Pre-optimization algorithm: independently replay every full prefix."""
    result = {}
    with torch.no_grad():
        for stop in stops:
            state = model.initial_state()
            for frame in inputs[:stop]:
                state = model.forward_step(frame, state).state
            result[stop] = state.detach()
    return result


class PrefixPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(8713)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.rollout = synthetic_rollout(self.model, turns=8)
        self.inputs = encode_history(self.rollout.history)

    def assert_tensors_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tensors_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tensors_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_every_boundary_matches_in_train_and_eval_without_changing_modes(self):
        stops = tuple(range(len(self.inputs) + 1))
        for train in (True, False):
            self.model.train(train)
            modes = [m.training for m in self.model.modules()]
            actual = self.model.prefix_states(self.inputs, stops)
            expected = uncached_prefixes(self.model, self.inputs, stops)
            self.assert_tensors_equal(actual, expected)
            self.assertTrue(all(not state.requires_grad for state in actual.values()))
            self.assertEqual(modes, [m.training for m in self.model.modules()])

    def test_each_call_rebuilds_after_in_place_changes_without_version_counter(self):
        stops = (len(self.inputs),)
        before = self.model.prefix_states(self.inputs, stops)
        # .data writes deliberately evade tensor version counters. The local
        # memo still cannot return anything from the preceding weight version.
        self.model.text_gru.bias_ih_l0.data.add_(0.15)
        self.model.gru.bias_hh.data.add_(0.12)
        after = self.model.prefix_states(self.inputs, stops)
        self.assertFalse(torch.equal(before[stops[0]], after[stops[0]]))
        self.assert_tensors_equal(after, uncached_prefixes(self.model, self.inputs, stops))

    def test_hooks_and_custom_forward_take_uncached_path(self):
        seen = []
        hook = self.model.text_gru.register_forward_hook(lambda *args: seen.append(1))
        try:
            self.assertFalse(self.model._can_cache_prefix())
            actual = self.model.prefix_states(self.inputs, (len(self.inputs),))
            calls = len(seen)
            seen.clear()
            expected = uncached_prefixes(self.model, self.inputs, (len(self.inputs),))
            self.assertEqual(calls, len(seen))
            self.assert_tensors_equal(actual, expected)
        finally:
            hook.remove()
        self.model.forward_step = self.model.forward_step
        self.assertFalse(self.model._can_cache_prefix())

    def test_losses_gradients_adam_state_weights_and_rng_match_across_updates(self):
        optimized, reference = deepcopy(self.model), deepcopy(self.model)
        reference.prefix_states = MethodType(uncached_prefixes, reference)
        config = PPOConfig(epochs=2, decisions_per_window=2, windows_per_minibatch=3)
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.0003) for model in (optimized, reference)]
        generators = [torch.Generator().manual_seed(1729) for _ in range(2)]
        # Two independently admitted updates also exercise nonempty Adam state.
        for update in range(2):
            rollouts = (synthetic_rollout(optimized, turns=8, episode=f"first-{update}"),
                        synthetic_rollout(optimized, turns=5, episode=f"cutoff-{update}", boundary="cutoff"))
            reports = [ppo_update(model, optimizer, rollouts, config=config, generator=generator)
                       for model, optimizer, generator in zip((optimized, reference), optimizers, generators)]
            self.assertEqual(replace(reports[0], timings_seconds=None), replace(reports[1], timings_seconds=None))
            self.assert_tensors_equal(optimized.state_dict(), reference.state_dict())
            self.assert_tensors_equal([p.grad for p in optimized.parameters()], [p.grad for p in reference.parameters()])
            self.assert_tensors_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
            self.assert_tensors_equal(generators[0].get_state(), generators[1].get_state())
            for report in reports:
                self.assertIsNone(report.timings_seconds)


if __name__ == "__main__":
    unittest.main()
