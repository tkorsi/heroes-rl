"""Exact replay of repeated text leaves: bit-identical updates, sensitivity, fail-closed guards (no gameplay)."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import pickle
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import contextual_model as cm
from experiments.learner import ppo
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history
from experiments.learner.ppo import PPOConfig, _evaluate_from_state, ppo_update
from tests.test_contextual_ppo import synthetic_rollout
from tests.test_ppo_phase2_acceleration import tensors_equal


def bits(value):
    """Exact bits, so that signed zeros and NaN payloads count."""
    if isinstance(value, torch.Tensor):
        return (value.dtype, value.shape, value.view(torch.int32) if value.dtype == torch.float32 else value)
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    return value


def bits_equal(left, right):
    return tensors_equal(bits(left), bits(right))


class Opaque(torch.autograd.Function):
    """Passes its input forward and sends no gradient back (an undefined incoming gradient)."""

    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return None


class Text(str):
    """A str subclass: its leaves take the reference path, never the replay."""


class ReplayedTextLeafTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2710)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.rollouts = (synthetic_rollout(self.model, turns=7, episode="replay-a"),
                         synthetic_rollout(self.model, turns=4, episode="replay-b", boundary="cutoff"))
        self.inputs = tuple(encode_history(item.history) for item in self.rollouts)
        self.addCleanup(setattr, ppo, "REPLAY_TEXT_LEAVES", ppo.REPLAY_TEXT_LEAVES)
        self.addCleanup(setattr, ppo, "TEXT_REPLAY_WORKERS", ppo.TEXT_REPLAY_WORKERS)
        self.addCleanup(setattr, ppo, "EXPERIMENTS", ppo.EXPERIMENTS)

    def update(self, model, replay, config, seed, workers=0):
        ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS = replay, workers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        generator = torch.Generator().manual_seed(seed)
        occurrences, apply = [], cm._TextOccurrence.apply

        def counting(*args):
            occurrences.append(args[0]); return apply(*args)
        with patch.object(cm._TextOccurrence, "apply", counting):
            report = ppo_update(model, optimizer, tuple(replace(item) for item in self.rollouts), config=config,
                                generator=generator)
        return (len(occurrences), replace(report, timings_seconds=None), model.state_dict(),
                [p.grad for p in model.parameters()], optimizer.state_dict(), generator.get_state())

    def window_gradients(self, context):
        self.model.zero_grad(set_to_none=True)
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with context:
            evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            loss = sum(e.log_prob + 0.5 * e.value for e in evaluated)
            loss.backward()
        return ([(e.log_prob.detach(), e.value.detach()) for e in evaluated],
                [None if p.grad is None else p.grad.clone() for p in self.model.parameters()])

    def test_updates_are_bit_identical_with_and_without_replay(self):
        for seed, config in ((44, PPOConfig(decisions_per_window=2, windows_per_minibatch=2)),
                             (45, PPOConfig(decisions_per_window=1, windows_per_minibatch=1)),
                             (46, PPOConfig(decisions_per_window=3, windows_per_minibatch=8, epochs=3))):
            replayed, reference = deepcopy(self.model), deepcopy(self.model)
            left = self.update(replayed, True, config, seed)
            right = self.update(reference, False, config, seed)
            self.assertGreater(left[0], 0)
            self.assertEqual(right[0], 0)
            self.assertEqual(left[1], right[1])
            for a, b in zip(left[2:], right[2:]):
                self.assertTrue(bits_equal(a, b))
            self.assertNotIn("_leaf_replay_state", replayed.__dict__)

    def test_updates_with_deferred_replays_on_worker_threads_are_bit_identical(self):
        for workers, seed, config in ((2, 47, PPOConfig(decisions_per_window=2, windows_per_minibatch=2)),
                                      (3, 48, PPOConfig(decisions_per_window=3, windows_per_minibatch=8, epochs=3)),
                                      (1, 49, PPOConfig(decisions_per_window=1, windows_per_minibatch=4))):
            replayed, reference = deepcopy(self.model), deepcopy(self.model)
            left = self.update(replayed, True, config, seed, workers)
            right = self.update(reference, False, config, seed)
            self.assertGreater(left[0], 0)
            self.assertEqual(left[1], right[1])
            for a, b in zip(left[2:], right[2:]):
                self.assertTrue(bits_equal(a, b))
            self.assertNotIn("_text_replay_deferred", replayed.__dict__)

    def deferred_window_gradients(self, workers, extra=None):
        self.model.zero_grad(set_to_none=True)
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with self.model.replayed_text_leaves(workers) as active:
            self.assertTrue(active)
            evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            loss = sum(e.log_prob + 0.5 * e.value for e in evaluated)
            if extra is not None:
                loss = loss + extra()
            loss.backward()
            self.model.finish_text_replays()
        return ([(e.log_prob.detach(), e.value.detach()) for e in evaluated],
                [None if p.grad is None else p.grad.clone() for p in self.model.parameters()])

    def test_deferred_window_gradients_are_bit_identical_and_need_finishing(self):
        reference = self.window_gradients(torch.enable_grad())
        for workers in (1, 2, 4):
            self.assertTrue(bits_equal(self.deferred_window_gradients(workers), reference))
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with self.assertRaisesRegex(RuntimeError, "not finished"):
            with self.model.replayed_text_leaves(2):
                sum(e.log_prob for e in _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)).backward()
        self.assertNotIn("_text_replay_deferred", self.model.__dict__)

    def test_a_direct_gradient_to_a_text_parameter_fails_closed_under_deferral(self):
        with self.assertRaisesRegex(RuntimeError, "outside its replayed occurrences"):
            self.deferred_window_gradients(2, extra=lambda: self.model.text_gru.bias_hh_l0.sum())

    def test_a_parameter_hook_keeps_the_immediate_replay(self):
        reference = self.window_gradients(torch.enable_grad())
        handle = self.model.byte.weight.register_hook(lambda grad: grad)
        try:
            with self.model.replayed_text_leaves(2):
                self.assertNotIn("_text_replay_deferred", self.model.__dict__)
            self.assertTrue(bits_equal(self.deferred_window_gradients(2), reference))
        finally:
            handle.remove()

    def test_window_values_and_gradients_are_bit_identical_and_leaves_repeat(self):
        made, applied = [], []
        build, apply = cm._ReplayedText.__init__, cm._TextOccurrence.apply

        def counting_build(leaf, *args):
            made.append(args[1]); build(leaf, *args)

        def counting_apply(*args):
            applied.append(args[0]); return apply(*args)
        with patch.object(cm._ReplayedText, "__init__", counting_build), \
                patch.object(cm._TextOccurrence, "apply", counting_apply):
            replayed = self.window_gradients(self.model.replayed_text_leaves())
        reference = self.window_gradients(torch.enable_grad())
        self.assertTrue(bits_equal(replayed, reference))
        # The test is meaningful only when leaves repeat across frames and have several characters.
        self.assertGreater(len(applied), len(made))
        self.assertTrue(any(len(raw) > 1 for raw in made))

    def test_presummed_contributions_are_detected(self):
        # Negative control: summing one occurrence's contributions before adding them changes the bits.
        replay = cm._ReplayedText.replay

        def presummed(leaf, grad):
            values, result, start = replay(leaf, grad), [], 0
            for count in leaf.counts:
                part = values[start:start + count]
                start += count
                total = part[0]
                for value in part[1:]:
                    total = total + value
                result.extend([total] + [torch.zeros_like(total)] * (count - 1))
            return result
        reference = self.window_gradients(torch.enable_grad())
        with patch.object(cm._ReplayedText, "replay", presummed):
            changed = self.window_gradients(self.model.replayed_text_leaves())
        self.assertTrue(bits_equal(changed[0], reference[0]))  # forward values are shared exactly
        self.assertFalse(bits_equal(changed[1], reference[1]))

    def test_replay_is_inert_with_any_hook_and_never_pickled(self):
        module = torch.nn.modules.module
        for register in (lambda: self.model.text_gru.register_forward_hook(lambda *a: None),
                         lambda: self.model.byte.register_full_backward_hook(lambda *a: None),
                         lambda: self.model.tree.register_full_backward_pre_hook(lambda *a: None),
                         lambda: module.register_module_full_backward_hook(lambda *a: None)):
            handle = register()
            try:
                with self.model.replayed_text_leaves() as active:
                    self.assertFalse(active)
                    self.assertNotIn("_leaf_replay_state", self.model.__dict__)
            finally:
                handle.remove()
        with self.model.replayed_text_leaves() as active:
            self.assertTrue(active)
            self.assertNotIn("_leaf_replay_state", pickle.loads(pickle.dumps(self.model)).__dict__)
            self.assertNotIn("_leaf_replay_state", deepcopy(self.model).__dict__)
        self.assertNotIn("_leaf_replay_state", self.model.__dict__)

    def test_a_parameter_change_inside_the_context_fails_closed(self):
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with self.model.replayed_text_leaves():
            evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            with torch.no_grad():
                self.model.text_gru.weight_hh_l0.add_(0.01)
            with self.assertRaises(RuntimeError):
                sum(e.log_prob for e in evaluated).backward()
        with self.model.replayed_text_leaves():
            replaced = torch.nn.Parameter(self.model.byte.weight.detach().clone())
            _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            self.model.byte.weight = replaced
            with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)

    def test_higher_order_gradients_are_refused(self):
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with self.model.replayed_text_leaves():
            evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
            with self.assertRaisesRegex(RuntimeError, "higher-order"):
                torch.autograd.grad(sum(e.log_prob for e in evaluated), list(self.model.parameters()),
                                    create_graph=True, allow_unused=True)

    def test_a_failed_minibatch_leaves_no_replay_state(self):
        ppo.REPLAY_TEXT_LEAVES = True
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)

        def fail():
            self.assertIn("_leaf_replay_state", self.model.__dict__)
            raise RuntimeError("injected")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            ppo._update(self.model, optimizer, self.rollouts, config=PPOConfig(), generator=torch.Generator().manual_seed(1),
                        before_step=fail)
        self.assertNotIn("_leaf_replay_state", self.model.__dict__)

    def test_the_numerical_experiment_takes_precedence_and_is_reported(self):
        ppo.EXPERIMENTS = frozenset({"shared_graph_encodings"})
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        with patch.object(cm._TextOccurrence, "apply", side_effect=AssertionError("replayed under the experiment")):
            report = ppo_update(self.model, optimizer, self.rollouts, config=PPOConfig(),
                                generator=torch.Generator().manual_seed(3))
        self.assertEqual(report.timings_seconds["experiment_shared_graph_encodings"], 1.0)

    def leaf_gradients(self, texts, replay):
        """Parameter gradients of weighted text leaf encodings, each computed in its own frame cache."""
        self.model.zero_grad(set_to_none=True)
        direction = torch.linspace(-1.0, 1.0, self.model.config.feature_size)
        with self.model.replayed_text_leaves() if replay else nullcontext(False) as active:
            self.assertEqual(active, replay)
            state = self.model.__dict__.get("_leaf_replay_state")
            loss = 0
            for weight, text, opaque in texts:
                value = self.model._encode_tree(("text", text), {} if state is None else {cm._LEAF_REPLAY: state})
                loss = loss + weight * ((Opaque.apply(value) if opaque else value) * direction).sum()
            loss.backward()
        return [None if p.grad is None else p.grad.clone() for p in self.model.parameters()]

    def test_undefined_gradients_and_reference_path_leaves_mix_exactly(self):
        texts = ((1.0, "ab", False), (0.5, Text("ab"), False), (2.0, "abc", False), (-1.5, "ab", True),
                 (0.25, "ab", False), (3.0, Text("abc"), False), (1.0, "x", False), (2.0, "abc", True))
        seen, backward = [], cm._TextOccurrence.backward

        def observing(ctx, grad):
            seen.append(grad is None)
            return backward(ctx, grad)
        with patch.object(cm._TextOccurrence, "backward", staticmethod(observing)):
            replayed = self.leaf_gradients(texts, True)
        self.assertEqual(sorted(seen), [False] * 4 + [True] * 2)
        self.assertTrue(bits_equal(replayed, self.leaf_gradients(texts, False)))

    def test_an_in_place_change_to_any_leaf_parameter_before_an_occurrence_fails_closed(self):
        for name in ("byte.weight", "text_gru.weight_ih_l0", "text_gru.weight_hh_l0", "text_gru.bias_ih_l0",
                     "text_gru.bias_hh_l0"):
            with self.model.replayed_text_leaves():
                state = self.model.__dict__["_leaf_replay_state"]
                self.model._encode_tree(("text", "a"), {cm._LEAF_REPLAY: state})
                with torch.no_grad():
                    self.model.get_parameter(name).add_(0.01)
                with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                    self.model._encode_tree(("text", "a"), {cm._LEAF_REPLAY: state})

    def test_a_frame_cache_used_after_its_context_is_computed_as_in_the_reference(self):
        with self.model.replayed_text_leaves():
            state = self.model.__dict__["_leaf_replay_state"]
            self.model._encode_tree(("text", "ab"), {cm._LEAF_REPLAY: state})
        with torch.no_grad():
            self.model.byte.weight.add_(0.01)  # as an optimizer step after the context would
        with patch.object(cm._TextOccurrence, "apply", side_effect=AssertionError("replayed after its context")):
            late = self.model._encode_tree(("text", "ab"), {cm._LEAF_REPLAY: state})
        self.assertTrue(bits_equal(late, self.model._encode_tree(("text", "ab"), {})))
        self.assertEqual(list(state), ["ab"])  # nothing added after the context, which released its leaf
        self.assertIsNone(state["ab"].value)

    def test_tied_leaf_parameters_and_saved_tensor_hooks_keep_the_reference(self):
        tied = deepcopy(self.model)
        with tied.replayed_text_leaves() as active:
            self.assertTrue(active)
        tied.text_gru.bias_hh_l0 = tied.text_gru.bias_ih_l0
        with tied.replayed_text_leaves() as active:
            self.assertFalse(active)
        identity = (lambda value: value, lambda value: value)
        with torch.autograd.graph.saved_tensors_hooks(*identity):
            with self.model.replayed_text_leaves() as active:
                self.assertFalse(active)

        @contextmanager
        def hooked_inside():
            with self.model.replayed_text_leaves() as active, torch.autograd.graph.saved_tensors_hooks(*identity):
                self.assertTrue(active)
                yield
        reference = self.window_gradients(torch.enable_grad())
        with patch.object(cm._TextOccurrence, "apply", side_effect=AssertionError("replayed under saved-tensor hooks")):
            self.assertTrue(bits_equal(self.window_gradients(hooked_inside()), reference))

    def test_a_second_backward_through_a_retained_graph_is_refused(self):
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        with self.model.replayed_text_leaves():
            loss = sum(e.log_prob for e in _evaluate_from_state(self.model, self.inputs[0], window, start, prefix))
            loss.backward(retain_graph=True)
            with self.assertRaisesRegex(RuntimeError, "single backward"):
                loss.backward()


if __name__ == "__main__":
    unittest.main()
