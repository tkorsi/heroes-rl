"""Exact replay of repeated text leaves: bit-identical updates, sensitivity, fail-closed guards (no gameplay)."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import gc
import pickle
import signal
import threading
import unittest
import weakref
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


def strides(grads):
    return [None if grad is None else grad.stride() for grad in grads]


class Raising(torch.autograd.Function):
    """Passes its input forward; its backward raises."""

    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("injected backward failure")


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


class NoHelpers:
    """Declared prefix helpers that compute nothing: every prefix stays in this process."""

    def __init__(self, *args):
        pass

    def vectors(self, *args):
        return None

    def close(self):
        pass


class Pending:
    """A queued replay that no worker ever starts: the main thread must take it."""

    def done(self):
        return False

    def cancel(self):
        return True


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
        for workers, helpers, seed, config in (
                (2, 0, 47, PPOConfig(decisions_per_window=2, windows_per_minibatch=2)),
                (3, 0, 48, PPOConfig(decisions_per_window=3, windows_per_minibatch=8, epochs=3)),
                (1, 0, 49, PPOConfig(decisions_per_window=1, windows_per_minibatch=4)),
                (None, 2, 50, PPOConfig(decisions_per_window=2, windows_per_minibatch=2))):  # threads = declared helpers
            replayed, reference = deepcopy(self.model), deepcopy(self.model)
            deferred, add = [], cm._DeferredTextReplays.add

            def counting(owner, leaf, grad):
                deferred.append(owner.pool._max_workers); return add(owner, leaf, grad)
            with patch.object(cm._DeferredTextReplays, "add", counting), patch.object(ppo, "PREFIX_HELPERS", helpers), \
                    patch("experiments.learner.prefix_pool.PrefixPool", NoHelpers):
                left = self.update(replayed, True, config, seed, workers)
            right = self.update(reference, False, config, seed)
            self.assertGreater(left[0], 0)
            self.assertTrue(deferred)  # the PPO path itself defers, on the expected number of threads
            self.assertEqual(set(deferred), {workers or helpers})
            self.assertEqual(left[1], right[1])
            for a, b in zip(left[2:], right[2:]):
                self.assertTrue(bits_equal(a, b))
            self.assertEqual(strides(left[3]), strides(right[3]))
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
            self.model.backward_with_text_replays(loss)
        return ([(e.log_prob.detach(), e.value.detach()) for e in evaluated],
                [None if p.grad is None else p.grad.clone() for p in self.model.parameters()])

    def test_deferred_window_gradients_are_bit_identical(self):
        reference = self.window_gradients(torch.enable_grad())
        for workers in (1, 2, 4):
            deferred, add = [], cm._DeferredTextReplays.add

            def counting(owner, leaf, grad):
                deferred.append(leaf); return add(owner, leaf, grad)
            with patch.object(cm._DeferredTextReplays, "add", counting):
                result = self.deferred_window_gradients(workers)
            self.assertTrue(deferred)
            self.assertTrue(bits_equal(result, reference))
            self.assertEqual(strides(result[1]), strides(reference[1]))
        self.assertNotIn("_text_replay_deferred", self.model.__dict__)

    def window_graph(self):
        window = self.rollouts[0].decisions[1:]
        start = self.rollouts[0].decisions[0].decision_seq + 1
        prefix = self.model.prefix_states(self.inputs[0], (start,))[start]
        evaluated = _evaluate_from_state(self.model, self.inputs[0], window, start, prefix)
        return sum(e.log_prob + 0.5 * e.value for e in evaluated)

    def test_other_backwards_in_a_deferred_context_replay_inside_the_engine_exactly(self):
        params = list(self.model.parameters())
        leaves = [self.model.byte.weight, self.model.text_gru.weight_hh_l0, self.model.tree[0].weight]
        reference = self.window_gradients(torch.enable_grad())
        self.model.zero_grad(set_to_none=True)
        expected = torch.autograd.grad(self.window_graph(), params, allow_unused=True)
        self.model.zero_grad(set_to_none=True)
        self.window_graph().backward(inputs=leaves)
        expected_inputs = [p.grad.clone() for p in leaves]
        with patch.object(cm._DeferredTextReplays, "add", side_effect=AssertionError("deferred outside the bracket")):
            for run in range(3):
                self.model.zero_grad(set_to_none=True)
                with self.model.replayed_text_leaves(2) as active:
                    self.assertTrue(active)
                    if run == 0:
                        self.window_graph().backward()
                        self.assertTrue(bits_equal([p.grad for p in params], reference[1]))
                    elif run == 1:
                        self.assertTrue(bits_equal(list(torch.autograd.grad(self.window_graph(), params,
                                                                            allow_unused=True)), list(expected)))
                    else:
                        self.window_graph().backward(inputs=leaves)
                        self.assertTrue(bits_equal([p.grad for p in leaves], expected_inputs))
        # Gradients already present (accumulation) also keep the engine's own replay, exactly.
        self.model.zero_grad(set_to_none=True)
        self.window_graph().backward()
        self.window_graph().backward()
        twice = [None if p.grad is None else p.grad.clone() for p in params]
        self.model.zero_grad(set_to_none=True)
        with self.model.replayed_text_leaves(2):
            self.model.backward_with_text_replays(self.window_graph())
        with patch.object(cm._DeferredTextReplays, "add", side_effect=AssertionError("deferred onto gradients")):
            with self.model.replayed_text_leaves(2):
                self.model.backward_with_text_replays(self.window_graph())
        self.assertTrue(bits_equal([p.grad for p in params], twice))

    def test_a_failed_backward_or_finish_refuses_further_brackets(self):
        for failure in ("backward", "finish"):
            self.model.zero_grad(set_to_none=True)
            replay = cm._ReplayedText.replay

            def failing(leaf, grad):
                if threading.current_thread() is threading.main_thread():
                    raise KeyboardInterrupt("injected finish interruption")
                return replay(leaf, grad)
            with self.model.replayed_text_leaves(1):
                deferred = self.model.__dict__["_text_replay_deferred"]
                if failure == "backward":
                    loss = self.window_graph() + Raising.apply(self.model.tree[0].bias).sum()
                    with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
                        self.model.backward_with_text_replays(loss)
                else:
                    with patch.object(cm._ReplayedText, "replay", failing), \
                            patch.object(cm._DeferredTextReplays, "add",
                                         lambda owner, leaf, grad: owner.pending.append((leaf, grad, Pending()))):
                        with self.assertRaises(KeyboardInterrupt):
                            self.model.backward_with_text_replays(self.window_graph())
                self.assertTrue(deferred.broken)
                self.assertFalse(deferred.active or deferred.pending)
                self.assertTrue(all(total is None for total in deferred.sums))
                with self.assertRaisesRegex(RuntimeError, "failed earlier"):
                    self.model.backward_with_text_replays(self.window_graph())
            self.assertFalse([t for t in threading.enumerate() if t.name.startswith("text-replay")])
            self.assertNotIn("_text_replay_deferred", self.model.__dict__)

    def test_the_main_thread_takes_unstarted_replays_after_the_backward_bit_identically(self):
        reference = self.window_gradients(torch.enable_grad())
        taken, workers, replay = threading.Event(), [], cm._ReplayedText.replay

        def tracked(leaf, grad):
            worker = threading.current_thread() is not threading.main_thread()
            if worker:
                taken.wait(30)  # the single worker stays on its first replay until the main thread takes one
            else:
                taken.set()
            workers.append(worker)
            return replay(leaf, grad)
        with patch.object(cm._ReplayedText, "replay", tracked):
            self.assertTrue(bits_equal(self.deferred_window_gradients(1), reference))
        self.assertIn(False, workers)
        self.assertIn(True, workers)

    def test_frozen_leaf_parameters_keep_the_reference_and_freezing_inside_fails_closed(self):
        for workers in (0, 2):
            frozen = deepcopy(self.model)
            frozen.text_gru.bias_hh_l0.requires_grad_(False)
            with frozen.replayed_text_leaves(workers) as active:
                self.assertFalse(active)
                self.assertNotIn("_text_replay_deferred", frozen.__dict__)
        with self.model.replayed_text_leaves():
            state = self.model.__dict__["_leaf_replay_state"]
            self.model._encode_tree(("text", "ab"), {cm._LEAF_REPLAY: state})
            self.model.byte.weight.requires_grad_(False)
            try:
                with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                    self.model._encode_tree(("text", "ab"), {cm._LEAF_REPLAY: state})
            finally:
                self.model.byte.weight.requires_grad_(True)

    def test_replay_workers_take_the_main_thread_count_first(self):
        calls, setter = [], torch.set_num_threads

        def recording(count):
            calls.append((threading.current_thread() is threading.main_thread(), count))
            setter(count)
        with patch.object(torch, "set_num_threads", recording):
            self.deferred_window_gradients(2)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {(False, torch.get_num_threads())})

    def test_replay_threads_leave_process_signals_to_the_main_thread(self):
        masks, replay = [], cm._ReplayedText.replay

        def recording(leaf, grad):
            if threading.current_thread() is not threading.main_thread():
                masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, []))
            return replay(leaf, grad)
        with patch.object(cm._ReplayedText, "replay", recording):
            self.deferred_window_gradients(2)
        self.assertTrue(masks)
        for mask in masks:
            self.assertLessEqual({signal.SIGALRM, signal.SIGINT, signal.SIGTERM}, mask)
        self.assertNotIn(signal.SIGINT, signal.pthread_sigmask(signal.SIG_BLOCK, []))

    def test_a_failed_replay_ends_its_threads_and_state_with_the_update(self):
        ppo.REPLAY_TEXT_LEAVES, ppo.TEXT_REPLAY_WORKERS = True, 2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        calls, replay = [], cm._ReplayedText.replay

        def failing(leaf, grad):
            calls.append(grad)
            if len(calls) == 3:
                raise RuntimeError("injected replay failure")
            return replay(leaf, grad)
        with patch.object(cm._ReplayedText, "replay", failing), \
                self.assertRaisesRegex(RuntimeError, "injected replay failure"):
            ppo._update(self.model, optimizer, self.rollouts, config=PPOConfig(), generator=torch.Generator().manual_seed(5))
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("text-replay")])
        self.assertNotIn("_leaf_replay_state", self.model.__dict__)
        self.assertNotIn("_text_replay_deferred", self.model.__dict__)

    def test_brackets_in_one_context_each_install_their_own_exact_gradients(self):
        reference = self.window_gradients(torch.enable_grad())
        with self.model.replayed_text_leaves(2):
            for _ in range(3):
                self.model.zero_grad(set_to_none=True)
                self.model.backward_with_text_replays(self.window_graph())
                self.assertTrue(bits_equal([p.grad for p in self.model.parameters()], reference[1]))

    def test_a_parameter_frozen_or_hooked_during_the_backward_fails_closed(self):
        byte = self.model.byte.weight
        for change, undo in ((lambda: byte.requires_grad_(False), lambda: byte.requires_grad_(True)),
                             (lambda: hooks.append(byte.register_post_accumulate_grad_hook(lambda p: None)),
                              lambda: hooks.pop().remove())):
            hooks = []
            probe = torch.zeros((), requires_grad=True)
            probe.register_hook(lambda grad: (change(), None)[1])
            try:
                with self.assertRaisesRegex(RuntimeError, "changed or received gradient"):
                    self.deferred_window_gradients(2, extra=lambda: 0.0 * probe)
            finally:
                undo()

    def test_a_parameter_frozen_or_hooked_after_its_last_occurrence_keeps_the_engine_replay(self):
        byte = self.model.byte.weight
        for change, undo in ((lambda: byte.requires_grad_(False), lambda: byte.requires_grad_(True)),
                             (lambda: hooks.append(byte.register_post_accumulate_grad_hook(lambda p: None)),
                              lambda: hooks.pop().remove())):
            hooks = []
            self.model.zero_grad(set_to_none=True)
            with self.model.replayed_text_leaves(2):
                loss = self.window_graph()
                change()
                try:
                    with patch.object(cm._DeferredTextReplays, "add", side_effect=AssertionError("deferred")):
                        self.model.backward_with_text_replays(loss)
                    replayed = [None if p.grad is None else p.grad.clone() for p in self.model.parameters()]
                finally:
                    undo()
            self.model.zero_grad(set_to_none=True)
            loss = self.window_graph()
            change()
            try:
                loss.backward()
                self.assertTrue(bits_equal(replayed, [p.grad for p in self.model.parameters()]))
            finally:
                undo()

    def test_a_direct_gradient_to_a_text_parameter_fails_closed_under_deferral(self):
        with self.assertRaisesRegex(RuntimeError, "outside its replayed occurrences"):
            self.deferred_window_gradients(2, extra=lambda: self.model.text_gru.bias_hh_l0.sum())

    def test_a_parameter_hook_keeps_the_immediate_replay(self):
        reference = self.window_gradients(torch.enable_grad())
        for register in (lambda: self.model.byte.weight.register_hook(lambda grad: grad),
                         lambda: self.model.text_gru.bias_hh_l0.register_post_accumulate_grad_hook(lambda p: None)):
            handle = register()
            try:
                with patch.object(cm._DeferredTextReplays, "add", side_effect=AssertionError("deferred with a hook")):
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

    def test_reordered_contributions_are_detected(self):
        # Negative control: each parameter's contributions of one occurrence, added in reverse order, change the bits.
        replay = cm._ReplayedText.replay

        def reversed_order(leaf, grad):
            values, result, start = replay(leaf, grad), [], 0
            for count in leaf.counts:
                result.extend(reversed(values[start:start + count]))
                start += count
            return result
        reference = self.window_gradients(torch.enable_grad())
        with patch.object(cm._ReplayedText, "replay", reversed_order):
            changed = self.window_gradients(self.model.replayed_text_leaves())
        self.assertTrue(bits_equal(changed[0], reference[0]))
        self.assertFalse(bits_equal(changed[1], reference[1]))

    def test_an_unreleased_leaf_is_freed_without_the_cycle_collector(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            leaf = cm._ReplayedText(self.model, "abc".encode())
            probe = weakref.ref(leaf)
            del leaf
            self.assertIsNone(probe())
        finally:
            if enabled:
                gc.enable()

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

    def test_a_data_replacement_of_a_leaf_parameter_before_an_occurrence_fails_closed(self):
        for name in ("byte.weight", "text_gru.weight_hh_l0"):
            with self.model.replayed_text_leaves():
                state = self.model.__dict__["_leaf_replay_state"]
                self.model._encode_tree(("text", "a"), {cm._LEAF_REPLAY: state})
                param = self.model.get_parameter(name)
                version = param._version
                param.data = param.data.clone()
                self.assertEqual(param._version, version)  # what the version alone would miss
                with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                    self.model._encode_tree(("text", "a"), {cm._LEAF_REPLAY: state})

    def test_autocast_anomaly_mode_and_forward_hooks_keep_the_reference(self):
        module = torch.nn.modules.module
        modes = (lambda: torch.autocast("cpu", dtype=torch.bfloat16),
                 lambda: torch.autograd.detect_anomaly(check_nan=False),
                 lambda: hooked(self.model.byte.register_forward_hook(lambda *a: None)),
                 lambda: hooked(self.model.text_gru.register_forward_pre_hook(lambda *a: None)),
                 lambda: hooked(module.register_module_forward_hook(lambda *a: None)))

        @contextmanager
        def hooked(handle):
            try:
                yield
            finally:
                handle.remove()
        for mode in modes:
            with mode():
                with self.model.replayed_text_leaves() as active:
                    self.assertFalse(active)
            with self.model.replayed_text_leaves() as active:
                self.assertTrue(active)
                state = self.model.__dict__["_leaf_replay_state"]
                with mode(), patch.object(cm._TextOccurrence, "apply", side_effect=AssertionError("replayed")):
                    inside = self.model._encode_tree(("text", "ab"), {cm._LEAF_REPLAY: state})
                    outside = self.model._encode_tree(("text", "ab"), {})
                self.assertTrue(bits_equal(inside, outside))

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
