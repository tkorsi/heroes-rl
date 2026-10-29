"""Exact first-order CPU backward tapes on disposable graphs, never training data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import unittest

import torch


@unittest.skipUnless(os.environ.get('HEROES_NATIVE_TEST_MANIFEST'), 'An explicitly built native learner is required')
class NativeReplayTapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.learner.native_backend import load_backend
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        manifest = Path(os.environ['HEROES_NATIVE_TEST_MANIFEST']).resolve()
        backend = load_backend(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
        cls.backend = backend
        cls.tape_class = backend._ReplayTape

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(43161)
        self.global_rng = torch.get_rng_state().clone()

    def tearDown(self):
        self.assertTrue(torch.equal(self.global_rng, torch.get_rng_state()))

    def leaf(self, *shape):
        return torch.randn(shape, generator=self.generator, dtype=torch.float32).requires_grad_()

    def gradient(self, shape):
        return torch.randn(shape, generator=self.generator, dtype=torch.float32)

    def exact(self, left, right):
        if left is None or right is None:
            self.assertIs(left, right)
            return
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(torch.equal(left.detach().contiguous().view(torch.uint8),
                                    right.detach().contiguous().view(torch.uint8)))

    def compile(self, output, aliases):
        tape, reason = self.tape_class.compile(output, aliases)
        self.assertIsNotNone(tape, reason)
        self.assertGreater(tape.node_count(), len(aliases))
        return tape

    def compare(self, output, aliases, repeats=3):
        tape = self.compile(output, aliases)
        for _ in range(repeats):
            gradient = self.gradient(output.shape)
            expected = torch.autograd.grad(output, aliases, gradient, retain_graph=True, allow_unused=True)
            ran, actual = tape.run(gradient)
            self.assertTrue(ran, tape.supported_now())
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual): self.exact(left, right)
            self.assertTrue(all(alias.grad is None for alias in aliases))
        return tape

    def test_shared_branch_accumulation_and_repeated_retained_graph(self):
        a, b = self.leaf(3, 4), self.leaf(3, 4)
        shared = a.tanh()
        output = ((shared * b) + (shared * shared) - b).sigmoid()
        self.compare(output, [a, b])

    def test_multiple_output_nodes_preserve_input_slot_accumulation(self):
        a = self.leaf(3, 4)
        first, second, third = a.unbind(0)
        output = ((first * second) + (first * third) + second).tanh()
        self.compare(output, [a])

    def test_unsafe_split_and_stack_routes_preserve_contributions(self):
        a = self.leaf(2, 8)
        first, second = torch.unsafe_chunk(a, 2, dim=1)
        output = torch.stack((first * second, first + second)).transpose(0, 1).tanh()
        self.compare(output, [a])

    def test_text_embedding_gru_graph_matches_engine_for_varied_lengths(self):
        for length in (1, 2, 9, 31):
            with self.subTest(length=length):
                parameters = [self.leaf(257, 4), self.leaf(12, 4), self.leaf(12, 4),
                              self.leaf(12), self.leaf(12)]
                indices = torch.tensor([1 + (i * 29) % 256 for i in range(length)])
                embedding = torch.nn.functional.embedding(indices, parameters[0], padding_idx=0).unsqueeze(0)
                _, hidden = torch._VF.gru(embedding, torch.zeros(1, 1, 4), parameters[1:],
                                         True, 1, 0.0, True, False, True)
                output = hidden.select(0, 0).select(0, 0)
                self.compare(output, parameters, repeats=2)

    def test_canonical_text_parameter_edge_contributions_match_byte_for_byte(self):
        parameters = [self.leaf(257, 4), self.leaf(12, 4), self.leaf(12, 4),
                      self.leaf(12), self.leaf(12)]
        for text in ('a', 'ab', 'adventure', 'long-label-' * 3, 'é🚀'):
            with self.subTest(text=text):
                gradients = [self.gradient((4,)), self.gradient((8,))[::2],
                             torch.tensor([-0.0]).expand(4)]
                expected, actual, stats, reason = self.backend._text_replay_probe(
                    parameters, text, 4, gradients)
                self.assertEqual(stats['tape_available'], 1, reason)
                self.assertEqual(stats['tape_replays'], len(gradients))
                self.assertEqual(stats['engine_replays'], len(gradients))
                self.assertEqual(len(expected), len(actual))
                for reference_terms, tape_terms in zip(expected, actual):
                    self.assertGreaterEqual(len(reference_terms), len(parameters))
                    if len(text.encode('utf-8')) > 1:
                        self.assertGreater(len(reference_terms), len(parameters))
                    self.assertEqual(len(reference_terms), len(tape_terms))
                    for left, right in zip(reference_terms, tape_terms): self.exact(left, right)
                self.assertTrue(all(parameter.grad is None for parameter in parameters))

    def test_signed_zero_large_and_small_finite_values_match_byte_for_byte(self):
        values = [0.0, -0.0, 1e-20, -1e-20, 1e10, -1e10,
                  0.125, -0.125, 1.0, -1.0, 1e-10, -1e-10]
        a = torch.tensor(values, dtype=torch.float32).reshape(3, 4).requires_grad_()
        b = torch.tensor(list(reversed(values)), dtype=torch.float32).reshape(3, 4).requires_grad_()
        output = ((a * b) + (a * a) - b).tanh()
        tape = self.compile(output, [a, b])
        gradients = [torch.tensor(values).reshape(3, 4), torch.tensor([-0.0]).expand_as(output),
                     torch.tensor([1e-20]).expand_as(output)]
        for gradient in gradients:
            expected = torch.autograd.grad(output, (a, b), gradient, retain_graph=True)
            ran, actual = tape.run(gradient)
            self.assertTrue(ran)
            for left, right in zip(expected, actual): self.exact(left, right)

    def test_noncontiguous_aliases_and_view_gradient_accumulation(self):
        a = self.leaf(5, 6).transpose(0, 1).detach().requires_grad_()
        b = self.leaf(6, 5)
        self.assertFalse(a.is_contiguous())
        shared = (a * b).transpose(0, 1)
        output = ((shared * shared) + shared).tanh()
        tape = self.compile(output, [a, b])
        gradients = [self.gradient((6, 5)).transpose(0, 1), torch.tensor([0.25]).expand_as(output)]
        for gradient in gradients:
            self.assertFalse(gradient.is_contiguous())
            expected = torch.autograd.grad(output, (a, b), gradient, retain_graph=True)
            ran, actual = tape.run(gradient)
            self.assertTrue(ran)
            for left, right in zip(expected, actual): self.exact(left, right)

    def test_forward_ad_output_and_gradient_require_pre_execution_fallback(self):
        alias = self.leaf(3)
        output = alias.tanh()
        tape = self.compile(output, [alias])
        with torch.autograd.forward_ad.dual_level():
            dual_output = torch.autograd.forward_ad.make_dual(output, torch.ones_like(output))
            rejected, reason = self.tape_class.compile(dual_output, [alias])
            self.assertIsNone(rejected)
            self.assertTrue(reason)
            dual_gradient = torch.autograd.forward_ad.make_dual(torch.ones_like(output), torch.ones_like(output))
            ran, gradients = tape.run(dual_gradient)
            self.assertFalse(ran)
            self.assertEqual(gradients, [])
        self.assertIsNone(alias.grad)

    def test_float64_requires_pre_execution_fallback(self):
        alias = torch.tensor([0.25, -0.5], dtype=torch.float64, requires_grad=True)
        output = alias.tanh()
        tape, reason = self.tape_class.compile(output, [alias])
        self.assertIsNone(tape)
        self.assertTrue(reason)
        self.assertIsNone(alias.grad)
        float_alias = self.leaf(2)
        float_tape = self.compile(float_alias.tanh(), [float_alias])
        ran, gradients = float_tape.run(torch.ones(2, dtype=torch.float64))
        self.assertFalse(ran)
        self.assertEqual(gradients, [])
        self.assertIsNone(float_alias.grad)

    def test_concurrent_replay_does_not_mutate_the_shared_graph_or_aliases(self):
        a, b = self.leaf(3, 4), self.leaf(3, 4)
        output = ((a * b) + (a * a)).tanh()
        tape = self.compile(output, [a, b])
        gradients = [self.gradient(output.shape) for _ in range(24)]
        expected = [torch.autograd.grad(output, (a, b), gradient, retain_graph=True) for gradient in gradients]
        snapshots = [value.detach().clone() for value in (a, b, output)]
        with ThreadPoolExecutor(max_workers=8, initializer=lambda: torch.set_num_threads(1)) as workers:
            actual = list(workers.map(tape.run, gradients))
        for reference, (ran, result) in zip(expected, actual):
            self.assertTrue(ran)
            for left, right in zip(reference, result): self.exact(left, right)
        for left, right in zip(snapshots, (a, b, output)): self.exact(left, right)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_unknown_builtin_and_unlisted_alias_force_pre_execution_fallback(self):
        a, b = self.leaf(3), self.leaf(3)
        for output, aliases in ((a.sin(), [a]), (a * b, [a]), (a.tanh(), [a, b]), (a.tanh(), [a, a])):
            tape, reason = self.tape_class.compile(output, aliases)
            self.assertIsNone(tape)
            self.assertTrue(reason)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_external_hook_added_after_compilation_forces_fallback_and_remains_visible(self):
        a = self.leaf(3)
        output = a.tanh()
        tape = self.compile(output, [a])
        calls = []
        handle = output.register_hook(lambda gradient: calls.append(gradient.clone()) or gradient)
        try:
            ran, result = tape.run(torch.ones_like(output))
            self.assertFalse(ran)
            self.assertEqual(result, [])
            self.assertEqual(calls, [])
            torch.autograd.grad(output, a, torch.ones_like(output))
            self.assertEqual(len(calls), 1)
        finally:
            handle.remove()

    def test_record_function_callback_forces_fallback_and_engine_events_remain_visible(self):
        a = self.leaf(3)
        output = a.tanh()
        tape = self.compile(output, [a])
        with torch.autograd.profiler.profile(use_cpu=True) as profile:
            ran, result = tape.run(torch.ones_like(output))
            self.assertFalse(ran)
            self.assertEqual(result, [])
            torch.autograd.grad(output, a, torch.ones_like(output))
        self.assertTrue(any('TanhBackward' in event.name for event in profile.function_events))

    def test_anomaly_mode_forces_fallback(self):
        a = self.leaf(3)
        output = a.tanh()
        tape = self.compile(output, [a])
        with torch.autograd.set_detect_anomaly(True):
            ran, result = tape.run(torch.ones_like(output))
        self.assertFalse(ran)
        self.assertEqual(result, [])

    def test_saved_tensor_hooks_are_rejected_after_creation_context_exits(self):
        a = self.leaf(3)
        unpacked = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value,
                                                     lambda value: unpacked.append(True) or value):
            output = a.tanh()
        tape, reason = self.tape_class.compile(output, [a])
        self.assertIsNone(tape)
        self.assertTrue(reason)
        self.assertEqual(unpacked, [])
        torch.autograd.grad(output, a, torch.ones_like(output))
        self.assertGreater(len(unpacked), 0)

    def test_enclosing_graph_task_forces_fallback(self):
        a = self.leaf(3)
        output = a.tanh()
        tape = self.compile(output, [a])
        observed = []

        class Outer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                observed.append(tape.run(torch.ones_like(output))[0])
                return gradient

        torch.autograd.grad(Outer.apply(a), a, torch.ones_like(a))
        self.assertEqual(observed, [False])

    def test_bad_root_metadata_raises_without_execution_or_retry(self):
        a = self.leaf(3)
        output = a.tanh()
        tape = self.compile(output, [a])
        with self.assertRaises(RuntimeError):
            tape.run(torch.ones(2))
        self.assertIsNone(a.grad)
        self.compare(output, [a], repeats=1)

    def test_saved_variable_version_failure_propagates_without_engine_retry(self):
        a = self.leaf(3)
        output = (a * a).tanh()
        tape = self.compile(output, [a])
        with torch.no_grad(): a.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            tape.run(torch.ones_like(output))
        self.assertIsNone(a.grad)


if __name__ == '__main__':
    unittest.main()
