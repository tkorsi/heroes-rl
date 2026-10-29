"""Actual compiled C++ learner parity on disposable synthetic state, never gameplay.

Set HEROES_TEST_NATIVE_MANIFEST to a freshly built external manifest. The suite
never compiles implicitly and never selects the backend for an official run.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Optional PyTorch is not installed")

from experiments.learner import ppo
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history
from experiments.learner.native_backend import load_backend, sha256, verified_manifest
from experiments.learner.native_recipes import compile_history
from tests.test_contextual_ppo import synthetic_rollout

MANIFEST = os.environ.get("HEROES_TEST_NATIVE_MANIFEST")


def equal_tree(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            equal_tree(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            equal_tree(test, left, right)
    else:
        test.assertEqual(actual, expected)


@unittest.skipUnless(MANIFEST, "Requires an explicitly built native learner manifest")
class NativeLearnerKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.digest = sha256(MANIFEST)
        cls.backend = load_backend(MANIFEST, cls.digest)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(291017)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.rollout = synthetic_rollout(self.model, turns=4)
        self.inputs = encode_history(self.rollout.history)
        self.history = compile_history(self.model, self.inputs, self.rollout.decisions, self.backend)
        self.native = self.backend.NativeModel(dict(self.model.named_parameters()), 4, 8)

    def test_every_prefix_boundary_matches_at_one_four_and_sixteen_workers(self):
        stops = list(range(len(self.inputs) + 1))
        expected = self.model.prefix_states(self.inputs, stops)
        rng = torch.get_rng_state().clone()
        for count in (1, 4, 16):
            actual = self.native.prefix(self.history, stops, count)
            equal_tree(self, actual, expected)
            self.assertTrue(all(not item.requires_grad for item in actual.values()))
            equal_tree(self, torch.get_rng_state(), rng)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_window_states_values_joint_and_conditional_probabilities_match(self):
        records = self.rollout.decisions[1:]
        start = self.rollout.decisions[0].decision_seq + 1
        state = self.model.prefix_states(self.inputs, (start,))[start]
        actual = self.native.evaluate_window_details(self.history, start, records[-1].decision_seq,
                                                    [item.decision_seq for item in records], state)
        expected = ppo.evaluate_window(self.model, self.inputs, records, start_seq=start)
        equal_tree(self, actual[0], torch.stack([item.log_prob for item in expected]))
        equal_tree(self, actual[1], torch.stack([item.value for item in expected]))
        current = state.detach()
        details = []
        by_seq = {item.decision_seq: item for item in records}
        for seq in range(start, records[-1].decision_seq + 1):
            output = self.model.forward_step(self.inputs[seq], current)
            current = output.state
            if seq in by_seq:
                trace = output.decision.distribution.evaluate(by_seq[seq].action).conditionals
                details.append(torch.stack([part.log_prob for part in trace]))
        equal_tree(self, actual[2], current)
        equal_tree(self, actual[3], details)

    def test_window_gradients_are_bit_exact(self):
        records = self.rollout.decisions
        native = self.native.evaluate_window(self.history, 0, records[-1].decision_seq,
                                             [item.decision_seq for item in records], self.model.initial_state())
        loss = (native[0] + native[1]).sum()
        loss.backward()
        actual = [None if p.grad is None else p.grad.clone() for p in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        expected = ppo.evaluate_window(self.model, self.inputs, records, start_seq=0)
        (torch.stack([item.log_prob for item in expected]) + torch.stack([item.value for item in expected])).sum().backward()
        equal_tree(self, actual, [p.grad for p in self.model.parameters()])

    def test_native_text_graph_reuse_and_parallel_backward_preserve_every_gradient(self):
        records = self.rollout.decisions
        expected = ppo.evaluate_window(self.model, self.inputs, records, start_seq=0)
        baseline = (torch.stack([item.log_prob for item in expected]) +
                    torch.stack([item.value for item in expected])).sum()
        baseline.backward()
        gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in self.model.parameters()]
        rng = torch.get_rng_state().clone()
        for workers in (0, 1, 2, 3, 15):
            self.model.zero_grad(set_to_none=True)
            self.native.begin_text_replay(workers)
            try:
                values = self.native.evaluate_window(self.history, 0, records[-1].decision_seq,
                    [item.decision_seq for item in records], self.model.initial_state())
                loss = (values[0] + values[1]).sum()
                equal_tree(self, loss, baseline)
                self.native.backward_with_text_replay(loss)
                equal_tree(self, [parameter.grad for parameter in self.model.parameters()], gradients)
            finally:
                self.native.end_text_replay()
            statistics = self.native.text_replay_statistics()
            self.assertEqual(statistics["workers"], workers)
            self.assertEqual(statistics["reducer_threads"], int(workers >= 2))
            self.assertEqual(statistics["replay_threads"], workers - int(workers >= 2))
            self.assertGreater(statistics["occurrences"], statistics["unique_texts"])
            if workers:
                self.assertGreater(statistics["completed_replays"], 0)
                self.assertGreater(statistics["folded_contributions"], statistics["completed_replays"])
                self.assertGreaterEqual(statistics["peak_parallel_replays"], 1)
                self.assertLessEqual(statistics["peak_parallel_replays"], workers - 1 if workers >= 2 else 2)
                self.assertLessEqual(statistics["max_pending_replays"], 1024)
                self.assertGreater(statistics["fold_nanoseconds"], 0)
                if workers >= 2:
                    self.assertEqual(statistics["main_thread_replays"], 0)
            self.assertGreater(statistics["producer_backward_nanoseconds"], 0)
            equal_tree(self, torch.get_rng_state(), rng)
            self.native.close()  # Cleanup is idempotent.

    def test_ordered_reducer_bounds_backlog_and_preserves_repeated_leaf_additions(self):
        history = self.backend.NativeHistory([("text", b"xy", [])], [])
        def loss():
            return torch.stack([self.native.tree(history, 0) * (index % 7 - 3)
                                for index in range(2050)]).sum()
        loss().backward()
        expected = [None if parameter.grad is None else parameter.grad.clone() for parameter in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        self.native.begin_text_replay(2)
        try:
            self.native.backward_with_text_replay(loss())
        finally:
            self.native.end_text_replay()
        actual = [parameter.grad for parameter in self.model.parameters()]
        equal_tree(self, actual, expected)
        for left, right in zip(actual, expected):
            if left is not None:
                self.assertTrue(torch.equal(left.view(torch.int32), right.view(torch.int32)))
        statistics = self.native.text_replay_statistics()
        self.assertEqual(statistics['completed_replays'], 2050)
        self.assertEqual(statistics['occurrences'], 2050)
        self.assertEqual(statistics['main_thread_replays'], 0)
        self.assertEqual(statistics['reducer_threads'], 1)
        self.assertEqual(statistics['replay_threads'], 1)
        self.assertGreater(statistics['max_pending_replays'], 0)
        self.assertLessEqual(statistics['max_pending_replays'], 1024)

    def test_outer_backward_failure_closes_reducer_after_work_was_queued(self):
        history = self.backend.NativeHistory([("text", b"queued cleanup", [])], [])
        self.native.begin_text_replay(3)
        outputs = [self.native.tree(history, 0) for _ in range(128)]
        def fail_after_other_occurrences(gradient):
            raise RuntimeError('outer backward failure after queued work')
        handle = outputs[0].register_hook(fail_after_other_occurrences)
        try:
            with self.assertRaisesRegex(RuntimeError, 'outer backward failure'):
                self.native.backward_with_text_replay(torch.stack(outputs).sum())
            self.native.close()
        finally:
            handle.remove()
            self.native.close()
        self.assertGreater(self.native.text_replay_statistics()['max_pending_replays'], 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in self.model.named_parameters()
                            if name == 'byte.weight' or name.startswith('text_gru.')))
        self.native.begin_text_replay(3)
        try:
            self.native.backward_with_text_replay(self.native.tree(history, 0).sum())
        finally:
            self.native.end_text_replay()

    def test_native_replay_rejects_hooks_mutation_higher_order_and_released_occurrences(self):
        for workers in (-1, 256):
            with self.assertRaises(RuntimeError):
                self.native.begin_text_replay(workers)
        hooked = deepcopy(self.model)
        hooked_native = self.backend.NativeModel(dict(hooked.named_parameters()), 4, 8)
        hook = hooked.byte.weight.register_hook(lambda value: value)
        try:
            with self.assertRaisesRegex(RuntimeError, "hooks"):
                hooked_native.begin_text_replay(3)
        finally:
            hook.remove()
        records = self.rollout.decisions
        def loss():
            values = self.native.evaluate_window(self.history, 0, records[-1].decision_seq,
                [item.decision_seq for item in records], self.model.initial_state())
            return (values[0] + values[1]).sum()
        self.native.begin_text_replay(3)
        released = loss()
        self.native.end_text_replay()
        with self.assertRaisesRegex(RuntimeError, "released scope"):
            released.backward()
        self.model.zero_grad(set_to_none=True)
        self.native.begin_text_replay(3)
        higher_order = loss()
        try:
            with self.assertRaisesRegex(RuntimeError, "higher-order"):
                torch.autograd.grad(higher_order, self.model.parameters(), create_graph=True, allow_unused=True)
        finally:
            self.native.end_text_replay()
        self.model.zero_grad(set_to_none=True)
        self.native.begin_text_replay(3)
        changed = loss()
        with torch.no_grad():
            self.model.byte.weight.add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.native.backward_with_text_replay(changed)
        self.native.close()
        self.model.zero_grad(set_to_none=True)
        self.native.begin_text_replay(3)
        try:
            self.native.backward_with_text_replay(loss())
        finally:
            self.native.close()

    def test_native_replay_worker_failure_is_joined_without_installing_partial_text_gradients(self):
        records = self.rollout.decisions
        def loss():
            values = self.native.evaluate_window(self.history, 0, records[-1].decision_seq,
                [item.decision_seq for item in records], self.model.initial_state())
            return (values[0] + values[1]).sum()
        self.native.begin_text_replay(3)
        changed = loss()
        def mutate_after_the_bracket_started(gradient):
            with torch.no_grad():
                self.model.byte.weight.add_(0.01)
            return gradient
        handle = changed.register_hook(mutate_after_the_bracket_started)
        try:
            with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                self.native.backward_with_text_replay(changed)
            self.native.close()  # Cleanup must not replace the primary worker failure.
        finally:
            handle.remove()
            self.native.close()
        self.assertGreater(self.native.text_replay_statistics()["completed_replays"], 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in self.model.named_parameters()
                            if name == "byte.weight" or name.startswith("text_gru.")))
        self.model.zero_grad(set_to_none=True)
        self.native.begin_text_replay(3)
        try:
            self.native.backward_with_text_replay(loss())
        finally:
            self.native.close()

    def test_two_complete_updates_preserve_adam_weights_gradients_and_rng_exactly(self):
        reference, candidate = deepcopy(self.model), deepcopy(self.model)
        models = (reference, candidate)
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.0003) for model in models]
        generators = [torch.Generator().manual_seed(4412) for _ in models]
        config = ppo.PPOConfig(epochs=2, decisions_per_window=2, windows_per_minibatch=2)
        execution = {"manifest": MANIFEST, "sha256": self.digest, "prefix_threads": 4}
        for update in range(2):
            rollouts = (synthetic_rollout(reference, turns=4, episode=f"native-{update}"),
                        synthetic_rollout(reference, turns=3, episode=f"native-cutoff-{update}", boundary="cutoff"))
            reports = []
            for model, optimizer, generator, selected in zip(models, optimizers, generators, (None, execution)):
                with patch.object(ppo, "NATIVE_EXECUTION", selected), patch.object(ppo, "PREFIX_HELPERS", 0), \
                        patch.object(ppo, "EXPERIMENTS", frozenset()):
                    reports.append(ppo.ppo_update(model, optimizer, rollouts, config=config, generator=generator))
            self.assertEqual(replace(reports[0], timings_seconds=None), replace(reports[1], timings_seconds=None))
            equal_tree(self, candidate.state_dict(), reference.state_dict())
            equal_tree(self, [p.grad for p in candidate.parameters()], [p.grad for p in reference.parameters()])
            equal_tree(self, optimizers[1].state_dict(), optimizers[0].state_dict())
            equal_tree(self, generators[1].get_state(), generators[0].get_state())

    def test_native_loss_matches_reference_and_detaches_behavior_inputs(self):
        values = [torch.tensor([-0.2, 0.4, -0.7], requires_grad=True) for _ in range(6)]
        for clip in (None, 0.2):
            config = ppo.PPOConfig(value_clip=clip)
            native = self.backend.clipped_losses(*values, config.clip_ratio, config.value_clip, config.value_coefficient)
            reference = ppo.clipped_losses(*values, config)
            equal_tree(self, native, reference)
            for item in values:
                item.grad = None
            native[0].backward()
            self.assertTrue(all(item.grad is None for item in values[2:]))

    def test_parameters_are_live_aliases_and_prefix_memos_expire(self):
        stop = len(self.inputs)
        before = self.native.prefix(self.history, [stop], 4)[stop]
        self.model.tree[0].bias.data.add_(0.1)
        after = self.native.prefix(self.history, [stop], 4)[stop]
        self.assertFalse(torch.equal(before, after))
        equal_tree(self, after, self.model.prefix_states(self.inputs, [stop])[stop])

    def test_thread_failure_is_propagated_and_a_later_prefix_still_works(self):
        parameter = self.model.tree[0].weight
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.set_(torch.zeros(2, 2))
        try:
            with self.assertRaises(RuntimeError):
                self.native.prefix(self.history, [len(self.inputs)], 16)
        finally:
            with torch.no_grad():
                parameter.set_(original)
        equal_tree(self, self.native.prefix(self.history, [len(self.inputs)], 16),
                   self.model.prefix_states(self.inputs, [len(self.inputs)]))

    def test_malformed_arenas_boundaries_and_parameters_reject(self):
        with self.assertRaises(RuntimeError):
            self.backend.NativeHistory([("node", 0, [])], [])
        with self.assertRaises(RuntimeError):
            self.backend.NativeHistory([("number", [float("nan"), 0.0], [])], [])
        with self.assertRaises(RuntimeError):
            self.backend.NativeModel({"byte.weight": self.model.byte.weight}, 4, 8)
        duplicate = dict(self.model.named_parameters())
        duplicate["text_gru.bias_hh_l0"] = duplicate["text_gru.bias_ih_l0"]
        with self.assertRaisesRegex(RuntimeError, "distinct parameter identities"):
            self.backend.NativeModel(duplicate, 4, 8)
        for count in (0, 257):
            with self.assertRaises(RuntimeError):
                self.native.prefix(self.history, [1], count)
        with self.assertRaises(RuntimeError):
            self.native.prefix(self.history, [len(self.inputs) + 1], 1)
        records = self.rollout.decisions
        with self.assertRaises(RuntimeError):
            self.native.evaluate_window(self.history, 0, records[-1].decision_seq,
                                        [records[1].decision_seq, records[0].decision_seq], self.model.initial_state())
        bad = torch.tensor([float("inf")])
        good = torch.tensor([0.0])
        with self.assertRaises(RuntimeError):
            self.backend.clipped_losses(bad, good, good, good, good, good, 0.2, 0.2, 0.5)

    def test_manifest_rejects_wrong_digest_source_identity_and_binary_bytes(self):
        with self.assertRaises(ValueError):
            load_backend(MANIFEST, "0" * 64)
        receipt = json.loads(Path(MANIFEST).read_text())
        with tempfile.TemporaryDirectory(prefix="heroes-native-tamper-") as directory:
            directory = Path(directory)
            fake = directory / "manifest.json"
            changed = deepcopy(receipt)
            changed["identity"]["contract"] = "tampered"
            fake.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "source.*ABI"):
                verified_manifest(fake, sha256(fake))
            binary = directory / "altered.so"
            binary.write_bytes(b"not the compiled native learner")
            changed = deepcopy(receipt)
            changed["binary"]["path"] = str(binary)
            fake.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "binary differs"):
                verified_manifest(fake, sha256(fake))


if __name__ == "__main__":
    unittest.main()
