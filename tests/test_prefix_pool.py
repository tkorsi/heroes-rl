"""Exact prefix helpers: bit-identical encodings and updates, audit, failure and descriptor isolation."""
from copy import deepcopy
from dataclasses import replace
import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import ppo, prefix_pool
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history, frame_key, frame_keys_until
from experiments.learner.ppo import PPOConfig, ppo_update
from experiments.learner.prefix_pool import PrefixPool
from tests.test_contextual_ppo import synthetic_rollout
from tests.test_ppo_phase2_acceleration import tensors_equal
from tests.test_ppo_prefix_performance import uncached_prefixes


def reaped(pid):
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


@unittest.skipUnless(hasattr(os, "fork"), "Prefix helpers fork")
class PrefixPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2611)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.rollouts = (synthetic_rollout(self.model, turns=9, episode="pool-a"),
                         synthetic_rollout(self.model, turns=5, episode="pool-b", boundary="cutoff"))
        self.inputs = tuple(encode_history(item.history) for item in self.rollouts)
        self.addCleanup(setattr, ppo, "PREFIX_HELPERS", ppo.PREFIX_HELPERS)

    def pool(self, helpers=2):
        pool = PrefixPool(self.model, self.inputs, helpers)
        self.addCleanup(pool.close)
        return pool

    def test_helper_encodings_reproduce_serial_and_uncached_states_across_weight_changes(self):
        pool = self.pool(3)
        for _ in range(2):
            for train in (True, False):
                self.model.train(train)
                for index, inputs in enumerate(self.inputs):
                    stops = tuple(range(len(inputs) + 1))
                    supplied = self.model.prefix_states(inputs, stops, vectors=pool.vectors(index, max(stops)))
                    self.assertTrue(tensors_equal(supplied, self.model.prefix_states(inputs, stops)))
                    self.assertTrue(tensors_equal(supplied, uncached_prefixes(self.model, inputs, stops)))
            with torch.no_grad():  # evades version counters; every request carries the current bits
                self.model.tree[0].bias.data.add_(0.1)
                self.model.text_gru.weight_hh_l0.data.mul_(1.5)

    def test_keys_computed_once_reproduce_per_call_keys(self):
        inputs = self.inputs[0]
        keys = frame_keys_until([], inputs, 3)
        self.assertEqual(len(keys), 3)
        self.assertIs(frame_keys_until(keys, inputs, len(inputs)), keys)
        self.assertEqual(keys, [None if f.semantics is None else frame_key(f) for f in inputs])
        stops = tuple(range(len(inputs) + 1))
        for train in (True, False):
            self.model.train(train)
            self.assertTrue(tensors_equal(self.model.prefix_states(inputs, stops, keys=keys),
                                          self.model.prefix_states(inputs, stops)))

    def test_update_with_helpers_is_bitwise_equal_to_the_serial_update(self):
        results = []
        for helpers in (0, 2):
            ppo.PREFIX_HELPERS = helpers
            model = deepcopy(self.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            generator = torch.Generator().manual_seed(45)
            report = ppo_update(model, optimizer, tuple(replace(item) for item in self.rollouts),
                                config=PPOConfig(decisions_per_window=2, windows_per_minibatch=2), generator=generator)
            self.assertEqual("prefix_helpers" in report.timings_seconds, helpers > 0)
            results.append((replace(report, timings_seconds=None), model.state_dict(), [p.grad for p in model.parameters()],
                            optimizer.state_dict(), generator.get_state()))
        self.assertEqual(results[0][0], results[1][0])
        for left, right in zip(results[0][1:], results[1][1:]):
            self.assertTrue(tensors_equal(left, right))

    def test_helpers_end_with_the_update_even_when_it_fails(self):
        ppo.PREFIX_HELPERS = 2
        seen = []
        original = PrefixPool.__init__
        def record(pool, *args):
            original(pool, *args)
            seen.append(list(pool.processes))
        PrefixPool.__init__ = record
        try:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
            def fail():
                raise RuntimeError("injected")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                ppo._update(self.model, optimizer, self.rollouts, config=PPOConfig(), generator=torch.Generator().manual_seed(1),
                            before_step=fail)
        finally:
            PrefixPool.__init__ = original
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(reaped(pid) for pid in seen[0]))

    def test_a_helper_with_different_bits_is_rejected(self):
        parent = os.getpid()
        original = prefix_pool._encode
        def poisoned(model, frames, seqs):
            vectors = original(model, frames, seqs)
            return [vector + 1e-3 for vector in vectors] if os.getpid() != parent else vectors
        prefix_pool._encode = poisoned
        try:
            pool = self.pool(2)
        finally:
            prefix_pool._encode = original
        stop = len(self.inputs[0])
        with self.assertRaisesRegex(RuntimeError, "differs from the parent"):
            pool.vectors(0, stop)

    def test_a_dead_helper_fails_the_request_and_is_reaped(self):
        pool = self.pool(1)
        pid = pool.processes[0]
        os.kill(pid, 9)
        with self.assertRaises((EOFError, OSError, RuntimeError)):
            pool.vectors(0, len(self.inputs[0]))
        pool.close()
        self.assertTrue(reaped(pid))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Descriptor listing uses /proc")
    def test_helpers_hold_only_their_own_socket_and_standard_streams(self):
        read, write = os.pipe()
        self.addCleanup(os.close, read)
        self.addCleanup(os.close, write)
        pool = self.pool(2)
        pool.vectors(0, len(self.inputs[0]))  # every helper answered, so it closed its inherited descriptors
        for pid in pool.processes:
            names = os.listdir(f"/proc/{pid}/fd")
            targets = {os.readlink(f"/proc/{pid}/fd/{name}") for name in names}
            self.assertLessEqual(len(names), 4)
            self.assertNotIn(os.readlink(f"/proc/self/fd/{write}"), targets)

    def test_helpers_need_the_exact_cached_path(self):
        hook = self.model.tree.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, "exact cached"):
                PrefixPool(self.model, self.inputs, 1)
            with self.assertRaisesRegex(ValueError, "exact cached"):
                self.model.prefix_states(self.inputs[0], (1,), vectors={})
        finally:
            hook.remove()
        for helpers in (0, -1, 16, 1.0, True):
            with self.assertRaises(ValueError):
                PrefixPool(self.model, self.inputs, helpers)


if __name__ == "__main__":
    unittest.main()
