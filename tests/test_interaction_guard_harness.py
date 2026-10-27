"""Regression checks for postcollection diagnostic bookkeeping; no native work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

from tools.validate_interaction_guard import state_snapshot, verify_state_snapshot, write_shutdown


class HarnessFinalizationTests(unittest.TestCase):
    def state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward(); optimizer.step(); optimizer.zero_grad()
        return SimpleNamespace(model=model, optimizer=optimizer, generator=torch.Generator().manual_seed(31),
            pool=SimpleNamespace(state_dict=lambda: {'frozen': torch.tensor([1, 2])}),
            sampler_state={'next': 12}, progress=SimpleNamespace(updates=7, rollouts=28, decisions=120))

    def test_full_state_comparison_accepts_same_state_and_detects_tensor_or_adam_mutation(self):
        state = self.state(); before = state_snapshot(state)
        verify_state_snapshot(before, state)
        with torch.no_grad(): state.model.weight.add_(1)
        with self.assertRaises(ValueError): verify_state_snapshot(before, state)
        state.model.load_state_dict(before['model'])
        first = next(iter(state.optimizer.state.values()))
        first['exp_avg'].add_(1)
        with self.assertRaises(ValueError): verify_state_snapshot(before, state)

    def test_sampler_and_rng_change_are_detected(self):
        state = self.state(); before = state_snapshot(state)
        torch.rand(1, generator=state.generator)
        with self.assertRaises(ValueError): verify_state_snapshot(before, state)
        state.generator.set_state(before['generator'])
        state.sampler_state['next'] += 1
        with self.assertRaises(ValueError): verify_state_snapshot(before, state)

    def test_shutdown_keeps_the_actual_cleanup_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cleanup = {'finished_utc': '2026-10-27T19:30:00Z', 'owned_cleanup_verified': True, 'errors': []}
            write_shutdown(path, 0, cleanup)
            self.assertEqual(json.loads((path/'shutdown.json').read_text()), {'returncode': 0, **cleanup})


if __name__ == '__main__': unittest.main()
