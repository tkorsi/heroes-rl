"""Regression checks for postcollection diagnostic bookkeeping; no native work."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch

from tools.validate_interaction_guard import (state_snapshot, verify_state_snapshot, write_shutdown,
    diagnostic_schedule, reserve_diagnostic_seeds, stopped_candidate_parent, configure_artifact_root)


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


class GarrisonCandidateTests(unittest.TestCase):
    def test_machine_root_is_set_before_preservation_import(self):
        with tempfile.TemporaryDirectory() as directory:
            code = """import sys
from pathlib import Path
from tools.validate_interaction_guard import configure_artifact_root
configure_artifact_root(sys.argv[1])
from experiments.learner.builtin_preservation import DURABLE_ROOT
assert DURABLE_ROOT == Path(sys.argv[1]).resolve(), DURABLE_ROOT
"""
            result = subprocess.run([sys.executable, '-c', code, directory],
                env=dict(os.environ, HEROES_ARTIFACT_ROOT='/workspace/stale-diagnostic-root'),
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_conflicting_cached_preservation_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            old = SimpleNamespace(DURABLE_ROOT=Path('/workspace/stale-diagnostic-root'))
            with patch.dict(sys.modules, {'experiments.learner.builtin_preservation': old}):
                with self.assertRaisesRegex(ValueError, 'already imported'):
                    configure_artifact_root(directory)
            with self.assertRaisesRegex(ValueError, 'absolute'):
                configure_artifact_root('relative-root')

    def test_fresh_candidate_schedule_is_eight_evaluations_with_separate_seed_ranges(self):
        schedule = diagnostic_schedule(83410000, 83450000)
        self.assertEqual([row['learner_player'] for row in schedule], [0]*4+[1]*4)
        self.assertEqual({row['phase'] for row in schedule}, {'evaluation'})
        self.assertEqual(len({row[key] for row in schedule for key in ('engine_seed', 'policy_seed')}), 16)
        for engine, policy in ((82010000, 82050000), (None, 83450000), (83410000, 83410001),
                               (83410000, 2**31-1), (True, 83450000)):
            with self.subTest(engine=engine, policy=policy), self.assertRaises(ValueError):
                diagnostic_schedule(engine, policy)

    def test_seed_reservation_keeps_failed_and_unstarted_ranges_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ledger = root/'seeds.jsonl'
            schedule = diagnostic_schedule(83410000, 83450000)
            arguments = dict(output=root/'attempt1', parent_run=root/'official', candidate={'sha256': 'c'*64})
            reserve_diagnostic_seeds(ledger, schedule, **arguments)
            first = ledger.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already consumed'):
                reserve_diagnostic_seeds(ledger, schedule, **arguments)
            with self.assertRaisesRegex(ValueError, 'already consumed'):
                reserve_diagnostic_seeds(ledger, diagnostic_schedule(83410007, 83460000), **arguments)
            self.assertEqual(first, ledger.read_bytes())
            reserve_diagnostic_seeds(ledger, diagnostic_schedule(83420000, 83460000), **arguments)
            self.assertTrue(ledger.read_bytes().startswith(first))
            self.assertEqual(len(ledger.read_text().splitlines()), 2)

    def test_failed_parent_requires_exact_selected_update_preservation_and_owned_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'official'; run.mkdir()
            config = dict(default_profile='original', run_root=str(root))
            values = {'session.json': dict(profile='original', run_dir=str(run), run_root=str(root)),
                'campaign-report.json': dict(status='stopped_failure', owned_cleanup_verified=True,
                    initial_state_unchanged=True, full_final_restore={'exact': True},
                    latest_progress={'updates': 770}, latest_parent={'policy': 'synthetic binding'}),
                'shutdown.json': dict(owned_cleanup_verified=True, within_hard_deadline=True, worker_returncode=1)}
            def save():
                for name, value in values.items(): (run/name).write_text(json.dumps(value))
            save()
            self.assertEqual(stopped_candidate_parent(run, config), values['campaign-report.json']['latest_parent'])
            for filename, field, bad in (('campaign-report.json', 'status', 'running'),
                ('campaign-report.json', 'latest_progress', {'updates': 769}),
                ('campaign-report.json', 'full_final_restore', {'exact': False}),
                ('shutdown.json', 'owned_cleanup_verified', False), ('shutdown.json', 'worker_returncode', -9)):
                original = values[filename][field]; values[filename][field] = bad; save()
                with self.subTest(field=field), self.assertRaises(ValueError): stopped_candidate_parent(run, config)
                values[filename][field] = original
            save()
            with self.assertRaises(ValueError): stopped_candidate_parent(run, dict(config, default_profile='easier8x8'))

            values['campaign-report.json']['latest_progress']['updates'] = 916
            save()
            with self.assertRaises(ValueError): stopped_candidate_parent(run, config)
            self.assertEqual(stopped_candidate_parent(run, config, 916),
                             values['campaign-report.json']['latest_parent'])
            for expected in (770, 915, 917, 0, -1, True, '916', None):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    stopped_candidate_parent(run, config, expected)


if __name__ == '__main__': unittest.main()
