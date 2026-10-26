"""Local adapter boundary checks; no native processes or training campaigns."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import subprocess
import unittest
from unittest.mock import patch
import zipfile

from experiments.learner import local_script_training as local


class ClockAndScheduleTests(unittest.TestCase):
    def clock(self):
        wall, mono = time.time(), time.monotonic()
        return dict(started_unix=wall-10, started_monotonic=mono-10,
                    compute_deadline_unix=wall+6890, compute_deadline_monotonic=mono+6890,
                    absolute_deadline_unix=wall+7190, hard_deadline_monotonic=mono+7190)

    def test_both_clock_domains_and_finalization_reserve(self):
        value = self.clock()
        local.check_clock(value)
        for key, delta in [('compute_deadline_monotonic', 1), ('hard_deadline_monotonic', 1),
                           ('compute_deadline_unix', 301)]:
            changed = {**value, key: value[key]+delta}
            with self.assertRaises(ValueError):
                local.check_clock(changed)
        value['compute_deadline_unix'] = value['started_unix']+1
        value['compute_deadline_monotonic'] = value['started_monotonic']+1
        with self.assertRaisesRegex(ValueError, 'expired'):
            local.check_clock(value)

    def test_nan_and_nonfinite_clocks_rejected(self):
        for invalid in (float('nan'), float('inf'), '7200', True):
            with self.assertRaisesRegex(ValueError, 'Invalid campaign clock'):
                local.check_clock({**self.clock(), 'started_unix': invalid})

    def test_fresh_training_sequences_and_fixed_both_seat_evaluation(self):
        session = dict(next_sequence=100, training_engine_seed_base=20000000,
                       training_policy_seed_base=30000000)
        observed = set()
        for sequence in range(100, 120):
            rows = local.schedule(session, sequence, sequence % 2)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row['learner_player'] for row in rows}, {sequence % 2})
            for row in rows:
                key = (row['engine_seed'], row['policy_seed'])
                self.assertNotIn(key, observed)
                observed.add(key)
        with self.assertRaisesRegex(ValueError, 'consumed'):
            local.schedule(session, 99, 0)
        for seat in (0, 1):
            self.assertEqual(local.schedule(session, 412, seat, True),
                             local.schedule(session, 500, seat, True))

    def test_native_seed_overflow_and_heldout_collision_rejected(self):
        session = dict(next_sequence=0, training_engine_seed_base=710001,
                       training_policy_seed_base=720001)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            local.schedule(session, 0, 0)
        session['training_engine_seed_base'] = 2**31-3
        with self.assertRaisesRegex(ValueError, 'limits'):
            local.schedule(session, 0, 0)


class ExactSourceAndMapTests(unittest.TestCase):
    def test_only_reviewed_local_adapter_transition_is_accepted(self):
        path = 'experiments/learner/local_script_training.py'
        current = (local.ROOT/path).read_bytes()
        for commit, digest in local.LOCAL_ADAPTER_ORIGINS.items():
            previous = subprocess.check_output(['git', 'show', commit+':'+path], cwd=local.ROOT, timeout=10)
            proof = local.local_adapter_transition(commit, digest, previous, current)
            self.assertEqual(proof['transition'], 'explicit_map_profile_and_seed_namespace')
            with self.assertRaisesRegex(ValueError, 'historical local adapter'):
                local.local_adapter_transition('0'*40, digest, previous, current)
            with self.assertRaisesRegex(ValueError, 'historical local adapter'):
                local.local_adapter_transition(commit, digest, previous+b'\n# changed', current)
            with self.assertRaisesRegex(ValueError, 'historical local adapter'):
                local.local_adapter_transition(commit, '0'*64, previous, current)
            with self.assertRaisesRegex(ValueError, 'reviewed profile transition'):
                local.local_adapter_transition(commit, digest, previous,
                    current.replace(b'original_torch, original_python =', b'changed_torch, original_python ='))

    def test_explicit_original_map_profile_uses_exact_authored_map(self):
        path = local.ROOT/'experiments/explorer/maps/hero-management-v2.vmap'
        for seat in (0, 1):
            result = local.map_identity(path, seat, 'original')
            self.assertEqual(result['scenario'], 'hero-management-v2')
            self.assertEqual(result['sha256'], 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3')
            self.assertTrue(result['semantics_verified'])
        with self.assertRaises(ValueError): local.map_identity(path, 0, 'easier8x8')
        with self.assertRaises(ValueError): local.map_identity(path, 0, 'unregistered')
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder)/'map.vmap'
            with zipfile.ZipFile(changed, 'w') as stream:
                for name, value in local._map_payloads(0).items(): stream.writestr(name, json.dumps(value))
            with self.assertRaises(ValueError): local.map_identity(changed, 0, 'original')

    def test_session_cannot_switch_profile_maps_or_seed_namespace(self):
        assets = dict(default_profile='original', maps=['map', 'map'],
                      seed_namespace=dict(engine_base=40_000_000, policy_base=50_000_000))
        session = dict(profile='original', maps=['map', 'map'],
                       training_engine_seed_base=40_000_000, training_policy_seed_base=50_000_000)
        local.check_profile(session, assets)
        for field, value in [('profile', 'easier8x8'), ('maps', ['map', 'other']),
                             ('training_engine_seed_base', 20_000_000)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                local.check_profile({**session, field: value}, assets)
        with self.assertRaisesRegex(ValueError, 'namespace'):
            local.check_profile(session, {k: v for k, v in assets.items() if k != 'seed_namespace'})

    def test_scope_route_removal_does_not_mask_collector_edits(self):
        original = 'def check_identity(plan):\n    return plan["source"]\n'
        routed = ('def check_identity(plan):\n'
            '    if plan.get("scope") == "heroes-local-script-training-v1":\n'
            '        from .local_script_training import check_identity as check_local\n'
            '        return check_local(plan)\n'
            '    return plan["source"]\n')
        self.assertEqual(local._without_scope_routes(original), local._without_scope_routes(routed))
        self.assertNotEqual(local._without_scope_routes(original),
                            local._without_scope_routes(routed.replace('plan["source"]', 'True')))
        self.assertNotEqual(local._without_scope_routes(original),
                            local._without_scope_routes(routed.replace('heroes-local-script-training-v1', 'unrecognized')))
        self.assertNotEqual(local._without_scope_routes(original),
                            local._without_scope_routes(routed.replace('return check_local(plan)', 'return True')))

    def test_invalid_hero_anchor_and_wrong_seat_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map.vmap'
            def write(value):
                with zipfile.ZipFile(path, 'w') as stream:
                    for name, content in value.items():
                        stream.writestr(name, json.dumps(content))
            payloads = local._map_payloads(0)
            write(payloads)
            self.assertEqual(local.map_identity(path, 0)['stage'], 3)
            with self.assertRaisesRegex(ValueError, 'distribution'):
                local.map_identity(path, 1)
            payloads['objects.json'][0]['x'] = 0
            write(payloads)
            with self.assertRaisesRegex(ValueError, 'distribution'):
                local.map_identity(path, 0)

    def test_relocated_bytes_and_full_dependency_are_required(self):
        local._same_bytes({'path': '/new/a', 'sha256': 'a', 'bytes': 4},
                          {'path': '/old/a', 'sha256': 'a', 'bytes': 4})
        with self.assertRaisesRegex(ValueError, 'historical bytes'):
            local._same_bytes({'sha256': 'b', 'bytes': 4}, {'sha256': 'a', 'bytes': 4})
        with self.assertRaisesRegex(ValueError, 'initial dependency'):
            local.read_descriptor({k: {} for k in ('plan', 'report', 'policy', 'resume')})


class BaselineAdmissionTests(unittest.TestCase):
    def report(self):
        session = dict(next_sequence=0, training_engine_seed_base=20000000, training_policy_seed_base=30000000)
        plan = dict(mode='evaluation', schedule=local.schedule(session, 412, 0, True))
        report = dict(schema=local.batches.EVALUATION+'-result', status='finished', training_performed=False,
            worker_complete=True, supervisor_cleanup_verified=True, within_shared_budget=True,
            optimizer_updates=0, optimizer_steps=0, training_samples=0,
            attempts=[dict(**row, status='passed', training_samples=0, completion_receipt=True,
                           cleanup_verified=True, identity_verified=True, replay_verified=True)
                      for row in plan['schedule']])
        return plan, report

    def test_failed_partial_or_admitted_baseline_is_rejected(self):
        plan, report = self.report()
        self.assertTrue(local.valid_evaluation(plan, report))
        for key, value in [('optimizer_steps', 1), ('training_samples', 1), ('optimizer_updates', 1),
                           ('worker_complete', False), ('supervisor_cleanup_verified', False),
                           ('within_shared_budget', False)]:
            self.assertFalse(local.valid_evaluation(plan, {**report, key: value}))
        broken = deepcopy(report)
        broken['attempts'] = broken['attempts'][:-1]
        self.assertFalse(local.valid_evaluation(plan, broken))
        broken = deepcopy(report)
        broken['attempts'][0]['replay_verified'] = False
        self.assertFalse(local.valid_evaluation(plan, broken))
        broken['attempts'][0]['replay_verified'] = True
        broken['attempts'][0]['engine_seed'] += 1
        self.assertFalse(local.valid_evaluation(plan, broken))

    def test_training_requires_both_seats_even_if_journal_claims_finished(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'campaign-report.json').write_text(json.dumps({'batches': []}))
            plan = {'local_session': {'path': 'fixture-session'}}
            with patch.object(local, 'read_session', return_value={'run_dir': folder}):
                with self.assertRaisesRegex(ValueError, 'Both current-runtime'):
                    local.require_baselines(plan)


if __name__ == '__main__':
    unittest.main()
