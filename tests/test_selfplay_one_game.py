"""Isolated evaluator checks use disposable files and mocks; never native games."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import tarfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_selfplay_one_game as evaluation


class OneFrozenGameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heroes-one-frozen-game-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'evaluation-seeds'
        patcher = patch.object(evaluation, 'BASE', self.base)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_default_check_never_reserves_launches_or_acquires_campaign_lease(self):
        checked = dict(checkpoint=1753, parent={}, source={})
        with patch.object(evaluation, 'inspect_parent', return_value=checked), \
             patch.object(evaluation, 'reserve') as reserve, patch.object(evaluation, 'supervise') as launch, \
             patch.object(evaluation.campaign, 'activity_lease') as lease, patch('builtins.print'):
            self.assertEqual(evaluation.main([]), 0)
            self.assertEqual(evaluation.main(['--check']), 0)
        reserve.assert_not_called(); launch.assert_not_called(); lease.assert_not_called()
        self.assertFalse(self.base.exists())

    def test_busy_campaign_fails_before_any_reservation_or_native_launch(self):
        with patch.object(evaluation.campaign, 'activity_lease', side_effect=ValueError('Another local campaign owns the machine')), \
             patch.object(evaluation, 'prepare') as prepare, patch.object(evaluation, 'supervise') as launch, \
             patch('builtins.print'), self.assertRaisesRegex(ValueError, 'post-run report.*no game or seeds started'):
            evaluation.main(['--run'])
        prepare.assert_not_called(); launch.assert_not_called()

    def test_budget_is_twice_choices_but_day_and_guard_are_unchanged(self):
        self.assertEqual(evaluation.BUDGET, dict(ordinary_choices_per_player=9920,
            reserved_dialog_choices_per_player=320, hard_choices_per_player=10240, max_days=365, capture_seconds=1200))
        self.assertEqual(evaluation.campaign.GUARD, 'visible-interaction-budget-revealed-v1')
        with self.assertRaises(SystemExit):
            evaluation.main(['--run', '--check'])
        with self.assertRaisesRegex(ValueError, 'finite total budget'):
            evaluation.main(['--seconds', 'nan'])

    def test_storage_release_pin_keeps_every_dependency_and_runtime_exact(self):
        original = {name: ('original ' + name).encode() for name in evaluation.STORAGE_PROGRESS_PATHS
                    if name not in evaluation.STORAGE_PROGRESS_NEW}
        original['engine/runtime.cpp'] = b'accepted engine'
        reviewed = {name: ('reviewed ' + name).encode() for name in evaluation.STORAGE_PROGRESS_PATHS}
        expected = {name: hashlib.sha256(data).hexdigest() for name, data in {**original, **reviewed}.items()}
        def git_read(command, *, cwd):
            self.assertEqual(command[:2], ['git', 'archive'])
            available = original if command[2] == evaluation.BASE_COMMIT else reviewed
            self.assertIn(command[2], (evaluation.BASE_COMMIT, evaluation.STORAGE_PROGRESS_COMMIT))
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode='w') as stream:
                for name in command[3:]:
                    data = available[name]
                    entry = tarfile.TarInfo(name); entry.size = len(data)
                    stream.addfile(entry, io.BytesIO(data))
            return archive.getvalue()
        with patch.object(evaluation.subprocess, 'check_output', side_effect=git_read):
            evaluation.verify_runtime_sources(expected)
            for name in expected:
                changed = dict(expected, **{name: '0' * 64})
                with self.subTest(tampered=name), self.assertRaisesRegex(ValueError, 'changed an existing'):
                    evaluation.verify_runtime_sources(changed)
            for name in evaluation.STORAGE_PROGRESS_PATHS:
                with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'omitted a reviewed'):
                    evaluation.verify_runtime_sources({key: value for key, value in expected.items() if key != name})

    def test_one_physical_game_reservation_uses_two_policy_seeds_and_never_reuses_unstarted(self):
        self.assertEqual(evaluation.seed_ledger()['reservations'], [])
        self.assertFalse(self.base.exists())
        first = evaluation.read(evaluation.bound(evaluation.reserve(self.root / 'never-started')))
        second = evaluation.read(evaluation.bound(evaluation.reserve(self.root / 'second-manual-attempt')))
        self.assertEqual(first['schedule'], dict(episode=1, engine_seed=1960000000, policy_seeds=[1962000000, 1962000001]))
        self.assertEqual(second['schedule'], dict(episode=1, engine_seed=1960000001, policy_seeds=[1962000002, 1962000003]))
        self.assertTrue(first['consumed_including_unstarted'])
        before = (self.base / 'seed-reservations.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            evaluation.reserve(self.root / 'never-started')
        self.assertEqual((self.base / 'seed-reservations.json').read_bytes(), before)

    def test_missing_or_altered_ledger_and_receipt_fail_closed(self):
        receipt = evaluation.reserve(self.root / 'first')
        ledger_path = self.base / 'seed-reservations.json'
        before = ledger_path.read_bytes(); ledger_path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing while prior artifacts'):
            evaluation.reserve(self.root / 'second')
        self.assertFalse(ledger_path.exists())
        ledger_path.write_bytes(before)
        raw = evaluation.read(ledger_path); raw['reservations'] = []
        evaluation.write(ledger_path, raw)
        with self.assertRaisesRegex(ValueError, 'Unmatched evaluation reservation'):
            evaluation.seed_ledger()
        ledger_path.write_bytes(before)
        Path(receipt['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            evaluation.reserve(self.root / 'second')
        self.assertEqual(ledger_path.read_bytes(), before)

    def capture_fixture(self):
        folder = self.root / 'episode-000001'; folder.mkdir()
        plan = dict(schedule=evaluation.seed_row(0), model_checkpoint='published-1753', weights_sha256='a' * 64,
            runtime={'runtime': 'bound'}, native={'map_sha256': 'map'}, data='/unused/data', replay_binary={})
        manifest = dict(episode='1', seed=1960000000, environment=evaluation.campaign.INTERFACE, status='truncated',
            identity={'native_runtime_sha256': plan['runtime'], **plan['native']})
        raw = {'paired': 'diagnostic'}
        evaluation.write(folder / 'manifest.json', manifest, new=True)
        evaluation.write(folder / 'neural-rollout.json', raw, new=True)
        models = tuple(SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {})) for _ in range(2))
        rollouts = [SimpleNamespace(player=i, purpose='diagnostic', origin='native',
            role='learner' if i == 0 else 'frozen_opponent', optimization_eligible=False,
            opponent_pool=evaluation.campaign.POOL, behavior_checkpoint='published-1753', opponent_checkpoint='published-1753',
            weights_sha256='a' * 64, opponent_weights_sha256='a' * 64, samples=[],
            decisions=[SimpleNamespace(decision_seq=0)], history=[SimpleNamespace(day=365)]) for i in range(2)]
        collected = SimpleNamespace(to_dict=lambda: raw, manifest_json=json.dumps(manifest),
                                    rollouts=rollouts, model_configs_json=('{}', '{}'))
        return folder, plan, models, collected

    def verify_fixture(self, mutate=None):
        from experiments.learner import contextual_rollout, contextual_model, selfplay_training
        from experiments.headless import contextual_replay
        folder, plan, models, collected = self.capture_fixture()
        if mutate is not None:
            mutate(collected)
        replay = dict(schema=contextual_replay.REPORT_SCHEMA, verified=True,
                      histories={'verified': True, 'players': {'red': {}, 'blue': {}}})
        def verify(*args, output, **kwargs):
            evaluation.write(output / 'contextual-replay-verification.json', replay, new=True)
            return replay
        with patch.object(contextual_rollout.CollectedEpisode, 'from_dict', return_value=collected), \
             patch.object(contextual_rollout, 'behavior_guard_version', return_value=evaluation.campaign.GUARD), \
             patch.object(contextual_model, 'weights_sha256', return_value='a' * 64), \
             patch.object(selfplay_training, '_files', return_value={'capture': 'unchanged'}), \
             patch.object(selfplay_training, 'reconstruct_policies', return_value=[{'player': 0}, {'player': 1}]) as reconstruct, \
             patch.object(contextual_replay, 'verify_episode', side_effect=verify) as native, \
             patch.object(evaluation, 'bound', return_value=Path('/unused/replay')), \
             patch.object(evaluation, 'check_identity'):
            result = evaluation.verify_capture(folder, plan, models, lambda: 30)
        reconstruct.assert_called_once(); native.assert_called_once()
        return result

    def test_verification_requires_both_frozen_policies_and_native_replay_without_training_receipts(self):
        from experiments.learner import selfplay_training
        with patch.object(selfplay_training, 'create_capture_binding') as admit_binding, \
             patch.object(selfplay_training, 'native_update') as update:
            proof, _ = self.verify_fixture()
        self.assertEqual(proof['choices_by_player'], [1, 1])
        self.assertFalse(proof['training_input']); self.assertEqual(proof['optimizer_updates'], 0)
        admit_binding.assert_not_called(); update.assert_not_called()

    def test_capture_rejects_training_purpose_or_excessive_choices_or_day366(self):
        changes = (
            (lambda c: setattr(c.rollouts[0], 'purpose', 'training'), 'exclusion differs'),
            (lambda c: setattr(c.rollouts[0], 'decisions', c.rollouts[0].decisions * 10241), 'sampled-day boundary'),
            (lambda c: setattr(c.rollouts[1].history[0], 'day', 366), 'sampled-day boundary'),
        )
        for mutate, message in changes:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory(dir=self.root) as directory:
                    original, self.root = self.root, Path(directory)
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            self.verify_fixture(mutate)
                    finally:
                        self.root = original

    def test_capture_receives_doubled_per_player_limits_and_preserves_day_guard_cleanup(self):
        from experiments.learner import contextual_rollout, contextual_storage
        from experiments.headless import env, process_deadline
        output = self.root / 'capture'; output.mkdir()
        events, observed = [], {}
        collected = SimpleNamespace(to_dict=lambda: {'diagnostic': True})
        class FakeCollector:
            def __init__(self, models, checkpoints, **kwargs):
                observed['collector'] = kwargs
            def act(self, request):
                return 'sampled'
            def run_episode(self, environment, **kwargs):
                observed['run'] = kwargs
                self.act(SimpleNamespace(record={'day': 365}))
                with self_test.assertRaisesRegex(ValueError, 'day365'):
                    self.act(SimpleNamespace(record={'day': 366}))
                return collected
        class FakeEnv:
            def __init__(self, launcher, folder, limits, **kwargs):
                observed['limits'], observed['environment'] = limits, kwargs
                self.episode_dir = folder / 'episode-000001'; self.episode_dir.mkdir()
            def close(self): events.append('environment_closed')
        class FakeLauncher:
            def __init__(self, *args, **kwargs): observed['launcher'] = kwargs
            def close_owned_groups(self, **kwargs): events.append('groups_closed')
            def assert_closed(self): events.append('groups_verified')
        self_test = self
        plan = dict(output=str(output), binary='/unused/client', data='/unused/data', game_map='/unused/map',
            model_checkpoint='published-1753', schedule=evaluation.seed_row(0),
            hard_deadline_monotonic=time.monotonic() + 3600, hard_deadline_unix=time.time() + 3600)
        with patch.object(contextual_rollout, 'ContextualCollector', FakeCollector), \
             patch.object(env, 'HeadlessSelfPlayEnv', FakeEnv), \
             patch.object(process_deadline, 'DeadlineVcmiLauncher', FakeLauncher), \
             patch.object(contextual_storage, 'save_collected') as save:
            folder = evaluation.collect(plan, (object(), object()))
        save.assert_called_once_with(folder / 'neural-rollout.json', collected)
        self.assertEqual(observed['collector']['purpose'], 'diagnostic')
        self.assertEqual(observed['collector']['guard_version'], evaluation.campaign.GUARD)
        self.assertEqual(observed['run']['max_requests_per_player'], 9920)
        self.assertEqual(observed['run']['max_dialog_requests_per_player'], 320)
        self.assertEqual((observed['limits'].max_actions, observed['limits'].max_days), (10240, 365))
        self.assertEqual(events, ['environment_closed', 'groups_closed', 'groups_verified'])

    def worker_simulation(self, *, change_plan=False):
        import signal
        from validation.probes.validate_hero_policy import wall_limit
        output = self.root / 'worker'; output.mkdir()
        plan = dict(output=str(output), supervisor_pid=os.getppid(), protected={}, execution={},
            hard_deadline_monotonic=time.monotonic() + 300, hard_deadline_unix=time.time() + 300)
        path = output / 'plan.json'; evaluation.write(path, plan, new=True)
        def capture(*args):
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
            with wall_limit(10):
                self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            return output
        def verify(*args):
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            evaluation.write(output / 'evaluation-policy-reconstruction.json', {}, new=True)
            if change_plan:
                evaluation.write(path, {**plan, 'changed': True})
            return ({'choices_by_player': [1, 1]}, dict(status='truncated', termination_reason='collection_window',
                    outcomes_by_player={'red': None, 'blue': None}, rewards_by_player={'red': 0, 'blue': 0}))
        with patch.object(evaluation, 'check_identity'), patch.object(evaluation.campaign, 'configure_native'), \
             patch.object(evaluation, 'frozen_models', return_value=(object(), object())), \
             patch.object(evaluation, 'collect', side_effect=capture), patch.object(evaluation, 'verify_capture', side_effect=verify):
            result = evaluation.worker(path, evaluation.binding(path)['sha256'])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        return result

    def test_worker_does_not_nest_capture_and_verification_deadline_timers(self):
        result = self.worker_simulation()
        self.assertEqual(result['status'], 'passed', result.get('error'))

    def test_worker_rejects_a_plan_changed_during_verification(self):
        result = self.worker_simulation(change_plan=True)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('plan changed during execution', result['error'])
        self.assertEqual(result['optimizer_updates'], 0)

    def test_same_checkpoint_ids_with_separate_models_reconstruct_both_synthetic_players(self):
        import torch
        from experiments.learner import contextual as c, contextual_model as m, selfplay_training
        from experiments.learner.contextual_rollout import ContextualCollector, CollectedEpisode
        from tests.test_selfplay_collection import AlternatingTurns
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng():
                torch.manual_seed(1753)
                first = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(c.EXPANDED_ENVIRONMENT),
                    hidden_size=8, feature_size=4)).eval().requires_grad_(False)
                models = first, deepcopy(first)
                self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(models[0].parameters(), models[1].parameters())))
                collector = ContextualCollector(models, ('published-1753',) * 2, learner_player=0,
                    opponent_pool=evaluation.campaign.POOL, interface=c.EXPANDED_ENVIRONMENT, purpose='diagnostic',
                    origin='synthetic', policy_seeds=(1, 2), guard_version=evaluation.campaign.GUARD)
                result = collector.run_episode(AlternatingTurns(), episode=1, seed=1,
                    max_requests_per_player=2, max_dialog_requests_per_player=1)
                loaded = CollectedEpisode.from_dict(result.to_dict())
                proof = selfplay_training.reconstruct_policies(loaded, models, (1, 2), lambda: 30)
                self.assertEqual([row['maximum_absolute_difference'] for row in proof], [0, 0])
                self.assertFalse(any(rollout.optimization_eligible for rollout in loaded.rollouts))
        finally:
            torch.set_num_threads(threads)

    def test_integrity_failure_still_persists_supervisor_cleanup_receipt(self):
        from experiments.headless import process_deadline
        output = self.root / 'supervised'; output.mkdir()
        plan = dict(output=str(output), protected={}, hard_deadline_monotonic=time.monotonic() + 300,
                    hard_deadline_unix=time.time() + 300)
        evaluation.write(output / 'plan.json', plan, new=True)
        evaluation.write(output / 'report.json', {'status': 'passed'}, new=True)
        process = SimpleNamespace(returncode=0, wait=lambda **kwargs: None)
        with patch.object(process_deadline, 'guarded_popen', return_value=process), \
             patch.object(process_deadline, 'close_owned_process') as close, \
             patch.object(evaluation.campaign, 'verify_protected', side_effect=ValueError('published pointer changed')), \
             self.assertRaisesRegex(ValueError, 'Evaluation stopped'):
            evaluation.supervise(plan)
        close.assert_called_once()
        shutdown = evaluation.read(output / 'shutdown.json')
        self.assertTrue(shutdown['owned_cleanup_verified'])
        self.assertFalse(shutdown['protected_inputs_unchanged'])
        self.assertIn('published pointer changed', shutdown['protection_error'])

    def test_supervisor_cleanup_failure_ends_stale_game_progress(self):
        from experiments.headless import process_deadline
        from experiments.learner.game_progress import GameProgress, progress_path
        output = self.root / 'progress-failure'; output.mkdir()
        plan = dict(output=str(output), protected={}, hard_deadline_monotonic=time.monotonic() + 300,
                    hard_deadline_unix=time.time() + 300)
        evaluation.write(output / 'plan.json', plan, new=True)
        progress = GameProgress(progress_path(output, 1), episode=1, ordinary=9920, reserved=320)
        progress((234, 123), 'collecting')
        process = SimpleNamespace(returncode=-9, wait=lambda **kwargs: -9)
        with patch.object(process_deadline, 'guarded_popen', return_value=process), \
             patch.object(process_deadline, 'close_owned_process', side_effect=RuntimeError('cleanup failed')), \
             patch.object(evaluation.campaign, 'verify_protected', return_value=True), \
             patch('sys.stderr', new=io.StringIO()) as display, \
             self.assertRaisesRegex(ValueError, 'Evaluation stopped'):
            evaluation.supervise(plan)
        saved = evaluation.read(progress_path(output, 1))
        self.assertEqual(saved['phase'], 'failed')
        self.assertEqual(saved['choices'], [234, 123])
        self.assertIn('Run failed or interrupted', display.getvalue())
        self.assertFalse(evaluation.read(output / 'shutdown.json')['owned_cleanup_verified'])

    def test_missing_plan_still_persists_failed_shutdown_and_cleanup(self):
        from experiments.headless import process_deadline
        output = self.root / 'supervised'; output.mkdir()
        plan = dict(output=str(output), protected={}, hard_deadline_monotonic=time.monotonic() + 300,
                    hard_deadline_unix=time.time() + 300)
        path = output / 'plan.json'; evaluation.write(path, plan, new=True)
        evaluation.write(output / 'report.json', {'status': 'passed', 'plan': evaluation.binding(path)}, new=True)
        process = SimpleNamespace(returncode=0, wait=lambda **kwargs: path.unlink())
        with patch.object(process_deadline, 'guarded_popen', return_value=process), \
             patch.object(process_deadline, 'close_owned_process') as close, \
             self.assertRaisesRegex(ValueError, 'Evaluation stopped'):
            evaluation.supervise(plan)
        close.assert_called_once()
        shutdown = evaluation.read(output / 'shutdown.json')
        self.assertTrue(shutdown['owned_cleanup_verified'])
        self.assertFalse(shutdown['plan_unchanged'])
        self.assertIn('FileNotFoundError', shutdown['plan_error'])


if __name__ == '__main__':
    unittest.main()
