"""Disposable offline checks for the manual tiny-model diagnostic launcher."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_tiny_selfplay_one_game as tiny
from tools import run_selfplay_one_game as evaluation


class TinyOneGameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heroes-tiny-one-game-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / 'tiny-seeds'
        patcher = patch.object(tiny, 'BASE', self.base)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_default_check_has_no_game_seed_lease_or_optimizer_effect(self):
        checked = dict(model_config=tiny.candidate.specification()[1].to_dict(), source={'bound': True})
        with patch.object(tiny, 'inspect_candidate', return_value=checked), \
             patch.object(tiny, 'reserve') as reserve, patch.object(tiny, 'prepare') as prepare, \
             patch.object(evaluation, 'supervise') as launch, \
             patch.object(tiny.campaign, 'activity_lease') as lease, \
             patch('sys.stdout', new=io.StringIO()) as printed:
            self.assertEqual(tiny.main([]), 0)
            self.assertEqual(tiny.main(['--check']), 0)
        for method in (reserve, prepare, launch, lease): method.assert_not_called()
        self.assertFalse(self.base.exists())
        result = json.JSONDecoder().raw_decode(printed.getvalue())[0]
        self.assertEqual(result['checkpoint'], tiny.CHECKPOINT)
        self.assertEqual(result['initialization'], 'random_initialization')
        self.assertFalse(result['gameplay_performed'])
        self.assertFalse(result['training_launcher_accepted'])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertNotIn('exact_model_adam_rng_progress_sampler_pool', result)

    def test_invalid_budget_and_busy_machine_stop_before_prepare(self):
        with patch.object(tiny, 'prepare') as prepare, patch.object(evaluation, 'supervise') as launch, \
             patch.object(tiny, 'inspect_candidate') as inspect, patch('builtins.print'):
            for budget in ('nan', 'inf', '179', '7201'):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'finite total budget'):
                    tiny.main(['--run', '--seconds', budget])
            with patch.object(tiny.campaign, 'activity_lease', side_effect=ValueError('busy')), \
                 self.assertRaisesRegex(ValueError, 'busy'):
                tiny.main(['--run'])
        for method in (prepare, launch, inspect): method.assert_not_called()

    def test_explicit_run_dispatches_tiny_worker_and_worker_binds_all_hooks(self):
        plan = {'output': 'mock-only'}
        with patch.object(tiny.campaign, 'activity_lease', return_value=nullcontext()), \
             patch.object(tiny, 'prepare', return_value=plan), \
             patch.object(evaluation, 'supervise', return_value={'status': 'passed'}) as supervise, \
             patch('builtins.print'):
            self.assertEqual(tiny.main(['--run']), 0)
        self.assertEqual(supervise.call_args.args, (plan,))
        self.assertEqual(supervise.call_args.kwargs['worker_script'], Path(tiny.__file__).resolve())
        self.assertEqual(supervise.call_args.kwargs['result_schema'], tiny.SCHEMA)
        self.assertEqual(supervise.call_args.kwargs['checkpoint'], tiny.CHECKPOINT)
        with patch.object(evaluation, 'worker', return_value={'status': 'failed'}) as worker:
            self.assertEqual(tiny.main(['_worker', '--plan', '/unused/plan', '--sha256', 'digest']), 1)
        worker.assert_called_once_with(Path('/unused/plan'), 'digest', identity_check=tiny.check_identity,
            model_loader=tiny.frozen_models, result_schema=tiny.SCHEMA, checkpoint=tiny.CHECKPOINT)
        with patch.object(evaluation, 'worker') as worker, self.assertRaisesRegex(ValueError, 'bound plan'):
            tiny.main(['_worker', '--run'])
        worker.assert_not_called()

    def test_seed_namespace_is_separate_and_unstarted_reservations_are_consumed(self):
        self.assertEqual(tiny.seed_ledger()['reservations'], [])
        self.assertFalse(self.base.exists())
        first = tiny.read(tiny.bound(tiny.reserve(self.root / 'unstarted')))
        second = tiny.read(tiny.bound(tiny.reserve(self.root / 'second')))
        self.assertEqual(first['schedule'], dict(episode=1, engine_seed=1974000000,
            policy_seeds=[1976000000, 1976000001]))
        self.assertEqual(second['schedule'], tiny.seed_row(1))
        self.assertTrue(first['consumed_including_unstarted'])
        new_seeds = {row['schedule']['engine_seed'] for row in (first, second)}
        new_seeds.update(seed for row in (first, second) for seed in row['schedule']['policy_seeds'])
        old = evaluation.seed_row(99999)
        self.assertTrue(new_seeds.isdisjoint({old['engine_seed'], *old['policy_seeds'], 1970000001,
                                            1970000002, 1970000003}))
        before = (self.base / 'seed-reservations.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            tiny.reserve(self.root / 'unstarted')
        self.assertEqual((self.base / 'seed-reservations.json').read_bytes(), before)
        for offset in (-1, 100000, True, 0.5):
            with self.subTest(offset=offset), self.assertRaises(ValueError): tiny.seed_row(offset)

    def test_missing_or_altered_seed_evidence_rejects_without_reusing_seed(self):
        receipt = tiny.reserve(self.root / 'first')
        ledger = self.base / 'seed-reservations.json'
        original = ledger.read_bytes(); ledger.unlink()
        with self.assertRaisesRegex(ValueError, 'missing while evidence'):
            tiny.reserve(self.root / 'second')
        self.assertFalse(ledger.exists())
        ledger.write_bytes(original)
        changed = tiny.read(ledger); changed['reservations'][0]['schedule']['policy_seeds'][0] += 1
        tiny.write(ledger, changed)
        with self.assertRaisesRegex(ValueError, 'reservation changed'): tiny.seed_ledger()
        ledger.write_bytes(original)
        Path(receipt['path']).unlink()
        with self.assertRaises(FileNotFoundError): tiny.reserve(self.root / 'second')
        self.assertEqual(ledger.read_bytes(), original)

    def policy_fixture(self, *, metadata=None, provenance='random_initialization', pool=True, config=None):
        import torch
        from experiments.learner import checkpoints as cp, contextual_model as model_api
        config = config or tiny.candidate.specification()[1]
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model = model_api.ContextualActorCritic(config)
            opponents = cp.FrozenOpponentPool() if pool else None
            if opponents is not None: opponents.add(tiny.CHECKPOINT, model)
            path = self.root / ('policy-' + str(len(list(self.root.glob('policy-*')))) + '.pt')
            cp.save_policy_checkpoint(path, model, checkpoint=tiny.CHECKPOINT, provenance=provenance,
                source_metadata=tiny.expected_metadata() if metadata is None else metadata,
                opponent_pool=opponents, interface=tiny.campaign.INTERFACE)
        return dict(tiny_policy=tiny.binding(path), weights_sha256=model_api.weights_sha256(model),
                    model_checkpoint=tiny.CHECKPOINT, tiny_metadata=tiny.expected_metadata())

    def pinned_fixture(self, plan):
        stack = ExitStack()
        stack.enter_context(patch.object(tiny, 'POLICY_SHA', plan['tiny_policy']['sha256']))
        stack.enter_context(patch.object(tiny, 'WEIGHTS_SHA', plan['weights_sha256']))
        return stack

    def test_real_checkpoint_load_yields_separate_frozen_tensors_without_rng_or_adam_changes(self):
        import torch
        import random
        from experiments.learner import checkpoints as cp, contextual_model
        plan = self.policy_fixture()
        before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
        before_bytes = Path(plan['tiny_policy']['path']).read_bytes()
        with self.pinned_fixture(plan), patch.object(cp, 'load_checkpoint') as resume, \
             patch.object(torch.optim, 'Adam') as adam:
            models = tiny.frozen_models(plan)
        resume.assert_not_called(); adam.assert_not_called()
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        self.assertEqual(Path(plan['tiny_policy']['path']).read_bytes(), before_bytes)
        self.assertEqual(len(models), 2)
        self.assertTrue(all(not model.training for model in models))
        self.assertTrue(all(not p.requires_grad for model in models for p in model.parameters()))
        self.assertFalse({p.data_ptr() for p in models[0].parameters()} &
                         {p.data_ptr() for p in models[1].parameters()})
        self.assertEqual([contextual_model.weights_sha256(model) for model in models], [plan['weights_sha256']] * 2)

    def test_wrong_provenance_metadata_shape_and_missing_pool_are_rejected(self):
        from experiments.learner.contextual_model import ModelConfig
        config = tiny.candidate.specification()[1].to_dict()
        config['hidden_size'] = 16
        cases = (
            {'provenance': 'synthetic_optimizer_test'},
            {'metadata': {**tiny.expected_metadata(), 'initialized_from_checkpoint': 1753}},
            {'pool': False},
            {'config': ModelConfig.from_dict(config)},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                plan = self.policy_fixture(**kwargs)
                with self.pinned_fixture(plan), self.assertRaises(ValueError): tiny.frozen_models(plan)

    def test_policy_plan_identity_and_corrupt_file_reject_before_loading(self):
        from experiments.learner import checkpoints as cp
        plan = self.policy_fixture()
        changes = ({'model_checkpoint': 'published-1753'}, {'weights_sha256': '0' * 64},
                   {'tiny_metadata': {**plan['tiny_metadata'], 'training_games': 1}})
        with self.pinned_fixture(plan), patch.object(cp, 'load_policy_checkpoint') as loader:
            for changeset in changes:
                with self.subTest(changeset=changeset), self.assertRaisesRegex(ValueError, 'initialization identity'):
                    tiny.frozen_models({**plan, **changeset})
            loader.assert_not_called()
            with Path(plan['tiny_policy']['path']).open('ab') as target: target.write(b'changed')
            with self.assertRaises(ValueError): tiny.frozen_models(plan)
            loader.assert_not_called()

    def test_shared_loaded_model_storage_is_rejected(self):
        from experiments.learner import checkpoints as cp
        plan = self.policy_fixture()
        loaded = cp.load_policy_checkpoint(Path(plan['tiny_policy']['path']),
            expected_source_metadata=tiny.expected_metadata(), interface=tiny.campaign.INTERFACE)
        with self.pinned_fixture(plan), patch.object(cp, 'load_policy_checkpoint', return_value=loaded), \
             self.assertRaisesRegex(ValueError, 'independent parameter storage'):
            tiny.frozen_models(plan)

    def test_inspection_protects_both_states_without_claiming_tiny_parent_or_adam(self):
        plan = self.policy_fixture()
        baseline_file = self.root / 'baseline.pt'; baseline_file.write_bytes(b'preserved baseline')
        baseline_binding = tiny.binding(baseline_file)
        baseline = dict(parent={'checkpoint': 1753}, checkpoint=1753, model_checkpoint='published-1753',
            weights_sha256='old', optimizer_steps=1753, source={'old': True},
            protected={str(baseline_file): baseline_binding}, binary='/unused/client')
        preparation = dict(status='prepared_untrained', progress=dict(updates=0, rollouts=0, decisions=0),
            policy=plan['tiny_policy'], weights_sha256=plan['weights_sha256'], exact_model_rng_pool_restore=True,
            training_launcher_accepted=False, protected={}, config=tiny.candidate.specification()[1].to_dict())
        receipt = self.root / 'preparation.json'; tiny.write(receipt, preparation, new=True)
        with self.pinned_fixture(plan), patch.object(tiny, 'PREPARATION', receipt), \
             patch.object(tiny, 'PREPARATION_SHA', tiny.binding(receipt)['sha256']), \
             patch.object(evaluation, 'inspect_parent', return_value=baseline), \
             patch.object(tiny, 'source_identity', return_value={'new': True}):
            checked = tiny.inspect_candidate()
            self.assertEqual(checked['comparison_baseline'], baseline['parent'])
            self.assertNotIn('parent', checked)
            self.assertNotIn('optimizer_steps', checked)
            self.assertIsNone(checked['initialized_from_checkpoint'])
            self.assertEqual(checked['optimizer_state'], 'not created')
            self.assertEqual(checked['checkpoint'], tiny.CHECKPOINT)
            self.assertIn(str(baseline_file), checked['protected'])
            self.assertIn(plan['tiny_policy']['path'], checked['protected'])
            self.assertIn(str(receipt), checked['protected'])
            baseline_file.write_bytes(b'changed')
            with self.assertRaises(ValueError): tiny.inspect_candidate()

    def test_modified_preparation_or_nonzero_progress_rejects_before_models(self):
        receipt = self.root / 'preparation.json'
        prepared = dict(status='prepared_untrained', progress=dict(updates=0, rollouts=0, decisions=0),
            policy={'sha256': tiny.POLICY_SHA}, weights_sha256=tiny.WEIGHTS_SHA,
            exact_model_rng_pool_restore=True, training_launcher_accepted=False)
        tiny.write(receipt, prepared, new=True)
        accepted_sha = tiny.binding(receipt)['sha256']
        with patch.object(evaluation, 'inspect_parent', return_value={}), \
             patch.object(tiny, 'PREPARATION', receipt), patch.object(tiny, 'frozen_models') as models:
            changed = deepcopy(prepared); changed['progress']['updates'] = 1
            tiny.write(receipt, changed)
            with patch.object(tiny, 'PREPARATION_SHA', accepted_sha), \
                 self.assertRaisesRegex(ValueError, 'receipt changed'):
                tiny.inspect_candidate()
            with patch.object(tiny, 'PREPARATION_SHA', tiny.binding(receipt)['sha256']), \
                 self.assertRaisesRegex(ValueError, 'preserved untrained initialization'):
                tiny.inspect_candidate()
            models.assert_not_called()

    def identity_fixture(self):
        files = {}
        for name in ('preparation', 'policy', 'replay', 'native'):
            path = self.root / (name + '.json'); path.write_text('{}')
            files[name] = tiny.binding(path)
        reservation = tiny.reserve(self.root / 'run')
        plan = dict(schema=tiny.SCHEMA + '-plan', source={'frozen': True}, budget=tiny.BUDGET,
            physical_games=1, purpose='diagnostic', training_input=False, optimizer_updates=0,
            policy_support=tiny.campaign.GUARD, checkpoint=tiny.CHECKPOINT,
            initialization='random_initialization', initialized_from_checkpoint=None,
            training_launcher_accepted=False, parameters=15460,
            model_config=tiny.candidate.specification()[1].to_dict(), optimizer_state='not created',
            binary='/unused/client', runtime={'binary': 'bound'}, replay_binary=files['replay'],
            native_manifest=files['native'], preparation=files['preparation'], tiny_policy=files['policy'],
            reservation=reservation, schedule=tiny.seed_row(0), protected={})
        return plan

    def test_source_runtime_and_learning_plan_tampering_fail_closed(self):
        from experiments.headless import contextual_evidence
        plan = self.identity_fixture()
        with patch.object(tiny, 'source_identity', return_value=plan['source']), \
             patch.object(tiny, 'PREPARATION', Path(plan['preparation']['path'])), \
             patch.object(tiny, 'PREPARATION_SHA', plan['preparation']['sha256']), \
             patch.object(contextual_evidence, 'runtime_identity', return_value=plan['runtime']):
            tiny.check_identity(plan)
            changes = ({'source': {}}, {'runtime': {}}, {'physical_games': 2}, {'purpose': 'training'},
                       {'training_input': True}, {'optimizer_updates': 1}, {'policy_support': 'none'},
                       {'checkpoint': 'published-1753'}, {'initialization': 'distillation'},
                       {'initialized_from_checkpoint': 1753}, {'training_launcher_accepted': True},
                       {'parameters': 74344}, {'optimizer_state': 'restored'},
                       {'model_config': {**plan['model_config'], 'hidden_size': 64}},
                       {'budget': {**tiny.BUDGET, 'max_days': 366}}, {'schedule': tiny.seed_row(1)})
            for changeset in changes:
                with self.subTest(changeset=changeset), self.assertRaises(ValueError):
                    tiny.check_identity({**plan, **changeset})

    def test_worker_uses_tiny_hooks_and_never_passes_failed_verification(self):
        for fail in (False, True):
            with self.subTest(failed_verification=fail):
                output = self.root / ('worker-' + str(fail)); output.mkdir()
                plan = dict(output=str(output), supervisor_pid=os.getppid(), protected={}, execution={},
                    hard_deadline_monotonic=time.monotonic() + 300, hard_deadline_unix=time.time() + 300)
                path = output / 'plan.json'; tiny.write(path, plan, new=True)
                identity, loader = Mock(), Mock(return_value=(object(), object()))
                def verify(*args, identity_check):
                    self.assertIs(identity_check, identity)
                    if fail: raise ValueError('native replay rejected')
                    identity_check(plan)
                    tiny.write(output / 'evaluation-policy-reconstruction.json', {}, new=True)
                    return ({'choices_by_player': [1, 1]}, dict(status='completed', termination_reason='victory',
                        outcomes_by_player={'red': 'win', 'blue': 'loss'}, rewards_by_player={'red': 1, 'blue': -1}))
                with patch.object(tiny.campaign, 'configure_native'), \
                     patch.object(evaluation, 'collect', return_value=output), \
                     patch.object(evaluation, 'verify_capture', side_effect=verify):
                    report = evaluation.worker(path, tiny.binding(path)['sha256'], identity_check=identity,
                        model_loader=loader, result_schema=tiny.SCHEMA, checkpoint=tiny.CHECKPOINT)
                self.assertEqual(report['status'], 'failed' if fail else 'passed')
                self.assertEqual(report['schema'], tiny.SCHEMA + '-result')
                self.assertEqual(report['checkpoint'], tiny.CHECKPOINT)
                self.assertEqual(report['optimizer_updates'], 0)
                self.assertEqual(report['training_samples_admitted'], 0)
                self.assertFalse(report['training_input'])
                self.assertEqual(loader.call_count, 2)
                if fail: self.assertNotIn('native_replay_verified', report)
                self.assertEqual(tiny.read(output / 'report.json'), report)

    def test_supervisor_uses_tiny_driver_in_guarded_child_command(self):
        from experiments.headless import process_deadline
        output = self.root / 'supervised'; output.mkdir()
        plan = dict(output=str(output), protected={}, hard_deadline_monotonic=time.monotonic() + 300,
                    hard_deadline_unix=time.time() + 300)
        tiny.write(output / 'plan.json', plan, new=True)
        report = dict(status='passed', schema=tiny.SCHEMA + '-result', checkpoint=tiny.CHECKPOINT,
                      plan=tiny.binding(output / 'plan.json'))
        tiny.write(output / 'report.json', report, new=True)
        process = SimpleNamespace(returncode=0, wait=lambda **kwargs: None)
        with patch.object(process_deadline, 'guarded_popen', return_value=process) as spawn, \
             patch.object(process_deadline, 'close_owned_process') as close:
            result = evaluation.supervise(plan, worker_script=Path(tiny.__file__),
                                          result_schema=tiny.SCHEMA, checkpoint=tiny.CHECKPOINT)
        self.assertEqual(result, report)
        self.assertEqual(spawn.call_args.args[0][1:3], [str(Path(tiny.__file__).resolve()), '_worker'])
        close.assert_called_once()
        self.assertTrue(tiny.read(output / 'shutdown.json')['owned_cleanup_verified'])

    def test_supervisor_rejects_report_from_wrong_checkpoint_or_schema(self):
        from experiments.headless import process_deadline
        for index, mismatch in enumerate(({'checkpoint': 1753}, {'schema': evaluation.SCHEMA + '-result'})):
            with self.subTest(mismatch=mismatch):
                output = self.root / f'mixed-report-{index}'; output.mkdir()
                plan = dict(output=str(output), protected={}, hard_deadline_monotonic=time.monotonic() + 300,
                            hard_deadline_unix=time.time() + 300)
                tiny.write(output / 'plan.json', plan, new=True)
                report = dict(status='passed', schema=tiny.SCHEMA + '-result', checkpoint=tiny.CHECKPOINT,
                              plan=tiny.binding(output / 'plan.json'))
                tiny.write(output / 'report.json', {**report, **mismatch}, new=True)
                process = SimpleNamespace(returncode=0, wait=lambda **kwargs: None)
                with patch.object(process_deadline, 'guarded_popen', return_value=process), \
                     patch.object(process_deadline, 'close_owned_process') as close, \
                     self.assertRaisesRegex(ValueError, 'Evaluation stopped'):
                    evaluation.supervise(plan, worker_script=Path(tiny.__file__),
                                         result_schema=tiny.SCHEMA, checkpoint=tiny.CHECKPOINT)
                close.assert_called_once()

    def test_custom_identity_does_not_bypass_policy_reconstruction_or_native_history_proofs(self):
        from experiments.learner import contextual_storage, contextual_rollout, contextual_model, selfplay_training
        from experiments.headless import contextual_replay
        cases = ('passed', 'native_rejected', 'history_rejected', 'missing_blue', 'identity_changed')
        for case in cases:
            with self.subTest(case=case):
                folder = self.root / case; folder.mkdir()
                plan = dict(schedule=tiny.seed_row(0), model_checkpoint=tiny.CHECKPOINT, weights_sha256='a' * 64,
                    runtime={'runtime': 'bound'}, native={'map_sha256': 'map'}, data='/unused/data', replay_binary={})
                manifest = dict(episode='1', seed=tiny.seed_row(0)['engine_seed'], environment=tiny.campaign.INTERFACE,
                    status='truncated', identity={'native_runtime_sha256': plan['runtime'], **plan['native']})
                tiny.write(folder / 'manifest.json', manifest, new=True)
                models = tuple(SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {})) for _ in range(2))
                rollouts = [SimpleNamespace(player=i, purpose='diagnostic', origin='native',
                    role='learner' if i == 0 else 'frozen_opponent', optimization_eligible=False,
                    opponent_pool=tiny.campaign.POOL, behavior_checkpoint=tiny.CHECKPOINT,
                    opponent_checkpoint=tiny.CHECKPOINT, weights_sha256='a' * 64, opponent_weights_sha256='a' * 64,
                    samples=[], decisions=[SimpleNamespace(decision_seq=0)], history=[SimpleNamespace(day=365)])
                    for i in range(2)]
                collected = SimpleNamespace(manifest_json=json.dumps(manifest), rollouts=rollouts,
                                            model_configs_json=('{}', '{}'))
                replay = dict(schema=contextual_replay.REPORT_SCHEMA, verified=case != 'native_rejected',
                    histories=dict(verified=case != 'history_rejected', players={'red': {}, 'blue': {}}))
                if case == 'missing_blue': del replay['histories']['players']['blue']
                def native_verify(*args, output, **kwargs):
                    tiny.write(output / 'contextual-replay-verification.json', replay, new=True)
                    return replay
                identity = Mock(side_effect=ValueError('identity changed') if case == 'identity_changed' else None)
                with patch.object(contextual_storage, 'load_collected', return_value=collected), \
                     patch.object(contextual_rollout, 'behavior_guard_version', return_value=tiny.campaign.GUARD), \
                     patch.object(contextual_model, 'weights_sha256', return_value='a' * 64), \
                     patch.object(selfplay_training, '_files', return_value={'capture': 'unchanged'}), \
                     patch.object(selfplay_training, 'reconstruct_policies', return_value=[{'player': 0}, {'player': 1}]) as reconstruct, \
                     patch.object(contextual_replay, 'verify_episode', side_effect=native_verify) as native, \
                     patch.object(evaluation, 'bound', return_value=Path('/unused/replay')), \
                     patch.object(evaluation, 'check_identity') as original_identity, \
                     patch.object(selfplay_training, 'native_update') as update:
                    if case == 'passed':
                        proof, _ = evaluation.verify_capture(folder, plan, models, lambda: 30, identity_check=identity)
                        self.assertEqual(proof['choices_by_player'], [1, 1])
                        self.assertFalse(proof['training_input'])
                        self.assertEqual(proof['optimizer_updates'], 0)
                    else:
                        with self.assertRaises(ValueError):
                            evaluation.verify_capture(folder, plan, models, lambda: 30, identity_check=identity)
                        self.assertFalse((folder / 'evaluation-policy-reconstruction.json').exists())
                reconstruct.assert_called_once_with(collected, models, plan['schedule']['policy_seeds'], unittest.mock.ANY)
                native.assert_called_once(); original_identity.assert_not_called(); update.assert_not_called()
                if case in ('passed', 'identity_changed'): identity.assert_called_once_with(plan)
                else: identity.assert_not_called()


if __name__ == '__main__':
    unittest.main()
