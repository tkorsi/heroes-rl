"""Disposable launcher controls; no native games or durable seeds are created."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_recruit_only_one_game as trained


class TerminalOutput(io.StringIO):
    def isatty(self):
        return True


class RecruitOnlyOneGameTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='heroes-trained-one-game-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        patcher = patch.object(trained.evaluation, 'BASE', self.root / 'large-seeds')
        patcher.start(); self.addCleanup(patcher.stop)

    def checked(self):
        return dict(checkpoint=1753, model_checkpoint='synthetic-trained-1753',
            weights_sha256='unchanged-trained-weights', model_config=dict(trained.MODEL_CONFIG),
            parameters=74344, source={'bound': 'synthetic-source'},
            parent={'checkpoint': {'metadata': {'original': True}}}, protected={'original': 'protected'},
            army_information=trained.OWN_ARMY_INFORMATION, experiment=trained.experiment(),
            training_launcher_accepted=False, binary='synthetic-native', runtime={'bound': 'native'},
            runtime_origin={'acceptance': 'bound'}, execution={'original': True})

    def plan(self):
        output = self.root / 'unstarted'
        receipt = trained.evaluation.reserve(output)
        return dict(schema=trained.SCHEMA + '-plan', **self.checked(), output=str(output),
            budget=trained.BUDGET, physical_games=1, purpose='diagnostic', training_input=False,
            optimizer_updates=0, policy_support=trained.GUARD, reservation=receipt,
            schedule=trained.read(trained.bound(receipt))['schedule'])

    def test_default_and_explicit_check_do_not_launch_or_reserve(self):
        with patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
             patch.object(trained.evaluation, 'reserve') as reserve, \
             patch.object(trained, 'prepare') as prepare, \
             patch.object(trained.evaluation, 'supervise') as supervise, \
             patch.object(trained.campaign, 'activity_lease') as lease, \
             patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(trained.main([]), 0)
            self.assertEqual(trained.main(['--check']), 0)
        for method in (reserve, prepare, supervise, lease): method.assert_not_called()
        self.assertFalse(trained.evaluation.BASE.exists())
        result = json.JSONDecoder().raw_decode(output.getvalue())[0]
        self.assertEqual(result['checkpoint'], 1753)
        self.assertEqual(result['parameters'], 74344)
        self.assertEqual(result['army_information'], 'own-army-details-v1')
        self.assertEqual(result['policy_support'], 'visible-interaction-budget-recruit-only-v1')
        self.assertTrue(result['exact_model_adam_rng_progress_sampler_pool'])
        self.assertFalse(result['gameplay_performed'])
        self.assertFalse(result['seeds_reserved'])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertIn('Total running time:', output.getvalue())

    def test_terminal_check_is_readable_and_still_cannot_start_a_game(self):
        for disable_color in (False, True):
            environment = {'TERM': 'xterm-256color'}
            if disable_color:
                environment['NO_COLOR'] = '1'
            with self.subTest(no_color=disable_color), \
                 patch.dict(os.environ, environment, clear=True), \
                 patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
                 patch.object(trained.evaluation, 'reserve') as reserve, \
                 patch.object(trained.evaluation, 'supervise') as supervise, \
                 patch.object(trained.campaign, 'activity_lease') as lease, \
                 patch('sys.stdout', new=TerminalOutput()) as output:
                self.assertEqual(trained.main(['--check']), 0)
            for method in (reserve, supervise, lease):
                method.assert_not_called()
            text = output.getvalue()
            self.assertIn('READY - checks passed; no game started.', text)
            self.assertIn('Learning OFF', text)
            self.assertIn('1,753', text)
            self.assertIn('74,344', text)
            self.assertIn('./play-one-selfplay.sh --run', text)
            self.assertNotIn('source_sha256', text)
            self.assertNotIn('visible-interaction-budget', text)
            self.assertEqual('\x1b[' in text, not disable_color)
            self.assertFalse(trained.evaluation.BASE.exists())

    def test_explicit_json_is_parseable_even_in_a_terminal(self):
        with patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
             patch('sys.stdout', new=TerminalOutput()) as output, \
             patch('sys.stderr', new=io.StringIO()) as diagnostics:
            self.assertEqual(trained.main(['--check', '--json']), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['checkpoint'], 1753)
        self.assertFalse(result['gameplay_performed'])
        self.assertNotIn('\x1b[', output.getvalue())
        self.assertIn('Total running time:', diagnostics.getvalue())

    def test_terminal_failure_is_clear_nonzero_and_never_retries(self):
        with patch.object(trained, 'inspect_candidate', side_effect=ValueError('Checkpoint differs')) as inspect, \
             patch.object(trained.evaluation, 'reserve') as reserve, \
             patch.object(trained.evaluation, 'supervise') as supervise, \
             patch('sys.stdout', new=TerminalOutput()) as output:
            self.assertEqual(trained.main(['--check']), 2)
        inspect.assert_called_once_with()
        reserve.assert_not_called(); supervise.assert_not_called()
        self.assertIn('RUN STOPPED - no automatic retry.', output.getvalue())
        self.assertIn('Checkpoint differs', output.getvalue())
        with patch.object(trained, 'inspect_candidate', side_effect=ValueError('Checkpoint differs')), \
             patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(ValueError, 'Checkpoint differs'):
            trained.main(['--check'])

    def test_rendered_colors_preserve_text_and_small_terminal_width(self):
        strip = lambda text: re.sub(r'\x1b\[[0-9;]*m', '', text)
        for width in (20, 48, 80):
            with self.subTest(columns=width):
                plain = trained.render_setup(ready=True, columns=width)
                colored = trained.render_setup(ready=True, columns=width, color=True)
                self.assertEqual(strip(colored), plain)
                self.assertTrue(all(len(line) <= width for line in plain.splitlines()))

    def test_result_separates_verification_from_win_and_unscored_cutoff(self):
        result = dict(status='passed', native_replay_verified=True,
            both_policies_reconstructed=True, owned_cleanup_verified=True,
            protected_inputs_unchanged=True, choices_by_player=[9920, 600],
            episode_status='truncated', termination_reason='collection_window',
            outcomes={'red': None, 'blue': None})
        for reason in ('collection_window', 'game_day_limit'):
            text = trained.render_result({**result, 'termination_reason': reason})
            self.assertIn('VERIFIED - recording checks passed.', text)
            self.assertIn('result is unscored', text)
            self.assertNotIn('wins.', text)
        won = trained.render_result({**result, 'episode_status': 'completed',
                                     'outcomes': {'red': 'loss', 'blue': 'win'}})
        self.assertIn('Blue wins.', won)
        incomplete = trained.render_result({**result, 'native_replay_verified': False})
        self.assertNotIn('VERIFIED - recording checks passed.', incomplete)
        self.assertIn('RUN STOPPED', incomplete)

    def test_explicit_run_and_worker_dispatch_the_large_policy_hooks(self):
        plan = {'synthetic': True}
        with patch.object(trained.campaign, 'activity_lease', return_value=nullcontext()), \
             patch.object(trained, 'prepare', return_value=plan), \
             patch.object(trained.evaluation, 'supervise', return_value={'status': 'passed'}) as supervise, \
             patch('builtins.print'):
            self.assertEqual(trained.main(['--run']), 0)
        supervise.assert_called_once_with(plan, worker_script=Path(trained.__file__).resolve(),
            result_schema=trained.SCHEMA, checkpoint=1753)
        with patch.object(trained.evaluation, 'worker', return_value={'status': 'passed'}) as worker:
            self.assertEqual(trained.main(['_worker', '--plan', '/synthetic/plan', '--sha256', 'digest']), 0)
        worker.assert_called_once_with(Path('/synthetic/plan'), 'digest', identity_check=trained.check_identity,
            model_loader=trained.frozen_models, result_schema=trained.SCHEMA, checkpoint=1753)
        with patch.object(trained.evaluation, 'worker') as worker, self.assertRaisesRegex(ValueError, 'bound plan'):
            trained.main(['_worker', '--run'])
        worker.assert_not_called()

    def test_invalid_budget_or_busy_owner_cannot_prepare(self):
        with patch.object(trained, 'prepare') as prepare, patch.object(trained, 'inspect_candidate') as inspect, \
             patch.object(trained.evaluation, 'supervise') as supervise, patch('builtins.print'):
            for value in ('nan', 'inf', '179', '7201'):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite total budget'):
                    trained.main(['--run', '--seconds', value])
            with patch.object(trained.campaign, 'activity_lease', side_effect=ValueError('busy')), \
                 self.assertRaisesRegex(ValueError, 'busy'):
                trained.main(['--run'])
        for method in (prepare, inspect, supervise): method.assert_not_called()

    def test_prepare_continues_the_existing_large_seed_namespace(self):
        previous = trained.evaluation.reserve(self.root / 'previous-unstarted')
        output = self.root / 'new-run'
        args = SimpleNamespace(seconds=180, output=output)
        with patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
             patch.object(trained.evaluation.copies, 'private_path', return_value=output) as private_path:
            plan = trained.prepare(args)
        private_path.assert_called_once_with(output, fresh=True)
        self.assertEqual(plan['schedule'], trained.evaluation.seed_row(1))
        self.assertEqual(plan['schedule']['engine_seed'], 1960000001)
        self.assertEqual(plan['checkpoint'], 1753)
        self.assertEqual(plan['policy_support'], trained.GUARD)
        self.assertEqual(plan['army_information'], trained.OWN_ARMY_INFORMATION)
        self.assertEqual(plan['experiment'], trained.experiment())
        self.assertEqual(plan['optimizer_updates'], 0)
        self.assertEqual(trained.read(output / 'plan.json'), plan)
        ledger = trained.evaluation.seed_ledger()
        self.assertEqual(len(ledger['reservations']), 2)
        self.assertEqual(ledger['reservations'][0], trained.read(trained.bound(previous)))
        self.assertTrue(all(row['consumed_including_unstarted'] for row in ledger['reservations']))

    def test_identity_rejects_contract_or_preserved_candidate_mutations(self):
        original = self.plan()
        with patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
             patch.object(trained.campaign, 'verify_protected', return_value=True):
            trained.check_identity(original)
            for key, value in (
                ('schema', 'historical-plan'), ('budget', {}), ('physical_games', 2),
                ('purpose', 'training'), ('training_input', True), ('optimizer_updates', 1),
                ('policy_support', trained.campaign.GUARD), ('army_information', None),
                ('checkpoint', 'tiny-v1-untrained-initial'), ('model_checkpoint', 'another-policy'),
                ('weights_sha256', 'new-weights'), ('parameters', 15460), ('model_config', {}),
                ('source', {}), ('runtime', {}), ('runtime_origin', {}), ('binary', 'old-runtime'),
                ('parent', {}), ('protected', {}), ('execution', {}), ('experiment', {}),
                ('training_launcher_accepted', True),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    trained.check_identity({**original, key: value})

    def test_identity_rejects_seed_reuse_or_different_output(self):
        original = self.plan()
        with patch.object(trained, 'inspect_candidate', return_value=self.checked()), \
             patch.object(trained.campaign, 'verify_protected', return_value=True):
            for update in ({'schedule': trained.evaluation.seed_row(1)}, {'output': '/different/output'}):
                with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'seed binding'):
                    trained.check_identity({**original, **update})
            with patch.object(trained.evaluation, 'seed_ledger', return_value={'reservations': []}), \
                 self.assertRaisesRegex(ValueError, 'seed binding'):
                trained.check_identity(original)

    def test_runtime_overlay_preserves_trained_parent_without_loading_tiny(self):
        baseline = self.checked()
        baseline.pop('army_information'); baseline.pop('runtime_origin')
        baseline['protected'] = {'original': 'protected'}
        before = deepcopy(baseline)
        runtime = dict(binary='accepted-new-native', runtime={'accepted': True},
            army_information=trained.OWN_ARMY_INFORMATION, runtime_origin={'accepted': 'receipt'},
            runtime_protected={'native': 'protected'})
        with patch.object(trained.evaluation, 'inspect_parent', return_value=baseline) as inspect_parent, \
             patch.object(trained, 'source_identity', return_value={'new': 'source'}), \
             patch.object(trained.runtime_validation, 'inspect_runtime', return_value=runtime) as inspect_runtime, \
             patch.object(trained.runtime_validation, 'inspect_candidate') as tiny_candidate, \
             patch.object(trained.runtime_validation, 'frozen_models') as tiny_models, \
             patch.object(trained, 'frozen_models') as models, \
             patch.object(trained.campaign, 'verify_protected', return_value=True):
            result = trained.inspect_candidate()
        inspect_parent.assert_called_once_with(trained.campaign.LINEAGE)
        inspect_runtime.assert_called_once_with(baseline)
        tiny_candidate.assert_not_called(); tiny_models.assert_not_called()
        self.assertEqual(baseline, before)
        self.assertEqual(result['parent'], before['parent'])
        self.assertEqual(result['weights_sha256'], before['weights_sha256'])
        self.assertEqual(result['protected'], {'original': 'protected', 'native': 'protected'})
        self.assertEqual(result['binary'], 'accepted-new-native')
        models.assert_called_once_with(result)

    def test_model_loader_rejects_tiny_shape_and_preserves_shared_loader_identity(self):
        def model(config, parameters):
            return SimpleNamespace(config=SimpleNamespace(to_dict=lambda: config),
                parameters=lambda: [SimpleNamespace(numel=lambda: parameters)])
        plan = self.checked()
        large = model(trained.MODEL_CONFIG, trained.PARAMETERS)
        with patch.object(trained.evaluation, 'frozen_models', return_value=(large, large)) as loader:
            self.assertEqual(trained.frozen_models(plan), (large, large))
            loader.assert_called_once_with(plan)
        for config, count in (({**trained.MODEL_CONFIG, 'hidden_size': 8}, 74344),
                              (trained.MODEL_CONFIG, 15460)):
            invalid = model(config, count)
            with patch.object(trained.evaluation, 'frozen_models', return_value=(invalid, invalid)), \
                 self.assertRaisesRegex(ValueError, 'large model architecture'):
                trained.frozen_models(plan)


if __name__ == '__main__':
    unittest.main()
