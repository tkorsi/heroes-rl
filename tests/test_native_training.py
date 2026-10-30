"""Native numerical parity on disposable typed fixtures; never gameplay admission."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.learner import contextual_model as m, ppo
from experiments.learner.native_recipes import compile_history, require_native_model
from tests.test_contextual_ppo import synthetic_rollout


class NativeModelGuardTests(unittest.TestCase):
    def test_compiled_autograd_context_is_rejected_before_native_execution(self):
        model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        for name in ('compiled_autograd_enabled', 'compiled_autograd_enabled_force_eager',
                     'in_compiled_autograd_region'):
            with patch.dict(sys.modules, {'torch._dynamo.compiled_autograd': SimpleNamespace(**{name: True})}):
                with self.assertRaisesRegex(ValueError, 'ordinary eager autograd'):
                    require_native_model(model)

    def test_noncanonical_execution_is_rejected(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            mutations = [lambda x: setattr(x, 'initial_state', lambda: torch.ones(x.config.hidden_size)),
                         lambda x: setattr(x.byte, 'padding_idx', 66),
                         lambda x: setattr(x.map_cnn[0], 'stride', (2, 2)),
                         lambda x: setattr(x.gru, 'bias', False),
                         lambda x: setattr(x.text_gru, 'bidirectional', True),
                         lambda x: setattr(x, '_long', lambda v: torch.tensor(v)),
                         lambda x: setattr(x, '_can_cache_prefix', lambda: True),
                         lambda x: setattr(x, 'named_modules', lambda: iter(())),
                         lambda x: setattr(x, 'number', torch.nn.Sequential(x.number[0])),
                         lambda x: setattr(x.number, '_modules', dict(reversed(list(x.number._modules.items())))),
                         lambda x: setattr(x, 'map_cnn', torch.nn.Sequential(*list(x.map_cnn.children())[:3])),
                         lambda x: x.quantity.append(torch.nn.Tanh()),
                         lambda x: setattr(x.text_gru, '_flat_weights',
                                           [x.text_gru._flat_weights[0].clone(), *x.text_gru._flat_weights[1:]]),
                         lambda x: setattr(x.ordered, '_flat_weights_names', list(reversed(x.ordered._flat_weights_names)))]
            for mutate in mutations:
                model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
                require_native_model(model)
                mutate(model)
                with self.assertRaises(ValueError): require_native_model(model)
        finally:
            torch.set_num_threads(previous)


@unittest.skipUnless(os.environ.get('HEROES_NATIVE_TEST_MANIFEST'), 'An explicitly built native learner is required')
class NativeTrainingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.learner.native_backend import load_backend
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.manifest = Path(os.environ['HEROES_NATIVE_TEST_MANIFEST']).resolve()
        cls.sha = hashlib.sha256(cls.manifest.read_bytes()).hexdigest()
        cls.backend = load_backend(cls.manifest, cls.sha)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def exact(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.exact(a, b)
        else:
            self.assertEqual(left, right)

    def test_two_updates_preserve_gradients_adam_weights_and_all_rng(self):
        torch.manual_seed(38120)
        reference = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        native = deepcopy(reference)
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.0003) for model in (reference, native)]
        generators = [torch.Generator().manual_seed(9231) for _ in range(2)]
        config = ppo.PPOConfig(epochs=2, decisions_per_window=2, windows_per_minibatch=2)
        for update in range(2):
            rollouts = (synthetic_rollout(reference, turns=5, episode=f'first-{update}'),
                        synthetic_rollout(reference, turns=3, episode=f'cutoff-{update}', boundary='cutoff'))
            before_rng = torch.get_rng_state().clone()
            with patch.object(ppo, 'NATIVE_EXECUTION', None), patch.object(ppo, 'TEXT_REPLAY_WORKERS', 2):
                expected = ppo.ppo_update(reference, optimizers[0], rollouts, config=config, generator=generators[0])
            with patch.object(ppo, 'NATIVE_EXECUTION', {'manifest': str(self.manifest), 'sha256': self.sha, 'prefix_threads': 16}):
                actual = ppo.ppo_update(native, optimizers[1], rollouts, config=config, generator=generators[1])
            self.assertEqual(expected, actual)
            self.assertIsNone(actual.timings_seconds)
            self.exact_bytes(reference.state_dict(), native.state_dict())
            self.exact_bytes([p.grad for p in reference.parameters()], [p.grad for p in native.parameters()])
            self.exact_bytes(optimizers[0].state_dict(), optimizers[1].state_dict())
            self.exact_bytes(generators[0].get_state(), generators[1].get_state())
            self.exact_bytes(before_rng, torch.get_rng_state())

    def exact_bytes(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                                        right.detach().contiguous().reshape(-1).view(torch.uint8)))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left: self.exact_bytes(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.exact_bytes(a, b)
        else:
            self.assertEqual(left, right)

    def frame_run(self, original, history, windows, workers):
        model = deepcopy(original)
        machine = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
        versions = [parameter._version for parameter in model.parameters()]
        machine.begin_text_replay(3)
        try:
            for start, end, decisions, initial in windows:
                machine.prepare_window_text(history, start, end, decisions, workers)
            outputs = [machine.evaluate_window_details(history, start, end, decisions, initial, workers)
                       for start, end, decisions, initial in windows]
            loss = sum(logs.sum() + values.square().sum() + state.square().sum() * 0.01
                       for logs, values, state, _ in outputs)
            machine.backward_with_text_replay(loss)
            machine.end_text_replay()
            self.assertEqual([parameter._version for parameter in model.parameters()], versions)
            self.exact_bytes(model.state_dict(), original.state_dict())
            return ([(logs.detach(), values.detach(), state.detach(), [part.detach() for part in details])
                     for logs, values, state, details in outputs],
                    [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()])
        finally:
            machine.close()

    def test_parallel_frame_features_preserve_gradient_bytes_across_windows(self):
        torch.manual_seed(41781)
        model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(model, turns=5, episode='parallel-frame-features')
        inputs = m.encode_history(rollout.history)
        history = compile_history(model, inputs, rollout.decisions, self.backend)
        decisions = [record.decision_seq for record in rollout.decisions]
        start = decisions[2] + 1
        initial = model.prefix_states(inputs, (start,))[start]
        # The first window skips a decision whose unused value head must still
        # be constructed. Both windows share one replay scope and parameter set.
        windows = [(0, decisions[2], [decisions[0], decisions[2]], model.initial_state()),
                   (start, decisions[4], decisions[3:], initial)]
        rng = torch.get_rng_state().clone()
        expected_outputs, expected_gradients = self.frame_run(model, history, windows, 1)
        for workers in (4, 16):
            with self.subTest(workers=workers):
                outputs, gradients = self.frame_run(model, history, windows, workers)
                self.exact_bytes(outputs, expected_outputs)
                self.exact_bytes(gradients, expected_gradients)
                self.exact_bytes(torch.get_rng_state(), rng)

    def test_parallel_frame_features_preserve_coordinate_quantity_and_management_gradients(self):
        from experiments.learner import contextual as c
        from tests.test_contextual_model import map_history, town_history, recruitment
        from tests.test_contextual_management import at_stack
        from tests.test_contextual_management_model import all_actions
        town = town_history(recruitment())
        stack = at_stack()
        simple_map = map_history()
        visible = simple_map.decision.observation
        multi_map = c.PlayerHistory('parallel-coordinate', 0, schema=simple_map.frames[0].schema)
        multi_map.emit('view', observation=visible)
        multi_map.emit('decision', observation=visible, offers=simple_map.decision.offers, revision=0)
        fixtures = [(multi_map, c.Action('move_to', 'hero', 2, 1)),
                    (town, c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA,
                         selection=town.decision.offers.panel.recruitment.selected_offer_ref,
                         destination=town.decision.offers.panel.recruitment.destination.ref, count=2)),
                    (stack, all_actions(stack.decision)[0])]
        for source, action in fixtures:
            with self.subTest(action=action.kind):
                torch.manual_seed(41782)
                model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(source.frames[0].schema),
                                                              hidden_size=8, feature_size=4))
                inputs = m.encode_history(source.frames)
                state = model.initial_state()
                with torch.no_grad():
                    for frame in inputs:
                        output = model.forward_step(frame, state)
                        state = output.state
                evaluated = output.decision.distribution.evaluate(action)
                record = SimpleNamespace(decision_seq=len(inputs)-1, action=action,
                    support=output.decision.distribution.support, conditionals=evaluated.conditionals)
                history = compile_history(model, inputs, (record,), self.backend)
                windows = [(0, len(inputs)-1, [len(inputs)-1], model.initial_state())]
                rng = torch.get_rng_state().clone()
                expected_outputs, expected_gradients = self.frame_run(model, history, windows, 1)
                for workers in (4, 16):
                    outputs, gradients = self.frame_run(model, history, windows, workers)
                    self.exact_bytes(outputs, expected_outputs)
                    self.exact_bytes(gradients, expected_gradients)
                    self.exact_bytes(torch.get_rng_state(), rng)

    def test_parallel_frame_missing_preparation_and_worker_failure_leave_retry_safe(self):
        torch.manual_seed(41783)
        model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(model, turns=3, episode='parallel-frame-failure')
        inputs = m.encode_history(rollout.history)
        history = compile_history(model, inputs, rollout.decisions, self.backend)
        decisions = [record.decision_seq for record in rollout.decisions]
        machine = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
        machine.begin_text_replay(3)
        try:
            with self.assertRaisesRegex(RuntimeError, 'prepar|prewarm'):
                machine.evaluate_window(history, 0, decisions[-1], decisions, model.initial_state(), 4)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            machine.prepare_window_text(history, 0, decisions[-1], decisions, 4)
            parameter = model.tree[0].weight
            original = parameter.detach().clone()
            with torch.no_grad(): parameter.set_(torch.zeros(2, 2))
            try:
                with self.assertRaises(RuntimeError):
                    machine.evaluate_window(history, 0, decisions[-1], decisions, model.initial_state(), 16)
            finally:
                with torch.no_grad(): parameter.set_(original)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            machine.close()
            machine.begin_text_replay(3)
            machine.prepare_window_text(history, 0, decisions[-1], decisions, 4)
            actual = machine.evaluate_window(history, 0, decisions[-1], decisions, model.initial_state(), 4)
            machine.backward_with_text_replay(actual[0].sum() + actual[1].sum())
            machine.end_text_replay()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        finally:
            machine.close()

    def test_prepared_window_text_preserves_outputs_gradients_and_rng(self):
        torch.manual_seed(41731)
        original = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(original, turns=4, episode='native-text-preparation')
        inputs = m.encode_history(rollout.history)
        history = compile_history(original, inputs, rollout.decisions, self.backend)
        decisions = [record.decision_seq for record in rollout.decisions]
        before_rng = torch.get_rng_state().clone()

        def evaluate(workers):
            model = deepcopy(original)
            machine = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
            machine.begin_text_replay(3)
            try:
                if workers is not None:
                    machine.prepare_window_text(history, 0, decisions[-1], decisions, workers)
                    machine.prepare_window_text(history, 0, decisions[-1], decisions, workers)
                logs, values, state, details = machine.evaluate_window_details(
                    history, 0, decisions[-1], decisions, model.initial_state())
                loss = logs.sum() + values.square().sum() + state.square().sum() * 0.01
                machine.backward_with_text_replay(loss)
                machine.end_text_replay()
                return ((logs.detach(), values.detach(), state.detach(), [item.detach() for item in details]),
                        [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()])
            finally:
                machine.close()

        expected = evaluate(None)
        for workers in (1, 4, 16):
            with self.subTest(workers=workers):
                self.exact(evaluate(workers), expected)
                self.exact(torch.get_rng_state(), before_rng)

    def test_window_text_preparation_rejects_invalid_requests_before_warming(self):
        torch.manual_seed(94113)
        model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        rollout = synthetic_rollout(model, turns=2, episode='native-text-preparation-invalid')
        inputs = m.encode_history(rollout.history)
        history = compile_history(model, inputs, rollout.decisions, self.backend)
        decisions = [record.decision_seq for record in rollout.decisions]
        end = decisions[-1]
        machine = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
        with self.assertRaisesRegex(RuntimeError, 'open replay scope'):
            machine.prepare_window_text(history, 0, end, decisions, 4)
        machine.begin_text_replay(0)
        try:
            requests = [(-1, end, decisions, 4), (0, len(inputs), decisions, 4),
                        (0, end, [], 4), (0, end, [decisions[0], decisions[0]], 4),
                        (0, end, decisions, 0), (0, end, decisions, 257)]
            for request in requests:
                with self.subTest(request=request), self.assertRaises(RuntimeError):
                    machine.prepare_window_text(history, *request)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        finally:
            machine.close()

    def check_history(self, history, actions):
        torch.manual_seed(5191)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(history.frames[0].schema),
                                                      hidden_size=8, feature_size=4))
        inputs = m.encode_history(history.frames)
        state = model.initial_state()
        with torch.no_grad():
            for frame in inputs:
                output = model.forward_step(frame, state)
                state = output.state
        distribution = output.decision.distribution
        for action in actions:
            expected = distribution.evaluate(action)
            record = SimpleNamespace(decision_seq=len(inputs)-1, action=action, support=distribution.support,
                                     conditionals=expected.conditionals)
            compiled = compile_history(model, inputs, (record,), self.backend)
            machine = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
            with torch.no_grad():
                logs, vals, actual_state, conditionals = machine.evaluate_window_details(
                    compiled, 0, len(inputs)-1, [len(inputs)-1], model.initial_state())
            self.exact(logs[0], expected.log_prob)
            self.exact(vals[0], output.decision.value)
            self.exact(actual_state, state)
            self.exact(conditionals[0], torch.stack([v.log_prob for v in expected.conditionals]))
            machine.begin_text_replay(0)
            try:
                machine.prepare_window_text(compiled, 0, len(inputs)-1, [len(inputs)-1], 4)
                prepared_values = machine.evaluate_window_details(
                    compiled, 0, len(inputs)-1, [len(inputs)-1], model.initial_state())
                self.exact(prepared_values, (logs, vals, actual_state, conditionals))
            finally:
                machine.close()

    def test_coordinates_quantities_management_dialogs_and_capabilities(self):
        from experiments.learner import contextual as c
        from tests.test_contextual_model import map_history, town_history, recruitment
        from tests.test_contextual_management import history_at, at_stack, at_confirmation
        from tests.test_contextual_management_model import all_actions
        from tests.test_contextual_dialogs import FAMILIES, history_at as dialog_history, panel, action as dialog_action
        from tests.test_contextual_capabilities import market_history, action as capability_action
        history = map_history()
        self.check_history(history, [c.Action('move_to', 'hero', 2, 1), c.Action('end_turn')])
        history = town_history(recruitment())
        self.check_history(history, [c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA,
            selection=history.decision.offers.panel.recruitment.selected_offer_ref,
            destination=history.decision.offers.panel.recruitment.destination.ref, count=2)])
        for history in (history_at(), history_at('hero'), at_stack(), at_confirmation()):
            self.check_history(history, all_actions(history.decision))
        for family in FAMILIES:
            history = dialog_history(panel(family))
            self.check_history(history, [dialog_action(selection=v.ref) for v in history.decision.offers.dialog_panel.choices])
        history = market_history()
        self.check_history(history, [capability_action('trade_resource', 'town', selection='choice-1', count=count)
                                     for count in (1, 2, 4)])


if __name__ == '__main__':
    unittest.main()
