"""Native numerical parity on disposable typed fixtures; never gameplay admission."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.learner import contextual_model as m, ppo
from experiments.learner.native_recipes import compile_history, require_native_model
from tests.test_contextual_ppo import synthetic_rollout


class NativeModelGuardTests(unittest.TestCase):
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
            self.exact(reference.state_dict(), native.state_dict())
            self.exact([p.grad for p in reference.parameters()], [p.grad for p in native.parameters()])
            self.exact(optimizers[0].state_dict(), optimizers[1].state_dict())
            self.exact(generators[0].get_state(), generators[1].get_state())
            self.exact(before_rng, torch.get_rng_state())

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
