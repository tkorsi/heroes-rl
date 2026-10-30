"""Synthetic v8 technical closes; no native-game or PPO success claim."""
from pathlib import Path
import tempfile
import unittest

from experiments.headless.contextual_protocol import ContextualProducer
from experiments.learner import contextual as c
from tests.test_contextual_capabilities import action, observation, market_history
from tests.test_contextual_replay_history import Fixture


class CapabilityFixture(Fixture):
    def __init__(self, directory, player, old_declaration=False):
        if old_declaration:
            self.allowed_actions = ["back", "close", "end_turn"]
        self.directory, self.profile = directory, c.EXPANDED_PROFILE
        self.producers = [ContextualProducer('1', p, interface=self.profile.environment) for p in (0, 1)]
        self.rows, self.natural, self.outcomes = [[], []], False, ['win', 'loss']
        self.identity = {'engine_revision': 'synthetic-only', 'map_sha256': 'a' * 64}
        for p in (0, 1):
            (directory / 'agents' / ('red', 'blue')[p]).mkdir(parents=True)
            self.engine(p, dict(protocol_version=self.profile.protocol, type='ready', episode='1', player=p,
                interface=self.profile.environment, history_schema=self.profile.history_schema,
                action_schema=self.profile.action_schema))
            self.batch(p, self.held(0, 0, observation()))
            self.navigate(p, 0, action('open_town', 'town'), observation('town'), 1, 'opened')
            self.navigate(p, 1, action('open_capability', 'town', panel='marketplace'),
                          market_history().decision.observation, 3, 'opened')
            self.producers[p].freeze_cutoff('collection_window')
            if p == player:
                self.navigate(p, 2, action('close_capability', 'town'), observation('town'), 5, 'closed', True)
                self.navigate(p, 3, action('close'), observation(), 7, 'closed', True)
                self.submit(p, action('end_turn'), outside=True)
                self.batch(p, dict(kind='consumed', day=1, decision_id=4, consumer='engine'),
                    dict(kind='issued', day=1, decision_id=4, command_index=0),
                    dict(kind='acknowledged', day=1, decision_id=4, command_index=0, accepted=True),
                    dict(kind='result', day=1, decision_id=4, result=dict(status='turn_ended', accepted=True,
                        position=None, movement=None, destination_reached=None)))
        self.write()

    def held(self, native, revision, obs):
        return dict(kind='decision', day=1, decision_id=native, revision=revision,
                    observation=c.encode_observation(obs, schema=self.profile.history_schema))

    def navigate(self, player, native, selected, obs, revision, status, outside=False):
        self.submit(player, selected, outside=outside)
        self.batch(player, dict(kind='consumed', day=1, decision_id=native, consumer='adapter'),
            dict(kind='context', day=1, decision_id=native, context=c._wire(obs.context), revision=revision),
            dict(kind='view', day=1, observation=c.encode_observation(obs, schema=self.profile.history_schema)),
            dict(kind='result', day=1, decision_id=native, result=dict(status=status, accepted=True,
                position=None, movement=None, destination_reached=None)), self.held(native + 1, revision + 1, obs))


class CapabilityCutoffReplayTests(unittest.TestCase):
    def test_both_colors_neutral_capability_close_is_outside_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            for player in (0, 1):
                f = CapabilityFixture(Path(temp) / str(player), player)
                result = f.verify()
                self.assertTrue(result['verified'])
                self.assertFalse(result['native_replay_verified'])
                self.assertEqual(f.producers[player].history.context.kind, 'capability')
                self.assertEqual([x.action.kind for x in f.producers[player].history.frames if x.kind == 'submitted'],
                                 ['open_town', 'open_capability'])
                self.assertEqual([x.action.kind for x in f.producers[player].live.frames if x.kind == 'submitted'][-3:],
                                 ['close_capability', 'close', 'end_turn'])

    def test_no_economic_choice_or_wrong_subject_can_be_technical_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, selected in [('trade', action('trade_resource', 'town', selection='choice-1', count=1)),
                                   ('wrong-subject', action('close_capability', 'hero'))]:
                f = CapabilityFixture(Path(temp) / name, 0)
                def mutate(rows):
                    row = next(r['message'] for r in rows if r['direction'] == 'policy' and r['message']['outside_learner_prefix'])
                    row['action'] = c.encode_action(selected)
                f.change_bridge(mutate)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    f.verify()

    def test_old_declaration_is_accepted_only_without_capability_close(self):
        with tempfile.TemporaryDirectory() as temp:
            for player in (None, 0):
                f = CapabilityFixture(Path(temp) / str(player), player, old_declaration=True)
                if player is not None:
                    with self.assertRaisesRegex(ValueError, 'technical continuation'):
                        f.verify()
                else:
                    self.assertTrue(f.verify()['verified'])
