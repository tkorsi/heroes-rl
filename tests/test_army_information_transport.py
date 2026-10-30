"""Offline negotiation and exact transport checks for passive own-army data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.headless import contextual_protocol as wire, contextual_replay as replay
from experiments.headless.env import HeadlessSelfPlayEnv
from experiments.learner import contextual as c


class ArmyInformationTransportTests(unittest.TestCase):
    def ready(self, extension=None):
        result = dict(protocol_version=10, type='ready', episode='1', player=0,
                      interface=c.EXPANDED_ENVIRONMENT, history_schema=c.EXPANDED_SCHEMA,
                      action_schema=c.EXPANDED_ACTION_SCHEMA)
        if extension is not None: result['army_information'] = extension
        return result

    def test_negotiation_is_explicit_and_legacy_ready_is_unchanged(self):
        legacy = wire.ContextualProducer('1', 0, interface=c.EXPANDED_ENVIRONMENT)
        legacy.ingest(self.ready()); self.assertTrue(legacy.ready)
        extended = wire.ContextualProducer('1', 0, interface=c.EXPANDED_ENVIRONMENT,
                                          army_information=c.OWN_ARMY_INFORMATION)
        extended.ingest(self.ready(c.OWN_ARMY_INFORMATION)); self.assertTrue(extended.ready)
        for selected, received in ((None, c.OWN_ARMY_INFORMATION), (c.OWN_ARMY_INFORMATION, None),
                                   (c.OWN_ARMY_INFORMATION, 'unknown')):
            with self.subTest(selected=selected, received=received):
                producer = wire.ContextualProducer('1', 0, interface=c.EXPANDED_ENVIRONMENT,
                                                  army_information=selected)
                with self.assertRaises(ValueError): producer.ingest(self.ready(received))
                self.assertFalse(producer.ready)
        for interface, selected in ((c.ENVIRONMENT, c.OWN_ARMY_INFORMATION), (c.EXPANDED_ENVIRONMENT, 'unknown')):
            with self.assertRaises(ValueError): wire.ContextualProducer('1', 0, interface=interface, army_information=selected)

    def test_observation_extension_cannot_disappear_or_appear_after_ready(self):
        for selected in (None, c.OWN_ARMY_INFORMATION):
            for kind in ('decision', 'view'):
                with self.subTest(selected=selected, kind=kind):
                    producer = wire.ContextualProducer('1', 0, interface=c.EXPANDED_ENVIRONMENT,
                                                      army_information=selected)
                    producer.ingest(self.ready(selected))
                    observation = c.encode_observation(c.Observation((2, 2, 1), (), (), (0,) * 7, (), ()),
                                                       schema=c.EXPANDED_SCHEMA)
                    if selected is None:
                        observation.update(army_information=c.OWN_ARMY_INFORMATION, own_armies=[])
                    record = dict(kind=kind, day=1, observation=observation)
                    if kind == 'decision': record.update(decision_id=0, revision=0)
                    message = dict(protocol_version=10, type='contextual', episode='1', player=0, records=[record])
                    with self.assertRaisesRegex(ValueError, 'negotiated extension'): producer.ingest(message)
                    self.assertEqual(producer.history.frames, ())

    def test_passive_armies_are_delivered_without_a_stack_action(self):
        producer = wire.ContextualProducer('1', 0, interface=c.EXPANDED_ENVIRONMENT,
                                          army_information=c.OWN_ARMY_INFORMATION)
        producer.ingest(self.ready(c.OWN_ARMY_INFORMATION))
        stack = c.OwnArmyStack(0, 0, 12, 'creatures.core.pikeman.name.plural',
                              4, 5, 1, 3, 10, 4, 0, 0, 0, 0, False, False, False)
        observation = c.Observation((2, 2, 1), (), (), (0,) * 7, (c.Hero('hero', (0, 0, 0), 100, in_garrison=False),), (),
                                    army_information=c.OWN_ARMY_INFORMATION,
                                    own_armies=(c.OwnArmy('hero', 'hero', (stack,)),))
        raw = c.encode_observation(observation, schema=c.EXPANDED_SCHEMA)
        frames = producer.ingest(dict(protocol_version=10, type='contextual', episode='1', player=0,
            records=[dict(kind='decision', day=1, decision_id=0, revision=0, observation=raw)]))
        self.assertEqual(frames[0].observation.own_armies[0].slots[0], stack)
        self.assertEqual([frame.kind for frame in frames], ['decision'])
        self.assertIsNone(frames[0].observation.window)

    def test_constructor_rejects_unknown_extension_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            for interface, selected in ((c.ENVIRONMENT, c.OWN_ARMY_INFORMATION), (c.EXPANDED_ENVIRONMENT, 'unknown')):
                with self.subTest(interface=interface, selected=selected), self.assertRaises(ValueError):
                    HeadlessSelfPlayEnv(object(), Path(directory), interface=interface, army_information=selected)

    def test_manifest_header_and_native_launch_must_agree_without_discarding_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(episode='1', environment=c.EXPANDED_ENVIRONMENT, status='truncated',
                            army_information=c.OWN_ARMY_INFORMATION,
                            launch={'environment': {'EXPLORER_RL_ARMY_INFORMATION': c.OWN_ARMY_INFORMATION}})
            tape = [dict(kind='init', schema=replay.SCHEMA, episode='1', interface=c.EXPANDED_ENVIRONMENT,
                         protocol_version=10, history_schema=c.EXPANDED_SCHEMA,
                         action_schema=c.EXPANDED_ACTION_SCHEMA, army_information=c.OWN_ARMY_INFORMATION)]
            for player, color in enumerate(('red', 'blue')):
                folder = root / 'agents' / color; folder.mkdir(parents=True)
                message = dict(player=player, records=[dict(kind='view', observation={
                    'army_information': c.OWN_ARMY_INFORMATION, 'own_armies': [{'count': 12}]})])
                (folder / 'bridge.jsonl').write_text(json.dumps(dict(direction='engine', message=message)) + '\n')
                tape.append(dict(seq=player, kind='message', player=player, message=message))
            self.assertEqual(replay.cross_check_wire(root, manifest, tape)['messages'], 2)
            for mutate in (lambda m, t: t[0].pop('army_information'),
                           lambda m, t: m['launch']['environment'].clear(),
                           lambda m, t: m.pop('army_information'),
                           lambda m, t: t[1]['message']['records'][0]['observation']['own_armies'][0].update(count=13)):
                changed_manifest, changed_tape = deepcopy(manifest), deepcopy(tape)
                mutate(changed_manifest, changed_tape)
                with self.assertRaises(ValueError): replay.cross_check_wire(root, changed_manifest, changed_tape)


if __name__ == '__main__':
    unittest.main()
