"""Player-owned passive army facts, strict wire checks and learned features.

These synthetic tests do not claim native engine capture or launch gameplay.
"""
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import unittest

import torch

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from tests.test_contextual_capabilities import observation
from tests.test_contextual_model import map_history


VERSION = "own-army-details-v1"


def stack(slot=0, creature=0, count=14, **changes):
    fields = dict(slot=slot, creature=creature, count=count,
        name_text_id="creatures.core.pikeman.name.plural", attack=5, defense=4,
        damage_min=1, damage_max=3, health=10, speed=4, shots=0, casts=0,
        morale=1, luck=-1, shooter=False, flying=False, undead=False)
    fields.update(changes)
    return c.OwnArmyStack(**fields)


def informed_observation(kind="map"):
    base = observation(kind)
    rows = (c.OwnArmy("hero", "hero", (stack(),)),
            c.OwnArmy("other", "hero", (stack(count=7),)),
            c.OwnArmy("town", "town_garrison", ()))
    if base.window is not None:
        displayed = {row.subject: row for row in base.window.armies or ()}
        rows = tuple(replace(row, slots=tuple(stack(slot.slot, slot.creature, slot.count)
                     for slot in displayed[row.subject].slots if slot.count))
                     if row.subject in displayed else row for row in rows)
    return replace(base, army_information=VERSION, own_armies=rows)


class PassiveArmyInformationTests(unittest.TestCase):
    def test_own_composition_and_current_stats_arrive_without_any_open_window(self):
        obs = informed_observation()
        self.assertIsNone(obs.window)
        self.assertEqual(obs.context.kind, "map")
        self.assertEqual({row.subject for row in obs.own_armies}, {"hero", "other", "town"})
        facts = obs.own_armies[0].slots[0]
        self.assertEqual((facts.creature, facts.count, facts.attack, facts.defense,
                          facts.damage_min, facts.damage_max, facts.health, facts.speed),
                         (0, 14, 5, 4, 1, 3, 10, 4))
        # Passive data add no sampled action and do not alter native legality.
        self.assertEqual(c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA),
                         c.Offers.from_observation(observation(), schema=c.EXPANDED_ACTION_SCHEMA))

    def test_additive_wire_roundtrip_keeps_complete_empty_town_row(self):
        for kind in ("map", "hero", "town"):
            with self.subTest(kind=kind):
                obs = informed_observation(kind)
                raw = c.encode_observation(obs, schema=c.EXPANDED_SCHEMA)
                self.assertEqual(raw["army_information"], VERSION)
                decoded = c.decode_observation(json.loads(json.dumps(raw)), schema=c.EXPANDED_SCHEMA)
                self.assertEqual(decoded, obs)
                if kind != "town":
                    self.assertEqual(decoded.own_armies[-1].slots, ())

    def test_passive_composition_must_agree_with_opened_army_rows(self):
        for kind in ("hero", "town"):
            obs = informed_observation(kind)
            row = next(row for row in obs.own_armies if row.subject == obs.window.armies[0].subject)
            changed = replace(row, slots=(replace(row.slots[0], count=row.slots[0].count + 1), *row.slots[1:]))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "disagrees with its displayed army"):
                replace(obs, own_armies=tuple(changed if item.subject == row.subject else item for item in obs.own_armies))

    def test_owned_town_stack_popup_cannot_contradict_passive_composition(self):
        obs = informed_observation("town")
        source = obs.window.armies[0].slots[0]
        panel = c.StackPanel(obs.context, source, (), None, None, None)
        context = c.Context("stack", "town", "overview", source.ref)
        window = c.Window(context, None, armies=(), stack_panel=panel, capability_controls=())
        opened = replace(obs, window=window)
        changed = replace(panel, source=replace(source, count=source.count + 1))
        with self.assertRaisesRegex(ValueError, "disagrees with its displayed stack"):
            replace(opened, window=replace(window, stack_panel=changed))

    def test_enemy_unknown_owner_missing_owned_row_and_duplicates_reject(self):
        obs = informed_observation()
        enemy = c.ObjectView("enemy", (1, 1, 0), 34, 0, "opponent")
        visible_enemy = replace(obs, objects=(enemy,))
        for rows in (obs.own_armies[:-1], obs.own_armies[1:],
                     (*obs.own_armies, obs.own_armies[0]),
                     (*obs.own_armies, c.OwnArmy("enemy", "hero", (stack(),))),
                     (*obs.own_armies, c.OwnArmy("unknown", "hero", (stack(),))),
                     (replace(obs.own_armies[0], role="town_garrison"), *obs.own_armies[1:])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replace(visible_enemy, own_armies=rows)
        for role in ("enemy", "ally", "joining_neutral", "allied_garrison"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                c.OwnArmy("hero", role, (stack(),))

    def test_marker_and_typed_payload_are_required_together(self):
        obs = informed_observation()
        for changes in (dict(army_information=None), dict(army_information="unknown"),
                        dict(own_armies=None), dict(own_armies=()),
                        dict(own_armies=list(obs.own_armies))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(obs, **changes)
        raw = c.encode_observation(obs, schema=c.EXPANDED_SCHEMA)
        for missing in ("army_information", "own_armies"):
            changed = deepcopy(raw); del changed[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                c.decode_observation(changed, schema=c.EXPANDED_SCHEMA)
        with self.assertRaises(ValueError):
            c.encode_observation(obs, schema=c.MANAGEMENT_SCHEMA)

    def test_unknown_private_fields_and_malformed_stack_payload_reject(self):
        raw = c.encode_observation(informed_observation(), schema=c.EXPANDED_SCHEMA)
        for target in ("observation", "army", "stack"):
            changed = deepcopy(raw)
            node = changed if target == "observation" else changed["own_armies"][0]
            if target == "stack": node = node["slots"][0]
            node["hidden_enemy_army"] = 42
            with self.subTest(target=target), self.assertRaises(ValueError):
                c.decode_observation(changed, schema=c.EXPANDED_SCHEMA)
        for changes in (dict(slot=7), dict(count=0), dict(count=True), dict(attack=True),
                        dict(damage_min=4, damage_max=2), dict(name_text_id="private.enemy.army"),
                        dict(shooter=1), dict(health=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                stack(**changes)
        for rows in ((stack(1), stack(0)), (stack(), stack()), [stack()]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                c.OwnArmy("hero", "hero", rows)

    def test_private_routing_identifiers_do_not_become_learned_features(self):
        obs = informed_observation()
        rename = {"hero": "private-101", "other": "private-203", "town": "private-307"}
        changed = replace(obs,
            heroes=tuple(replace(hero, handle=rename[hero.handle]) for hero in obs.heroes),
            towns=tuple(replace(town, handle=rename[town.handle]) for town in obs.towns),
            own_armies=tuple(replace(row, subject=rename[row.subject]) for row in obs.own_armies))
        original, renamed = m.observation_semantics(obs), m.observation_semantics(changed)
        self.assertEqual(original, renamed)
        self.assertTrue(all(value not in repr(renamed) for value in rename.values()))

    def test_each_delivered_stat_and_composition_are_consumed_as_features(self):
        obs = informed_observation()
        original = m.observation_semantics(obs)
        current = obs.own_armies[0].slots[0]
        changes = dict(creature=1, count=15, name_text_id="creatures.core.halberdier.name.plural",
                       attack=7, defense=8, damage_min=2, damage_max=4, health=12,
                       speed=5, shots=8, casts=2, morale=2, luck=0,
                       shooter=True, flying=True, undead=True)
        for field, value in changes.items():
            with self.subTest(field=field):
                first = replace(obs.own_armies[0], slots=(replace(current, **{field: value}),))
                changed = replace(obs, own_armies=(first, *obs.own_armies[1:]))
                self.assertNotEqual(original, m.observation_semantics(changed))
        with torch.random.fork_rng():
            torch.manual_seed(31)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA,
                                                        hidden_size=8, feature_size=4))
        outputs = []
        with torch.no_grad():
            for observed in (observation(), obs, changed):
                history = map_history(observed, schema=c.EXPANDED_SCHEMA)
                frame = m.encode_history(history.frames, guard_version=None)[0]
                outputs.append(model.forward_step(frame, model.initial_state()))
        self.assertFalse(torch.equal(outputs[0].state, outputs[1].state))
        self.assertFalse(torch.equal(outputs[1].state, outputs[2].state))

    def test_old_wire_and_features_are_bit_identical_to_prechange_snapshots(self):
        # Captured before implementing own-army-details-v1 from the existing
        # expanded fixtures, preserving optional-field absence exactly.
        golden = {
            "map": ("af296be685823b9724e895210f0d8b7b88979036eb9a75dc923ef477e12986c5",
                    "94d05f2ff30476365e58d6fab1c004c8932b7c1538ef245dbce11607053d79f1"),
            "hero": ("61aa8128f6b4b06b4e317ca2d11d49f80bf39e3ad92a9225c6988b2f74c2d748",
                     "7c4441de47233cf8a50f744860468dc1ac2ca2cb669ca12ef5cdf5f4a87f0854"),
            "town": ("5c4765213b0e0f83e12f48f2bd5e7cdf4914b74999b0fa9ea7f45c75149dbcb5",
                     "a32f6f9894298070fbc9d3a358ffe1f124b703f2837051c5adaf5859749fef51"),
        }
        for kind, (wire_digest, feature_digest) in golden.items():
            with self.subTest(kind=kind):
                obs = observation(kind)
                raw = c.encode_observation(obs, schema=c.EXPANDED_SCHEMA)
                self.assertNotIn("army_information", raw)
                self.assertNotIn("own_armies", raw)
                wire = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(wire).hexdigest(), wire_digest)
                self.assertEqual(hashlib.sha256(repr(m.observation_semantics(obs)).encode()).hexdigest(), feature_digest)


if __name__ == "__main__":
    unittest.main()
