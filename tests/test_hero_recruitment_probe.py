"""Synthetic receipt checks; these do not claim native Tavern validation."""
from dataclasses import replace
import unittest

from experiments.learner import contextual as c
from validation.probes.validate_contextual_native import ProbeFailure
from validation.probes.validate_hero_recruitment import exact_hire


def frame(seq, kind, **payload):
    return c.Frame("synthetic-hero-probe", 0, seq, 1, kind, schema=c.HEROES_SCHEMA, **payload)


def army(ref, role, subject, creature=None):
    return c.ArmyView(ref, role, subject, role != "town_garrison", tuple(
        c.ArmySlot(f"{ref}-{slot}", slot, creature if slot == 0 else None,
                   3 if slot == 0 and creature is not None else 0) for slot in range(7)))


def evidence(recovery=False):
    offer = c.HeroRecruitChoice("offer-1", 1, "hero.core.edric.name")
    tavern = c.TavernPanel((offer,), offer.ref, c.HeroRecruitDetails(1, 0, 0), 2500, "allowed")
    before_window = c.Window(c.Context("town", "town", "tavern"), None,
                             town_panel=c.TownPanel("tavern", tavern=tavern))
    heroes = () if recovery else (c.Hero("original", (2, 1, 0), 100, 100),)
    towns = (c.Town("town", (1, 1, 0)),)
    before_obs = c.Observation((4, 4, 1), (), (), (10, 2, 3, 4, 5, 6, 5000), heroes, towns, before_window)
    added = c.Hero("new", (1, 1, 0), 100, 100)
    exchange = c.ArmyExchange(army("upper", "town_garrison", "town"),
                               army("visitor", "visiting_hero", "new", 0))
    after_window = c.Window(c.Context("town", "town", "overview"), (), faction=0,
                            town_panel=c.TownPanel("overview", armies=exchange))
    after_obs = replace(before_obs, resources=(10, 2, 3, 4, 5, 6, 2500), heroes=heroes + (added,), window=after_window)
    before = frame(0, "decision", observation=before_obs, revision=1,
                   offers=c.Offers.from_observation(before_obs, schema=c.HEROES_ACTION_SCHEMA))
    after = frame(7, "decision", observation=after_obs, revision=3,
                  offers=c.Offers.from_observation(after_obs, schema=c.HEROES_ACTION_SCHEMA))
    action = c.Action("hire_hero", "town", schema=c.HEROES_ACTION_SCHEMA, selection=offer.ref)
    result = c.Result("hired", True)
    additions = (
        frame(1, "consumed", decision_seq=0, consumer="engine"),
        frame(2, "issued", decision_seq=0, command_index=0),
        frame(3, "context", decision_seq=0, context=after_obs.context, revision=2),
        frame(4, "acknowledged", decision_seq=0, command_index=0, accepted=True),
        frame(5, "view", observation=after_obs),
        frame(6, "result", decision_seq=0, result=result),
    )
    return before, after, action, additions, result


class HeroRecruitmentProbeTests(unittest.TestCase):
    def test_exact_hire_can_add_second_or_replacement_hero(self):
        for recovery in (False, True):
            with self.subTest(recovery=recovery):
                report = exact_hire(*evidence(recovery))
                self.assertEqual(report["heroes_after"], 1 if recovery else 2)
                self.assertEqual(report["gold_cost"], 2500)
                self.assertEqual(report["new_hero"], "new")
                self.assertFalse(report["native_rejection"])

    def test_hired_status_cannot_replace_missing_or_conflicting_native_receipt(self):
        before, after, action, additions, result = evidence()
        variants = (additions[:3] + additions[4:],
                    additions[:3] + (replace(additions[3], accepted=False),) + additions[4:],
                    additions[:3] + (replace(additions[3], command_index=1),) + additions[4:],
                    additions + (replace(additions[3], seq=8),),
                    (replace(additions[0], consumer="adapter"),) + additions[1:])
        for changed in variants:
            with self.subTest(additions=changed), self.assertRaises(ProbeFailure):
                exact_hire(before, after, action, changed, result)

    def test_exact_resources_and_owned_hero_creation_are_required(self):
        before, after, action, additions, result = evidence()
        obs = after.observation
        variants = (
            replace(obs, resources=(10, 2, 3, 4, 5, 6, 2501)),
            replace(obs, resources=(9, 2, 3, 4, 5, 6, 2500)),
            replace(obs, heroes=before.observation.heroes, window=None),
            replace(obs, heroes=obs.heroes + (replace(obs.heroes[-1], handle="extra"),)),
            replace(obs, heroes=(replace(obs.heroes[0], movement=99), obs.heroes[1])),
            replace(obs, heroes=(obs.heroes[0], replace(obs.heroes[1], position=(3, 3, 0)))),
        )
        for changed in variants:
            with self.subTest(observation=changed), self.assertRaises(ProbeFailure):
                exact_hire(before, replace(after, observation=changed,
                           offers=c.Offers.from_observation(changed, schema=c.HEROES_ACTION_SCHEMA)), action, additions, result)

    def test_selected_offer_and_actual_visitor_must_match(self):
        before, after, action, additions, result = evidence()
        with self.assertRaises(ProbeFailure):
            exact_hire(before, after, replace(action, selection="foreign"), additions, result)
        for window in (None, before.observation.window,
                       replace(after.observation.window, town_panel=c.TownPanel("overview", armies=c.ArmyExchange(
                           after.observation.window.town_panel.armies.source, None)))):
            with self.subTest(window=window), self.assertRaises(ProbeFailure):
                changed = replace(after.observation, window=window)
                exact_hire(before, replace(after, observation=changed,
                           offers=c.Offers.from_observation(changed, schema=c.HEROES_ACTION_SCHEMA)),
                           action, additions, result)


if __name__ == "__main__":
    unittest.main()
