"""Synthetic evidence checks; these do not claim native transfer validation."""
from dataclasses import replace
import unittest

from experiments.learner import contextual as c
from validation.probes.validate_contextual_native import ProbeFailure
from validation.probes.validate_town_transfer import exact_transfer


def row(prefix, role, subject, stacks):
    return c.ArmyView(prefix, role, subject, role != "town_garrison", tuple(
        c.ArmySlot(f"{prefix}-{slot}", slot, *stacks.get(slot, (None, 0))) for slot in range(7)))


def observed(source, destination):
    exchange = c.ArmyExchange(source, destination)
    window = c.Window(c.Context("town", "town", "overview"),
                      tuple(c.Stack(s.slot, s.creature, s.count) for s in source.slots if s.count),
                      faction=0, town_panel=c.TownPanel("overview", armies=exchange))
    return c.Observation((4, 4, 1), (), (), (0, 0, 0, 0, 0, 0, 100),
                         (c.Hero("hero", (1, 1, 0), 100, 100),), (c.Town("town", (1, 1, 0)),), window)


def frame(seq, kind, **payload):
    return c.Frame("synthetic-transfer-probe", 0, seq, 1, kind, schema=c.ARMIES_SCHEMA, **payload)


def evidence(*, count=2, destination=0):
    upper = row("upper", "town_garrison", "town", {0: (0, 3), 4: (1, 1)})
    lower = row("lower", "visiting_hero", "hero", {0: (0, 5), 1: (1, 1)})
    before_obs = observed(upper, lower)
    before = frame(0, "decision", observation=before_obs, revision=1,
                   offers=c.Offers.from_observation(before_obs, schema=c.ARMIES_ACTION_SCHEMA))
    action = c.Action("transfer_stack", "town", schema=c.ARMIES_ACTION_SCHEMA,
                      selection="upper-0", destination=f"lower-{destination}", count=count)
    after_upper = list(upper.slots)
    after_upper[0] = replace(after_upper[0], count=3 - count, creature=0 if count < 3 else None)
    after_lower = list(lower.slots)
    after_lower[destination] = replace(after_lower[destination], creature=0,
                                       count=after_lower[destination].count + count)
    after_obs = observed(replace(upper, slots=tuple(after_upper)), replace(lower, slots=tuple(after_lower)))
    after = frame(6, "decision", observation=after_obs, revision=2,
                  offers=c.Offers.from_observation(after_obs, schema=c.ARMIES_ACTION_SCHEMA))
    result = c.Result("transferred", True)
    additions = (
        frame(1, "consumed", decision_seq=0, consumer="engine"),
        frame(2, "issued", decision_seq=0, command_index=0),
        frame(3, "acknowledged", decision_seq=0, command_index=0, accepted=True),
        frame(4, "view", observation=after_obs),
        frame(5, "result", decision_seq=0, result=result),
    )
    return before, after, action, additions, result


class TransferProbeTests(unittest.TestCase):
    def test_exact_partial_merge_and_full_empty_slot_move_keep_explicit_count(self):
        for count, destination, total in ((2, 0, 7), (3, 6, 3)):
            with self.subTest(count=count, destination=destination):
                outcome = exact_transfer(*evidence(count=count, destination=destination))
                self.assertEqual(outcome["native_destination_total"], total)
                self.assertEqual(outcome["count"], count)
                self.assertEqual(outcome["source_remaining"], 3 - count)
                self.assertFalse(outcome["native_rejection"])

    def test_status_or_ack_without_exact_effect_is_not_success(self):
        before, after, action, additions, result = evidence()
        wrong = evidence(count=1)[1]
        for after_frame in (replace(after, observation=before.observation,
                                    offers=before.offers), wrong):
            with self.assertRaises(ProbeFailure):
                exact_transfer(before, after_frame, action, additions, result)
        for changed in (additions[:2] + additions[3:],
                        additions[:2] + (replace(additions[2], accepted=False),) + additions[3:],
                        additions[:2] + (replace(additions[2], command_index=1),) + additions[3:],
                        additions + (replace(additions[2], seq=7),)):
            with self.assertRaises(ProbeFailure):
                exact_transfer(before, after, action, changed, result)

    def test_unrelated_slot_resource_or_window_changes_fail(self):
        before, after, action, additions, result = evidence()
        obs = after.observation
        rows = obs.window.town_panel.armies
        slots = list(rows.destination.slots)
        slots[1] = replace(slots[1], count=2)
        wrong_slot = observed(rows.source, replace(rows.destination, slots=tuple(slots)))
        for changed in (wrong_slot, replace(obs, resources=(0,) * 7)):
            after_frame = replace(after, observation=changed,
                                  offers=c.Offers.from_observation(changed, schema=c.ARMIES_ACTION_SCHEMA))
            with self.assertRaises(ProbeFailure):
                exact_transfer(before, after_frame, action, additions, result)
        navigation = frame(7, "context", decision_seq=0, context=obs.context, revision=2)
        with self.assertRaises(ProbeFailure):
            exact_transfer(before, after, action, additions + (navigation,), result)

    def test_legitimate_public_hero_movement_max_change_is_not_a_false_failure(self):
        before, after, action, additions, result = evidence()
        obs = replace(after.observation, heroes=(replace(after.observation.heroes[0], movement_max=120),))
        updated = replace(after, observation=obs, offers=c.Offers.from_observation(obs, schema=c.ARMIES_ACTION_SCHEMA))
        self.assertEqual(exact_transfer(before, updated, action, additions, result)["count"], 2)


if __name__ == "__main__":
    unittest.main()
