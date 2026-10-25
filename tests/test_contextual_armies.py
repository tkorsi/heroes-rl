"""Synthetic v4 contract evidence, never simulated native-game validation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from experiments.learner import contextual as c


def army(prefix, role, subject, stacks):
    return c.ArmyView(prefix, role, subject, role != "town_garrison",
                      tuple(c.ArmySlot(f"{prefix}-{slot}", slot, stacks.get(slot, (None, 0))[0],
                                      stacks.get(slot, (None, 0))[1]) for slot in range(7)))


def exchange(source=None, destination=None, *, visitor=True):
    return c.ArmyExchange(source or army("upper", "town_garrison", "town", {0: (0, 3)}),
                          (destination or army("lower", "visiting_hero", "hero", {0: (0, 2), 1: (1, 1)})) if visitor else None)


def observation(armies=None, *, opened=True):
    window = None
    if opened:
        armies = armies or exchange()
        panel = c.TownPanel("overview", armies=armies)
        window = c.Window(c.Context("town", "town", "overview"),
                          tuple(c.Stack(slot.slot, slot.creature, slot.count) for slot in armies.source.slots if slot.count),
                          faction=0, town_panel=panel)
    return c.Observation((4, 4, 1), (c.Tile((1, 1, 0), 1, 0, False, False),), (), (0, 0, 0, 0, 0, 0, 100),
                         (c.Hero("hero", (1, 1, 0), 100, 100), c.Hero("garrison", (1, 1, 0), 100, 100)),
                         (c.Town("town", (1, 1, 0)),), window)


def decide(history, obs):
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=c.ARMIES_ACTION_SCHEMA),
                        revision=history.revision + 1)


def history_at_armies(player=0, armies=None):
    history = c.PlayerHistory("synthetic-armies", player, schema=c.ARMIES_SCHEMA)
    start = decide(history, observation(opened=False))
    history.emit("submitted", decision_seq=start.seq, action=c.Action("open_town", "town", schema=c.ARMIES_ACTION_SCHEMA))
    history.emit("consumed", decision_seq=start.seq, consumer="adapter")
    obs = observation(armies)
    history.emit("context", decision_seq=start.seq, context=obs.context, revision=history.revision + 1)
    history.emit("view", observation=obs)
    history.emit("result", decision_seq=start.seq, result=c.Result("opened", True))
    decide(history, obs)
    return history


def selected(count=1, source="upper-0", destination="lower-0"):
    return c.Action("transfer_stack", "town", schema=c.ARMIES_ACTION_SCHEMA, selection=source,
                    destination=destination, count=count)


def begin_transfer(history, action=None, *, accepted=True):
    action = action or selected()
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=action)
    history.emit("consumed", decision_seq=seq, consumer="engine")
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=accepted)
    return seq


def after_transfer(before, action):
    original = before.window.town_panel.armies
    source = next(slot for slot in original.source.slots if slot.ref == action.selection)
    source_slots = tuple(replace(slot, count=slot.count - action.count,
                                 creature=slot.creature if slot.count > action.count else None)
                         if slot.ref == action.selection else slot for slot in original.source.slots)
    dest_slots = tuple(replace(slot, count=slot.count + action.count, creature=source.creature)
                       if slot.ref == action.destination else slot for slot in original.destination.slots)
    changed = c.ArmyExchange(replace(original.source, slots=source_slots), replace(original.destination, slots=dest_slots))
    return replace(before, window=replace(before.window, town_panel=replace(before.window.town_panel, armies=changed),
        army=tuple(c.Stack(slot.slot, slot.creature, slot.count) for slot in source_slots if slot.count)))


class ArmyContractTests(unittest.TestCase):
    def test_wire_roundtrip_is_immutable_and_rejects_older_or_hidden_payloads(self):
        frame = history_at_armies().decision
        raw = c.encode_frame(frame)
        self.assertEqual(c.decode_frame(json.loads(json.dumps(raw))), frame)
        with self.assertRaises(FrozenInstanceError):
            frame.observation.window.town_panel.armies.source.slots[0].count = 5
        for schema in (c.SCHEMA, c.TOWN_SCHEMA):
            changed = deepcopy(raw); changed["schema"] = schema
            with self.assertRaises(ValueError): c.decode_frame(changed)
        for path in (("observation", "window", "town_panel", "armies", "source"),
                     ("offers", "panel", "armies", "destination")):
            changed = deepcopy(raw); node = changed
            for key in path: node = node[key]
            node["hidden_army_id"] = 42
            with self.assertRaises(ValueError): c.decode_frame(changed)
        changed = deepcopy(raw)
        del changed["observation"]["window"]["town_panel"]["armies"]
        with self.assertRaises(ValueError): c.decode_frame(changed)

    def test_support_uses_slots_and_exact_native_count_constraints(self):
        offers = history_at_armies().decision.offers
        self.assertTrue(offers.permits(selected(3)))
        self.assertTrue(offers.permits(selected(3, destination="lower-6")))
        for action in (selected(4), selected(destination="lower-1"), selected(source="upper-1"),
                       selected(source="lower-0", destination="upper-0"), selected(destination="unknown")):
            self.assertFalse(offers.permits(action))
        self.assertFalse(history_at_armies(armies=exchange(visitor=False)).decision.offers.permits(selected()))
        for invalid in (0, -1, True, 1.5, 2**31):
            with self.assertRaises(ValueError): selected(invalid)
        with self.assertRaises(ValueError): c.Action("transfer_stack", "town", schema=c.TOWN_ACTION_SCHEMA,
                                                    selection="upper-0", destination="lower-0", count=1)

    def test_last_stack_and_entire_pair_overflow_are_not_silently_clamped(self):
        source = army("upper", "garrison_hero", "garrison", {0: (0, 3)})
        offered = history_at_armies(armies=exchange(source=source)).decision.offers
        self.assertTrue(offered.permits(selected(2)))
        self.assertFalse(offered.permits(selected(3)))
        source = army("upper", "town_garrison", "town", {0: (0, 2**31 - 2)})
        offered = history_at_armies(armies=exchange(source=source)).decision.offers
        self.assertFalse(offered.permits(selected(1)))  # total with destination's 2 overflows
        self.assertTrue(offered.permits(selected(1, destination="lower-6")))

    def test_split_full_move_and_merge_require_exact_effects_and_one_ack(self):
        for destination in ("lower-0", "lower-6"):
            for count in (1, 3):
                with self.subTest(destination=destination, count=count):
                    history = history_at_armies()
                    before = history.decision.observation
                    action = selected(count, destination=destination)
                    seq = begin_transfer(history, action)
                    changed = after_transfer(before, action)
                    # Army changes may legitimately alter this public hero statistic.
                    changed = replace(changed, heroes=(replace(changed.heroes[0], movement_max=150), changed.heroes[1]))
                    history.emit("view", observation=changed)
                    history.emit("result", decision_seq=seq, result=c.Result("transferred", True))
                    self.assertEqual(history.context, before.context)
        for corruption in ("missing_effect", "wrong_count", "resource_debit", "unrelated_slot"):
            history = history_at_armies(); before = history.decision.observation
            seq = begin_transfer(history)
            changed = after_transfer(before, selected(2 if corruption == "wrong_count" else 1))
            if corruption == "missing_effect": changed = before
            if corruption == "resource_debit": changed = replace(changed, resources=(0,) * 7)
            if corruption == "unrelated_slot":
                armies = changed.window.town_panel.armies
                slots = list(armies.destination.slots); slots[1] = replace(slots[1], count=2)
                changed = replace(changed, window=replace(changed.window, town_panel=replace(changed.window.town_panel,
                    armies=replace(armies, destination=replace(armies.destination, slots=tuple(slots))))))
            history.emit("view", observation=changed)
            with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("transferred", True))

    def test_rejected_ack_cannot_claim_success_or_change_armies(self):
        for accepted in (False, True):
            history = history_at_armies(); seq = begin_transfer(history, accepted=accepted)
            if accepted:
                with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
            else:
                history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
        history = history_at_armies(); before = history.decision.observation
        seq = begin_transfer(history, accepted=False)
        history.emit("view", observation=after_transfer(before, selected()))
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("rejected", False))

    def test_no_command_or_adapter_consumption_cannot_be_transfer_success(self):
        history = history_at_armies(); seq = history.decision.seq
        history.emit("submitted", decision_seq=seq, action=selected())
        with self.assertRaises(ValueError): history.emit("consumed", decision_seq=seq, consumer="adapter")
        history.emit("consumed", decision_seq=seq, consumer="engine")
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("transferred", True))

    def test_close_removes_both_armies_and_stale_transfer_is_unoffered(self):
        history = history_at_armies(); seq = history.decision.seq
        history.emit("submitted", decision_seq=seq, action=c.Action("close", schema=c.ARMIES_ACTION_SCHEMA))
        history.emit("consumed", decision_seq=seq, consumer="adapter")
        history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision + 1)
        history.emit("view", observation=observation(opened=False))
        history.emit("result", decision_seq=seq, result=c.Result("closed", True))
        frame = decide(history, observation(opened=False))
        self.assertIsNone(frame.observation.window)
        self.assertFalse(frame.offers.permits(selected()))
        self.assertNotIn('"armies":', json.dumps(c.encode_frame(frame)))


if __name__ == "__main__":
    unittest.main()
