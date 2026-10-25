"""Synthetic v6 contract regressions; not evidence of native gameplay correctness."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

from experiments.learner import contextual as c
from experiments.headless.contextual_protocol import ContextualProducer
from tests.test_contextual_armies import army


def action(kind, subject=None, **kwargs):
    return c.Action(kind, subject, schema=c.MANAGEMENT_ACTION_SCHEMA, **kwargs)


def observation(kind="map", *, armies=None, heroes=None, stack_panel=None, panel="overview", gold=1000):
    own = heroes or (c.Hero("hero", (1, 1, 0), 100, 100, False), c.Hero("other", (2, 1, 0), 80, 100, False))
    window = None
    if kind == "town":
        rows = armies if armies is not None else (army("upper", "town_garrison", "town", {0: (0, 14)}),
                                                   army("lower", "visiting_hero", "hero", {0: (0, 7), 1: (2, 2)}))
        window = c.Window(c.Context("town", "town", "overview"),
            tuple(c.Stack(slot.slot, slot.creature, slot.count) for slot in rows[0].slots if slot.count),
            faction=0, town_panel=c.TownPanel("overview", (c.PanelControl("hall", "hall"),)), armies=rows)
    elif kind == "hero":
        rows = armies if armies is not None else (army("own", "hero", "hero", {0: (0, 14), 1: (2, 2)}),)
        window = c.Window(c.Context("hero", "hero"),
            tuple(c.Stack(slot.slot, slot.creature, slot.count) for slot in rows[0].slots if slot.count),
            (1, 1, 1, 1), 10, 1, armies=rows)
    elif kind == "exchange":
        rows = armies if armies is not None else (army("first", "hero", "hero", {0: (0, 14)}),
                                                   army("second", "hero", "other", {0: (0, 7), 1: (2, 2)}))
        window = c.Window(c.Context("exchange", "hero", None, "other"), None, armies=rows)
    elif kind == "stack":
        window = c.Window(c.Context("stack", "hero", panel, stack_panel.source.ref), None, armies=(), stack_panel=stack_panel)
    return c.Observation((4, 3, 1), (c.Tile((1, 1, 0), 1, 0, False, False),), (),
                         (0, 0, 0, 0, 0, 0, gold), own, (c.Town("town", (1, 1, 0)),), window)


def decide(history, obs):
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=c.MANAGEMENT_ACTION_SCHEMA),
                        revision=history.revision + 1)


def begin(history, selected, *, consumer="engine"):
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=selected)
    history.emit("consumed", decision_seq=seq, consumer=consumer)
    return seq


def ack(history, seq, accepted=True):
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=accepted)


def navigation(history, selected, after, status="opened"):
    seq = begin(history, selected, consumer="adapter")
    history.emit("context", decision_seq=seq, context=after.context, revision=history.revision + 1)
    history.emit("view", observation=after)
    history.emit("result", decision_seq=seq, result=c.Result(status, True))
    return decide(history, after)


def history_at(kind="town", **kwargs):
    history = c.PlayerHistory("synthetic-management", 0, schema=c.MANAGEMENT_SCHEMA)
    obs = observation(kind, **kwargs)
    decide(history, replace(obs, window=None))
    navigation(history, action("open_town" if kind == "town" else "open_hero", "town" if kind == "town" else "hero"), obs)
    return history


def changed_slots(obs, updates):
    rows = tuple(replace(row, slots=tuple(replace(slot, creature=updates[slot.ref][0], count=updates[slot.ref][1])
                                         if slot.ref in updates else slot for slot in row.slots)) for row in obs.window.armies)
    primary = None if obs.window.army is None else tuple(c.Stack(s.slot, s.creature, s.count) for s in rows[0].slots if s.count)
    return replace(obs, window=replace(obs.window, armies=rows, army=primary))


def at_stack(*, gold=1000):
    history = history_at("hero", gold=gold)
    before = history.decision.observation
    payload = c.StackPanel(before.context, before.window.armies[0].slots[0],
                           (c.UpgradeChoice("upgrade-1", 1), c.UpgradeChoice("upgrade-2", 3)), None, None, None)
    after = observation("stack", stack_panel=payload, gold=gold)
    navigation(history, action("open_stack", "hero", selection="own-0"), after)
    return history


def at_confirmation(*, gold=1000):
    history = at_stack(gold=gold)
    before = history.decision.observation
    payload = replace(before.window.stack_panel, selected_upgrade_ref="upgrade-1", total_cost=(0, 0, 0, 0, 0, 0, 140), affordable=gold >= 140)
    after = observation("stack", stack_panel=payload, panel="upgrade", gold=gold)
    navigation(history, action("select_upgrade", "hero", selection="upgrade-1"), after, "selected")
    return history


def pending_exchange():
    history = c.PlayerHistory("synthetic-management", 0, schema=c.MANAGEMENT_SCHEMA)
    before = observation()
    decide(history, before)
    seq = begin(history, action("move_to", "hero", x=2, y=1))
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("event", event=c.VisibleEvent("army_exchange_requested", "hero", partner="other"))
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    return history, seq


def complete_move(history, seq):
    history.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1, 1, 0), 100, False))


class ManagementContractTests(unittest.TestCase):
    def test_profile_and_old_wire_compatibility(self):
        self.assertEqual((c.MANAGEMENT_PROFILE.protocol, c.MANAGEMENT_PROFILE.version), (8, 6))
        self.assertFalse(c.MANAGEMENT_PROFILE.army_transfer)
        frame = history_at().decision
        raw = c.encode_frame(frame)
        self.assertEqual(c.decode_frame(json.loads(json.dumps(raw))), frame)
        native = deepcopy(raw); native["observation"]["window"]["stack_panel"] = None
        self.assertEqual(c.decode_frame(native), frame)
        self.assertNotIn("armies", raw["observation"]["window"]["town_panel"])
        for profile in (c.WINDOWS_PROFILE, c.TOWN_PROFILE, c.ARMIES_PROFILE, c.HEROES_PROFILE):
            altered = deepcopy(raw); altered["schema"] = profile.history_schema
            with self.assertRaises(ValueError): c.decode_frame(altered)
        with self.assertRaises(ValueError): action("transfer_stack", "town", selection="upper-0", destination="lower-0", count=1)
        for location in (raw["observation"]["window"]["armies"][0], raw["offers"]["armies"][0]):
            location["hidden_army_id"] = 40
            with self.assertRaises(ValueError): c.decode_frame(raw)
            location.pop("hidden_army_id")
        from tests.test_contextual_heroes import history_at_tavern
        old = c.encode_frame(history_at_tavern().decision)
        self.assertNotIn("in_garrison", old["observation"]["heroes"][0])
        self.assertNotIn("armies", old["observation"]["window"])
        self.assertNotIn("movable_hero_refs", old["offers"])

    def test_garrison_hero_stays_owned_openable_but_not_movable(self):
        obs = observation(heroes=(c.Hero("hero", (1, 1, 0), 100, 100, True), c.Hero("other", (1, 1, 0), 100, 100, False)))
        offers = c.Offers.from_observation(obs, schema=c.MANAGEMENT_ACTION_SCHEMA)
        self.assertTrue(offers.permits(action("open_hero", "hero")))
        self.assertFalse(offers.permits(action("move_to", "hero", x=0, y=0)))
        self.assertTrue(offers.permits(action("move_to", "other", x=0, y=0)))
        with self.assertRaises(ValueError): c.encode_observation(replace(obs, heroes=(replace(obs.heroes[0], in_garrison=None),)), schema=c.MANAGEMENT_SCHEMA)

    def test_support_same_army_both_directions_last_stack_and_overflow(self):
        support = history_at().decision.offers
        for selected in (action("split_stack", "town", selection="upper-0", destination="lower-0", count=14),
                         action("split_stack", "town", selection="lower-0", destination="upper-0", count=7),
                         action("split_stack", "town", selection="lower-0", destination="lower-6", count=7),
                         action("merge_stacks", "town", selection="lower-0", destination="upper-0"),
                         action("swap_stacks", "town", selection="upper-0", destination="lower-0"),
                         action("swap_stacks", "town", selection="lower-6", destination="upper-0")):
            self.assertTrue(support.permits(selected), selected)
        one = (army("upper", "town_garrison", "town", {}), army("lower", "visiting_hero", "hero", {0: (0, 7)}))
        support = history_at(armies=one).decision.offers
        self.assertFalse(support.permits(action("split_stack", "town", selection="lower-0", destination="upper-0", count=7)))
        self.assertTrue(support.permits(action("split_stack", "town", selection="lower-0", destination="lower-1", count=7)))
        self.assertFalse(support.permits(action("swap_stacks", "town", selection="upper-0", destination="lower-0")))
        huge = (army("upper", "town_garrison", "town", {0: (0, 2**31-1)}), one[1])
        self.assertFalse(c.army_operation_maximum(huge, "split_stack", "upper-0", "lower-0"))
        self.assertEqual(c.army_operation_maximum(huge, "swap_stacks", "upper-0", "lower-0"), 1)

    def test_exact_native_army_effects_and_affected_speed_refresh(self):
        cases = (("split_stack", "upper-0", "lower-0", 3, {"upper-0": (0, 11), "lower-0": (0, 10)}),
                 ("merge_stacks", "lower-0", "upper-0", None, {"lower-0": (None, 0), "upper-0": (0, 21)}),
                 ("swap_stacks", "upper-0", "lower-0", None, {"upper-0": (0, 7), "lower-0": (0, 14)}),
                 ("swap_stacks", "lower-6", "upper-0", None, {"upper-0": (None, 0), "lower-6": (0, 14)}))
        for kind, source, dest, count, updates in cases:
            with self.subTest(kind=kind, source=source):
                history = history_at(); before = history.decision.observation
                selected = action(kind, "town", selection=source, destination=dest, count=count)
                seq = begin(history, selected); ack(history, seq)
                after = changed_slots(before, updates)
                after = replace(after, heroes=(replace(after.heroes[0], movement_max=150), after.heroes[1]))
                history.emit("view", observation=after)
                history.emit("result", decision_seq=seq, result=c.Result({"split_stack":"split", "merge_stacks":"merged", "swap_stacks":"swapped"}[kind], True))
        for corruption in ("missing", "gold", "movement", "unaffected_speed"):
            history = history_at(); before = history.decision.observation
            seq = begin(history, action("swap_stacks", "town", selection="upper-0", destination="lower-0")); ack(history, seq)
            after = changed_slots(before, {"upper-0": (0, 7), "lower-0": (0, 14)})
            if corruption == "missing": after = before
            elif corruption == "gold": after = replace(after, resources=(0,)*7)
            elif corruption == "movement": after = replace(after, heroes=(replace(after.heroes[0], movement=99), after.heroes[1]))
            else: after = replace(after, heroes=(after.heroes[0], replace(after.heroes[1], movement_max=150)))
            history.emit("view", observation=after)
            with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("swapped", True))

    def test_rejected_army_action_requires_rejected_native_ack_and_no_effect(self):
        history = history_at(); seq = begin(history, action("merge_stacks", "town", selection="upper-0", destination="lower-0"))
        ack(history, seq, accepted=False)
        history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
        history = history_at(); seq = begin(history, action("merge_stacks", "town", selection="upper-0", destination="lower-0")); ack(history, seq)
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("rejected", False))

    def test_stack_fields_are_explicit_and_price_appears_only_after_selection(self):
        history = at_stack(); frame = history.decision
        raw = c.encode_frame(frame)
        self.assertEqual(c.decode_frame(raw), frame)
        panel = frame.observation.window.stack_panel
        self.assertIsNone(panel.total_cost)
        self.assertEqual(frame.observation.window.armies, ())
        self.assertTrue(frame.offers.permits(action("select_upgrade", "hero", selection="upgrade-2")))
        self.assertFalse(frame.offers.permits(action("upgrade_stack", "hero", selection="upgrade-1")))
        with self.assertRaises(ValueError): replace(panel, total_cost=(0,)*7)
        with self.assertRaises(ValueError): replace(frame.observation.window, armies=observation("hero").window.armies)
        self.assertEqual(c.decode_frame(c.encode_frame(at_confirmation().decision)), at_confirmation().decision)

    def test_only_selected_affordable_upgrade_can_be_accepted(self):
        frame = at_confirmation().decision
        self.assertTrue(frame.offers.permits(action("upgrade_stack", "hero", selection="upgrade-1")))
        self.assertFalse(frame.offers.permits(action("upgrade_stack", "hero", selection="upgrade-2")))
        low = at_confirmation(gold=10).decision
        self.assertFalse(low.offers.permits(action("upgrade_stack", "hero", selection="upgrade-1")))
        self.assertTrue(low.offers.permits(action("back", "hero")))
        self.assertTrue(at_stack(gold=10).decision.offers.permits(action("select_upgrade", "hero", selection="upgrade-1")))

    def test_upgrade_closes_before_native_issue_and_matches_whole_stack_cost(self):
        for corrupt in (None, "cost", "count", "target", "source"):
            history = at_confirmation(); seq = begin(history, action("upgrade_stack", "hero", selection="upgrade-1"))
            parent = observation("hero", gold=860)
            after = changed_slots(parent, {"own-0": (1, 14)})
            after = replace(after, heroes=(replace(after.heroes[0], movement_max=150), after.heroes[1]))
            history.emit("context", decision_seq=seq, context=parent.context, revision=history.revision+1)
            ack(history, seq)
            if corrupt == "cost": after = replace(after, resources=(0,0,0,0,0,0,861))
            elif corrupt == "count": after = changed_slots(after, {"own-0": (1, 13)})
            elif corrupt == "target": after = changed_slots(after, {"own-0": (3, 14)})
            elif corrupt == "source": after = changed_slots(after, {"own-0": (0, 14), "own-1": (1, 2)})
            history.emit("view", observation=after)
            if corrupt:
                with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("upgraded", True))
            else: history.emit("result", decision_seq=seq, result=c.Result("upgraded", True))

    def test_stack_back_steps_restore_the_frozen_parent(self):
        history = at_confirmation(); before = history.decision.observation
        payload = replace(before.window.stack_panel, selected_upgrade_ref=None, total_cost=None, affordable=None)
        navigation(history, action("back", "hero"), observation("stack", stack_panel=payload), "closed")
        navigation(history, action("back", "hero"), observation("hero"), "closed")
        self.assertEqual(history.context, c.Context("hero", "hero"))
        self.assertFalse(any(frame.kind == "issued" for frame in history.frames))

    def test_garrison_blocked_notice_is_profile_bound_and_has_no_native_packet(self):
        notice = c.Notice("modal", (c.NoticeToken("append_text", text_id="core.genrltxt.18"),
                                   c.NoticeToken("replace_number", number=8)), ())
        with self.assertRaises(ValueError): c.Frame("old", 0, 0, 1, "event", event=c.VisibleEvent("notice", notice=notice), schema=c.HEROES_SCHEMA)
        history = history_at(); seq = begin(history, action("swap_garrison", "town"), consumer="adapter")
        history.emit("event", event=c.VisibleEvent("notice", notice=notice))
        with self.assertRaises(ValueError): history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
        history = history_at(); seq = begin(history, action("swap_garrison", "town"), consumer="adapter")
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("rejected", False))

    def test_native_garrison_enter_and_exit_conserve_creatures_and_bind_roles(self):
        history = history_at(); before = history.decision.observation
        seq = begin(history, action("swap_garrison", "town")); ack(history, seq)
        rows = (army("lower", "garrison_hero", "hero", {0: (0, 21), 1: (2, 2)}),)
        heroes = (replace(before.heroes[0], in_garrison=True, movement_max=120), before.heroes[1])
        entered = observation("town", armies=rows, heroes=heroes)
        history.emit("view", observation=entered)
        history.emit("result", decision_seq=seq, result=c.Result("garrison_swapped", True))
        decide(history, entered); seq = begin(history, action("swap_garrison", "town")); ack(history, seq)
        exited = observation("town", armies=(army("town-new", "town_garrison", "town", {}), replace(rows[0], role="visiting_hero")),
                             heroes=(replace(heroes[0], in_garrison=False), heroes[1]))
        history.emit("view", observation=exited)
        history.emit("result", decision_seq=seq, result=c.Result("garrison_swapped", True))

    def test_occupied_garrison_swap_changes_roles_only(self):
        rows = (army("upper", "garrison_hero", "other", {0: (1, 3)}),
                army("lower", "visiting_hero", "hero", {0: (0, 7), 1: (2, 2)}))
        heroes = (c.Hero("hero", (1,1,0), 100, 100, False), c.Hero("other", (1,1,0), 80, 100, True))
        for corrupt in (False, True):
            history = history_at(armies=rows, heroes=heroes)
            seq = begin(history, action("swap_garrison", "town")); ack(history, seq)
            after = observation("town", armies=(replace(rows[1], role="garrison_hero"), replace(rows[0], role="visiting_hero")),
                                heroes=(replace(heroes[0], in_garrison=True), replace(heroes[1], in_garrison=False)))
            if corrupt:
                after = changed_slots(after, {"lower-0": (1, 3), "upper-0": (0, 7)})
            history.emit("view", observation=after)
            if corrupt:
                with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("garrison_swapped", True))
            else: history.emit("result", decision_seq=seq, result=c.Result("garrison_swapped", True))

    def test_exchange_opens_only_after_parent_result_then_close_uses_native_reply(self):
        history, seq = pending_exchange()
        ctx = observation("exchange").context
        with self.assertRaises(ValueError): history.emit("context_opened", context=ctx, revision=history.revision+1, reason="own_hero_exchange")
        complete_move(history, seq)
        with self.assertRaises(ValueError): decide(history, observation())
        with self.assertRaises(ValueError): history.emit("context_opened", context=replace(ctx, selection="unknown"), revision=history.revision+1, reason="own_hero_exchange")
        history.emit("context_opened", context=ctx, revision=history.revision+1, reason="own_hero_exchange")
        exchange = observation("exchange"); history.emit("view", observation=exchange); decide(history, exchange)
        seq = begin(history, action("close"))
        history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision+1)
        ack(history, seq)
        history.emit("view", observation=observation())
        history.emit("result", decision_seq=seq, result=c.Result("closed", True))
        self.assertEqual(history.context, c.Context())
        with self.assertRaises(ValueError): history.emit("context_opened", context=ctx, revision=history.revision+1, reason="own_hero_exchange")
        replay = c.PlayerHistory(history.episode, history.player, schema=history.schema)
        for frame in history.frames: replay.append(c.decode_frame(c.encode_frame(frame)))
        self.assertEqual(history.digest(), replay.digest())

    def test_exchange_event_cannot_be_forged_from_nonmovement_or_foreign_hero(self):
        history = history_at()
        with self.assertRaises(ValueError): history.emit("event", event=c.VisibleEvent("army_exchange_requested", "hero", partner="other"))
        history = c.PlayerHistory("synthetic-management", 0, schema=c.MANAGEMENT_SCHEMA)
        decide(history, observation()); seq = begin(history, action("move_to", "hero", x=2, y=1))
        history.emit("issued", decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError): history.emit("event", event=c.VisibleEvent("army_exchange_requested", "hero", partner="foreign"))

    def test_transport_context_opened_has_no_fabricated_decision_identifier(self):
        history, seq = pending_exchange(); complete_move(history, seq)
        history.emit("context_opened", context=observation("exchange").context, revision=history.revision+1, reason="own_hero_exchange")
        p = ContextualProducer(history.episode, 0, interface=c.MANAGEMENT_ENVIRONMENT)
        p.ingest(dict(protocol_version=8, type="ready", episode=history.episode, player=0,
                      interface=c.MANAGEMENT_ENVIRONMENT, history_schema=c.MANAGEMENT_SCHEMA, action_schema=c.MANAGEMENT_ACTION_SCHEMA))
        for frame in history.frames:
            if frame.kind == "submitted": p.submitted(frame.action); continue
            raw = c.encode_frame(frame)
            from experiments.headless.contextual_protocol import RECORD_FIELDS
            record = {"kind":frame.kind, "day":frame.day}
            for key in RECORD_FIELDS[frame.kind]:
                record[key] = 42 if key == "decision_id" else raw[key]
            p.ingest(dict(protocol_version=8, type="contextual", episode=history.episode, player=0, records=[record]))
        self.assertEqual(p.live.digest(), history.digest())
        self.assertIsNone(p.current_id)


if __name__ == "__main__": unittest.main()
