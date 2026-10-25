"""Synthetic v7 chronology/support tests; these are not native-game evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from experiments.learner import contextual as c
from experiments.headless.contextual_protocol import ContextualProducer, RECORD_FIELDS, clone_history
from tests.test_contextual_management import observation as map_observation


FAMILIES = ("guarded_mine", "neutral_free_join", "neutral_paid_join", "neutral_flee",
            "chest", "arena", "school_magic", "school_war")


def panel(family="chest", ref="dialog-1"):
    number = dict(zip(FAMILIES, (84, 86, 90, 91, 146, 0, 71, 158)))[family]
    text = (c.NoticeToken("append_text", text_id=f"core.advevent.{number}"),)
    components = ()
    selection = family in ("chest", "arena", "school_magic", "school_war")
    cancel = family not in ("chest", "arena")
    if family.startswith("neutral_"):
        if family == "neutral_paid_join":
            text += (c.NoticeToken("replace_number", number=6), c.NoticeToken("replace_number", number=360))
            components = (c.DialogComponent("resource", "gold", 360),)
        text += (c.NoticeToken("replace_text", text_id="creatures.core.pikeman.name.plural"),)
    if family == "chest":
        components = (c.DialogComponent("resource", "gold", 1000), c.DialogComponent("experience", amount=500))
    elif family in ("arena", "school_magic", "school_war"):
        skills = ("spellpower", "knowledge") if family == "school_magic" else ("attack", "defence")
        components = tuple(c.DialogComponent("primary_skill", amount=2 if family == "arena" else 1, skill=skill) for skill in skills)
    if selection:
        choices = tuple(c.DialogChoice(f"{ref}-choice-{index}", "select_component", index) for index in range(2))
        if cancel: choices += (c.DialogChoice(f"{ref}-cancel", "cancel"),)
    else:
        kinds = ("pursue", "let_flee") if family == "neutral_flee" else ("accept", "decline")
        choices = tuple(c.DialogChoice(f"{ref}-{kind}", kind) for kind in kinds)
    return c.DialogPanel(ref, family, text, components, selection, cancel, choices)


def observation(delivered=None):
    base = map_observation()
    if delivered is None: return base
    return replace(base, window=c.Window(c.Context("dialog", "hero", None, delivered.ref), None,
                                        armies=(), dialog_panel=delivered))


def action(kind="choose_dialog", subject="hero", **kwargs):
    return c.Action(kind, subject, schema=c.DIALOGS_ACTION_SCHEMA, **kwargs)


def decide(history, obs):
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=c.DIALOGS_ACTION_SCHEMA),
                        revision=history.revision+1)


def start_move():
    history = c.PlayerHistory("synthetic-dialogs", 0, schema=c.DIALOGS_SCHEMA)
    decision = decide(history, observation())
    history.emit("submitted", decision_seq=decision.seq, action=action("move_to", x=3, y=2))
    history.emit("consumed", decision_seq=decision.seq, consumer="engine")
    history.emit("issued", decision_seq=decision.seq, command_index=0)
    return history, decision.seq


def pending_dialog(delivered=None):
    delivered = panel() if delivered is None else delivered
    history, seq = start_move()
    history.emit("event", event=c.VisibleEvent("dialog_requested", "hero", dialog=delivered))
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    return history, seq


def finish_move(history, seq):
    history.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1, 1, 0), 100, False))


def open_dialog(history, delivered):
    history.emit("context_opened", context=observation(delivered).context, revision=history.revision+1, reason="adventure_dialog")
    return decide(history, observation(delivered))


def history_at(delivered=None):
    delivered = panel() if delivered is None else delivered
    history, seq = pending_dialog(delivered)
    finish_move(history, seq)
    open_dialog(history, delivered)
    return history


def begin_answer(history, choice=0):
    decision = history.decision
    selected = action(selection=decision.offers.dialog_panel.choices[choice].ref)
    history.emit("submitted", decision_seq=decision.seq, action=selected)
    history.emit("consumed", decision_seq=decision.seq, consumer="engine")
    history.emit("context", decision_seq=decision.seq, context=c.Context(), revision=history.revision+1)
    history.emit("issued", decision_seq=decision.seq, command_index=0)
    return decision.seq


def finish_answer(history, seq):
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    history.emit("result", decision_seq=seq, result=c.Result("dialog_answered", True))


class DialogContractTests(unittest.TestCase):
    def test_all_families_exact_wire_and_finite_support(self):
        self.assertEqual((c.DIALOGS_PROFILE.protocol, c.DIALOGS_PROFILE.version), (9, 7))
        self.assertTrue(c.DIALOGS_PROFILE.army_management)
        self.assertFalse(c.DIALOGS_PROFILE.army_transfer)
        for family in FAMILIES:
            with self.subTest(family=family):
                history = history_at(panel(family))
                for frame in history.frames:
                    self.assertEqual(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))), frame)
                support = history.decision.offers
                for choice in support.dialog_panel.choices:
                    self.assertTrue(support.permits(action(selection=choice.ref)))
                for selected in (action(selection="stale"), action(subject="other", selection=support.dialog_panel.choices[0].ref),
                                 action("close", None), action("back"), action("end_turn", None), action("move_to", x=1, y=1)):
                    self.assertFalse(support.permits(selected))
        # Insufficient own gold does not silently remove the paid-join accept button.
        obs = replace(observation(panel("neutral_paid_join")), resources=(0,)*7)
        self.assertTrue(c.Offers.from_observation(obs, schema=c.DIALOGS_ACTION_SCHEMA).permits(action(selection=obs.window.dialog_panel.choices[0].ref)))

    def test_versioned_window_null_and_legacy_bytes(self):
        legacy = map_observation("hero")
        raw = c.encode_observation(legacy, schema=c.MANAGEMENT_SCHEMA)
        self.assertNotIn("dialog_panel", raw["window"])
        current = c.encode_observation(legacy, schema=c.DIALOGS_SCHEMA)
        self.assertIsNone(current["window"]["dialog_panel"])
        self.assertEqual(c.decode_observation(current, schema=c.DIALOGS_SCHEMA), legacy)
        current["window"].pop("dialog_panel")
        with self.assertRaises(ValueError): c.decode_observation(current, schema=c.DIALOGS_SCHEMA)
        raw["window"]["dialog_panel"] = None
        with self.assertRaises(ValueError): c.decode_observation(raw, schema=c.MANAGEMENT_SCHEMA)
        for profile in (c.WINDOWS_PROFILE, c.TOWN_PROFILE, c.ARMIES_PROFILE, c.HEROES_PROFILE, c.MANAGEMENT_PROFILE):
            with self.assertRaises(ValueError): c.Action("choose_dialog", "hero", selection="choice", schema=profile.action_schema)
            with self.assertRaises(ValueError): c.Frame("old", 0, 0, 1, "event", event=c.VisibleEvent("dialog_requested", "hero", dialog=panel()), schema=profile.history_schema)

    def test_panel_rejects_hidden_fields_wrong_flags_counts_and_choices(self):
        raw = c._wire(panel())
        for key in ("query_id", "hidden_reward", "safe_to_autoaccept", "hero_army"):
            changed = deepcopy(raw); changed[key] = 7
            with self.assertRaises(ValueError): c.decode_dialog_panel(changed)
        for change in (dict(cancel=True), dict(choices=panel().choices[:1]),
                       dict(components=tuple(reversed(panel().components))), dict(selection=False)):
            with self.assertRaises(ValueError): replace(panel(), **change)
        for family in ("neutral_free_join", "neutral_flee"):
            delivered = panel(family)
            with self.assertRaises(ValueError): replace(delivered, text=delivered.text+(c.NoticeToken("replace_number", number=6),))
        paid = panel("neutral_paid_join")
        with self.assertRaises(ValueError): replace(paid, components=(c.DialogComponent("resource", "gold", 359),))
        with self.assertRaises(ValueError): replace(panel(), choices=(panel().choices[0], panel().choices[0]))
        with self.assertRaises(ValueError): replace(panel(), ref=panel().choices[0].ref)
        with self.assertRaises(FrozenInstanceError): panel().family = "arena"

    def test_callback_parent_result_then_context_order_is_required(self):
        history, seq = pending_dialog()
        with self.assertRaises(ValueError): open_dialog(history, panel())
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("movement_limit", True, (1,1,0),100,False))
        finish_move(history, seq)
        with self.assertRaises(ValueError): decide(history, observation())
        open_dialog(history, panel())
        with self.assertRaises(ValueError): history.emit("context_opened", context=history.context, revision=history.revision+1, reason="adventure_dialog")

    def test_payload_is_frozen_across_callback_context_and_decision(self):
        history, seq = pending_dialog(); finish_move(history, seq)
        history.emit("context_opened", context=observation(panel()).context, revision=history.revision+1, reason="adventure_dialog")
        changed = replace(panel(), components=(c.DialogComponent("resource","gold",1500), c.DialogComponent("experience",amount=1000)))
        before = history.digest()
        with self.assertRaises(ValueError): decide(history, observation(changed))
        self.assertEqual(history.digest(), before)
        decide(history, observation(panel()))

    def test_requested_dialog_requires_current_executing_owned_hero(self):
        for subject in ("other", "enemy"):
            history, _ = start_move()
            with self.assertRaises(ValueError): history.emit("event", event=c.VisibleEvent("dialog_requested", subject, dialog=panel()))
        obs = observation(panel())
        with self.assertRaises(ValueError): replace(obs, heroes=tuple(hero for hero in obs.heroes if hero.handle != "hero"))
        with self.assertRaises(ValueError): replace(obs.window, army=(c.Stack(0,0,25),))
        with self.assertRaises(ValueError): replace(obs.window, armies=map_observation("hero").window.armies)

    def test_chained_answers_get_new_contexts_and_old_choices_cannot_route(self):
        first = panel("neutral_free_join")
        history = history_at(first)
        seq = begin_answer(history, 1)
        second = panel("neutral_flee", "dialog-2")
        history.emit("event", event=c.VisibleEvent("dialog_requested", "hero", dialog=second))
        with self.assertRaises(ValueError): open_dialog(history, second)
        finish_answer(history, seq); open_dialog(history, second)
        self.assertFalse(history.decision.offers.permits(action(selection=first.choices[0].ref)))
        seq = begin_answer(history)
        with self.assertRaises(ValueError): history.emit("event", event=c.VisibleEvent("dialog_requested", "hero", dialog=first))
        finish_answer(history, seq); decide(history, observation())
        self.assertEqual([f.result.status for f in history.frames if f.result], ["interaction","dialog_answered","dialog_answered"])
        self.assertIsNone(history.terminal)
        self.assertTrue(all(t.reward is None and not t.eligible for t in history.transitions()))
        self.assertEqual(clone_history(history).digest(), history.digest())

    def test_reply_requires_map_return_one_native_ack_and_never_outcome_claim(self):
        history = history_at(); seq = history.decision.seq
        history.emit("submitted", decision_seq=seq, action=action(selection=panel().choices[0].ref))
        with self.assertRaises(ValueError): history.emit("consumed", decision_seq=seq, consumer="adapter")
        history.emit("consumed", decision_seq=seq, consumer="engine")
        with self.assertRaises(ValueError): history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision+1)
        history.emit("issued", decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("dialog_answered", True))
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=False)
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("dialog_answered", True))
        with self.assertRaises(ValueError): history.emit("result", decision_seq=seq, result=c.Result("rejected", False))
        history.emit("failure", reason="native_query_reply_rejected")
        self.assertEqual(history.boundary,"failure")
        self.assertTrue(all(t.reward is None and not t.eligible for t in history.transitions()))

    def test_strategic_dialog_cutoff_fails_before_prefix_freeze(self):
        history = history_at(); before = history.digest()
        with self.assertRaises(ValueError): history.emit("cutoff", decision_seq=history.decision.seq, reason="budget")
        self.assertEqual(history.digest(), before)
        producer = ContextualProducer(history.episode,0,interface=c.DIALOGS_ENVIRONMENT)
        producer.live = producer.history = history
        with self.assertRaises(ValueError): producer.freeze_cutoff("budget")
        self.assertIsNone(producer.cutoff)
        self.assertIsNone(history.boundary)
        self.assertIs(producer.history, history)

    def test_producer_roundtrips_query_without_transport_id_leak(self):
        source = history_at(panel("arena"))
        producer = ContextualProducer(source.episode,0,interface=c.DIALOGS_ENVIRONMENT)
        envelope = dict(protocol_version=9,episode=source.episode,player=0)
        producer.ingest(dict(envelope,type="ready",interface=c.DIALOGS_ENVIRONMENT,history_schema=c.DIALOGS_SCHEMA,action_schema=c.DIALOGS_ACTION_SCHEMA))
        for frame in source.frames:
            if frame.kind == "submitted":
                producer.submitted(frame.action); continue
            record = {key:value for key,value in c.encode_frame(frame).items() if key in {"kind","day"} | RECORD_FIELDS[frame.kind]}
            if frame.kind == "decision": record["decision_id"] = frame.seq+100
            elif frame.decision_seq is not None: record["decision_id"] = frame.decision_seq+100
            producer.ingest(dict(envelope,type="contextual",records=[record]))
        self.assertEqual(producer.live.frames,source.frames)
        response = producer.response(action(selection=panel("arena").choices[0].ref))
        self.assertEqual(response["decision_id"],source.decision.seq+100)
        self.assertNotIn("query_id",json.dumps(c.encode_frame(producer.live.decision)))


if __name__ == "__main__": unittest.main()
