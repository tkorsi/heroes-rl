"""Synthetic recruitment schema/lifecycle tests, never native game evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest

from experiments.learner import contextual as c
from test_contextual_town import action, observation, decide, submit, navigate, building_history, build
from contextual_protocol import ContextualProducer, RECORD_FIELDS


def panel(*, selected="creature-0", stock=10, maximum=10, role="town_garrison"):
    return c.RecruitmentPanel("dwelling-0", (c.RecruitChoice("creature-0", 0), c.RecruitChoice("creature-1", 1)),
        selected, stock, (0, 0, 0, 0, 0, 0, 60 if selected == "creature-0" else 75), maximum,
        c.RecruitDestination("army-0", role))


def recruit_observation(context, *, payload=None, resources=None):
    if context.panel == "overview":
        window = c.Window(context, (), faction=0, town_panel=c.TownPanel("overview", (
            c.PanelControl("hall", "panel-0"), c.PanelControl("recruitment", "dwelling-0"))))
    else:
        window = c.Window(context, None, town_panel=c.TownPanel("recruitment", recruitment=payload or panel()))
    kwargs = {"window": window}
    if resources is not None: kwargs["resources"] = resources
    return replace(observation(), **kwargs)


def history_at_recruitment(player=0, *, payload=None):
    history = c.PlayerHistory("recruit-test", player, schema=c.TOWN_SCHEMA)
    decide(history)
    overview = c.Context("town", "town", "overview")
    navigate(history, action("open_town", subject="town"), overview, obs=recruit_observation(overview))
    context = c.Context("town", "town", "recruitment", "dwelling-0")
    navigate(history, action("open_panel", subject="town", panel="recruitment", selection="dwelling-0"), context,
             obs=recruit_observation(context, payload=payload))
    return history


def purchase(**kwargs):
    return action("recruit", subject="town", selection=kwargs.pop("selection", "creature-0"),
                  count=kwargs.pop("count", 2), destination=kwargs.pop("destination", "army-0"), **kwargs)


def consume(history, *, native=True, selected=None):
    seq = history.decision.seq
    history.emit("submitted", action=selected or purchase(), decision_seq=seq)
    history.emit("consumed", consumer="engine" if native else "adapter", decision_seq=seq)
    return seq


def notice():
    return c.Notice("modal", (c.NoticeToken("append_text", text_id="core.genrltxt.17"),), ())


def no_capacity(history):
    seq = consume(history, native=False)
    history.emit("event", event=c.VisibleEvent("notice", notice=notice()))
    history.emit("result", decision_seq=seq, result=c.Result("no_capacity", False))
    decide(history, recruit_observation(history.context))


def purchase_result(history, accepted=True):
    seq = consume(history)
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("context", context=c.Context("town", "town", "overview"), revision=history.revision + 1, decision_seq=seq)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=accepted)
    obs = recruit_observation(history.context, resources=(1, 2, 3, 4, 5, 6, 6880 if accepted else 7000))
    history.emit("view", observation=obs)
    history.emit("result", decision_seq=seq, result=c.Result("recruited" if accepted else "rejected", accepted))
    decide(history, obs)


class RecruitmentTests(unittest.TestCase):
    def test_pre_recruitment_v3_canonical_history_is_unchanged(self):
        history = building_history(); build(history)
        # Captured at 1beba77, before TownPanel gained recruitment.
        self.assertEqual(history.digest(), "17166f61fdc0d57536ca2ae581d52cbe20c0ac437120ebc69b46b98294e05d91")
        self.assertNotIn("recruitment", c.encode_frame(history.decision)["observation"]["window"]["town_panel"])

    def test_recruitment_payload_is_immutable_and_exclusive(self):
        value = panel()
        with self.assertRaises(FrozenInstanceError): value.stock = 100
        for invalid in (lambda: replace(value, offers=[]), lambda: replace(value, selected_offer_ref="foreign-card"),
                        lambda: replace(value, slider_max=11), lambda: replace(value, stock=True),
                        lambda: replace(value, unit_cost=(-1,) * 7),
                        lambda: c.RecruitDestination("hero-army", "visiting_hero"),
                        lambda: c.TownPanel("overview", recruitment=value),
                        lambda: c.Window(c.Context("town", "town", "recruitment", "dwelling-0"), (),
                                         town_panel=c.TownPanel("recruitment", recruitment=value))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): invalid()
        self.assertEqual(panel(role="garrison_hero").destination.role, "garrison_hero")

    def test_counts_and_destination_are_explicit_and_offered(self):
        history = history_at_recruitment()
        before = history.digest()
        for count in (0, -1, True, 1.5, 2**31):
            with self.subTest(count=count), self.assertRaises(ValueError): purchase(count=count)
        for selected in (purchase(count=11), purchase(selection="creature-1"), purchase(destination="foreign-army"),
                         action("open_panel", subject="town", panel="recruitment", selection="hidden-dwelling")):
            self.assertFalse(history.decision.offers.permits(selected))
            with self.assertRaises(ValueError): history.emit("submitted", action=selected, decision_seq=history.decision.seq)
            self.assertEqual(history.digest(), before)
        self.assertTrue(history.decision.offers.permits(purchase(count=10)))
        depleted = history_at_recruitment(payload=panel(stock=0, maximum=0))
        self.assertFalse(depleted.decision.offers.permits(purchase(count=1)))
        self.assertTrue(depleted.decision.offers.permits(action("select_recruit", subject="town", selection="creature-1")))

    def test_slider_matches_exact_visible_affordability_without_capacity_masking(self):
        context = c.Context("town", "town", "recruitment", "dwelling-0")
        for maximum in (0, 1, 9):
            with self.subTest(narrowed=maximum), self.assertRaises(ValueError):
                recruit_observation(context, payload=panel(maximum=maximum))
        resources = (1, 2, 3, 4, 5, 6, 119)
        obs = recruit_observation(context, payload=panel(maximum=1), resources=resources)
        self.assertEqual(obs.window.town_panel.recruitment.slider_max, 1)
        with self.assertRaises(ValueError): recruit_observation(context, payload=panel(), resources=resources)
        multiple = replace(panel(maximum=2), unit_cost=(3, 0, 5, 0, 0, 0, 60))
        recruit_observation(context, payload=multiple, resources=(7, 0, 24, 0, 0, 0, 7000))
        free = replace(panel(stock=2**31 - 1, maximum=2147483645), unit_cost=(0,) * 7)
        recruit_observation(context, payload=free, resources=(0,) * 7)
        with self.assertRaises(ValueError):
            recruit_observation(context, payload=replace(free, slider_max=2**31 - 1), resources=(0,) * 7)

    def test_selection_is_an_explicit_adapter_transition(self):
        history = history_at_recruitment()
        selected = action("select_recruit", subject="town", selection="creature-1")
        seq = submit(history, selected)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("selected", True), decision_seq=seq)
        history.emit("context", context=history.context, revision=history.revision + 1, decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("selected", True), decision_seq=seq)
        obs = recruit_observation(history.context, payload=panel(selected="creature-1"))
        history.emit("view", observation=obs)
        history.emit("result", result=c.Result("selected", True), decision_seq=seq)
        decide(history, obs)
        self.assertFalse(history.decision.offers.permits(purchase()))
        self.assertTrue(history.decision.offers.permits(purchase(selection="creature-1")))
        self.assertFalse(any(frame.kind in ("issued", "acknowledged") for frame in history.frames))

    def test_rejected_selection_cannot_change_card_or_context_revision(self):
        for change_context, change_card in ((True, False), (False, True), (True, True)):
            history = history_at_recruitment()
            seq = submit(history, action("select_recruit", subject="town", selection="creature-1"))
            if change_context:
                history.emit("context", context=history.context, revision=history.revision + 1, decision_seq=seq)
            if change_card:
                history.emit("view", observation=recruit_observation(history.context, payload=panel(selected="creature-1")))
            with self.subTest(context=change_context, card=change_card), self.assertRaises(ValueError):
                history.emit("result", result=c.Result("rejected", False), decision_seq=seq)
        history = history_at_recruitment()
        seq = submit(history, action("select_recruit", subject="town", selection="creature-1"))
        history.emit("result", result=c.Result("rejected", False), decision_seq=seq)
        self.assertEqual(history.context.panel, "recruitment")

    def test_success_and_server_rejection_bind_one_command_and_overview(self):
        for accepted in (True, False):
            history = history_at_recruitment()
            prefix = tuple(c.encode_frame(frame) for frame in history.frames)
            purchase_result(history, accepted)
            self.assertEqual(tuple(c.encode_frame(frame) for frame in history.frames[:len(prefix)]), prefix)
            self.assertEqual(history.context.panel, "overview")
            self.assertEqual(sum(frame.kind == "issued" for frame in history.frames), 1)
            restored = c.PlayerHistory(history.episode, 0, schema=c.TOWN_SCHEMA)
            for frame in history.frames: restored.append(c.decode_frame(c.encode_frame(frame)))
            self.assertEqual(restored.digest(), history.digest())
            self.assertTrue(all(item.reward is None for item in history.transitions()))

    def test_no_capacity_requires_the_actual_notice_and_no_native_purchase(self):
        history = history_at_recruitment()
        seq = consume(history, native=False)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("no_capacity", False), decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("issued", command_index=0, decision_seq=seq)
        history.emit("event", event=c.VisibleEvent("notice", notice=notice()))
        with self.assertRaises(ValueError): history.emit("issued", command_index=0, decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("event", event=c.VisibleEvent("notice", notice=notice()))
        history.emit("result", result=c.Result("no_capacity", False), decision_seq=seq)
        self.assertEqual(history.context.panel, "recruitment")

    def test_capacity_notice_is_v3_only_and_scoped_to_consumed_attempt(self):
        event = c.VisibleEvent("notice", notice=notice())
        with self.assertRaises(ValueError): c.Frame("v2", 0, 0, 1, "event", event=event)
        wire = c.encode_frame(c.Frame("v3", 0, 0, 1, "event", event=event, schema=c.TOWN_SCHEMA))
        with self.assertRaises(ValueError): c.decode_notice(wire["event"]["notice"])
        self.assertEqual(c.decode_notice(wire["event"]["notice"], schema=c.TOWN_SCHEMA), notice())
        with self.assertRaises(ValueError): c.PlayerHistory("v3", 0, schema=c.TOWN_SCHEMA).append(c.decode_frame(wire))
        for changed in (replace(notice(), display_mode="info"), replace(notice(), components=(c.NoticeComponent("flag", owner="self"),)),
                        replace(notice(), text=notice().text + (c.NoticeToken("newline"),))):
            with self.assertRaises(ValueError):
                c.Frame("v3", 0, 0, 1, "event", event=c.VisibleEvent("notice", notice=changed), schema=c.TOWN_SCHEMA)
        history = history_at_recruitment(); consume(history)
        with self.assertRaises(ValueError): history.emit("event", event=event)

    def test_no_capacity_cannot_hide_changed_resources_or_stock(self):
        for changed in (lambda obs: replace(obs, resources=(1, 2, 3, 4, 5, 6, 6999)),
                        lambda obs: recruit_observation(obs.context, payload=panel(stock=9, maximum=9))):
            history = history_at_recruitment(); seq = consume(history, native=False)
            history.emit("event", event=c.VisibleEvent("notice", notice=notice()))
            history.emit("view", observation=changed(history.decision.observation))
            with self.assertRaises(ValueError): history.emit("result", result=c.Result("no_capacity", False), decision_seq=seq)

    def test_recruitment_cannot_succeed_without_accepted_ack_or_correct_context(self):
        history = history_at_recruitment(); seq = consume(history)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("recruited", True), decision_seq=seq)
        history.emit("issued", command_index=0, decision_seq=seq)
        history.emit("acknowledged", command_index=0, accepted=False, decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("recruited", True), decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("issued", command_index=1, decision_seq=seq)

    def test_hidden_or_inactive_recruitment_fields_are_rejected(self):
        history = history_at_recruitment()
        for mutate in (lambda raw: raw["window"]["town_panel"]["recruitment"].update(hidden_hero_id=4),
                       lambda raw: raw["window"]["town_panel"]["recruitment"]["offers"][0].update(unit_cost=[1] * 7),
                       lambda raw: raw["window"]["town_panel"]["recruitment"]["destination"].update(engine_id=2)):
            raw = c.encode_observation(history.decision.observation, schema=c.TOWN_SCHEMA); mutate(raw)
            with self.assertRaises(ValueError): c.decode_observation(raw, schema=c.TOWN_SCHEMA)
        raw = c.encode_observation(recruit_observation(c.Context("town", "town", "overview")), schema=c.TOWN_SCHEMA)
        raw["window"]["town_panel"]["recruitment"] = None
        with self.assertRaises(ValueError): c.decode_observation(raw, schema=c.TOWN_SCHEMA)

    def test_native_transport_reconstructs_complete_purchase_and_rejection_history(self):
        for finish in (purchase_result, no_capacity):
            source = history_at_recruitment(); finish(source)
            producer = ContextualProducer(source.episode, 0, interface=c.TOWN_ENVIRONMENT)
            producer.ingest(dict(protocol_version=5, type="ready", episode=source.episode, player=0,
                                 interface=c.TOWN_ENVIRONMENT, history_schema=c.TOWN_SCHEMA, action_schema=c.TOWN_ACTION_SCHEMA))
            for frame in source.frames:
                if frame.kind == "submitted":
                    producer.submitted(frame.action)
                    continue
                raw = c.encode_frame(frame)
                record = dict(kind=frame.kind, day=frame.day)
                for key in RECORD_FIELDS[frame.kind]:
                    if key == "decision_id": record[key] = frame.seq if frame.kind == "decision" else frame.decision_seq
                    else: record[key] = raw[key]
                producer.ingest(dict(protocol_version=5, type="contextual", episode=source.episode, player=0, records=[record]))
            self.assertEqual(producer.history.digest(), source.digest())

    def test_paired_cutoff_keeps_zero_rewards_and_unwinds_to_overview(self):
        pair = tuple(history_at_recruitment(player) for player in (0, 1))
        for history in pair:
            no_capacity(history)
            history.emit("cutoff", decision_seq=history.decision.seq, reason="synthetic-budget")
        c.finalize_cutoff(pair, evidence_ref="synthetic-not-a-game", histories_complete=(True, True), failure_free=True,
                          shutdown_verified=True, continuation_limit=8, continuation_steps=(3, 0))
        self.assertTrue(all(t.reward == 0 for history in pair for t in history.transitions()))
        history = history_at_recruitment()
        navigate(history, action("back", subject="town"), c.Context("town", "town", "overview"), status="closed",
                 obs=recruit_observation(c.Context("town", "town", "overview")))
        self.assertIsNone(history.decision.observation.window.town_panel.recruitment)


if __name__ == "__main__":
    unittest.main()
