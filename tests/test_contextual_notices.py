"""Informational-notice schema tests; no native run or template-semantics claim."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import unittest

from experiments.learner.contextual import (
    Action, Context, Frame, Hero, Notice, NoticeComponent, NoticeToken, Observation,
    Offers, PlayerHistory, Result, Stack, Tile, Town, VisibleEvent, Window,
    decode_frame, decode_notice, encode_frame,
)


def gold_notice(mode="info"):
    return Notice(mode,
                  (NoticeToken("append_text", text_id="core.advevent.113"),
                   NoticeToken("replace_text", text_id="core.restypes.6")),
                  (NoticeComponent("resource", resource="gold", amount=500),))


def notice_frame(notice=None, *, seq=0):
    return Frame("notice-test", 0, seq, 1, "event", event=VisibleEvent("notice", notice=notice or gold_notice()))


def observation(window=None):
    return Observation((20, 20, 1), (Tile((1, 1, 0), 2, 0, False, False),), (), (0,) * 7,
                       (Hero("own-hero", (1, 1, 0), 1000),), (Town("own-town", (1, 1, 0)),), window)


def decision(history, view=None):
    view = view or observation()
    return history.emit("decision", observation=view, offers=Offers.from_observation(view), revision=history.revision + 1)


class ContextualNoticeTests(unittest.TestCase):
    def test_verified_static_text_id_families(self):
        ids = ("core.advevent.113", "hero.core.orrin.name", "hero.core.edric.name",
               "heroes.baseline.all_opponents_defeated", "heroes.baseline.opponent_won",
               "heroes.baseline.seven_days_without_town", "heroes.baseline.opponent_lost_last_town",
               *(f"core.restypes.{n}" for n in range(7)), *(f"core.mineevnt.{n}" for n in range(7)),
               *(f"core.genrltxt.{n}" for n in (1, 5, 7, 8, 659)),
               *(f"core.arraytxt.{n}" for n in (*range(32, 42), *range(43, 58), *range(130, 136))),
               "creatures.core.pikeman.name.singular", "creatures.core.efreetSultan.name.plural")
        for text_id in ids:
            with self.subTest(text_id=text_id):
                self.assertEqual(NoticeToken("append_text", text_id=text_id).text_id, text_id)
                self.assertEqual(NoticeToken("replace_text", text_id=text_id).text_id, text_id)

    def test_static_ids_do_not_allow_raw_text_global_scopes_or_nearby_unused_ids(self):
        for text_id in ("core.advobtxt.113", "core.advevent.112", "core.restypes.7", "core.mineevnt.8",
                        "core.genrltxt.2", "core.arraytxt.42", "core.arraytxt.58", "core.arraytxt.129",
                        "core.arraytxt.136", "hero.core.catherine.name", "hero.hota.orrin.name",
                        "creature.core.pikeman.name.plural", "creatures.hota.pikeman.name.plural",
                        "creatures.core.pikeman.description", "creatures.core.9pikeman.name.plural",
                        "creatures.core.hidden/position.name.plural", "creatures.core.hidden.position.name.plural",
                        "creatures.core.hidden position.name.plural", "creatures.core.雪.name.plural",
                        "creatures.core." + "x" * 65 + ".name.plural", "You have won!", "a" * 64,
                        "heroes.baseline.hidden_enemy_dead", "core.restypes.6\n"):
            with self.subTest(text_id=text_id), self.assertRaises(ValueError):
                NoticeToken("append_text", text_id=text_id)

    def test_token_payload_is_discriminated_and_numbers_are_signed_int64(self):
        for op in ("append_number", "replace_number", "replace_positive_number"):
            for number in (-(2**63), -1, 0, 1, 2**63 - 1):
                self.assertEqual(NoticeToken(op, number=number).number, number)
            for number in (-(2**63) - 1, 2**63, True, 1.0, None):
                with self.subTest(op=op, number=number), self.assertRaises(ValueError):
                    NoticeToken(op, number=number)
            with self.assertRaises(ValueError):
                NoticeToken(op, text_id="core.restypes.0", number=2)
            with self.assertRaises(ValueError):
                NoticeToken(op, number=2, owner="self")
        for op in ("append_text", "replace_text"):
            for owner in ("self", "opponent"):
                self.assertEqual(NoticeToken(op, owner=owner).owner, owner)
            for payload in ({}, {"text_id": "core.restypes.0", "owner": "self"},
                            {"text_id": "core.restypes.0", "number": 5}, {"owner": "neutral"},
                            {"owner": "red"}, {"owner": 0}):
                with self.subTest(op=op, payload=payload), self.assertRaises(ValueError):
                    NoticeToken(op, **payload)
        self.assertEqual(NoticeToken("newline").op, "newline")
        for payload in ({"number": 0}, {"text_id": "core.restypes.0"}, {"owner": "self"}):
            with self.assertRaises(ValueError):
                NoticeToken("newline", **payload)
        with self.assertRaises(ValueError):
            NoticeToken("append_raw_string", text_id="core.restypes.0")

    def test_component_payload_is_discriminated_and_signed_int32(self):
        for kind in ("resource", "resource_per_day"):
            for resource in ("wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold"):
                for amount in (-(2**31), 0, 2**31 - 1):
                    self.assertEqual(NoticeComponent(kind, resource, amount).amount, amount)
            for payload in ({"resource": "gold"}, {"resource": "hidden", "amount": 1},
                            {"resource": 6, "amount": 1}, {"resource": "gold", "amount": True},
                            {"resource": "gold", "amount": 2**31},
                            {"resource": "gold", "amount": -(2**31) - 1},
                            {"resource": "gold", "amount": 1, "owner": "self"}):
                with self.subTest(kind=kind, payload=payload), self.assertRaises(ValueError):
                    NoticeComponent(kind, **payload)
        for owner in ("self", "opponent"):
            self.assertEqual(NoticeComponent("flag", owner=owner).owner, owner)
        for payload in ({}, {"owner": "neutral"}, {"owner": "blue"}, {"owner": 1},
                        {"owner": "self", "amount": 0}, {"owner": "self", "resource": "gold"}):
            with self.assertRaises(ValueError):
                NoticeComponent("flag", **payload)
        with self.assertRaises(ValueError):
            NoticeComponent("creature", amount=25)

    def test_immutable_typed_bounded_collections(self):
        notice = gold_notice()
        with self.assertRaises(FrozenInstanceError):
            notice.display_mode = "auto"
        with self.assertRaises(FrozenInstanceError):
            notice.text[0].text_id = "core.restypes.0"
        with self.assertRaises(FrozenInstanceError):
            notice.components[0].amount = 900
        for text, components in ((list(notice.text), notice.components),
                                 (notice.text, list(notice.components)),
                                 ((asdict(notice.text[0]),), notice.components),
                                 (notice.text, (asdict(notice.components[0]),))):
            with self.assertRaises(ValueError):
                Notice("info", text, components)
        Notice("auto", (NoticeToken("newline"),) * 64, (NoticeComponent("flag", owner="self"),) * 32)
        for text, components in (((NoticeToken("newline"),) * 65, ()),
                                 ((), (NoticeComponent("flag", owner="self"),) * 33)):
            with self.assertRaises(ValueError):
                Notice("info", text, components)
        with self.assertRaises(ValueError):
            Notice("query", (), ())

    def test_notice_is_exclusive_event_payload_and_preserves_existing_constructors(self):
        notice = gold_notice()
        self.assertEqual(VisibleEvent("notice", notice=notice).notice, notice)
        self.assertIsNone(VisibleEvent("visit", "object-1", 79, 0, "neutral").notice)
        for payload in ({"subject": "object-1"}, {"object_kind": 79}, {"object_subtype": 0},
                        {"owner": "self"}, {"side": "attacker"}, {"won": True}):
            with self.assertRaises(ValueError):
                VisibleEvent("notice", notice=notice, **payload)
        with self.assertRaises(ValueError):
            VisibleEvent("notice")
        with self.assertRaises(ValueError):
            VisibleEvent("notice", notice=asdict(notice))
        with self.assertRaises(ValueError):
            VisibleEvent("visit", notice=notice)

    def test_wire_roundtrip_and_no_shared_mutable_data(self):
        for mode in ("auto", "modal", "info"):
            frame = notice_frame(gold_notice(mode))
            wire = encode_frame(frame)
            self.assertEqual(decode_frame(wire), frame)
            self.assertEqual(decode_notice(wire["event"]["notice"]), frame.event.notice)
            wire["event"]["notice"]["components"][0]["amount"] = 1000
            self.assertEqual(frame.event.notice.components[0].amount, 500)
        frame = notice_frame(Notice("modal", (NoticeToken("append_text", owner="opponent"),
                                               NoticeToken("replace_positive_number", number=-5),
                                               NoticeToken("newline")), (NoticeComponent("flag", owner="self"),)))
        self.assertEqual(decode_frame(encode_frame(frame)), frame)

    def test_nested_hidden_raw_and_unused_pool_fields_rejected(self):
        for path, field, value in (
                (("event",), "engine_id", 99),
                (("event", "notice"), "exactStrings", ["hidden enemy is dead"]),
                (("event", "notice"), "localStrings", [[0, 659]]),
                (("event", "notice"), "stringsTextID", ["hidden.enemy.current.position"]),
                (("event", "notice"), "numbers", [19, 19]),
                (("event", "notice"), "text_hash", "a" * 64),
                (("event", "notice", "text", 0), "raw_text", "hidden enemy position"),
                (("event", "notice", "text", 0), "unused_replacements", ["opponent"]),
                (("event", "notice", "components", 0), "position", [19, 19, 0]),
                (("event", "notice", "components", 0), "global_player_id", 1)):
            wire = encode_frame(notice_frame())
            destination = wire
            for key in path:
                destination = destination[key]
            destination[field] = value
            with self.subTest(path=path, field=field), self.assertRaises(ValueError):
                decode_frame(wire)
        for raw in ({}, {"display_mode": "info", "text": [], "components": {}},
                    {"display_mode": "info", "text": (), "components": []},
                    {"display_mode": "info", "text": [{}], "components": []}):
            with self.assertRaises(ValueError):
                decode_notice(raw)

    def test_notice_during_consumed_move_does_not_acknowledge_or_reward(self):
        history = PlayerHistory("notice-test", 0)
        held = decision(history)
        history.emit("submitted", decision_seq=held.seq, action=Action("move_to", "own-hero", 2, 1))
        history.emit("consumed", decision_seq=held.seq, consumer="engine")
        history.emit("issued", decision_seq=held.seq, command_index=0)
        prefix = history.frames
        revision, context, pending = history.revision, history.context, history.pending_command
        history.emit("event", event=VisibleEvent("notice", notice=gold_notice("modal")))
        self.assertEqual(history.frames[:len(prefix)], prefix)
        self.assertEqual((history.revision, history.context, history.pending_command), (revision, context, pending))
        self.assertEqual(history.frames[-1].kind, "event")
        self.assertEqual(history.transitions(), ())
        self.assertIsNone(history.terminal)
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=held.seq, result=Result("interaction", True, (2, 1, 0), 900, True))
        history.emit("acknowledged", decision_seq=held.seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=held.seq, result=Result("interaction", True, (2, 1, 0), 900, True))
        self.assertIsNone(history.transitions()[0].reward)
        self.assertFalse(history.transitions()[0].eligible)

    def test_notice_after_supersession_preserves_window_and_final_views(self):
        history = PlayerHistory("notice-test", 0)
        held = decision(history)
        history.emit("submitted", decision_seq=held.seq, action=Action("open_hero", "own-hero"))
        history.emit("consumed", decision_seq=held.seq, consumer="adapter")
        hero_context = Context("hero", "own-hero")
        history.emit("context", decision_seq=held.seq, context=hero_context, revision=history.revision + 1)
        panel = Window(hero_context, (Stack(0, 0, 25),), primary_skills=(1, 1, 1, 1), mana=10, level=1)
        observed = history.emit("view", observation=observation(panel))
        history.emit("result", decision_seq=held.seq, result=Result("opened", True))
        held = decision(history, observation(panel))
        history.emit("superseded", decision_seq=held.seq, reason="terminal")
        prefix = history.frames
        revision = history.revision
        notice = Notice("modal", (NoticeToken("append_text", text_id="heroes.baseline.all_opponents_defeated"),), ())
        history.emit("event", event=VisibleEvent("notice", notice=notice))
        self.assertEqual(history.frames[:len(prefix)], prefix)
        self.assertEqual(history.context, hero_context)
        self.assertEqual(history.revision, revision)
        self.assertEqual(observed.observation.window.army[0].count, 25)
        self.assertIsNone(history.frames[-1].observation)
        self.assertIsNone(history.terminal)
        self.assertIsNone(history.transitions()[0].reward)
        history.emit("terminal", outcome="win")
        history.emit("event", event=VisibleEvent("notice", notice=gold_notice("auto")))
        history.emit("view", observation=observation(panel))
        self.assertIsNone(history.transitions()[0].reward)
        self.assertFalse(history.transitions()[0].eligible)

    def test_notice_text_is_not_a_query_action_or_engine_result(self):
        history = PlayerHistory("notice-test", 0)
        for mode in ("auto", "modal", "info"):
            history.emit("event", event=VisibleEvent("notice", notice=gold_notice(mode)))
        self.assertEqual([frame.kind for frame in history.frames], ["event"] * 3)
        self.assertIsNone(history.action)
        self.assertIsNone(history.decision)
        self.assertIsNone(history.pending_command)
        self.assertEqual(history.context, Context())
        self.assertEqual(history.transitions(), ())
        self.assertIsNone(history.terminal)
        self.assertIsNone(history.boundary)

    def test_schema_does_not_guess_native_template_operation_semantics(self):
        # Native producer must validate template/operation sequences. The learner
        # schema preserves their order without inventing a template interpreter.
        notice = Notice("modal", (NoticeToken("replace_number", number=0),
                                   NoticeToken("newline"), NoticeToken("append_text", owner="self")), ())
        self.assertEqual(decode_notice(json.loads(json.dumps(asdict(notice)))), notice)


if __name__ == "__main__":
    unittest.main()
