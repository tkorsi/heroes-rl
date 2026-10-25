"""Exact post-battle notice admission; these tests run no games or optimization."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner import contextual as c
from tests.test_contextual_dialogs import decide, observation, start_move


def necromancy(count=4, creature="skeleton"):
    tokens = (c.NoticeToken("append_text", text_id=f"core.genrltxt.{145 if count > 1 else 146}"),)
    if count > 1:
        tokens += (c.NoticeToken("replace_number", number=count),)
    tokens += (c.NoticeToken("replace_text", text_id=f"creatures.core.{creature}.name.{'plural' if count > 1 else 'singular'}"),)
    return c.Notice("auto", tokens, (c.NoticeComponent("creature", amount=count, content_id=f"core:{creature}"),))


def history_with_necromancy(count=4, creature="skeleton"):
    history, seq = start_move()
    history.emit("event", event=c.VisibleEvent("battle_start", side="attacker"))
    history.emit("event", event=c.VisibleEvent("notice", notice=necromancy(count, creature)))
    history.emit("event", event=c.VisibleEvent("battle_complete", won=True))
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    history.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1, 1, 0), 100, False))
    decide(history, observation())
    return history


class NecromancyNoticeTests(unittest.TestCase):
    def test_exact_singular_plural_and_core_creature_payloads(self):
        for count in (1, 2, 4, 2**31 - 1):
            for creature in ("skeleton", "skeletonWarrior", "lich"):
                with self.subTest(count=count, creature=creature):
                    notice = necromancy(count, creature)
                    self.assertEqual(c.decode_notice(c._wire(notice), schema=c.DIALOGS_SCHEMA), notice)

    def test_legacy_profiles_reject_new_templates_and_components(self):
        for count in (1, 4):
            notice = necromancy(count)
            for profile in (c.WINDOWS_PROFILE, c.TOWN_PROFILE, c.ARMIES_PROFILE,
                            c.HEROES_PROFILE, c.MANAGEMENT_PROFILE):
                with self.subTest(count=count, profile=profile.history_schema):
                    with self.assertRaises(ValueError):
                        c.decode_notice(c._wire(notice), schema=profile.history_schema)
                    with self.assertRaises(ValueError):
                        c.Frame("notice", 0, 0, 62, "event", event=c.VisibleEvent("notice", notice=notice),
                                schema=profile.history_schema)
        with self.assertRaises(ValueError):
            c.decode_notice(c._wire(necromancy()), schema=c.DIALOGS_ACTION_SCHEMA)

    def test_malformed_templates_count_or_content_mismatch_fail_closed(self):
        notice = necromancy()
        wrong = (
            replace(notice, display_mode="modal"), replace(notice, display_mode="info"),
            replace(notice, text=notice.text + (c.NoticeToken("newline"),)),
            replace(notice, text=notice.text[1:]), replace(notice, text=notice.text[:-1]),
            replace(notice, text=tuple(reversed(notice.text))),
            replace(notice, text=(notice.text[0], c.NoticeToken("replace_positive_number", number=4), notice.text[-1])),
            replace(notice, text=(notice.text[0], c.NoticeToken("replace_number", number=5), notice.text[-1])),
            replace(notice, text=necromancy(1).text),
            replace(notice, text=necromancy(4, "skeletonWarrior").text),
            replace(notice, text=(notice.text[0], notice.text[1], c.NoticeToken("replace_text", text_id="creatures.core.skeleton.name.singular"))),
            replace(notice, components=()), replace(notice, components=notice.components * 2),
            replace(notice, components=(c.NoticeComponent("resource", resource="gold", amount=4),)),
            replace(notice, components=(replace(notice.components[0], amount=1),)),
            replace(necromancy(1), text=necromancy(1).text + (c.NoticeToken("replace_number", number=1),)),
            replace(notice, text=(c.NoticeToken("append_text", text_id="core.genrltxt.30"),)),
        )
        for changed in wrong:
            with self.subTest(notice=changed), self.assertRaises(ValueError):
                c.decode_notice(c._wire(changed), schema=c.DIALOGS_SCHEMA)

    def test_creature_component_requires_only_static_core_id_and_positive_int32(self):
        for amount in (None, True, 0, -1, 1.0, 2**31):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                c.NoticeComponent("creature", amount=amount, content_id="core:skeleton")
        for content_id in (None, "skeleton", "hota:skeleton", "core:0skeleton", "core:hidden.position", "core:skeleton\n"):
            with self.subTest(content_id=content_id), self.assertRaises(ValueError):
                c.NoticeComponent("creature", amount=4, content_id=content_id)
        for field, value in (("resource", "gold"), ("owner", "self"), ("skill", "attack")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.NoticeComponent("creature", amount=4, content_id="core:skeleton", **{field: value})

    def test_raw_payload_does_not_accept_actions_hidden_fields_or_unknown_text(self):
        original = c._wire(necromancy())
        for field, value in (("native_id", 56), ("hero", "Thant"), ("reward", 1), ("choices", ["accept"])):
            raw = deepcopy(original)
            raw["components"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.decode_notice(raw, schema=c.DIALOGS_SCHEMA)
        for text_id in ("core.genrltxt.144", "core.genrltxt.149", "creatures.hota.skeleton.name.plural"):
            raw = deepcopy(original)
            raw["text"][0]["text_id"] = text_id
            with self.subTest(text_id=text_id), self.assertRaises(ValueError):
                c.decode_notice(raw, schema=c.DIALOGS_SCHEMA)

    def test_typed_history_continues_without_action_query_or_reward(self):
        history = history_with_necromancy()
        restored = c.PlayerHistory(history.episode, history.player, schema=c.DIALOGS_SCHEMA)
        for frame in history.frames:
            restored.append(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))))
        self.assertEqual(restored.digest(), history.digest())
        self.assertIsNone(restored.pending_dialog)
        self.assertIsNone(restored.active_dialog)
        self.assertEqual(restored.context, c.Context())
        self.assertEqual(sum(frame.kind == "submitted" for frame in restored.frames), 1)
        transitions = restored.transitions()
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action.kind, "move_to")
        self.assertIsNone(transitions[0].reward)
        self.assertFalse(transitions[0].eligible)
        self.assertEqual(restored.decision.offers.schema, c.DIALOGS_ACTION_SCHEMA)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch is not installed")
class NecromancyModelTests(unittest.TestCase):
    def test_existing_v7_checkpoint_consumes_notice_with_unchanged_weights_and_shapes(self):
        import torch
        from experiments.learner import contextual_model as m
        from experiments.learner.checkpoints import load_policy_checkpoint, save_policy_checkpoint
        from tests.test_contextual_management_model import last_output

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(212)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
            identity = m.weights_sha256(model)
            shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "existing-v7.pt"
                save_policy_checkpoint(path, model, checkpoint="synthetic-v7",
                                       source_metadata={"purpose": "notice_compatibility_test"}, interface=c.DIALOGS_ENVIRONMENT)
                restored = load_policy_checkpoint(path, interface=c.DIALOGS_ENVIRONMENT).model
            with torch.no_grad():
                original = last_output(restored, history_with_necromancy().frames)
                changed_count = last_output(restored, history_with_necromancy(5).frames)
                changed_creature = last_output(restored, history_with_necromancy(4, "lich").frames)
            self.assertTrue(torch.isfinite(original.state).all())
            self.assertTrue(torch.isfinite(original.decision.value))
            self.assertFalse(torch.equal(original.state, changed_count.state))
            self.assertFalse(torch.equal(original.state, changed_creature.state))
            self.assertEqual(shapes, {key: tuple(value.shape) for key, value in restored.state_dict().items()})
            self.assertEqual(m.weights_sha256(restored), identity)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
