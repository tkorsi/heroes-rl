"""Strict v8 capability contracts; no gameplay or training is launched here."""
from dataclasses import replace
from dataclasses import fields
from copy import deepcopy
import json
import unittest

from experiments.learner import contextual as c
from tests.test_contextual_management import observation as base_observation


def choice(kind, *, ref="choice-1", subject="town", **values):
    data = {field.name: None for field in fields(c.CapabilityChoice)}
    data.update(kind=kind, ref=ref, subject=subject, **values)
    return c.CapabilityChoice(**data)


def observation(kind="map"):
    obs = base_observation(kind)
    if obs.window is not None:
        controls = ("marketplace", "purchases") if kind == "town" else ("artifacts", "spellbook")
        obs = replace(obs, window=replace(obs.window, capability_controls=controls))
    return obs


def decide(history, obs):
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA), revision=history.revision + 1)


def action(kind, subject=None, **kwargs):
    return c.Action(kind, subject, schema=c.EXPANDED_ACTION_SCHEMA, **kwargs)


def begin(history, selected, consumer="adapter"):
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=selected)
    history.emit("consumed", decision_seq=seq, consumer=consumer)
    return seq


def navigate(history, selected, after, status="opened"):
    seq = begin(history, selected)
    history.emit("context", decision_seq=seq, context=after.context, revision=history.revision + 1)
    history.emit("view", observation=after)
    history.emit("result", decision_seq=seq, result=c.Result(status, True))
    return decide(history, after)


def market_history():
    history = c.PlayerHistory("synthetic-capabilities", 0, schema=c.EXPANDED_SCHEMA)
    decide(history, observation())
    navigate(history, action("open_town", "town"), observation("town"))
    trade = choice("trade_resource", resource_from=6, resource_to=0, give=250, receive=1, max_count=4)
    panel = c.CapabilityPanel("market-1", "marketplace", c.Context("town", "town", "overview"), False, None, None, (), (trade,))
    obs = replace(observation(), window=c.Window(c.Context("capability", "town", "marketplace"), None,
                      armies=(), capability_controls=(), capability_panel=panel))
    navigate(history, action("open_capability", "town", panel="marketplace"), obs)
    return history


class ExpandedContractTests(unittest.TestCase):
    def test_market_navigation_lots_and_exact_native_ack(self):
        history = market_history()
        support = history.decision.offers
        for count in range(1, 5):
            self.assertTrue(support.permits(action("trade_resource", "town", selection="choice-1", count=count)))
        self.assertFalse(support.permits(action("trade_resource", "town", selection="choice-1", count=5)))
        self.assertFalse(support.permits(action("trade_resource", "town", selection="stale", count=1)))
        seq = begin(history, action("trade_resource", "town", selection="choice-1", count=1), "engine")
        history.emit("issued", decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
        restored = c.PlayerHistory(history.episode, history.player, schema=c.EXPANDED_SCHEMA)
        for frame in history.frames:
            restored.append(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))))
        self.assertEqual(restored.digest(), history.digest())

    def test_capabilities_are_absent_before_window_opening(self):
        history = c.PlayerHistory("hidden", 0, schema=c.EXPANDED_SCHEMA)
        frame = decide(history, observation())
        self.assertFalse(frame.offers.permits(action("open_capability", "town", panel="marketplace")))
        raw = c.encode_observation(observation("town"), schema=c.EXPANDED_SCHEMA)
        self.assertIsNone(raw["window"]["capability_panel"])
        self.assertEqual(c.decode_observation(raw, schema=c.EXPANDED_SCHEMA), observation("town"))
        for key in ("capability_controls", "capability_panel"):
            changed = deepcopy(raw); changed["window"].pop(key)
            with self.assertRaises(ValueError):
                c.decode_observation(changed, schema=c.EXPANDED_SCHEMA)

    def test_legacy_schema_rejects_new_actions_and_data(self):
        with self.assertRaises(ValueError):
            c.Action("open_capability", "town", schema=c.DIALOGS_ACTION_SCHEMA, panel="marketplace")
        with self.assertRaises(ValueError):
            c.encode_observation(observation("town"), schema=c.DIALOGS_SCHEMA)
        raw = c._wire(market_history().decision.observation.window.capability_panel)
        for key in ("native_id", "utility", "distance", "hidden_inventory"):
            changed = deepcopy(raw); changed[key] = 1
            with self.assertRaises(ValueError): c.decode_capability_panel(changed)

    def test_level_up_waits_for_native_parent_ack_and_requires_real_choice(self):
        history = c.PlayerHistory("level-up", 0, schema=c.EXPANDED_SCHEMA)
        decide(history, observation())
        seq = begin(history, action("move_to", "hero", x=2, y=1), "engine")
        history.emit("issued", decision_seq=seq, command_index=0)
        panel = c.CapabilityPanel("level-1", "level_up", c.Context(), True, 2, 0, (), (
            choice("choose_level_up", ref="skill-a", subject="hero", skill=1, skill_level=1),
            choice("choose_level_up", ref="skill-b", subject="hero", skill=2, skill_level=1)))
        history.emit("event", event=c.VisibleEvent("capability_requested", "hero", capability=panel))
        context = c.Context("capability", "hero", "level_up", panel.ref)
        with self.assertRaises(ValueError):
            history.emit("context_opened", context=context, revision=history.revision + 1, reason="capability_dialog")
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1,1,0), 100, False))
        history.emit("context_opened", context=context, revision=history.revision + 1, reason="capability_dialog")
        obs = replace(observation(), window=c.Window(context, None, armies=(), capability_controls=(), capability_panel=panel))
        decision = decide(history, obs)
        self.assertTrue(history.requires_choice)
        with self.assertRaises(ValueError):
            history.emit("cutoff", decision_seq=decision.seq, reason="agent_action_limit")
        self.assertFalse(decision.offers.permits(action("close_capability", "hero")))
        seq = begin(history, action("choose_level_up", "hero", selection="skill-a"), "engine")
        history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision + 1)
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
        self.assertFalse(history.requires_choice)


class ExpandedModelTests(unittest.TestCase):
    def test_finite_trade_distribution_has_exact_support_and_gradients(self):
        import torch
        from experiments.learner import contextual_model as m
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
            history = market_history()
            state = model.initial_state()
            for frame in m.encode_history(history.frames):
                output = model.forward_step(frame, state); state = output.state
            distribution = output.decision.distribution
            actions = [action("close_capability", "town"), *(action("trade_resource", "town", selection="choice-1", count=count) for count in range(1, 5))]
            log_probs = torch.stack([distribution.log_prob(item) for item in actions])
            self.assertAlmostEqual(float(log_probs.exp().sum().detach()), 1.0, places=6)
            (-log_probs[1] + output.decision.value).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            encoded = repr(m._window_semantic(history.decision.observation))
            self.assertNotIn("choice-1", encoded)
            self.assertNotIn("market-1", encoded)
        finally:
            torch.set_num_threads(threads)


def eagle_eye(spells=("magicArrow", "haste", "slow")):
    text = [c.NoticeToken("append_text", text_id="core.genrltxt.221"),
            c.NoticeToken("replace_text", text_id="hero.core.sanya.name")]
    for index, spell in enumerate(spells):
        if index:
            text.append(c.NoticeToken("append_text", text_id="core.genrltxt.141"
                if index == len(spells) - 1 else "heroes.notice.comma_space"))
        text.append(c.NoticeToken("replace_text" if index == 0 else "append_text",
                                 text_id=f"spell.core.{spell}.name"))
    text.append(c.NoticeToken("append_text", text_id="heroes.notice.period"))
    return c.Notice("modal", tuple(text), tuple(c.NoticeComponent("spell", content_id=f"core:{spell}") for spell in spells))


class ExpandedNoticeTests(unittest.TestCase):
    def test_eagle_eye_single_and_ordered_lists(self):
        for spells in (("magicArrow",), ("magicArrow", "haste"), ("magicArrow", "haste", "slow")):
            notice = eagle_eye(spells)
            self.assertEqual(c.decode_notice(c._wire(notice), schema=c.EXPANDED_SCHEMA), notice)

    def test_stables_is_information_without_an_action_or_reward(self):
        notice = c.Notice("modal", (c.NoticeToken("append_text", text_id="core.genrltxt.580"),), ())
        self.assertEqual(c.decode_notice(c._wire(notice), schema=c.EXPANDED_SCHEMA), notice)
        for changed in (replace(notice, display_mode="auto"),
                        replace(notice, components=(c.NoticeComponent("resource", resource="gold", amount=400),))):
            with self.assertRaises(ValueError):
                c.decode_notice(c._wire(changed), schema=c.EXPANDED_SCHEMA)

    def test_old_profiles_do_not_silently_expand_their_contract(self):
        for schema in (c.SCHEMA, c.TOWN_SCHEMA, c.ARMIES_SCHEMA, c.HEROES_SCHEMA,
                       c.MANAGEMENT_SCHEMA, c.DIALOGS_SCHEMA):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                c.decode_notice(c._wire(eagle_eye()), schema=schema)

    def test_spell_components_and_punctuation_must_match_delivered_text(self):
        notice = eagle_eye()
        malformed = (replace(notice, components=tuple(reversed(notice.components))),
                     replace(notice, components=notice.components[:-1]),
                     replace(notice, text=notice.text[:-1]),
                     replace(notice, text=notice.text + (c.NoticeToken("newline"),)),
                     replace(notice, display_mode="auto"),
                     replace(notice, text=(notice.text[0], *notice.text[2:])),
                     eagle_eye(("haste", "haste")))
        for changed in malformed:
            with self.subTest(notice=changed), self.assertRaises(ValueError):
                c.decode_notice(c._wire(changed), schema=c.EXPANDED_SCHEMA)


if __name__ == "__main__":
    unittest.main()
