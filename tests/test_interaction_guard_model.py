"""Synthetic guarded probability/reconstruction checks; no games or PPO."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import interaction_guard as g
from tests.test_contextual_model import observation, map_history, town_history
from tests.test_contextual_management import history_at
from tests.test_contextual_management_model import all_actions
from tests.test_contextual_capabilities import market_history, observation as expanded_observation, action, decide, navigate
from tests.test_interaction_guard import artifacts


class InteractionGuardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(182)

    def distribution(self, frame, guard, model=None):
        model = model or m.ContextualActorCritic(m.ModelConfig(
            schema=m.model_schema_for(frame.schema), hidden_size=8, feature_size=4))
        encoded = m.ContextualInput(frame, m._node("frame"), frame.observation, guard)
        return model.forward_step(encoded, model.initial_state()).decision.distribution

    def assert_normalizes(self, distribution, actions):
        probability = sum(distribution.log_prob(selected).exp().item() for selected in actions)
        self.assertAlmostEqual(probability, 1., places=5)
        generator = torch.Generator().manual_seed(291)
        for _ in range(20):
            selected, evaluated = distribution.sample_evaluated(generator)
            self.assertIn(selected, actions)
            self.assertTrue(torch.equal(evaluated.log_prob, distribution.log_prob(selected)))

    def test_map_subject_coordinate_and_kind_filter_before_normalization(self):
        obs = observation()
        obs = replace(obs, heroes=(*obs.heroes, c.Hero("spent", (2, 1, 0), 0)))
        frame = map_history(obs).decision
        guard = g.Tracker(g.VERSION).observe(frame)
        guard = replace(guard, blocked_navigation=(("open_hero", "hero"),),
                        blocked_coordinates=tuple(sorted((*guard.blocked_coordinates, ("hero", 1, 0)))))
        policy = self.distribution(frame, guard)
        permitted = [c.Action("end_turn"), c.Action("open_hero", "spent"), c.Action("open_town", "town")]
        permitted += [c.Action("move_to", "hero", x, y) for y in range(2) for x in range(3)
                      if ("hero", x, y) not in guard.blocked_coordinates]
        self.assert_normalizes(policy, permitted)
        for selected in (c.Action("move_to", "spent", 1, 0), c.Action("move_to", "hero", 0, 0),
                         c.Action("move_to", "hero", 1, 0), c.Action("open_hero", "hero")):
            self.assertTrue(frame.offers.permits(selected))
            with self.assertRaises(ValueError):
                policy.log_prob(selected)
        trace = policy.trace(c.Action("move_to", "hero", 2, 1))
        self.assertEqual([(part.name, part.support_size) for part in trace],
                         [("kind", 4), ("subject", 1), ("coordinate", 4)])
        self.assertEqual(policy.support.guard, guard)
        self.assertEqual(policy.support.distribution_schema, m.DISTRIBUTION_SCHEMA + "+" + g.VERSION)

    def test_no_remaining_movement_removes_kind_and_keeps_escape(self):
        obs = replace(observation(), size=(1, 1, 1), towns=())
        frame = map_history(obs).decision
        guard = g.Tracker(g.VERSION).observe(frame)
        guard = replace(guard, blocked_navigation=(("open_hero", "hero"),))
        policy = self.distribution(frame, guard)
        self.assertEqual(policy._kinds(), ("end_turn",))
        self.assertEqual(policy.sample(), c.Action("end_turn"))
        self.assertEqual(policy.log_prob(c.Action("end_turn")).item(), 0.)

    def test_navigation_controls_filter_individual_semantic_entries(self):
        panel = c.TownPanel("overview", controls=(c.PanelControl("hall", "hall"),
            c.PanelControl("recruitment", "first"), c.PanelControl("recruitment", "second")))
        frame = town_history(panel).decision
        selected = c.Action("open_panel", "town", schema=c.TOWN_ACTION_SCHEMA,
                            panel="recruitment", selection="dwelling")
        guard = g.GuardSupport(g.VERSION, blocked_navigation=(g.nav_key(selected, frame.observation, frame.offers),))
        policy = self.distribution(frame, guard)
        actions = [c.Action("close", schema=c.TOWN_ACTION_SCHEMA),
            c.Action("open_panel", "town", schema=c.TOWN_ACTION_SCHEMA, panel="hall"),
            replace(selected, selection="dwelling2")]
        self.assert_normalizes(policy, actions)
        self.assertEqual(policy.trace(actions[-1])[-1].support_size, 2)
        with self.assertRaises(ValueError): policy.evaluate(selected)

    def test_management_filters_both_actor_pairs_without_count_holes(self):
        frame = history_at().decision
        guard = g.GuardSupport(g.VERSION, blocked_management_subjects=("hero",))
        policy = self.distribution(frame, guard)
        native = all_actions(frame)
        allowed = [selected for selected in native if guard.permits(selected, frame.observation, frame.offers)]
        self.assertTrue(any(selected.kind == "split_stack" for selected in allowed))
        self.assertTrue(any(selected.kind == "split_stack" and selected not in allowed for selected in native))
        self.assert_normalizes(policy, allowed)
        for selected in native:
            if selected not in allowed:
                with self.assertRaises(ValueError): policy.evaluate(selected)
        complete = self.distribution(frame, replace(guard, blocked_management_subjects=("hero", "town")))
        self.assertFalse(set(complete._kinds()).intersection(g.MANAGEMENT_KINDS))
        self.assertIn("close", complete._kinds())

    def test_artifact_destination_budget_and_mandatory_queries(self):
        for mandatory in (False, True):
            obs = artifacts(mandatory=mandatory)
            frame = c.Frame("guard-model", 0, 0, 1, "decision", schema=c.EXPANDED_SCHEMA,
                            observation=obs, offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA), revision=0)
            guard = g.GuardSupport(g.VERSION, blocked_management_subjects=("other",))
            policy = self.distribution(frame, guard)
            if mandatory:
                allowed = [action("choose_level_up", "hero", selection=choice.ref)
                           for choice in frame.offers.capability_panel.choices]
                self.assertEqual(policy._kinds(), ("choose_level_up",))
            else:
                allowed = [action("close_capability", "hero")]
                self.assertEqual(policy._kinds(), ("close_capability",))
            self.assert_normalizes(policy, allowed)

    def test_capability_choice_filter_preserves_unblocked_actor(self):
        obs = artifacts()
        panel = obs.window.capability_panel
        unblocked = replace(panel.choices[0], ref="same-hero", destination="hero")
        obs = replace(obs, window=replace(obs.window,
                      capability_panel=replace(panel, choices=(*panel.choices, unblocked))))
        frame = c.Frame("guard-choice", 0, 0, 1, "decision", schema=c.EXPANDED_SCHEMA,
                        observation=obs, offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA), revision=0)
        policy = self.distribution(frame, g.GuardSupport(g.VERSION, blocked_management_subjects=("other",)))
        selected = action("move_artifact", "hero", selection="same-hero")
        self.assert_normalizes(policy, [action("close_capability", "hero"), selected])
        self.assertEqual(policy.trace(selected)[-1].support_size, 1)
        with self.assertRaises(ValueError): policy.evaluate(action("move_artifact", "hero", selection="choice-1"))

    def test_unaffected_quantity_distribution_and_legacy_path_are_exact(self):
        frame = market_history().decision
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        old = self.distribution(frame, None, model)
        guarded = self.distribution(frame, g.GuardSupport(g.VERSION), model)
        choices = [action("close_capability", "town")]
        choices += [action("trade_resource", "town", selection="choice-1", count=count) for count in range(1, 5)]
        for selected in choices:
            self.assertTrue(torch.equal(old.log_prob(selected), guarded.log_prob(selected)))
        self.assert_normalizes(guarded, choices)
        self.assertEqual(old.support.distribution_schema, m.EXPANDED_DISTRIBUTION_SCHEMA)
        self.assertIsNone(old.support.guard)

    def test_encoder_tracks_all_frames_and_explicit_none_reconstructs_legacy(self):
        history = c.PlayerHistory("guard-encoder", 0, schema=c.EXPANDED_SCHEMA)
        decide(history, expanded_observation())
        navigate(history, action("open_hero", "hero"), expanded_observation("hero"))
        for _ in range(2):
            navigate(history, action("open_capability", "hero", panel="artifacts"), artifacts())
            navigate(history, action("close_capability", "hero"), expanded_observation("hero"), "closed")
        enabled = m.encode_history(history.frames, guard_version=g.VERSION)
        expected = g.Tracker(g.VERSION)
        self.assertEqual([item.guard for item in enabled], [expected.observe(frame) for frame in history.frames])
        selected = action("open_capability", "hero", panel="artifacts")
        self.assertFalse(enabled[-1].guard.permits(selected, history.decision.observation, history.decision.offers))
        with patch.object(g, "ACTIVE_VERSION", g.VERSION):
            default = m.encode_history(history.frames)
            legacy = m.encode_history(history.frames, guard_version=None)
        self.assertEqual(default, enabled)
        self.assertTrue(all(item.guard is None for item in legacy))
        self.assertEqual([item.semantics for item in enabled], [item.semantics for item in legacy])
        self.assertEqual([item.observation for item in enabled], [item.observation for item in legacy])
        self.assertNotEqual(m.frame_key(enabled[-1]), m.frame_key(legacy[-1]))
        self.assertEqual(m.frame_key(legacy[-1]), repr((legacy[-1].semantics, legacy[-1].observation)))
        self.assertEqual(m.decision_support(history.decision, enabled[-1].guard).guard, enabled[-1].guard)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        before, after = model.initial_state(), model.initial_state()
        for old_frame, new_frame in zip(legacy, enabled):
            old, new = model.forward_step(old_frame, before), model.forward_step(new_frame, after)
            self.assertTrue(torch.equal(old.state, new.state))
            if old.decision is not None:
                self.assertTrue(torch.equal(old.decision.value, new.decision.value))
            before, after = old.state, new.state

    def test_encoder_rejects_bad_chronology_without_spending_guard_budget(self):
        history = map_history()
        encoder = m.ContextualEncoder(history.episode, history.player, guard_version=g.VERSION)
        first = encoder.encode(history.decision)
        previous = encoder.guard_tracker.copy()
        with self.assertRaises(ValueError): encoder.encode(history.decision)
        self.assertEqual(encoder.guard_tracker.__dict__, previous.__dict__)
        self.assertEqual(encoder.history.frames, (first.frame,))

    def test_guard_schema_and_nondecision_attachment_are_rejected(self):
        history = map_history()
        guard = g.Tracker(g.VERSION).observe(history.decision)
        with self.assertRaises(ValueError):
            m.DecisionSupport(history.decision.offers.schema, history.decision.offers.context,
                              history.decision.offers, m.DISTRIBUTION_SCHEMA, guard)
        with self.assertRaises(ValueError): m.distribution_schema_for(c.SCHEMA, "unknown")
        history.emit("submitted", action=c.Action("end_turn"), decision_seq=history.decision.seq)
        encoded = m.encode_history(history.frames, guard_version=None)[-1]
        model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))
        with self.assertRaises(ValueError): model.forward_step(replace(encoded, guard=guard), model.initial_state())


if __name__ == "__main__":
    unittest.main()
