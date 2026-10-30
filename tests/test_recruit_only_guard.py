"""Offline support checks for the temporary recruitment-only army experiment."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import torch

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import interaction_guard as g
from tests.test_contextual_armies import army, history_at_armies, selected as transfer
from tests.test_contextual_capabilities import action, observation, decide
from tests.test_contextual_dialogs import history_at as dialog_history, panel as dialog_panel
from tests.test_contextual_garrisons import at_garrison
from tests.test_contextual_management import history_at, at_stack, at_confirmation, observation as management_observation
from tests.test_contextual_management_model import all_actions
from tests.test_contextual_model import map_history, recruitment, town_history
from tests.test_interaction_guard import artifacts


def support(frame, version=g.RECRUIT_ONLY_VERSION):
    return g.Tracker(version).observe(frame)


def permitted(frame, guard, selected):
    return guard.permits(selected, frame.observation, frame.offers)


class RecruitOnlyGuardTests(unittest.TestCase):
    def test_version_is_explicit_and_historical_snapshot_shape_is_preserved(self):
        self.assertEqual(g.RECRUIT_ONLY_VERSION, "visible-interaction-budget-recruit-only-v1")
        self.assertIsNone(g.ACTIVE_VERSION)
        frame = history_at().decision
        old, new = support(frame, g.REVEALED_VERSION), support(frame)
        self.assertEqual(replace(new, version=g.REVEALED_VERSION), old)
        self.assertEqual(set(new.to_dict()), set(old.to_dict()))
        self.assertEqual(g.GuardSupport.from_dict(json.loads(json.dumps(new.to_dict()))), new)
        self.assertEqual(g.GuardSupport.from_dict(old.to_dict()), old)
        with self.assertRaises(ValueError):
            g.GuardSupport.from_dict({**new.to_dict(), "version": "recruit-only-v2"})

    def test_all_existing_slot_operations_and_inspection_are_explicitly_excluded(self):
        for history in (history_at(), history_at("hero"), at_stack(), at_confirmation()):
            frame = history.decision
            native_before = c.encode_frame(frame)
            for selected in all_actions(frame):
                with self.subTest(context=frame.offers.context, action=selected):
                    self.assertTrue(frame.offers.permits(selected))
                    for version in (g.VERSION, g.REVEALED_VERSION):
                        self.assertTrue(permitted(frame, support(frame, version), selected))
                    self.assertEqual(permitted(frame, support(frame), selected),
                                     selected.kind not in g.RECRUIT_ONLY_BLOCKED_KINDS)
            self.assertEqual(c.encode_frame(frame), native_before)
        frame = history_at_armies().decision
        self.assertTrue(frame.offers.permits(transfer()))
        self.assertFalse(permitted(frame, support(frame), transfer()))
        self.assertTrue(permitted(frame, support(frame, g.REVEALED_VERSION), transfer()))

    def test_new_day_and_world_progress_do_not_remove_the_restriction(self):
        frame = history_at("hero").decision
        selected = next(a for a in all_actions(frame) if a.kind == "swap_stacks")
        tracker = g.Tracker(g.RECRUIT_ONLY_VERSION)
        for seq, day, gold in ((0, 1, 1000), (1, 2, 1000), (2, 3, 2000)):
            obs = replace(frame.observation, resources=(0, 0, 0, 0, 0, 0, gold))
            changed = replace(frame, seq=seq, day=day, observation=obs)
            self.assertFalse(permitted(changed, tracker.observe(changed), selected))

    def test_whole_hero_garrison_role_retains_existing_daily_budget(self):
        frame = history_at().decision
        selected = next(a for a in all_actions(frame) if a.kind == "swap_garrison")
        self.assertTrue(permitted(frame, support(frame), selected))
        limited = replace(support(frame), blocked_management_subjects=("hero",))
        self.assertFalse(permitted(frame, limited, selected))

    def test_garrison_queries_keep_policy_controlled_close_for_all_roles(self):
        for owner in ("self", "ally"):
            for locked in (False, True):
                frame = at_garrison(object_kind=33, owner=owner, locked=locked,
                                    stacks={0: (0, 3), 1: (2, 2)}).decision
                guard = support(frame)
                selected = action("close")
                self.assertTrue(frame.offers.permits(selected))
                self.assertTrue(permitted(frame, guard, selected))
                blocked = 0
                for kind in ("split_stack", "merge_stacks", "swap_stacks"):
                    for source, destination, maximum in c.army_operation_choices(frame.offers.armies, kind):
                        selected = action(kind, "hero", selection=source, destination=destination,
                                          **({"count": maximum} if kind == "split_stack" else {}))
                        self.assertTrue(frame.offers.permits(selected))
                        self.assertFalse(permitted(frame, guard, selected))
                        blocked += 1
                self.assertGreater(blocked, 0)

    def test_neutral_overflow_and_own_hero_exchange_keep_explicit_close(self):
        own = army("own", "hero", "hero", {i: (i, i + 1) for i in range(7)})
        neutral = replace(army("joining", "town_garrison", "neutral", {0: (20, 10)}), role="joining_neutral")
        target = c.ObjectView("neutral", (2, 1, 0), 54, 0, "neutral")
        obs = replace(observation(), objects=(target,),
            tiles=(*observation().tiles, c.Tile((2, 1, 0), 1, 0, False, True)),
            window=c.Window(c.Context("exchange", "hero", None, "neutral"), None,
                            armies=(own, neutral), capability_controls=()))
        frame = c.Frame("synthetic-overflow", 0, 0, 1, "decision", schema=c.EXPANDED_SCHEMA,
                        observation=obs, offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA), revision=0)
        guard = support(frame)
        self.assertTrue(permitted(frame, guard, action("close")))
        self.assertTrue(frame.offers.permits(action("open_stack", "hero", selection="joining-0")))
        self.assertFalse(permitted(frame, guard, action("open_stack", "hero", selection="joining-0")))
        exchange = management_observation("exchange")
        exchange = replace(exchange, window=replace(exchange.window, capability_controls=()))
        native = c.Offers.from_observation(exchange, schema=c.EXPANDED_ACTION_SCHEMA)
        self.assertTrue(guard.permits(action("close"), exchange, native))
        selected = action("swap_stacks", "hero", selection="first-0", destination="second-1")
        self.assertTrue(native.permits(selected))
        self.assertFalse(guard.permits(selected, exchange, native))

    def test_restriction_cannot_be_bypassed_by_a_mandatory_context(self):
        # A future query must not inherit the generic mandatory-choice exemption
        # for slot operations. Full typed garrison/overflow contracts are above.
        offers = SimpleNamespace(permits=lambda selected: True,
                                 context=SimpleNamespace(kind="dialog"), capability_panel=None)
        for kind in g.RECRUIT_ONLY_BLOCKED_KINDS:
            self.assertFalse(g.GuardSupport(g.RECRUIT_ONLY_VERSION).permits(SimpleNamespace(kind=kind), None, offers))

    def test_revealed_movement_and_failed_target_rules_stay_identical(self):
        obs = observation()
        obs = replace(obs, tiles=(*obs.tiles, c.Tile((3, 2, 0), 1, 0, True, False)))
        frame = map_history(obs, schema=c.EXPANDED_SCHEMA).decision
        old, new = support(frame, g.REVEALED_VERSION), support(frame)
        for hero in obs.heroes:
            for x in range(obs.size[0]):
                for y in range(obs.size[1]):
                    selected = action("move_to", hero.handle, x=x, y=y)
                    self.assertEqual(permitted(frame, old, selected), permitted(frame, new, selected))
            self.assertEqual(tuple(old.movement_coordinates(hero.handle, obs, frame.offers)),
                             tuple(new.movement_coordinates(hero.handle, obs, frame.offers)))
        for selected in (action("open_hero", "hero"), action("open_town", "town"), action("end_turn")):
            self.assertTrue(permitted(frame, new, selected))


class RecruitOnlyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def distribution(self, frame, guard):
        with torch.random.fork_rng():
            torch.manual_seed(197)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(frame.schema),
                                                         hidden_size=8, feature_size=4))
        encoded = m.ContextualInput(frame, m._node("frame"), frame.observation, guard)
        return model.forward_step(encoded, model.initial_state()).decision.distribution

    def assert_normalized_and_sampled(self, policy, actions):
        self.assertAlmostEqual(sum(policy.log_prob(a).exp().item() for a in actions), 1, places=5)
        first, second = torch.Generator().manual_seed(977), torch.Generator().manual_seed(977)
        for _ in range(24):
            selected, evaluated = policy.sample_evaluated(first)
            self.assertIn(selected, actions)
            self.assertEqual(selected, policy.sample(second))
            self.assertTrue(torch.equal(evaluated.log_prob, policy.log_prob(selected)))
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_guarded_model_filters_before_normalization_and_preserves_original_offers(self):
        for player in (0, 1):
            for history in (history_at(), history_at("hero"), at_stack(), at_confirmation()):
                frame = replace(history.decision, player=player)
                guard = support(frame)
                policy = self.distribution(frame, guard)
                allowed = [a for a in all_actions(frame) if a.kind not in g.RECRUIT_ONLY_BLOCKED_KINDS]
                self.assert_normalized_and_sampled(policy, allowed)
                self.assertEqual(policy.support.offers, frame.offers)
                self.assertTrue(policy.support.distribution_schema.endswith("+" + g.RECRUIT_ONLY_VERSION))
                self.assertFalse(set(policy._kinds()) & g.RECRUIT_ONLY_BLOCKED_KINDS)
                for selected in all_actions(frame):
                    if selected not in allowed:
                        with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
                            policy.log_prob(selected)

    def test_recruitment_keeps_all_native_counts_and_exact_probabilities(self):
        frame = town_history(recruitment()).decision
        old = self.distribution(frame, support(frame, g.REVEALED_VERSION))
        new = self.distribution(frame, support(frame))
        actions = [c.Action("back", "town", schema=c.TOWN_ACTION_SCHEMA)]
        actions += [c.Action("select_recruit", "town", selection=ref, schema=c.TOWN_ACTION_SCHEMA)
                    for ref in ("base", "upgrade")]
        actions += [c.Action("recruit", "town", selection="base", destination="upper", count=count,
                             schema=c.TOWN_ACTION_SCHEMA) for count in (1, 2, 3)]
        self.assert_normalized_and_sampled(new, actions)
        for selected in actions:
            self.assertTrue(torch.equal(old.log_prob(selected), new.log_prob(selected)))

    def test_forced_dialog_answers_and_capability_choices_remain_policy_choices(self):
        frame = dialog_history(dialog_panel("neutral_paid_join")).decision
        policy = self.distribution(frame, support(frame))
        actions = [c.Action("choose_dialog", "hero", selection=item.ref, schema=c.DIALOGS_ACTION_SCHEMA)
                   for item in frame.offers.dialog_panel.choices]
        self.assertEqual(len(actions), 2)
        self.assert_normalized_and_sampled(policy, actions)
        obs = artifacts("query", mandatory=True)
        frame = c.Frame("synthetic-level", 0, 0, 1, "decision", schema=c.EXPANDED_SCHEMA,
                        observation=obs, offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA), revision=0)
        policy = self.distribution(frame, support(frame))
        actions = [action("choose_level_up", "hero", selection=item.ref) for item in frame.offers.capability_panel.choices]
        self.assert_normalized_and_sampled(policy, actions)

    def test_garrison_empty_management_support_still_requires_sampled_close(self):
        frame = at_garrison(stacks={0: (0, 3), 1: (2, 2)}).decision
        original = c.encode_frame(frame)
        policy = self.distribution(frame, support(frame))
        self.assertEqual(policy._kinds(), ("close",))
        self.assert_normalized_and_sampled(policy, [action("close")])
        self.assertEqual(c.encode_frame(frame), original)

    def test_history_reconstruction_keeps_model_value_state_and_old_versions_exact(self):
        history = c.PlayerHistory("synthetic-recruit-guard", 0, schema=c.EXPANDED_SCHEMA)
        decide(history, observation())
        frames = {version: m.encode_history(history.frames, guard_version=version)
                  for version in (None, g.VERSION, g.REVEALED_VERSION, g.RECRUIT_ONLY_VERSION)}
        with torch.random.fork_rng():
            torch.manual_seed(318)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        outputs = {version: model.forward_step(encoded[0], model.initial_state())
                   for version, encoded in frames.items()}
        for version, output in outputs.items():
            self.assertTrue(torch.equal(output.state, outputs[None].state))
            self.assertTrue(torch.equal(output.decision.value, outputs[None].decision.value))
            self.assertEqual(frames[version][0].semantics, frames[None][0].semantics)
        self.assertNotEqual(m.frame_key(frames[g.REVEALED_VERSION][0]),
                            m.frame_key(frames[g.RECRUIT_ONLY_VERSION][0]))


if __name__ == "__main__":
    unittest.main()
