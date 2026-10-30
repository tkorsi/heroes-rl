"""Current-observation movement support; synthetic models only, no native games."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import torch

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import interaction_guard as g
from experiments.learner.ppo import PPOConfig
from tests.test_contextual_model import observation, map_history
from tests.test_contextual_capabilities import observation as expanded_observation, action
from tests.test_interaction_guard import Trace, artifacts, permits


def revealed_trace():
    trace = Trace()
    trace.tracker = g.Tracker(g.REVEALED_VERSION)
    return trace


def tile(x, y, *, blocked=False, visitable=False):
    return c.Tile((x, y, 0), 1, 0, blocked, visitable)


class RevealedMovementGuardTests(unittest.TestCase):
    def test_level_membership_is_subject_specific_without_extending_map_contract(self):
        obs = observation()
        with self.assertRaisesRegex(ValueError, "one map level"):
            replace(obs, size=(3, 2, 2))
        # The public typed contract is still one-level. Exercise the helper's
        # defensive level binding directly without claiming two-level support.
        observed = SimpleNamespace(heroes=(obs.heroes[0], replace(obs.heroes[0], handle="underground",
                                                                position=(0, 0, 1))),
                                   tiles=(tile(1, 0), c.Tile((2, 1, 1), 1, 0, True, False)))
        support = g.GuardSupport(g.REVEALED_VERSION)
        self.assertEqual(support._revealed_coordinates("hero", observed), frozenset({(1, 0)}))
        self.assertEqual(support._revealed_coordinates("underground", observed), frozenset({(2, 1)}))

    def test_version_is_explicit_and_snapshot_shape_stays_historical(self):
        self.assertEqual(g.VERSION, "visible-interaction-budget-v1")
        self.assertIsNone(g.ACTIVE_VERSION)
        self.assertEqual(g.SUPPORTED_VERSIONS, (g.VERSION, g.REVEALED_VERSION))
        frame = map_history().decision
        old = g.Tracker(g.VERSION).observe(frame)
        new = g.Tracker(g.REVEALED_VERSION).observe(frame)
        self.assertEqual(replace(new, version=g.VERSION), old)
        self.assertEqual(set(new.to_dict()), set(old.to_dict()))
        self.assertEqual(g.GuardSupport.from_dict(json.loads(json.dumps(new.to_dict()))), new)
        self.assertEqual(g.GuardSupport.from_dict(old.to_dict()), old)
        self.assertIsNone(g.Tracker(None).observe(frame))
        for version in ("visible-interaction-budget-v2", "unknown"):
            with self.assertRaises(ValueError):
                g.Tracker(version)
            with self.assertRaises(ValueError):
                g.GuardSupport(version)

    def test_only_delivered_tile_membership_is_added_native_offers_unchanged(self):
        obs = replace(observation(), tiles=(tile(0, 0), tile(1, 0, blocked=True),
                                           tile(2, 1, blocked=True, visitable=True)))
        frame = map_history(obs).decision
        native_before = frame.offers
        old = g.Tracker(g.VERSION).observe(frame)
        new = g.Tracker(g.REVEALED_VERSION).observe(frame)
        for x, y in ((1, 0), (2, 1)):
            selected = c.Action("move_to", "hero", x, y)
            self.assertTrue(new.permits(selected, obs, frame.offers))
        hidden = c.Action("move_to", "hero", 2, 0)
        self.assertTrue(frame.offers.permits(hidden))
        self.assertTrue(old.permits(hidden, obs, frame.offers))
        self.assertFalse(new.permits(hidden, obs, frame.offers))
        self.assertFalse(new.permits(c.Action("move_to", "hero", 0, 0), obs, frame.offers))
        self.assertEqual(frame.offers, native_before)
        self.assertEqual(tuple(new.movement_coordinates("hero", obs, frame.offers)), (1, 5))

    def test_newly_revealed_and_rehidden_targets_use_current_observation(self):
        trace = revealed_trace()
        hidden = expanded_observation()
        selected = action("move_to", "hero", x=3, y=2)
        hidden = replace(hidden, tiles=tuple(t for t in hidden.tiles if t.position != (3, 2, 0)))
        shown = replace(hidden, tiles=(*hidden.tiles, tile(3, 2, blocked=True)))
        first = trace.decide(hidden)
        self.assertFalse(permits(first, selected, hidden))
        self.assertTrue(permits(trace.decide(shown), selected, shown))
        self.assertFalse(permits(trace.decide(hidden), selected, hidden))
        # The immutable guard contains no accumulated last-seen tile cache.
        self.assertTrue(permits(first, selected, shown))
        self.assertNotIn("tiles", first.to_dict())

    def test_each_hero_keeps_its_own_current_failed_and_movement_restrictions(self):
        obs = replace(observation(), tiles=(tile(0, 0), tile(1, 0), tile(2, 1)),
                      heroes=(c.Hero("hero", (0, 0, 0), 100, in_garrison=False),
                              c.Hero("other", (2, 1, 0), 100, in_garrison=False),
                              c.Hero("spent", (1, 1, 0), 0, in_garrison=False),
                              c.Hero("garrison", (2, 0, 0), 100, in_garrison=True)))
        frame = map_history(obs, schema=c.EXPANDED_SCHEMA).decision
        support = g.Tracker(g.REVEALED_VERSION).observe(frame)
        support = replace(support, blocked_coordinates=tuple(sorted((*support.blocked_coordinates,
                                                                          ("hero", 1, 0)))))
        self.assertEqual(tuple(support.movement_coordinates("hero", obs, frame.offers)), (5,))
        self.assertEqual(tuple(support.movement_coordinates("other", obs, frame.offers)), (0, 1))
        for subject in ("spent", "garrison"):
            self.assertEqual(tuple(support.movement_coordinates(subject, obs, frame.offers)), ())
        for hero in obs.heroes:
            expected = tuple(y*obs.size[0]+x for y in range(obs.size[1]) for x in range(obs.size[0])
                             if support.permits(c.Action("move_to", hero.handle, x, y,
                                                         schema=c.EXPANDED_ACTION_SCHEMA), obs, frame.offers))
            self.assertEqual(tuple(support.movement_coordinates(hero.handle, obs, frame.offers)), expected)

    def test_failed_revealed_target_retains_v1_retry_budget(self):
        trace, obs = revealed_trace(), expanded_observation()
        obs = replace(obs, tiles=(*obs.tiles, tile(3, 2)))
        selected = action("move_to", "hero", x=3, y=2)
        self.assertTrue(permits(trace.decide(obs), selected, obs))
        trace.consume(selected, "no_route_known", position=obs.heroes[0].position,
                      movement=obs.heroes[0].movement, destination_reached=False)
        self.assertFalse(permits(trace.decide(obs), selected, obs))
        progressed = replace(obs, resources=(*obs.resources[:6], obs.resources[6]+1))
        self.assertTrue(permits(trace.decide(progressed), selected, progressed))

    def test_navigation_management_escape_and_mandatory_query_budgets_are_preserved(self):
        for version in g.SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                trace, obs = Trace(), expanded_observation("hero")
                trace.tracker = g.Tracker(version)
                opened = action("open_capability", "hero", panel="artifacts")
                for _ in range(g.NAVIGATION_LIMIT):
                    self.assertTrue(permits(trace.decide(obs), opened, obs))
                    trace.consume(opened)
                self.assertFalse(permits(trace.decide(obs), opened, obs))
                self.assertTrue(permits(trace.decide(obs), action("close"), obs))
                panel = artifacts()
                moved = action("move_artifact", "hero", selection="choice-1")
                for _ in range(g.MANAGEMENT_LIMIT):
                    self.assertTrue(permits(trace.decide(panel), moved, panel))
                    trace.consume(moved, "capability_applied")
                self.assertFalse(permits(trace.decide(panel), moved, panel))
                self.assertTrue(permits(trace.decide(panel), action("close_capability", "hero"), panel))
                mandatory = artifacts("query", mandatory=True)
                self.assertTrue(permits(trace.decide(mandatory),
                    action("choose_level_up", "hero", selection="choice-1"), mandatory))


class RevealedMovementModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(973)
        self.model = m.ContextualActorCritic(m.ModelConfig(hidden_size=8, feature_size=4))

    def policy(self, observation, version=g.REVEALED_VERSION):
        history = map_history(observation)
        frame = m.encode_history(history.frames, guard_version=version)[0]
        return self.model.forward_step(frame, self.model.initial_state()).decision.distribution

    def test_sparse_support_sampling_log_probability_and_normalization(self):
        obs = replace(observation(), tiles=(tile(2, 1, blocked=True), tile(0, 0), tile(1, 0)))
        policy = self.policy(obs)
        actions = [c.Action("end_turn"), c.Action("open_hero", "hero"), c.Action("open_town", "town"),
                   c.Action("move_to", "hero", 1, 0), c.Action("move_to", "hero", 2, 1)]
        probabilities = torch.stack([policy.log_prob(a).exp() for a in actions])
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.), rtol=0, atol=1e-6)
        self.assertEqual(tuple(policy._coordinates("hero")), (1, 5))
        self.assertEqual(policy.trace(actions[-1])[-1].support_size, 2)
        self.assertEqual(policy.support.distribution_schema, m.DISTRIBUTION_SCHEMA+"+"+g.REVEALED_VERSION)
        with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
            policy.log_prob(c.Action("move_to", "hero", 2, 0))
        first, second = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
        for _ in range(40):
            selected, evaluated = policy.sample_evaluated(first)
            self.assertIn(selected, actions)
            self.assertEqual(selected, policy.sample(second))
            self.assertTrue(torch.equal(evaluated.log_prob, policy.log_prob(selected)))
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        # Enumerated finite support has a valid entropy; the existing general
        # hierarchical entropy API and zero PPO coefficient remain unchanged.
        self.assertTrue(torch.isfinite(-(probabilities*probabilities.log()).sum()))
        self.assertFalse(hasattr(policy, "entropy"))
        self.assertEqual(PPOConfig().entropy_coefficient, 0)

    def test_empty_revealed_destinations_remove_move_without_selecting_escape_for_policy(self):
        for tiles in ((), (tile(0, 0),)):
            with self.subTest(tiles=tiles):
                obs = replace(observation(), tiles=tiles, towns=())
                policy = self.policy(obs)
                self.assertEqual(policy._kinds(), ("open_hero", "end_turn"))
                self.assertEqual(policy._subjects("move_to"), ())
                self.assertTrue(torch.isfinite(policy.log_prob(c.Action("open_hero", "hero"))))
                self.assertTrue(torch.isfinite(policy.log_prob(c.Action("end_turn"))))
                self.assertIn("move_to", self.policy(obs, g.VERSION)._kinds())

    def test_multihero_sampling_removes_only_subjects_without_revealed_destinations(self):
        obs = replace(observation(), heroes=(c.Hero("hero", (0, 0, 0), 100),
                      c.Hero("other", (2, 1, 0), 100), c.Hero("spent", (1, 0, 0), 0)))
        policy = self.policy(obs)
        self.assertEqual(policy._subjects("move_to"), ("other",))
        selected = c.Action("move_to", "other", 0, 0)
        self.assertEqual([(part.name, part.support_size) for part in policy.trace(selected)],
                         [("kind", 4), ("subject", 1), ("coordinate", 1)])
        actions = [c.Action("end_turn"), c.Action("open_town", "town"), selected]
        actions += [c.Action("open_hero", hero.handle) for hero in obs.heroes]
        probability = torch.stack([policy.log_prob(a).exp() for a in actions]).sum()
        torch.testing.assert_close(probability, torch.tensor(1.), rtol=0, atol=1e-6)

    def test_reordered_tiles_keep_coordinate_order_and_exact_probabilities(self):
        obs = replace(observation(), tiles=(tile(2, 1), tile(0, 0), tile(1, 0)))
        a = self.policy(obs)
        b = self.policy(replace(obs, tiles=tuple(reversed(obs.tiles))))
        self.assertEqual(tuple(a._coordinates("hero")), tuple(b._coordinates("hero")))
        for selected in (c.Action("move_to", "hero", 1, 0), c.Action("move_to", "hero", 2, 1),
                         c.Action("end_turn")):
            self.assertTrue(torch.equal(a.log_prob(selected), b.log_prob(selected)))

    def test_version_changes_support_only_not_model_shape_value_or_recurrent_state(self):
        history = map_history()
        shapes = {name: tuple(tensor.shape) for name, tensor in self.model.state_dict().items()}
        outputs = []
        encoded = []
        for version in (None, g.VERSION, g.REVEALED_VERSION):
            frame = m.encode_history(history.frames, guard_version=version)[0]
            encoded.append(frame)
            outputs.append(self.model.forward_step(frame, self.model.initial_state()))
        for output in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0].state, output.state))
            self.assertTrue(torch.equal(outputs[0].decision.value, output.decision.value))
        self.assertEqual(shapes, {name: tuple(tensor.shape) for name, tensor in self.model.state_dict().items()})
        self.assertEqual(encoded[0].semantics, encoded[2].semantics)
        self.assertNotEqual(m.frame_key(encoded[1]), m.frame_key(encoded[2]))
        # Historical v1 still permits unseen coordinates, even if the default
        # collection version changes elsewhere.
        self.assertTrue(torch.isfinite(outputs[1].decision.distribution.log_prob(c.Action("move_to", "hero", 2, 1))))
        with self.assertRaises(ValueError):
            outputs[2].decision.distribution.log_prob(c.Action("move_to", "hero", 2, 1))


if __name__ == "__main__":
    unittest.main()
