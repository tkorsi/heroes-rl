"""Synthetic support-budget tests; these do not launch gameplay or PPO."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from experiments.learner import contextual as c
from experiments.learner import interaction_guard as g
from tests.test_contextual_capabilities import observation, action, choice


class Trace:
    def __init__(self):
        self.tracker = g.Tracker(g.VERSION)
        self.frames = []
        self.day = 1
        self.decision = None

    def emit(self, kind, **values):
        frame = c.Frame("guard-test", 0, len(self.frames), self.day, kind,
                        schema=c.EXPANDED_SCHEMA, **values)
        self.frames.append(frame)
        support = self.tracker.observe(frame)
        if kind == "decision":
            self.decision = frame
        return support

    def decide(self, obs):
        return self.emit("decision", observation=obs,
                         offers=c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA),
                         revision=len(self.frames))

    def consume(self, selected, status="opened", **result):
        seq = self.decision.seq
        self.emit("submitted", decision_seq=seq, action=selected)
        self.emit("consumed", decision_seq=seq, consumer="engine")
        self.emit("result", decision_seq=seq, result=c.Result(status, status != "rejected", **result))


def permits(support, selected, obs):
    offers = c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA)
    return support.permits(selected, obs, offers)


def artifacts(ref="panel-0", *, mandatory=False):
    if mandatory:
        entries = (choice("choose_level_up", subject="hero", skill=1, skill_level=1),
                   choice("choose_level_up", subject="hero", ref="choice-2", skill=2, skill_level=1))
        panel = c.CapabilityPanel(ref, "level_up", c.Context("hero", "hero"), True, 2, 1, (), entries)
        context = c.Context("capability", "hero", "level_up", ref)
    else:
        entries = (choice("move_artifact", subject="hero", artifact=1, slot=0,
                         destination="other", destination_slot=19),)
        panel = c.CapabilityPanel(ref, "artifacts", c.Context("hero", "hero"), False, None, None, (), entries)
        context = c.Context("capability", "hero", "artifacts")
    return replace(observation(), window=c.Window(context, None, armies=(),
                   capability_controls=(), capability_panel=panel))


class InteractionGuardTests(unittest.TestCase):
    def test_default_disabled_and_explicit_version(self):
        trace = Trace()
        trace.decide(observation())
        self.assertIsNone(g.Tracker().observe(trace.frames[0]))
        with patch.object(g, "ACTIVE_VERSION", g.VERSION):
            self.assertIsInstance(g.Tracker().observe(trace.frames[0]), g.GuardSupport)
            self.assertIsNone(g.Tracker(None).observe(trace.frames[0]))
        with self.assertRaises(ValueError):
            g.Tracker("future")

    def test_two_artifact_reopens_then_safe_escape(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        for index in range(2):
            self.assertTrue(permits(trace.decide(obs), selected, obs))
            trace.consume(selected)
            panel = artifacts(f"panel-{index}")
            support = trace.decide(panel)
            self.assertTrue(permits(support, action("close_capability", "hero"), panel))
            trace.consume(action("close_capability", "hero"), "closed")
        support = trace.decide(obs)
        self.assertFalse(permits(support, selected, obs))
        self.assertTrue(permits(support, action("open_capability", "hero", panel="spellbook"), obs))
        self.assertTrue(permits(support, action("close"), obs))
        trace.consume(action("close"), "closed")
        map_obs = observation()
        self.assertTrue(permits(trace.decide(map_obs), action("end_turn"), map_obs))

    def test_real_visible_progress_resets_navigation_but_not_ui_metadata(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        for _ in range(2):
            trace.decide(obs)
            trace.consume(selected)
        self.assertFalse(permits(trace.decide(replace(obs, window=replace(obs.window, mana=9))), selected, obs))
        moved = replace(obs, heroes=(replace(obs.heroes[0], movement=99), obs.heroes[1]))
        self.assertTrue(permits(trace.decide(moved), selected, moved))
        for _ in range(2):
            trace.consume(selected)
            trace.decide(moved)
        richer = replace(moved, resources=(*moved.resources[:6], 1001))
        self.assertTrue(permits(trace.decide(richer), selected, richer))

    def test_zero_movement_and_current_tile_blocked_without_new_native_mask(self):
        obs = observation()
        obs = replace(obs, heroes=(replace(obs.heroes[0], movement=0), obs.heroes[1]))
        support = Trace().decide(obs)
        native = c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA)
        selected = action("move_to", "hero", x=3, y=2)
        self.assertTrue(native.permits(selected))
        self.assertFalse(support.permits(selected, obs, native))
        self.assertFalse(permits(support, action("move_to", "other", x=2, y=1), obs))
        self.assertTrue(permits(support, action("move_to", "other", x=3, y=2), obs))

    def test_failed_target_retries_only_after_visible_progress(self):
        trace, obs = Trace(), observation()
        selected = action("move_to", "hero", x=3, y=2)
        trace.decide(obs)
        trace.consume(selected, "no_route_known", position=obs.heroes[0].position, movement=100, destination_reached=False)
        support = trace.decide(obs)
        self.assertFalse(permits(support, selected, obs))
        self.assertTrue(permits(support, action("move_to", "hero", x=0, y=2), obs))
        changed = replace(obs, heroes=(replace(obs.heroes[0], movement=90), obs.heroes[1]))
        self.assertTrue(permits(trace.decide(changed), selected, changed))

    def test_partial_movement_and_free_interaction_are_not_failed_targets(self):
        for status, position, movement in (("movement_limit", (3, 1, 0), 0),
                                            ("interaction", (1, 1, 0), 100)):
            trace, obs = Trace(), observation()
            selected = action("move_to", "hero", x=3, y=2)
            trace.decide(obs)
            trace.consume(selected, status, position=position, movement=movement, destination_reached=False)
            self.assertTrue(permits(trace.decide(obs), selected, obs))

    def test_reversible_management_bound_per_actor_not_count_or_slot(self):
        trace, obs = Trace(), observation("hero")
        for index in range(g.MANAGEMENT_LIMIT):
            selected = action("split_stack", "hero", selection="own-0", destination="own-2", count=index % 13 + 1)
            self.assertTrue(permits(trace.decide(obs), selected, obs))
            trace.consume(selected, "split")
        support = trace.decide(obs)
        self.assertFalse(permits(support, action("swap_stacks", "hero", selection="own-0", destination="own-1"), obs))
        self.assertTrue(permits(support, action("close"), obs))
        changed = replace(obs, heroes=(replace(obs.heroes[0], movement=90), obs.heroes[1]))
        self.assertFalse(permits(trace.decide(changed), selected, changed))
        trace.day = 2
        self.assertTrue(permits(trace.decide(changed), selected, changed))

    def test_artifact_transfers_charge_both_actors_and_do_not_reset_on_refs(self):
        trace = Trace()
        for index in range(g.MANAGEMENT_LIMIT):
            obs = artifacts(f"inventory-{index}")
            selected = action("move_artifact", "hero", selection="choice-1")
            self.assertTrue(permits(trace.decide(obs), selected, obs))
            trace.consume(selected, "capability_applied")
        support = trace.decide(artifacts("new-ref"))
        self.assertEqual(support.blocked_management_subjects, ("hero", "other"))
        self.assertFalse(permits(support, selected, obs))
        mandatory = artifacts("query", mandatory=True)
        self.assertTrue(permits(trace.decide(mandatory), action("choose_level_up", "hero", selection="choice-1"), mandatory))

    def test_stack_navigation_key_ignores_references_and_layout(self):
        obs = observation("hero")
        selected = action("open_stack", "hero", selection="own-0")
        offers = c.Offers.from_observation(obs, schema=c.EXPANDED_ACTION_SCHEMA)
        before = g.nav_key(selected, obs, offers)
        row = replace(obs.window.armies[0], ref="new-row", slots=tuple(
            replace(slot, ref=f"new-{slot.slot}") for slot in obs.window.armies[0].slots))
        changed = replace(obs, window=replace(obs.window, armies=(row,)))
        after = g.nav_key(action("open_stack", "hero", selection="new-0"), changed,
                          c.Offers.from_observation(changed, schema=c.EXPANDED_ACTION_SCHEMA))
        self.assertEqual(before, after)

    def test_free_management_and_observation_order_do_not_renew_navigation(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        for _ in range(2):
            trace.decide(obs)
            trace.consume(selected)
        trace.decide(obs)
        trace.consume(action("swap_stacks", "hero", selection="own-0", destination="own-1"), "swapped")
        # Even if a free operation changes a delivered stat, it is not a fresh
        # navigation epoch. UI/reference/order changes are not progress either.
        changed = replace(obs, heroes=(obs.heroes[1], replace(obs.heroes[0], movement=99)))
        self.assertFalse(permits(trace.decide(changed), selected, changed))
        reordered = replace(changed, heroes=tuple(reversed(changed.heroes)))
        self.assertFalse(permits(trace.decide(reordered), selected, reordered))

    def test_completed_cost_effect_and_new_day_allow_inspection_again(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        for _ in range(2):
            trace.decide(obs)
            trace.consume(selected)
        self.assertFalse(permits(trace.decide(obs), selected, obs))
        trace.emit("event", event=c.VisibleEvent("battle_complete", "hero", side="attacker", won=False))
        self.assertTrue(permits(trace.decide(obs), selected, obs))
        for _ in range(2):
            trace.consume(selected)
            trace.decide(obs)
        self.assertTrue(trace.tracker.navigation)
        trace.day = 2
        self.assertTrue(permits(trace.decide(obs), selected, obs))

    def test_cast_target_then_cancel_does_not_renew_free_navigation(self):
        trace, hero = Trace(), observation("hero")
        opened = action("open_capability", "hero", panel="spellbook")
        spellbook = replace(observation(), window=c.Window(
            c.Context("capability", "hero", "spellbook"), None, mana=10, armies=(), capability_controls=(),
            capability_panel=c.CapabilityPanel("book", "spellbook", hero.context, False, None, None, (),
                (choice("cast_adventure_spell", subject="hero", spell=1),))))
        target = replace(observation(), window=c.Window(
            c.Context("capability", "hero", "spell_target", "query"), None, armies=(), capability_controls=(),
            capability_panel=c.CapabilityPanel("query", "spell_target", c.Context(), True, None, None, (),
                (choice("cancel_spell_target", subject="hero", spell=1),))))
        for _ in range(2):
            self.assertTrue(permits(trace.decide(hero), opened, hero))
            trace.consume(opened)
            trace.decide(spellbook)
            trace.consume(action("cast_adventure_spell", "hero", selection="choice-1"), "capability_applied")
            support = trace.decide(target)
            cancel = action("cancel_spell_target", "hero", selection="choice-1")
            self.assertTrue(permits(support, cancel, target))
            trace.consume(cancel, "capability_applied")
        self.assertFalse(permits(trace.decide(hero), opened, hero))

    def test_snapshot_json_is_strict_immutable_and_recomputed_from_history(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        first = trace.decide(obs)
        for _ in range(2):
            trace.consume(selected)
            last = trace.decide(obs)
        self.assertEqual(first.blocked_navigation, ())
        restored = g.GuardSupport.from_dict(json.loads(json.dumps(last.to_dict())))
        self.assertEqual(restored, last)
        tracker = g.Tracker(g.VERSION)
        rebuilt = [tracker.observe(frame) for frame in trace.frames]
        self.assertEqual(rebuilt[-1], last)
        for invalid in ({**last.to_dict(), "extra": 1}, {**last.to_dict(), "version": "unknown"},
                        {**last.to_dict(), "blocked_coordinates": [["hero", True, 0]]},
                        {**last.to_dict(), "blocked_navigation": list(reversed(last.to_dict()["blocked_navigation"])) * 2}):
            with self.assertRaises(ValueError):
                g.GuardSupport.from_dict(invalid)
        clone = tracker.copy()
        self.assertIs(clone.decision, tracker.decision)
        clone.navigation.clear()
        self.assertNotEqual(clone.navigation, tracker.navigation)

    def test_episode_isolation_and_consumption_not_submission(self):
        trace, obs = Trace(), observation("hero")
        selected = action("open_capability", "hero", panel="artifacts")
        trace.decide(obs)
        trace.emit("submitted", decision_seq=trace.decision.seq, action=selected)
        self.assertFalse(trace.tracker.navigation)
        with self.assertRaises(ValueError):
            trace.tracker.observe(replace(trace.frames[-1], episode="other", seq=len(trace.frames)))
        self.assertEqual(g.Tracker(g.VERSION).navigation, {})


if __name__ == "__main__":
    unittest.main()
