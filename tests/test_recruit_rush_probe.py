"""Regression checks for diagnostic planning and terminal action accounting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.learner import contextual as c
from validation.probes.validate_recruit_rush import Rush, diagnostic_accepted, summarize_episode


def army(ref, role, subject, count):
    return c.ArmyView(ref, role, subject, role != "town_garrison", tuple(
        c.ArmySlot(f"{ref}-{slot}", slot, 0 if count and slot == 0 else None,
                   count if slot == 0 else 0) for slot in range(7)))


class RecruitRushTests(unittest.TestCase):
    def test_verified_replay_cannot_hide_summary_or_cleanup_failure(self):
        report = {"replay": {"verified": True}, "owned_processes_closed": True, "status": "completed"}
        self.assertTrue(diagnostic_accepted(report))
        for change in ({"error": "summary failed"}, {"cleanup_error": "process alive"},
                       {"owned_processes_closed": False}, {"status": "diagnostic_failed"}):
            self.assertFalse(diagnostic_accepted({**report, **change}))

    def test_v8_recruits_are_transferred_from_window_armies_before_departure(self):
        plan = Rush()
        plan.stage, plan.hero, plan.town, plan.direction = "purchase", "hero", "town", 1
        # v8 removed the older town_panel.armies field. The live first probe
        # exposed this regression: buying troops alone leaves them in town.
        context = c.Context("town", "town", "overview")
        window = SimpleNamespace(town_panel=c.TownPanel("overview"), armies=(
            army("upper", "town_garrison", "town", 14),
            army("visitor", "visiting_hero", "hero", 25)))
        frame = SimpleNamespace(observation=SimpleNamespace(context=context, window=window))
        action, _ = plan.choose(frame)
        self.assertEqual(action.kind, "merge_stacks")
        self.assertEqual((action.selection, action.destination), ("upper-0", "visitor-0"))
        self.assertEqual(plan.stage, "purchase")

    def test_frontier_exploration_uses_only_observed_free_tiles(self):
        plan = Rush()
        plan.stage, plan.hero, plan.town, plan.direction = "rush", "hero", "town", 1
        hero = c.Hero("hero", (4, 7, 0), 1560, 1560, False)
        tiles = (c.Tile((8, 7, 0), 0, 0, False, False),
                 c.Tile((9, 7, 0), 0, 0, False, False),
                 c.Tile((15, 7, 0), 0, 0, True, True))
        frame = SimpleNamespace(day=1, observation=SimpleNamespace(
            context=c.Context(), heroes=(hero,), objects=(), tiles=tiles))
        action, label = plan.choose(frame)
        self.assertEqual((action.x, action.y), (9, 7))
        self.assertEqual(label, "explore_visible_frontier")

    def test_new_creature_stack_uses_empty_slot_swap_not_merge(self):
        plan = Rush()
        plan.stage, plan.hero, plan.town, plan.direction = "purchase", "hero", "town", 1
        source = army("upper", "town_garrison", "town", 14)
        visitor = army("visitor", "visiting_hero", "hero", 25)
        from dataclasses import replace
        visitor = replace(visitor, slots=(replace(visitor.slots[0], creature=2), *visitor.slots[1:]))
        frame = SimpleNamespace(observation=SimpleNamespace(
            context=c.Context("town", "town", "overview"),
            window=SimpleNamespace(town_panel=c.TownPanel("overview"), armies=(source, visitor))))
        action, _ = plan.choose(frame)
        self.assertEqual(action.kind, "swap_stacks")
        self.assertEqual(action.destination, "visitor-1")

    def test_final_modal_reply_counts_even_when_terminal_supersedes_result(self):
        frames = [
            {"kind": "consumed", "decision_seq": 10},
            {"kind": "event", "seq": 12, "day": 1,
             "event": {"kind": "battle_start", "side": "attacker"}},
            {"kind": "result", "decision_seq": 10, "seq": 13, "day": 1,
             "result": {"status": "interaction", "accepted": True}},
            {"kind": "consumed", "decision_seq": 20},
            {"kind": "event", "seq": 22, "day": 1,
             "event": {"kind": "battle_complete", "won": True}},
            {"kind": "superseded", "seq": 23, "decision_seq": 20, "reason": "terminal"},
            {"kind": "terminal", "seq": 24, "day": 1, "outcome": "win"},
        ]
        actions = [{"index": 1, "decision_seq": 10}, {"index": 2, "decision_seq": 20}]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agents/red/history-v8.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
            evidence = summarize_episode(Path(folder), actions, "red")
        self.assertEqual(evidence["consumed_actions"], [1, 2])
        self.assertEqual(evidence["typed_actions_attempted"], 2)
        self.assertEqual(evidence["battles"][0]["start_action_index"], 1)
        self.assertEqual(evidence["battles"][0]["completion_action_index"], 2)
        self.assertEqual(evidence["terminal"][0]["outcome"], "win")
        self.assertEqual(evidence["superseded"][0]["action_index"], 2)


if __name__ == "__main__":
    unittest.main()
