"""Descriptive synthetic evidence tests; no gameplay, model or optimizer runs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from experiments.learner import builtin_behavior
from experiments.learner import builtin_training_metrics as metrics
from experiments.learner import contextual as c
from experiments.learner import evaluation_behavior as shared
from tests.test_builtin_replay import HistoryFixture, write
from tests.test_contextual_armies import army
from tests.test_contextual_management import observation
from tests.test_evaluation_behavior import Frames


def extras(frames):
    basic, issues = shared._player_metrics(frames.frames)
    assert not issues
    return metrics._extras(frames.frames, basic)


def recruitment_observation():
    payload = c.RecruitmentPanel("dwelling", (c.RecruitChoice("card", 0),), "card", 14,
        (0, 0, 0, 0, 0, 0, 60), 14, c.RecruitDestination("destination", "town_garrison"))
    window = c.Window(c.Context("town", "town", "recruitment", "dwelling"), None,
        town_panel=c.TownPanel("recruitment", recruitment=payload), armies=())
    return replace(observation(), window=window)


class DetailedMetricsTests(unittest.TestCase):
    def test_recruitment_quantity_has_explicit_result_and_display_coverage(self):
        frames = Frames()
        for accepted in (True, False):
            seq = frames.decision(recruitment_observation())
            frames.submit(seq, "recruit", "town", selection="card", count=5, destination="destination")
            frames.add("view", observation=observation("town"))
            frames.add("result", decision_seq=seq, result=c.Result("recruited" if accepted else "rejected", accepted))
        frames.cutoff(observation())
        result = extras(frames)["recruitment"]
        self.assertEqual((result["attempts"], result["successful"], result["successful_nonzero"], result["successful_zero"]), (2, 1, 1, 0))
        self.assertEqual(result["accepted_quantity"], 5)
        self.assertEqual(result["quantity_observed_successes"], 1)
        self.assertEqual(result["displayed_delta_comparisons"], 0)
        self.assertIsNone(result["actions"][1]["accepted_quantity"])
        self.assertTrue(all(row["displayed_creature_delta"] is None for row in result["actions"]))

    def test_cross_army_counts_and_concentration_use_the_action_interval(self):
        before = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, 14)}),
                                                   army("second", "hero", "other", {0: (0, 6)})))
        after = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, 10)}),
                                                  army("second", "hero", "other", {0: (0, 10)})))
        frames = Frames(); seq = frames.decision(before)
        frames.submit(seq, "split_stack", "hero", selection="first-0", destination="second-0", count=4)
        frames.add("view", observation=after)
        frames.add("result", decision_seq=seq, result=c.Result("split", True))
        frames.cutoff(observation())
        action = extras(frames)["armies"]["actions"][0]
        self.assertEqual([row["creature_count"] for row in action["before"]], [14, 6])
        self.assertEqual([row["creature_count"] for row in action["after"]], [10, 10])
        self.assertEqual(action["effect"]["observed_cross_army_net_redistribution"], [{"creature": 0, "count": 4}])
        self.assertEqual(action["count_concentration"], {"subjects": ["hero", "other"], "total_before": 20,
            "total_after": 20, "largest_count_share_before": .7, "largest_count_share_after": .5})

    def test_equal_stack_swap_preserves_counts_and_does_not_invent_net_transfer(self):
        before = observation("exchange", armies=(army("first", "hero", "hero", {0: (0, 12)}),
                                                   army("second", "hero", "other", {0: (0, 12)})))
        frames = Frames(); seq = frames.decision(before)
        frames.submit(seq, "swap_stacks", "hero", selection="first-0", destination="second-0")
        frames.add("view", observation=before)
        frames.add("result", decision_seq=seq, result=c.Result("swapped", True))
        frames.cutoff(before)
        action = extras(frames)["armies"]["actions"][0]
        self.assertEqual(action["effect"]["observed_cross_army_net_redistribution"], [])
        self.assertEqual([row["creature_count"] for row in action["before"]], [12, 12])
        self.assertEqual(action["count_concentration"]["largest_count_share_after"], .5)

    def test_later_panel_is_never_used_as_an_earlier_action_effect(self):
        frames = Frames(); seq = frames.decision(observation("town"))
        frames.submit(seq, "swap_stacks", "town", selection="upper-0", destination="lower-0")
        frames.add("view", observation=observation())
        frames.add("result", decision_seq=seq, result=c.Result("swapped", True))
        frames.cutoff(observation("town"))
        action = extras(frames)["armies"]["actions"][0]
        self.assertEqual((action["before"], action["after"]), ([], []))
        self.assertEqual(action["effect"]["comparison"], "unavailable")
        self.assertIsNone(action["count_concentration"])

    def test_army_first_max_last_are_observations_not_hidden_current_counts(self):
        frames = Frames(); frames.add("view", observation=observation())
        first = frames.add("view", observation=observation("hero", armies=(army("a", "hero", "hero", {0: (0, 10)}),)))
        maximum = frames.add("view", observation=observation("hero", armies=(army("a", "hero", "hero", {0: (0, 20)}),)))
        final = frames.add("view", observation=observation("hero", armies=(army("a", "hero", "hero", {0: (0, 7)}),)))
        frames.cutoff(observation())
        result = extras(frames)["armies"]
        row = result["totals_by_subject"][0]
        self.assertEqual((row["first_observed"]["seq"], row["maximum_observed"]["seq"], row["last_observed"]["seq"]), (first, maximum, final))
        self.assertEqual([row[key]["creature_count"] for key in ("first_observed", "maximum_observed", "last_observed")], [10, 20, 7])
        self.assertFalse(row["current_at_final_observation"])
        self.assertIsNone(row["final_observation"])
        self.assertEqual(result["coverage"]["subjects_never_displayed"], ["other", "town"])
        self.assertFalse(result["coverage"]["whole_owned_army_census"])

    def test_current_final_row_and_non_owned_army_guard(self):
        frames = Frames(); frames.cutoff(observation("hero"))
        result = extras(frames)["armies"]["totals_by_subject"][0]
        self.assertTrue(result["current_at_final_observation"])
        self.assertEqual(result["final_observation"]["creature_count"], 16)
        frames.frames[0]["observation"]["window"]["armies"][0]["subject"] = "enemy"
        with self.assertRaisesRegex(ValueError, "current own entity"):
            extras(frames)

    def test_town_ownership_events_bind_day_sequence_and_capture_corroboration(self):
        frames = Frames(); base = observation(); frames.add("view", observation=base)
        lost = replace(base, towns=()); loss_seq = frames.add("view", observation=lost)
        frames.frames[-1]["day"] = 3
        extra = c.Town("extra", (3, 2, 0))
        addition = replace(lost, towns=(extra,)); frames.add("view", observation=addition)
        enemy = c.ObjectView("captured", (3, 1, 0), 98, 0, "opponent")
        visible = replace(addition, tiles=base.tiles + (c.Tile((3, 1, 0), 1, 0, False, False),), objects=(enemy,))
        frames.add("view", observation=visible)
        frames.add("event", event=c.VisibleEvent("visit", "captured", 98, 0, "opponent"))
        gained = replace(visible, towns=visible.towns + (c.Town("captured", (3, 1, 0)),), objects=(replace(enemy, owner="self"),))
        gain_seq = frames.add("view", observation=gained); frames.frames[-1]["day"] = 4
        frames.cutoff(gained)
        result = extras(frames)["towns"]
        self.assertEqual(result["ownership_losses"], [{"seq": loss_seq, "day": 3, "subject": "town"}])
        self.assertEqual(len(result["roster_additions"]), 2)
        self.assertEqual([(row["subject"], row["seq"], row["day"]) for row in result["corroborated_captures"]], [("captured", gain_seq, 4)])

    def test_repetition_records_exact_choice_runs_without_strategy_labels(self):
        frames = Frames()
        for kind in ("open_hero", "open_hero", "close", "end_turn", "open_hero"):
            seq = frames.decision(observation())
            frames.submit(seq, kind, "hero" if kind == "open_hero" else None)
            frames.add("result", decision_seq=seq, result=c.Result({"open_hero": "opened", "close": "closed", "end_turn": "turn_ended"}[kind], True))
        frames.cutoff(observation())
        self.assertEqual(extras(frames)["repetition"], {"submitted_choices": 5,
            "consecutive_identical_action_repeats": 1, "maximum_consecutive_same_kind": 2,
            "maximum_consecutive_navigation_selection": 3, "maximum_consecutive_army_operations": 0})


class BoundMetricsTests(unittest.TestCase):
    def test_bound_both_seat_histories_and_existing_behavior_recompute_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seat in (0, 1):
                root = Path(tmp) / str(seat); HistoryFixture(root, seat, natural=True)
                write(root / "behavior.json", builtin_behavior.analyze_episode(root))
                result = metrics.per_episode(root)
                self.assertEqual(result["boundary"], "natural_win")
                self.assertEqual(result["learner_player"], seat)
                self.assertEqual(result["details"]["terminal"]["day"], 1)
                self.assertFalse(result["native_replay_verified_by_this_module"])
                self.assertIn("behavior.json", result["bindings"])
                behavior = builtin_behavior.analyze_episode(root); behavior["aggregate"]["choices"] += 1
                write(root / "behavior.json", behavior)
                with self.assertRaisesRegex(ValueError, "Saved behavior differs"):
                    metrics.per_episode(root)

    def test_missing_or_tampered_history_remains_unavailable_or_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); missing = metrics.per_episode(root)
            self.assertEqual(missing["boundary"], "unavailable")
            self.assertIsNone(missing["details"])
            fixture = HistoryFixture(root, 0)
            path = root / fixture.manifest["players"]["red"]["history_v7"]["path"]
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "history bytes changed"):
                metrics.per_episode(root)

    def test_aggregate_does_not_turn_cutoffs_failed_terminals_or_unstarted_into_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for index, (seat, natural, status) in enumerate(((0, True, "passed"), (1, False, "passed"), (0, True, "failed")), 1):
                root = Path(tmp) / str(index); HistoryFixture(root, seat, natural=natural)
                item = metrics.per_episode(root)
                # Fixture uses episode 1; separate schedule IDs are deliberately
                # changed in these report objects only for aggregation tests.
                item["episode"]["episode"] = str(index)
                rows.append({"episode": index, "learner_player": seat, "engine_seed": 99,
                             "status": status, "metrics": item})
            rows.append({"episode": 4, "learner_player": 1, "engine_seed": 99, "status": "not_started", "metrics": None})
            result = metrics.aggregate(rows, phase="monitor-1", checkpoint="weights-a")
            outcomes = result["outcomes"]
            self.assertEqual((outcomes["natural_win"], outcomes["natural_loss"], outcomes["cutoff"], outcomes["failure"], outcomes["not_started"]), (1, 0, 1, 1, 1))
            self.assertEqual((outcomes["attempted"], outcomes["natural_endings"], outcomes["natural_win_rate"]), (3, 1, 1.))
            self.assertEqual(result["passed_attempt_metrics"]["basic"]["episodes"], 2)
            self.assertEqual(result["all_available_prefix_metrics"]["basic"]["episodes"], 3)
            self.assertIsNone(result["by_seat"]["blue"]["natural_win_rate"])
            self.assertFalse(result["best_checkpoint_selection"])
            self.assertTrue(any("stopping bias" in text for text in result["limitations"]))

    def test_aggregate_rejects_wrong_seat_seed_phase_checkpoint_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = HistoryFixture(root, 0, natural=True)
            item = metrics.per_episode(root); item["episode"]["seed"] = 99
            row = {"episode": 1, "learner_player": 0, "engine_seed": 99, "status": "passed", "metrics": item}
            for change in ({"learner_player": 1}, {"engine_seed": True}, {"phase": "training"}, {"checkpoint": "other"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    metrics.aggregate([{**row, **change}], phase="evaluation", checkpoint="frozen")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                metrics.aggregate([row, row], phase="evaluation", checkpoint="frozen")
            altered = deepcopy(row); altered["metrics"]["coverage"]["complete"] = False
            with self.assertRaisesRegex(ValueError, "Passed attempt"):
                metrics.aggregate([altered], phase="evaluation", checkpoint="frozen")

    def test_batch_rejected_verified_histories_are_separate_unscored_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for player in (0, 1):
                root = Path(tmp) / str(player)
                HistoryFixture(root, player, natural=True)
                item = metrics.per_episode(root)
                item['episode']['episode'] = str(player+1)
                rows.append(dict(episode=player+1, learner_player=player, engine_seed=99,
                    status='rejected', batch_admission='rejected', completion_receipt=False,
                    replay_verified=True, metrics=item))
            result = metrics.aggregate(rows, phase='evaluation', checkpoint='frozen')
        self.assertEqual(result['outcomes']['rejected'], 2)
        self.assertEqual(result['outcomes']['failure'], 0)
        self.assertEqual(result['outcomes']['attempted'], 2)
        self.assertEqual(result['outcomes']['natural_endings'], 0)
        self.assertEqual(result['passed_attempt_metrics']['basic']['episodes'], 0)
        self.assertEqual(result['all_available_prefix_metrics']['basic']['episodes'], 2)
        for color in ('red', 'blue'):
            self.assertEqual(result['by_seat'][color]['rejected'], 1)
            self.assertIsNone(result['by_seat'][color]['natural_win_rate'])
        self.assertTrue(all(row['classification'] == 'rejected' for row in result['attempts']))

    def test_wilson_correct_denominator_edges_and_strict_counts(self):
        self.assertIsNone(metrics.wilson(0, 0))
        self.assertAlmostEqual(metrics.wilson(0, 10)["upper"], .2775327998628892)
        self.assertAlmostEqual(metrics.wilson(10, 10)["lower"], .7224672001371107)
        self.assertAlmostEqual(metrics.wilson(5, 10)["lower"], .236593090512564)
        for args in ((True, 2), (1, 0), (0, -1), (1., 2)):
            with self.assertRaises(ValueError): metrics.wilson(*args)
        result = metrics.aggregate([], phase="unstarted", checkpoint="frozen")
        self.assertEqual(result["scheduled"], 0)
        self.assertIsNone(result["outcomes"]["natural_win_rate_wilson_95"])


if __name__ == "__main__":
    unittest.main()
