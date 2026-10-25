"""Comparison evidence must retain failures and use only the learner's seat."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner import evaluate_builtin_opponent as e
from experiments.learner import evaluate_long_dialogs as n
from validation.probes import summarize_builtin_evaluation as s
from tests.test_evaluation_behavior import Frames, observation, write_episode


class BuiltinSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("checkpoint", "vcmiclient", "vcmireplay", "map"):
            (self.root / name).write_text(name)
        self.plans, self.reports = {}, {}
        for label, module in (("neural", n), ("builtin", e)):
            output = self.root / label
            (output / "execution").mkdir(parents=True)
            plan = {"schema": module.SCHEMA + "-plan", "configuration": module.config(),
                "schedule": module.schedule(module.config()), "execution_output": str(output / "execution"),
                "checkpoint": e.binding(self.root / "checkpoint"), "replay_binary": e.binding(self.root / "vcmireplay"),
                "frozen": {}, "source": {"source_sha256": {}}, "implementation": {},
                "binary": str(self.root / "vcmiclient"), "native": {"binary_sha256": e.digest(self.root / "vcmiclient")},
                "runtime": {"vcmireplay": e.digest(self.root / "vcmireplay")},
                "game_map": str(self.root / "map"), "map": {"sha256": e.digest(self.root / "map")},
                "choice_scope": "explicit approximate half-budget", "replay_scope": "captured native inputs"}
            report = {"schema": module.SCHEMA + "-result", "status": "stopped", "training_performed": False,
                "optimizer_updates": 0, "optimizer_steps": 0, "training_samples": 0, "frozen_before": {},
                "frozen_after_file_sha256": plan["checkpoint"]["sha256"],
                "attempts": [module.empty_attempt(row) for row in plan["schedule"]]}
            self.plans[label], self.reports[label] = plan, report
        self.save("neural")
        self.plans["builtin"]["reference"] = e.reference(self.root / "neural/plan.json", self.root / "neural/execution/report.json")

    def save(self, label):
        module = e if label == "builtin" else n
        path = self.root / label / "plan.json"
        module.write(path, self.plans[label])
        self.reports[label]["plan_sha256"] = module.digest(path)
        module.write(path.parent / "execution/report.json", self.reports[label])

    def run_summary(self):
        self.save("neural")
        self.plans["builtin"]["reference"] = e.reference(self.root / "neural/plan.json", self.root / "neural/execution/report.json")
        self.save("builtin")
        return s.summarize(self.root / "builtin/plan.json", self.root / "builtin/execution/report.json")

    def test_all_scheduled_rows_survive_and_unstarted_is_not_complete(self):
        result = self.run_summary()
        self.assertFalse(result["coverage"]["all_planned_games_verified"])
        self.assertEqual(len(result["matched_pairs"]), 8)
        for cohort in result["cohorts"].values():
            self.assertEqual(cohort["categories"], {"not_started": 8})
            self.assertEqual(cohort["started"], 0)
            self.assertIsNone(cohort["learner_behavior"])

    def test_failures_stay_unscored_and_missing_prefix_stays_unavailable(self):
        self.reports["builtin"]["attempts"][0].update(status="native_failure", termination_reason="engine_crash")
        result = self.run_summary()
        cohort = result["cohorts"]["builtin_nullkiller_easy"]
        self.assertEqual(cohort["categories"], {"native_failure": 1, "not_started": 7})
        self.assertIsNone(cohort["attempts"][0]["learner_behavior"])
        self.assertEqual(cohort["attempts"][0]["rewards"], {"red": None, "blue": None})
        self.assertEqual(result["coverage"]["issues"][0]["issue"], "episode_directory_missing")

    def test_neural_metrics_exclude_frozen_opponents_choices(self):
        folder = self.root / "neural/execution/episode-000001"
        red, blue = Frames(0), Frames(1)
        red.cutoff(observation())
        for _ in range(3):
            decision = blue.decision(observation())
            blue.submit(decision, "end_turn")
        blue.cutoff(observation())
        manifest = write_episode(folder, red.frames, blue.frames, status="failed", complete=False)
        manifest["seed"] = self.plans["neural"]["schedule"][0]["engine_seed"]
        (folder / "manifest.json").write_text(json.dumps(manifest))
        self.reports["neural"]["attempts"][0].update(status="failed", episode_directory=str(folder))
        result = self.run_summary()
        aggregate = result["cohorts"]["frozen_neural_opponent"]["learner_behavior"]
        self.assertEqual(aggregate["choices"], 0)
        self.assertEqual(aggregate["available_player_histories"], 1)
        self.assertEqual(aggregate["expected_player_histories"], 1)

    def test_scores_and_training_cannot_be_added_to_failed_attempt(self):
        self.reports["builtin"]["attempts"][0]["rewards"]["red"] = 0
        with self.assertRaisesRegex(ValueError, "received a score"):
            self.run_summary()
        self.reports["builtin"]["attempts"][0]["rewards"]["red"] = None
        self.reports["builtin"]["training_samples"] = 1
        with self.assertRaisesRegex(ValueError, "reports training"):
            self.run_summary()

    def test_hash_drift_and_fixed_schedule_drift_are_rejected(self):
        (self.root / "checkpoint").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Bound artifact changed"):
            self.run_summary()
        (self.root / "checkpoint").write_text("checkpoint")
        self.reports["builtin"]["attempts"][1]["policy_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Fixed schedule changed"):
            self.run_summary()

    def test_running_reference_cannot_launch_comparison(self):
        self.reports["neural"]["status"] = "running"
        self.save("neural")
        with self.assertRaisesRegex(ValueError, "Finish the neural cohort"):
            e.reference(self.root / "neural/plan.json", self.root / "neural/execution/report.json")

    def test_category_requires_actual_audited_complementary_terminal(self):
        row = {**e.empty_attempt(e.schedule(e.config())[1]), "status": "passed", "replay_verified": True,
            "weights_unchanged": True, "termination_reason": "natural_terminal",
            "outcomes": {"red": "win", "blue": "loss"}, "rewards": {"red": 1, "blue": -1}}
        self.assertEqual(s.category(row), "natural_loss")
        for mutate in (lambda x: x.update(replay_verified=False),
                       lambda x: x["rewards"].update(blue=0),
                       lambda x: x["rewards"].update(red=True),
                       lambda x: x["outcomes"].update(blue="win")):
            bad = copy.deepcopy(row); mutate(bad)
            with self.assertRaises(ValueError):
                s.category(bad)


if __name__ == "__main__":
    unittest.main()
