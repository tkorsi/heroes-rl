"""Offline summarizer tests: no native process, policy, or optimizer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import evaluate_long_dialogs as evaluation
from experiments.learner.evaluation_behavior import analyze_episode
from validation.probes import summarize_long_dialog_evaluation as summary


class LongDialogSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.execution = self.root / "execution"
        self.execution.mkdir()
        for name in ("checkpoint.pt", "client", "replay", "map.vmap", "freshness.json"):
            (self.root / name).write_text(name)
        self.cfg = evaluation.config()
        self.cfg["checkpoint_sha256"] = evaluation.digest(self.root / "checkpoint.pt")
        self.config_patch = patch.object(evaluation, "config", return_value=self.cfg)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.plan = {"schema": evaluation.SCHEMA + "-plan", "configuration": self.cfg,
            "schedule": evaluation.schedule(self.cfg), "execution_output": str(self.execution),
            "checkpoint": evaluation.binding(self.root / "checkpoint.pt"),
            "replay_binary": evaluation.binding(self.root / "replay"),
            "seed_freshness": evaluation.binding(self.root / "freshness.json"),
            "source": {"source_sha256": {}}, "implementation": {},
            "frozen": {"checkpoint_file_sha256": self.cfg["checkpoint_sha256"]},
            "binary": str(self.root / "client"), "native": {"binary_sha256": evaluation.digest(self.root / "client")},
            "runtime": {"replay": evaluation.digest(self.root / "replay")},
            "game_map": str(self.root / "map.vmap"), "map": {"sha256": evaluation.digest(self.root / "map.vmap")},
            "checkpoint_provenance": "random_initialization", "training_performed": False}
        self.plan_path = self.root / "plan.json"
        evaluation.write(self.plan_path, self.plan)
        self.report = {"schema": evaluation.SCHEMA + "-result", "plan_sha256": evaluation.digest(self.plan_path),
            "status": "stopped", "attempts": [evaluation.empty_attempt(r) for r in self.plan["schedule"]],
            "training_performed": False, "optimizer_updates": 0, "optimizer_steps": 0, "training_samples": 0,
            "frozen_before": self.plan["frozen"], "frozen_after_file_sha256": self.cfg["checkpoint_sha256"]}
        self.report_path = self.execution / "report.json"

    def run_summary(self, **kwargs):
        evaluation.write(self.report_path, self.report)
        return summary.summarize(self.plan_path, self.report_path, reference_episodes=kwargs.pop("reference_episodes", ()), **kwargs)

    def test_keeps_eight_rows_unscored_failure_and_missing_evidence(self):
        self.report["attempts"][0].update(status="native_failure", termination_reason="engine_crash")
        result = self.run_summary()
        self.assertEqual(len(result["attempts"]), 8)
        self.assertEqual(result["attempt_counts"], {"native_failure": 1, "not_started": 7})
        self.assertIsNone(result["attempts"][0]["behavior"])
        self.assertEqual(result["attempts"][0]["rewards"], {"red": None, "blue": None})
        self.assertFalse(result["coverage"]["complete"])
        self.assertIn("episode_directory_missing", [x["issue"] for x in result["coverage"]["issues"]])
        self.assertIsNone(result["behavior_summary"]["aggregate"])

    def test_stopped_all_not_started_is_not_complete(self):
        result = self.run_summary()
        self.assertEqual(result["attempt_counts"], {"not_started": 8})
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["behavior_summary"]["episodes"], 0)

    def test_partial_directory_and_reference_stay_incomplete(self):
        folder = self.execution / "episode-000001"
        folder.mkdir()
        self.report["attempts"][0].update(status="failed", episode_directory=str(folder))
        result = self.run_summary(reference_episodes=(self.root / "old-1", self.root / "old-2"))
        self.assertEqual(result["attempts"][0]["behavior"]["status"], "incomplete")
        self.assertIsNone(result["attempts"][0]["behavior"]["aggregate"])
        self.assertEqual(result["earlier_ordinary_comparison"]["summary"]["incomplete_episodes"], 2)
        self.assertEqual(result["earlier_ordinary_comparison"]["new_replays"], 0)

    def test_binding_corruption_and_report_schedule_drift_raise(self):
        (self.root / "checkpoint.pt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Bound artifact changed"):
            self.run_summary()
        (self.root / "checkpoint.pt").write_text("checkpoint.pt")
        self.report["attempts"][4]["policy_seeds"] = self.report["attempts"][4]["policy_seeds"][::-1]
        with self.assertRaisesRegex(ValueError, "fixed schedule"):
            self.run_summary()

    def test_plan_hash_and_versioned_schedule_are_enforced(self):
        self.plan_path.write_text(self.plan_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            self.run_summary()
        self.plan["schedule"][0]["engine_seed"] = 123
        evaluation.write(self.plan_path, self.plan)
        self.report["plan_sha256"] = evaluation.digest(self.plan_path)
        with self.assertRaisesRegex(ValueError, "scope or output changed"):
            self.run_summary()

    def test_unverified_scores_and_training_are_rejected(self):
        self.report["attempts"][0]["rewards"]["red"] = 0
        with self.assertRaisesRegex(ValueError, "received a score"):
            self.run_summary()
        self.report["attempts"][0]["rewards"]["red"] = None
        self.report["optimizer_updates"] = 1
        with self.assertRaisesRegex(ValueError, "training activity"):
            self.run_summary()

    def test_raw_binding_excludes_temp_and_own_output(self):
        (self.execution / "worker-01.log").write_text("retained failure")
        (self.execution / "report.json.tmp-42").write_text("partial")
        output = self.execution / "summary.json"
        output.write_text("old output is excluded only by explicit path")
        result = self.run_summary(output_path=output)
        self.assertIn(str(self.execution / "worker-01.log"), result["bindings"])
        self.assertNotIn(str(output), result["bindings"])
        self.assertNotIn(str(self.execution / "report.json.tmp-42"), result["bindings"])
        self.assertEqual(result["plan"], self.plan)

    def test_compaction_discards_detailed_arrays_but_retains_metrics(self):
        value = analyze_episode(self.root / "missing")
        value["players"]["red"]["metrics"] = {"choices": 8, "movement": {"attempts": 3},
            "army_effects": [{"secretly_large": [1, 2]}], "movement_attempts": [{"x": 5}],
            "town_capture_evidence": [{"seq": 3}], "battles": {"starts": 1, "completions": 1, "visible_results": [{}]}}
        compact = summary.compact_behavior(value)["players"]["red"]["metrics"]
        self.assertEqual(compact["choices"], 8)
        self.assertEqual(compact["movement"], {"attempts": 3})
        self.assertEqual(compact["counts"]["town_capture_evidence"], 1)
        self.assertNotIn("army_effects", compact)
        self.assertNotIn("movement_attempts", compact)
        self.assertEqual(compact["battles"], {"starts": 1, "completions": 1})

    def test_saved_behavior_hash_is_verified_before_comparison(self):
        folder = self.execution / "episode-000001"
        folder.mkdir()
        behavior = folder / "behavior.json"
        evaluation.write(behavior, analyze_episode(folder))
        self.report["attempts"][0].update(status="failed", episode_directory=str(folder), behavior=evaluation.binding(behavior))
        behavior.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Bound artifact changed"):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
