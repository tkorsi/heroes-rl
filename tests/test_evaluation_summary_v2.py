"""Versioned offline evidence retains originals and never executes a model/game."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.learner import evaluate_builtin_opponent as builtin
from experiments.learner import evaluate_long_dialogs as neural
from experiments.learner import evaluation_revalidation as revalidation
from tests import test_builtin_summary as fixtures
from validation.probes import summarize_builtin_evaluation as comparison
from validation.probes import summarize_long_dialog_evaluation as summary


class RevalidationSummaryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.BuiltinSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        for label, module in (("neural", neural), ("builtin", builtin)):
            cfg = fixture.plans[label]["configuration"]
            cfg.update(checkpoint_sha256=neural.digest(self.root / "checkpoint"),
                map_sha256=neural.digest(self.root / "map"))
            config_patch = patch.object(module, "config", return_value=cfg)
            config_patch.start()
            self.addCleanup(config_patch.stop)
        self.plans, self.reports, self.paths = {}, {}, {}
        self.original_bindings = []
        for label, module in (("neural", neural), ("builtin", builtin)):
            old_plan = fixture.plans[label]
            old_plan.update(training_performed=False, data=str(self.root / "data"),
                checkpoint_source={"interface": neural.INTERFACE}, checkpoint_provenance="random_initialization")
            # These historical files intentionally no longer exist at HEAD.
            old_plan["source"] = {"source_sha256": {"retired/source.py": "a" * 64}}
            old_plan["implementation"] = {"retired/runner.py": "b" * 64}
            if label == "builtin":
                old_plan["reference"] = builtin.reference(self.root / "neural/plan.json",
                    self.root / "neural/execution/report.json")
            fixture.save(label)
            old_path = self.root / label / "plan.json"
            old_report = old_path.parent / "execution/report.json"
            prior = revalidation.prior_cohort(old_path, old_report,
                reader=module.read_plan, legacy_schema=module.SCHEMA)
            self.original_bindings.extend((prior["plan"], prior["report"]))
            plan = copy.deepcopy(old_plan)
            path = self.root / (label + "-v2") / "plan.json"
            (path.parent / "execution").mkdir(parents=True)
            plan.update(schema=module.V2_SCHEMA + "-plan", execution_output=str(path.parent / "execution"),
                authorization=dict(revalidation.AUTHORIZATION),
                revalidation={**revalidation.PROVENANCE, "prior": prior},
                source={"source_sha256": {}, "tracked_worktree_dirty": False}, implementation={})
            plan.pop("seed_freshness", None)
            self.paths[label], self.plans[label] = path, plan
            self.reports[label] = copy.deepcopy(fixture.reports[label])
            self.reports[label]["schema"] = module.V2_SCHEMA + "-result"
            if label == "builtin":
                plan["reference"] = builtin.reference(self.paths["neural"],
                    self.paths["neural"].parent / "execution/report.json")
            self.save(label)

    def save(self, label):
        module = builtin if label == "builtin" else neural
        path = self.paths[label]
        module.write(path, self.plans[label])
        self.reports[label]["plan_sha256"] = module.digest(path)
        module.write(path.parent / "execution/report.json", self.reports[label])

    def run_neural(self):
        self.save("neural")
        return summary.summarize(self.paths["neural"], self.paths["neural"].parent / "execution/report.json",
            reference_episodes=())

    def run_comparison(self):
        self.save("neural")
        self.plans["builtin"]["reference"] = builtin.reference(self.paths["neural"],
            self.paths["neural"].parent / "execution/report.json")
        self.save("builtin")
        return comparison.summarize(self.paths["builtin"], self.paths["builtin"].parent / "execution/report.json")

    def test_neural_v2_preserves_prior_and_reused_seeds_without_freshness(self):
        result = self.run_neural()
        self.assertEqual(result["schema"], "heroes-long-dialog-evaluation-evidence-v2")
        self.assertEqual(result["attempt_counts"], {"not_started": 8})
        self.assertFalse(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["issues"], [])
        self.assertFalse(result["plan"]["revalidation"]["held_out"])
        self.assertEqual(result["plan"]["revalidation"]["seed_use"], "reused_matched_seeds")
        self.assertNotIn("seed_freshness", result["plan"])
        for item in self.original_bindings[:2]:
            self.assertEqual(result["bindings"][item["path"]]["sha256"], item["sha256"])
            self.assertEqual(neural.binding(item["path"]), item)

    def test_comparison_v2_binds_both_original_cohorts_and_separate_budgets(self):
        result = self.run_comparison()
        self.assertEqual(result["schema"], "heroes-builtin-opponent-comparison-evidence-v2")
        self.assertEqual(len(result["matched_pairs"]), 8)
        self.assertFalse(result["coverage"]["all_planned_games_verified"])
        for item in self.original_bindings:
            self.assertEqual(result["bindings"][item["path"]]["sha256"], item["sha256"])
            self.assertEqual(neural.binding(item["path"]), item)
        for label, cap in (("frozen_neural_opponent", 4096), ("builtin_nullkiller_easy", 2048)):
            cohort = result["cohorts"][label]
            self.assertEqual(cohort["categories"], {"not_started": 8})
            self.assertEqual(cohort["configuration"]["max_total_choices"], cap)
            self.assertEqual(cohort["configuration"]["shared_wall_seconds"], 3600)
            self.assertFalse(cohort["revalidation"]["held_out"])
            self.assertIsNone(cohort["learner_behavior"])

    def test_partial_failed_rows_and_phase_times_are_retained_without_scores(self):
        durations = {"initialization": 0.25, "collection": 1.5, "cleanup": 0.75}
        for label in ("neural", "builtin"):
            folder = self.paths[label].parent / "execution/episode-000001"
            folder.mkdir()
            self.reports[label]["attempts"][0].update(status="failed", episode_directory=str(folder),
                wall_seconds=2.5, phase_wall_seconds=durations, sampled_choices=0)
            self.reports[label].update(execution_wall_seconds=4.0, offline_report_seconds=0.5)
        neural_result = self.run_neural()
        self.assertEqual(neural_result["attempts"][0]["phase_wall_seconds"], durations)
        self.assertEqual(neural_result["execution"]["offline_report_seconds"], 0.5)
        result = self.run_comparison()
        for cohort in result["cohorts"].values():
            self.assertEqual(cohort["categories"], {"failed": 1, "not_started": 7})
            self.assertEqual(cohort["attempts"][0]["phase_wall_seconds"], durations)
            self.assertEqual(cohort["attempts"][0]["rewards"], {"red": None, "blue": None})
            self.assertEqual(cohort["execution_wall_seconds"], 4.0)
            self.assertEqual(cohort["offline_report_seconds"], 0.5)
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["training_samples"], 0)

    def test_report_version_must_match_its_validated_plan(self):
        for label, module, run in (("neural", neural, self.run_neural),
                                   ("builtin", builtin, self.run_comparison)):
            with self.subTest(label=label):
                self.reports[label]["schema"] = module.SCHEMA + "-result"
                with self.assertRaisesRegex(ValueError, "Unexpected .*report"):
                    run()
                self.reports[label]["schema"] = module.V2_SCHEMA + "-result"

    def test_changed_prior_report_is_not_rebound_or_ignored(self):
        old = Path(self.original_bindings[1]["path"])
        old.write_text(old.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Preserved prior cohort artifacts changed"):
            self.run_neural()

    def test_v2_cannot_claim_fresh_or_held_out_seeds(self):
        self.plans["neural"]["seed_freshness"] = self.original_bindings[0]
        with self.assertRaisesRegex(ValueError, "cannot claim new seed freshness"):
            self.run_neural()
        del self.plans["neural"]["seed_freshness"]
        self.plans["neural"]["revalidation"]["held_out"] = True
        with self.assertRaisesRegex(ValueError, "seed or preservation claim changed"):
            self.run_neural()

    def test_v1_schema_remains_v1_and_does_not_require_revalidation(self):
        result = comparison.summarize(self.root / "builtin/plan.json",
            self.root / "builtin/execution/report.json")
        # The old source is missing, explicitly reported rather than matched to
        # the new v2 implementation; no v2 provenance is manufactured for it.
        self.assertEqual(result["schema"], comparison.SCHEMA)
        self.assertNotIn("revalidation", result["plan"])
        self.assertTrue(result["coverage"]["issues"])


if __name__ == "__main__":
    unittest.main()
