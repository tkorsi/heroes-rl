"""Pure cohort planning/reporting checks; no native game or simulated engine."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
else:
    from experiments.headless.env import Limits
    from experiments.learner.checkpoints import FrozenOpponentPool
    from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, weights_sha256
    from experiments.learner.inference import (
        BASELINE, BASELINE_SHA256, attempt_category, attempt_roster, diagnostic_summary,
        ordinary_map_identity, ordinary_summary,
    )
    from validation.probes.create_boundary_fixture import create_fixture


def attempt(status, seat=0, *, reason=None, winner="red"):
    outcomes = {color: "win" if color == winner else "loss" for color in ("red", "blue")} if status == "completed" else dict.fromkeys(("red", "blue"))
    rewards = {color: 1 if result == "win" else -1 for color, result in outcomes.items()} if status == "completed" else dict.fromkeys(("red", "blue"))
    return {"status": status, "learner_player": seat, "outcomes": outcomes, "rewards": rewards,
            "audit_consistent": True, "weights_unchanged": True,
            "termination_reason": reason or {"completed": "natural_terminal", "truncated": "collection_window",
                                               "failed": "engine_error", "collection_error": "collection_error"}[status]}


@unittest.skipIf(torch is None, "Optional PyTorch not installed")
class ContextualInferenceTests(unittest.TestCase):
    def test_all_attempt_denominator_and_completed_only_outcomes_are_distinct(self):
        rows = [attempt("completed", 0), attempt("completed", 1),
                attempt("truncated", 0), attempt("failed", 1, reason="unsupported_interface"),
                attempt("failed", 0, reason="process_crash"), attempt("collection_error", 1)]
        summary = ordinary_summary(rows, planned_attempts=6)
        self.assertEqual(summary["attempts"], 6)
        self.assertEqual(summary["finished_attempts"], 6)
        self.assertEqual(summary["not_started"], 0)
        self.assertEqual(summary["natural_outcome_frequency"], {"numerator": 2, "denominator": 6, "value": 1 / 3})
        self.assertEqual(summary["learner_wins_among_completed"], {"numerator": 1, "denominator": 2, "value": .5})
        for name, expected in (("verified_natural", 2), ("verified_cutoff", 1), ("unsupported_interface", 1),
                               ("technical_failure", 1), ("collection_or_audit_error", 1)):
            self.assertEqual(summary[name], expected)
        self.assertEqual(summary["by_learner_seat"]["red"]["learner_wins"], 1)
        self.assertEqual(summary["by_learner_seat"]["blue"]["learner_losses"], 1)
        self.assertEqual(summary["by_learner_seat"]["blue"]["unsupported_interface"], 1)
        self.assertEqual(summary["outcomes_by_player"], {"red": {"win": 2, "loss": 0}, "blue": {"win": 0, "loss": 2}})
        self.assertEqual(summary["training_samples"], 0)
        self.assertEqual(summary["optimizer_updates"], 0)

    def test_no_completed_games_and_incomplete_roster_never_create_draws(self):
        empty = ordinary_summary([], planned_attempts=4)
        self.assertIsNone(empty["natural_outcome_frequency"]["value"])
        self.assertEqual(empty["not_started"], 4)
        rows = [attempt("truncated"), {"status": "running", "learner_player": 1}]
        summary = ordinary_summary(rows, planned_attempts=4)
        self.assertEqual(summary["natural_outcome_frequency"], {"numerator": 0, "denominator": 2, "value": 0})
        self.assertEqual(summary["learner_wins_among_completed"], {"numerator": 0, "denominator": 0, "value": None})
        self.assertEqual((summary["in_progress"], summary["finished_attempts"], summary["not_started"]), (1, 1, 2))
        self.assertEqual(summary["learner_losses"], 0)

    def test_unaudited_or_contradictory_outcomes_are_collection_errors(self):
        good = attempt("completed")
        bad = []
        for field in ("audit_consistent", "weights_unchanged"):
            row = deepcopy(good); row[field] = False; bad.append(row)
        row = deepcopy(good); row["outcomes"]["blue"] = "win"; bad.append(row)
        row = deepcopy(good); row["rewards"]["red"] = True; bad.append(row)
        row = deepcopy(good); row["status"] = "collection_error"; bad.append(row)
        row = attempt("truncated"); row["rewards"]["red"] = 0; bad.append(row)
        self.assertEqual([attempt_category(row) for row in bad], ["collection_or_audit_error"] * len(bad))
        self.assertEqual(ordinary_summary(bad, planned_attempts=len(bad))["verified_natural"], 0)

    def test_default_diagnostic_summary_keeps_rates_disabled(self):
        summary = diagnostic_summary([attempt("completed"), attempt("truncated"),
                                      attempt("failed", reason="unsupported_interface"), attempt("collection_error")])
        self.assertIsNone(summary["outcome_rate"])
        self.assertEqual((summary["attempts"], summary["verified_natural"], summary["verified_cutoff"],
                          summary["technical_failure"], summary["collection_or_audit_error"]), (4, 1, 1, 1, 1))

    def test_roster_pins_both_seats_seeds_checkpoints_and_actual_limits(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng():
                torch.manual_seed(201)
                model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
            pool = FrozenOpponentPool(); pool.add("frozen", model)
            limits = Limits(max_days=100, max_actions=2049)
            rows = attempt_roster([(96101, "frozen"), (96102, "frozen")], learner_id="initial",
                                  learner_digest=weights_sha256(model), pool=pool, policy_seed=96003,
                                  max_requests=2048, limits=limits)
            self.assertEqual([(row["episode"], row["engine_seed"], row["learner_player"]) for row in rows],
                             [(1, 96101, 0), (2, 96101, 1), (3, 96102, 0), (4, 96102, 1)])
            self.assertEqual([row["policy_seeds"] for row in rows], [[96005, 96006], [96007, 96008], [96009, 96010], [96011, 96012]])
            for row in rows:
                self.assertEqual(row["learner_checkpoint"], "initial")
                self.assertEqual(row["opponent_checkpoint"], "frozen")
                self.assertEqual(row["learner_weights_sha256"], weights_sha256(model))
                self.assertEqual(row["opponent_weights_sha256"], weights_sha256(model))
                self.assertEqual(row["budgets"]["max_requests"], 2048)
                self.assertEqual(row["budgets"]["environment_limits"]["max_actions"], 2049)
                self.assertEqual(row["budgets"]["environment_limits"]["max_days"], 100)
                self.assertEqual(row["budgets"]["environment_limits"]["watchdog_seconds"], 60)
        finally:
            torch.set_num_threads(threads)

    def test_ordinary_map_requires_exact_baseline_bytes_and_semantics(self):
        identity = ordinary_map_identity(BASELINE)
        self.assertEqual(identity["sha256"], BASELINE_SHA256)
        self.assertTrue(identity["semantics_verified"])
        self.assertFalse(identity["fixture"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            copy = root / "copy.vmap"; copy.write_bytes(BASELINE.read_bytes())
            self.assertEqual(ordinary_map_identity(copy)["sha256"], BASELINE_SHA256)
            create_fixture(root / "fixture", kind="natural-townless")
            with self.assertRaisesRegex(ValueError, "exact unchanged"):
                ordinary_map_identity(root / "fixture/natural-townless.vmap")
            copy.write_bytes(copy.read_bytes() + b"unexpected archive tail")
            with self.assertRaisesRegex(ValueError, "exact unchanged"):
                ordinary_map_identity(copy)


if __name__ == "__main__":
    unittest.main()
