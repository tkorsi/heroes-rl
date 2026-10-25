"""Coverage aggregation preserves failures and randomized missed branches."""
from copy import deepcopy
import unittest

from validation.probes import summarize_capabilities as summary


def full_report():
    report = {"report_path": "synthetic-report", "implementation_sha256": "same-source", "runtime_sha256": "same-runtime",
            "status": "passed", "not_started": 0,
            "rows": [{"case": case, "player": player, "status": "passed"}
                     for case in summary.CASES for player in ("red", "blue")]}
    for row in report["rows"]:
        if row["case"].startswith("dimension-door-"):
            blocked = row["case"] == "dimension-door-blocked"
            row.update(seed=17001, repeat=0, checks=[{"check": "native_dimension_door_fog_attempt", "status": "passed",
                       "blocked_fixture": blocked, "target": [4, 18], "player_view_before_sha256": "same-view",
                       "offered_spell_targets_sha256": "same-targets", "offered_spell_target_count": 224,
                       "result_status": "rejected" if blocked else "capability_applied"}])
    return report


class CapabilitySummaryTests(unittest.TestCase):
    def test_every_feature_and_both_players_are_required(self):
        report = full_report()
        self.assertTrue(summary.summarize([report])["native_coverage_complete"])
        report["rows"].pop()
        self.assertFalse(summary.summarize([report])["native_coverage_complete"])

    def test_retained_random_branch_misses_do_not_erase_an_actual_witness(self):
        report = full_report()
        report["status"] = "unverified"
        report["rows"].insert(0, {"case": "eagle-eye", "player": "red", "status": "unverified"})
        result = summary.summarize([report])
        self.assertTrue(result["native_coverage_complete"])
        self.assertEqual(result["unverified_attempts"], 1)
        self.assertEqual(len(result["reports"][0]["rows"]), len(summary.CASES) * 2 + 1)

    def test_a_failed_or_unstarted_attempt_keeps_the_gate_closed(self):
        for change in ({"status": "failed"}, {"not_started": 1}):
            report = full_report()
            report.update(change)
            self.assertFalse(summary.summarize([report])["native_coverage_complete"])

    def test_old_and_new_runtime_evidence_cannot_be_combined(self):
        first = full_report()
        second = deepcopy(first)
        second.update(report_path="other", runtime_sha256="rebuilt-runtime")
        result = summary.summarize([first, second])
        self.assertFalse(result["native_coverage_complete"])
        self.assertFalse(result["one_frozen_implementation_and_runtime"])
        with self.assertRaises(ValueError):
            summary.summarize([first, first])

    def test_hidden_layout_candidate_difference_closes_the_gate(self):
        report = full_report()
        row = next(row for row in report["rows"] if row["case"] == "dimension-door-blocked" and row["player"] == "red")
        row["checks"][0]["offered_spell_targets_sha256"] = "leaks-hidden-blockage"
        self.assertFalse(summary.summarize([report])["native_coverage_complete"])


if __name__ == "__main__":
    unittest.main()
