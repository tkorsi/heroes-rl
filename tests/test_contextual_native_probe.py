"""Pure evidence-comparison tests; no engine process or simulated game adapter."""
from copy import deepcopy
import unittest

from experiments.learner.contextual import Frame, Result, VisibleEvent
from validation.probes import validate_contextual_native as probe


def record(seq, kind, **payload):
    return Frame("evidence", 0, seq, 1, kind, **payload)


def native_ack(seq, accepted):
    return record(seq, "acknowledged", decision_seq=0, command_index=0, accepted=accepted)


def native_issue(seq, index=0):
    return record(seq, "issued", decision_seq=0, command_index=index)


def route_result(seq, status):
    result = Result(status, True, (2, 1, 0), 900, False)
    return record(seq, "result", decision_seq=0, result=result), result


def observed_prefix():
    return [{"episode": "a", "player": 0, "seq": 0, "day": 1, "kind": "view",
             "observation": {"size": [5, 5, 1],
                             "tiles": [{"position": [x, 1, 0], "terrain": 2, "road": 0,
                                        "blocked": False, "visitable": False} for x in (1, 2, 3)],
                             "objects": [], "resources": [0] * 7,
                             "heroes": [{"handle": "own-a", "position": [1, 1, 0], "movement": 1000}],
                             "towns": [], "window": None}}]


def add_fixture_opponent(prefix, position):
    prefix[-1]["observation"]["objects"].append(
        {"handle": "seen-enemy", "position": list(position), "owner": "opponent", "kind": 34, "subtype": 0})


class ContextualNativeProbeTests(unittest.TestCase):
    def test_handle_and_episode_renaming_preserves_permitted_comparison(self):
        left = observed_prefix()
        right = deepcopy(left)
        right[0]["episode"] = "b"
        right[0]["observation"]["heroes"][0]["handle"] = "own-b"
        self.assertEqual(probe.compare_permitted_prefixes(left, right, set())["status"], "passed")

    def test_own_state_or_native_metadata_difference_cannot_be_normalized_away(self):
        for field in ("resources", "position", "movement", "seq", "day"):
            with self.subTest(field=field):
                left = observed_prefix()
                right = deepcopy(left)
                if field in ("seq", "day"):
                    right[0][field] += 1
                elif field == "resources":
                    right[0]["observation"][field][0] = 1
                elif field == "position":
                    right[0]["observation"]["heroes"][0][field] = [2, 1, 0]
                else:
                    right[0]["observation"]["heroes"][0][field] = 900
                self.assertEqual(probe.compare_permitted_prefixes(left, right, set())["status"], "failed")

    def test_legitimate_fixture_visibility_stops_comparison_without_passing_it(self):
        left, right = observed_prefix(), observed_prefix()
        add_fixture_opponent(left, (2, 1, 0))
        compared = probe.compare_permitted_prefixes(left, right, {(2, 1, 0), (3, 1, 0)})
        self.assertEqual(compared["status"], "unverified")
        self.assertTrue(compared["comparison_stopped_at_legitimate_visibility_difference"])

    def test_fixture_visibility_does_not_excuse_changed_own_route(self):
        left, right = observed_prefix(), observed_prefix()
        add_fixture_opponent(left, (2, 1, 0))
        left[0]["observation"]["heroes"][0]["position"] = [2, 1, 0]
        self.assertEqual(probe.compare_permitted_prefixes(left, right, {(2, 1, 0)})["status"], "failed")

    def test_missing_history_tail_cannot_pass_equal_prefix(self):
        left = observed_prefix()
        right = deepcopy(left)
        right.append({"episode": "a", "player": 0, "seq": 1, "day": 1, "kind": "issued",
                      "decision_seq": 0, "command_index": 0})
        self.assertEqual(probe.compare_permitted_prefixes(left, right, set())["status"], "failed")

    def test_encounter_label_requires_event_evidence(self):
        final, result = route_result(3, "interaction")
        evidence = probe.route_stop_evidence([native_issue(1), native_ack(2, True), final], result)
        self.assertEqual(evidence["encounter"]["status"], "unverified")

    def test_encounter_event_followed_by_no_further_command_passes(self):
        final, result = route_result(4, "interaction")
        frames = [native_issue(1), record(2, "event", event=VisibleEvent("visit")), native_ack(3, True), final]
        self.assertEqual(probe.route_stop_evidence(frames, result)["encounter"]["status"], "passed")

    def test_native_command_after_encounter_is_failed_evidence(self):
        final, result = route_result(6, "interaction")
        frames = [native_issue(1), record(2, "event", event=VisibleEvent("battle_start")),
                  native_ack(3, True), native_issue(4, 1),
                  record(5, "acknowledged", decision_seq=0, command_index=1, accepted=True), final]
        self.assertEqual(probe.route_stop_evidence(frames, result)["encounter"]["status"], "failed")

    def test_interruption_label_requires_rejected_native_acknowledgement(self):
        final, result = route_result(3, "interrupted")
        for accepted, expected in ((True, "unverified"), (False, "passed")):
            with self.subTest(accepted=accepted):
                frames = [native_issue(1), native_ack(2, accepted), final]
                self.assertEqual(probe.route_stop_evidence(frames, result)["interruption"]["status"], expected)

    def test_native_command_after_rejection_is_failed_evidence(self):
        final, result = route_result(4, "interrupted")
        frames = [native_issue(1), native_ack(2, False), native_issue(3, 1), final]
        self.assertEqual(probe.route_stop_evidence(frames, result)["interruption"]["status"], "failed")

    def test_natural_gate_counts_only_successfully_audited_random_runs(self):
        base = {"case": "random-001", "episode_status": "completed", "probe_status": "passed",
                "history_audit_passed": True}
        rows = [base, {**base, "probe_status": "failed"}, {**base, "probe_status": "unverified"},
                {**base, "history_audit_passed": False}, {**base, "episode_status": "truncated"},
                {**base, "case": "mechanics"}]
        self.assertEqual(probe.validated_natural_count(rows), 1)

    def test_paired_comparison_cannot_pass_failed_or_unaudited_native_runs(self):
        passed = {"probe_status": "passed", "history_audit_passed": True}
        comparisons = [{"status": "passed"}, {"status": "passed"}]
        for overrides, expected in (({}, "passed"), ({"probe_status": "failed"}, "failed"),
                                    ({"history_audit_passed": False}, "unverified"),
                                    ({"probe_status": "unverified"}, "unverified")):
            with self.subTest(overrides=overrides):
                self.assertEqual(probe.paired_comparison_status([passed, {**passed, **overrides}], comparisons), expected)
        self.assertEqual(probe.paired_comparison_status([passed, passed], [{"status": "failed"}]), "failed")
        self.assertEqual(probe.paired_comparison_status([passed, passed], [{"status": "unverified"}]), "unverified")


if __name__ == "__main__":
    unittest.main()
