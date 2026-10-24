"""Transport-binding and tamper-generator checks, without a fake replay engine."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.headless import contextual_replay as replay


class ContextualReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {"episode": "1", "environment": "adventure-town-v1", "status": "completed"}
        self.tape = [{"kind": "init", "schema": replay.SCHEMA, "episode": "1", "interface": "adventure-town-v1",
                      "protocol_version": 5, "history_schema": "heroes-player-history-v3",
                      "action_schema": "heroes-context-actions-v3"}]
        for player, color in enumerate(("red", "blue")):
            folder = self.root / "agents" / color
            folder.mkdir(parents=True)
            message = {"player": player, "records": [{"kind": "view", "observation": {"tiles": [{"road": 0}],
                       "window": {"context": {"kind": "town"}}}}]}
            response = {"player": player, "action": {"kind": "recruit", "count": 3, "destination": "town"}}
            self.tape.extend([{"kind": "enter", "actor": str(player), "operation": "worker"},
                              {"kind": "message", "player": player, "message": message},
                              {"kind": "response", "player": player, "response": response},
                              {"kind": "request", "player": player, "request_id": player, "pack_type": 2,
                               "pack": "123456", "source": "adapter"}, {"kind": "leave"}])
            rows = [{"direction": "engine", "message": message},
                    {"direction": "policy", "message": {**response, "delivered": True, "outside_learner_prefix": False}}]
            (folder / "bridge.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.tape.append({"kind": "close", "barrier_id": 1, "client_packets": 10})
        self.renumber()

    def tearDown(self):
        self.temp.cleanup()

    def renumber(self):
        for seq, row in enumerate(self.tape[1:]): row["seq"] = seq

    def check(self):
        return replay.cross_check_wire(self.root, self.manifest, self.tape)

    def test_exact_wire_and_policy_receipts_are_bound(self):
        result = self.check()
        self.assertEqual((result["messages"], result["responses"], result["requests"]), (2, 2, 2))
        self.assertTrue(all(row["verified"] for row in result["players"].values()))

    def test_rehashed_omission_or_observation_change_fails_wire_binding(self):
        original = deepcopy(self.tape)
        for mode in ("omit", "road", "recipient", "response"):
            with self.subTest(mode=mode):
                self.tape = deepcopy(original)
                if mode == "omit": self.tape.pop(2)
                elif mode == "road": self.tape[2]["message"]["records"][0]["observation"]["tiles"][0]["road"] = 1
                elif mode == "recipient": self.tape[2]["player"] = 1
                else: self.tape[3]["response"]["action"]["count"] = 4
                self.renumber()
                with self.assertRaises(ValueError): self.check()

    def test_final_unconsumed_intention_is_not_allowed_at_cutoff(self):
        self.tape.pop(3)
        self.renumber()
        result = self.check()
        self.assertEqual(result["players"]["red"]["submitted_without_receipt"], 1)
        self.manifest["status"] = "truncated"
        with self.assertRaises(ValueError): self.check()

    def test_tamper_generator_preserves_original_and_renumbers_mutations(self):
        before = deepcopy(self.tape)
        cases = replay.tamper_cases(self.tape)
        names = {name for name, _ in cases}
        self.assertTrue({"observed_road", "open_window_state", "exact_recruitment_count", "native_command_bytes",
                         "omitted_message", "wrong_recipient", "request_identity", "missing_natural_close"} <= names)
        self.assertEqual(self.tape, before)
        for name, changed in cases:
            self.assertNotEqual(changed, before, name)
            self.assertEqual([row["seq"] for row in changed[1:]], list(range(len(changed) - 1)))

    def test_old_recordings_are_rejected_before_output_creation(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "historical"):
            replay.verify_episode(self.root, binary=self.root / "missing", data=self.root, output=output)
        self.assertFalse(output.exists())

    def test_construction_selection_has_a_separate_semantic_mutation(self):
        self.tape.insert(-1, {"kind": "response", "player": 0,
                             "response": {"action": {"kind": "construct", "selection": "building-3"}}})
        self.renumber()
        before = deepcopy(self.tape)
        changed = dict(replay.tamper_cases(self.tape))["construction_selection"]
        self.assertEqual(self.tape, before)
        self.assertEqual(changed[-2]["response"]["action"]["selection"], "building-3-changed")


class NativeReportAdmissionTests(unittest.TestCase):
    """Pure report admission tests; no synthetic report is native evidence."""
    def setUp(self):
        self.barrier = {"schema": replay.evidence.SCHEMA, "kind": "client", "episode": "1",
            "interface": "adventure-town-v1", "barrier_id": 1, "origin": "request",
            "request": {"player": 0, "id": 42, "type": 90}, "client_packets": 123,
            "outcomes": ["win", "loss"], "callbacks_complete": True, "observed_count": 123}
        self.wire = dict(transactions=10, messages=5, responses=3, requests=2)
        counts = {**self.wire, "client_packets": 123}
        self.native = {"verified": True, "returncode": 0, "summary": {
            "server_verified": True, "loaded": counts, "checked": dict(counts),
            "unchecked": dict.fromkeys(counts, 0), "open_final_transaction": False,
            "captured_prefix_only": False, "regenerated_server_packets_after_prefix": 0,
            "boundary": {"kind": "natural", "barrier_id": 1, "client_packets": 123},
            "terminal_barrier": deepcopy(self.barrier)}}

    def check(self, *, cutoff=False):
        replay.validate_native_report(self.native, self.wire, {"terminal_barrier": {"client": self.barrier}}, cutoff=cutoff)

    def test_exact_regenerated_receipt_is_required(self):
        self.check()
        mutations = (lambda r: r["request"].update(id=43), lambda r: r["request"].update(player=1),
                     lambda r: r["request"].update(type=91), lambda r: r.update(outcomes=["loss", "win"]),
                     lambda r: r.update(observed_count=122), lambda r: r.update(extra="unbound"),
                     lambda r: r["request"].update(player=False), lambda r: r.update(barrier_id=1.0))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.native["summary"]["terminal_barrier"] = deepcopy(self.barrier)
                mutate(self.native["summary"]["terminal_barrier"])
                with self.assertRaisesRegex(ValueError, "Regenerated terminal barrier"): self.check()

    def test_matching_saved_receipts_cannot_override_the_regenerated_request(self):
        self.barrier["request"]["id"] = 999
        with self.assertRaisesRegex(ValueError, "Regenerated terminal barrier"): self.check()

    def test_incomplete_coverage_failure_or_extra_native_packet_is_rejected(self):
        original = deepcopy(self.native)
        mutations = (lambda n: n.update(verified=False), lambda n: n.update(returncode=3),
                     lambda n: n["summary"]["loaded"].update(messages=6),
                     lambda n: n["summary"]["checked"].pop("requests"),
                     lambda n: n["summary"].update(unchecked={}),
                     lambda n: n["summary"].update(regenerated_server_packets_after_prefix=1))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.native = deepcopy(original); mutate(self.native)
                with self.assertRaises(ValueError): self.check()

    def test_cutoff_requires_absent_regenerated_natural_barrier(self):
        summary = self.native["summary"]
        summary["captured_prefix_only"] = True
        summary["boundary"].update(kind="cutoff", barrier_id=None)
        with self.assertRaisesRegex(ValueError, "cannot claim"): self.check(cutoff=True)
        summary["terminal_barrier"] = None
        self.check(cutoff=True)
        del summary["terminal_barrier"]
        with self.assertRaises(ValueError): self.check(cutoff=True)


class TamperPositiveControlTests(unittest.TestCase):
    """Synthetic admission-flow tests; mocked calls never execute a game."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "engine").mkdir()
        self.manifest = {"status": "truncated", "launch": {"requested_settings": {}}}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        for name in ("contextual-finalization.json", *replay.evidence.REPLAY_FILES):
            (self.root / name).write_text('{"synthetic_admission_test":true}\n')
        self.order = []

    def tearDown(self):
        self.temporary.cleanup()

    def control(self, passed, *, mutate=False):
        def run(episode, **kwargs):
            self.order.append("positive")
            output = kwargs["output"]; output.mkdir()
            result = {"verified": passed, "synthetic_admission_test": True}
            (output / "contextual-replay-verification.json").write_text(json.dumps(result))
            if mutate:
                (self.root / "engine/contextual-replay.jsonl").write_text('{"changed":true}\n')
            return result
        return run

    def run_tamper(self):
        return replay.tamper_episode(self.root, binary=self.root / "vcmireplay", data=self.root,
                                      output=self.root / "tamper")

    def test_failed_original_cannot_make_universal_rejection_pass(self):
        with patch.object(replay, "verify_episode", side_effect=self.control(False)), \
             patch.object(replay, "run_native") as native, patch.object(replay, "bound_evidence") as bound:
            result = self.run_tamper()
        self.assertFalse(result["passed"])
        self.assertFalse(result["positive_control"]["verified"])
        self.assertEqual(result["cases"], [])
        native.assert_not_called(); bound.assert_not_called()

    def test_original_inputs_cannot_change_between_control_and_mutations(self):
        with patch.object(replay, "verify_episode", side_effect=self.control(True, mutate=True)), \
             patch.object(replay, "run_native") as native:
            result = self.run_tamper()
        self.assertFalse(result["passed"])
        self.assertIn("inputs changed", result["error"])
        native.assert_not_called()

    def test_successful_control_precedes_rehashed_negative_and_is_bound(self):
        def reject(*args, **kwargs):
            self.order.append("negative")
            return {"verified": False, "returncode": 3, "synthetic_admission_test": True}
        with patch.object(replay, "verify_episode", side_effect=self.control(True)), \
             patch.object(replay, "bound_evidence", side_effect=lambda *a: self.order.append("bound")), \
             patch.object(replay.legacy, "complete_replay_records", side_effect=lambda rows, **k: (rows, 0)), \
             patch.object(replay, "tamper_cases", return_value=[("mutation", [{"changed": True}])]), \
             patch.object(replay.legacy, "replay_profile", return_value=({}, self.root / "map")), \
             patch.object(replay, "run_native", side_effect=reject):
            result = self.run_tamper()
        self.assertEqual(self.order, ["positive", "bound", "negative"])
        self.assertTrue(result["passed"])
        control = result["positive_control"]
        path = self.root / "tamper" / control["path"]
        self.assertEqual(control["identity"], replay.evidence.file_identity(path))
        self.assertEqual(set(control["inputs"]), {"manifest.json", "contextual-finalization.json", *replay.evidence.REPLAY_FILES})


if __name__ == "__main__":
    unittest.main()
