"""Offline auditor tamper tests; synthetic records are not native game evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner import contextual as c
from validation.probes.audit_contextual_histories import audit, main


def observation(window=None, enemy=False):
    return c.Observation(
        (6, 6, 1), (c.Tile((1, 1, 0), 2, 0, False, False), c.Tile((2, 1, 0), 2, 1, False, False)),
        (c.ObjectView("enemy", (2, 1, 0), 34, 0, "opponent"),) if enemy else (),
        (0,) * 7, (c.Hero("hero", (1, 1, 0), 1000, 1000),), (c.Town("town", (1, 1, 0)),), window)


def decision(history, obs=None):
    obs = obs or observation()
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs), revision=history.revision + 1)


def submit(history, action):
    seq = history.decision.seq
    history.emit("submitted", action=action, decision_seq=seq)
    history.emit("consumed", consumer="adapter", decision_seq=seq)
    return seq


def no_op(history):
    decision(history)
    seq = submit(history, c.Action("move_to", "hero", 1, 1))
    history.emit("result", decision_seq=seq, result=c.Result("reached", True, (1, 1, 0), 1000, True))


def finalize_cutoff(pair):
    return c.finalize_cutoff(pair, evidence_ref="synthetic-native-evidence", histories_complete=(True, True),
                             failure_free=True, shutdown_verified=True, continuation_limit=4, continuation_steps=(0, 0))


def histories(boundary="natural"):
    pair = (c.PlayerHistory("synthetic-audit", 0), c.PlayerHistory("synthetic-audit", 1))
    for history in pair:
        no_op(history)
        if boundary == "natural":
            history.emit("terminal", outcome="win" if history.player == 0 else "loss")
        elif boundary == "cutoff":
            frame = decision(history)
            history.emit("cutoff", decision_seq=frame.seq, reason="decision_budget")
        else:
            history.emit("failure", reason="reader_error")
    if boundary == "natural":
        c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                           process_exit_code=0, evidence_ref="synthetic-native-evidence")
    elif boundary == "cutoff":
        finalize_cutoff(pair)
    return pair


class ContextualAuditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = (self.root / "red.jsonl", self.root / "blue.jsonl")

    def tearDown(self):
        self.temp.cleanup()

    def persist(self, pair):
        for history, path in zip(pair, self.paths):
            # Deliberately noncanonical whitespace demonstrates the distinction
            # between file-byte hashes and canonical contract-prefix hashes.
            path.write_text("".join(json.dumps(c.encode_frame(frame), sort_keys=True) + "\n" for frame in history.frames))
        manifest = {
            "schema_version": 1, "kind": "contextual-headless-self-play", "environment": c.ENVIRONMENT,
            "history_schema": c.SCHEMA, "action_schema": c.ACTION_SCHEMA, "episode": pair[0].episode,
            "status": {"natural": "completed", "cutoff": "truncated", "failure": "failed"}[pair[0].boundary],
            "termination_reason": "synthetic_fixture", "returncode": 0, "evidence_ref": "synthetic-native-evidence",
            "outcomes_by_player": {color: history.terminal if history.boundary == "natural" else None
                                   for color, history in zip(("red", "blue"), pair)},
            "rewards_by_player": {color: reward if pair[0].boundary == "natural" else None
                                  for color, reward in (("red", 1), ("blue", -1))},
            "identity": {"engine_revision": "synthetic-no-game", "bridge_source_sha256": "a" * 64, "map_sha256": "b" * 64},
            "players": {},
        }
        for color, path, history in zip(("red", "blue"), self.paths, pair):
            proof = (history.frames[-1].verification if history.boundary == "natural" else
                     history.frames[-1].cutoff_verification if history.boundary == "cutoff" else None)
            prefix = proof.history_sha256[history.player] if proof else history.digest()
            manifest["players"][color] = {"history_v2": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "canonical_sha256": history.digest(), "prefix_sha256": prefix,
                "boundary": history.boundary, "complete": history.boundary != "failure"}}
        return manifest

    def rows(self, player=0):
        return [json.loads(line) for line in self.paths[player].read_text().splitlines()]

    def write_rows(self, rows, player=0):
        self.paths[player].write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_natural_pair_binds_distinct_file_and_contract_hashes(self):
        manifest = self.persist(histories())
        report = audit(*self.paths, manifest)
        self.assertTrue(report["consistent"] and report["manifest_bound"] and report["scoring_binding_verified"])
        self.assertTrue(report["record_contract_eligible"] and report["has_eligible_samples"] and report["learner_eligible"])
        red = report["players"]["red"]
        self.assertNotEqual(red["file_sha256"], red["canonical_sha256"])
        self.assertNotEqual(red["prefix_sha256"], red["canonical_sha256"])
        self.assertEqual(red["transitions"]["contract_terminal_rewards"], [1])
        self.assertFalse(report["source_visibility_proven"])
        self.assertIn("does not prove", report["scope"])

    def test_no_manifest_is_structural_only(self):
        self.persist(histories())
        report = audit(*self.paths)
        self.assertTrue(report["consistent"])
        self.assertFalse(report["manifest_bound"] or report["scoring_binding_verified"] or report["learner_eligible"])
        self.assertFalse(report["record_contract_eligible"] or report["has_eligible_samples"])
        self.assertEqual(report["players"]["red"]["transitions"]["effective_eligible"], 0)

    def test_empty_natural_record_is_valid_but_has_no_learner_samples(self):
        # A verified natural result can exist before a seat has a completed
        # action. Record validity does not manufacture a learner transition.
        for red_has_action in (False, True):
            with self.subTest(red_has_action=red_has_action):
                pair = (c.PlayerHistory("synthetic-audit", 0), c.PlayerHistory("synthetic-audit", 1))
                if red_has_action:
                    no_op(pair[0])
                for history, outcome in zip(pair, ("win", "loss")):
                    history.emit("terminal", outcome=outcome)
                c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                                   process_exit_code=0, evidence_ref="synthetic-native-evidence")
                manifest = self.persist(pair)
                report = audit(*self.paths, manifest)
                self.assertTrue(report["consistent"] and report["scoring_binding_verified"])
                self.assertTrue(report["record_contract_eligible"])
                self.assertEqual(report["has_eligible_samples"], red_has_action)
                self.assertEqual(report["learner_eligible"], red_has_action)
                for color, has_samples in (("red", red_has_action), ("blue", False)):
                    player = report["players"][color]
                    self.assertTrue(player["record_contract_eligible"])
                    self.assertEqual(player["has_eligible_samples"], has_samples)
                    self.assertEqual(player["learner_eligible"], has_samples)
                    self.assertEqual(player["transitions"]["effective_eligible"], int(has_samples))
                    self.assertEqual(player["transitions"]["effective_bootstrap_allowed"], 0)

    def test_cutoff_and_failure_never_gain_terminal_rewards(self):
        for boundary in ("cutoff", "failure"):
            with self.subTest(boundary=boundary):
                manifest = self.persist(histories(boundary))
                report = audit(*self.paths, manifest)
                self.assertTrue(report["consistent"], report["failures"])
                self.assertFalse(report["scoring_binding_verified"])
                for player in report["players"].values():
                    self.assertEqual(player["transitions"]["contract_terminal_rewards"], [])
                    self.assertEqual(player["transitions"]["effective_bootstrap_allowed"], int(boundary == "cutoff"))
                    self.assertEqual(player["learner_eligible"], boundary == "cutoff")

    def test_wrong_player_episode_sequence_and_missing_schema_fail_closed(self):
        for field, value in (("player", 1), ("episode", "different"), ("seq", 19), ("schema", None)):
            with self.subTest(field=field):
                self.persist(histories())
                rows = self.rows()
                if field == "schema":
                    del rows[1][field]
                else:
                    rows[1][field] = value
                self.write_rows(rows)
                report = audit(*self.paths)
                self.assertFalse(report["consistent"] or report["learner_eligible"])
                self.assertEqual(report["failures"][0]["line"], 2)
        self.persist(histories())
        self.assertFalse(audit(self.paths[1], self.paths[0])["consistent"])

    def test_cross_episode_pair_even_when_individually_valid_is_rejected(self):
        pair = histories("failure")
        self.persist(pair)
        rows = self.rows(1)
        for row in rows:
            row["episode"] = "another-game"
        self.write_rows(rows, 1)
        report = audit(*self.paths)
        self.assertFalse(report["consistent"])
        self.assertIn("Different player episode", str(report["failures"]))

    def test_manifest_status_reward_evidence_provenance_and_hash_mismatch(self):
        original = self.persist(histories())
        mutations = [lambda m: m.update(status="failed"), lambda m: m.update(returncode=1),
                     lambda m: m.update(evidence_ref="other-evidence"), lambda m: m.update(environment="movement-v1"),
                     lambda m: m["rewards_by_player"].update(red=0), lambda m: m["identity"].pop("map_sha256"),
                     lambda m: m["players"]["blue"]["history_v2"].update(sha256="0" * 64),
                     lambda m: m["players"]["red"]["history_v2"].update(prefix_sha256="0" * 64),
                     lambda m: m["players"]["red"]["history_v2"].update(complete=False)]
        for mutation in mutations:
            manifest = deepcopy(original)
            mutation(manifest)
            report = audit(*self.paths, manifest)
            self.assertFalse(report["consistent"] or report["manifest_bound"] or report["learner_eligible"])
            self.assertEqual(report["players"]["red"]["transitions"]["effective_eligible"], 0)

    def test_final_proofs_must_bind_both_actual_prefixes(self):
        self.persist(histories())
        rows = self.rows()
        rows[-1]["verification"]["history_sha256"][1] = "f" * 64
        self.write_rows(rows)
        report = audit(*self.paths)
        self.assertFalse(report["consistent"])
        self.assertIn("same two complete prefixes", str(report["failures"]))

    def test_immutable_prefix_tamper_and_postfinal_append_rejected(self):
        for tamper_prefix in (True, False):
            self.persist(histories())
            rows = self.rows()
            if tamper_prefix:
                rows[0]["observation"]["resources"][0] = 1
            else:
                rows.append({**rows[0], "seq": len(rows), "revision": 12})
            self.write_rows(rows)
            report = audit(*self.paths)
            self.assertFalse(report["consistent"] or report["learner_eligible"])

    def test_native_acknowledgement_cannot_be_dropped(self):
        pair = (c.PlayerHistory("synthetic-audit", 0), c.PlayerHistory("synthetic-audit", 1))
        for history in pair:
            decision(history)
            seq = submit(history, c.Action("move_to", "hero", 2, 1))
            history.emit("issued", decision_seq=seq, command_index=0)
            history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
            history.emit("result", decision_seq=seq, result=c.Result("reached", True, (2, 1, 0), 900, True))
            frame = decision(history, replace(observation(), heroes=(c.Hero("hero", (2, 1, 0), 900),)))
            history.emit("cutoff", decision_seq=frame.seq, reason="budget")
        finalize_cutoff(pair)
        self.persist(pair)
        self.assertTrue(audit(*self.paths)["consistent"])
        rows = self.rows()
        del rows[4]
        for seq, row in enumerate(rows):
            row["seq"] = seq
        rows[-2]["decision_seq"] -= 1
        self.write_rows(rows)
        report = audit(*self.paths)
        self.assertFalse(report["consistent"])
        self.assertIn("acknowledgement", str(report["failures"]))

    def test_coverage_counts_windows_roads_and_transient_sightings(self):
        pair = (c.PlayerHistory("synthetic-audit", 0), c.PlayerHistory("synthetic-audit", 1))
        panel = c.Window(c.Context("hero", "hero"), (c.Stack(0, 0, 25),), primary_skills=(1, 1, 1, 1), mana=10, level=1)
        for history in pair:
            decision(history)
            seq = submit(history, c.Action("open_hero", "hero"))
            history.emit("context", decision_seq=seq, context=panel.context, revision=1)
            history.emit("view", observation=observation(panel, enemy=True))
            history.emit("view", observation=observation(panel))
            history.emit("result", decision_seq=seq, result=c.Result("opened", True))
            decision(history, observation(panel))
            seq = submit(history, c.Action("close"))
            history.emit("context", decision_seq=seq, context=c.Context(), revision=3)
            history.emit("view", observation=observation())
            history.emit("result", decision_seq=seq, result=c.Result("closed", True))
            frame = decision(history)
            history.emit("cutoff", decision_seq=frame.seq, reason="budget")
        finalize_cutoff(pair)
        manifest = self.persist(pair)
        report = audit(*self.paths, manifest)
        self.assertTrue(report["consistent"], report["failures"])
        coverage = report["players"]["red"]["coverage"]
        self.assertEqual(coverage["observation_contexts"], {"hero": 3, "map": 3})
        self.assertEqual(coverage["road_id_values"], [0, 1])
        self.assertEqual(coverage["opponent_sightings"]["fully_transient_position_samples"], 1)
        self.assertFalse(coverage["closed_window_nonrefresh_proven"])

    def test_cutoff_proofs_require_both_complete_prefixes_and_no_late_failure(self):
        self.persist(histories("cutoff"))
        rows = self.rows()
        rows[-1]["cutoff_verification"]["history_sha256"][1] = "f" * 64
        self.write_rows(rows)
        report = audit(*self.paths)
        self.assertFalse(report["consistent"])
        self.assertIn("same two complete prefixes", str(report["failures"]))

        manifest = self.persist(histories("cutoff"))
        rows = self.rows()
        del rows[-1]
        self.write_rows(rows)
        report = audit(*self.paths, manifest)
        self.assertFalse(report["consistent"] or report["learner_eligible"])
        self.assertEqual(report["players"]["red"]["boundary"], "cutoff_pending")
        self.assertEqual(report["players"]["red"]["transitions"]["contract_eligible"], 0)

        pair = (c.PlayerHistory("synthetic-audit", 0), c.PlayerHistory("synthetic-audit", 1))
        for history in pair:
            no_op(history)
            held = decision(history)
            history.emit("cutoff", decision_seq=held.seq, reason="budget")
            history.emit("failure", reason="trailing_reader_error")
        manifest = self.persist(pair)
        report = audit(*self.paths, manifest)
        self.assertTrue(report["consistent"], report["failures"])
        self.assertFalse(report["learner_eligible"])
        self.assertEqual(report["players"]["red"]["transitions"]["effective_bootstrap_allowed"], 0)

    def test_malformed_and_empty_json_produce_diagnostics_not_success(self):
        for bad in ("", "\n", "{", '{"schema":"x","schema":"y"}\n'):
            self.persist(histories())
            self.paths[0].write_text(bad)
            self.assertFalse(audit(*self.paths)["consistent"])
        self.persist(histories())
        for manifest in ([], {"status": []}, {"status": "completed", "players": []}):
            self.assertFalse(audit(*self.paths, manifest)["consistent"])

    def test_cli_output_is_new_and_read_only(self):
        manifest = self.persist(histories("failure"))
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        before = [path.read_bytes() for path in (*self.paths, manifest_path)]
        output = self.root / "audit.json"
        args = ["--red", str(self.paths[0]), "--blue", str(self.paths[1]), "--manifest", str(manifest_path), "--output", str(output)]
        self.assertEqual(main(args), 0)
        self.assertTrue(json.loads(output.read_text())["consistent"])
        self.assertEqual(before, [path.read_bytes() for path in (*self.paths, manifest_path)])
        with self.assertRaises(FileExistsError):
            main(args)

    def test_cli_explicit_null_manifest_is_rejected_not_treated_as_omitted(self):
        self.persist(histories())
        manifest = self.root / "null-manifest.json"
        manifest.write_text("null\n")
        output = self.root / "null-manifest-audit.json"
        before = [path.read_bytes() for path in (*self.paths, manifest)]
        result = main(["--red", str(self.paths[0]), "--blue", str(self.paths[1]),
                       "--manifest", str(manifest), "--output", str(output)])
        self.assertEqual(result, 1)
        report = json.loads(output.read_text())
        for flag in ("consistent", "manifest_bound", "record_contract_eligible", "has_eligible_samples", "learner_eligible"):
            self.assertFalse(report[flag])
        self.assertIn("Provided manifest must be a JSON object", str(report["failures"]))
        self.assertEqual(before, [path.read_bytes() for path in (*self.paths, manifest)])


if __name__ == "__main__":
    unittest.main()
