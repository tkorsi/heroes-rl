"""Synthetic mixed replay admission; these tests never run a native engine."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.headless import builtin_replay as replay
from experiments.headless import builtin_evidence as evidence
from experiments.headless.builtin_replay_history import verify_history
from experiments.headless.contextual_protocol import ContextualProducer
from experiments.headless.contextual_replay_history import _wire
from experiments.learner import contextual as c
from tests.test_contextual_dialogs import observation


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


class WireTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture(0)

    def fixture(self, player, *, cutoff=False):
        color = replay.COLORS[player]
        folder = self.root / "agents" / color
        folder.mkdir(parents=True, exist_ok=True)
        self.manifest = {"episode": "1", "environment": c.DIALOGS_ENVIRONMENT, "status": "truncated" if cutoff else "completed",
                         "learner_player": player, "controllers": replay.controllers(player)}
        message = {"player": player, "records": [{"kind": "view", "observation": {"own": "synthetic"}}]}
        response = {"player": player, "action": {"kind": "end_turn"}}
        rows = [{"direction": "engine", "message": message},
                {"direction": "policy", "message": {**response, "delivered": True, "outside_learner_prefix": False}}]
        (folder / "bridge.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.tape = [{"kind": "init", "schema": replay.SCHEMA, "episode": "1", "interface": c.DIALOGS_ENVIRONMENT,
            "protocol_version": 9, "history_schema": c.DIALOGS_SCHEMA, "action_schema": c.DIALOGS_ACTION_SCHEMA,
            "controllers": replay.controllers(player)},
            {"kind": "enter", "actor": f"player{player}", "operation": "worker", "packet": None},
            {"kind": "message", "player": player, "message": message},
            {"kind": "response", "player": player, "response": response},
            {"kind": "request", "source": "adapter", "player": player, "request_id": 3, "pack_type": 90, "pack": "aabb"},
            {"kind": "leave", "actor": f"player{player}", "operation": "worker"},
            {"kind": "enter", "actor": "client", "operation": "packet", "packet": None},
            {"kind": "request", "source": "native_opponent", "player": 1-player, "request_id": 4, "pack_type": 91, "pack": "ccdd"},
            {"kind": "packet_complete"}, {"kind": "leave", "actor": "client", "operation": "packet"}]
        if not cutoff:
            self.tape.append({"kind": "close", "barrier_id": 1, "client_packets": 1})
        self.renumber()
        self.server = [{"kind": "init"}, {"kind": "start"}]
        for index, request in enumerate(row for row in self.tape if row["kind"] == "request"):
            self.server.extend([{"kind": "input", "seq": index, "player": request["player"], "request": request["request_id"],
                "type": request["pack_type"], "pack": request["pack"]}, {"kind": "applied", "seq": index}])

    def renumber(self):
        for seq, row in enumerate(self.tape[1:]):
            row["seq"] = seq

    def check(self):
        return replay.cross_check_wire(self.root, self.manifest, self.tape, self.server)

    def test_both_colors_bind_native_commands_without_opponent_history(self):
        for player in (0, 1):
            for cutoff in (False, True):
                self.fixture(player, cutoff=cutoff)
                result = self.check()
                self.assertEqual(result["native_opponent_requests"], 1)
                self.assertEqual(result["requests"], 2)
                self.assertEqual(result["learner_player"], player)
                self.assertIsNone(result["opponent_history"])

    def test_reordered_or_changed_native_bytes_fail_server_binding(self):
        original = deepcopy(self.server)
        for mutation in (lambda r: r[4].update(pack="ccee"), lambda r: r[4].update(player=0),
                         lambda r: r[4].update(request=5), lambda r: r[4].update(type=True),
                         lambda r: r.__setitem__(slice(2,6), r[4:6]+r[2:4])):
            self.server = deepcopy(original)
            mutation(self.server)
            with self.assertRaises(ValueError):
                self.check()

    def test_opponent_cannot_be_relabelled_or_acquire_wire_messages(self):
        original = deepcopy(self.tape)
        for mutation in (lambda t: t[7].update(source="automatic"), lambda t: t[7].update(source="adapter"),
                         lambda t: t[7].update(player=0), lambda t: t[2].update(player=1),
                         lambda t: t[0]["controllers"]["explorer_players"].append(1)):
            self.tape = deepcopy(original)
            mutation(self.tape)
            with self.assertRaises(ValueError):
                self.check()

    def test_omissions_and_wrong_boundary_are_rejected(self):
        self.tape.pop(7)
        self.renumber()
        with self.assertRaises(ValueError):
            self.check()
        self.fixture(0)
        self.manifest["status"] = "truncated"
        with self.assertRaises(ValueError):
            self.check()

    def test_paired_capture_is_rejected_before_creating_output(self):
        write(self.root / "manifest.json", {"replay": {"schema": "heroes-contextual-replay-v1"}})
        with self.assertRaisesRegex(ValueError, "Fresh mixed"):
            replay.verify_episode(self.root, binary=self.root / "absent", data=self.root, output=self.root / "out")
        self.assertFalse((self.root / "out").exists())


class NativeReportTests(unittest.TestCase):
    def setUp(self):
        self.wire = dict(transactions=3, messages=2, responses=1, requests=4, client_packets=10,
                         native_opponent_requests=2, controllers=replay.controllers(1))
        self.barrier = {"controllers": replay.controllers(1), "barrier_id": 1, "client_packets": 10,
            "request": {"player": 0, "id": 42, "type": 90}, "outcomes": ["win", "loss"],
            "native_terminal_callbacks": [0, 1], "explorer_terminal_callbacks": [1]}
        counts = {k: v for k, v in self.wire.items() if k != "controllers"}
        self.native = {"schema": replay.REPORT_SCHEMA, "verified": True, "returncode": 0, "summary": {
            "controllers": replay.controllers(1), "server_verified": True, "loaded": counts,
            "checked": dict(counts), "unchecked": dict.fromkeys(counts, 0), "server": {"inputs": 4},
            "captured_prefix_only": False, "open_final_transaction": False, "regenerated_server_packets_after_prefix": 0,
            "boundary": {"kind": "natural", "barrier_id": 1, "client_packets": 10}, "terminal_barrier": deepcopy(self.barrier)}}

    def check(self):
        replay.validate_native_report(self.native, self.wire, {"terminal_barrier": {"client": self.barrier}}, cutoff=False)

    def test_complete_mixed_coverage_is_admitted(self):
        self.check()

    def test_missing_native_coverage_or_changed_controller_or_effect_count_fails(self):
        original = deepcopy(self.native)
        mutations = (lambda n: n["summary"]["checked"].update(native_opponent_requests=1),
            lambda n: n["summary"]["loaded"].pop("native_opponent_requests"),
            lambda n: n["summary"]["unchecked"].update(native_opponent_requests=False),
            lambda n: n["summary"].update(controllers=replay.controllers(0)),
            lambda n: n["summary"]["server"].update(inputs=3),
            lambda n: n["summary"]["terminal_barrier"].update(explorer_terminal_callbacks=[0,1]),
            lambda n: n.update(schema="heroes-contextual-replay-verification-v1"))
        for mutate in mutations:
            self.native = deepcopy(original)
            mutate(self.native)
            with self.assertRaises(ValueError):
                self.check()


class EffectiveEnvironmentTests(unittest.TestCase):
    def test_actual_easy_flags_and_both_controller_seats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "engine").mkdir()
            (root / "logs").mkdir()
            for player in (0, 1):
                metadata = {"schema_version": 1, "phase": "before_first_turn", "game_day": 0,
                    "cheats_allowed": False, "requested_no_cheats": True, "difficulty_index": 0,
                    "map_width": 4, "map_height": 4, "map_levels": 1, "requested_agent_player": player,
                    "controllers": replay.controllers(player), "players": [
                        {"player_index": p, "team_index": p, "human": False, "visible_tiles": 3, "total_tiles": 16,
                         "adventure_ai": "ExplorerRL" if p == player else "Nullkiller"} for p in (0, 1)]}
                log = (f"Experiment AI visibility: player={1-player} cheats_allowed=0 open_map=0 use_object_graph=0\n"
                       "Experiment engine RNG: initial_seed=123 initial_sample=9\n")
                write(root / "engine/experiment-metadata.json", metadata)
                (root / "logs/process.log").write_text(log)
                checked = {"metadata": replay.legacy.scoring.validate_experiment_metadata(metadata, player, log, agent_player=player),
                           "seed": replay.legacy.scoring.engine_seed_evidence(log, 123)}
                manifest = {"learner_player": player, "seed": 123, "engine_evidence": checked}
                final = {"engine_evidence": checked}
                self.assertTrue(replay.effective_environment(root, manifest, final)["verified"])
                metadata["players"][1-player]["adventure_ai"] = "Nullkiller2"
                write(root / "engine/experiment-metadata.json", metadata)
                with self.assertRaisesRegex(ValueError, "fixed Nullkiller"):
                    replay.effective_environment(root, manifest, final)


class HistoryFixture:
    def __init__(self, root, player, *, natural=False, late=False):
        self.root, self.player, self.color = root, player, replay.COLORS[player]
        self.producer = ContextualProducer("1", player, interface=c.DIALOGS_ENVIRONMENT)
        self.rows = []
        self.engine(dict(protocol_version=9, type="ready", episode="1", player=player,
            interface=c.DIALOGS_ENVIRONMENT, history_schema=c.DIALOGS_SCHEMA, action_schema=c.DIALOGS_ACTION_SCHEMA))
        self.batch(self.decision(0))
        action = c.Action("end_turn", schema=c.DIALOGS_ACTION_SCHEMA)
        self.rows.append({"t": .1, "direction": "policy", "message": {
            **self.producer.response(action), "delivered": True, "outside_learner_prefix": False}})
        self.producer.submitted(action)
        prefix = [{"kind": "consumed", "day": 1, "decision_id": 0, "consumer": "engine"},
                  {"kind": "issued", "day": 1, "decision_id": 0, "command_index": 0}]
        if natural:
            self.batch(*prefix, {"kind": "superseded", "day": 1, "decision_id": 0, "reason": "terminal"},
                       {"kind": "terminal", "day": 1, "outcome": "win"})
        else:
            self.batch(*prefix, {"kind": "acknowledged", "day": 1, "decision_id": 0, "command_index": 0, "accepted": True},
                {"kind": "result", "day": 1, "decision_id": 0, "result": {"status": "turn_ended", "accepted": True,
                 "position": None, "movement": None, "destination_reached": None}}, self.decision(1))
            self.producer.freeze_cutoff("collection_window")
            if late:
                self.batch({"kind": "superseded", "day": 1, "decision_id": 1, "reason": "visible_update"}, self.decision(2))
        folder = root / "agents" / self.color
        folder.mkdir(parents=True)
        self.bridge_path = folder / "bridge.jsonl"
        self.bridge_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        for label, frames in (("history-v7.jsonl", self.producer.history.frames), ("live-history-v7.jsonl", self.producer.live.frames)):
            (folder / label).write_bytes(b"".join(_wire(frame) for frame in frames))
        sha = self.producer.history.digest()
        outcome = {self.color: "win", replay.COLORS[1-player]: "loss"} if natural else dict.fromkeys(replay.COLORS)
        declaration = {"ready": True, "error": None, "terminal": {"outcome": "win"} if natural else None,
            "actions_sent": 1, "actions_accepted": 0 if natural else 1, "actions_rejected": 0, "responses_undelivered": 0,
            "history_v7": {"path": f"agents/{self.color}/history-v7.jsonl", "sha256": sha,
                "canonical_sha256": sha, "prefix_sha256": sha, "boundary": "natural" if natural else "cutoff", "complete": True},
            "live_evidence": {"path": f"agents/{self.color}/live-history-v7.jsonl",
                "sha256": evidence.native.file_identity(folder / "live-history-v7.jsonl")["sha256"],
                "continuation_steps": 0, "learner_input": False}}
        self.manifest = {"schema_version": 1, "kind": evidence.MANIFEST_KIND, "episode": "1", "learner_player": player,
            "controllers": evidence.controllers(player), "opponent_identity": evidence.opponent_identity(player, "a"*64),
            "environment": c.DIALOGS_ENVIRONMENT, "history_schema": c.DIALOGS_SCHEMA, "action_schema": c.DIALOGS_ACTION_SCHEMA,
            "status": "completed" if natural else "truncated", "returncode": 0 if natural else -15,
            "termination_reason": "natural_terminal" if natural else "collection_window", "identity": {"synthetic_only": True},
            "replay": {"synthetic_only": True}, "dialog_cutoff_drain": None, "players": {self.color: declaration},
            "outcomes_by_player": outcome, "rewards_by_player": {k: 1 if v == "win" else -1 if v == "loss" else None for k,v in outcome.items()}}
        self.final = {"schema": evidence.FINAL_SCHEMA, "episode": "1", "learner_player": player,
            "controllers": evidence.controllers(player), "returncode": self.manifest["returncode"], "identity": self.manifest["identity"],
            "native_capture": self.manifest["replay"], "dialog_cutoff_drain": None, "history_sha256": sha,
            "stream_complete": True, "engine_evidence_error": None, "terminal_barrier_error": None,
            "socket_outcome": "win" if natural else None, "engine_verified_outcomes": outcome if natural else None,
            "declared_stop": not natural, "shutdown_verified": not natural, "reason": None if natural else "collection_window"}
        self.save_bindings()

    def engine(self, message):
        self.rows.append({"t": len(self.rows)*.1, "direction": "engine", "message": message})
        self.producer.ingest(message)

    def batch(self, *records):
        self.engine(dict(protocol_version=9, type="contextual", episode="1", player=self.player, records=list(records)))

    def decision(self, index):
        return {"kind": "decision", "day": 1, "decision_id": index, "revision": index,
                "observation": c.encode_observation(observation(), schema=c.DIALOGS_SCHEMA)}

    def save_bindings(self):
        self.final["raw_bridge_sha256"] = {self.color: evidence.native.file_identity(self.bridge_path)["sha256"]}
        write(self.root / "builtin-finalization.json", self.final)
        self.manifest["evidence_ref"] = "builtin-finalization.json#sha256=" + evidence.native.file_identity(self.root / "builtin-finalization.json")["sha256"]
        write(self.root / "manifest.json", self.manifest)


class HistoryTests(unittest.TestCase):
    def test_one_seat_natural_and_cutoff_reconstruct_without_paired_frames(self):
        for player in (0, 1):
            for natural in (False, True):
                with tempfile.TemporaryDirectory() as tmp:
                    fixture = HistoryFixture(Path(tmp), player, natural=natural, late=not natural)
                    result = verify_history(fixture.root, fixture.manifest, fixture.final)
                    self.assertTrue(result["verified"])
                    self.assertFalse(result["native_replay_verified"])
                    self.assertIsNone(result["opponent_history"])
                    self.assertEqual(result["counts"]["delivered_actions"], 1)
                    if not natural:
                        self.assertGreater(result["live"]["frames"], result["learner"]["frames"])

    def test_rehashed_bridge_mutation_still_fails_history_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = HistoryFixture(Path(tmp), 1)
            rows = deepcopy(fixture.rows)
            rows[1]["message"]["records"][0]["observation"]["resources"][0] += 1
            fixture.bridge_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            fixture.save_bindings()
            with self.assertRaises(ValueError):
                verify_history(fixture.root, fixture.manifest, fixture.final)

    def test_native_opponent_declaration_cannot_be_an_empty_fake_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = HistoryFixture(Path(tmp), 0)
            fixture.manifest["players"]["blue"] = {"history_v7": None}
            fixture.save_bindings()
            with self.assertRaisesRegex(ValueError, "sole learner"):
                verify_history(fixture.root, fixture.manifest, fixture.final)


if __name__ == "__main__":
    unittest.main()
