"""Synthetic transport reconstruction tests; no engine or replay-success claim."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.headless.contextual_protocol import ContextualProducer
from experiments.headless.contextual_replay_history import verify_histories
from experiments.learner import contextual as c


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation():
    return c.Observation((4, 4, 1), (c.Tile((1, 1, 0), 2, 0, False, False),), (), (0,) * 7,
                         (c.Hero("local-hero", (1, 1, 0), 100),), (c.Town("local-town", (1, 1, 0)),))


class Fixture:
    """Explicit synthetic producer streams with independent persisted artifacts."""
    def __init__(self, directory, *, natural=False, town=False, reverse=False, late_cutoff_view=False, profile=None):
        self.directory = directory
        self.profile = profile or (c.TOWN_PROFILE if town else c.WINDOWS_PROFILE)
        self.producers = [ContextualProducer("1", p, interface=self.profile.environment) for p in (0, 1)]
        self.rows = [[], []]
        self.natural = natural
        self.outcomes = ["loss", "win"] if reverse else ["win", "loss"]
        self.identity = {"engine_revision": "synthetic-only", "map_sha256": "a" * 64}
        for p in (0, 1):
            folder = directory / "agents" / ("red", "blue")[p]
            folder.mkdir(parents=True)
            self.engine(p, dict(protocol_version=self.profile.protocol, type="ready", episode="1", player=p,
                interface=self.profile.environment, history_schema=self.profile.history_schema,
                action_schema=self.profile.action_schema))
            self.batch(p, self.decision(0, 0))
            self.submit(p, c.Action("move_to", "local-hero", 1, 1, schema=self.profile.action_schema))
            self.batch(p, dict(kind="consumed", day=1, decision_id=0, consumer="adapter"),
                dict(kind="result", day=1, decision_id=0,
                     result=dict(status="reached", accepted=True, position=[1, 1, 0], movement=100,
                                 destination_reached=True)), self.decision(1, 1))
            if natural:
                self.submit(p, c.Action("end_turn", schema=self.profile.action_schema))
                self.batch(p, dict(kind="consumed", day=1, decision_id=1, consumer="engine"),
                    dict(kind="issued", day=1, decision_id=1, command_index=0),
                    dict(kind="superseded", day=1, decision_id=1, reason="terminal"),
                    dict(kind="terminal", day=1, outcome=self.outcomes[p]))
                # Complete natural history includes the actual postterminal view.
                self.batch(p, dict(kind="view", day=1,
                                   observation=c.encode_observation(observation(), schema=self.profile.history_schema)))
            else:
                self.producers[p].freeze_cutoff("collection_window")
                if p == 0:
                    self.submit(p, c.Action("end_turn", schema=self.profile.action_schema), outside=True)
                    self.batch(p, dict(kind="consumed", day=1, decision_id=1, consumer="engine"),
                        dict(kind="issued", day=1, decision_id=1, command_index=0),
                        dict(kind="acknowledged", day=1, decision_id=1, command_index=0, accepted=True),
                        dict(kind="result", day=1, decision_id=1,
                             result=dict(status="turn_ended", accepted=True, position=None, movement=None,
                                         destination_reached=None)))
                elif late_cutoff_view:
                    self.batch(p, dict(kind="superseded", day=1, decision_id=1, reason="visible_update"),
                        dict(kind="view", day=1,
                             observation=c.encode_observation(observation(), schema=self.profile.history_schema)),
                        self.decision(2, 2))
        self.write()

    def decision(self, native, revision):
        return dict(kind="decision", day=1, decision_id=native, revision=revision,
                    observation=c.encode_observation(observation(), schema=self.profile.history_schema))

    def engine(self, player, message):
        self.rows[player].append(dict(t=len(self.rows[player]) * 0.1, direction="engine", message=message))
        self.producers[player].ingest(message)

    def batch(self, player, *records):
        self.engine(player, dict(protocol_version=self.profile.protocol, type="contextual", episode="1",
                                player=player, records=list(records)))

    def submit(self, player, action, outside=False):
        producer = self.producers[player]
        message = dict(producer.response(action), delivered=True, outside_learner_prefix=outside)
        self.rows[player].append(dict(t=len(self.rows[player]) * 0.1, direction="policy", message=message))
        producer.submitted(action)

    def write(self):
        for p, color in enumerate(("red", "blue")):
            path = self.directory / f"agents/{color}/bridge.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self.rows[p]))
        self.final = dict(schema="heroes-contextual-finalization-evidence-v1", episode="1",
            returncode=0 if self.natural else -15, declared_stop=not self.natural,
            shutdown_verified=not self.natural, expected_stop_signals=[15, 9],
            reason=None if self.natural else "collection_window", streams_complete=[True, True],
            engine_evidence_error=None, natural_terminal_after_cutoff=None, identity=self.identity,
            socket_outcomes=self.outcomes if self.natural else [None, None],
            engine_verified_outcomes=dict(zip(("red", "blue"), self.outcomes)) if self.natural else None,
            continuation=dict(limit_per_player=8, actions=[p.continuation_steps for p in self.producers],
                allowed_actions=getattr(self, "allowed_actions", ["back", "close", "close_capability", "end_turn"] if self.profile == c.EXPANDED_PROFILE
                    else ["back", "close", "end_turn"] if self.profile != c.WINDOWS_PROFILE else ["close", "end_turn"]),
                learner_excluded=True),
            raw_bridge_sha256={color: sha(self.directory / f"agents/{color}/bridge.jsonl") for color in ("red", "blue")})
        if self.profile.dialogs:
            self.final["dialog_cutoff_drain"] = None
        dump(self.directory / "contextual-finalization.json", self.final)
        ref = "contextual-finalization.json#sha256=" + sha(self.directory / "contextual-finalization.json")
        self.prefixes = [p.history.digest() for p in self.producers]
        pair = tuple(p.history for p in self.producers)
        if self.natural:
            c.finalize_natural(pair, engine_outcomes=tuple(self.outcomes), socket_outcomes=tuple(self.outcomes),
                               process_exit_code=0, evidence_ref=ref)
        else:
            c.finalize_cutoff(pair, evidence_ref=ref, histories_complete=(True, True), failure_free=True,
                shutdown_verified=True, continuation_limit=8,
                continuation_steps=tuple(p.continuation_steps for p in self.producers))
        declarations = {}
        for p, color in enumerate(("red", "blue")):
            producer = self.producers[p]
            names = [f"agents/{color}/{self.profile.history_filename}", f"agents/{color}/live-{self.profile.history_filename}"]
            for name, history in zip(names, (producer.history, producer.live)):
                (self.directory / name).write_text("".join(json.dumps(c.encode_frame(frame), sort_keys=True,
                    separators=(",", ":")) + "\n" for frame in history.frames))
            count = Counter(f.kind for f in producer.live.frames)
            declarations[color] = dict(ready=True, error=None, terminal={"outcome": producer.terminal} if producer.terminal else None,
                actions_sent=count["submitted"], actions_accepted=sum(f.kind == "acknowledged" and f.accepted for f in producer.live.frames),
                actions_rejected=0, responses_undelivered=0,
                live_evidence=dict(path=names[1], sha256=sha(self.directory / names[1]), continuation_steps=producer.continuation_steps,
                                   learner_input=False))
            declarations[color][self.profile.history_key] = dict(path=names[0], sha256=sha(self.directory / names[0]),
                canonical_sha256=producer.history.digest(), prefix_sha256=self.prefixes[p],
                boundary=producer.history.boundary, complete=True)
        self.manifest = dict(schema_version=1, kind="contextual-headless-self-play", episode="1",
            environment=self.profile.environment, history_schema=self.profile.history_schema,
            action_schema=self.profile.action_schema, identity=self.identity, returncode=self.final["returncode"],
            status="completed" if self.natural else "truncated",
            termination_reason="natural_terminal" if self.natural else "collection_window", evidence_ref=ref,
            outcomes_by_player=dict(zip(("red", "blue"), self.outcomes)) if self.natural else dict.fromkeys(("red", "blue")),
            rewards_by_player={color: 1 if outcome == "win" else -1 for color, outcome in zip(("red", "blue"), self.outcomes)}
                if self.natural else dict.fromkeys(("red", "blue")), players=declarations)
        if self.profile.dialogs:
            self.manifest["dialog_cutoff_drain"] = None
        dump(self.directory / "manifest.json", self.manifest)

    def rebind(self):
        """Update outer hashes for a deliberate tamper; never rebuild histories."""
        for color in ("red", "blue"):
            self.final["raw_bridge_sha256"][color] = sha(self.directory / f"agents/{color}/bridge.jsonl")
        dump(self.directory / "contextual-finalization.json", self.final)
        self.manifest["evidence_ref"] = "contextual-finalization.json#sha256=" + sha(self.directory / "contextual-finalization.json")
        dump(self.directory / "manifest.json", self.manifest)

    def change_bridge(self, mutate, player=0):
        mutate(self.rows[player])
        path = self.directory / f"agents/{('red', 'blue')[player]}/bridge.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.rows[player]))
        self.rebind()

    def verify(self):
        return verify_histories(self.directory, self.manifest, self.final)


class HistoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_natural_both_versions_and_outcome_orders_with_postterminal_views(self):
        for town in (False, True):
            for reverse in (False, True):
                with self.subTest(town=town, reverse=reverse):
                    fixture = Fixture(self.root / f"{town}-{reverse}", natural=True, town=town, reverse=reverse)
                    result = fixture.verify()
                    self.assertTrue(result["verified"])
                    self.assertFalse(result["native_replay_verified"])
                    self.assertEqual(result["boundary"], "natural")
                    for player in result["players"].values():
                        self.assertEqual(player["learner"], player["live"])
                        self.assertEqual(player["counts"]["view"], 1)
                        self.assertEqual(player["counts"]["delivered_actions"], 2)

    def test_cutoff_reconstructs_zero_continuation_prefix_before_late_native_view(self):
        for town in (False, True):
            fixture = Fixture(self.root / str(town), town=town, late_cutoff_view=True)
            result = fixture.verify()
            self.assertEqual(result["boundary"], "cutoff")
            self.assertEqual([p["continuation_steps"] for p in result["players"].values()], [1, 0])
            for p in result["players"].values():
                self.assertIsNotNone(p["cutoff_decision_seq"])
                self.assertNotEqual(p["learner"]["sha256"], p["live"]["sha256"])

    def test_armies_profile_reconstructs_both_boundaries_without_legacy_downgrade(self):
        for natural in (False, True):
            fixture = Fixture(self.root / str(natural), natural=natural, profile=c.ARMIES_PROFILE)
            result = fixture.verify()
            self.assertTrue(result["verified"])
            self.assertEqual(result["boundary"], "natural" if natural else "cutoff")
            fixture.manifest["history_schema"] = c.TOWN_SCHEMA
            with self.assertRaises(ValueError): fixture.verify()

    def test_delivered_flags_request_identity_and_continuation_cannot_be_forged(self):
        mutations = [lambda m: m.update(delivered=False), lambda m: m.update(outside_learner_prefix=True),
                     lambda m: m.update(decision_id=9), lambda m: m.update(revision=99),
                     lambda m: m.update(player=1), lambda m: m.update(protocol_version=5)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                fixture = Fixture(self.root / str(index))
                fixture.change_bridge(lambda rows: mutate(next(r["message"] for r in rows if r["direction"] == "policy")))
                with self.assertRaises(ValueError): fixture.verify()

    def test_continuation_cannot_turn_into_player_move(self):
        fixture = Fixture(self.root)
        def tamper(rows):
            message = next(r["message"] for r in rows if r["direction"] == "policy" and r["message"]["outside_learner_prefix"])
            message["action"] = c.encode_action(c.Action("move_to", "local-hero", 1, 1))
        fixture.change_bridge(tamper)
        with self.assertRaisesRegex(ValueError, "technical continuation"): fixture.verify()

    def test_new_native_event_cannot_be_hidden_by_persisted_histories(self):
        fixture = Fixture(self.root, natural=True)
        fixture.change_bridge(lambda rows: rows[-1]["message"]["records"].append(
            dict(kind="view", day=1, observation=c.encode_observation(observation()))))
        with self.assertRaisesRegex(ValueError, "prefix digest"): fixture.verify()

    def test_persisted_history_with_recomputed_file_hash_is_still_compared(self):
        fixture = Fixture(self.root, natural=True)
        target = fixture.manifest["players"]["red"][fixture.profile.history_key]
        path = self.root / target["path"]
        rows = path.read_text().splitlines()
        value = json.loads(rows[0]); value["revision"] = 91
        rows[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(rows) + "\n")
        target["sha256"] = sha(path)
        dump(self.root / "manifest.json", fixture.manifest)
        with self.assertRaisesRegex(ValueError, "History differs from raw bridge"): fixture.verify()

    def test_wrong_boundary_hash_and_continuation_counts_fail(self):
        for index, mutate in enumerate((lambda f: f.manifest["players"]["blue"][f.profile.history_key].update(prefix_sha256="0" * 64),
                                       lambda f: f.final["continuation"].update(actions=[0, 0]),
                                       lambda f: f.final.update(streams_complete=[True, False]),
                                       lambda f: f.final.update(returncode=1),
                                       lambda f: f.manifest.update(status="failed"))):
            fixture = Fixture(self.root / str(index)); mutate(fixture); fixture.rebind()
            with self.subTest(index=index), self.assertRaises(ValueError): fixture.verify()

    def test_source_documents_must_match_supplied_objects_and_digests(self):
        fixture = Fixture(self.root)
        changed = deepcopy(fixture.final); changed["identity"] = {"unrelated": True}
        with self.assertRaisesRegex(ValueError, "Supplied finalization"): verify_histories(self.root, fixture.manifest, changed)
        changed = deepcopy(fixture.manifest); changed["episode"] = "2"
        with self.assertRaisesRegex(ValueError, "Supplied manifest"): verify_histories(self.root, changed, fixture.final)
        (self.root / "contextual-finalization.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "digest mismatch"): fixture.verify()

    def test_partial_and_duplicate_key_bridge_records_fail_even_if_rehashed(self):
        for index, suffix in enumerate((b'{"t":1}', b'{"t":1,"t":2,"direction":"engine","message":{}}\n')):
            fixture = Fixture(self.root / str(index))
            with (fixture.directory / "agents/red/bridge.jsonl").open("ab") as out: out.write(suffix)
            fixture.rebind()
            with self.subTest(index=index), self.assertRaises(ValueError): fixture.verify()

    def test_symlinked_or_relocated_saved_history_is_rejected(self):
        fixture = Fixture(self.root)
        target = fixture.manifest["players"]["red"][fixture.profile.history_key]
        path = self.root / target["path"]
        other = self.root / "other.jsonl"; path.rename(other); path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlink"): fixture.verify()


if __name__ == "__main__":
    unittest.main()
