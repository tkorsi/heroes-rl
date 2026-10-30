"""Synthetic protocol/lifecycle regressions; these do not validate native games."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/headless"))
from experiments.learner import contextual as c
from contextual_protocol import ContextualProducer, NativeFailure, loads
from env import HeadlessSelfPlayEnv, Limits, _Side
from contextual_env import CONTINUATION_LIMIT
from test_contextual_evidence import write_barriers


def obs():
    return c.encode_observation(c.Observation((4, 4, 1),
        (c.Tile((1, 1, 0), 2, 0, False, False),), (), (0,) * 7,
        (c.Hero("local-hero", (1, 1, 0), 100),), (c.Town("local-town", (1, 1, 0)),)))


def ready(player=0):
    return dict(protocol_version=4, type="ready", episode="1", player=player,
                interface=c.ENVIRONMENT, history_schema=c.SCHEMA, action_schema=c.ACTION_SCHEMA)


def batch(*records, player=0):
    return dict(protocol_version=4, type="contextual", episode="1", player=player, records=list(records))


def decision(native=0, revision=0, day=1):
    return dict(kind="decision", day=day, decision_id=native, revision=revision, observation=obs())


def producer(player=0):
    result = ContextualProducer("1", player)
    result.ingest(ready(player))
    result.ingest(batch(decision(), player=player))
    return result


def complete_noop(p):
    action = c.Action("move_to", "local-hero", 1, 1)
    p.submitted(action)
    p.ingest(batch(dict(kind="consumed", day=1, decision_id=0, consumer="adapter"),
        dict(kind="result", day=1, decision_id=0,
             result=dict(status="reached", accepted=True, position=[1, 1, 0], movement=100, destination_reached=True)), player=p.player))


class ProtocolTests(unittest.TestCase):
    def test_schema_and_cross_player_fail_closed(self):
        for mutate in (lambda r: r.update(protocol_version=3), lambda r: r.update(player=1),
                       lambda r: r.update(episode="2"), lambda r: r.update(secret=99)):
            p = ContextualProducer("1", 0)
            raw = ready()
            mutate(raw)
            with self.assertRaises(ValueError): p.ingest(raw)
            self.assertFalse(p.ready)

    def test_nested_hidden_fields_and_python_owned_kinds_rejected(self):
        for record in (dict(kind="submitted", day=1, decision_id=0, action=asdict(c.Action("end_turn"))),
                       dict(kind="finalized", day=1), dict(kind="cutoff", day=1, decision_id=0, reason="budget")):
            with self.assertRaises(ValueError): producer().ingest(batch(record))
        p = ContextualProducer("1", 0)
        p.ingest(ready())
        rec = decision()
        rec["observation"]["enemy_current_position"] = [3, 3, 0]
        with self.assertRaises(ValueError): p.ingest(batch(rec))
        self.assertEqual(p.history.frames, ())

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in ('{"player":0,"player":1}', '{"value":NaN}', '{"value":Infinity}'):
            with self.assertRaises(ValueError): loads(raw)

    def test_offer_support_and_transport_id_mapping(self):
        p = producer()
        before = p.history.digest()
        response = p.response(c.Action("move_to", "local-hero", 3, 3))
        self.assertEqual(response["decision_id"], 0)
        self.assertEqual(p.history.digest(), before)
        p.submitted(c.Action("move_to", "local-hero", 3, 3))
        self.assertFalse(p.live.consumed)
        p.ingest(batch(dict(kind="consumed", day=1, decision_id=0, consumer="adapter")))
        self.assertTrue(p.live.consumed)
        self.assertEqual(p.history.frames[-1].decision_seq, 0)
        self.assertNotIn("decision_id", c.encode_frame(p.history.frames[-1]))

    def test_result_without_consumption_and_unknown_reference_rejected(self):
        p = producer()
        p.submitted(c.Action("end_turn"))
        with self.assertRaises(ValueError):
            p.ingest(batch(dict(kind="issued", day=1, decision_id=0, command_index=0)))
        with self.assertRaises(ValueError):
            p.ingest(batch(dict(kind="consumed", day=1, decision_id=2, consumer="adapter")))

    def test_supersession_then_transient_view_is_preserved(self):
        p = producer()
        with self.assertRaises(ValueError): p.ingest(batch(dict(kind="view", day=1, observation=obs())))
        p.ingest(batch(dict(kind="superseded", day=1, decision_id=0, reason="visible_update"),
                       dict(kind="view", day=1, observation=obs()), decision(1, 1)))
        self.assertEqual([f.kind for f in p.history.frames], ["decision", "superseded", "view", "decision"])
        self.assertEqual(p.response(c.Action("end_turn"))["decision_id"], 1)

    def test_native_error_has_no_terminal_or_reward(self):
        p = producer()
        with self.assertRaises(NativeFailure):
            p.ingest(dict(protocol_version=4, type="error", episode="1", player=0, message="unsupported notice"))
        self.assertIsNone(p.terminal)
        self.assertEqual(p.history.transitions(), ())

    def test_cutoff_freezes_prefix_while_continuation_remains_separate(self):
        p = producer()
        p.freeze_cutoff("collection_window")
        frozen = p.history.frames
        complete_noop(p)
        p.ingest(batch(decision(1, 1)))
        self.assertEqual(p.history.frames, frozen)
        self.assertGreater(len(p.live.frames), len(frozen))
        self.assertEqual(p.continuation_steps, 1)
        p.fail("reader_error")
        self.assertEqual(p.history.boundary, "failure")
        self.assertFalse(any(t.eligible for t in p.history.transitions()))


class Process:
    """Synthetic lifecycle object; never a fake claim of a native engine run."""
    def __init__(self, exited=0): self.exited = exited
    def poll(self): return self.exited
    def wait(self, timeout=None): return self.exited


class Connection:
    def __init__(self, fail=False): self.fail, self.sent = fail, []
    def sendall(self, data):
        if self.fail: raise BrokenPipeError("injected")
        self.sent.append(data)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.env = HeadlessSelfPlayEnv(SimpleNamespace(), self.root, interface=c.ENVIRONMENT)
        e = self.env
        e.episode, e.seed, e.episode_dir = 1, 12345, self.root
        e.started = e.request_time = time.monotonic()
        e.created_at, e.launch_info = "synthetic-test", {}
        e.costs = {key: 0 for key in ("respond", "ingest", "ledger", "act")}
        e.reason = e.detail = None
        e.stopping = threading.Event()
        e.shutdown_deadline = None
        e.events = queue.Queue()
        e.engine_wait = e.agent_wait = 0
        e.step_latencies = []
        e.process = Process()
        (self.root / "logs").mkdir()
        e.log = (self.root / "logs/process.log").open("w")
        for index, color in enumerate(("red", "blue")):
            p = producer(index)
            side = _Side(index, color, p, p.history, None)
            side.connection = Connection()
            side.eof = True
            folder = self.root / "agents" / color
            folder.mkdir(parents=True)
            side.ledger = (folder / "bridge.jsonl").open("x")
            e.sides[index] = side
        e._engine_evidence = lambda log: ({"synthetic": True}, None)
        e._identity = lambda: dict(engine_revision="synthetic", bridge_source_sha256="a" * 64, map_sha256="b" * 64)
        write_barriers(self.root)
        for name in ("replay.jsonl", "contextual-replay.jsonl"):
            (self.root / "engine" / name).write_text('{"synthetic_unit_fixture":true}\n')
        with (self.root / "engine/contextual-replay.jsonl").open("a") as stream:
            stream.write('{"seq":0,"kind":"close","barrier_id":1,"client_packets":123}\n')

    def tearDown(self):
        # No subprocess or real socket exists in these explicit unit fixtures.
        for side in self.env.sides.values():
            if not side.ledger.closed: side.ledger.close()
        if not self.env.log.closed: self.env.log.close()
        self.directory.cleanup()

    def audit(self, result):
        spec = importlib.util.spec_from_file_location("contextual_test_audit", ROOT / "validation/probes/audit_contextual_histories.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        report = mod.audit(self.root / "agents/red/history-v2.jsonl", self.root / "agents/blue/history-v2.jsonl", self.root / "manifest.json")
        self.assertTrue(report["consistent"], report["failures"])
        self.assertTrue(report["manifest_bound"])
        return report

    def test_opt_in_keeps_default_legacy_class(self):
        legacy = HeadlessSelfPlayEnv(SimpleNamespace(), self.root)
        self.assertIs(type(legacy), HeadlessSelfPlayEnv)
        self.assertIsInstance(self.env, HeadlessSelfPlayEnv)
        with self.assertRaises(ValueError): HeadlessSelfPlayEnv(SimpleNamespace(), self.root, interface="unknown")

    def test_failed_send_never_records_submitted_or_consumed(self):
        e = self.env
        side = e.sides[0]
        side.connection = Connection(fail=True)
        e._issue(0)
        before = e.history_v2("red")
        e._send_current(c.Action("end_turn"))
        self.assertEqual(e.history_v2("red"), before)
        self.assertEqual(e.reason, "action_delivery_failed")

    def test_step_on_preexisting_crash_does_not_submit(self):
        e = self.env
        e.process.exited = -9
        e._issue(0)
        result = e.step(c.Action("end_turn"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(e.sides[0].connection.sent, [])
        self.assertFalse(any(f.kind == "submitted" for f in e.sides[0].validator.history.frames))
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_invalid_step_does_not_consume_pending_error(self):
        e = self.env
        request = e._issue(0)
        before = e.history_v2("red")
        e.events.put(("bridge_error", 1, "bad stream", time.monotonic()))
        with self.assertRaises(ValueError): e.step(c.Action("close"))
        self.assertIs(e.request, request)
        self.assertEqual(e.history_v2("red"), before)
        self.assertEqual(e.events.qsize(), 1)

    def test_held_decision_supersession_returns_replacement_without_submission(self):
        e = self.env
        e.process.exited = None
        for side in e.sides.values(): side.eof = False
        e._issue(0)
        e.events.put(("message", 0, batch(dict(kind="superseded", day=1, decision_id=0, reason="visible_update"),
                                        dict(kind="view", day=1, observation=obs()), decision(1, 1)), time.monotonic()))
        replacement = e.step(c.Action("end_turn"))
        self.assertFalse(replacement.done)
        self.assertEqual(replacement.frames[-1]["kind"], "decision")
        self.assertEqual(replacement.frames[-1]["seq"], 3)
        self.assertEqual(e.sides[0].connection.sent, [])
        self.assertFalse(any(f.kind == "submitted" for f in e.sides[0].validator.history.frames))

    def test_invalid_action_keeps_request_and_history_identity(self):
        e = self.env
        request = e._issue(0)
        before = e.history_v2("red")
        with self.assertRaises(ValueError): e._send_current(c.Action("close"))
        self.assertIs(e.request, request)
        self.assertEqual(e.history_v2("red"), before)
        self.assertEqual(e.sides[0].connection.sent, [])

    def test_successful_send_is_not_consumption(self):
        e = self.env
        e._issue(0)
        e._send_current(c.Action("end_turn"))
        self.assertEqual(e.sides[0].validator.live.frames[-1].kind, "submitted")
        self.assertFalse(e.sides[0].validator.live.consumed)

    def test_natural_finalization_and_manifest_are_bound_after_drain(self):
        e = self.env
        for index, side in e.sides.items():
            complete_noop(side.validator)
            side.actions_sent = 1
            side.validator.ingest(batch(dict(kind="terminal", day=1, outcome="win" if index == 0 else "loss"), player=index))
        e.log.write("Experiment player 0 won. Terminal received.\nExperiment player 1 lost. Terminal received.\nExperiment self-play complete. Ending game.\n")
        result = e._finish(0)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.rewards, {"red": 1, "blue": -1})
        self.assertTrue(self.audit(result)["scoring_binding_verified"])
        self.assertFalse(result.manifest["deterministic_replay_verified"])
        self.assertEqual(result.manifest["replay"]["status"], "recorded")

    def _terminal_pair(self):
        for index, side in self.env.sides.items():
            complete_noop(side.validator)
            side.actions_sent = 1
            side.validator.ingest(batch(dict(kind="terminal", day=1, outcome="win" if index == 0 else "loss"), player=index))
        self.env.log.write("Experiment player 0 won. Terminal received.\nExperiment player 1 lost. Terminal received.\nExperiment self-play complete. Ending game.\n")

    def test_late_error_envelope_after_terminal_and_exit_is_not_swallowed(self):
        e = self.env
        self._terminal_pair()
        e.events.put(("exit", None, 0, time.monotonic()))
        e.events.put(("message", 1, dict(protocol_version=4, type="error", episode="1", player=1,
                                      message="unsupported final notice"), time.monotonic()))
        result = e._finish(0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "unsupported_interface")
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_clean_exit_and_complete_sockets_without_barrier_are_unscored(self):
        self._terminal_pair()
        (self.root / "engine/terminal-barrier-client.jsonl").unlink()
        result = self.env._finish(0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "terminal_barrier_invalid")
        self.assertEqual(result.rewards, {"red": None, "blue": None})
        self.assertEqual(result.bootstrap, {"red": None, "blue": None})
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_late_engine_failure_after_valid_barrier_is_unscored(self):
        self._terminal_pair()
        self.env.log.write("Headless packet handling failed: injected after receipt\n")
        result = self.env._finish(0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.rewards, {"red": None, "blue": None})
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_explicit_unsupported_before_terminal_keeps_specific_reason(self):
        e = self.env
        e._ingest(e.sides[0], dict(protocol_version=4, type="error", episode="1", player=0,
                                message="Unsupported scenario interaction: special dialog"), time.monotonic())
        result = e._finish(1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "unsupported_interface")
        self.assertEqual(result.rewards, {"red": None, "blue": None})
        self.assertEqual(result.bootstrap, {"red": None, "blue": None})
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_authoritative_late_failure_log_overrides_cutoff(self):
        e = self.env
        e.cutoff_reason = e.reason = "collection_window"
        for side in e.sides.values(): e._capture_cutoff(side)
        e.process.exited = None
        e.log.write("Experiment failure: ExplorerRL unsupported final interaction: informational notice\n")
        with patch("contextual_env.stop_process_group", lambda p: setattr(p, "exited", -15)):
            result = e._finish(None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "engine_error")
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_postterminal_view_is_included_before_final_proof(self):
        e = self.env
        self._terminal_pair()
        changed = obs()
        changed["resources"][6] = 500
        e.events.put(("exit", None, 0, time.monotonic()))
        e.events.put(("message", 0, batch(dict(kind="view", day=1, observation=changed)), time.monotonic()))
        result = e._finish(0)
        self.assertEqual(result.status, "completed")
        frames = e.sides[0].validator.history.frames
        self.assertEqual([f.kind for f in frames[-3:]], ["terminal", "view", "finalized"])
        self.assertEqual(frames[-2].observation.resources[6], 500)
        self.assertTrue(self.audit(result)["scoring_binding_verified"])

    def test_reader_error_after_exit_invalidates_both_provisional_cutoffs(self):
        e = self.env
        e.cutoff_reason = "collection_window"
        for side in e.sides.values(): e._capture_cutoff(side)
        e.reason = "collection_window"
        e.events.put(("exit", None, 0, time.monotonic()))
        e.events.put(("bridge_error", 1, "trailing partial JSON", time.monotonic()))
        result = e._finish(None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.rewards, {"red": None, "blue": None})
        self.assertTrue(all(p["history_v2"]["boundary"] == "failure" for p in result.manifest["players"].values()))
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_pair_cutoff_finalizes_only_after_explicit_owned_shutdown(self):
        e = self.env
        e.cutoff_reason = e.reason = "collection_window"
        for side in e.sides.values(): e._capture_cutoff(side)
        e.process.exited = None
        def stop(process): process.exited = -15
        with patch("contextual_env.stop_process_group", stop):
            result = e._finish(None)
        self.assertEqual(result.status, "truncated")
        self.assertTrue(all(result.bootstrap.values()))
        self.assertTrue(self.audit(result)["record_contract_eligible"])

    def test_reset_rejects_edited_loaded_source_before_preflight_or_launch(self):
        e = self.env
        # Exercise reset ordering without closing the intentionally synthetic
        # socket objects used by this unit fixture.
        e.close = lambda: None
        e._imported_sources = {"experiments/headless/contextual_env.py": "0" * 64}
        e._preflight_contextual = lambda: self.fail("Preflight ran before source guard")
        with self.assertRaisesRegex(ValueError, "restart Python"):
            e.reset(episode=2, seed=12345)

    def test_identity_freezes_binary_and_python_hashes_before_finish(self):
        e = self.env
        del e._identity  # use the real identity method rather than fixture metadata
        calls = []
        def native_identity():
            calls.append(True)
            return {"binary_sha256": "c" * 64, "map": {"sha256": "b" * 64}}
        e.launcher.identity = native_identity
        first = e._identity()
        e.launcher.identity = lambda: self.fail("Identity was recomputed after launch")
        self.assertIs(e._identity(), first)
        self.assertEqual(len(calls), 1)
        self.assertTrue(first["captured_before_launch"])
        self.assertEqual(first["contextual_python_sha256"], e._imported_sources)
        self.assertEqual(first["binary_sha256"], "c" * 64)

    def test_cached_history_returns_detached_mutable_json(self):
        e = self.env
        first = e.history_v2("red")
        first[0]["observation"]["heroes"][0]["movement"] = 999
        second = e.history_v2("red")
        self.assertEqual(second[0]["observation"]["heroes"][0]["movement"], 100)
        self.assertEqual(e.sides[0].validator.history.frames[0].observation.heroes[0].movement, 100)
        complete_noop(e.sides[0].validator)
        self.assertEqual(e.history_v2("red")[-1]["kind"], "result")

    def test_engine_failure_racing_owned_stop_is_not_a_cutoff(self):
        e = self.env
        e.cutoff_reason = e.reason = "collection_window"
        for side in e.sides.values(): e._capture_cutoff(side)
        e.process.exited = None
        with patch("contextual_env.stop_process_group", lambda p: setattr(p, "exited", 1)):
            result = e._finish(None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "unverified_cutoff_shutdown")
        self.assertEqual(result.bootstrap, {"red": None, "blue": None})
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_preexisting_crash_before_truncate_has_no_bootstrap(self):
        e = self.env
        e.process.exited = -9
        e._issue(0)
        result = e.truncate()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.bootstrap, {"red": None, "blue": None})
        self.assertFalse(self.audit(result)["learner_eligible"])

    def test_one_cutoff_cannot_finalize_without_other_real_decision(self):
        e = self.env
        e.cutoff_reason = e.reason = "collection_window"
        e._capture_cutoff(e.sides[0])
        e.process.exited = None
        with patch("contextual_env.stop_process_group", lambda p: setattr(p, "exited", -15)):
            result = e._finish(None)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "unverified_paired_cutoff")
        self.assertFalse(self.audit(result)["record_contract_eligible"])

    def test_finalizer_bound_failure_never_submits_another_action(self):
        e = self.env
        e.cutoff_reason = "collection_window"
        side = e.sides[0]
        e._capture_cutoff(side)
        side.validator.continuation_steps = CONTINUATION_LIMIT
        e._issue(0, continuation=True)
        e._send_current(c.Action("end_turn"), continuation=True)
        self.assertEqual(e.reason, "cutoff_continuation_limit")
        self.assertEqual(side.connection.sent, [])


if __name__ == "__main__": unittest.main()
