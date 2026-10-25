"""In-process backend adapter with a scripted engine; no game, replay or training claim.

The fake engine hands out the same tiny native-shaped stream as the socket
synthetic in test_builtin_rollout, through take/submit/finish. Native proof
producers are mocked exactly as there; the real adapter, files, model and
reconstruction run.
"""
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import importlib.machinery
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.headless import builtin_evidence as e, builtin_replay as replay
from experiments.headless import embedded_env as ee, embedded_evidence as ev
from experiments.headless.builtin_replay_history import verify_history
from experiments.learner import contextual as c, contextual_model as m
from experiments.learner.builtin_rollout import BuiltinCollector, audit_persisted_builtin_policy
from tests import test_builtin_env as boundaries
from tests import test_builtin_replay_diagnostic as diagnostic

initial = boundaries.initial


def delivery(generation=1, decision=0, messages=(), needs_action=False, terminal=False, failure=None,
             closed=False, decode_error=None):
    return {"generation": generation, "decision": decision, "messages": list(messages), "needs_action": needs_action,
            "terminal": terminal, "closed": closed, "timed_out": not (messages or needs_action or terminal or failure or closed),
            "failure": failure, "decode_error": decode_error}


def report(*, exit_code, natural, host_stopped, clean=True, messages=(), channel_failure=None, decode_error=None,
           poisoned=False):
    return {"exit_code": exit_code, "natural": natural, "host_stopped": host_stopped, "shutdown_deadline": False,
            "close_clean": clean, "poisoned": poisoned, "experiment_failed": False, "channel_failure": channel_failure,
            "close_error": None, "messages": list(messages), "decode_error": decode_error,
            "opponent": {"module": "Nullkiller", "player": 1, "difficulty": 0, "created": True, "released": True,
                         "module_reset": True, "turns": 1, "turn_seconds": 0.01, "battles": 0, "battle_seconds": 0.0},
            "wait_seconds": 0.0, "close_seconds": 0.001}


class FakeEngine:
    """Scripted take/submit/finish with the native channel's token rules."""

    profile_environment = {"XDG_DATA_HOME": "/synthetic"}

    def __init__(self, boundary):
        self.boundary = boundary
        self.active = self.poisoned = False
        self.generation, self.responses, self.resets, self.finishes = 0, [], [], []

    def require_ready(self):
        if self.active or self.poisoned:
            raise ValueError("not ready")

    def reset(self, **kwargs):
        self.resets.append(kwargs)
        self.active, self.generation = True, self.generation + 1
        Path(kwargs["capture_directory"]).mkdir()
        Path(kwargs["log_path"]).write_text("")
        self.folder = Path(kwargs["capture_directory"]).parent
        player, episode = kwargs["learner"], kwargs["episode"]
        self.player = player
        self.base = {"protocol_version": 9, "episode": episode, "player": player}
        observation = c.encode_frame(initial(player).frames[0])["observation"]
        self.decision = lambda index: {"kind": "decision", "day": 1, "decision_id": index, "revision": index,
                                       "observation": observation}
        ready = {**self.base, "type": "ready", "interface": c.DIALOGS_ENVIRONMENT,
                 "history_schema": c.DIALOGS_SCHEMA, "action_schema": c.DIALOGS_ACTION_SCHEMA}
        if self.boundary == "startup_failure":
            self.script = deque([delivery(self.generation, 0, failure="Headless game start failed")])
        elif self.boundary == "decode_error":
            self.script = deque([delivery(self.generation, 0, [ready], decode_error="UnicodeDecodeError: invalid")])
        else:
            self.script = deque([delivery(self.generation, 1, [ready, self.batch(self.decision(0))], needs_action=True)])
        self.expected = (self.generation, 1)

    def batch(self, *records):
        return {**self.base, "type": "contextual", "records": list(records)}

    def take(self, timeout, until_boundary):
        if not self.active:
            raise RuntimeError("No active native episode")
        return self.script.popleft() if self.script else delivery(self.generation, 0)

    def submit_text(self, generation, decision, text):
        response = json.loads(text)
        if (generation, decision) != self.expected:
            raise ValueError("Stale, duplicate or nondecision native step token")
        self.expected = None
        self.responses.append(response)
        if response["action"]["kind"] != "end_turn":
            raise AssertionError("Only offered action is end_turn")
        result = self.batch({"kind": "consumed", "day": 1, "decision_id": 0, "consumer": "engine"},
                            {"kind": "issued", "day": 1, "decision_id": 0, "command_index": 0},
                            {"kind": "acknowledged", "day": 1, "decision_id": 0, "command_index": 0, "accepted": True},
                            {"kind": "result", "day": 1, "decision_id": 0,
                             "result": {"status": "turn_ended", "accepted": True, "position": None,
                                        "movement": None, "destination_reached": None}})
        if self.boundary == "natural":
            for name, record in zip(e.native.FILES, boundaries.BuiltinReceiptTests().records(self.player)):
                record["episode"] = self.base["episode"]
                (self.folder / name).write_text(json.dumps(record) + "\n")
            self.script.append(delivery(self.generation, 2, [result, self.batch(
                {"kind": "terminal", "day": 1, "outcome": ("win", "loss")[self.player]})], terminal=True))
        elif self.boundary == "failure_after_action":
            error = {**self.base, "type": "error", "message": "unsupported fixture interaction"}
            self.script.append(delivery(self.generation, 1, [result, error], failure="ExplorerRL: unsupported fixture interaction"))
        else:
            self.script.append(delivery(self.generation, 2, [result, self.batch(self.decision(1))], needs_action=True))
            self.expected = (self.generation, 2)

    def finish(self, natural, timeout):
        self.finishes.append((natural, timeout))
        self.active = False
        if self.boundary == "natural":
            return report(exit_code=0, natural=True, host_stopped=False)
        if self.boundary == "startup_failure":
            return report(exit_code=1, natural=False, host_stopped=False, clean=False,
                          channel_failure="Headless game start failed")
        if self.boundary == "failure_after_action":
            return report(exit_code=None, natural=False, host_stopped=True, clean=False,
                          channel_failure="ExplorerRL: unsupported fixture interaction", poisoned=True)
        if self.boundary == "decode_error":
            return report(exit_code=None, natural=False, host_stopped=True)
        return report(exit_code=None, natural=False, host_stopped=True)


class SyntheticEmbeddedLauncher(ee.EmbeddedLauncher):
    def __init__(self, root):
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        binary = root / "bin/vcmiclient"
        library = root / ("bin/AI/libExplorerRL" + suffix)
        library.parent.mkdir(parents=True)
        binary.write_bytes(b"heroes-contextual-terminal-barrier-v1")
        library.write_bytes(" ".join((c.DIALOGS_ENVIRONMENT, c.DIALOGS_SCHEMA, c.DIALOGS_ACTION_SCHEMA)).encode())
        for name in ("vcmireplay", "libvcmi" + suffix, "AI/libBattleAI" + suffix, "AI/libNullkiller" + suffix):
            (binary.parent / name).write_bytes(b"EXPLORER_RL_BUILTIN_OPPONENT")
        (root / "assets/Data").mkdir(parents=True)
        (root / "maps").mkdir()
        (root / "maps/synthetic.vmap").write_bytes(b"synthetic map")
        super().__init__(binary, root / "assets", root / "maps/synthetic.vmap")
        self.runtime = e.native.runtime_identity(binary.parent)

    def preflight(self): pass


class EmbeddedCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def run_case(self, root, player, boundary, engine=None, episode=1):
        launcher = SyntheticEmbeddedLauncher(root) if not (root / "bin").exists() else self.launcher
        self.launcher = launcher
        engine = engine or FakeEngine(boundary)
        engine.boundary = boundary
        identity = {"engine_revision": "synthetic-not-native", "map_sha256": "b"*64,
                    "native_runtime_sha256": launcher.runtime,
                    "backend": {"kind": ev.BACKEND, "protocol": ev.PROTOCOL}}
        opponent = e.opponent_identity(player, next(v for k, v in launcher.runtime.items() if "libNullkiller" in k))
        torch.manual_seed(172)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        collector = BuiltinCollector(model, "synthetic-frozen-checkpoint", learner_player=player,
                                     opponent_identity=opponent, policy_seed=17)
        env = ee.EmbeddedBuiltinEnv(launcher, root / "episodes", learner_player=player, engine=engine)
        with ExitStack() as stack:
            stack.enter_context(patch.object(env, "_identity", return_value=identity))
            stack.enter_context(patch.object(env, "_engine_evidence", return_value=({"synthetic": True}, None)))
            stack.enter_context(patch.object(e, "capture_files", return_value={"schema": e.REPLAY_SCHEMA, "recorded": True, "files": {}}))
            stack.enter_context(patch.object(ee.native, "terminal_capture_close"))
            try:
                collected = collector.run_episode(env, episode=episode, seed=11, max_requests=1)
            finally:
                env.close()
        (env.episode_dir / "builtin-rollout.json").write_text(json.dumps(collected.to_dict()))
        return collected, env, engine, model, opponent

    def test_both_seats_natural_and_cutoff_with_persisted_reconstruction(self):
        for player in (0, 1):
            for boundary in ("cutoff", "natural"):
                with self.subTest(player=player, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    got, env, engine, model, opponent = self.run_case(Path(tmp), player, boundary)
                    folder = env.episode_dir
                    self.assertEqual(len(got.decisions), 1)
                    self.assertEqual(len(engine.responses), 1)
                    self.assertEqual(engine.resets[0]["learner"], player)
                    self.assertEqual(engine.resets[0]["map"], "Maps/headless.vmap")
                    self.assertEqual(engine.resets[0]["policy_timeout"], env.limits.bridge_timeout)
                    self.assertEqual([natural for natural, _ in engine.finishes], [boundary == "natural"])
                    manifest = got.manifest
                    self.assertEqual(manifest["launch"]["backend"], ev.BACKEND)
                    self.assertIsNone(manifest["launch"]["command"])
                    self.assertEqual(manifest["launch"]["requested_settings"]["server"]["seed"], 11)
                    self.assertEqual(len(sorted((folder / "profile").rglob("Maps/headless.vmap"))), 1)
                    final = e.read_finalization(folder, manifest)
                    self.assertTrue(verify_history(folder, manifest, final)["verified"])
                    audit = audit_persisted_builtin_policy(folder, model, got.checkpoint, 17, opponent)
                    self.assertTrue(audit["passed"])
                    if boundary == "cutoff":
                        self.assertEqual(manifest["status"], "truncated")
                        self.assertIsNone(manifest["returncode"])
                        self.assertTrue(final["shutdown_verified"] and final["declared_stop"])
                        self.assertTrue(final["native_close"]["host_stopped"])
                        cold = deepcopy(manifest); del cold["identity"]["backend"]
                        with self.assertRaises(ValueError):
                            e._host_stop_proven(cold, final) or e.require(False, "socket rule needs a signal status")
                    else:
                        self.assertEqual(manifest["status"], "completed")
                        self.assertEqual(manifest["returncode"], 0)
                        self.assertEqual(manifest["rewards_by_player"], {"red": 1, "blue": -1})
                        self.assertTrue(ev.natural_closed(final))

    def test_one_engine_serves_consecutive_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = FakeEngine("cutoff")
            statuses = []
            for episode, (player, boundary) in enumerate([(0, "cutoff"), (1, "natural"), (0, "natural"), (1, "cutoff")], 1):
                got, *_ = self.run_case(Path(tmp), player, boundary, engine=engine, episode=episode)
                statuses.append(got.manifest["status"])
            self.assertEqual(statuses, ["truncated", "completed", "completed", "truncated"])
            self.assertEqual(engine.generation, 4)
            self.assertEqual(len(engine.finishes), 4)

    def test_failures_stay_unscored_with_their_stream_prefix(self):
        for boundary, reason in (("startup_failure", "process_exit_without_verified_terminal"), ("decode_error", "protocol_error"),
                                 ("failure_after_action", "unsupported_interface")):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                got, env, engine, *_ = self.run_case(Path(tmp), 0, boundary)
                manifest = got.manifest
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["termination_reason"], reason)
                self.assertEqual(manifest["outcomes_by_player"], dict.fromkeys(e.COLORS))
                self.assertEqual(engine.finishes[0][0], False)
                ledger = [json.loads(line) for line in (env.episode_dir / "agents/red/bridge.jsonl").read_text().splitlines()]
                if boundary == "decode_error":
                    self.assertEqual([r["message"]["type"] for r in ledger], ["ready"])
                if boundary == "failure_after_action":
                    self.assertEqual(ledger[-1]["message"]["type"], "error")

    def test_relative_output_and_foreign_launcher_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = SyntheticEmbeddedLauncher(Path(tmp))
            with self.assertRaisesRegex(ValueError, "absolute output"):
                ee.EmbeddedBuiltinEnv(launcher, Path("relative"), learner_player=0, engine=FakeEngine("cutoff"))
            with self.assertRaisesRegex(ValueError, "EmbeddedLauncher"):
                ee.EmbeddedBuiltinEnv(object(), Path(tmp) / "out", learner_player=0)
            with self.assertRaisesRegex(ValueError, "never launches"):
                launcher.launch(Path(tmp), 1, {}, None)


class EventOrderTests(unittest.TestCase):
    def events(self, engine):
        env = type("Env", (), {})()
        env.engine, env.learner_player, env.shutdown_deadline = engine, 0, None
        env.limits = ee.Limits(shutdown_seconds=5)
        env.process = ee.EmbeddedProcess()
        return ee.EmbeddedEvents(env), env

    def test_stream_end_keeps_socket_order_and_exit_only_when_reported(self):
        class Engine:
            def __init__(self, end):
                self.script = deque([delivery(messages=[{"type": "a"}], terminal=True)])
                self.end = end
            def take(self, timeout, until_boundary): return self.script.popleft()
            def finish(self, natural, timeout):
                self.natural, self.timeout = natural, timeout
                return self.end
        events, env = self.events(Engine(report(exit_code=0, natural=True, host_stopped=False, messages=[{"type": "late"}])))
        kinds = [events.get(1)[0] for _ in range(4)]
        self.assertEqual(kinds, ["message", "message", "eof", "exit"])
        self.assertTrue(env.process.finished and env.process.poll() == 0)
        with self.assertRaises(queue.Empty):
            events.get_nowait()
        events, env = self.events(Engine(report(exit_code=None, natural=True, host_stopped=True, clean=False,
                                                channel_failure="Native policy message queue overflow")))
        items = [events.get(1) for _ in range(3)]
        self.assertEqual([i[0] for i in items], ["message", "bridge_error", "eof"])
        self.assertIsNone(env.process.poll())

    def test_a_decision_is_answered_with_its_own_token_once(self):
        class Engine:
            def __init__(self):
                self.script = deque([delivery(messages=[{"type": "contextual"}]), delivery(generation=3, decision=7, needs_action=True)])
                self.submitted = []
            def take(self, timeout, until_boundary): return self.script.popleft() if self.script else delivery()
            def submit_text(self, generation, decision, text): self.submitted.append((generation, decision, json.loads(text)))
        engine = Engine()
        events, _ = self.events(engine)
        events.env.limits = ee.Limits(bridge_timeout=1)
        self.assertEqual(events.get_nowait()[0], "message")
        self.assertIsNone(events.submit({"type": "context_action"}))  # waits briefly for the native boundary
        self.assertEqual(engine.submitted, [(3, 7, {"type": "context_action"})])
        self.assertEqual(events.submit({"type": "context_action"}), "No native decision is waiting for a response")


    def test_an_answer_the_own_terminal_superseded_is_written_like_a_socket_line(self):
        class Engine:
            def __init__(self, applied):
                self.applied = applied
                self.script = deque([delivery(generation=2, decision=1, needs_action=True)])
            def take(self, timeout, until_boundary): return self.script.popleft() if self.script else delivery()
            def submit_text(self, generation, decision, text): return self.applied
        for applied in (True, False):
            with self.subTest(applied=applied):
                events, _ = self.events(Engine(applied))
                events.env.limits = ee.Limits(bridge_timeout=1)
                self.assertIsNone(events.submit({"type": "context_action"}))
                self.assertEqual(events.superseded, 0 if applied else 1)
        # The terminal arrived before the decision's native wait: no token was ever offered.
        class Ended:
            def take(self, timeout, until_boundary): return delivery(messages=[{"type": "contextual"}], terminal=True)
        events, _ = self.events(Ended())
        events.env.limits = ee.Limits(bridge_timeout=1)
        self.assertEqual(events.get_nowait()[0], "message")
        self.assertIsNone(events.submit({"type": "context_action"}))
        self.assertEqual(events.superseded, 1)
        # A failure, unlike the own terminal, still refuses the answer.
        class Failed:
            def take(self, timeout, until_boundary): return delivery(failure="Native policy response timeout")
        events, _ = self.events(Failed())
        events.env.limits = ee.Limits(bridge_timeout=1)
        self.assertEqual(events.submit({"type": "context_action"}), "No native decision is waiting for a response")

    def test_signal_handler_exceptions_propagate_and_nothing_is_lost(self):
        class Budget(RuntimeError): pass
        class EngineFailure(RuntimeError): pass
        class Engine:
            errors = (EngineFailure, ValueError)
            def __init__(self):
                self.takes, self.finishes = 0, 0
            def take(self, timeout, until_boundary):
                self.takes += 1
                if self.takes == 1:
                    raise Budget("Global wall-time budget expired")
                return delivery(messages=[{"type": "a"}], terminal=True)
            def finish(self, natural, timeout):
                self.finishes += 1
                if self.finishes == 1:
                    raise Budget("expired during the close")
                return report(exit_code=0, natural=True, host_stopped=False)
        events, env = self.events(Engine())
        with self.assertRaises(Budget):
            events.get(1)
        self.assertIsNone(events.end_pending)
        self.assertEqual(events.get(1)[0], "message")
        with self.assertRaises(Budget):
            events.get(1)  # the natural close was interrupted; the binding keeps its report
        self.assertFalse(events.ended or env.process.finished)
        self.assertEqual([events.get(1)[0] for _ in range(2)], ["eof", "exit"])
        self.assertTrue(env.process.finished and env.process.poll() == 0)
        class Refusing(Engine):
            def take(self, timeout, until_boundary): raise EngineFailure("No active native episode")
        events, _ = self.events(Refusing())
        self.assertEqual(events.get(1)[:3], ("bridge_error", 0, "EngineFailure: No active native episode"))

    def test_a_decode_refusal_ends_the_stream_without_an_end_of_stream(self):
        class Engine:
            def take(self, timeout, until_boundary):
                return delivery(messages=[{"type": "ready"}], decode_error="ValueError: Duplicate JSON field: a")
            def finish(self, natural, timeout):
                return report(exit_code=None, natural=False, host_stopped=True, messages=[{"type": "late"}])
        events, _ = self.events(Engine())
        kinds = []
        while True:
            try:
                kinds.append(events.get(1)[0])
            except queue.Empty:
                break
        self.assertEqual(kinds, ["message", "bridge_error"])


class EvidenceRuleTests(unittest.TestCase):
    def test_host_stop_and_natural_rules_refuse_every_unclean_variant(self):
        stop = {"returncode": None, "native_close": ev.close_record(report(exit_code=None, natural=False, host_stopped=True))}
        self.assertTrue(ev.truncation_closed(stop))
        natural = {"returncode": 0, "native_close": ev.close_record(report(exit_code=0, natural=True, host_stopped=False))}
        self.assertTrue(ev.natural_closed(natural))
        self.assertFalse(ev.truncation_closed(natural))
        self.assertFalse(ev.natural_closed(stop))
        for mutate in (lambda f: f.update(returncode=-15), lambda f: f["native_close"].update(exit_code=1),
                       lambda f: f["native_close"].update(close_clean=False), lambda f: f["native_close"].update(poisoned=True),
                       lambda f: f["native_close"].update(channel_failure="x"), lambda f: f["native_close"].update(host_stopped=False),
                       lambda f: f["native_close"]["opponent"].update(released=False),
                       lambda f: f["native_close"]["opponent"].update(module_reset=False),
                       lambda f: f["native_close"]["opponent"].update(created=False), lambda f: f.pop("native_close")):
            changed = deepcopy(stop); mutate(changed)
            with self.subTest(mutate=mutate):
                self.assertFalse(ev.truncation_closed(changed) if "native_close" in changed else False)
        with self.assertRaises(ValueError):
            ev.truncation_closed({"returncode": None})
        for key in ("module_reset", "created"):
            changed = deepcopy(natural); changed["native_close"]["opponent"][key] = False
            self.assertFalse(ev.natural_closed(changed))

    def test_every_bound_field_names_one_backend(self):
        close = ev.close_record(report(exit_code=None, natural=False, host_stopped=True))
        embedded = {"identity": {"backend": {"kind": ev.BACKEND}}, "launch": {"command": None, "backend": ev.BACKEND,
                    "environment": {"EXPLORER_RL_TRANSPORT": "in-process"}}}
        cold = {"identity": {}, "launch": {"command": ["vcmiclient", "--headless"], "environment": {}}}
        ev.require_consistent_backend(embedded, {"native_close": close})
        ev.require_consistent_backend(cold, {})
        stripped = deepcopy(embedded); del stripped["identity"]["backend"]
        ev.require_consistent_backend({"identity": {}, "launch": {"environment": {}}}, {})  # older synthetic records
        for manifest, final in ((stripped, {"native_close": close}), (stripped, {}), (cold, {"native_close": close}),
                                (embedded, {}), ({**cold, "identity": embedded["identity"]}, {"native_close": close})):
            with self.subTest(manifest=manifest, final=final), self.assertRaises(ValueError):
                ev.require_consistent_backend(manifest, final)

    def test_loaded_runtime_changes_are_detected_by_inventory_and_stat_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = SyntheticEmbeddedLauncher(Path(tmp))
            module = Path(tmp) / "embedded" / (ev.MODULE_NAME + ".so")
            module.parent.mkdir()
            module.write_bytes(b"module")
            engine = object.__new__(ee.EmbeddedEngine)
            engine.files = ee.runtime_files(launcher, module)
            self.assertIn(launcher.game_map.resolve(), engine.files)
            engine.fingerprint = ee.fingerprint(engine.files)
            engine.require_unchanged()
            library = launcher.binary.parent / ("AI/libNullkiller" + (".dylib" if sys.platform == "darwin" else ".so"))
            for change in (lambda: library.write_bytes(b"rebuilt"), lambda: module.unlink()):
                change()
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "restart Python"):
                    engine.require_unchanged()


class EmbeddedIdentityTests(unittest.TestCase):
    file, sha, check = diagnostic.IdentityTests.file, diagnostic.IdentityTests.sha, diagnostic.IdentityTests.check

    def setUp(self):
        diagnostic.IdentityTests.setUp(self)
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        self.file("embedded/" + ev.MODULE_NAME + suffix, "synthetic module " + ev.PROTOCOL)
        self.file("engine/NativeSession/EngineSession.cpp", "synthetic host")
        sources = set(self.manifest["identity"]["contextual_python_sha256"]) | ev.ADAPTER_SOURCES
        for name in ev.ADAPTER_SOURCES: self.file(name, "synthetic python " + name)
        self.manifest["identity"]["contextual_python_sha256"] = {name: self.sha(name) for name in sources}
        self.manifest["identity"]["backend"] = ev.backend_identity(self.binary.parent, self.root)

    def test_embedded_capture_binds_module_host_sources_and_adapter(self):
        self.assertTrue(self.check()["embedded_backend"]["embedded_module_unchanged"])
        for name in ("embedded/" + ev.MODULE_NAME + importlib.machinery.EXTENSION_SUFFIXES[0],
                     "engine/NativeSession/EngineSession.cpp", "experiments/headless/embedded_env.py"):
            path = self.root / name; old = path.read_bytes(); path.write_bytes(old + b" changed")
            with self.subTest(name=name), self.assertRaises(ValueError): self.check()
            with self.subTest(name=name, diagnostic=True), self.assertRaises(ValueError): self.check(True)
            path.write_bytes(old)
        self.file("engine/NativeSession/Extra.cpp", "new host source")
        with self.assertRaisesRegex(ValueError, "host sources"): self.check()

    def test_the_captured_module_file_is_verified_under_any_python_abi(self):
        other = "embedded/" + ev.MODULE_NAME + ".cpython-399-synthetic.so"
        self.file(other, "module built for another interpreter")
        self.manifest["identity"]["backend"].update(module=other, module_sha256=self.sha(other))
        self.assertTrue(self.check()["embedded_backend"]["embedded_module_unchanged"])
        for name in ("../escape.so", "bin/" + ev.MODULE_NAME + ".so"):
            self.manifest["identity"]["backend"]["module"] = name
            with self.subTest(name=name), self.assertRaises(ValueError): self.check()

    def test_cold_capture_cannot_carry_embedded_sources(self):
        del self.manifest["identity"]["backend"]
        with self.assertRaisesRegex(ValueError, "inventory"): self.check()


if __name__ == "__main__": unittest.main()
