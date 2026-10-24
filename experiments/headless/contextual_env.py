"""Opt-in real-engine contextual lifecycle, with immutable paired boundaries.

The legacy environment supplies socket ownership and its established timing
loop. This mixin replaces every semantic fold and scoring boundary. Complete
raw and live streams remain evidence; frozen cutoff histories exclude the
explicit bounded close/end-turn finalizer, which is never learner experience.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time
import weakref

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.learner import contextual as contract
import contextual_protocol
from contextual_protocol import ContextualProducer, NativeFailure, loads
from env import COLORS, Limits, Request, EpisodeEnd, _Side, READER_DRAIN_SECONDS, scoring, stop_process_group

# Record loaded adapter inputs immediately. Each reset refuses later source
# changes; the experiment runner must reload modules to use an edited version.
_IMPORTED_SOURCE_HASHES = {
    str(Path(__file__).resolve().relative_to(ROOT)): scoring.digest(Path(__file__)),
    str(Path(contextual_protocol.__file__).resolve().relative_to(ROOT)): contextual_protocol.IMPORTED_SOURCE_SHA256,
    str(Path(contract.__file__).resolve().relative_to(ROOT)): scoring.digest(Path(contract.__file__)),
}

CONTINUATION_LIMIT = 8  # per seat; close then end-turn normally needs at most two
# These are explicit engine failure channels in the owned headless patch. In
# particular, a late unsupported event can report only through the client hook
# after the policy worker has already observed terminal.
ENGINE_FAILURE_MARKERS = (
    "Experiment failure:", "Headless network thread failed:",
    "Headless connection to the server failed:", "Headless packet handling failed:",
    "Headless game start failed:", "Headless run received a silent game end",
)


_WIRE_CACHE = weakref.WeakKeyDictionary()


def canonical_records(history):
    """Cache only immutable serialized frames, never a mutable public object."""
    frames = history.frames
    cache = _WIRE_CACHE.setdefault(history, [])
    for frame in frames[len(cache):]:
        cache.append(json.dumps(contract.encode_frame(frame), sort_keys=True, separators=(",", ":")))
    return cache


def encoded(history, start=0):
    # A new JSON graph on every call preserves caller isolation; historical
    # dataclass trees are serialized once, rather than once per decision.
    return json.loads("[" + ",".join(canonical_records(history)[start:]) + "]")


def persist_history(path, history):
    with path.open("x") as stream:
        for record in canonical_records(history):
            stream.write(record + "\n")
    return scoring.digest(path)


class ContextualMixin:
    _decode_message = staticmethod(loads)

    def __init__(self, launcher, output, limits=Limits(), *, ledger=True, profile=False, agents=None,
                 snapshot_every=1, replay=True, interface=contract.ENVIRONMENT):
        self.context_profile = contract.profile_for(interface)
        if interface != self.context_profile.environment:
            raise ValueError("Unknown contextual interface")
        if not ledger:
            raise ValueError("Contextual validation requires complete raw bridge and action ledgers")
        super().__init__(launcher, output, limits, ledger=ledger, profile=profile, agents=agents,
                         snapshot_every=snapshot_every, replay=replay)
        self.interface = interface
        self.cutoff_reason = None
        self._context_identity = None
        # The public env module may be loaded under a probe-specific name. Read
        # the actual base class module's import hash, not a second env import.
        base = next(cls for cls in type(self).__mro__ if cls.__name__ == "HeadlessSelfPlayEnv")
        base_module = sys.modules[base.__module__]
        self._imported_sources = {**_IMPORTED_SOURCE_HASHES,
            "experiments/headless/env.py": base_module.IMPORTED_SOURCE_SHA256}

    def _assert_sources_unchanged(self):
        changed = [name for name, digest in self._imported_sources.items()
                   if scoring.digest(ROOT / name) != digest]
        if changed:
            raise ValueError("Loaded contextual source changed; restart Python before reset: " + ", ".join(changed))

    def _preflight_contextual(self):
        self.launcher.preflight()
        library = getattr(self.launcher, "ai_library", None)
        if library is None:
            raise ValueError("Contextual launch requires a native ExplorerRL library identity")
        contents = Path(library).read_bytes()
        required = tuple(value.encode() for value in (self.context_profile.environment,
                         self.context_profile.history_schema, self.context_profile.action_schema))
        if any(marker not in contents for marker in required):
            raise ValueError(f"Rebuild ExplorerRL with the contextual protocol-{self.context_profile.protocol} interface; no legacy fallback")

    def reset(self, episode, seed):
        if type(episode) is not int or episode < 1 or type(seed) is not int or not 1 <= seed <= 2147483647:
            raise ValueError("Use a positive episode and signed-32-bit engine seed")
        self.close()
        self._assert_sources_unchanged()
        self._preflight_contextual()
        self._identity_cache = self._context_identity = None
        self._identity()  # freeze source, recipe, map and binary identity BEFORE launch
        self.episode, self.seed = episode, seed
        self.episode_dir = self.output / f"episode-{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.events = queue.Queue()
        self.request, self.result, self.env_step = None, None, 0
        self.closing = self.settling = False
        self.stopping = threading.Event()
        self.reason = self.detail = self.cutoff_reason = None
        self.engine_wait = self.agent_wait = 0.0
        self.step_latencies = []
        self.costs = {"ingest": 0.0, "act": 0.0, "ledger": 0.0, "respond": 0.0}
        self.replay_tail_wait = 0.0
        self.last_engine_message = self.shutdown_deadline = None
        self.sides = {}
        environment = {"EXPLORER_RL_PLAYERS": "0,1", "EXPLORER_RL_EPISODE": str(episode),
                       "EXPLORER_RL_TIMEOUT": str(self.limits.bridge_timeout),
                       "EXPLORER_RL_PROFILE": "1" if self.profile else "0",
                       "EXPLORER_RL_INTERFACE": self.interface}
        # These optional files are raw evidence, explicitly unsupported by the
        # contextual verifier; they never inherit a legacy replay success label.
        if self.replay:
            environment["VCMI_EXPERIMENT_REPLAY"] = "1"
        for player, color in enumerate(COLORS):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            producer = ContextualProducer(str(episode), player, interface=self.interface)
            side = _Side(player, color, producer, producer.history, listener)
            folder = self.episode_dir / "agents" / color
            folder.mkdir(parents=True)
            side.ledger = (folder / "bridge.jsonl").open("x")
            environment[f"EXPLORER_RL_PORT_{player}"] = str(listener.getsockname()[1])
            self.sides[player] = side
        self.started = time.monotonic()
        self.created_at = datetime.now(timezone.utc).isoformat()
        (self.episode_dir / "logs").mkdir()
        self.log = (self.episode_dir / "logs" / "process.log").open("w")
        try:
            self.process, self.launch_info = self.launcher.launch(self.episode_dir, seed, environment, self.log)
        except BaseException:
            self.close()
            raise
        for side in self.sides.values():
            side.reader = threading.Thread(target=self._read, args=(side,), name=f"contextual-{side.color}", daemon=True)
            side.reader.start()
        threading.Thread(target=self._watch_process, args=(self.process, self.events), daemon=True).start()
        return self._advance()

    def history_v2(self, color):
        if self.context_profile != contract.WINDOWS_PROFILE:
            raise ValueError("history_v2 requires adventure-windows-v1")
        return self.history(color)

    def history_v3(self, color):
        if self.context_profile != contract.TOWN_PROFILE:
            raise ValueError("history_v3 requires adventure-town-v1")
        return self.history(color)

    def history(self, color):
        if color not in COLORS:
            raise ValueError("Unknown player color")
        return encoded(self.sides[COLORS.index(color)].validator.history)

    def history_v1(self, color):
        raise ValueError("Contextual history has no compatible v1 projection")

    def memory(self, color):
        raise ValueError("Contextual mode exposes history only; no engineered memory input")

    def _send_current(self, action, *, continuation=False):
        if self.request is None:
            raise ValueError("No current contextual request")
        side = self.sides[self.request.player]
        producer = side.validator
        response = producer.response(action)  # validation before send or mutation
        if continuation and producer.cutoff is None:
            raise ValueError("Finalizer actions require a previously frozen own cutoff")
        if continuation and producer.continuation_steps >= CONTINUATION_LIMIT:
            self._note_problem("cutoff_continuation_limit", "Could not obtain both real player boundaries")
            self.request = None
            return
        self.agent_wait += time.monotonic() - self.request_time
        started = time.perf_counter()
        try:
            side.connection.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
            delivered = True
        except OSError as error:
            side.write_error = f"{type(error).__name__}: {error}"
            side.responses_undelivered += 1
            delivered = False
        self.costs["respond"] += time.perf_counter() - started
        self._write_ledger(side, "policy", {**response, "delivered": delivered,
                                            "outside_learner_prefix": continuation})
        if delivered:
            producer.submitted(action)
            side.actions_sent += 1
        else:
            # Even an undelivered action cannot turn an impending real terminal
            # into an invented consumed sample. Actual streams/exit still decide.
            self._note_problem("action_delivery_failed", f"{side.color}: {side.write_error}")
        self.env_step += 1
        self.request = None

    def step(self, action):
        if self.request is None:
            raise ValueError("No current contextual request")
        side = self.sides[self.request.player]
        side.validator.response(action)  # invalid caller input cannot mutate request/history
        held = side.validator.live.decision
        exited = self._settle()
        if self.reason is not None or exited is not None:
            self.request = None
            return self._finish(exited)
        if any(other.validator.terminal for other in self.sides.values()):
            self.request = None
            self.closing = True
            return self._advance()
        if side.validator.live.decision is not held:
            # An intervening visible event withdrew the offered decision. Return
            # the real replacement (or await it), never apply the old choice to it.
            self.request = None
            if side.validator.live.decision is not None:
                return self._issue(side.player)
            return self._advance()
        self._send_current(action)
        started = time.monotonic()
        result = self._advance()
        self.step_latencies.append(time.monotonic() - started)
        return result

    def _capture_cutoff(self, side):
        producer = side.validator
        if producer.cutoff is not None:
            return
        producer.freeze_cutoff(self.cutoff_reason)
        side.history = producer.history
        side.cutoff = contract.encode_frame(producer.cutoff)
        side.cutoff_seq = producer.cutoff.seq
        side.cutoff_reason = side.limit = self.cutoff_reason
        side.cutoff_actions = side.actions_sent

    def truncate(self, reason="collection_window"):
        if self.request is None or reason not in scoring.HEADLESS_TRUNCATIONS:
            raise ValueError("Truncate only at a pending own request with a declared horizon")
        request = self.request
        exited = self._settle()
        if self.reason is not None or exited is not None:
            self.request = None
            return self._finish(exited)
        if any(side.validator.terminal for side in self.sides.values()):
            self.request = None
            self.closing = True
            return self._advance()
        side = self.sides[request.player]
        if side.validator.live.decision is None or side.validator.live.action is not None:
            self._note_problem("cutoff_boundary_unavailable", "Pending decision was superseded before cutoff")
            self.request = None
            return self._finish(None)
        self.cutoff_reason = reason
        self._capture_cutoff(side)
        self.request = self._issue(side.player, continuation=True)
        return self._continue_cutoff()

    def _continue_cutoff(self):
        while True:
            if all(side.validator.cutoff is not None for side in self.sides.values()):
                self.request = None
                self._note_problem(self.cutoff_reason, "Both real own-decision cutoffs captured")
                return self._finish(None)
            if self.request is not None:
                side = self.sides[self.request.player]
                if side.validator.cutoff is None:
                    self._capture_cutoff(side)
                    continue
                exited = self._settle()
                if self.reason is not None or exited is not None:
                    self.request = None
                    return self._finish(exited)
                if any(other.validator.terminal for other in self.sides.values()):
                    self.request = None
                    self.closing = True
                    return super()._advance()
                if side.validator.live.decision is None:
                    self.request = None
                else:
                    context = side.validator.live.context
                    kind = "end_turn" if context.kind == "map" else "back" if context.panel in ("hall", "building", "recruitment") else "close"
                    action = contract.Action(kind, subject=context.subject if kind == "back" else None,
                                             schema=self.context_profile.action_schema)
                    self._send_current(action, continuation=True)
            state = super()._advance()
            if state.done:
                return state
            # Every request during this bounded finalizer remains internal.

    def _advance(self):
        state = super()._advance()
        if not state.done and self.cutoff_reason is not None:
            return self._continue_cutoff()
        return state

    def _ingest_checked(self, side, message, received, draining):
        try:
            frames = side.validator.ingest(message)
        except NativeFailure as error:
            side.error = str(error)
            reason = "unsupported_interface" if "unsupported" in side.error.lower() else "engine_error"
            self._note_problem(reason, f"{side.color}: {error}")
            return None
        if message["type"] == "ready":
            side.ready_at = received - self.started
        for frame in frames:
            if frame.kind == "acknowledged":
                if frame.accepted:
                    side.accepted += 1
                else:
                    side.rejected += 1
            elif frame.kind == "terminal":
                side.terminal_at = received - self.started
        if draining or not frames or frames[-1].kind != "decision":
            return None
        producer = side.validator
        decision = producer.live.decision
        if self.cutoff_reason is None and (decision.day > self.limits.max_days or side.actions_sent >= self.limits.max_actions):
            self.cutoff_reason = "game_day_limit" if decision.day > self.limits.max_days else "agent_action_limit"
        if self.cutoff_reason is not None:
            self._capture_cutoff(side)
        if self.request is not None:
            raise ValueError("Concurrent outstanding contextual requests")
        return self._issue(side.player, continuation=self.cutoff_reason is not None)

    def _issue(self, player, continuation=False):
        side = self.sides[player]
        history = side.validator.live if continuation else side.validator.history
        if history.decision is None or history.frames[-1] != history.decision:
            raise ValueError("Request must end at the exact current player decision")
        frames = encoded(history, side.unread_frames_from)
        # Internal continuation may have already delivered its boundary decision.
        if not frames and continuation:
            frames = [contract.encode_frame(history.decision)]
        if not frames:
            raise ValueError("Empty contextual request history")
        if not continuation:
            side.unread_frames_from = len(history.frames)
        self.request = Request(side.color, player, "contextual", frames[-1], frames, frames,
                               past_horizon=continuation)
        self.request_time = time.monotonic()
        return self.request

    def _note_problem(self, reason, detail):
        # A later technical failure always invalidates provisional cutoffs,
        # including failure discovered while draining a stopped process.
        if self.reason is None or self.reason in scoring.HEADLESS_TRUNCATIONS and reason not in scoring.HEADLESS_TRUNCATIONS:
            self.reason, self.detail = reason, detail

    def _settle(self):
        """Drain all queued events, including errors queued after an exit event."""
        exited = None
        while True:
            try:
                kind, player, payload, received = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "exit":
                exited = payload
            elif kind == "message":
                self._ingest(self.sides[player], payload, received, draining=True)
            elif kind == "bridge_error":
                self._note_problem("protocol_error", f"{COLORS[player]}: {payload}")
            elif kind == "eof":
                self.sides[player].eof = True
        polled = self.process.poll()
        if exited is None and polled is None and self.reason not in scoring.HEADLESS_TRUNCATIONS:
            for side in self.sides.values():
                if side.eof and not side.validator.terminal:
                    self._note_problem("bridge_disconnected", f"{side.color} closed before terminal")
        return exited if exited is not None else polled

    def _identity(self):
        if self._context_identity is None:
            identity = super()._identity()
            recipe = identity["engine_recipe"]
            bridge = {name: digest for name, digest in recipe.items() if name.startswith("engine/ExplorerRL/")}
            self._context_identity = {**identity,
                "engine_revision": (identity.get("engine") or {}).get("commit", "unavailable"),
                "bridge_source_sha256": hashlib.sha256(json.dumps(bridge, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "map_sha256": identity.get("map", {}).get("sha256"),
                "contextual_python_sha256": dict(self._imported_sources),
                "captured_before_launch": True}
        return self._context_identity

    def _finish(self, exited):
        if self.result is not None:
            return self.result
        # A pre-existing process exit or queued failure wins over a requested
        # cutoff. Do not manufacture success by killing an already failed game.
        pending_exit = self._settle()
        exited = exited if exited is not None else pending_exit
        declared_stop = self.reason in scoring.HEADLESS_TRUNCATIONS and exited is None
        if exited is not None and self.reason in scoring.HEADLESS_TRUNCATIONS:
            self.reason = self.detail = None
        if exited is None and self.reason is not None:
            stop_process_group(self.process)
        returncode = self.process.wait()
        # Never silence reader errors before both real socket streams finish.
        # On an intentional stop EOF is expected; invalid/truncated JSON is not.
        connected = [side for side in self.sides.values() if side.connection is not None and side.reader is not None]
        deadline = time.monotonic() + READER_DRAIN_SECONDS
        for side in connected:
            side.reader.join(timeout=max(0, deadline - time.monotonic()))
        unfinished = [side.color for side in connected if side.reader.is_alive()]
        self._settle()
        if unfinished:
            self._note_problem("protocol_error", "Unfinished contextual reader: " + ", ".join(unfinished))
        self.stopping.set()
        for side in self.sides.values():
            if side.reader is not None and side.connection is None:
                side.reader.join(timeout=1)
        self._settle()
        self.request = None
        for side in self.sides.values():
            if side.ledger is not None:
                side.ledger.close()
        self.log.close()
        log = (self.episode_dir / "logs/process.log").read_text(errors="replace")
        failure_marker = next((marker for marker in ENGINE_FAILURE_MARKERS if marker in log), None)
        if failure_marker:
            self._note_problem("engine_error", f"Authoritative engine failure log: {failure_marker}")
        executions = {side.color: {"ready": side.validator.ready, "actions_sent": side.actions_sent,
                                   "error": side.error, "terminal": {"outcome": side.validator.terminal}
                                   if side.validator.terminal else None} for side in self.sides.values()}
        status, outcomes, termination, rewards = scoring.classify_headless_self_play(returncode, self.reason, log, executions)
        if status == "failed" and self.reason == "unsupported_interface":
            # Legacy scoring groups socket errors as policy_error. Preserve the
            # explicit contextual unsupported boundary for diagnosis/cohort counts.
            termination = self.reason
        histories = tuple(self.sides[player].validator.history for player in (0, 1))
        complete = tuple(self.sides[player].eof and self.sides[player].validator.ready for player in (0, 1))
        evidence, evidence_error = self._engine_evidence(log)
        # A terminal after continuation is preserved as evidence, but cannot
        # retroactively score the earlier frozen collection prefix.
        natural_after_cutoff = outcomes if status == "completed" and self.cutoff_reason else None
        if status == "completed" and self.cutoff_reason:
            status, termination = "failed", "natural_terminal_before_paired_cutoff"
        if status in ("completed", "truncated") and evidence_error:
            status, termination = "failed", "engine_evidence_invalid"
        if status in ("completed", "truncated") and not all(complete):
            status, termination = "failed", "incomplete_contextual_streams"
        shutdown_verified = declared_stop and returncode in (-signal.SIGTERM, -signal.SIGKILL)
        if status == "truncated" and not shutdown_verified:
            # An engine failure can win between the last poll and our signal.
            # Its positive/other-signal exit is never attributed to our cutoff.
            status, termination = "failed", "unverified_cutoff_shutdown"
        if status == "truncated" and any(h.boundary != "cutoff_pending" for h in histories):
            status, termination = "failed", "unverified_paired_cutoff"
        identity = self._identity()
        if status in ("completed", "truncated") and (identity["engine_revision"] == "unavailable" or not identity.get("map_sha256")):
            status, termination = "failed", "missing_source_identity"
        evidence_path = self.episode_dir / "contextual-finalization.json"
        evidence_document = {
            "schema": "heroes-contextual-finalization-evidence-v1", "episode": str(self.episode),
            "returncode": returncode, "declared_stop": declared_stop, "shutdown_verified": shutdown_verified,
            "expected_stop_signals": [signal.SIGTERM, signal.SIGKILL], "reason": self.reason,
            "detail": self.detail, "streams_complete": list(complete), "engine_evidence": evidence,
            "engine_evidence_error": evidence_error, "process_log_sha256": scoring.digest(self.episode_dir / "logs/process.log"),
            "socket_outcomes": [self.sides[p].validator.terminal for p in (0, 1)],
            "engine_verified_outcomes": outcomes if status == "completed" else None,
            "natural_terminal_after_cutoff": natural_after_cutoff,
            "continuation": {"limit_per_player": CONTINUATION_LIMIT, "actions": [self.sides[p].validator.continuation_steps for p in (0, 1)],
                             "allowed_actions": (["back", "close", "end_turn"] if self.context_profile == contract.TOWN_PROFILE
                                                 else ["close", "end_turn"]), "learner_excluded": True},
            "raw_bridge_sha256": {side.color: scoring.digest(self.episode_dir / "agents" / side.color / "bridge.jsonl")
                                  for side in self.sides.values()}, "identity": identity,
        }
        scoring.write_json(evidence_path, evidence_document)
        evidence_ref = evidence_path.name + "#sha256=" + scoring.digest(evidence_path)
        try:
            if status == "completed":
                contract.finalize_natural(histories, engine_outcomes=tuple(outcomes[c] for c in COLORS),
                    socket_outcomes=tuple(self.sides[p].validator.terminal for p in (0, 1)),
                    process_exit_code=returncode, evidence_ref=evidence_ref, histories_complete=complete)
            elif status == "truncated":
                contract.finalize_cutoff(histories, evidence_ref=evidence_ref, histories_complete=complete,
                    failure_free=True, shutdown_verified=shutdown_verified, continuation_limit=CONTINUATION_LIMIT,
                    continuation_steps=tuple(self.sides[p].validator.continuation_steps for p in (0, 1)))
        except (ValueError, TypeError) as error:
            status, termination = "failed", "contextual_finalization_invalid"
            self.detail = str(error)
        if status == "failed":
            for side in self.sides.values():
                side.validator.fail(termination)
        if status != "completed":
            outcomes, rewards = {color: None for color in COLORS}, {color: None for color in COLORS}
        declarations = {}
        for side in self.sides.values():
            producer = side.validator
            folder = self.episode_dir / "agents" / side.color
            path = folder / self.context_profile.history_filename
            digest = persist_history(path, producer.history)
            prefix = producer.history.frames[:-1] if producer.history.frames[-1].kind in ("finalized", "cutoff_finalized") else producer.history.frames
            digestor = hashlib.sha256()
            for frame in prefix:
                digestor.update(json.dumps(contract.encode_frame(frame), sort_keys=True, separators=(",", ":")).encode() + b"\n")
            live = folder / ("live-" + self.context_profile.history_filename)
            live_digest = persist_history(live, producer.live)
            declarations[side.color] = {**executions[side.color], "actions_sent": side.actions_sent,
                "actions_accepted": side.accepted, "actions_rejected": side.rejected,
                "responses_undelivered": side.responses_undelivered,
                self.context_profile.history_key: {"path": str(path.relative_to(self.episode_dir)), "sha256": digest,
                               "canonical_sha256": producer.history.digest(), "prefix_sha256": digestor.hexdigest(),
                               "boundary": producer.history.boundary, "complete": complete[side.player]},
                "live_evidence": {"path": str(live.relative_to(self.episode_dir)), "sha256": live_digest,
                                  "continuation_steps": producer.continuation_steps, "learner_input": False}}
        manifest = {"schema_version": 1, "kind": "contextual-headless-self-play", "episode": str(self.episode),
                    "environment": self.interface, "history_schema": self.context_profile.history_schema,
                    "action_schema": self.context_profile.action_schema,
                    "seed": self.seed, "created_at": self.created_at, "status": status, "termination_reason": termination,
                    "failure_detail": self.detail, "returncode": returncode, "outcomes_by_player": outcomes,
                    "rewards_by_player": rewards, "evidence_ref": evidence_ref, "players": declarations,
                    "identity": identity, "launch": self.launch_info, "engine_evidence": evidence,
                    "env_steps": self.env_step, "wall_seconds": time.monotonic() - self.started,
                    "replay": {"requested": self.replay, "status": "unsupported", "verified": False,
                               "reason": "Contextual fields and action/native-command associations are not replay verified"},
                    "deterministic_replay_verified": False, "training_performed": False,
                    "natural_terminal_after_horizon": natural_after_cutoff}
        scoring.write_json(self.episode_dir / "manifest.json", manifest)
        final = {side.color: encoded(side.validator.history, min(side.unread_frames_from, len(side.validator.history.frames) - 1))
                 for side in self.sides.values()}
        bootstrap = {side.color: side.cutoff if status == "truncated" else None for side in self.sides.values()}
        self.result = EpisodeEnd(status, termination, outcomes, rewards, final, final, bootstrap, manifest)
        self.process = None
        return self.result
