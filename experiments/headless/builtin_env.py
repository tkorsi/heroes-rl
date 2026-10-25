"""One v7 controlled player against native Nullkiller Easy, for evaluation only.

Reuse the strict player-local transport. Boundary proof and process scoring are
separate from the immutable two-policy contract; no opponent history is made.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import signal
import socket
import threading
import time

from experiments.headless.env import HeadlessSelfPlayEnv, Limits, EpisodeEnd, _Side, READER_DRAIN_SECONDS, scoring, stop_process_group
from experiments.headless.contextual_env import ContextualMixin, ContextualProducer, encoded, persist_history, ENGINE_FAILURE_MARKERS
from experiments.headless import builtin_evidence as evidence
from experiments.headless import contextual_evidence as native
from experiments.learner import contextual as c

COLORS = evidence.COLORS
IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class BuiltinHeadlessEnv(ContextualMixin, HeadlessSelfPlayEnv):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, launcher, output, *, learner_player, limits=Limits(), agents=None,
                 ledger=True, replay=True, interface=c.DIALOGS_ENVIRONMENT):
        evidence.controllers(learner_player)
        if interface != c.DIALOGS_ENVIRONMENT:
            raise ValueError("Built-in evaluation requires unchanged v7 player semantics")
        self.learner_player = learner_player
        super().__init__(launcher, output, limits, agents=agents, ledger=ledger, replay=replay, interface=interface)
        self._imported_sources.update({"experiments/headless/builtin_env.py": IMPORTED_SOURCE_SHA256,
            "experiments/headless/builtin_evidence.py": evidence.IMPORTED_SOURCE_SHA256})

    def _preflight_contextual(self):
        super()._preflight_contextual()
        directory = Path(self.launcher.binary).parent
        runtime = native.runtime_identity(directory)
        rules = next(directory / name for name in ("libvcmi.dylib", "libvcmi.so") if name in runtime)
        if b"EXPLORER_RL_BUILTIN_OPPONENT" not in rules.read_bytes():
            raise ValueError("Rebuild native rules library with explicit built-in opponent evaluation support")

    def reset(self, episode, seed):
        if type(episode) is not int or episode < 1 or type(seed) is not int or not 1 <= seed <= 2147483647:
            raise ValueError("Use a positive episode and signed-32-bit engine seed")
        self.close()
        self._assert_sources_unchanged()
        self._preflight_contextual()
        self._identity_cache = self._context_identity = None
        identity = self._identity()
        library = next((sha for name, sha in identity["native_runtime_sha256"].items()
                        if name in ("AI/libNullkiller.dylib", "AI/libNullkiller.so")), None)
        self.opponent_identity = evidence.opponent_identity(self.learner_player, library)
        self.episode, self.seed = episode, seed
        self.episode_dir = self.output / f"episode-{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.events = queue.Queue()
        self.request, self.result, self.env_step = None, None, 0
        self.closing = self.settling = False
        self.stopping = threading.Event()
        self.reason = self.detail = self.cutoff_reason = self.dialog_cutoff_drain = None
        self.engine_wait = self.agent_wait = 0.0
        self.step_latencies = []
        self.costs = {"ingest": 0.0, "act": 0.0, "ledger": 0.0, "respond": 0.0}
        self.replay_tail_wait = 0.0
        self.last_engine_message = self.shutdown_deadline = None
        player, color = self.learner_player, COLORS[self.learner_player]
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        producer = ContextualProducer(str(episode), player, interface=self.interface)
        side = _Side(player, color, producer, producer.history, listener)
        self.sides = {player: side}
        folder = self.episode_dir / "agents" / color
        folder.mkdir(parents=True)
        side.ledger = (folder / "bridge.jsonl").open("x")
        environment = {"EXPLORER_RL_PLAYERS": str(player), "EXPLORER_RL_BUILTIN_OPPONENT": "Nullkiller",
            "EXPLORER_RL_EPISODE": str(episode), "EXPLORER_RL_TIMEOUT": str(self.limits.bridge_timeout),
            "EXPLORER_RL_PROFILE": "0", "EXPLORER_RL_INTERFACE": self.interface,
            f"EXPLORER_RL_PORT_{player}": str(listener.getsockname()[1]),
            "VCMI_EXPERIMENT_REPLAY": "1", "VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE": "1"}
        self.started = time.monotonic()
        self.created_at = datetime.now(timezone.utc).isoformat()
        (self.episode_dir / "logs").mkdir()
        self.log = (self.episode_dir / "logs/process.log").open("w")
        try:
            self.process, self.launch_info = self.launcher.launch(self.episode_dir, seed, environment, self.log)
        except BaseException:
            self.close()
            raise
        side.reader = threading.Thread(target=self._read, args=(side,), name=f"builtin-learner-{color}", daemon=True)
        side.reader.start()
        threading.Thread(target=self._watch_process, args=(self.process, self.events), daemon=True).start()
        return self._advance()

    def history(self, color):
        if color != COLORS[self.learner_player]:
            raise ValueError("Nullkiller has no external-agent history")
        return encoded(self.sides[self.learner_player].validator.history)

    def defer_dialog_cutoff(self, reason):
        if (self.request is None or self.request.player != self.learner_player or self.request.past_horizon
                or reason not in scoring.HEADLESS_TRUNCATIONS or self.cutoff_reason is not None
                or self.sides[self.learner_player].validator.live.context.kind != "dialog"):
            raise ValueError("Defer only the current learner's unfrozen strategic dialog")
        if self.dialog_cutoff_drain is None:
            side = self.sides[self.learner_player]
            self.dialog_cutoff_drain = {"reason": reason, "trigger": {"player": self.learner_player,
                "decision_seq": side.validator.live.decision.seq, "actions_sent": side.actions_sent},
                "limit": 8, "choices": [], "learner_included": True}

    def _send_current(self, action, *, continuation=False):
        if continuation:
            raise ValueError("Single-player cutoff never sends technical or strategic continuation")
        return super()._send_current(action, continuation=False)

    def _continue_cutoff(self):
        side = self.sides[self.learner_player]
        if side.validator.cutoff is None or side.validator.history.context.kind == "dialog":
            raise ValueError("Single-player cutoff requires its actual safe held decision")
        self.request = None
        self._note_problem(self.cutoff_reason, "The controlled player's real decision cutoff was captured")
        return self._finish(None)

    def _engine_evidence(self, log_text):
        try:
            metadata = native.strict_json((self.episode_dir / "engine/experiment-metadata.json").read_bytes())
            if metadata.get("controllers") != evidence.controllers(self.learner_player):
                raise ValueError("Effective mixed controllers differ from requested routing")
            validated = scoring.validate_experiment_metadata(metadata, self.learner_player, log_text,
                                                              agent_player=self.learner_player)
            return {"metadata": validated, "seed": scoring.engine_seed_evidence(log_text, self.seed)}, None
        except (OSError, ValueError) as error:
            return None, f"{type(error).__name__}: {error}"

    def _finish(self, exited):
        if self.result is not None:
            return self.result
        pending = self._settle()
        exited = exited if exited is not None else pending
        declared_stop = self.reason in scoring.HEADLESS_TRUNCATIONS and exited is None
        if exited is not None and self.reason in scoring.HEADLESS_TRUNCATIONS:
            self.reason = self.detail = None
        if exited is None and self.reason is not None:
            stop_process_group(self.process)
        returncode = self.process.wait()
        side = self.sides[self.learner_player]
        if side.connection is not None and side.reader is not None:
            side.reader.join(timeout=READER_DRAIN_SECONDS)
        self._settle()
        if side.reader is not None and side.connection is not None and side.reader.is_alive():
            self._note_problem("protocol_error", "Controlled bridge reader did not drain")
        self.stopping.set()
        if side.reader is not None and side.connection is None:
            side.reader.join(timeout=1)
        self._settle()
        self.request = None
        if side.ledger is not None:
            side.ledger.close()
        self.log.close()
        log = (self.episode_dir / "logs/process.log").read_text(errors="replace")
        marker = next((m for m in ENGINE_FAILURE_MARKERS if m in log), None)
        if marker:
            self._note_problem("engine_error", "Authoritative engine failure: " + marker)
        if side.error or "ExplorerRL: " in log or "ExplorerRL terminal delivery failed" in log:
            self._note_problem("protocol_error", side.error or "Native policy failure")
        status = "truncated" if self.reason in scoring.HEADLESS_TRUNCATIONS else "failed"
        termination = self.reason or "process_exit_without_verified_terminal"
        history = side.validator.history
        complete = side.eof and side.validator.ready
        engine, engine_error = self._engine_evidence(log)
        barrier, barrier_error, outcomes = None, None, dict.fromkeys(COLORS)
        if self.reason is None and returncode == 0 and side.validator.terminal is not None:
            try:
                raw = native.strict_json((self.episode_dir / native.FILES[0]).read_bytes())
                barrier = evidence.terminal_barrier(self.episode_dir, episode=str(self.episode), interface=self.interface,
                    outcomes=raw.get("outcomes"), learner_player=self.learner_player)
                native.terminal_capture_close(self.episode_dir, barrier)
                outcomes = dict(zip(COLORS, barrier["server"]["outcomes"]))
                if outcomes[side.color] != side.validator.terminal:
                    raise ValueError("Own socket and native terminal disagree")
                if self.cutoff_reason is not None:
                    raise ValueError("A terminal after frozen cutoff cannot score its earlier prefix")
                status, termination = "completed", "natural_terminal"
            except (OSError, ValueError, TypeError, KeyError) as error:
                barrier_error = f"{type(error).__name__}: {error}"
                status, termination, self.detail = "failed", "terminal_barrier_invalid", barrier_error
        shutdown_verified = declared_stop and returncode in (-signal.SIGTERM, -signal.SIGKILL)
        if status == "truncated" and (not shutdown_verified or history.boundary != "cutoff_pending"):
            status, termination = "failed", "unverified_single_player_cutoff"
        if status in ("completed", "truncated") and (engine_error or not complete):
            status, termination = "failed", "engine_evidence_invalid" if engine_error else "incomplete_contextual_stream"
        identity = self._identity()
        if status in ("completed", "truncated") and (identity["engine_revision"] == "unavailable" or not identity.get("map_sha256")):
            status, termination = "failed", "missing_source_identity"
        try:
            capture = evidence.capture_files(self.episode_dir)
        except (OSError, ValueError) as error:
            capture = {"schema": evidence.REPLAY_SCHEMA, "recorded": False, "files": {}, "error": str(error)}
        if status in ("completed", "truncated") and not capture["recorded"]:
            status, termination = "failed", "contextual_capture_incomplete"
        if status == "failed":
            side.validator.fail(termination)
        if status != "completed":
            outcomes = dict.fromkeys(COLORS)
        rewards = {color: 1 if o == "win" else -1 if o == "loss" else None for color, o in outcomes.items()}
        folder = self.episode_dir / "agents" / side.color
        path, live = folder / "history-v7.jsonl", folder / "live-history-v7.jsonl"
        sha, live_sha = persist_history(path, history), persist_history(live, side.validator.live)
        boundary = {"completed": "natural", "truncated": "cutoff", "failed": "failure"}[status]
        declaration = {"ready": side.validator.ready, "actions_sent": side.actions_sent, "error": side.error,
            "terminal": {"outcome": side.validator.terminal} if side.validator.terminal else None,
            "actions_accepted": side.accepted, "actions_rejected": side.rejected,
            "responses_undelivered": side.responses_undelivered,
            "history_v7": {"path": str(path.relative_to(self.episode_dir)), "sha256": sha,
                "canonical_sha256": sha, "prefix_sha256": sha, "boundary": boundary, "complete": complete},
            "live_evidence": {"path": str(live.relative_to(self.episode_dir)), "sha256": live_sha,
                              "continuation_steps": 0, "learner_input": False}}
        final = {"schema": evidence.FINAL_SCHEMA, "episode": str(self.episode), "learner_player": self.learner_player,
            "controllers": evidence.controllers(self.learner_player), "returncode": returncode,
            "declared_stop": declared_stop, "shutdown_verified": shutdown_verified, "reason": self.reason,
            "detail": self.detail, "stream_complete": complete, "engine_evidence": engine,
            "engine_evidence_error": engine_error, "process_log_sha256": scoring.digest(self.episode_dir / "logs/process.log"),
            "terminal_barrier": barrier, "terminal_barrier_error": barrier_error, "native_capture": capture,
            "socket_outcome": side.validator.terminal, "engine_verified_outcomes": outcomes if status == "completed" else None,
            "raw_bridge_sha256": {side.color: scoring.digest(folder / "bridge.jsonl")}, "identity": identity,
            "dialog_cutoff_drain": self.dialog_cutoff_drain, "history_sha256": sha}
        proof_path = self.episode_dir / "builtin-finalization.json"
        scoring.write_json(proof_path, final)
        manifest = {"schema_version": 1, "kind": evidence.MANIFEST_KIND, "episode": str(self.episode),
            "environment": self.interface, "history_schema": c.DIALOGS_SCHEMA, "action_schema": c.DIALOGS_ACTION_SCHEMA,
            "learner_player": self.learner_player, "controllers": evidence.controllers(self.learner_player),
            "opponent_identity": self.opponent_identity, "seed": self.seed, "created_at": self.created_at,
            "status": status, "termination_reason": termination, "failure_detail": self.detail,
            "returncode": returncode, "outcomes_by_player": outcomes, "rewards_by_player": rewards,
            "evidence_ref": proof_path.name + "#sha256=" + scoring.digest(proof_path), "players": {side.color: declaration},
            "identity": identity, "launch": self.launch_info, "engine_evidence": engine, "env_steps": self.env_step,
            "wall_seconds": time.monotonic() - self.started, "replay": capture, "dialog_cutoff_drain": self.dialog_cutoff_drain,
            "deterministic_replay_verified": False, "training_performed": False, "training_samples": 0}
        evidence.validate_manifest_history(history, manifest)
        scoring.write_json(self.episode_dir / "manifest.json", manifest)
        frames = {side.color: encoded(history, min(side.unread_frames_from, len(history.frames) - 1))}
        self.result = EpisodeEnd(status, termination, outcomes, rewards, frames, frames,
            {side.color: side.cutoff if status == "truncated" else None}, manifest)
        self.process = None
        return self.result
