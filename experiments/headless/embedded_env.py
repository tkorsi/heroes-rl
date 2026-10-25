"""Opt-in in-process backend for the one-seat built-in evaluation environment.

The real VCMI client and server, ExplorerRL for the learner, Nullkiller Easy
for the other seat and BattleAI for every automatic battle run inside this
Python process through the heroes_embedded_engine extension. No game process,
socket, renderer or disk polling is on the decision path: the learner's own
messages arrive as native values identical to what the socket backend decodes,
and responses return as typed values. Game rules, routing, battles, evidence
files, the v7 history, finalization and independent replay are unchanged.

One engine per Python process, loaded once and reset for each game. It owns the
process working directory (the VCMI build output, as for the executable client)
and environment variables. A failed engine is poisoned and needs a new process;
run each embedded worker in its own process. The socket backend
(BuiltinHeadlessEnv with VcmiLauncher) remains the reference and fallback.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import time

from experiments.headless.builtin_env import (BuiltinHeadlessEnv, COLORS, ContextualProducer, ENGINE_FAILURE_MARKERS,
    EpisodeEnd, Limits, _Side, c, encoded, evidence, native, persist_history, scoring)
from experiments.headless.env import ROOT, VcmiLauncher, _git_state
from experiments.headless import embedded_evidence as embedded

IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BACKEND = embedded.BACKEND
# Waits return to Python at least this often, so signal handlers (budget
# alarms, Ctrl-C) run while the engine plays a long opponent turn.
WAIT_SLICE_SECONDS = 0.5
# The cold launcher removes these from the inherited environment of its child.
STRIPPED_PREFIXES = ("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_")


def profile_layout(profile):
    """The executable launcher's per-platform private profile layout."""
    if sys.platform == "darwin":
        return profile, profile / "config", {"VCMI_USER_DATA_DIR": str(profile)}
    return (profile / "data" / "vcmi", profile / "config" / "vcmi",
            {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
             "XDG_CACHE_HOME": str(profile / "cache")})


def runtime_files(launcher, module):
    """Every file whose loaded copy this process runs or records as identity."""
    directory = launcher.binary.parent.resolve()
    paths = {directory / "vcmireplay", launcher.binary, launcher.ai_library, launcher.game_map, Path(module),
             *directory.glob("*.dylib"), *directory.glob("*.so*"), *directory.glob("AI/*.dylib"), *directory.glob("AI/*.so*")}
    return sorted(Path(path).resolve() for path in paths)


def fingerprint(paths):
    """Inventory and stat identity: a rebuild, copy or in-place write changes it."""
    result = {}
    for path in paths:
        try:
            status = os.stat(path)
        except OSError:
            result[str(path)] = None
            continue
        result[str(path)] = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
    return result


def write_profile(profile, data, game_map, settings):
    data_root, config_root, environment = profile_layout(profile)
    for folder in (data_root / "Maps", config_root):
        folder.mkdir(parents=True, exist_ok=True)
    (data_root / "Data").symlink_to(data / "Data", target_is_directory=True)
    local_map = data_root / "Maps" / ("headless" + game_map.suffix.lower())
    shutil.copyfile(game_map, local_map)
    scoring.write_json(config_root / "settings.json", settings)
    return local_map, environment


class EmbeddedLauncher(VcmiLauncher):
    """The prepared runtime of VcmiLauncher, hosted in-process; never spawns."""

    def __init__(self, binary, data, game_map, log_level="info"):
        super().__init__(Path(binary), Path(data), Path(game_map), log_level)

    def preflight(self):
        super().preflight()
        content = Path(embedded.module_path(self.binary.parent)).read_bytes()
        if embedded.PROTOCOL.encode() not in content:
            raise ValueError("Rebuild the embedded engine module for " + embedded.PROTOCOL)

    def identity(self):
        return {**super().identity(), "backend": embedded.backend_identity(self.binary.parent)}

    def launch(self, *args, **kwargs):
        raise ValueError("The embedded backend never launches a game process")

    def episode_profile(self, directory, seed, environment, game_map=None):
        """Episode evidence of the map and settings this game used, at the cold paths."""
        settings = self.settings(seed)
        write_profile(directory / "profile", self.data, game_map or self.game_map, settings)
        return {"command": None, "backend": BACKEND, "environment": environment, "requested_settings": settings}


class EmbeddedEngine:
    """The process's one loaded engine. Reset per game; never reloaded."""

    _owner = None
    _lock = threading.Lock()

    @classmethod
    def acquire(cls, launcher, workspace):
        with cls._lock:
            if cls._owner is None:
                cls._owner = cls(launcher, workspace)
            cls._owner.require_compatible(launcher)
            return cls._owner

    def __init__(self, launcher, workspace):
        workspace = Path(workspace).expanduser()
        if not workspace.is_absolute() or workspace.exists():
            raise ValueError("The embedded engine needs a new absolute workspace")
        self.binary_directory = launcher.binary.parent
        self.data, self.game_map = launcher.data, launcher.game_map
        self.module_path = embedded.module_path(self.binary_directory)
        # The process keeps running the copies it loads now: libvcmi with the
        # module, ExplorerRL/Nullkiller/BattleAI at the first game (never closed),
        # the worker map copy. Identity is hashed once, from these files, and every
        # later reset and finish requires them unchanged on disk.
        self.launcher = launcher
        self.files = runtime_files(launcher, self.module_path)
        self.fingerprint = fingerprint(self.files)
        launcher.preflight()
        self.runtime_sha256 = native.runtime_identity(self.binary_directory)
        self.loaded_identity = {**launcher.identity(), "engine": _git_state(ROOT / "vcmi"), "project": _git_state(ROOT),
                                "engine_recipe": scoring.engine_recipe_digests(ROOT)}
        self.map_sha256 = self.loaded_identity["map"]["sha256"]
        self.module_sha256 = self.loaded_identity["backend"]["module_sha256"]
        self.require_unchanged("while loading")
        self.contextual_preflight = set()  # interfaces whose ExplorerRL markers were checked
        self.workspace = workspace
        workspace.mkdir(parents=True)
        self.profile = workspace / "profile"
        self.local_map, self.profile_environment = write_profile(self.profile, launcher.data, launcher.game_map,
                                                                 launcher.settings(1))
        self.original_cwd = Path.cwd()
        self.stripped_environment = sorted(key for key in os.environ if key.startswith(STRIPPED_PREFIXES))
        for key in self.stripped_environment:
            del os.environ[key]
        os.environ.update(self.profile_environment)
        # VCMI's development layout resolves data and libraries against ".".
        sys.path[:] = [str((self.original_cwd / entry).resolve()) if not Path(entry).is_absolute() else entry
                       for entry in sys.path]
        os.chdir(self.binary_directory)
        spec = importlib.util.spec_from_file_location(embedded.MODULE_NAME, self.module_path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        if self.module.PROTOCOL != embedded.PROTOCOL:
            raise ValueError("Embedded engine module protocol differs")
        # Only the binding's own failures are engine failures; anything else,
        # such as a budget alarm raised by a signal handler, propagates.
        self.errors = (self.module.EngineError, ValueError, TypeError, OverflowError)
        self.resources_before_load = self.module.resources()
        started = time.perf_counter()
        self.engine = self.module.Engine(console_log=False)
        self.engine.load()
        self.load_seconds = time.perf_counter() - started
        self.resources_after_load = self.module.resources()
        self.require_unchanged("while loading")
        if scoring.digest(self.local_map) != self.map_sha256:
            raise ValueError("Embedded worker map copy differs from the loaded map")
        self.games = 0

    def require_unchanged(self, when="since this process loaded it"):
        # Listed again: an added library or a retargeted alias changes the inventory, not any old stat.
        current = runtime_files(self.launcher, self.module_path)
        if current != self.files or fingerprint(current) != self.fingerprint:
            raise ValueError(f"Native runtime, module or map changed on disk {when}; restart Python")

    def require_compatible(self, launcher):
        if (launcher.binary.parent != self.binary_directory or launcher.data != self.data
                or scoring.digest(launcher.game_map) != self.map_sha256):
            raise ValueError("One embedded engine serves one prepared runtime, data and map per process")

    def require_ready(self):
        if Path.cwd() != self.binary_directory:
            raise ValueError("The embedded engine owns the working directory; it was changed")
        if self.engine.poisoned:
            raise ValueError("Embedded engine is poisoned by an earlier failure; start a new process")
        if self.engine.active:
            raise ValueError("Embedded engine still holds an unfinished game")
        self.require_unchanged()
        if scoring.digest(self.local_map) != self.map_sha256:
            raise ValueError("Embedded worker map copy changed")

    def reset(self, **kwargs):
        self.games += 1
        self.engine.reset(**kwargs)

    def take(self, timeout, until_boundary):
        return self.engine.take(timeout, until_boundary)

    def submit_text(self, generation, decision, text):
        return self.engine.submit_text(generation, decision, text)

    def finish(self, natural, timeout):
        return self.engine.finish(natural=natural, timeout=timeout)

    def timings(self):
        return self.engine.timings()

    def resources(self):
        return self.module.resources()

    @property
    def poisoned(self):
        return self.engine.poisoned

    @property
    def active(self):
        return self.engine.active


class EmbeddedProcess:
    """What the environment asks of a finished game process, without one."""
    pid = None

    def __init__(self):
        self.finished, self.returncode, self.report, self.error = False, None, None, None

    def complete(self, report, error=None):
        self.finished, self.report, self.error = True, report, error
        self.returncode = report["exit_code"] if report is not None else None

    def poll(self):
        return self.returncode if self.finished else None

    def wait(self, timeout=None):
        # finish() has joined every episode thread before any end-of-stream.
        return self.returncode


class EmbeddedEvents:
    """queue.Queue view of the learner's own stream, filled from the engine.

    Items keep the socket backend's shape and order: messages, then at the end
    of the game the stream's end and, when the engine reported one, its exit
    status. A decode refusal is delivered after its prefix, like a reader that
    stopped at an undecodable line.
    """

    def __init__(self, env):
        self.env, self.engine, self.player = env, env.engine, env.learner_player
        # The binding's own refusals; a signal handler's exception (budget alarm,
        # Ctrl-C) propagates to the driver exactly as from the socket queue.
        self.errors = getattr(self.engine, "errors", (RuntimeError, ValueError))
        self.items = deque()
        self.token = None
        self.end_pending = None  # natural flag once the engine reported its end
        self.ended = self.refused = False
        self.superseded = 0  # answers the player's own terminal made moot, as on a socket

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        if not self.items:
            self._refill(WAIT_SLICE_SECONDS if timeout is None else timeout, True)
        if not self.items:
            raise queue.Empty
        return self.items.popleft()

    def get_nowait(self):
        if not self.items:
            self._refill(0.0, False)
        if not self.items:
            raise queue.Empty
        return self.items.popleft()

    def _refill(self, timeout, until_boundary):
        if self.ended:
            return
        if self.end_pending is None:
            self._pull(timeout, until_boundary)
        if not self.items and self.end_pending is not None:
            self._end(self.end_pending)

    def _pull(self, timeout, until_boundary):
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            wait = min(WAIT_SLICE_SECONDS, max(0.0, deadline - time.monotonic()))
            try:
                delivery = self.engine.take(wait, until_boundary)
            except self.errors as error:
                self.items.append(("bridge_error", self.player, f"{type(error).__name__}: {error}", time.monotonic()))
                self.end_pending = False
                return
            if self._deliver(delivery) or time.monotonic() >= deadline:
                return

    def _deliver(self, delivery):
        received = time.monotonic()
        if not self.refused:
            for message in delivery["messages"]:
                self.items.append(("message", self.player, message, received))
            if delivery["decode_error"] is not None:
                self.items.append(("bridge_error", self.player, delivery["decode_error"], received))
                self.refused = True
        if delivery["needs_action"]:
            self.token = (delivery["generation"], delivery["decision"])
        if delivery["failure"] is not None or delivery["closed"]:
            self.end_pending = False
        elif delivery["terminal"]:
            self.end_pending = True
        return bool(delivery["messages"]) or delivery["needs_action"] or self.end_pending is not None

    def submit(self, response):
        """Answer the waiting native decision; returns a refusal text or None."""
        if self.token is None and not self.ended and self.end_pending is None:
            # The decision message precedes its native wait by a moment.
            deadline = time.monotonic() + self.env.limits.bridge_timeout
            while self.token is None and self.end_pending is None and time.monotonic() < deadline:
                self._pull(deadline - time.monotonic(), True)
        if self.token is None:
            if self.end_pending is True and not self.ended:
                # The player's own terminal came before the decision's native wait:
                # the socket backend writes this line to a peer that already ended.
                self.superseded += 1
                return None
            return "No native decision is waiting for a response"
        (generation, decision), self.token = self.token, None
        try:
            # The socket backend's own line and parser: identical admission and values.
            applied = self.engine.submit_text(generation, decision, json.dumps(response, separators=(",", ":")))
        except self.errors as error:
            return f"{type(error).__name__}: {error}"
        if applied is False:
            self.superseded += 1  # admitted after the own terminal, never applied
        return None

    def stop(self):
        """The host ends the game now: the socket backend's SIGTERM, without a process."""
        if not self.ended:
            self._end(False)

    def _end(self, natural):
        self.token = None
        env = self.env
        if natural:
            deadline = env.shutdown_deadline or time.monotonic() + env.limits.shutdown_seconds
            timeout = max(0.0, deadline - time.monotonic())
        else:
            timeout = 0.0
        try:
            # A foreign exception leaves ended unset: the binding keeps the report
            # and the next stop() collects it.
            report = self.engine.finish(natural, timeout)
        except self.errors as error:
            self.ended = True
            text = f"{type(error).__name__}: {error}"
            env.process.complete(None, text)
            self.items.append(("bridge_error", self.player, "Embedded engine close failed: " + text, time.monotonic()))
            return
        self.ended = True
        received = time.monotonic()
        if not self.refused:
            for message in report["messages"]:
                self.items.append(("message", self.player, message, received))
            if report["decode_error"] is not None:
                self.items.append(("bridge_error", self.player, report["decode_error"], received))
        if report["channel_failure"] is not None and report["exit_code"] is None:
            # Only when no engine exit status explains the end; after the stream,
            # so an ExplorerRL error record in it classifies first.
            self.items.append(("bridge_error", self.player, report["channel_failure"], received))
        env.process.complete(report)
        if not self.refused and report["decode_error"] is None:
            # A reader that stopped at an undecodable line never reaches its end of stream.
            self.items.append(("eof", self.player, None, received))
        if report["exit_code"] is not None:
            self.items.append(("exit", None, report["exit_code"], received))


class EmbeddedBuiltinEnv(BuiltinHeadlessEnv):
    """BuiltinHeadlessEnv semantics over the in-process engine."""

    def __init__(self, launcher, output, *, learner_player, limits=Limits(), agents=None, ledger=True,
                 replay=True, interface=c.DIALOGS_ENVIRONMENT, engine=None, workspace=None):
        if not isinstance(launcher, EmbeddedLauncher) and engine is None:
            raise ValueError("The embedded backend requires an EmbeddedLauncher")
        output = Path(output).expanduser()
        if not output.is_absolute():
            raise ValueError("The embedded backend owns the working directory; use an absolute output")
        super().__init__(launcher, output, learner_player=learner_player, limits=limits, agents=agents,
                         ledger=ledger, replay=replay, interface=interface)
        self._imported_sources.update({"experiments/headless/embedded_env.py": IMPORTED_SOURCE_SHA256,
            "experiments/headless/embedded_evidence.py": embedded.IMPORTED_SOURCE_SHA256})
        self.engine = engine if engine is not None else EmbeddedEngine.acquire(
            launcher, workspace or self.output / f"embedded-worker-{os.getpid()}")
        self.native_close = None

    def _preflight_contextual(self):
        # The engine's files are fingerprinted at load and checked on every reset,
        # so their contents are read once per process and interface, not once per game.
        checked = getattr(self.engine, "contextual_preflight", None)
        if checked is None or self.interface not in checked:
            super()._preflight_contextual()
            if checked is not None:
                checked.add(self.interface)

    def _runtime_identity(self, directory):
        loaded = getattr(self.engine, "runtime_sha256", None)
        return dict(loaded) if loaded is not None else super()._runtime_identity(directory)

    def reset(self, episode, seed):
        if type(episode) is not int or episode < 1 or type(seed) is not int or not 1 <= seed <= 2147483647:
            raise ValueError("Use a positive episode and signed-32-bit engine seed")
        self.close()
        self._assert_sources_unchanged()
        self._preflight_contextual()
        self.engine.require_ready()
        self._identity_cache = self._context_identity = None
        loaded = getattr(self.engine, "loaded_identity", None)
        if loaded is not None:
            # What this process loaded, not what is on disk now (require_ready proved they agree).
            self._identity_cache = dict(loaded)
        identity = self._identity()
        library = next((sha for name, sha in identity["native_runtime_sha256"].items()
                        if name in ("AI/libNullkiller.dylib", "AI/libNullkiller.so")), None)
        self.opponent_identity = evidence.opponent_identity(self.learner_player, library)
        self.episode, self.seed = episode, seed
        self.episode_dir = self.output / f"episode-{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.request, self.result, self.env_step = None, None, 0
        self.closing = self.settling = False
        self.stopping = threading.Event()
        self.reason = self.detail = self.cutoff_reason = self.dialog_cutoff_drain = None
        self.engine_wait = self.agent_wait = 0.0
        self.step_latencies = []
        self.costs = {"ingest": 0.0, "act": 0.0, "ledger": 0.0, "respond": 0.0}
        self.replay_tail_wait = 0.0
        self.last_engine_message = self.shutdown_deadline = None
        self.native_close = None
        player, color = self.learner_player, COLORS[self.learner_player]
        producer = ContextualProducer(str(episode), player, interface=self.interface)
        side = _Side(player, color, producer, producer.history, None)
        self.sides = {player: side}
        folder = self.episode_dir / "agents" / color
        folder.mkdir(parents=True)
        side.ledger = (folder / "bridge.jsonl").open("x")
        (self.episode_dir / "logs").mkdir()
        capture = self.episode_dir / "engine"
        # What EngineSession::reset sets in-process, recorded as launch evidence.
        environment = {**getattr(self.engine, "profile_environment", {}), "EXPLORER_RL_PLAYERS": str(player),
            "EXPLORER_RL_BUILTIN_OPPONENT": "Nullkiller", "EXPLORER_RL_INTERFACE": self.interface,
            "EXPLORER_RL_TRANSPORT": "in-process", "EXPLORER_RL_EPISODE": str(episode),
            "EXPLORER_RL_TIMEOUT": str(self.limits.bridge_timeout), "EXPLORER_RL_PROFILE": "0",
            "VCMI_EXPERIMENT_DIR": str(capture), "VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE": "1",
            "VCMI_EXPERIMENT_REPLAY": "1", "VCMI_EXPERIMENT_NO_CHEATS": "1"}
        self.started = time.monotonic()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.process = EmbeddedProcess()
        self.events = EmbeddedEvents(self)
        try:
            self.launch_info = self.launcher.episode_profile(self.episode_dir, seed, environment,
                                                             getattr(self.engine, "local_map", None))
            self.engine.reset(episode=str(episode), map="Maps/headless" + self.launcher.game_map.suffix.lower(),
                              capture_directory=str(capture), seed=seed, learner=player,
                              policy_timeout=self.limits.bridge_timeout,
                              log_path=str(self.episode_dir / "logs/process.log"), interface=self.interface)
        except BaseException:
            self.close()
            raise
        return self._advance()

    def _send_current(self, action, *, continuation=False):
        if continuation:
            raise ValueError("Single-player cutoff never sends technical or strategic continuation")
        return super()._send_current(action)

    def _write_response(self, side, response):
        """Through the engine instead of a socket: a refusal text, or None once admitted."""
        return self.events.submit(response)

    def close(self):
        self.request = None
        if getattr(self, "stopping", None) is not None:
            self.stopping.set()
        events = self.events
        if self.process is not None and isinstance(events, EmbeddedEvents) and not events.ended:
            events.stop()
        for side in self.sides.values():
            if side.ledger is not None and not side.ledger.closed:
                side.ledger.close()
        self.process = None

    def _finish(self, exited):
        """BuiltinHeadlessEnv._finish with the engine's close report as process evidence."""
        if self.result is not None:
            return self.result
        pending = self._settle()
        exited = exited if exited is not None else pending
        ended = self.process.finished
        declared_stop = self.reason in scoring.HEADLESS_TRUNCATIONS and not ended
        if ended and self.reason in scoring.HEADLESS_TRUNCATIONS:
            self.reason = self.detail = None  # the engine ended on its own first; its stream decides
        if not ended:
            self.events.stop()
        returncode = self.process.returncode
        side = self.sides[self.learner_player]
        self._settle()
        self.stopping.set()
        self.request = None
        if side.ledger is not None:
            side.ledger.close()
        report = self.process.report
        close = embedded.close_record(report) if report is not None else None
        self.native_close = close
        log_path = self.episode_dir / "logs/process.log"
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        marker = next((m for m in ENGINE_FAILURE_MARKERS if m in log), None)
        if marker:
            self._note_problem("engine_error", "Authoritative engine failure: " + marker)
        if side.error or "ExplorerRL: " in log or "ExplorerRL terminal delivery failed" in log:
            self._note_problem("protocol_error", side.error or "Native policy failure")
        if hasattr(self.engine, "require_unchanged"):
            try:
                self.engine.require_unchanged("during this game")
            except ValueError as error:
                self._note_problem("engine_error", str(error))
        if report is None:
            self._note_problem("engine_error", "Embedded engine close failed: " + str(self.process.error))
        elif report["poisoned"] or report["close_error"] is not None or report["experiment_failed"]:
            self._note_problem("engine_error", "Embedded engine close found a failure: "
                               + (report["close_error"] or report["channel_failure"] or "native failure"))
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
                    raise ValueError("Own stream and native terminal disagree")
                if self.cutoff_reason is not None:
                    raise ValueError("A terminal after frozen cutoff cannot score its earlier prefix")
                if not embedded.natural_closed({"returncode": returncode, "native_close": close}):
                    raise ValueError("Embedded engine did not close cleanly after the natural terminal")
                status, termination = "completed", "natural_terminal"
            except (OSError, ValueError, TypeError, KeyError) as error:
                barrier_error = f"{type(error).__name__}: {error}"
                status, termination, self.detail = "failed", "terminal_barrier_invalid", barrier_error
        shutdown_verified = (declared_stop and close is not None
                             and embedded.truncation_closed({"returncode": returncode, "native_close": close}))
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
        profile = evidence.profile_for(self.interface)
        path, live = folder / profile.history_filename, folder / ("live-" + profile.history_filename)
        sha, live_sha = persist_history(path, history), persist_history(live, side.validator.live)
        boundary = {"completed": "natural", "truncated": "cutoff", "failed": "failure"}[status]
        declaration = {"ready": side.validator.ready, "actions_sent": side.actions_sent, "error": side.error,
            "terminal": {"outcome": side.validator.terminal} if side.validator.terminal else None,
            "actions_accepted": side.accepted, "actions_rejected": side.rejected,
            "responses_undelivered": side.responses_undelivered,
            profile.history_key: {"path": str(path.relative_to(self.episode_dir)), "sha256": sha,
                "canonical_sha256": sha, "prefix_sha256": sha, "boundary": boundary, "complete": complete},
            "live_evidence": {"path": str(live.relative_to(self.episode_dir)), "sha256": live_sha,
                              "continuation_steps": 0, "learner_input": False}}
        final = {"schema": evidence.FINAL_SCHEMA, "episode": str(self.episode), "learner_player": self.learner_player,
            "controllers": evidence.controllers(self.learner_player), "returncode": returncode,
            "declared_stop": declared_stop, "shutdown_verified": shutdown_verified, "reason": self.reason,
            "detail": self.detail, "stream_complete": complete, "engine_evidence": engine,
            "engine_evidence_error": engine_error, "process_log_sha256": scoring.digest(log_path) if log_path.is_file() else None,
            "terminal_barrier": barrier, "terminal_barrier_error": barrier_error, "native_capture": capture,
            "socket_outcome": side.validator.terminal, "engine_verified_outcomes": outcomes if status == "completed" else None,
            "raw_bridge_sha256": {side.color: scoring.digest(folder / "bridge.jsonl")}, "identity": identity,
            "dialog_cutoff_drain": self.dialog_cutoff_drain, "history_sha256": sha, "native_close": close}
        proof_path = self.episode_dir / "builtin-finalization.json"
        scoring.write_json(proof_path, final)
        manifest = {"schema_version": 1, "kind": evidence.MANIFEST_KIND, "episode": str(self.episode),
            "environment": self.interface, "history_schema": profile.history_schema, "action_schema": profile.action_schema,
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


def fresh_workspace(prefix="heroes-embedded-"):
    """A new absolute directory for a worker's private engine profile."""
    return Path(tempfile.mkdtemp(prefix=prefix)).resolve() / "engine"
