"""Headless two-player self-play environment over the real VCMI rules.

The engine owns the game loop and asks each ExplorerRL side for a response at
its own decision points. This adapter turns that into an agent-environment
cycle: ``reset`` launches an isolated engine process and returns the first
request; ``step`` answers the current request and returns the next one, or the
end of the episode. Each request belongs to exactly one player and carries only
that player's sanitized history (see ``player_history.py``).

Rewards are 0 on every ordinary transition. A +1/-1 reward exists only after a
natural result was delivered to both players, the engine exited cleanly and the
evidence was verified. Day/action horizons are truncations; a watchdog, crash,
protocol violation or engine error is a technical failure. Neither is scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bridge_protocol import BridgeValidator, ProtocolError  # noqa: E402
from player_history import PlayerHistory  # noqa: E402

_SCORING_SPEC = importlib.util.spec_from_file_location("heroes_episode_scoring", ROOT / "experiments/episodes/scoring.py")
scoring = importlib.util.module_from_spec(_SCORING_SPEC)
_SCORING_SPEC.loader.exec_module(scoring)

COLORS = scoring.SELF_PLAY_COLORS
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Limits:
    max_days: int = 40
    max_actions: int = 2000  # adventure actions per player
    watchdog_seconds: float = 60.0  # longest engine silence while the environment waits for it
    startup_seconds: float = 180.0  # launch until both bridges are ready
    shutdown_seconds: float = 30.0  # last terminal until the process exits
    bridge_timeout: int = 60  # EXPLORER_RL_TIMEOUT: engine-side wait for each response

    def __post_init__(self):
        if (type(self.max_days) is not int or type(self.max_actions) is not int or self.max_days < 1
                or self.max_actions < 1 or not 1 <= self.bridge_timeout <= 300
                or min(self.watchdog_seconds, self.startup_seconds, self.shutdown_seconds) <= 0):
            raise ValueError("Use positive integer horizons, positive time limits and a 1..300 s bridge timeout")


@dataclass
class Request:
    """The current player must answer this request with ``env.step(value)``."""
    color: str
    player: int
    kind: str  # "action" or "decision"
    record: dict  # learner record of the observation or decision
    since_last: list[dict]  # this player's records since its previous response, ending with ``record``
    frames: list[dict]  # this player's history-v1 frames since its previous request; shared, do not mutate
    mask: list[bool] | None = None
    choices: list[int] | None = None
    reward: int = 0  # ordinary transitions are never rewarded
    done: bool = False
    # True at and after this player's own horizon boundary: the request is
    # answered so the other player can reach its boundary, but transitions
    # from here on belong to no training trajectory.
    past_horizon: bool = False


@dataclass
class EpisodeEnd:
    status: str  # completed, truncated or failed
    termination_reason: str
    outcomes: dict
    rewards: dict  # +1/-1 per player only for a verified natural result, else None
    final_records: dict  # per player: records since its last response
    final_frames: dict  # per player: history-v1 frames since its last request
    bootstrap: dict  # per player at a truncation: its first decision past the horizon, else None
    manifest: dict
    done: bool = True
    bootstrap_v1_seq: dict = field(default_factory=dict)  # per player: that decision's history-v1 sequence, else None


@dataclass
class _Side:
    player: int
    color: str
    validator: BridgeValidator
    history: PlayerHistory
    listener: socket.socket
    connection: socket.socket | None = None
    ledger: object = None
    reader: threading.Thread | None = None
    eof: bool = False
    error: str | None = None
    limit: str | None = None
    actions_sent: int = 0
    decisions_sent: int = 0
    accepted: int = 0
    rejected: int = 0
    unread_from: int = 0  # first history record not yet shown to the agent
    unread_frames_from: int = 0  # first history-v1 frame not yet shown to the agent
    ready_at: float | None = None
    terminal_at: float | None = None
    cutoff: dict | None = None  # this player's first decision observation past a declared horizon
    cutoff_reason: str | None = None  # the horizon that decision passed
    write_error: str | None = None  # a response the engine could no longer receive
    cutoff_seq: int | None = None  # that decision's history-v1 sequence while the projection is valid
    bytes_received: int = 0
    perf: dict = field(default_factory=dict)


class VcmiLauncher:
    """Launches the real headless client with an isolated per-episode profile."""

    def __init__(self, binary: Path, data: Path, game_map: Path, log_level: str = "info"):
        self.binary = binary.expanduser().resolve()
        self.data = data.expanduser().resolve()
        self.game_map = game_map.expanduser().resolve()
        if log_level not in {"trace", "debug", "info"}:
            raise ValueError("Use log level trace, debug or info; scoring needs info-level engine evidence")
        self.log_level = log_level
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        self.ai_library = self.binary.parent / "AI" / f"libExplorerRL{suffix}"

    def preflight(self) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError(f"Build the client first: {self.binary}")
        content = self.binary.read_bytes()
        if b"VCMI_EXPERIMENT_DIR" not in content or b"EXPLORER_RL_PLAYERS" not in content:
            raise ValueError("Client lacks the headless simulation patch; rebuild vcmi/build")
        if not self.ai_library.is_file() or b"visible_update-v2" not in self.ai_library.read_bytes():
            raise ValueError(f"Rebuild the protocol-3 ExplorerRL module with visible_update-v2: {self.ai_library}")
        if not (self.data / "Data").is_dir():
            raise ValueError(f"Missing original game assets: {self.data / 'Data'}")
        if not self.game_map.is_file() or self.game_map.suffix.lower() not in {".vmap", ".h3m"}:
            raise ValueError(f"Expected a real .vmap or .h3m map: {self.game_map}")

    def identity(self) -> dict:
        return {"binary": str(self.binary), "binary_sha256": scoring.digest(self.binary),
                "explorer_rl_sha256": scoring.digest(self.ai_library),
                "map": {"name": self.game_map.name, "sha256": scoring.digest(self.game_map)},
                "log_level": self.log_level}

    def settings(self, seed: int) -> dict:
        return {
            "general": {"lastDifficulty": 0, "music": 0, "sound": 0},
            "server": {"seed": seed, "useProcess": False, "localPort": 0},
            "ai": {"adventureEnemyAI": "Nullkiller", "adventureAlliedAI": "Nullkiller",
                   "combatEnemyAI": "BattleAI", "combatAlliedAI": "BattleAI", "combatNeutralAI": "BattleAI"},
            # Trace logging is a measured hot spot; game evidence is logged at info.
            "logging": {"loggers": [{"domain": "global", "level": self.log_level},
                                    {"domain": "rng", "level": "info"}]},
        }

    def launch(self, directory: Path, seed: int, environment: dict, log) -> tuple[subprocess.Popen, dict]:
        profile, logs, engine = directory / "profile", directory / "logs", directory / "engine"
        for folder in (logs, engine):
            folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            data_root, config_root = profile, profile / "config"
            profile_env = {"VCMI_USER_DATA_DIR": str(profile)}
        else:
            data_root, config_root = profile / "data" / "vcmi", profile / "config" / "vcmi"
            profile_env = {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
                           "XDG_CACHE_HOME": str(profile / "cache")}
        for folder in (data_root / "Maps", config_root):
            folder.mkdir(parents=True, exist_ok=True)
        (data_root / "Data").symlink_to(self.data / "Data", target_is_directory=True)
        local_map = data_root / "Maps" / ("headless" + self.game_map.suffix.lower())
        shutil.copyfile(self.game_map, local_map)
        settings = self.settings(seed)
        scoring.write_json(config_root / "settings.json", settings)
        command = [str(self.binary), "--headless", "--testmap", "Maps/" + local_map.name,
                   "--nointro", "--disable-video", "--logLocation", str(logs)]
        launch_env = {**profile_env, "VCMI_EXPERIMENT_DIR": str(engine), "VCMI_EXPERIMENT_NO_CHEATS": "1", **environment}
        inherited = {key: value for key, value in os.environ.items()
                     if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
        process = subprocess.Popen(command, cwd=self.binary.parent, env={**inherited, **launch_env},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process, {"command": command, "environment": launch_env, "requested_settings": settings}


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop the private process group this environment started, stragglers included."""
    group = process.pid  # started with start_new_session, so the leader's PID names the group
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(group, signal.SIGKILL)  # anything still alive in the group
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


class HeadlessSelfPlayEnv:
    """Agent-environment cycle over one isolated engine process per episode."""

    def __init__(self, launcher, output: Path, limits: Limits = Limits(), *, ledger: bool = True,
                 profile: bool = False, agents: dict | None = None, snapshot_every: int = 1):
        if type(snapshot_every) is not int or not 0 <= snapshot_every <= 1000000:
            raise ValueError("snapshot_every must be an integer from 0 to 1000000")
        self.launcher, self.limits, self.ledger_enabled, self.profile = launcher, limits, ledger, profile
        # Decision observations repeat the visible tiles every N decisions (never
        # when 0) so the adapter can verify its event fold; events are never thinned.
        self.snapshot_every = snapshot_every
        self._identity_cache = None
        self.output = output.expanduser().resolve()
        self.agent_descriptions = agents or {}
        self.process = None
        self.sides: dict[int, _Side] = {}
        self.events: queue.Queue = queue.Queue()
        self.request: Request | None = None
        self.result: EpisodeEnd | None = None
        self.episode_dir = None
        self.env_step = 0

    # -- public API --------------------------------------------------------
    def reset(self, episode: int, seed: int) -> Request | EpisodeEnd:
        if type(episode) is not int or episode < 1 or type(seed) is not int or not 1 <= seed <= 2147483647:
            raise ValueError("Use a positive episode number and a positive signed-32-bit engine seed")
        self.close()
        self.episode, self.seed = episode, seed
        self.episode_dir = self.output / f"episode-{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.events = queue.Queue()
        self.request, self.result, self.env_step = None, None, 0
        self.stopping = threading.Event()
        self.reason = self.detail = None
        self.engine_wait = self.agent_wait = 0.0
        self.step_latencies = []  # seconds from each response until the next request or the end
        self.last_engine_message = None
        self.shutdown_deadline = None
        self.sides = {}
        environment = {"EXPLORER_RL_PLAYERS": "0,1", "EXPLORER_RL_EPISODE": str(episode),
                       "EXPLORER_RL_TIMEOUT": str(self.limits.bridge_timeout),
                       "EXPLORER_RL_PROFILE": "1" if self.profile else "0",
                       "EXPLORER_RL_SNAPSHOT_EVERY": str(self.snapshot_every)}
        for player, color in enumerate(COLORS):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            side = _Side(player, color, BridgeValidator(player, str(episode)), PlayerHistory(player, color, str(episode)),
                         listener)
            if self.ledger_enabled:
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
            side.reader = threading.Thread(target=self._read, args=(side,), name=f"bridge-{side.color}", daemon=True)
            side.reader.start()
        threading.Thread(target=self._watch_process, args=(self.process, self.events), name="engine-exit", daemon=True).start()
        return self._advance()

    def step(self, value: int) -> Request | EpisodeEnd:
        if self.request is None:
            raise ValueError("No current request; call reset() or start a new episode")
        side = self.sides[self.request.player]
        response = side.validator.respond(value)  # raises ValueError without sending on an invalid choice
        self.agent_wait += time.monotonic() - self.request_time
        try:
            side.connection.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
        except OSError as error:
            # The engine closes the socket after its own terminal or failure. Its
            # queued messages and exit status say which: a missing terminal still
            # fails the episode through the end-of-stream and exit paths.
            side.write_error = f"{type(error).__name__}: {error}"
        self._write_ledger(side, "policy", response)
        if "action" in response:
            side.history.act(response["action"])
            side.actions_sent += 1
        else:
            side.decisions_sent += 1
        self.env_step += 1
        self.request = None
        sent = time.monotonic()
        result = self._advance()
        self.step_latencies.append(time.monotonic() - sent)
        return result

    def truncate(self, reason: str = "collection_window") -> EpisodeEnd:
        """End the episode at the current request as a declared horizon."""
        if self.request is None or reason not in scoring.HEADLESS_TRUNCATIONS:
            raise ValueError("Truncate only at a pending request, with a declared horizon")
        side = self.sides[self.request.player]
        if self.request.kind == "action" and side.cutoff is None:
            side.cutoff, side.cutoff_seq, side.cutoff_reason = self.request.record, side.history.v1.decision, reason
        side.limit = side.cutoff_reason or reason
        self._note_problem(reason, f"{side.color} stopped at a declared horizon")
        self.request = None
        return self._finish(None)

    def history(self, color: str) -> list[dict]:
        """A copy of one player's complete learner history."""
        side = next(side for side in self.sides.values() if side.color == color)
        return json.loads(json.dumps(side.history.records))

    def history_v1(self, color: str) -> list[dict]:
        """A copy of one player's complete heroes-player-history-v1 frames."""
        side = next(side for side in self.sides.values() if side.color == color)
        return json.loads(json.dumps(side.history.v1.frames))

    def memory(self, color: str) -> list[dict]:
        side = next(side for side in self.sides.values() if side.color == color)
        return side.history.memory()

    def close(self) -> None:
        self.request = None
        if getattr(self, "stopping", None) is not None:
            self.stopping.set()
        if self.process is not None:
            stop_process_group(self.process)
        for side in self.sides.values():
            for sock in (side.connection, side.listener):
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            if side.reader is not None:
                side.reader.join(timeout=3)
            if side.ledger is not None and not side.ledger.closed:
                side.ledger.close()
        if getattr(self, "log", None) is not None and not self.log.closed:
            self.log.close()
        self.process = None

    # -- engine stream -----------------------------------------------------
    def _read(self, side: _Side) -> None:
        events, stopping = self.events, self.stopping
        try:
            # Accept polls only before the bridge connects, so that an engine
            # that dies during startup cannot leave this thread blocked.
            side.listener.settimeout(0.25)
            while True:
                try:
                    connection, _ = side.listener.accept()
                    break
                except socket.timeout:
                    if stopping.is_set():
                        return
            connection.settimeout(None)
            # Responses are tiny single lines; never let Nagle's algorithm hold them.
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            side.connection = connection
            if stopping.is_set():
                return  # close() may have run before the connection was published
            buffer, scanned = bytearray(), 0
            while True:
                try:
                    chunk = connection.recv(1 << 16)
                except (ConnectionResetError, BrokenPipeError):
                    chunk = b""  # an engine that exits with unread input resets the connection
                if not chunk:
                    if buffer:
                        raise ProtocolError("Bridge stream ended inside a message")
                    events.put(("eof", side.player, None, time.monotonic()))
                    return
                side.bytes_received += len(chunk)
                buffer += chunk
                start = 0
                while (newline := buffer.find(b"\n", scanned)) >= 0:
                    events.put(("message", side.player, json.loads(buffer[start:newline]), time.monotonic()))
                    start = scanned = newline + 1
                if start:
                    del buffer[:start]
                scanned = len(buffer)
                if len(buffer) > MAX_MESSAGE_BYTES:
                    raise ProtocolError("Bridge message exceeds 16 MiB")
        except Exception as error:  # a reader must report every failure, never die silently
            if not stopping.is_set():
                events.put(("bridge_error", side.player, f"{type(error).__name__}: {error}", time.monotonic()))

    @staticmethod
    def _watch_process(process: subprocess.Popen, events: queue.Queue) -> None:
        events.put(("exit", None, process.wait(), time.monotonic()))

    def _note_problem(self, reason: str, detail: str) -> None:
        if self.reason is None:
            self.reason, self.detail = reason, detail

    def _advance(self) -> Request | EpisodeEnd:
        """Consume engine events until a request needs an answer or the episode ends."""
        exited = None
        while True:
            all_ready = all(side.validator.ready for side in self.sides.values())
            all_terminal = all(side.validator.terminal for side in self.sides.values())
            if self.reason or exited is not None:
                return self._finish(exited)
            if all_terminal:
                if self.shutdown_deadline is None:
                    self.shutdown_deadline = time.monotonic() + self.limits.shutdown_seconds
                deadline_kind, timeout = "shutdown", max(0.0, self.shutdown_deadline - time.monotonic())
            elif not all_ready:
                deadline_kind = "startup"
                timeout = max(0.0, self.started + self.limits.startup_seconds - time.monotonic())
            else:
                deadline_kind, timeout = "watchdog", self.limits.watchdog_seconds
            waited = time.monotonic()
            try:
                kind, player, payload, received = self.events.get(timeout=timeout)
            except queue.Empty:
                self._note_problem({"startup": "startup_timeout", "watchdog": "watchdog_timeout",
                                    "shutdown": "shutdown_timeout"}[deadline_kind],
                                   f"No engine event for {timeout:.1f} s")
                continue
            finally:
                self.engine_wait += time.monotonic() - waited
            if kind == "exit":
                exited = payload
                continue
            side = self.sides[player]
            if kind == "eof":
                side.eof = True
                if not side.validator.terminal:
                    # A crashed engine also closes the socket; prefer its exit status.
                    exited = self._await_exit(grace=3.0)
                    if exited is None:
                        self._note_problem("bridge_disconnected", f"{side.color} closed before its terminal")
                continue
            if kind == "bridge_error":
                self._note_problem("protocol_error", f"{side.color}: {payload}")
                continue
            request = self._ingest(side, payload, received)
            if request is not None:
                return request

    def _await_exit(self, grace: float):
        try:
            return self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return None

    def _ingest(self, side: _Side, message: dict, received: float, draining: bool = False) -> Request | None:
        """Validate, record and fold one engine message.

        While draining after the engine exited, a message is only recorded: it
        never becomes a request and never declares a horizon.
        """
        self.last_engine_message = received
        self._write_ledger(side, "engine", message)
        try:
            return self._ingest_checked(side, message, received, draining)
        except ProtocolError as error:
            self._note_problem("protocol_error", f"{side.color}: {error}")
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
            self._note_problem("protocol_error", f"{side.color}: malformed message ({type(error).__name__}: {error})")
        return None

    def _ingest_checked(self, side: _Side, message: dict, received: float, draining: bool) -> Request | None:
        kind = side.validator.check(message)
        if kind == "ready":
            side.ready_at = received - self.started
            side.history.start(message["map_size"])
            return None
        if kind == "error":
            side.error = str(message.get("message"))
            self._note_problem("engine_error", f"{side.color}: {side.error}")
            return None
        record = side.history.ingest(message)
        if "perf" in message:
            for key, value in message["perf"].items():
                if type(value) is int:
                    side.perf[key] = side.perf.get(key, 0) + value
        if kind == "action_result":
            if message["accepted"]:
                side.accepted += 1
            else:
                side.rejected += 1
        elif kind == "terminal":
            side.terminal_at = received - self.started
            return None
        if draining:
            return None
        if kind == "observation":
            past_day = record["day"] > self.limits.max_days
            if side.cutoff is None and (past_day or side.actions_sent >= self.limits.max_actions):
                side.cutoff_reason = "game_day_limit" if past_day else "agent_action_limit"
                side.cutoff, side.cutoff_seq = record, side.history.v1.decision
            # Both players pass a day horizon within one round, so play continues
            # until each has its own next-decision boundary. The action horizon is
            # a runaway guard and stops at once, also during that continuation.
            over_actions = side.actions_sent >= self.limits.max_actions
            if side.cutoff is not None and (over_actions or all(s.cutoff is not None for s in self.sides.values())):
                reason = "agent_action_limit" if over_actions else side.cutoff_reason
                for other in self.sides.values():
                    if other.cutoff is not None:
                        other.limit = other.cutoff_reason
                self._note_problem(reason, f"{side.color} reached the horizon")
                return None
        if kind in {"observation", "decision"} and self.request is None:
            return self._issue(side.player, record)
        return None

    def _issue(self, player: int, record: dict) -> Request:
        side = self.sides[player]
        kind = "action" if record["type"] == "observation" else "decision"
        # One copy for the agent; the stored history is never handed out.
        since_last = json.loads(json.dumps(side.history.records[side.unread_from:record["index"] + 1]))
        side.unread_from = record["index"] + 1
        frames = side.history.v1.frames[side.unread_frames_from:]
        side.unread_frames_from = len(side.history.v1.frames)
        record = since_last[-1]
        self.request = Request(side.color, player, kind, record, since_last, frames,
                               mask=record.get("action_mask") if kind == "action" else None,
                               choices=[c["choice"] for c in record["choices"]] if kind == "decision" else None,
                               past_horizon=side.cutoff is not None)
        self.request_time = time.monotonic()
        return self.request

    def _write_ledger(self, side: _Side, direction: str, message: dict) -> None:
        # Internal evidence with engine IDs: never a policy input.
        if side.ledger is not None:
            side.ledger.write(json.dumps({"t": round(time.monotonic() - self.started, 6), "direction": direction,
                                          "message": message}, separators=(",", ":")) + "\n")

    # -- episode end -------------------------------------------------------
    def _finish(self, exited) -> EpisodeEnd:
        stopped_by_environment = self.reason is not None
        if stopped_by_environment:
            stop_process_group(self.process)
        returncode = self.process.wait()
        finished = time.monotonic()
        self.stopping.set()
        # Drain messages already written before the exit; they belong to the evidence.
        for side in self.sides.values():
            if side.reader is not None:
                side.reader.join(timeout=2)
        while True:
            try:
                kind, player, payload, received = self.events.get_nowait()
            except queue.Empty:
                break
            if kind != "message":
                continue
            if stopped_by_environment:
                self._write_ledger(self.sides[player], "engine", payload)  # evidence only
            else:
                self._ingest(self.sides[player], payload, received, draining=True)
        self.request = None
        for side in self.sides.values():
            if side.ledger is not None:
                side.ledger.close()
        self.log.close()
        log_text = (self.episode_dir / "logs" / "process.log").read_text(errors="replace")
        executions = {side.color: self._execution(side) for side in self.sides.values()}
        status, outcomes, termination, rewards = scoring.classify_headless_self_play(returncode, self.reason, log_text, executions)
        natural_after_horizon = None
        horizon = next((side.cutoff_reason for side in self.sides.values() if side.cutoff is not None), None)
        if status == "completed" and horizon:
            # A natural ending during the horizon continuation follows actions
            # that belong to no trajectory, so it is reported but never scored.
            natural_after_horizon, outcomes = outcomes, {c: None for c in COLORS}
            status, termination, rewards = "truncated", horizon, {c: None for c in COLORS}
            self.detail = "natural terminal after a player's horizon"
        evidence, evidence_error = self._engine_evidence(log_text)
        if status in {"completed", "truncated"} and evidence_error:
            status, termination = "failed", "engine_evidence_invalid"
            outcomes, rewards = {c: None for c in COLORS}, {c: None for c in COLORS}
        terminals = [side.terminal_at for side in self.sides.values() if side.terminal_at is not None]
        readies = [side.ready_at for side in self.sides.values() if side.ready_at is not None]
        manifest = {
            "schema_version": 1, "kind": "headless-self-play", "episode": self.episode, "seed": self.seed,
            "created_at": self.created_at, "status": status, "termination_reason": termination,
            "failure_detail": self.detail if status != "completed" else None,
            "outcomes_by_player": outcomes, "rewards_by_player": rewards, "returncode": returncode,
            "natural_terminal_after_horizon": natural_after_horizon,
            "limits": {"max_days": self.limits.max_days, "max_actions_per_player": self.limits.max_actions,
                       "watchdog_seconds": self.limits.watchdog_seconds, "bridge_timeout": self.limits.bridge_timeout},
            "env_steps": self.env_step,
            "step_latency_seconds": latency_summary(self.step_latencies),
            "timing": {"wall_seconds": round(finished - self.started, 6),
                       "startup_seconds": round(max(readies), 6) if len(readies) == 2 else None,
                       "shutdown_seconds": round(finished - self.started - max(terminals), 6) if len(terminals) == 2 else None,
                       "engine_wait_seconds": round(self.engine_wait, 6), "agent_seconds": round(self.agent_wait, 6)},
            "players": {side.color: {**executions[side.color], **self._counters(side)} for side in self.sides.values()},
            "engine_evidence": evidence, "engine_evidence_error": evidence_error,
            "agents": self.agent_descriptions, "ledger": self.ledger_enabled,
            "identity": self._identity(),
            "launch": self.launch_info,
            "deterministic_replay_verified": False,
        }
        scoring.write_json(self.episode_dir / "manifest.json", manifest)
        bootstrap = {side.color: (json.loads(json.dumps(side.cutoff)) if status == "truncated" and side.cutoff else None)
                     for side in self.sides.values()}
        bootstrap_seq = {side.color: (side.cutoff_seq if status == "truncated" and side.cutoff is not None else None)
                         for side in self.sides.values()}
        final = {side.color: json.loads(json.dumps(side.history.records[side.unread_from:])) for side in self.sides.values()}
        final_frames = {side.color: side.history.v1.frames[side.unread_frames_from:] for side in self.sides.values()}
        self.result = EpisodeEnd(status, termination, outcomes, rewards, final, final_frames, bootstrap, manifest,
                                 bootstrap_v1_seq=bootstrap_seq)
        self.process = None
        return self.result

    def _execution(self, side: _Side) -> dict:
        terminal = {"outcome": side.validator.terminal} if side.validator.terminal else None
        return {"ready": side.validator.ready, "actions_sent": side.actions_sent, "error": side.error,
                "limit": side.limit, "terminal": terminal, "write_error": side.write_error,
                "cutoff": None if side.cutoff is None else {"day": side.cutoff["day"], "history_v1_seq": side.cutoff_seq}}

    def _counters(self, side: _Side) -> dict:
        history = side.history
        return {"decisions_sent": side.decisions_sent, "actions_accepted": side.accepted,
                "actions_rejected": side.rejected, "turns_completed": history.turns_completed,
                "last_day": history.day, "battles_started": history.battles_started,
                "battles_completed": history.battles_completed, "handles_assigned": history.handle_count(),
                "records": len(history.records), "visible_updates": side.validator.next_visible_sequence,
                "bytes_received": side.bytes_received, "ready_seconds": side.ready_at,
                "terminal_seconds": side.terminal_at, "native_perf_totals": side.perf,
                "coverage": coverage(history.records), "history_v1": history.v1.summary()}

    def _identity(self) -> dict:
        """Build identity, computed once per environment: it hashes binaries and runs git."""
        if self._identity_cache is None:
            self._identity_cache = {**(self.launcher.identity() if hasattr(self.launcher, "identity") else {}),
                                    "engine": _git_state(ROOT / "vcmi"), "project": _git_state(ROOT),
                                    "engine_recipe": scoring.engine_recipe_digests(ROOT)}
        return self._identity_cache

    def _engine_evidence(self, log_text: str) -> tuple[dict | None, str | None]:
        metadata_path = self.episode_dir / "engine" / "experiment-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
            validated = scoring.validate_experiment_metadata(metadata, 0, log_text, agent_players=[0, 1])
            seed = scoring.engine_seed_evidence(log_text, self.seed)
            return {"metadata": validated, "seed": seed}, None
        except (OSError, ValueError) as error:
            return None, f"{type(error).__name__}: {error}"


def _git_state(path: Path) -> dict | None:
    try:
        return scoring.git_state(path)
    except (OSError, subprocess.CalledProcessError):
        return None


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def quantile(q):
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 6)
    return {"count": len(ordered), "p50": quantile(0.5), "p90": quantile(0.9), "p99": quantile(0.99),
            "max": round(ordered[-1], 6), "mean": round(sum(ordered) / len(ordered), 6)}


def coverage(records: list[dict]) -> dict:
    """Which gameplay paths one player's history demonstrates; all from its own records."""
    counts = dict.fromkeys(("successful_moves", "stationary_accepted_moves", "rejected_attempts", "end_turns",
                            "town_only_turns", "town_only_decisions", "resource_gains_on_move", "mine_captures",
                            "town_captures", "neutral_battles", "hero_battles", "other_battles", "dialogs",
                            "visits"), 0)
    previous = None
    for record in records:
        for event in record.get("events", []):
            if event["kind"] == "battle_start":
                key = {"neutral": "neutral_battles", "hero": "hero_battles"}.get(event.get("opponent"), "other_battles")
                counts[key] += 1
            elif event["kind"] == "decision":
                counts["dialogs"] += 1
            elif event["kind"] == "visit":
                counts["visits"] += 1
        if record["type"] == "action_result":
            before, after = record["before"], record["after"]
            if not record["accepted"]:
                counts["rejected_attempts"] += 1
            elif record["action"] == 8:
                counts["end_turns"] += 1
                counts["town_only_turns"] += before.get("own_heroes") == 0 and before.get("own_towns", 0) > 0
            else:
                moved = (before.get("x"), before.get("y"), before.get("z")) != (after.get("x"), after.get("y"), after.get("z"))
                counts["successful_moves" if moved else "stationary_accepted_moves"] += 1
                gained = [b - a for a, b in zip(before.get("resources", []), after.get("resources", []))]
                counts["resource_gains_on_move"] += any(delta > 0 for delta in gained)
        state = record.get("state") or record.get("after")
        if isinstance(state, dict):
            if previous is not None:
                counts["mine_captures"] += state.get("own_mines", 0) > previous.get("own_mines", 0)
                counts["town_captures"] += state.get("own_towns", 0) > previous.get("own_towns", 0)
            counts["town_only_decisions"] += (record["type"] == "observation" and state.get("own_heroes") == 0
                                              and state.get("own_towns", 0) > 0)
            previous = state
    return counts
