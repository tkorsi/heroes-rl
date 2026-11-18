"""Two-seat self-play over one headless VCMI game.

``reset`` starts a game and returns the first ``Request``; ``step`` answers it
and returns the next one, or the ``End`` of the game. A request belongs to one
seat and carries that seat's new frames, the last of which is the decision to
answer. Only one request is outstanding at a time.

A game ends in one of three ways:

completed  the engine finished it and both seats received complementary
           win/loss terminals, confirmed by the engine's own log and a clean
           exit. Only this is ever scored.
truncated  a decision arrived past the day or action limit and the caller
           stopped the game there.
failed     anything else: a crash, a protocol error, a silent engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import socket
import subprocess
import threading
import time

from . import process as engine_process
from .protocol import NativeFailure, Producer, loads

COLORS = ("red", "blue")
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
READER_DRAIN_SECONDS = 5.0  # after the engine exits, for both readers to reach the end of their streams
# Lines the patched engine logs when it gives up on a game.
ENGINE_FAILURE_MARKERS = (
    "Experiment failure:", "Headless network thread failed:",
    "Headless connection to the server failed:", "Headless packet handling failed:",
    "Headless game start failed:", "Headless run received a silent game end",
    "ExplorerRL: ", "ExplorerRL terminal delivery failed",
)


@dataclass(frozen=True)
class Limits:
    max_days: int = 365  # a decision on a later game day ends the game
    max_actions: int = 163840  # answered decisions per seat
    idle_seconds: float | None = None  # longest engine silence; None waits forever


@dataclass
class Request:
    player: int
    frames: tuple  # the seat's frames since its previous request, ending with the decision
    limit: str | None = None  # set when the decision is past a limit: call ``stop`` instead of ``step``
    done = False


@dataclass
class End:
    status: str  # completed, truncated or failed
    reason: str
    outcomes: tuple | None = None  # ("win", "loss") or ("loss", "win") by seat, for a completed game
    detail: str | None = None
    done = True


class _Side:
    def __init__(self, player, producer, listener):
        self.player, self.producer, self.listener = player, producer, listener
        self.connection = None
        self.reader = None
        self.eof = False
        self.unread = 0  # first frame not yet handed to the policy
        self.issued = None  # sequence number of the last decision handed out
        self.actions = 0  # answers written to the engine


class Env:
    def __init__(self, engine, limits=Limits()):
        self.engine, self.limits = engine, limits
        self.process = None
        self.sides = ()
        self.log = None

    # -- the caller's interface ------------------------------------------------
    def reset(self, directory, episode, seed):
        """Start game ``episode`` with engine ``seed`` in the new folder ``directory``."""
        self.close()
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True)
        self.events = queue.Queue()
        self.stopping, self.exited = threading.Event(), threading.Event()
        self.request = self.result = self.failure = self.exit_code = self.limit = None
        sides = []
        for player in (0, 1):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            producer = Producer(str(episode), player, interface=self.engine.interface,
                                army_information=self.engine.army_information)
            sides.append(_Side(player, producer, listener))
        self.sides = tuple(sides)
        (self.directory / "logs").mkdir()
        self.log = (self.directory / "logs" / "process.log").open("w")
        try:
            self.process = self.engine.launch(self.directory, episode, seed,
                                              [side.listener.getsockname()[1] for side in self.sides], self.log)
        except BaseException:
            self.close()
            raise
        for side in self.sides:
            side.reader = threading.Thread(target=self._read, args=(side, self.events, self.stopping, self.exited),
                                           name=f"engine-{COLORS[side.player]}", daemon=True)
            side.reader.start()
        threading.Thread(target=lambda process=self.process, events=self.events:
                         events.put(("exit", None, process.wait())), daemon=True).start()
        return self._advance()

    def step(self, action):
        """Answer the current request and wait for the next one."""
        if self.request is None:
            raise ValueError("There is no request to answer")
        side = self.sides[self.request.player]
        response = side.producer.response(action)  # refuses an action the decision does not offer
        held = side.producer.history.decision
        self.request = None
        # Fold what arrived while the policy was thinking. The engine may have
        # withdrawn the decision or ended the game; the answer is then dropped
        # and whatever replaced the decision comes back as the next request.
        self._drain()
        if (self.failure is None and self.exit_code is None and side.producer.history.decision is held
                and not self._ending()):
            try:
                side.connection.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
            except OSError as error:
                self._fail("action_delivery_failed", f"{COLORS[side.player]}: {type(error).__name__}: {error}")
            else:
                side.producer.submitted(action)
                side.actions += 1
        return self._advance()

    def stop(self, reason):
        """Stop the game at a limit. A game that already ended on its own keeps its real result."""
        self.request = None
        self._drain()
        if self.failure is None and self.exit_code is None and not self._ending():
            self.limit = reason
            return self._finish()
        return self._advance()

    def frames(self, player):
        """Every frame the seat has received or produced so far."""
        return tuple(self.sides[player].producer.history._frames)

    @property
    def day(self):
        return max((side.producer.history.day for side in self.sides), default=1)

    def close(self):
        """Stop the engine and release sockets and files. The seats' frames stay readable."""
        if self.process is not None:
            engine_process.stop(self.process)
            self.process = None
        if self.sides:
            self.stopping.set()
        for side in self.sides:
            for sock in (side.connection, side.listener):
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            side.connection = side.listener = None
            if side.reader is not None:
                side.reader.join(timeout=3)
        if self.log is not None and not self.log.closed:
            self.log.close()

    # -- engine events ---------------------------------------------------------
    @staticmethod
    def _read(side, events, stopping, exited):
        """Reader thread: accept the seat's connection and queue its messages."""
        try:
            # Poll while waiting, so an engine that dies at startup cannot block this thread.
            listener = side.listener
            listener.settimeout(0.25)
            while True:
                # Read before the attempt: a connection made before the engine exited is still accepted.
                over = stopping.is_set() or exited.is_set()
                try:
                    connection, _ = listener.accept()
                    break
                except socket.timeout:
                    if over:
                        return
            connection.settimeout(None)
            # Answers are tiny single lines; never let Nagle's algorithm hold them.
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            side.connection = connection
            if stopping.is_set():
                return  # close() ran before the connection was published
            buffer, scanned = bytearray(), 0
            while True:
                try:
                    chunk = connection.recv(1 << 16)
                except (ConnectionResetError, BrokenPipeError):
                    chunk = b""  # an engine that exits with unread input resets the connection
                if not chunk:
                    if buffer:
                        raise ValueError("Stream ended inside a message")
                    events.put(("eof", side.player, None))
                    return
                buffer += chunk
                start = 0
                while (newline := buffer.find(b"\n", scanned)) >= 0:
                    events.put(("message", side.player, loads(buffer[start:newline])))
                    start = scanned = newline + 1
                if start:
                    del buffer[:start]
                scanned = len(buffer)
                if len(buffer) > MAX_MESSAGE_BYTES:
                    raise ValueError("Message exceeds 16 MiB")
        except Exception as error:  # a reader reports every failure; it never dies silently
            if not stopping.is_set():
                events.put(("error", side.player, f"{type(error).__name__}: {error}"))

    def _fail(self, reason, detail):
        if self.failure is None:  # the first failure is the cause
            self.failure = (reason, detail)

    def _ending(self):
        return any(side.producer.terminal for side in self.sides)

    def _fold(self, event):
        kind, player, payload = event
        if kind == "exit":
            self.exit_code = payload
            return
        side = self.sides[player]
        if kind == "message":
            try:
                side.producer.ingest(payload)
            except NativeFailure as error:
                self._fail("engine_error", f"{COLORS[player]}: {error}")
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                self._fail("protocol_error", f"{COLORS[player]}: {type(error).__name__}: {error}")
        elif kind == "error":
            self._fail("protocol_error", f"{COLORS[player]}: {payload}")
        elif kind == "eof":
            side.eof = True
            if side.producer.terminal is None and self.exit_code is None:
                # A crashing engine closes its sockets first; prefer its exit status.
                try:
                    self.exit_code = self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._fail("bridge_disconnected", f"{COLORS[player]} closed its connection before the game ended")

    def _drain(self):
        """Fold every queued event without waiting."""
        while True:
            try:
                self._fold(self.events.get_nowait())
            except queue.Empty:
                break
        if self.exit_code is None:
            self.exit_code = self.process.poll()

    def _next_request(self):
        for side in self.sides:
            history = side.producer.history
            decision = history.decision
            if decision is None or history.action is not None or side.issued == decision.seq:
                continue
            side.issued = decision.seq
            frames = tuple(history._frames[side.unread:])
            side.unread = len(history._frames)
            limit = ("game_day_limit" if decision.day > self.limits.max_days else
                     "agent_action_limit" if side.actions >= self.limits.max_actions else None)
            self.request = Request(side.player, frames, limit)
            return self.request
        return None

    def _advance(self):
        """Consume engine events until a seat must decide or the game ends."""
        while True:
            if self.failure is not None or self.exit_code is not None:
                return self._finish()
            # Once either seat has its terminal the engine is shutting the game down.
            if not self._ending():
                request = self._next_request()
                if request is not None:
                    return request
            event = self._wait()
            if event is None:
                self._fail("engine_silent", f"No engine event for {self.limits.idle_seconds} s")
            else:
                self._fold(event)

    def _wait(self):
        """The next engine event, or None once the engine was silent for too long.

        It waits a second at a time. A wait without a timeout is not interrupted
        by signals on every platform, and an actor that is told to stop must not
        sit out a silent engine.
        """
        idle = self.limits.idle_seconds
        deadline = None if idle is None else time.monotonic() + idle
        while True:
            try:
                return self.events.get(timeout=1 if deadline is None else max(0, min(1, deadline - time.monotonic())))
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return None

    # -- the end of a game -----------------------------------------------------
    def _finish(self):
        if self.result is not None:
            return self.result
        self.request = None
        stopped = self.process.poll() is None
        if stopped:
            # Only a limit or a failure gets here with the engine still running.
            engine_process.stop(self.process)
        returncode = self.process.wait()
        self.exited.set()  # a seat that never connected stops waiting for the engine
        deadline = time.monotonic() + READER_DRAIN_SECONDS
        for side in self.sides:
            side.reader.join(timeout=max(0, deadline - time.monotonic()))
        if not stopped:
            # The engine ended on its own: its last messages, the terminals among
            # them, may still be queued behind the exit event.
            if any(side.reader.is_alive() for side in self.sides):
                self._fail("protocol_error", "A reader did not reach the end of its stream")
            self._drain()
        self.process = None
        self.close()
        self.result = self._classify(returncode, stopped)
        return self.result

    def _classify(self, returncode, stopped):
        log = (self.directory / "logs" / "process.log").read_text(errors="replace")
        marker = next((marker for marker in ENGINE_FAILURE_MARKERS if marker in log), None)
        if marker is not None:
            self._fail("engine_error", "The engine log reports: " + marker.strip())
        if self.failure is not None:
            return End("failed", self.failure[0], detail=self.failure[1])
        if stopped:
            return End("truncated", self.limit)
        if returncode != 0:
            return End("failed", "unclean_exit", detail=f"The engine exited with status {returncode}")
        if any(not side.producer.ready or side.actions < 1 for side in self.sides):
            return End("failed", "seat_never_acted")
        if log.count("Experiment self-play complete. Ending game.") != 1:
            return End("failed", "unverified_terminal", detail="The engine log has no single end-of-game line")
        outcomes = []
        for side in self.sides:
            logged = re.findall(rf"Experiment player {side.player} (won|lost)\. Terminal received\.", log)
            outcome = {"won": "win", "lost": "loss"}[logged[0]] if len(logged) == 1 else None
            if outcome is None or side.producer.terminal != outcome:
                return End("failed", "unverified_terminal",
                           detail=f"{COLORS[side.player]}: the engine log and the socket disagree on the outcome")
            outcomes.append(outcome)
        if set(outcomes) != {"win", "loss"}:
            return End("failed", "unverified_terminal", detail="Both seats report the same outcome")
        return End("completed", "natural_terminal", tuple(outcomes))
