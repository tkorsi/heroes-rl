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

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time

from . import process as engine_process
from .protocol import NativeFailure, Producer, loads

COLORS = ("red", "blue")
TRAINING_DAY_LIMIT = 365
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
READER_DRAIN_SECONDS = 5.0  # after the engine exits, for both readers to reach the end of their streams
SETTLE_SECONDS = 5.0  # at a limit, for the engine's report on an action it was sent before the limit was met
# Lines the patched engine logs when it gives up on a game.
ENGINE_FAILURE_MARKERS = (
    "Experiment failure:", "Headless network thread failed:",
    "Headless connection to the server failed:", "Headless packet handling failed:",
    "Headless game start failed:", "Headless run received a silent game end",
    "ExplorerRL: ", "ExplorerRL terminal delivery failed",
)


def _send(connection, data):
    """Send all of ``data`` on a non-blocking connection, waiting while the engine's buffer is full."""
    view = memoryview(data)
    while view:
        try:
            view = view[connection.send(view):]
        except (BlockingIOError, InterruptedError):
            select.select([], [connection], [], 1.0)


class _Events:
    """The engine's events in the order they are read, as a queue the caller's thread fills itself.

    Each seat's connection is accepted by a thread of its own, and a thread
    waits for the engine's exit; every message is read and parsed here, in the
    thread that asks for the next event. Handing each message from a reader
    thread to this one cost two thread switches per message, more than the
    parsing itself. The events and their order per connection are those the
    reader threads queued: a seat's messages, its end of stream or a read
    failure, and the engine's exit. ``get`` and ``get_nowait`` are those of
    ``queue.Queue``; ``put`` may be called from any thread.
    """

    def __init__(self, sides, raw=False):
        self.sides = sides
        self.raw = raw  # queue each message as its line of bytes, for a producer that reads it itself
        self.items = deque()
        self.lock = threading.Lock()
        self.wake_read, self.wake_write = os.pipe()
        os.set_blocking(self.wake_read, False)
        os.set_blocking(self.wake_write, False)
        self.buffers = {side.player: bytearray() for side in sides}
        self.closed = set()  # players whose stream ended or failed

    def put(self, item):
        with self.lock:
            self.items.append(item)
            self._wake()

    def wake(self):
        with self.lock:
            self._wake()

    def _wake(self):
        # Under the lock: once close() has run the descriptors may belong to another file.
        if self.wake_write is None:
            return
        try:
            os.write(self.wake_write, b"x")
        except (BlockingIOError, OSError):
            pass  # the pipe is full: a wake-up is pending already

    def _readable(self):
        return [side for side in self.sides if side.connection is not None and side.player not in self.closed]

    def _pump(self, side):
        """Read what the seat's connection has without waiting, and queue its complete messages."""
        buffer = self.buffers[side.player]
        while True:
            try:
                chunk = side.connection.recv(1 << 16)
            except (BlockingIOError, InterruptedError):
                return
            except (ConnectionResetError, BrokenPipeError):
                chunk = b""  # an engine that exits with unread input resets the connection
            except OSError as error:
                self._end(side, ("error", side.player, f"{type(error).__name__}: {error}"))
                return
            if not chunk:
                if buffer:
                    self._end(side, ("error", side.player, "ValueError: Stream ended inside a message"))
                else:
                    self._end(side, ("eof", side.player, None))
                return
            start = len(buffer)
            buffer += chunk
            scanned, begin = start, 0
            while (newline := buffer.find(b"\n", scanned)) >= 0:
                try:
                    message = bytes(buffer[begin:newline]) if self.raw else loads(buffer[begin:newline])
                except Exception as error:  # a reader reports every failure; it never dies silently
                    self._end(side, ("error", side.player, f"{type(error).__name__}: {error}"))
                    return
                self.items.append(("message", side.player, message))
                begin = scanned = newline + 1
            if begin:
                del buffer[:begin]
            if len(buffer) > MAX_MESSAGE_BYTES:
                self._end(side, ("error", side.player, "ValueError: Message exceeds 16 MiB"))
                return
            if len(chunk) < (1 << 16):
                return  # most likely all there was; the next call reads the rest

    def _end(self, side, item):
        self.closed.add(side.player)
        self.items.append(item)

    def pump(self):
        for side in self._readable():
            self._pump(side)

    def get_nowait(self):
        # The oldest queued event is the one to hand out whether or not the connections are read first;
        # they are read when nothing is queued.
        if not self.items:
            self.pump()
        try:
            with self.lock:
                return self.items.popleft()
        except IndexError:
            raise queue.Empty from None

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                pass
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise queue.Empty
            waiting = [side.connection for side in self._readable()] + ([self.wake_read] if self.wake_read is not None else [])
            try:
                ready, _, _ = select.select(waiting, [], [], remaining)
            except (OSError, ValueError):
                ready = []  # a connection closed under the wait; the next round sees it
            if self.wake_read is not None and self.wake_read in ready:
                try:
                    while os.read(self.wake_read, 4096):
                        pass
                except (BlockingIOError, OSError):
                    pass
            else:
                # Only a connection that has something is read; the next round reads every one if nothing came.
                for side in self._readable():
                    if side.connection in ready:
                        self._pump(side)

    def read_to_end(self, deadline):
        """Read every connection to its end, or until ``deadline``; the events stay queued in their order."""
        while not self.ended():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.pump()
            if self.ended():
                return
            try:
                select.select([side.connection for side in self._readable()], [], [], min(remaining, 0.25))
            except (OSError, ValueError):
                pass

    def ended(self):
        """Whether every connection that was made has come to its end."""
        return all(side.player in self.closed for side in self.sides if side.connection is not None)

    def close(self):
        with self.lock:
            descriptors, self.wake_read, self.wake_write = (self.wake_read, self.wake_write), None, None
        for descriptor in descriptors:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


@dataclass(frozen=True)
class Limits:
    max_days: int = TRAINING_DAY_LIMIT  # a decision on a later game day ends the game
    max_actions: int = 327_680  # answered decisions per seat
    idle_seconds: float | None = None  # longest engine silence; None waits forever


@dataclass
class Request:
    player: int
    frames: tuple  # the seat's frames since its previous request, ending with the decision
    limit: str | None = None  # set when the decision is past a limit: call ``stop`` instead of ``step``
    decision: int | None = None  # the decision's frame (NativeEnv, whose requests carry no frames)
    done = False


@dataclass
class End:
    status: str  # completed, truncated or failed
    reason: str
    outcomes: tuple | None = None  # ("win", "loss") or ("loss", "win") by seat, for a completed game
    detail: str | None = None
    done = True


def _local_folder():
    """A new folder for the seats' local sockets, short enough for a socket path, or None where there is none."""
    if not hasattr(socket, "AF_UNIX") or os.environ.get("HEROES_LOCAL_SOCKETS", "1") == "0":
        return None
    try:
        folder = tempfile.mkdtemp(prefix="hrl-", dir="/tmp" if os.path.isdir("/tmp") else None)
    except OSError:
        return None
    if len(os.path.join(folder, "p0").encode()) >= 100:  # sockaddr_un's path holds 104 bytes on macOS
        shutil.rmtree(folder, ignore_errors=True)
        return None
    return folder


class _Side:
    def __init__(self, player, producer, listener, local=None):
        self.player, self.producer, self.listener = player, producer, listener
        self.local = local  # a listener on a local (Unix) socket too, or None: an engine that knows it connects there
        self.connection = None
        self.reader = None
        self.eof = False
        self.unread = 0  # first frame not yet handed to the policy
        self.issued = None  # sequence number of the last decision handed out
        self.actions = 0  # answers written to the engine


class Env:
    raw = False  # whether the producers read the engine's lines themselves (NativeEnv)

    def __init__(self, engine, limits=Limits()):
        self.engine, self.limits = engine, limits
        self.process = None
        self.sides = ()
        self.log = None
        self.local_folder = None  # the folder of the seats' local sockets while a game runs
        self.idle = None  # called with no argument whenever the environment is about to wait for the engine

    def _producer(self, episode, player):
        return Producer(str(episode), player, interface=self.engine.interface, army_information=self.engine.army_information)

    # -- the caller's interface ------------------------------------------------
    def reset(self, directory, episode, seed, game_map=None):
        """Start game ``episode`` with engine ``seed`` in the new folder ``directory``.

        ``game_map`` is another map file for this game alone; without it the game is the engine's own map.
        """
        self.close()
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True)
        self.stopping, self.exited = threading.Event(), threading.Event()
        self.request = self.result = self.failure = self.exit_code = self.limit = None
        sides = []
        for player in (0, 1):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            sides.append(_Side(player, self._producer(episode, player), listener))
        self.local_folder = _local_folder()
        if self.local_folder is not None:
            try:
                for side in sides:
                    side.local = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    side.local.bind(os.path.join(self.local_folder, f"p{side.player}"))
                    side.local.listen(1)
            except OSError:  # no local sockets here: the port alone
                for side in sides:
                    if side.local is not None:
                        side.local.close()
                        side.local = None
                shutil.rmtree(self.local_folder, ignore_errors=True)
                self.local_folder = None
        self.sides = tuple(sides)
        self.events = _Events(self.sides, raw=self.raw)
        (self.directory / "logs").mkdir()
        self.log = (self.directory / "logs" / "process.log").open("w")
        try:
            ports = [side.listener.getsockname()[1] for side in self.sides]
            extra = {} if game_map is None else {"game_map": game_map}
            if self.local_folder is not None:
                extra["sockets"] = [side.local.getsockname() for side in self.sides]
            self.process = self.engine.launch(self.directory, episode, seed, ports, self.log, **extra)
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
                _send(side.connection, (json.dumps(response, separators=(",", ":")) + "\n").encode())
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
        self._settle()
        if self.failure is None and self.exit_code is None and not self._ending():
            self.limit = reason
            return self._finish()
        return self._advance()

    def _settle(self):
        """Wait for the engine's report on every action it has been sent.

        A limit is met in one seat's decision while the other seat's last
        action, the end of its turn, may still be on its way through the
        engine: sent, its result not yet read. The two travel on two
        connections, so which arrived first was a matter of milliseconds, and
        a seat's history ended with or without the result of its last action
        accordingly: the same game gave two trajectories. The engine reports
        on every action it was sent, so the stop waits for those reports, for
        ``SETTLE_SECONDS`` at most; an engine that stays silent is stopped as
        it is, as before.
        """
        deadline = time.monotonic() + SETTLE_SECONDS
        while (self.failure is None and self.exit_code is None and not self._ending()
               and any(side.producer.history.action is not None for side in self.sides)):
            try:
                self._fold(self.events.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                break

    def frames(self, player):
        """Every frame the seat has received or produced so far."""
        return tuple(self.sides[player].producer.history._frames)

    def received(self, player):
        """The seat's frames so far as the growing list itself; frame ``n`` is at position ``n``. Read only."""
        return self.sides[player].producer.history._frames

    def forget(self, player, before):
        """Drop the observations and offers of the seat's frames before ``before``, to bound a long game's memory.

        ``before`` is at most the seat's latest decision. The frames stay in
        their places, without what they showed.
        """
        self.sides[player].producer.history.forget(before)

    @property
    def day(self):
        return max((side.producer.history.day for side in self.sides), default=1)

    @property
    def gold_mine_captures(self):
        return tuple(side.producer.gold_mine_captures for side in self.sides)

    def close(self):
        """Stop the engine and release sockets and files. The seats' frames stay readable."""
        if self.process is not None:
            engine_process.stop(self.process)
            self.process = None
        if self.sides:
            self.stopping.set()
        for side in self.sides:
            for sock in (side.connection, side.listener, side.local):
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    sock.close()
            side.connection = side.listener = side.local = None
            if side.reader is not None:
                side.reader.join(timeout=3)
        if getattr(self, "local_folder", None) is not None:
            shutil.rmtree(self.local_folder, ignore_errors=True)
            self.local_folder = None
        if self.sides and isinstance(getattr(self, "events", None), _Events):
            self.events.close()
        if self.log is not None and not self.log.closed:
            self.log.close()

    # -- engine events ---------------------------------------------------------
    @staticmethod
    def _read(side, events, stopping, exited):
        """Reader thread: accept the seat's connection and queue its messages."""
        try:
            # Poll while waiting, so an engine that dies at startup cannot block this thread. An engine told the
            # local socket's path connects there; an engine that does not know it connects to the port.
            listeners = [listener for listener in (side.listener, side.local) if listener is not None]
            while True:
                # Read before the attempt: a connection made before the engine exited is still accepted.
                over = stopping.is_set() or exited.is_set()
                ready, _, _ = select.select(listeners, [], [], 0.25)
                if ready:
                    connection, _ = ready[0].accept()
                    break
                if over:
                    return
            connection.setblocking(False)  # read by the thread that asks for events (_Events); answers are sent whole
            if connection.family == socket.AF_INET:
                # Answers are tiny single lines; never let Nagle's algorithm hold them.
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            side.connection = connection
            events.wake()
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
            if self.idle is not None:
                self.idle()  # the engine is working: the caller may use the time
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
            side.reader.join(timeout=max(0, deadline - time.monotonic()))  # the accepting threads
        if not stopped:
            # The engine ended on its own: its last messages, the terminals among
            # them, may still be unread behind the exit event.
            self.events.read_to_end(deadline)
            if not self.events.ended():
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
        # A seat that connected and was never asked anything lost before its first turn: red's hero can reach
        # blue's town on day 1 and win the game there. That is a game with a result, and the engine's account of
        # it is checked below like any other. A seat that never connected, or a game nobody acted in, is not.
        if any(not side.producer.ready for side in self.sides) or not any(side.actions for side in self.sides):
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


_LOOP_REQUEST, _LOOP_PENDING, _LOOP_ENDING, _LOOP_EVENTS, _LOOP_EOF, _LOOP_ERROR, _LOOP_SILENT, _LOOP_INGEST = range(8)


class NativeEnv(Env):
    """``Env`` whose seats' frames are kept by ``heroes.model.native``'s producer.

    The engine's lines go to the producer as they were read, and an answer is
    an action's value id, written by the producer: no message is decoded into
    Python objects. The game, its frames and its end are those ``Env`` gives;
    a request names its decision's frame (``decision``) and carries no frames.
    """
    raw = True

    @staticmethod
    def usable(engine):
        """Whether native seats can play this engine's games here (``HEROES_NATIVE_SEAT=0`` keeps the Python ones)."""
        from ..game import contract
        from ..model import native
        return (os.environ.get("HEROES_NATIVE_SEAT", "1") != "0" and native.module() is not None
                and engine.interface == contract.EXPANDED_ENVIRONMENT
                and engine.army_information == contract.OWN_ARMY_INFORMATION)

    def _producer(self, episode, player):
        from ..game import contract
        from ..model import native
        profile = contract.profile_for(self.engine.interface)
        return native.module().Producer(str(episode), player, profile.protocol, profile.environment, profile.history_schema,
                                        profile.action_schema, self.engine.army_information)

    def producer(self, player):
        return self.sides[player].producer

    def reset(self, directory, episode, seed, game_map=None):
        # The values and trees of the games before are dropped once nothing of theirs is alive: a worker plays
        # game after game, and every game interns its own observations.
        self.close()
        self.sides = ()
        from ..model import native
        native.module().reset_values()
        return super().reset(directory, episode, seed, game_map=game_map)

    def step(self, action):
        """Answer the current request with the action of value id ``action`` and wait for the next one."""
        if self.request is None:
            raise ValueError("There is no request to answer")
        side = self.sides[self.request.player]
        line = side.producer.answer(action)  # refuses an action the decision does not offer
        held = self.request.decision
        self.request = None
        self._drain()
        decision, answered, _ = side.producer.open()
        if self.failure is None and self.exit_code is None and decision == held and not answered and not self._ending():
            try:
                _send(side.connection, line)
            except OSError as error:
                self._fail("action_delivery_failed", f"{COLORS[side.player]}: {type(error).__name__}: {error}")
            else:
                side.producer.submit(action)
                side.actions += 1
        return self._advance()

    # -- the game loop in heroes.model.native ------------------------------------
    def play(self, seats, request):
        """Answer ``request`` and the requests after it with native ``seats`` until something is Python's.

        The loop (``native.cpp``, ``py_loop``) does what ``step`` and ``_advance`` do for engine lines and
        answers; it returns at a request past a limit, every second (for progress), and for anything else (the
        game ending, an event of the environment's threads, the end of a stream, an error, silence), which is
        then folded here as ``_advance`` folds it. Returns the next state, or None when the loop cannot take
        this request (an event other than an engine line is queued, or a seat is not connected): answer it with
        ``step`` then.
        """
        from ..model import native
        events = self.events
        if any(side.connection is None for side in self.sides) or self.failure is not None or self.exit_code is not None:
            return None
        with events.lock:
            if any(item[0] != "message" for item in events.items):
                return None
            queued = [bytearray(), bytearray()]
            for _, player, line in events.items:
                queued[player] += line + b"\n"
            events.items.clear()
        buffers = tuple(bytes(queued[side.player] + events.buffers[side.player]) for side in self.sides)
        for side in self.sides:
            events.buffers[side.player] = bytearray()
        self.request = None
        result = native.module().loop(
            tuple(side.producer for side in self.sides), tuple(seats), tuple(side.connection.fileno() for side in self.sides),
            events.wake_read, buffers, tuple(side.issued for side in self.sides), tuple(side.actions for side in self.sides),
            self.limits.max_days, self.limits.max_actions, self.limits.idle_seconds or 0.0, request.player, request.decision)
        code, player, decision, value, rest, issued, actions, text, error, pending = result
        for side in self.sides:
            side.issued, side.actions = issued[side.player], actions[side.player]
            # complete lines go back as the reader queued them; a partial line stays in its buffer
            data = rest[side.player]
            end = data.rfind(b"\n") + 1
            with events.lock:
                for line in data[:end].split(b"\n")[:-1]:
                    events.items.append(("message", side.player, line))
            events.buffers[side.player] = bytearray(data[end:])
        if code == _LOOP_REQUEST:
            limit = "game_day_limit" if value == 1 else "agent_action_limit" if value == 2 else None
            self.request = Request(player, (), limit, decision)
            return self.request
        side = self.sides[value] if code in (_LOOP_EOF, _LOOP_ERROR, _LOOP_INGEST) else None
        if code == _LOOP_EOF:
            if events.buffers[side.player]:
                events._end(side, ("error", side.player, "ValueError: Stream ended inside a message"))
            else:
                events._end(side, ("eof", side.player, None))
        elif code == _LOOP_ERROR:
            events._end(side, ("error", side.player, text if text.startswith("ValueError") else f"OSError: {text}"))
        elif code == _LOOP_INGEST:
            if isinstance(error, NativeFailure):
                self._fail("engine_error", f"{COLORS[side.player]}: {error}")
            elif isinstance(error, (KeyError, TypeError, ValueError, AttributeError, IndexError)):
                self._fail("protocol_error", f"{COLORS[side.player]}: {type(error).__name__}: {error}")
            else:
                raise error
        elif code == _LOOP_SILENT:
            self._fail("engine_silent", f"No engine event for {self.limits.idle_seconds} s")
        if pending is not None:  # the rest of step: fold what is there, then send or drop the answer
            held, line, action = pending
            asker = self.sides[player]
            self._drain()
            decision, answered, _ = asker.producer.open()
            if self.failure is None and self.exit_code is None and decision == held and not answered and not self._ending():
                try:
                    _send(asker.connection, line)
                except OSError as failure:
                    self._fail("action_delivery_failed", f"{COLORS[asker.player]}: {type(failure).__name__}: {failure}")
                else:
                    asker.producer.submit(action)
                    asker.actions += 1
        return self._advance()

    def _settle(self):
        deadline = time.monotonic() + SETTLE_SECONDS
        while (self.failure is None and self.exit_code is None and not self._ending()
               and any(side.producer.open()[1] for side in self.sides)):
            try:
                self._fold(self.events.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                break

    def frames(self, player):
        raise ValueError("A native producer keeps its frames; see heroes.model.native")

    received = frames

    def forget(self, player, before):
        pass  # a native frame holds value ids, not objects

    @property
    def day(self):
        return max((side.producer.day for side in self.sides), default=1)

    def _next_request(self):
        for side in self.sides:
            decision, answered, day = side.producer.open()
            if decision < 0 or answered or side.issued == decision:
                continue
            side.issued = decision
            limit = ("game_day_limit" if day > self.limits.max_days else
                     "agent_action_limit" if side.actions >= self.limits.max_actions else None)
            self.request = Request(side.player, (), limit, decision)
            return self.request
        return None
