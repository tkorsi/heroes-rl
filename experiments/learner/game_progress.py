"""Best-effort display of submitted policy choices, separate from evidence.

These small, replaceable snapshots are never used for admission or recovery.
They contain no model, history, resource profiling or hidden game information.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SCHEMA = "heroes-game-progress-v1"
PHASES = {"starting", "collecting", "finalizing", "storing", "awaiting verification", "verifying", "verified", "failed"}


def progress_path(output, episode):
    return Path(output) / "progress" / f"game-{episode:06d}.json"


def _valid(value):
    return (type(value) is dict
            and {"schema", "episode", "choices", "ordinary", "reserved", "phase"} <= value.keys()
            and value.keys() <= {"schema", "episode", "choices", "ordinary", "reserved", "phase", "days", "max_days"}
            and value["schema"] == SCHEMA
            and type(value["episode"]) is int and value["episode"] >= 0
            and type(value["ordinary"]) is int and value["ordinary"] > 0
            and type(value["reserved"]) is int and value["reserved"] >= 0
            and type(value["choices"]) is list and len(value["choices"]) == 2
            and all(type(n) is int and 0 <= n <= value["ordinary"] + value["reserved"] for n in value["choices"])
            and type(value["phase"]) is str and value["phase"] in PHASES
            and ("days" not in value or type(value["days"]) is list and len(value["days"]) == 2
                 and all(day is None or type(day) is int and day >= 1 for day in value["days"]))
            and (value.get("max_days") is None
                 or type(value["max_days"]) is int and value["max_days"] >= 1))


def _read(path):
    with Path(path).open("rb") as source:
        payload = source.read(4097)
    if len(payload) > 4096:
        raise ValueError("Oversized progress snapshot")
    try:
        value = json.loads(payload)
    except RecursionError as error:
        raise ValueError("Progress snapshot nesting is excessive") from error
    if not _valid(value):
        raise ValueError("Invalid progress snapshot")
    return value


def _write(path, value):
    # No fsync: this is a disposable display snapshot, never scientific evidence.
    path = Path(path)
    temporary = path.with_name(f".{path.name}-{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GameProgress:
    """Collector callback, throttled to one small write per second."""

    def __init__(self, path, *, episode, ordinary, reserved, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.value = dict(schema=SCHEMA, episode=episode, choices=[0, 0],
                          ordinary=ordinary, reserved=reserved, phase="starting", days=[None, None], max_days=None)
        if not _valid(self.value):
            raise ValueError("Invalid progress configuration")
        self.last_write = float("-inf")
        self.warned = False

    def observe_day(self, player, day, *, max_days=None):
        """Remember the delivered calendar without reading or changing policy state.

        Both seats share one game calendar; their last delivered days may differ
        while the other seat is acting. No extra write or history scan is needed.
        """
        if type(max_days) is int and max_days >= 1:
            self.value["max_days"] = max_days
        if type(player) is int and player in (0, 1) and type(day) is int and day >= 1:
            previous = self.value["days"][player]
            self.value["days"][player] = max(day, previous or day)

    def __call__(self, counts, phase="collecting"):
        now = self.clock()
        changed_phase = phase != self.value["phase"]
        self.value = {**self.value, "choices": list(counts), "phase": phase}
        if not _valid(self.value):
            raise ValueError("Invalid progress update")
        if not changed_phase and now - self.last_write < 1:
            return
        try:
            _write(self.path, self.value)
        except OSError as error:
            if not self.warned:
                try:
                    print(f"Game progress display unavailable: {error}", file=sys.stderr, flush=True)
                except (OSError, ValueError):
                    pass
                self.warned = True
        self.last_write = now

    def phase(self, value):
        self(self.value["choices"], value)


def set_phase(path, phase):
    """Update display after worker handoff; missing progress never blocks work."""
    if phase not in PHASES:
        raise ValueError("Invalid progress phase")
    try:
        value = _read(path)
        _write(path, {**value, "phase": phase})
    except (OSError, ValueError):
        pass


def finish_unfinished(directory):
    """After owned-process cleanup, terminate stale UI without altering proofs."""
    unfinished = False
    try:
        for path in Path(directory).glob("game-*.json"):
            try:
                value = _read(path)
                if value["phase"] not in ("verified", "failed"):
                    unfinished = True
                    _write(path, {**value, "phase": "failed"})
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return unfinished


def format_progress(value):
    """A full bar means ordinary allowance spent, not that a game is verified."""
    parts = []
    for color, count in zip(("Red", "Blue"), value["choices"]):
        used = min(count, value["ordinary"])
        filled = 16 * used // value["ordinary"]
        bar = "#" * filled + "-" * (16 - filled)
        reserve = max(0, count - value["ordinary"])
        parts.append(f"{color} [{bar}] {used:,}/{value['ordinary']:,} +{reserve}/{value['reserved']} reserved")
    observed = [day for day in value.get("days", ()) if day is not None]
    day = str(max(observed)) if observed else "?"
    if value.get("max_days") is not None:
        day += f"/{value['max_days']}"
    return f"Game slot {value['episode']} | Day {day} | " + " | ".join(parts) + f" | {value['phase']}"


def terminal_lines(value):
    text = format_progress(value).split(" | ")
    return [text[0] + " | " + text[1] + " | " + text[-1], "  " + text[2], "  " + text[3]]


class ProgressDisplay:
    """Read tiny sidecars in the terminal-owning supervisor, not worker stdout."""

    def __init__(self, directory, *, stream=None, clock=time.monotonic):
        self.directory = Path(directory)
        self.stream = sys.stderr if stream is None else stream
        self.clock, self.last_output = clock, float("-inf")
        self.previous, self.lines = {}, 0
        self.disabled = False
        try:
            self.tty = (bool(getattr(self.stream, "isatty", lambda: False)())
                        and shutil.get_terminal_size(fallback=(80, 24)).columns >= 80)
        except (OSError, ValueError):
            self.tty, self.disabled = False, True

    def poll(self, *, final=False):
        if self.disabled:
            return
        try:
            values = {}
            for path in sorted(self.directory.glob("game-*.json")):
                try:
                    value = _read(path)
                    values[value["episode"]] = value
                except (OSError, ValueError):
                    continue
            changed = {key: value for key, value in values.items() if value != self.previous.get(key)}
            if not changed:
                return
            now = self.clock()
            phase_change = any(value["phase"] != self.previous.get(key, {}).get("phase") for key, value in changed.items())
            if not self.tty and not final and not phase_change and now - self.last_output < 10:
                return
            if self.tty:
                # Keep parallel games visible while replacing finished batches.
                active = [v for v in values.values() if v["phase"] not in ("verified", "failed")]
                shown = active or list(changed.values())
                if self.lines:
                    self.stream.write(f"\x1b[{self.lines}F\x1b[J")
                else:
                    self.stream.write("Choice budget used (not completion time); slots restart each batch.\n"
                                      "Replay verification follows gameplay.\n")
                lines = [line for value in shown for line in terminal_lines(value)]
                for line in lines:
                    self.stream.write(line + "\n")
                self.lines = len(lines)
            else:
                for value in changed.values():
                    self.stream.write(format_progress(value) + "\n")
            self.stream.flush()
            self.previous, self.last_output = values, now
        except (OSError, ValueError):
            self.disabled = True

    def finish(self, *, failed=False):
        unfinished = finish_unfinished(self.directory)
        self.poll(final=True)
        if (failed or unfinished) and not self.disabled:
            try:
                self.stream.write("Run failed or interrupted; see shutdown.json for cleanup status.\n")
                self.stream.flush()
            except (OSError, ValueError):
                self.disabled = True

    def wait(self, process, *, timeout):
        """Preserve the caller's deadline and propagate subprocess failures."""
        deadline = self.clock() + timeout
        try:
            while True:
                self.poll()
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                try:
                    return process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    if self.clock() >= deadline:
                        raise
        finally:
            self.poll(final=True)
