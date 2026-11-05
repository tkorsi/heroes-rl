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
import textwrap
import time


SCHEMA = "heroes-game-progress-v1"
PHASES = {"starting", "collecting", "finalizing", "storing", "awaiting verification",
          "preparing PPO", "captured", "verifying", "verified", "failed"}
PHASE_LABELS = {
    "starting": "Starting", "collecting": "Playing", "finalizing": "Finishing game",
    "storing": "Saving game", "awaiting verification": "Waiting for replay checks",
    "preparing PPO": "Preparing PPO", "captured": "Captured for training",
    "verifying": "Verifying game", "verified": "Verified", "failed": "Failed",
}


def terminal_color_enabled(stream):
    """Use color only for a capable terminal, including explicit opt-out support."""
    try:
        return ("NO_COLOR" not in os.environ and os.environ.get("TERM", "").lower() != "dumb"
                and bool(getattr(stream, "isatty", lambda: False)()))
    except (OSError, ValueError):
        return False


def color_text(text, code, enabled):
    return f"\x1b[{code}m{text}\x1b[0m" if enabled else text


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
                if value["phase"] not in ("verified", "captured", "failed"):
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


def terminal_lines(value, *, width=80, color=False):
    """Render a readable terminal card; colors never carry the only meaning."""
    width = max(1, width)
    observed = [day for day in value.get("days", ()) if day is not None]
    day = str(max(observed)) if observed else "?"
    if value.get("max_days") is not None:
        day += f"/{value['max_days']}"
    phase = PHASE_LABELS[value["phase"]]
    phase_code = ("1;32" if value["phase"] == "verified" else "1;31" if value["phase"] == "failed"
                  else "1;36" if value["phase"] in ("starting", "collecting") else "1;33")
    rows = [(f"Game {value['episode']}   Day {day}   |   {phase}", phase_code)]
    ordinary = value["ordinary"]
    digits = len(f"{ordinary:,}")
    bar_width = min(24, max(8, width - 32 - 2 * digits))
    for name, count, code in zip(("Red", "Blue"), value["choices"], ("1;91", "1;94")):
        used, reserve = min(count, ordinary), max(0, count - ordinary)
        filled = bar_width * used // ordinary
        bar = "#" * filled + "-" * (bar_width - filled)
        counts = f"{used:>{digits},} / {ordinary:,}"
        if width < 38:
            rows.extend(((f"  {name:<5} [{bar}]", code), (f"  {counts} choices", code)))
        else:
            suffix = " choices" if width >= 55 else ""
            rows.append((f"  {name:<5} [{bar}]  {counts}{suffix}", code))
        indent = "        " if width >= 55 else "  "
        rows.append((f"{indent}Dialog reserve: {reserve:,} / {value['reserved']:,}", "33" if reserve else "2"))
    # Wrap before applying ANSI so terminal cell widths never include escapes.
    # Leave the last terminal column unused to avoid implicit line wrapping.
    return [color_text(line, code, color) for text, code in rows
            for line in textwrap.wrap(text, width=width, replace_whitespace=False,
                                      drop_whitespace=True, break_on_hyphens=False) or [""]]


class ProgressDisplay:
    """Read tiny sidecars in the terminal-owning supervisor, not worker stdout."""

    def __init__(self, directory, *, stream=None, clock=time.monotonic):
        self.directory = Path(directory)
        self.stream = sys.stderr if stream is None else stream
        self.clock, self.last_output = clock, float("-inf")
        self.previous, self.lines = {}, 0
        self.disabled = False
        try:
            self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())
            self.columns = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)
            self.tty = self.terminal and self.columns >= 80 and os.environ.get("TERM", "").lower() != "dumb"
            self.color = self.tty and terminal_color_enabled(self.stream)
        except (OSError, ValueError):
            self.terminal, self.tty, self.color, self.columns, self.disabled = False, False, False, 80, True

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
                    self.stream.write(color_text("ACTION BUDGET", "1", self.color) + "\n"
                                      "Bars show choices used, not time remaining.\n"
                                      "Game results follow replay checks.\n\n")
                lines = [line for value in shown for line in terminal_lines(
                    value, width=self.columns - 1, color=self.color)]
                for line in lines:
                    self.stream.write(line + "\n")
                self.lines = len(lines)
            else:
                for value in changed.values():
                    if self.terminal:
                        self.stream.write("\n".join(terminal_lines(value, width=max(1, self.columns - 1))) + "\n")
                    else:
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
                self.stream.write(color_text("Run failed or interrupted; see shutdown.json for cleanup status.",
                                             "1;31", self.color) + "\n")
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


def batch_terminal_lines(slots, *, width=80, color=False):
    """Five columns, two rows; each card keeps result separate from preparation."""
    cell = (width - 4) // 5
    inner = cell - 2
    cards = []
    for slot in slots:
        value, attempt = slot['value'], slot['attempt']
        state = attempt.get('status', 'not_started')
        phase = value.get('phase', 'starting')
        done = state in ('captured', 'passed')
        failed = state in ('failed', 'interrupted') or phase == 'failed'
        outcomes = attempt.get('outcomes', {})
        result = ('RED WIN' if outcomes.get('red') == 'win' and outcomes.get('blue') == 'loss'
                  else 'BLUE WIN' if outcomes.get('blue') == 'win' and outcomes.get('red') == 'loss'
                  else 'DAY LIMIT' if attempt.get('termination_reason') == 'game_day_limit'
                  else 'CUTOFF' if attempt.get('episode_status') == 'truncated' else None)
        active = state == 'running' and not failed
        label = result or ('FAILED' if failed else 'READY' if done else
                           {'collecting': 'PLAYING', 'starting': 'STARTING', 'finalizing': 'FINISHING',
                            'storing': 'SAVING', 'preparing PPO': 'PREPARING PPO',
                            'verifying': 'VERIFYING', 'awaiting verification': 'CHECKS QUEUED',
                            'captured': 'CAPTURED', 'verified': 'VERIFIED'}.get(phase, phase.upper())
                           if active else 'WAITING')
        code = '1;31' if failed else '1;36' if active else '33' if result in ('DAY LIMIT', 'CUTOFF') else '32' if done else '2'
        days = [n for n in value.get('days', ()) if n is not None]
        day = f"Day {max(days) if days else '?'}/{value.get('max_days') or '?'}"
        counts = value.get('choices', attempt.get('sampled_by_player', [0, 0]))
        total = value.get('ordinary', 1) + value.get('reserved', 0)
        used = min(inner, inner * sum(counts) // max(1, 2 * total))
        footer = ('ERROR ' + attempt.get('failure_stage', 'run') if failed else
                  'READY FOR PPO' if state == 'captured' else 'CHECKS PASSED' if state == 'passed' else
                  '#' * used + '-' * (inner - used) if active else '')
        def fit(text): return text[:inner].ljust(inner)
        title = f"{'>' if active else ' '} Game {slot['episode']:02d} "
        rows = [color_text('+' + title[:inner].ljust(inner, '-') + '+', code, color)]
        for text, tint in ((label, code), (day, '2'),
                           (f"R {counts[0]:,}", '91'), (f"B {counts[1]:,}", '94'), (footer, code)):
            rows.append(color_text('|', code, color) + color_text(fit(text), tint, color) + color_text('|', code, color))
        rows.append(color_text('+' + '-' * inner + '+', code, color))
        cards.append(rows)
    return [' '.join(cards[col][row] for col in range(start, start + 5))
            for start in (0, 5) for row in range(7)]


class BatchProgressDisplay(ProgressDisplay):
    """A stationary ten-slot view of existing disposable progress and reports."""

    def __init__(self, directory, *, launch, stream=None, clock=time.monotonic):
        super().__init__(directory, stream=stream, clock=clock)
        self.launch, self.report = launch, {}
        self.started, self.opened = self.clock(), False
        self.saved, self.phase_name, self.phase_start = {}, None, 0
        self.dashboard = self.terminal and os.environ.get("TERM", "").lower() != "dumb"

    def update_report(self, report):
        self.report = report

    def _phase(self):
        batches, evaluations = self.report.get('batches', []), self.report.get('evaluations', [])
        batch = batches[-1] if batches else {}
        evaluation = evaluations[-1] if evaluations else {}
        evaluating = bool(evaluation and (not batch or batch.get('status') == 'checkpoint_verified'
                          and evaluation.get('checkpoint_before') == batch.get('checkpoint_after')))
        return batch, evaluation if evaluating else None

    def poll(self, *, final=False):
        if not self.dashboard:
            return super().poll(final=final)
        if self.disabled:
            return
        try:
            size = shutil.get_terminal_size(fallback=(80, 24))
            width, height = max(1, size.columns - 1), max(1, size.lines - 1)
            batch, evaluation = self._phase()
            phase = evaluation or batch
            name = phase.get('name')
            if name != self.phase_name:
                self.phase_name = name
                try:
                    self.phase_start = Path(phase['plan']['path']).stat().st_mtime
                except (KeyError, OSError):
                    self.phase_start = float('inf')
            current = {}
            for path in self.directory.glob('game-*.json'):
                try:
                    if path.stat().st_mtime >= self.phase_start:
                        value = _read(path)
                        current[value['episode']] = value
                except (OSError, ValueError):
                    continue
            if not evaluation:
                for attempt in batch.get('attempts', []):
                    episode = attempt['episode']
                    if attempt['status'] != 'not_started' and episode in current:
                        self.saved[(batch.get('name'), episode)] = current[episode]
            attempts = {a['episode']: a for a in batch.get('attempts', [])}
            cfg = self.launch.get('configuration', {})
            defaults = dict(max_days=cfg.get('max_days'), ordinary=cfg.get('ordinary_choice_limit', 1),
                            reserved=cfg.get('reserved_dialog_choices', 0))
            slots = [dict(episode=n, attempt=attempts.get(n, {}),
                          value={**defaults, **self.saved.get((batch.get('name'), n), {})}) for n in range(1, 11)]
            elapsed = int(self.clock() - self.started)
            checkpoint = self.report.get('latest_checkpoint', self.launch.get('parent_pair', {}).get('progress', {}).get('updates', '?'))
            completed = sum(a.get('status') in ('captured', 'passed') for a in attempts.values())
            rows = [color_text('HEROES  /  TEN-GAME TRAINING'[:width], '1', self.color),
                    f"Batch {len(self.report.get('batches', [])) or 1}/{self.launch.get('batches', 1)} | Ready {completed}/10 | Checkpoint {checkpoint} | {elapsed//3600:02d}:{elapsed//60%60:02d}:{elapsed%60:02d}"]
            if width >= 74 and height >= 21:
                rows += batch_terminal_lines(slots, width=width, color=self.color)
            else:
                rows += ['Widen to 80 columns for the 5 x 2 grid.']
                for slot in slots:
                    a, v = slot['attempt'], slot['value']
                    days = [d for d in v.get('days', ()) if d is not None]
                    rows.append(f"Game {slot['episode']:02d} | {a.get('status', 'waiting')} | Day {max(days) if days else '?'} | R/B "
                                + '/'.join(str(n) for n in v.get('choices', a.get('sampled_by_player', [0, 0]))))
            counts = batch.get('counts', {})
            red = counts.get('engine_red_wins', counts.get('red_wins', 0))
            blue = counts.get('engine_blue_wins', counts.get('blue_wins', 0))
            rows.append(f"Engine results: Red {red} | Blue {blue} | Cutoffs {counts.get('game_cutoffs', 0)} | Errors {counts.get('failures', 0)}")
            if evaluation:
                v = current.get(1, {})
                days = [d for d in v.get('days', ()) if d is not None]
                check = ('PASSED' if evaluation.get('status') == 'evaluation_verified' else PHASE_LABELS.get(v.get('phase'), 'Starting'))
                rows.append(f"PPO updates: {self.report.get('optimizer_updates', 0)} | Validation: {check} | Day {max(days) if days else '?'}")
            else:
                status = 'UPDATING' if batch.get('status') == 'updating' else 'SAVED' if batch.get('status') == 'checkpoint_verified' else 'waiting for 10 ready games'
                rows.append(f"PPO: {status} | Updates {self.report.get('optimizer_updates', 0)} | Validation: after saved update")
            rows.append('> active | R/B = choices used | Bars = action budget, not time | Wins above are unverified')
            if self.report.get('error'):
                rows[-1] = 'STOPPED: ' + self.report['error']
            # Clip before styling is applied to cards; other rows have no ANSI
            # except the short heading. Never write into the last cell/row.
            rows = [row if '\x1b' in row else row[:width] for row in rows][:height]
            if not self.opened:
                self.stream.write('\x1b[?1049h\x1b[?25l\x1b[2J')
                self.opened = True
            self.stream.write('\x1b[H' + '\n'.join(row + '\x1b[K' for row in rows) + '\x1b[J')
            self.stream.flush()
        except Exception:
            # Display data is never a training gate; malformed UI/report fields
            # must not propagate into the process-owning supervisor.
            self.disabled = True
            self._close_screen()

    def _close_screen(self):
        if self.opened:
            try:
                self.stream.write('\x1b[?25h\x1b[?1049l')
                self.stream.flush()
            except (OSError, ValueError):
                pass
            self.opened = False

    def finish(self, *, failed=False):
        if not self.dashboard:
            return super().finish(failed=failed)
        try:
            finish_unfinished(self.directory)
            self.poll(final=True)
        finally:
            self._close_screen()
