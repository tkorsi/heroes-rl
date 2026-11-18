"""The training loop: play a batch of games in parallel, update once, save, repeat.

Every game of a batch is played by the same weights, the ones in the newest
checkpoint. Actor processes play the games and hand back the learner seat's
trajectory; this process runs one PPO update over the whole batch and writes
the next checkpoint, which the actors then load for the next batch.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import multiprocessing
import os
import pickle
import queue
import secrets
import shutil
import signal
import sys
import time
import traceback

import torch

from . import checkpoint
from .actor import Game, Job, compile_recipe, play
from .engine.env import COLORS, Env


def restore(config, *, fresh=False):
    """The newest training state: this pipeline's own, else the legacy one, else a new policy."""
    path = checkpoint.latest(config.state)
    if path is not None:
        return checkpoint.load(path, learning_rate=config.ppo.learning_rate)
    if fresh:
        state = checkpoint.fresh(learning_rate=config.ppo.learning_rate)
    else:
        pair = checkpoint.find_legacy(config.legacy)
        if pair is None:
            raise SystemExit(f"No checkpoint in {config.state} and no earlier one to continue in {config.legacy}.\n"
                             "Pass --fresh to start from a randomly initialized policy.")
        state = checkpoint.import_legacy(*pair, learning_rate=config.ppo.learning_rate)
    checkpoint.save(config.state, state)
    return state


def update_function(config):
    """The PPO update of the configured backend, checked before any game is played."""
    if config.backend == "torch":
        from .learner.ppo import update
        return update
    if config.backend != "mlx":
        raise SystemExit(f"Unknown backend {config.backend!r}; use mlx or torch.")
    try:
        import mlx.core as mx
        available = mx.metal.is_available()
    except Exception:
        available = False
    if not available:
        raise SystemExit("The MLX backend needs an Apple GPU and the mlx package. "
                         "Use --backend torch for the slow reference update.")
    from .learner.mlx.update import update
    return update


# -- actor processes -----------------------------------------------------------

def _play_job(env, model, job, config, report):
    directory = config.state / "games" / f"game-{job.number:07d}"
    shutil.rmtree(directory, ignore_errors=True)
    started = time.monotonic()
    try:
        game = play(env, model, job, directory, guard=config.guard, on_progress=report)
        if game.trajectory is not None and config.backend == "mlx":
            compiling = time.monotonic()
            compile_recipe(model, game.trajectory)
            game.compile_seconds = time.monotonic() - compiling
    except Exception:
        game = Game(job, "failed", "actor_error", None, traceback.format_exc(limit=8), 0, (0, 0),
                    time.monotonic() - started, None)
    finally:
        env.close()
    if game.status == "failed":
        try:
            tail = (directory / "logs" / "process.log").read_text(errors="replace")[-1500:]
            game.detail = f"{game.detail or ''}\n--- end of the engine log ---\n{tail}".strip()
        except OSError:
            pass
    if not config.keep_game_files:
        shutil.rmtree(directory, ignore_errors=True)
    return game


def worker(index, config, inbox, results):
    """One actor process. It plays the games it is handed, one at a time."""
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the trainer handles Ctrl-C and stops its actors
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    parent = os.getppid()
    env = Env(config.engine(), config.limits)
    model = loaded = None
    try:
        while True:
            try:
                item = inbox.get(timeout=5)
            except queue.Empty:
                if os.getppid() != parent:
                    return  # the trainer is gone
                continue
            if item is None:
                return
            path, job = item
            if path != loaded:
                model, loaded = checkpoint.load_model(path), path
            last = [time.monotonic()]

            def report(day, choices, job=job):
                now = time.monotonic()
                if now - last[0] >= 1:
                    last[0] = now
                    if os.getppid() != parent:
                        raise SystemExit(0)
                    results.put(("progress", index, (job.number, day, choices)))

            game = _play_job(env, model, job, config, report)
            # Plain pickle keeps tensors inside the message instead of in shared memory files.
            results.put(("game", index, pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)))
    finally:
        env.close()


class Pool:
    """The actor processes and which game each of them is playing."""

    def __init__(self, config):
        self.config = config
        self.context = multiprocessing.get_context("spawn")
        self.results = self.context.Queue()
        self.inboxes = [self.context.Queue() for _ in range(config.workers)]
        self.processes = [self._start(index) for index in range(config.workers)]
        self.idle = list(range(config.workers))
        self.busy = {}  # worker index -> its job

    def _start(self, index):
        process = self.context.Process(target=worker, args=(index, self.config, self.inboxes[index], self.results),
                                       name=f"actor-{index}", daemon=True)
        process.start()
        return process

    def submit(self, path, job):
        index = self.idle.pop()
        self.busy[index] = job
        self.inboxes[index].put((str(path), job))

    def next(self):
        """The next message from an actor, or None after a quiet second.

        Returns ``("progress", index, (number, day, choices))`` or
        ``("game", index, Game)``. An actor that died is replaced, and the game
        it was playing comes back as failed.
        """
        try:
            kind, index, payload = self.results.get(timeout=1)
        except queue.Empty:
            for index, process in enumerate(self.processes):
                if not process.is_alive():
                    self.processes[index] = self._start(index)
                    self.idle.append(index)
                    job = self.busy.pop(index, None)
                    if job is not None:
                        return "game", index, Game(job, "failed", "actor_died", None,
                                                   f"The actor process exited with status {process.exitcode}",
                                                   0, (0, 0), 0.0, None)
            return None
        if kind == "game":
            payload = pickle.loads(payload)
            self.busy.pop(index, None)
            self.idle.append(index)
        return kind, index, payload

    def shutdown(self):
        for inbox in self.inboxes:
            inbox.put(None)
        deadline = time.monotonic() + 5
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()  # its handler stops the game it is playing
        for process in self.processes:
            process.join(timeout=15)


# -- what the person watching sees ---------------------------------------------

def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Display:
    """One permanent line per finished game and update; on a terminal also a live status line."""

    def __init__(self, out):
        self.out = out
        self.live = out.isatty()
        self.playing = {}  # worker index -> (game number, day, choices)

    def line(self, text):
        self.out.write(("\r\x1b[2K" if self.live else "") + text + "\n")
        self.out.flush()

    def status(self, text):
        if self.live:
            self.out.write("\r\x1b[2K" + text[:shutil.get_terminal_size().columns - 1])
            self.out.flush()

    def collecting(self, update, done, target, tally, started):
        days = " ".join(str(day) for _, day, _ in sorted(self.playing.values()))
        self.status(f"update {update}  games {done}/{target}  red {tally['red']}  blue {tally['blue']}  "
                    f"stopped {tally['truncated']}  failed {tally['failed']}  "
                    f"{_clock(time.monotonic() - started)}  days in play: {days}")

    def game(self, game, samples):
        if game.status == "completed":
            result = f"{COLORS[game.outcomes.index('win')]} wins"
        elif game.status == "truncated":
            result = "stopped at the " + ("day" if game.reason == "game_day_limit" else "action") + " limit"
        else:
            result = f"FAILED ({game.reason})"
        self.line(f"game {game.job.number}  {result}  day {game.day}  learner {COLORS[game.job.learner]}  "
                  f"{samples} samples  {_clock(game.seconds)}")


class Log:
    """Append-only ``log.jsonl``: one record per game and per update."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a")

    def write(self, event, **fields):
        record = {"event": event, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()


# -- the loop --------------------------------------------------------------------

def collect(pool, config, path, target, numbers, update, display, log):
    """Play until ``target`` games have given a trajectory. All use the weights at ``path``."""
    games, tally, started = [], {"red": 0, "blue": 0, "truncated": 0, "failed": 0}, time.monotonic()
    while len(games) < target:
        while pool.idle and len(games) + len(pool.busy) < target:
            number = next(numbers)
            pool.submit(path, Job(number, secrets.randbelow(2**31 - 1) + 1, number % 2,
                                  (secrets.randbits(62), secrets.randbits(62))))
        message = pool.next()
        if message is not None and message[0] == "progress":
            display.playing[message[1]] = message[2]
        elif message is not None:
            _, index, game = message
            display.playing.pop(index, None)
            samples = len(game.trajectory.decisions) if game.trajectory is not None else 0
            log.write("game", number=game.job.number, engine_seed=game.job.engine_seed,
                      policy_seeds=list(game.job.policy_seeds), learner=COLORS[game.job.learner],
                      status=game.status, reason=game.reason, outcomes=game.outcomes, day=game.day,
                      choices=list(game.choices), samples=samples, seconds=round(game.seconds, 1),
                      compile_seconds=round(game.compile_seconds, 1), update=update, detail=game.detail)
            display.game(game, samples)
            if game.trajectory is None:
                tally["failed"] += 1
                if tally["failed"] > target or (not games and tally["failed"] >= max(3, config.workers)):
                    raise RuntimeError(f"{tally['failed']} games failed, the last one with {game.reason}:\n{game.detail}\n"
                                       f"Every failed game is in {config.state / 'log.jsonl'}.")
            else:
                tally["truncated" if game.status == "truncated" else COLORS[game.outcomes.index("win")]] += 1
                games.append(game)
        display.collecting(update, len(games), target, tally, started)
    return games, tally


def train(config, games, *, fresh=False, out=sys.stderr):
    """Play ``games`` games in batches of ``config.games_per_update`` and update after each batch."""
    config.engine().check()
    update_policy = update_function(config)
    state = restore(config, fresh=fresh)
    display, log = Display(out), Log(config.state / "log.jsonl")
    display.line(f"Continuing from update {state.updates} ({state.progress['games']} games, "
                 f"{state.progress['decisions']} decisions) in {config.state}")
    log.write("start", games=games, backend=config.backend, workers=config.workers, progress=state.progress,
              ppo=config.ppo.to_dict(), limits=asdict(config.limits), guard=config.guard,
              record_replay=config.record_replay)
    numbers = itertools.count(state.progress["games"] + 1)
    pool = Pool(config)
    try:
        while games > 0:
            target = min(config.games_per_update, games)
            update, started = state.updates + 1, time.monotonic()
            batch, tally = collect(pool, config, checkpoint.latest(config.state), target, numbers, update, display, log)
            collected = time.monotonic()

            def progress(step, total, update=update, collected=collected):
                display.status(f"update {update}  PPO step {step}/{total}  {_clock(time.monotonic() - collected)}")

            report = update_policy(state.model, state.optimizer, [game.trajectory for game in batch],
                                   config=config.ppo, generator=state.generator, progress=progress)
            updated = time.monotonic()
            state.progress = {"updates": update, "games": state.progress["games"] + len(batch),
                              "decisions": state.progress["decisions"] + report.samples,
                              "optimizer_steps": state.progress["optimizer_steps"] + report.optimizer_steps}
            saved = checkpoint.save(config.state, state)
            del batch
            log.write("update", update=update, games=target, **tally, samples=report.samples,
                      optimizer_steps=report.optimizer_steps, policy_loss=report.policy_loss,
                      value_loss=report.value_loss, approximate_kl=report.approximate_kl,
                      clip_fraction=report.clip_fraction, collect_seconds=round(collected - started, 1),
                      update_seconds=round(updated - collected, 1), update_timings=report.timings_seconds,
                      checkpoint=saved.name, progress=state.progress)
            display.line(f"update {update} saved  {report.samples} samples  {report.optimizer_steps} steps  "
                         f"policy loss {report.policy_loss:+.4f}  value loss {report.value_loss:.4f}  "
                         f"KL {report.approximate_kl:.5f}  games {_clock(collected - started)}  "
                         f"PPO {_clock(updated - collected)}")
            games -= target
    except KeyboardInterrupt:
        display.line(f"Stopped. The next run continues from update {state.updates}; "
                     "games of the unfinished batch are discarded.")
    finally:
        pool.shutdown()
    return state
