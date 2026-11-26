"""The training loop: play a batch of games in parallel, update once, save, repeat.

Every game of a batch is played by the same weights, the ones in the newest
checkpoint. Actor processes play the games and hand back the learner seat's
trajectory; this process runs one PPO update over the whole batch and writes
the next checkpoint, which the actors then load for the next batch.

With ``play_during_update`` the actors do not wait for the update: they start
on the next batch at once, with the weights from before it. Those games are one
update behind; the rest of that batch is played with the new weights. Only the
reference backend takes such a batch: the MLX update refuses games that the
weights it starts from did not play.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
import pickle
import re
import secrets
import shutil
import signal
import sys
import threading
import time
import traceback

import torch

from . import checkpoint
from .actor import Game, Job, compact_recipe, compile_recipe, play
from .engine.env import COLORS, Env, NativeEnv
from .learner.ppo import WINDOW_REPEAT_PENALTY
from .model.policy import weights_sha256


def restore(config, *, fresh=False, save=True):
    """Load the existing state or initialize it from the selected source.

    A state that does not come from this folder is saved as its first checkpoint.
    """
    path = checkpoint.latest(config.state)
    if path is not None:
        return checkpoint.load(path, learning_rate=config.ppo.learning_rate)
    if fresh:
        state = checkpoint.fresh(learning_rate=config.ppo.learning_rate)
    elif config.initial_checkpoint is not None:
        state = checkpoint.load(config.initial_checkpoint, learning_rate=config.ppo.learning_rate)
    else:
        pair = checkpoint.find_legacy(config.legacy)
        if pair is None:
            raise SystemExit(f"No checkpoint in {config.state} and no earlier one to continue in {config.legacy}.\n"
                             "Pass --fresh to start from a randomly initialized policy.")
        state = checkpoint.import_legacy(*pair, learning_rate=config.ppo.learning_rate)
    earlier = checkpoint.unlisted(config.state, state.updates)
    if earlier:
        raise SystemExit(f"{config.state} holds checkpoints up to {earlier[-1]} but no latest.json that names the "
                         "newest, so starting here would replace them. Nothing was changed. Write latest.json, "
                         f'for example {{"checkpoint": "checkpoints/{earlier[-1]}"}}, or use another --state folder.')
    if save:
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
    directory = config.state / ("kept-games" if config.keep_game_files else "games") / f"game-{job.number:07d}"
    shutil.rmtree(directory, ignore_errors=True)
    started = time.monotonic()
    try:
        try:
            # The update by whole games reads arrays; an actor then records for it alone and keeps no frames.
            record = config.backend == "mlx" and ("arrays" if config.ppo.games_per_minibatch is not None else True)
            game = play(env, model, job, directory, guard=config.guard, on_progress=report,
                        both_seats=config.both_seats, record=record)
            if config.backend == "mlx":
                compiling = time.monotonic()
                for trajectory in game.trajectories:
                    if trajectory.compact is not None:  # recorded as arrays by native seats
                        continue
                    if trajectory.recipe is None:  # not recorded while playing
                        compile_recipe(model, trajectory)
                    if config.ppo.games_per_minibatch is not None:
                        compact_recipe(trajectory)
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
        return game
    finally:  # also when the actor is told to stop in the middle of the game
        if not config.keep_game_files:
            shutil.rmtree(directory, ignore_errors=True)


def _end_with_trainer(trainer):
    """Stop this actor once the trainer is gone, even while it waits on a silent engine."""
    while os.getppid() == trainer:
        time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)


def _stop_actor(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # once: nothing interrupts the cleanup this starts
    sys.exit(0)  # unwinds through the game in play, stopping its engine and removing its folder


def worker(config, pipe, trainer):
    """One actor process. It plays the games it is handed, one at a time."""
    torch.set_num_threads(1)
    # A game allocates millions of objects that are freed by reference counting. The cyclic collector
    # finds nearly nothing among them and took a tenth of a game's time; it runs between games instead.
    gc.disable()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the trainer handles Ctrl-C and stops its actors
    signal.signal(signal.SIGTERM, _stop_actor)
    threading.Thread(target=_end_with_trainer, args=(trainer,), daemon=True).start()
    engine = config.engine()
    # Native seats play exactly as the Python ones do; they record only the arrays of the update by whole games.
    native_seats = (config.backend == "mlx" and config.ppo.games_per_minibatch is not None and NativeEnv.usable(engine))
    env = (NativeEnv if native_seats else Env)(engine, config.limits)
    model = loaded = weights = None

    def send(kind, payload):
        # Plain pickle keeps tensors inside the message instead of in shared memory files.
        pipe.send_bytes(pickle.dumps((kind, payload), protocol=pickle.HIGHEST_PROTOCOL))

    try:
        while True:
            item = pipe.recv()
            if item is None:
                return
            path, job = item
            if path != loaded:
                model, loaded = checkpoint.load_model(path).freeze(), path  # an actor only plays with it
                weights = weights_sha256(model)  # of the weights as this process holds them
            last = [time.monotonic()]

            def report(day, choices, job=job):
                now = time.monotonic()
                if now - last[0] >= 1:
                    last[0] = now
                    send("progress", (job.number, day, choices))

            game = _play_job(env, model, job, config, report)
            game.weights = weights
            send("game", game)
            del game
            gc.collect()
    except (EOFError, ConnectionError):
        pass  # the pipe closed: the trainer is gone
    finally:
        env.close()


RENEW_AFTER_DECISIONS = 20_000  # an actor that played a game this long is replaced by a new process


class Pool:
    """The actor processes and which game each of them is playing.

    Every actor has its own pipe to the trainer. An actor that dies closes its
    end, which the trainer sees at once; a trainer that dies closes the other
    end, which stops the actor and its engine. An actor that finished a long
    game is replaced by a new process: Python does not give the memory of a
    game back, so fourteen actors would each stay as large as their largest game.
    """

    def __init__(self, config):
        self.config = config
        self.context = multiprocessing.get_context("spawn")
        self.retired = []  # replaced actors that have not exited yet
        self.pipes, self.processes = [None] * config.workers, [None] * config.workers
        for index in range(config.workers):
            self._start(index)
        self.idle = list(range(config.workers))
        self.busy = {}  # worker index -> its job

    def _start(self, index):
        ours, theirs = self.context.Pipe()
        process = self.context.Process(target=worker, args=(self.config, theirs, os.getpid()),
                                       name=f"actor-{index}", daemon=True)
        process.start()
        theirs.close()  # the actor now holds the only copy, so its death ends our stream
        self.pipes[index], self.processes[index] = ours, process

    def submit(self, path, job):
        index = self.idle.pop()
        self.busy[index] = job
        try:
            self.pipes[index].send((str(path), job))
        except OSError:
            pass  # the actor is gone; next() replaces it and reports the game as failed

    def next(self, timeout=1):
        """The next message from an actor, or None after ``timeout`` quiet seconds.

        Returns ``("progress", index, (number, day, choices))`` or
        ``("game", index, Game)``. An actor that died is replaced, and the game
        it was playing comes back as failed.
        """
        ready = multiprocessing.connection.wait(self.pipes, timeout=timeout)
        if not ready:
            return None
        index = self.pipes.index(ready[0])
        try:
            kind, payload = pickle.loads(ready[0].recv_bytes())
        except (EOFError, OSError):
            process = self.processes[index]
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            ready[0].close()
            self._start(index)
            job = self.busy.pop(index, None)
            if job is None:
                return None  # it was waiting for a game
            self.idle.append(index)
            return "game", index, Game(job, "failed", "actor_died", None,
                                       f"The actor process exited with status {process.exitcode}",
                                       0, (0, 0), 0.0, None)
        if kind == "game":
            del self.busy[index]
            if sum(payload.choices) >= RENEW_AFTER_DECISIONS:
                self._renew(index)
            self.idle.append(index)
        return kind, index, payload

    def _renew(self, index):
        """Replace an idle actor by a new process; the old one ends by itself."""
        pipe, process = self.pipes[index], self.processes[index]
        try:
            pipe.send(None)
        except OSError:
            pass
        pipe.close()
        self.retired = [old for old in self.retired if old.is_alive()] + [process]
        self._start(index)

    def shutdown(self):
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except OSError:
                pass
        deadline = time.monotonic() + 5
        processes = self.processes + self.retired
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()  # its handler stops the game it is playing
        for process in processes:
            process.join(timeout=15)


# -- what the person watching sees ---------------------------------------------

def _clock(seconds, *, tenths=False):
    whole, fraction = divmod(round(seconds * 10), 10) if tenths else (int(seconds), 0)
    clock = f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
    return f"{clock}.{fraction}" if tenths else clock


class Display:
    """Keep a rolling table above one live status line."""

    rows = 10
    team_labels = re.compile(r"\b(?:Red|Blue|[RB][0-9]+)\b")
    header = f"{'Game':>8}  {'Result':<18}  {'Learner':<7}  {'Day':>3}  {'Gold mines':<11}  {'Samples':>9}  {'Time':>11}"

    def __init__(self, out):
        self.out = out
        self.live = out.isatty() and os.environ.get("TERM", "").lower() != "dumb"
        self.color = self.live and "NO_COLOR" not in os.environ
        self.table_live = self.live and shutil.get_terminal_size().lines >= self.rows + 2
        self.table_visible = False
        self.recent = deque(maxlen=self.rows)
        self.playing = {}  # worker index -> (game number, day, choices)

    def color_teams(self, text):
        if not self.color:
            return text
        return self.team_labels.sub(
            lambda match: f"\x1b[{91 if match[0].startswith('R') else 94}m{match[0]}\x1b[0m", text)

    def line(self, text):
        self.out.write(("\r\x1b[2K" if self.live else "") + self.color_teams(text) + "\n")
        self.out.flush()

    def status(self, *fields):
        if not self.live:
            return
        width = max(0, shutil.get_terminal_size().columns - 1)
        text = fields[0][:width]
        for field in fields[1:]:
            if len(text) + 2 + len(field) > width:
                break
            text += "  " + field
        self.out.write("\r\x1b[2K" + self.color_teams(text))
        self.out.flush()

    def table(self, *, pad):
        rows = [self.header, *self.recent]
        if pad:
            rows.extend("       ·" for _ in range(self.rows - len(self.recent)))
        if self.live:
            width = max(0, shutil.get_terminal_size().columns - 1)
            above = f"\x1b[{self.rows + 1}A" if self.table_visible else ""
            self.out.write(above + "".join("\r\x1b[2K" + self.color_teams(row[:width]) + "\n" for row in rows))
        else:
            self.out.write("\n".join(rows) + "\n")
        self.out.flush()

    def finish_games(self):
        if self.table_visible or self.recent:
            self.table(pad=self.table_visible)
        self.table_visible = False
        self.recent.clear()

    def collecting(self, update, done, target, tally, started, average_days=None):
        if not self.live:
            return
        if self.table_live:
            self.table(pad=True)
            self.table_visible = True
        days = " ".join(str(day) for _, day, _ in sorted(self.playing.values()))
        average = f"Avg days {average_days:.1f}" if average_days is not None else "Avg days –"
        self.status(f"Games {done}/{target}", average, *self.score(tally), _clock(time.monotonic() - started),
                    f"Update {update}", f"Days in play {days}" if days else "")

    def game(self, game, samples):
        if game.status == "completed":
            result = COLORS[game.outcomes.index("win")].title() + " won"
        elif game.status == "truncated":
            limit = "day" if game.reason == "game_day_limit" else "action"
            result = f"Cutoff ({limit})"
        else:
            result = f"Failed ({game.reason})"
        if len(result) > 18:
            result = result[:17] + "…"
        learner = COLORS[game.job.learner].title()
        gold = "?"
        if game.gold_mine_captures is not None:
            red, blue = game.gold_mine_captures
            total = red + blue
            gold = "-" if total == 0 else "Red" if red else "Blue"
            if total > 1:
                gold = f"({total}) R{red}/B{blue}"
        self.recent.append(f"{game.job.number:>8}  {result:<18}  {learner:<7}  {game.day:>3}"
                           f"  {gold:<11}  {samples:>9,}  {_clock(game.seconds, tenths=True):>11}")

    @staticmethod
    def score(tally):
        return (f"Blue wins {tally['blue']:,}", f"Red wins {tally['red']:,}",
                f"Cutoffs {tally['truncated']:,}", f"Failed {tally['failed']:,}")


class Log:
    """Append-only ``log.jsonl``: one record per game and per update."""

    def __init__(self, path):
        self.last_game = 0  # the highest game number earlier runs handed out
        line = "\n"
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    if line.startswith('{"event": "job"'):
                        try:
                            self.last_game = max(self.last_game, json.loads(line)["number"])
                        except ValueError:
                            pass  # cut short by a crash
        self.stream = path.open("a")
        if not line.endswith("\n"):
            self.stream.write("\n")  # the previous run stopped inside a line

    def write(self, event, **fields):
        record = {"event": event, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()


# -- the loop --------------------------------------------------------------------

class Batch:
    """The games of one update while they are being played."""

    def __init__(self, pool, config, target, numbers, update, display, log, seeds=None, starts=None):
        self.pool, self.config, self.target, self.numbers, self.update = pool, config, target, numbers, update
        self.display, self.log = display, log
        self.seeds = seeds  # a ``random.Random`` that gives the games' seeds, for runs that must play the same games
        self.starts = starts  # practice games: the start positions a game's own is drawn from (``heroes.starts``)
        self.began = {}  # game number -> the name of its start position
        self.games, self.tally, self.started = [], {"red": 0, "blue": 0, "truncated": 0, "failed": 0}, time.monotonic()

        self.days_played = 0

    @property
    def average_days(self):
        return self.days_played / len(self.games) if self.games else None

    @property
    def done(self):
        return len(self.games) >= self.target

    def step(self, path, timeout=1):
        """Hand games to idle actors, played with the weights at ``path``, and take one message.

        Returns False when none came within ``timeout`` seconds.
        """
        pool, games, tally = self.pool, self.games, self.tally
        while pool.idle and len(games) + len(pool.busy) < self.target:
            number = next(self.numbers)
            self.log.write("job", number=number)  # before the game exists, so its number is never handed out twice
            if self.seeds is None:
                job = Job(number, secrets.randbelow(2**31 - 1) + 1, number % 2,
                          (secrets.randbits(62), secrets.randbits(62)))
            else:
                job = Job(number, self.seeds.randrange(1, 2**31 - 1), number % 2,
                          (self.seeds.getrandbits(62), self.seeds.getrandbits(62)))
            if self.starts is not None:  # drawn after the seeds: a run without start positions draws what it drew
                self.began[number], file, days = self.starts.choose(self.seeds)
                job = replace(job, game_map=file, max_days=days)
            pool.submit(path, job)
        message = pool.next(timeout)
        if message is None:
            return False
        if message[0] == "progress":
            self.display.playing[message[1]] = message[2]
            return True
        _, index, game = message
        self.display.playing.pop(index, None)
        samples = sum(len(trajectory.decisions) for trajectory in game.trajectories)
        seat_trajectories = {COLORS[seat]: trajectory for seat, trajectory in
                             ((game.job.learner, game.trajectory), (1 - game.job.learner, game.other))
                             if trajectory is not None}
        self.log.write("game", number=game.job.number, engine_seed=game.job.engine_seed,
                       policy_seeds=list(game.job.policy_seeds), learner=COLORS[game.job.learner],
                       status=game.status, reason=game.reason, outcomes=game.outcomes, day=game.day,
                       training_rewards={seat: t.reward for seat, t in seat_trajectories.items()},
                       window_repeats={seat: len(t.window_repeat_indices) for seat, t in seat_trajectories.items()},
                       window_penalties={seat: t.window_penalty for seat, t in seat_trajectories.items()},
                       gold_mine_captures=(dict(zip(COLORS, game.gold_mine_captures))
                                           if game.gold_mine_captures is not None else None),
                       choices=list(game.choices), samples=samples, seconds=round(game.seconds, 1),
                       compile_seconds=round(game.compile_seconds, 1), update=self.update,
                       weights=game.weights, detail=game.detail,
                       **({} if self.starts is None else {"start": self.began.pop(game.job.number, None)}),
                       **({} if game.job.max_days is None else {"max_days": game.job.max_days}))
        self.display.game(game, samples)
        if not game.trajectories:
            tally["failed"] += 1
            if tally["failed"] > self.target or (not games and tally["failed"] >= max(3, self.config.workers)):
                raise RuntimeError(f"{tally['failed']} games failed, the last one with {game.reason}:\n{game.detail}\n"
                                   f"Every failed game is in {self.config.state / 'log.jsonl'}.")
        else:
            tally["truncated" if game.status == "truncated" else COLORS[game.outcomes.index("win")]] += 1
            games.append(game)
            # A day cutoff is delivered on the first request of the following day.
            self.days_played += min(game.day, game.job.max_days or self.config.limits.max_days)
        return True


def collect(pool, config, path, target, numbers, update, display, log, batch=None, seeds=None, starts=None):
    """Play until ``target`` games have given a trajectory, with the weights at ``path``.

    ``batch`` is a batch begun earlier, during the previous update.
    """
    batch = batch or Batch(pool, config, target, numbers, update, display, log, seeds, starts)
    started = time.monotonic()
    display.collecting(update, len(batch.games), target, batch.tally, started, batch.average_days)
    while not batch.done:
        batch.step(path)
        display.collecting(update, len(batch.games), target, batch.tally, started, batch.average_days)
    return batch.games, batch.tally, batch.average_days


def train(config, games, *, fresh=False, out=sys.stderr, seed=None):
    """Play ``games`` games in batches of ``config.games_per_update`` and update after each batch.

    The games' seeds are random. With ``seed`` they are drawn from that number
    instead, so that two runs from the same checkpoint play the same games.
    """
    batch_size = games if config.games_per_update is None else min(games, config.games_per_update)
    config = replace(config, games_per_update=batch_size, workers=min(config.workers, batch_size))
    if config.play_during_update and config.backend == "mlx":
        # Before its first step the MLX update compares what it computes for a
        # minibatch with what was recorded while playing, and stops if they
        # differ. Games begun during an update were played by the weights from
        # before it, so that check would stop the next update.
        raise SystemExit("--play-during-update works with --backend torch only: the MLX update checks its first "
                         "minibatch against the weights that played the games, and the first games of each "
                         "batch would have been played by the weights from before the previous update.")
    config.engine().check()
    update_policy = update_function(config)
    config.state.mkdir(parents=True, exist_ok=True)
    with (config.state / "lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another training run is using {config.state}.")
        return _train(config, games, update_policy, fresh, out, seed)


def _adam_steps(optimizer):
    """The steps Adam has taken, as its state counts them: one number, unless its parameters disagree."""
    return sorted({int(entry["step"]) for entry in optimizer.state_dict()["state"].values() if "step" in entry})


def _train(config, games, update_policy, fresh, out, seed=None):
    import random
    seeds = None if seed is None else random.Random(seed)
    state = restore(config, fresh=fresh)
    if config.new_optimizer:
        # The moments the checkpoint kept are those of the update that made it; this run's updates start their own.
        state.optimizer.state.clear()
    shutil.rmtree(config.state / "games", ignore_errors=True)  # left behind by actors that were killed
    display, log = Display(out), Log(config.state / "log.jsonl")
    display.line(f"Continuing from update {state.updates} ({state.progress['games']} games, "
                 f"{state.progress['decisions']} decisions) in {config.state}")
    starts = None
    if config.starts is not None:
        from .starts import load
        starts = load(config.starts)
        display.line(f"Practice games: every game begins from one of {len(starts.names)} starts ({config.starts}); "
                     f"a start without a day limit of its own is stopped after day {config.limits.max_days}")
    log.write("start", games=games, games_per_update=config.games_per_update,
              backend=config.backend, workers=config.workers, progress=state.progress,
              game_map=str(config.game_map), window_repeat_penalty=WINDOW_REPEAT_PENALTY,
              ppo=config.ppo.to_dict(), limits=asdict(config.limits), guard=config.guard,
              both_seats=config.both_seats, weights=weights_sha256(state.model), seed=seed,
              new_optimizer=config.new_optimizer,
              **({} if starts is None else {"starts": starts.shares(), "starts_file": str(config.starts)}))
    # Failed and discarded games keep their numbers, so a number never names two games.
    numbers = itertools.count(max(state.progress["games"], log.last_game) + 1)
    pool, saved_update = Pool(config), state.updates
    following = None  # the next batch, begun while an update ran
    try:
        while games > 0:
            target = min(config.games_per_update, games)
            update, started = state.updates + 1, time.monotonic()
            path = checkpoint.latest(config.state)
            batch, tally, average_days = collect(
                pool, config, path, target, numbers, update, display, log, following, seeds, starts)
            collected = time.monotonic()
            display.finish_games()
            display.line("  |  ".join(display.score(tally)))
            following = None
            if config.play_during_update and games > target:
                following = Batch(pool, config, min(config.games_per_update, games - target), numbers, update + 1,
                                  display, log, seeds, starts)

            failed = []  # what stopped the next batch while this update ran

            def progress(step, total, update=update, collected=collected, path=path, following=following,
                         failed=failed):
                display.status(f"PPO {update}  step {step}/{total}", _clock(time.monotonic() - collected))
                # The actors play on with the weights this update started from.
                # If that batch fails, this update still finishes and is saved first.
                try:
                    while following is not None and not failed and not following.done and following.step(path, timeout=0):
                        pass
                except Exception as error:
                    failed.append(error)

            trajectories = [trajectory for game in batch for trajectory in game.trajectories]
            # What the update started from and what it trained on, for whoever checks the round afterwards.
            audit = {"weights_before": weights_sha256(state.model), "adam_steps_before": _adam_steps(state.optimizer),
                     "window_memory_weight_before": state.model.window_memory.weight.item(),
                     "played_with": sorted({str(game.weights) for game in batch}), "trajectories": len(trajectories),
                     "trajectory_samples": sum(len(trajectory.decisions) for trajectory in trajectories)}
            report = update_policy(state.model, state.optimizer, trajectories,
                                   config=config.ppo, generator=state.generator, progress=progress)
            del trajectories
            updated = time.monotonic()
            audit.update(weights_after=weights_sha256(state.model), adam_steps_after=_adam_steps(state.optimizer),
                         window_memory_weight_after=state.model.window_memory.weight.item(),
                         steps=report.steps, initial_check=report.initial_check)
            state.progress = {"updates": update, "games": state.progress["games"] + len(batch),
                              "decisions": state.progress["decisions"] + report.samples,
                              "optimizer_steps": state.progress["optimizer_steps"] + report.optimizer_steps}
            saved = checkpoint.save(config.state, state)
            saved_update = update
            del batch
            log.write("update", update=update, games=target, **tally, samples=report.samples,
                      optimizer_steps=report.optimizer_steps, policy_loss=report.policy_loss,
                      value_loss=report.value_loss, approximate_kl=report.approximate_kl,
                      clip_fraction=report.clip_fraction, collect_seconds=round(collected - started, 1),
                      update_seconds=round(updated - collected, 1), update_timings=report.timings_seconds,
                      checkpoint=saved.name, progress=state.progress, average_days=round(average_days, 2), **audit)
            display.line(f"update {update} saved  |  {saved.name}  |  games {_clock(collected - started, tenths=True)}"
                         f"  |  PPO {_clock(updated - collected, tenths=True)}"
                         f"  |  {report.optimizer_steps:,} steps  |  {report.samples:,} samples")
            display.line(f"  Total {_clock(updated - started, tenths=True)}  |  Avg days {average_days:.1f}"
                         f"  |  policy loss {report.policy_loss:+.4f}"
                         f"  |  value loss {report.value_loss:.4f}  |  KL {report.approximate_kl:.5f}")
            games -= target
            if failed:
                raise failed[0]
    except KeyboardInterrupt:
        display.finish_games()
        display.line(f"Stopped. The next run continues from update {saved_update}; "
                     "games of the unfinished batch are discarded.")
    finally:
        display.finish_games()
        pool.shutdown()
    return state
