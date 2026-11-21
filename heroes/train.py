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

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
import pickle
import secrets
import shutil
import signal
import sys
import threading
import time
import traceback

import torch

from . import checkpoint
from .actor import Game, Job, compile_recipe, play
from .engine.env import COLORS, Env


def against_reference(config, number):
    """Whether game ``number`` has the reference checkpoint in the opponent's seat: every second pair of games.

    The learner's seat alternates by game number, so the pairs give the
    learner both colours against either opponent.
    """
    return config.reference is not None and number // 2 % 2 == 0


def restore(config, *, fresh=False, save=True):
    """The newest training state: this pipeline's own, else the legacy one, else a new policy.

    A state that does not come from this folder is saved as its first checkpoint.
    """
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
            # Against another checkpoint only the learner's seat was played by the weights the update trains.
            learner = model[job.learner] if isinstance(model, tuple) and job.learner is not None else model
            game = play(env, model, job, directory, guard=config.guard, on_progress=report,
                        both_seats=config.both_seats and learner is model, stopped_reward=config.stopped_game_reward)
            if config.backend == "mlx":
                compiling = time.monotonic()
                for trajectory in game.trajectories:
                    compile_recipe(learner, trajectory)
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
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the trainer handles Ctrl-C and stops its actors
    signal.signal(signal.SIGTERM, _stop_actor)
    threading.Thread(target=_end_with_trainer, args=(trainer,), daemon=True).start()
    env = Env(config.engine(), config.limits)
    policies = {}  # checkpoint path -> its policy, for the checkpoints of the game in hand

    def send(kind, payload):
        # Plain pickle keeps tensors inside the message instead of in shared memory files.
        pipe.send_bytes(pickle.dumps((kind, payload), protocol=pickle.HIGHEST_PROTOCOL))

    try:
        while True:
            item = pipe.recv()
            if item is None:
                return
            path, job = item  # one checkpoint for both seats, or a pair: red's and blue's
            paths = path if isinstance(path, tuple) else (path,)
            policies = {part: policies[part] if part in policies else checkpoint.load_model(part) for part in paths}
            model = tuple(policies[part] for part in paths) if isinstance(path, tuple) else policies[path]
            last = [time.monotonic()]

            def report(day, choices, job=job):
                now = time.monotonic()
                if now - last[0] >= 1:
                    last[0] = now
                    send("progress", (job.number, day, choices))

            send("game", _play_job(env, model, job, config, report))
    except (EOFError, ConnectionError):
        pass  # the pipe closed: the trainer is gone
    finally:
        env.close()


class Pool:
    """The actor processes and which game each of them is playing.

    Every actor has its own pipe to the trainer. An actor that dies closes its
    end, which the trainer sees at once; a trainer that dies closes the other
    end, which stops the actor and its engine.
    """

    def __init__(self, config):
        self.config = config
        self.context = multiprocessing.get_context("spawn")
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
        """Hand ``job`` to an idle actor. ``path`` is the checkpoint both seats play with, or a pair of them."""
        index = self.idle.pop()
        self.busy[index] = job
        try:
            self.pipes[index].send((tuple(map(str, path)) if isinstance(path, tuple) else str(path), job))
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
            self.idle.append(index)
        return kind, index, payload

    def shutdown(self):
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except OSError:
                pass
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

    def __init__(self, pool, config, target, numbers, update, display, log):
        self.pool, self.config, self.target, self.numbers, self.update = pool, config, target, numbers, update
        self.display, self.log = display, log
        self.games, self.tally, self.started = [], {"red": 0, "blue": 0, "truncated": 0, "failed": 0}, time.monotonic()
        if config.reference is not None:  # the learner's results in its games against the reference
            self.tally.update(reference_wins=0, reference_losses=0, reference_stopped=0)

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
            job = Job(number, secrets.randbelow(2**31 - 1) + 1, number % 2, (secrets.randbits(62), secrets.randbits(62)))
            if against_reference(self.config, number):  # red's weights, then blue's
                pool.submit((path, self.config.reference) if job.learner == 0 else (self.config.reference, path), job)
            else:
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
        reference = against_reference(self.config, game.job.number)
        self.log.write("game", number=game.job.number, engine_seed=game.job.engine_seed,
                       policy_seeds=list(game.job.policy_seeds), learner=COLORS[game.job.learner],
                       opponent="reference" if reference else "self",
                       status=game.status, reason=game.reason, outcomes=game.outcomes, day=game.day,
                       choices=list(game.choices), samples=samples, seconds=round(game.seconds, 1),
                       compile_seconds=round(game.compile_seconds, 1), update=self.update, detail=game.detail)
        self.display.game(game, samples)
        if not game.trajectories:
            tally["failed"] += 1
            if tally["failed"] > self.target or (not games and tally["failed"] >= max(3, self.config.workers)):
                raise RuntimeError(f"{tally['failed']} games failed, the last one with {game.reason}:\n{game.detail}\n"
                                   f"Every failed game is in {self.config.state / 'log.jsonl'}.")
        else:
            tally["truncated" if game.status == "truncated" else COLORS[game.outcomes.index("win")]] += 1
            if reference:
                tally["reference_stopped" if game.status == "truncated" else
                      "reference_wins" if game.outcomes[game.job.learner] == "win" else "reference_losses"] += 1
            games.append(game)
        return True


def collect(pool, config, path, target, numbers, update, display, log, batch=None):
    """Play until ``target`` games have given a trajectory, with the weights at ``path``.

    ``batch`` is a batch begun earlier, during the previous update.
    """
    batch = batch or Batch(pool, config, target, numbers, update, display, log)
    started = time.monotonic()
    while not batch.done:
        batch.step(path)
        display.collecting(update, len(batch.games), target, batch.tally, started)
    return batch.games, batch.tally


def train(config, games, *, fresh=False, out=sys.stderr):
    """Play ``games`` games in batches of ``config.games_per_update`` and update after each batch."""
    if config.stopped_game_reward not in (None, -1, 0):
        raise SystemExit("A game stopped at a limit is worth -1 (both seats lost) or 0 (a draw) to a seat, "
                         f"not {config.stopped_game_reward!r}.")
    if config.reference is not None and not config.reference.is_file():
        raise SystemExit(f"No reference checkpoint at {config.reference}")
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
        return _train(config, games, update_policy, fresh, out)


def _train(config, games, update_policy, fresh, out):
    state = restore(config, fresh=fresh)
    shutil.rmtree(config.state / "games", ignore_errors=True)  # left behind by actors that were killed
    display, log = Display(out), Log(config.state / "log.jsonl")
    display.line(f"Continuing from update {state.updates} ({state.progress['games']} games, "
                 f"{state.progress['decisions']} decisions) in {config.state}")
    log.write("start", games=games, backend=config.backend, workers=config.workers, progress=state.progress,
              ppo=config.ppo.to_dict(), limits=asdict(config.limits), guard=config.guard,
              both_seats=config.both_seats, stopped_game_reward=config.stopped_game_reward,
              reference=None if config.reference is None else str(config.reference))
    # Failed and discarded games keep their numbers, so a number never names two games.
    numbers = itertools.count(max(state.progress["games"], log.last_game) + 1)
    pool, saved_update = Pool(config), state.updates
    following = None  # the next batch, begun while an update ran
    try:
        while games > 0:
            target = min(config.games_per_update, games)
            update, started = state.updates + 1, time.monotonic()
            path = checkpoint.latest(config.state)
            batch, tally = collect(pool, config, path, target, numbers, update, display, log, following)
            collected = time.monotonic()
            following = None
            if config.play_during_update and games > target:
                following = Batch(pool, config, min(config.games_per_update, games - target), numbers, update + 1,
                                  display, log)

            failed = []  # what stopped the next batch while this update ran

            def progress(step, total, update=update, collected=collected, path=path, following=following,
                         failed=failed):
                display.status(f"update {update}  PPO step {step}/{total}  {_clock(time.monotonic() - collected)}")
                # The actors play on with the weights this update started from.
                # If that batch fails, this update still finishes and is saved first.
                try:
                    while following is not None and not failed and not following.done and following.step(path, timeout=0):
                        pass
                except Exception as error:
                    failed.append(error)

            report = update_policy(state.model, state.optimizer,
                                   [trajectory for game in batch for trajectory in game.trajectories],
                                   config=config.ppo, generator=state.generator, progress=progress)
            updated = time.monotonic()
            state.progress = {"updates": update, "games": state.progress["games"] + len(batch),
                              "decisions": state.progress["decisions"] + report.samples,
                              "optimizer_steps": state.progress["optimizer_steps"] + report.optimizer_steps}
            saved = checkpoint.save(config.state, state)
            saved_update = update
            del batch
            log.write("update", update=update, games=target, **tally, samples=report.samples,
                      optimizer_steps=report.optimizer_steps, planned_steps=report.planned_steps,
                      policy_loss=report.policy_loss,
                      value_loss=report.value_loss, approximate_kl=report.approximate_kl,
                      clip_fraction=report.clip_fraction, collect_seconds=round(collected - started, 1),
                      update_seconds=round(updated - collected, 1), update_timings=report.timings_seconds,
                      checkpoint=saved.name, progress=state.progress)
            steps = (f"{report.optimizer_steps} steps" if report.optimizer_steps >= report.planned_steps else
                     f"{report.optimizer_steps} of {report.planned_steps} steps (stopped at the KL limit)")
            if config.reference is not None:
                steps += (f"  against the reference {tally['reference_wins']} won, {tally['reference_losses']} lost, "
                          f"{tally['reference_stopped']} stopped")
            display.line(f"update {update} saved  {report.samples} samples  {steps}  "
                         f"policy loss {report.policy_loss:+.4f}  value loss {report.value_loss:.4f}  "
                         f"KL {report.approximate_kl:.5f}  games {_clock(collected - started)}  "
                         f"PPO {_clock(updated - collected)}")
            games -= target
            if failed:
                raise failed[0]
    except KeyboardInterrupt:
        display.line(f"Stopped. The next run continues from update {saved_update}; "
                     "games of the unfinished batch are discarded.")
    finally:
        pool.shutdown()
    return state
