"""Play self-play games and collect trajectories for PPO.

Both seats use the same weights with separate recurrent states and sampling.
The designated learner alternates by game. A 365-day draw trains both seats
with a final reward of -1 each. ``both_seats`` also collects both in other games.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import os
import time

import torch

from .engine.env import COLORS, TRAINING_DAY_LIMIT, Request
from .game.window_repeats import RepeatedOpens, repeated_empty_opens
from .learner.ppo import Trajectory
from .model import fast
from .model.policy import ContextualEncoder, Recording


@dataclass(frozen=True)
class Job:
    number: int  # game number; also the engine's episode id
    engine_seed: int  # 1 .. 2**31 - 1
    learner: int | None  # 0 plays red, 1 plays blue; None when the game is played for its result alone
    policy_seeds: tuple  # sampling seed of each seat
    game_map: str | None = None  # this game alone begins from another map file: a start position (``heroes.starts``)
    max_days: int | None = None  # this game alone is stopped after another day than the run's limit


@dataclass
class Game:
    job: Job
    status: str  # completed, truncated or failed
    reason: str
    outcomes: tuple | None  # by seat, for a completed game
    detail: str | None
    day: int  # last game day reached
    choices: tuple  # decisions answered by each seat
    seconds: float
    trajectory: Trajectory | None  # the learner seat's; None when that seat cannot be trained on
    compile_seconds: float = 0.0  # time spent compiling the recipes
    other: Trajectory | None = None  # the opponent seat's trajectory, when both seats are trained on
    weights: str | None = None  # SHA-256 of the weights both seats played with, as the actor holds them
    gold_mine_captures: tuple[int, int] | None = None  # Red, Blue, when reported by the actors

    @property
    def trajectories(self):
        """What the game gives the update: the learner seat's trajectory, then the opponent's, each if there is one."""
        return tuple(item for item in (self.trajectory, self.other) if item is not None)


class Seat:
    """One player's policy during a game: encoder, recurrent state and sampling."""

    def __init__(self, model, seed, *, guard, record=False):
        self.model = model
        # What the seat plays with: the frozen model's arithmetic in C where that can be had (heroes.model.fast),
        # else the model itself. The recorder below reads the model only for what no weight enters.
        self.acting = fast.acting(model)
        self.encoder = ContextualEncoder(guard)
        self.state = self.acting.initial_state()
        self.generator = torch.Generator().manual_seed(seed)
        self.values = {}  # decision frame -> value estimate
        self.choices = {}  # decision frame -> (action, log-probability)
        self.pending = None  # (decision frame, its action distribution)
        self.observed = 0  # frames the policy has read
        self.recorder = None  # compiles the seat's recipe while it plays
        self.repeats = None  # finds the repeated empty window visits while it plays, when frames are forgotten
        if record:
            from .learner.recipes import Recorder
            self.recorder = Recorder(model, small_maps=record == "arrays")
            if record == "arrays":
                self.repeats = RepeatedOpens()

    def forget(self, env, player):
        """Let the environment drop what the seat's earlier frames showed, once nothing reads it again.

        The policy has read them, the recorder holds what it compiles from
        them, and the search for repeated window visits is past them. What is
        left of a game of 150,000 decisions is then a few gigabytes smaller.
        """
        if self.repeats is not None:
            self.repeats.read(env.received(player))
            env.forget(player, min(self.repeats.needed, self.observed))

    def observe(self, frames, stop=None):
        """Advance the recurrent state over the seat's frames from the first unread one up to ``stop``.

        ``frames`` are all the seat's frames, frame ``n`` at position ``n``;
        without ``stop`` every frame there is.
        """
        recorder = self.recorder
        stop = len(frames) if stop is None else stop
        with torch.no_grad():
            for position in range(self.observed, stop):
                frame = frames[position]
                item = self.encoder.encode(frame)
                output = self.acting.forward_step(item, self.state)
                self.state = output.state
                if output.decision is not None:
                    self.values[frame.seq] = float(output.decision.value)
                    self.pending = (frame.seq, output.decision.distribution)
                if recorder is not None:
                    recorder.feed(item)
        self.observed = max(self.observed, stop)

    def observe_ahead(self, frames):
        """Read the frames that are already there, up to the seat's next decision.

        The frames a seat receives while the engine works on an action are read
        at once instead of with the next decision; a decision is read only when
        the environment hands it out. The recurrent state takes the same frames
        in the same order either way.
        """
        stop = position = self.observed
        count = len(frames)
        while position < count and frames[position].kind != "decision":
            position += 1
        if position > stop:
            self.observe(frames, position)

    def act(self):
        """Sample the answer to the decision observed last."""
        seq, distribution = self.pending
        if seq not in self.choices:
            if self.recorder is not None:
                distribution.recording = Recording()
            with torch.no_grad():
                action, evaluated = distribution.sample_evaluated(generator=self.generator)
            self.choices[seq] = (action, float(evaluated.log_prob))
            if self.recorder is not None:
                self.recorder.sampled(seq, distribution.recording)
                distribution.recording = None
        return self.choices[seq][0]

    def recorded(self, frames, trajectory):
        """Give ``trajectory`` the recipe recorded while playing; False when it has to be compiled instead.

        ``frames`` are all the seat's frames. The policy read a prefix of them;
        the recipe covers the rest as well, as the compiler's does.
        """
        if self.recorder is None:
            return False
        rest = [self.encoder.encode(frame) for frame in frames[self.observed:]]
        recipe = self.recorder.finish(rest, trajectory.decisions)
        self.recorder = None
        if recipe is None:
            return False
        trajectory.recipe, trajectory.frames = recipe, None
        return True


class Unusable(ValueError):
    """The game ended in a state that gives no well-defined trajectory."""


def learner_samples(frames, *, natural):
    """The learner's resolved actions as ``[(decision frame, successor frame)]``.

    An action is resolved once the engine reported its result. Its successor is
    the seat's next decision, whose value the action bootstraps from. In a
    naturally finished game the last action has no successor and owns the
    terminal reward; that also holds for an action the engine had consumed when
    the game ended, which it closes with a terminal instead of a result.
    """
    decisions, submitted, consumed, superseded, results, terminals = [], set(), [], {}, [], 0
    for frame in frames:
        kind = frame.kind
        if kind == "decision":
            decisions.append(frame.seq)
        elif kind == "submitted":
            submitted.add(frame.decision_seq)
        elif kind == "consumed":
            consumed.append(frame.decision_seq)
        elif kind == "superseded":
            superseded.setdefault(frame.decision_seq, []).append(frame.reason)
        elif kind == "result":
            results.append((frame.decision_seq, frame.seq))
        elif kind == "terminal":
            terminals += 1
    acted = set(consumed)
    # A decision the engine withdrew before using its answer is not part of the trajectory.
    live = [seq for seq in decisions if seq in acted or seq not in superseded]
    resolved = {decision for decision, _ in results}
    if (natural and consumed and consumed[-1] == decisions[-1] and consumed[-1] in submitted
            and consumed[-1] not in resolved and superseded.get(consumed[-1]) == ["terminal"] and terminals == 1):
        results.append((consumed[-1], None))
        resolved.add(consumed[-1])
    samples = []
    for decision, result in results:
        position = len(live) if result is None else bisect_right(live, result)
        samples.append((decision, live[position] if position < len(live) else None))
    if natural:
        if acted != resolved:
            raise Unusable("an action was still unresolved when the game ended")
        if samples:
            samples[-1] = (samples[-1][0], None)
    return samples


def build_trajectory(frames, seat, *, natural, outcome, guard, day_limit_draw=False):
    """Join the seat's resolved actions with what its policy recorded while playing."""
    samples = learner_samples(frames, natural=natural)
    reward = 0
    if natural or day_limit_draw:
        reward = 1 if outcome == "win" else -1
    else:
        # A technical cutoff needs a following decision for its bootstrap.
        while samples and samples[-1][1] not in seat.values:
            samples.pop()
    if not samples:
        raise Unusable("this seat completed no action")
    for (_, successor), (following, _) in zip(samples, samples[1:]):
        if successor != following:
            raise Unusable("this seat's actions do not form one unbroken sequence")
    decisions = tuple(decision for decision, _ in samples)
    if any(decision not in seat.choices for decision in decisions):
        raise Unusable("an action in the history was not sampled by this policy")
    found = getattr(seat, "repeats", None)  # while playing, when the frames no longer hold their observations
    repeat_opens = set(repeated_empty_opens(frames) if found is None else found.finish(frames))
    return Trajectory(
        decisions=decisions,
        log_probs=tuple(seat.choices[decision][1] for decision in decisions),
        values=tuple(seat.values[decision] for decision in decisions),
        reward=reward, bootstrap=0.0 if reward else seat.values[samples[-1][1]],
        actions=tuple(seat.choices[decision][0] for decision in decisions),
        frames=tuple(frames), guard=guard,
        window_repeat_indices=tuple(i for i, seq in enumerate(decisions) if seq in repeat_opens))


def compile_recipe(model, trajectory):
    """Replace the trajectory's frames with the compiled recipe the MLX learner reads."""
    from .learner.recipes import compile_history
    encoder = ContextualEncoder(trajectory.guard)
    trajectory.recipe = compile_history(model, (encoder.encode(frame) for frame in trajectory.frames),
                                        dict(zip(trajectory.decisions, trajectory.actions)))
    trajectory.frames = None
    return trajectory


def compact_recipe(trajectory):
    """Replace the trajectory's compiled recipe with its arrays, which is what the update by whole games reads.

    The arrays are a quarter of the recipe's size and travel between processes
    in a fraction of a second; the recipe of a long game takes half a minute.
    """
    from .learner.compact import Compact
    trajectory.compact = Compact.from_history(trajectory.recipe, trajectory.decisions)
    trajectory.recipe = None
    return trajectory


def play(env, model, job, directory, *, guard, on_progress=None, both_seats=False, record=False):
    """Play one game to its end and return it with the learner's trajectory (see ``_play`` for the rest).

    A job with a day limit of its own is played under that limit and the
    environment has its own again afterwards: a mix of start positions that
    are stopped early and ordinary games that are not is one batch.
    """
    limits = env.limits
    if job.max_days is not None:
        env.limits = replace(limits, max_days=job.max_days)
    try:
        if getattr(env, "raw", False):  # a NativeEnv: its producers and the seats are heroes.model.native's
            return _play_native(env, model, job, directory, guard=guard, on_progress=on_progress, both_seats=both_seats,
                                record=record)
        return _play(env, model, job, directory, guard=guard, on_progress=on_progress, both_seats=both_seats,
                     record=record)
    finally:
        env.limits = limits


def _play(env, model, job, directory, *, guard, on_progress=None, both_seats=False, record=False):
    """Play one game to its end and return it with the learner's trajectory.

    A 365-day draw requires a usable trajectory from each seat. For other
    games, one usable learner trajectory suffices (either seat with ``both_seats``).
    A game can end before one seat has had a turn (red takes blue's town on
    day 1): it is a completed game with its outcome, the seat that acted has
    its trajectory, and the seat that did not has none. When that seat is the
    only learner the game gives no trajectory and ends as
    ``learner_never_acted``, with its outcome kept.
    With ``record`` the recipe the MLX learner reads is compiled while the game
    is played, for both seats because a draw trains both, and a trajectory
    comes back with it and without its frames; a trajectory whose recipe could
    not be recorded keeps its frames for ``compile_recipe``. With ``record``
    ``"arrays"`` the recipe is recorded for the update by whole games alone:
    its maps are small arrays, and what a seat was shown is dropped from its
    frames once it has been read, which bounds the memory of a long game. A
    seat whose recipe could not be recorded then gives no trajectory.

    ``model`` may be a pair, red's policy and blue's: a match between two
    checkpoints. A job without a learner gives no trajectory and is not a
    failed game for that; it ends as the engine ended it.
    """
    started = time.monotonic()
    models = tuple(model) if isinstance(model, (tuple, list)) else (model, model)
    seats = [Seat(models[player], seed, guard=guard, record=record) for player, seed in enumerate(job.policy_seeds)]
    def ahead():
        # While the engine works on an action, both seats read what they have received of it.
        for player, seat in enumerate(seats):
            seat.observe_ahead(env.received(player))

    env.idle = ahead
    try:
        if job.game_map is None:
            state = env.reset(directory, job.number, job.engine_seed)
        else:
            state = env.reset(directory, job.number, job.engine_seed, game_map=job.game_map)
        while isinstance(state, Request):
            seat = seats[state.player]
            # The request ends with its decision, and a frame's number is its position.
            seat.observe(env.received(state.player), state.frames[-1].seq + 1)
            seat.forget(env, state.player)
            if state.limit is not None:
                state = env.stop(state.limit)
                break
            state = env.step(seat.act())
            if on_progress is not None:
                on_progress(env.day, tuple(len(item.choices) for item in seats))
    finally:
        env.idle = None
    status, reason, detail = state.status, state.reason, state.detail
    built = {}  # seat -> its trajectory
    if job.learner is None:
        return Game(job, status, reason, state.outcomes, detail, env.day,
                    tuple(len(item.choices) for item in seats), time.monotonic() - started, None,
                    gold_mine_captures=env.gold_mine_captures)
    if status in ("completed", "truncated"):
        day_limit_draw = reason == "game_day_limit" and env.limits.max_days == TRAINING_DAY_LIMIT
        collect_both = both_seats or day_limit_draw
        unusable = []
        for seat in (job.learner, 1 - job.learner) if collect_both else (job.learner,):
            try:
                frames = env.frames(seat)
                built[seat] = build_trajectory(
                    frames, seats[seat], natural=status == "completed",
                    outcome=state.outcomes[seat] if state.outcomes else None, guard=guard,
                    day_limit_draw=day_limit_draw)
                if not seats[seat].recorded(frames, built[seat]) and record == "arrays":
                    del built[seat]
                    raise Unusable("its recipe could not be recorded while playing, and its frames were not kept")
            except Unusable as error:
                unusable.append(f"{COLORS[seat]}: {error}" if collect_both else str(error))
        if day_limit_draw and unusable:
            built.clear()
        if not built:
            # The game may have ended before the learner's seat had a turn: a game with a result and nothing
            # to learn from for that seat. Its outcome stays in the record; another game is played for the batch.
            idle = status == "completed" and not seats[job.learner].choices
            status, reason = "failed", "learner_never_acted" if idle else "unusable_trajectory"
        if unusable:
            detail = "; ".join(unusable)
    return Game(job, status, reason, state.outcomes, detail, env.day,
                tuple(len(item.choices) for item in seats), time.monotonic() - started,
                built.get(job.learner), other=built.get(1 - job.learner),
                gold_mine_captures=env.gold_mine_captures)


class _Played:
    """What ``build_trajectory`` reads of a seat, for a native seat's results."""

    def __init__(self, choices, values, repeats):
        self.choices, self.values, self._repeats = choices, values, repeats
        self.repeats = self

    def finish(self, frames):
        return self._repeats


def _play_native(env, model, job, directory, *, guard, on_progress=None, both_seats=False, record=False):
    """``_play`` with ``heroes.model.native``'s seats on a ``NativeEnv``: the same game, actions and trajectories.

    Each seat follows its producer's frames, answers with the native walk scored by ``FastPolicy``'s arithmetic
    and sampled with torch's generator arithmetic, and records the learner's recipe as ``Compact`` arrays, which
    is what the update by whole games reads (``record="arrays"``; the other recordings need Python frames).
    """
    from .model import native
    if record != "arrays":
        raise ValueError("Native seats record only for the update by whole games (record='arrays')")
    started = time.monotonic()
    models = tuple(model) if isinstance(model, (tuple, list)) else (model, model)
    policies = [fast.acting(item) for item in models]
    if not all(isinstance(policy, fast.FastPolicy) for policy in policies):
        raise RuntimeError("Native seats need fastnet.c's policy, which is not available for this model")
    seats = [None, None]

    def ahead():
        # While the engine works on an action, both seats read what they have received of it.
        for seat in seats:
            if seat is not None:
                seat.observe_ahead()

    env.idle = ahead
    try:
        if job.game_map is None:
            state = env.reset(directory, job.number, job.engine_seed)
        else:
            state = env.reset(directory, job.number, job.engine_seed, game_map=job.game_map)
        for player, seed in enumerate(job.policy_seeds):
            seats[player] = native.seat(policies[player], models[player], env.producer(player), seed, guard, True)
        native_loop = os.environ.get("HEROES_NATIVE_LOOP", "1") != "0"
        while isinstance(state, Request):
            seat = seats[state.player]
            if state.limit is not None:
                seat.observe(state.decision + 1)
                state = env.stop(state.limit)
                break
            # The game loop in native.cpp answers this request and the ones after it while nothing needs Python.
            following = env.play(seats, state) if native_loop else None
            if following is None:
                seat.observe(state.decision + 1)
                following = env.step(seat.act())
            state = following
            if on_progress is not None:
                on_progress(env.day, tuple(item.answered for item in seats))
    finally:
        env.idle = None
    choices = tuple(item.answered if item is not None else 0 for item in seats)
    status, reason, detail = state.status, state.reason, state.detail
    built = {}
    if job.learner is None:
        return Game(job, status, reason, state.outcomes, detail, env.day, choices, time.monotonic() - started, None,
                    gold_mine_captures=env.gold_mine_captures)
    if status in ("completed", "truncated"):
        day_limit_draw = reason == "game_day_limit" and env.limits.max_days == TRAINING_DAY_LIMIT
        collect_both = both_seats or day_limit_draw
        unusable = []
        module = native.module()
        for player in (job.learner, 1 - job.learner) if collect_both else (job.learner,):
            try:
                producer, seat = env.producer(player), seats[player]
                frames = [native.SampleFrame(*item) for item in module.sample_frames(producer)]
                played = _Played(*seat.results(), module.repeated_empty_opens(producer))
                trajectory = build_trajectory(frames, played, natural=status == "completed",
                                              outcome=state.outcomes[player] if state.outcomes else None, guard=guard,
                                              day_limit_draw=day_limit_draw)
                arrays = seat.finish(trajectory.decisions)
                if arrays is None:
                    raise Unusable("its recipe could not be recorded while playing, and its frames were not kept")
                trajectory.frames, trajectory.compact = None, native.compact_of(arrays)
                built[player] = trajectory
            except Unusable as error:
                unusable.append(f"{COLORS[player]}: {error}" if collect_both else str(error))
        if day_limit_draw and unusable:
            built.clear()
        if not built:
            idle = status == "completed" and not seats[job.learner].answered
            status, reason = "failed", "learner_never_acted" if idle else "unusable_trajectory"
        if unusable:
            detail = "; ".join(unusable)
    return Game(job, status, reason, state.outcomes, detail, env.day, choices, time.monotonic() - started,
                built.get(job.learner), other=built.get(1 - job.learner), gold_mine_captures=env.gold_mine_captures)
