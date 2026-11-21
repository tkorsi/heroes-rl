"""Play one self-play game and turn the learner's seat into a PPO trajectory.

Both seats are played by the same weights. Each seat has its own recurrent
state, its own view of the game and its own random generator; one of the two is
the learner whose experience is trained on, the other is its opponent. With
``both_seats`` the opponent's experience is trained on as well: the same game
then gives two trajectories, one that ends in a win and one that ends in a loss.

A game stopped at a limit has no winner. Its actions are scored with the
policy's own estimate of what would have followed, unless ``stopped_reward``
says what such a game is worth to a seat: -1 makes it a game both seats lost,
0 a draw.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import time

import torch

from .engine.env import COLORS, Request
from .learner.ppo import Trajectory
from .model.policy import ContextualEncoder


@dataclass(frozen=True)
class Job:
    number: int  # game number; also the engine's episode id
    engine_seed: int  # 1 .. 2**31 - 1
    learner: int | None  # 0 plays red, 1 plays blue; None when the game is only played, as in a match
    policy_seeds: tuple  # sampling seed of each seat
    # For a match: game days a seat remembers. Its recurrent state starts anew
    # when it ends its turn on every such day; None keeps it for the whole game.
    memory_days: tuple = (None, None)


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
    # Of a game without a learner: by seat, the value estimate at the seat's first
    # decision of every game day from day 1 on, None for a day without one.
    daily_values: tuple | None = None

    @property
    def trajectories(self):
        """What the game gives the update: the learner seat's trajectory, then the opponent's, each if there is one."""
        return tuple(item for item in (self.trajectory, self.other) if item is not None)


class Seat:
    """One player's policy during a game: encoder, recurrent state and sampling."""

    def __init__(self, model, seed, *, guard, memory_days=None):
        self.model = model
        self.encoder = ContextualEncoder(guard)
        self.state = model.initial_state()
        self.generator = torch.Generator().manual_seed(seed)
        self.values = {}  # decision frame -> value estimate
        self.choices = {}  # decision frame -> (action, log-probability)
        self.pending = None  # (decision frame, its action distribution)
        self.memory_days = memory_days  # see ``Job``
        self.day = 0  # game day of the frame observed last
        self.daily_values = {}  # game day -> value estimate at the first decision of that day

    def observe(self, frames):
        """Advance the recurrent state over the seat's new frames."""
        with torch.no_grad():
            for frame in frames:
                self.day = frame.day
                output = self.model.forward_step(self.encoder.encode(frame), self.state)
                self.state = output.state
                if output.decision is not None:
                    value = float(output.decision.value)
                    self.values[frame.seq] = value
                    self.daily_values.setdefault(frame.day, value)
                    self.pending = (frame.seq, output.decision.distribution)

    def act(self):
        """Sample the answer to the decision observed last."""
        seq, distribution = self.pending
        if seq not in self.choices:
            with torch.no_grad():
                action, evaluated = distribution.sample_evaluated(generator=self.generator)
            self.choices[seq] = (action, float(evaluated.log_prob))
            if self.memory_days and action.kind == "end_turn" and self.day % self.memory_days == 0:
                # The seat forgets here: what the opponent does next and its own next turn start a new memory.
                self.state = self.model.initial_state()
        return self.choices[seq][0]


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


def build_trajectory(frames, seat, *, natural, outcome, guard, stopped_reward=None):
    """Join the seat's resolved actions with what its policy recorded while playing.

    ``stopped_reward`` is what a game stopped at a limit is worth to the seat.
    None scores it with the value of the decision after the last action.
    """
    samples = learner_samples(frames, natural=natural)
    if not natural:
        # A stopped game ends at the last action whose following decision the
        # policy saw; that decision's value stands in for the rest of the game.
        # With a fixed reward the same actions are trained on, so the two ways
        # of scoring a stopped game differ in nothing else.
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
    if natural:
        reward, bootstrap = (1 if outcome == "win" else -1), 0.0
    elif stopped_reward is None:
        reward, bootstrap = 0, seat.values[samples[-1][1]]
    else:
        reward, bootstrap = stopped_reward, 0.0
    return Trajectory(
        decisions=decisions,
        log_probs=tuple(seat.choices[decision][1] for decision in decisions),
        values=tuple(seat.values[decision] for decision in decisions),
        reward=reward, bootstrap=bootstrap,
        actions=tuple(seat.choices[decision][0] for decision in decisions),
        frames=tuple(frames), guard=guard)


def compile_recipe(model, trajectory):
    """Replace the trajectory's frames with the compiled recipe the MLX learner reads."""
    from .learner.recipes import compile_history
    encoder = ContextualEncoder(trajectory.guard)
    trajectory.recipe = compile_history(model, (encoder.encode(frame) for frame in trajectory.frames),
                                        dict(zip(trajectory.decisions, trajectory.actions)))
    trajectory.frames = None
    return trajectory


def play(env, model, job, directory, *, guard, on_progress=None, both_seats=False, stopped_reward=None):
    """Play one game to its end and return it with the learner's trajectory.

    A game whose learner seat gives no trajectory is a failed game. With
    ``both_seats`` each seat's trajectory is built on its own and stays with
    its seat: the game counts as long as one of the two gives a trajectory, and
    ``detail`` then says why the other gave none.

    ``model`` is the policy of both seats, or a pair: red's policy and blue's.
    A job without a learner is played for its result alone and gives no
    trajectory. Only such a game may shorten a seat's memory: the update
    rebuilds every state from the first frame of the game.
    """
    started = time.monotonic()
    if job.learner is not None and any(job.memory_days):
        raise ValueError("A game played with a shortened memory cannot be trained on")
    policies = model if isinstance(model, tuple) else (model, model)
    seats = [Seat(policy, seed, guard=guard, memory_days=days)
             for policy, seed, days in zip(policies, job.policy_seeds, job.memory_days, strict=True)]
    state = env.reset(directory, job.number, job.engine_seed)
    while isinstance(state, Request):
        seat = seats[state.player]
        seat.observe(state.frames)
        if state.limit is not None:
            state = env.stop(state.limit)
            break
        state = env.step(seat.act())
        if on_progress is not None:
            on_progress(env.day, tuple(len(item.choices) for item in seats))
    status, reason, detail = state.status, state.reason, state.detail
    built = {}  # seat -> its trajectory
    trained = () if job.learner is None else (job.learner, 1 - job.learner) if both_seats else (job.learner,)
    if status in ("completed", "truncated") and trained:
        unusable = []
        for seat in trained:
            try:
                built[seat] = build_trajectory(
                    env.frames(seat), seats[seat], natural=status == "completed",
                    outcome=state.outcomes[seat] if state.outcomes else None, guard=guard,
                    stopped_reward=stopped_reward)
            except Unusable as error:
                unusable.append(f"{COLORS[seat]}: {error}" if both_seats else str(error))
        if not built:
            status, reason = "failed", "unusable_trajectory"
        if unusable:
            detail = "; ".join(unusable)
    daily = None if job.learner is not None else tuple(
        tuple(item.daily_values.get(day) for day in range(1, max(item.daily_values, default=0) + 1)) for item in seats)
    return Game(job, status, reason, state.outcomes, detail, env.day,
                tuple(len(item.choices) for item in seats), time.monotonic() - started,
                built.get(job.learner), other=built.get(trained[1]) if len(trained) == 2 else None,
                daily_values=daily)
