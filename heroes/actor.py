"""Play one self-play game and turn the learner's seat into a PPO trajectory.

Both seats are played by the same weights. Each seat has its own recurrent
state, its own view of the game and its own random generator; one of the two is
the learner whose experience is trained on, the other is its opponent. With
``both_seats`` the opponent's experience is trained on as well: the same game
then gives two trajectories, one that ends in a win and one that ends in a loss.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import time

import torch

from .engine.env import Request
from .learner.ppo import Trajectory
from .model.policy import ContextualEncoder


@dataclass(frozen=True)
class Job:
    number: int  # game number; also the engine's episode id
    engine_seed: int  # 1 .. 2**31 - 1
    learner: int  # 0 plays red, 1 plays blue
    policy_seeds: tuple  # sampling seed of each seat


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
    trajectory: Trajectory | None  # None when the game cannot be trained on
    compile_seconds: float = 0.0  # time spent compiling the trajectory's recipe
    other: Trajectory | None = None  # the opponent seat's trajectory, when both seats are trained on

    @property
    def trajectories(self):
        return tuple(item for item in (self.trajectory, self.other) if item is not None)


class Seat:
    """One player's policy during a game: encoder, recurrent state and sampling."""

    def __init__(self, model, seed, *, guard):
        self.model = model
        self.encoder = ContextualEncoder(guard)
        self.state = model.initial_state()
        self.generator = torch.Generator().manual_seed(seed)
        self.values = {}  # decision frame -> value estimate
        self.choices = {}  # decision frame -> (action, log-probability)
        self.pending = None  # (decision frame, its action distribution)

    def observe(self, frames):
        """Advance the recurrent state over the seat's new frames."""
        with torch.no_grad():
            for frame in frames:
                output = self.model.forward_step(self.encoder.encode(frame), self.state)
                self.state = output.state
                if output.decision is not None:
                    self.values[frame.seq] = float(output.decision.value)
                    self.pending = (frame.seq, output.decision.distribution)

    def act(self):
        """Sample the answer to the decision observed last."""
        seq, distribution = self.pending
        if seq not in self.choices:
            with torch.no_grad():
                action, evaluated = distribution.sample_evaluated(generator=self.generator)
            self.choices[seq] = (action, float(evaluated.log_prob))
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


def build_trajectory(frames, seat, *, natural, outcome, guard):
    """Join the seat's resolved actions with what its policy recorded while playing."""
    samples = learner_samples(frames, natural=natural)
    if not natural:
        # A stopped game ends at the last action whose following decision the
        # policy saw; that decision's value stands in for the rest of the game.
        while samples and samples[-1][1] not in seat.values:
            samples.pop()
    if not samples:
        raise Unusable("the learner completed no action")
    for (_, successor), (following, _) in zip(samples, samples[1:]):
        if successor != following:
            raise Unusable("the learner's actions do not form one unbroken sequence")
    decisions = tuple(decision for decision, _ in samples)
    if any(decision not in seat.choices for decision in decisions):
        raise Unusable("an action in the history was not sampled by this policy")
    return Trajectory(
        decisions=decisions,
        log_probs=tuple(seat.choices[decision][1] for decision in decisions),
        values=tuple(seat.values[decision] for decision in decisions),
        reward=(1 if outcome == "win" else -1) if natural else 0,
        bootstrap=0.0 if natural else seat.values[samples[-1][1]],
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


def play(env, model, job, directory, *, guard, on_progress=None, both_seats=False):
    """Play one game to its end and return it with the learner's trajectory.

    With ``both_seats`` the game also carries the opponent seat's trajectory,
    unless that seat's history gives none; the game then counts with one.
    """
    started = time.monotonic()
    seats = [Seat(model, seed, guard=guard) for seed in job.policy_seeds]
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
    trajectory, other, status, reason, detail = None, None, state.status, state.reason, state.detail

    def of(seat):
        return build_trajectory(env.frames(seat), seats[seat], natural=status == "completed",
                                outcome=state.outcomes[seat] if state.outcomes else None, guard=guard)

    if status in ("completed", "truncated"):
        try:
            trajectory = of(job.learner)
        except Unusable as error:
            status, reason, detail = "failed", "unusable_trajectory", str(error)
        else:
            if both_seats:
                try:
                    other = of(1 - job.learner)
                except Unusable:
                    pass
    return Game(job, status, reason, state.outcomes, detail, env.day,
                tuple(len(item.choices) for item in seats), time.monotonic() - started, trajectory, other=other)
