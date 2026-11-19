"""Recurrent PPO: trajectories, targets, loss windows and a torch reference update.

A trajectory is one game seen from the learner's seat. A loss window starts
from the recurrent state the learner's own weights give for everything before
it, and only the window itself is differentiated; hidden states stored during
play are never reused. That state is rebuilt by replaying the game from its
first frame, which costs time quadratic in the length of a game.

An update is a sequence of rounds. A round replays each of its games once,
with the weights it starts from, takes the state before each of its windows
from that replay, and then makes its optimizer steps of
``windows_per_minibatch`` windows each. With ``windows_per_replay`` 1 a round
is a single step, so every state is computed with the weights the step
differentiates. A larger value puts that many windows of a game into a round:
one replay serves all of them, the steps keep their size and number, and a
window starts from a state that is at most ``windows_per_replay - 1`` steps
old. The MLX backend in ``heroes.learner.mlx`` runs the same update on Apple
GPUs. ``update`` below is the slow, plain-autograd reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time

import torch


@dataclass(frozen=True)
class PPOConfig:
    gae_lambda: float = 1.0
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 1
    decisions_per_window: int = 16
    windows_per_minibatch: int = 2
    normalize_advantages: bool = True
    learning_rate: float = 3e-4
    # Windows of a game that one replay of it serves. 1 replays a game for every
    # window, with the weights of the very step. More share the replay among the
    # steps of a round, whose states are then up to windows_per_replay - 1 steps old.
    windows_per_replay: int = 1

    def __post_init__(self):
        if not (0 <= self.gae_lambda <= 1 and 0 < self.clip_ratio < 1 and self.value_coefficient >= 0
                and self.max_grad_norm > 0 and self.learning_rate > 0
                and (self.value_clip is None or self.value_clip > 0)
                and type(self.windows_per_replay) is int
                and min(self.epochs, self.decisions_per_window, self.windows_per_minibatch,
                        self.windows_per_replay) >= 1):
            raise ValueError("Invalid PPO configuration")

    def to_dict(self):
        return asdict(self)


@dataclass
class Trajectory:
    """One finished game from the learner's seat.

    Rewards are undiscounted and terminal only: every action scores zero except
    the last one of a naturally finished game, which scores +1 or -1. A game
    stopped at a limit instead bootstraps from the value of the learner's next
    decision.
    """
    decisions: tuple[int, ...]  # frame index of each trained decision, ascending
    log_probs: tuple[float, ...]  # behaviour log-probability of the action taken there
    values: tuple[float, ...]  # behaviour value estimate there
    reward: float  # +1 or -1 after the last action of a natural game, else 0
    bootstrap: float  # value after the last action: 0 for a natural game
    actions: tuple = ()  # the actions taken, aligned with ``decisions``
    frames: tuple | None = None  # the seat's typed frames (torch reference backend)
    guard: str | None = None  # interaction-guard version the behaviour policy used
    recipe: object = None  # compiled ``heroes.learner.plan.History`` (MLX backend)

    def __post_init__(self):
        count = len(self.decisions)
        if not count or len(self.log_probs) != count or len(self.values) != count:
            raise ValueError("A trajectory needs aligned, nonempty decisions")
        if list(self.decisions) != sorted(set(self.decisions)):
            raise ValueError("Trajectory decisions must be strictly ascending")
        if self.reward not in (-1, 0, 1) or (self.reward and self.bootstrap):
            raise ValueError("A trajectory ends with a terminal reward or a bootstrap value, not both")


@dataclass(frozen=True)
class AdvantageTargets:
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


def gae(rewards, values, successor_values, *, gae_lambda=1.0):
    """Undiscounted generalized advantage estimation for one trajectory."""
    if not rewards or len(rewards) != len(values) or len(values) != len(successor_values):
        raise ValueError("GAE requires aligned nonempty vectors")
    if any(not math.isfinite(value) for vector in (rewards, values, successor_values) for value in vector):
        raise ValueError("GAE inputs must be finite")
    result = [0.0] * len(rewards)
    following = 0.0
    for index in reversed(range(len(rewards))):
        following = rewards[index] + successor_values[index] - values[index] + gae_lambda * following
        result[index] = following
    return AdvantageTargets(tuple(result), tuple(a + v for a, v in zip(result, values)))


@dataclass(frozen=True)
class LossWindow:
    rollout_index: int
    sample_indices: tuple[int, ...]
    start_seq: int  # first differentiated frame; everything before it is a no-gradient prefix
    end_seq: int  # the window's last decision frame


@dataclass
class Batch:
    """Targets and loss windows of one update, shared by both backends."""
    targets: list
    advantages: list  # flat over all trajectories, normalized when configured
    offsets: list  # first flat index of each trajectory
    windows: list
    samples: int
    starts: list = field(default_factory=list)  # per trajectory, the first frame of each of its windows, ascending

    def rounds(self, config, generator):
        """One epoch as rounds; a round is a list of minibatches, each the indices of its windows in ``windows``.

        Every window is used once. The states the windows of a round start
        from are computed together, before its first step. With
        ``windows_per_replay`` 1 the windows of all games are shuffled together
        and every minibatch is a round of its own. Otherwise each game's
        windows are shuffled and cut into groups of that size, the groups of
        all games are shuffled, and a round takes ``windows_per_minibatch`` of
        them, shuffles their windows together and cuts them into minibatches.
        A full round then makes ``windows_per_replay`` steps on windows of at
        most ``windows_per_minibatch`` games. A game's last group may be
        smaller, and so may the round and the last minibatch it is in.
        """
        size, step = config.windows_per_replay, config.windows_per_minibatch

        def shuffled(count):
            return torch.randperm(count, generator=generator).tolist()

        if size == 1:
            order = shuffled(len(self.windows))
            return [[order[left:left + step]] for left in range(0, len(order), step)]
        groups, first = [], 0
        for starts in self.starts:  # ``windows`` lists the games one after another
            order = shuffled(len(starts))
            groups.extend([first + index for index in order[left:left + size]]
                          for left in range(0, len(order), size))
            first += len(starts)
        rounds, order = [], shuffled(len(groups))
        for left in range(0, len(order), step):
            pool = [window for group in order[left:left + step] for window in groups[group]]
            pool = [pool[index] for index in shuffled(len(pool))]
            rounds.append([pool[at:at + step] for at in range(0, len(pool), step)])
        return rounds

    def schedule(self, config, generator):
        """The rounds of every epoch, and the number of optimizer steps they hold."""
        epochs = [self.rounds(config, generator) for _ in range(config.epochs)]
        return epochs, sum(len(minibatches) for rounds in epochs for minibatches in rounds)


def prepare(trajectories, config):
    targets = [gae((0.0,) * (len(item.values) - 1) + (float(item.reward),), item.values,
                   item.values[1:] + (float(item.bootstrap),), gae_lambda=config.gae_lambda)
               for item in trajectories]
    advantages = [value for target in targets for value in target.advantages]
    if config.normalize_advantages:
        mean = sum(advantages) / len(advantages)
        scale = math.sqrt(sum((value - mean) ** 2 for value in advantages) / len(advantages)) + 1e-8
        advantages = [(value - mean) / scale for value in advantages]
    offsets, offset, windows, starts = [], 0, [], []
    for index, item in enumerate(trajectories):
        offsets.append(offset)
        offset += len(item.decisions)
        starts.append([])
        for start in range(0, len(item.decisions), config.decisions_per_window):
            indices = tuple(range(start, min(start + config.decisions_per_window, len(item.decisions))))
            # The window's graph includes every frame since the preceding trained decision.
            begin = 0 if start == 0 else item.decisions[start - 1] + 1
            windows.append(LossWindow(index, indices, begin, item.decisions[indices[-1]]))
            starts[-1].append(begin)
    return Batch(targets, advantages, offsets, windows, offset, starts)


def clipped_losses(log_probs, values, old_log_probs, old_values, advantages, returns, config):
    """PPO-clip policy loss and clipped value loss over the trained decisions."""
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all().item():
        raise ValueError("Nonfinite PPO importance ratio")
    policy_loss = -torch.minimum(ratio * advantages,
                                ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages).mean()
    squared = (values - returns).square()
    if config.value_clip is not None:
        clipped = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
        squared = torch.maximum(squared, (clipped - returns).square())
    value_loss = 0.5 * squared.mean()
    total = policy_loss + config.value_coefficient * value_loss
    return total, policy_loss, value_loss, ((ratio - 1) - log_ratio).mean(), ((ratio - 1).abs() > config.clip_ratio).float().mean()


@dataclass(frozen=True)
class PPOReport:
    samples: int
    optimizer_steps: int
    policy_loss: float
    value_loss: float
    approximate_kl: float
    clip_fraction: float
    timings_seconds: dict = field(default_factory=dict)


def _states_before(model, frames, starts):
    """One pass over a game without gradients: the state before each frame in ``starts`` (ascending)."""
    states, state, seq = {}, model.initial_state(), 0
    with torch.no_grad():
        for start in starts:
            while seq < start:
                state = model.forward_step(frames[seq], state).state
                seq += 1
            states[start] = state
    return states


def update(model, optimizer, trajectories, *, config, generator, progress=None):
    """Reference update in plain torch autograd. Correct but slow: it replays
    games frame by frame. Production training uses MLX."""
    from ..model.policy import encode_history
    started = time.perf_counter()
    batch = prepare(trajectories, config)
    inputs = [encode_history(item.frames, guard=item.guard) for item in trajectories]
    epochs, planned = batch.schedule(config, generator)
    totals, count_seen, steps = [0.0] * 4, 0, 0
    for rounds in epochs:
        for minibatches in rounds:
            # One replay of each game of the round, with the weights the round starts from.
            stops = {}
            for window in (batch.windows[index] for selected in minibatches for index in selected):
                stops.setdefault(window.rollout_index, set()).add(window.start_seq)
            states = {index: _states_before(model, inputs[index], sorted(starts)) for index, starts in stops.items()}
            for selected in minibatches:
                new_logs, new_values, old_logs, old_values, advantages, returns = [], [], [], [], [], []
                for window in (batch.windows[index] for index in selected):
                    item, frames = trajectories[window.rollout_index], inputs[window.rollout_index]
                    state = states[window.rollout_index][window.start_seq]
                    wanted = {item.decisions[index]: index for index in window.sample_indices}
                    for seq in range(window.start_seq, window.end_seq + 1):
                        output = model.forward_step(frames[seq], state)
                        state = output.state
                        if seq in wanted:
                            index = wanted[seq]
                            new_logs.append(output.decision.distribution.log_prob(item.actions[index]))
                            new_values.append(output.decision.value)
                            old_logs.append(item.log_probs[index])
                            old_values.append(item.values[index])
                            advantages.append(batch.advantages[batch.offsets[window.rollout_index] + index])
                            returns.append(batch.targets[window.rollout_index].returns[index])
                log_probs, values = torch.stack(new_logs), torch.stack(new_values)
                make = log_probs.new_tensor
                loss, *metrics = clipped_losses(log_probs, values, make(old_logs), make(old_values),
                                                make(advantages), make(returns), config)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                for index, metric in enumerate(metrics):
                    totals[index] += metric.detach().item() * len(old_logs)
                count_seen += len(old_logs)
                steps += 1
                if progress is not None:
                    progress(steps, planned)
    return PPOReport(batch.samples, steps, *(value / count_seen for value in totals),
                     timings_seconds={"full_update": time.perf_counter() - started})
