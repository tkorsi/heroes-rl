"""Recurrent PPO: trajectories, targets, loss windows and a torch reference update.

A trajectory is one game seen from the learner's seat. A loss window starts
from the recurrent state the learner's own weights give for everything before
it, and only the window itself is differentiated; hidden states stored during
play are never reused. With ``state_refresh_steps`` 1 that state is rebuilt
from the start of the game with the current weights at every optimizer step,
which costs time quadratic in the length of a game. A larger value replays a
game once, keeps the state before each of its windows and reuses them for that
many optimizer steps before replaying the game again, so a window can start
from a state that is at most that many steps old. The MLX backend in
``heroes.learner.mlx`` runs the same update on Apple GPUs. ``update`` below is
the slow, plain-autograd reference.
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
    # Windows of a minibatch that come from the same game and so share one replay
    # of it. 1 draws every window from all games alike; it must divide
    # windows_per_minibatch.
    windows_per_game: int = 1
    # Optimizer steps for which the states before a game's windows are reused
    # once the game was replayed. 1 replays the game at every step it is used.
    state_refresh_steps: int = 1

    def __post_init__(self):
        if not (0 <= self.gae_lambda <= 1 and 0 < self.clip_ratio < 1 and self.value_coefficient >= 0
                and self.max_grad_norm > 0 and self.learning_rate > 0
                and (self.value_clip is None or self.value_clip > 0)
                and type(self.state_refresh_steps) is int and type(self.windows_per_game) is int
                and min(self.epochs, self.decisions_per_window, self.windows_per_minibatch,
                        self.state_refresh_steps, self.windows_per_game) >= 1
                and self.windows_per_minibatch % self.windows_per_game == 0):
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

    def planned_steps(self, config):
        groups = sum(math.ceil(len(starts) / config.windows_per_game) for starts in self.starts)
        return config.epochs * math.ceil(groups * config.windows_per_game / config.windows_per_minibatch)

    def minibatches(self, config, generator):
        """One epoch: for every optimizer step, the indices of its windows in ``windows``.

        Every window is used once. With ``windows_per_game`` 1 the windows of
        all games are shuffled together. Otherwise each game's windows are
        shuffled and cut into groups of that size, the groups of all games are
        shuffled, and a minibatch takes whole groups. Its windows then come
        from few games, and one replay of a game serves all its windows in the
        step. A game's last group may be smaller.
        """
        size = config.windows_per_game
        if size == 1:
            order = torch.randperm(len(self.windows), generator=generator).tolist()
            return [order[left:left + config.windows_per_minibatch]
                    for left in range(0, len(order), config.windows_per_minibatch)]
        groups, first = [], 0
        for starts in self.starts:  # ``windows`` lists the games one after another
            order = torch.randperm(len(starts), generator=generator).tolist()
            groups.extend([first + index for index in order[left:left + size]]
                          for left in range(0, len(order), size))
            first += len(starts)
        order = torch.randperm(len(groups), generator=generator).tolist()
        count = config.windows_per_minibatch // size
        return [[window for group in order[left:left + count] for window in groups[group]]
                for left in range(0, len(order), count)]

    def kept(self, config):
        """Trajectories whose window states are kept for ``state_refresh_steps`` optimizer steps.

        Keeping pays once a game is expected to be used again before its
        states expire: replaying it to the end costs about twice the average
        replay up to one window. Every other game is replayed up to the window
        at each use, with the current weights.
        """
        if config.state_refresh_steps == 1:
            return frozenset()
        return frozenset(index for index, starts in enumerate(self.starts) if len(starts) > 1 and
                         config.state_refresh_steps * config.windows_per_minibatch * len(starts) >= len(self.windows))


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
    game_replays: int = 0  # whole-game replays that refreshed kept window states


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
    kept, saved = batch.kept(config), {}  # trajectory -> (step its states were computed at, the states)
    totals, count_seen, steps, replays = [0.0] * 4, 0, 0, 0
    for _ in range(config.epochs):
        for selected in batch.minibatches(config, generator):
            new_logs, new_values, old_logs, old_values, advantages, returns = [], [], [], [], [], []
            for window in (batch.windows[index] for index in selected):
                item, frames = trajectories[window.rollout_index], inputs[window.rollout_index]
                if window.rollout_index in kept:
                    if (window.rollout_index not in saved
                            or steps - saved[window.rollout_index][0] >= config.state_refresh_steps):
                        saved[window.rollout_index] = (steps, _states_before(
                            model, frames, batch.starts[window.rollout_index]))
                        replays += 1
                    state = saved[window.rollout_index][1][window.start_seq]
                else:
                    state = _states_before(model, frames, [window.start_seq])[window.start_seq]
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
                progress(steps, batch.planned_steps(config))
    return PPOReport(batch.samples, steps, *(value / count_seen for value in totals),
                     timings_seconds={"full_update": time.perf_counter() - started}, game_replays=replays)
