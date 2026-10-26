"""Small recurrent PPO implementation with synthetic-only optimizer admission.

Every minibatch reconstructs recurrent state with the *current* parameters from
the complete player prefix. Only the loss window participates in backpropagation;
stored behavior hidden tensors are never used. The public entry point admits
synthetic tests only; a separate approval-bound runner verifies native replay
before using the shared numerical kernel.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math
from time import perf_counter

import torch


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    epochs: int = 2
    decisions_per_window: int = 16
    windows_per_minibatch: int = 2
    normalize_advantages: bool = True

    def __post_init__(self):
        for name in ("gamma", "gae_lambda", "clip_ratio", "value_coefficient", "entropy_coefficient", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("PPO coefficients must be finite numbers")
        if (self.gamma != 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_ratio < 1
                or self.value_coefficient < 0 or self.max_grad_norm <= 0 or self.entropy_coefficient != 0):
            raise ValueError("PPO requires gamma1, valid clipping and disabled hierarchical entropy")
        if self.value_clip is not None and (type(self.value_clip) not in (int, float)
                or not math.isfinite(self.value_clip) or self.value_clip <= 0):
            raise ValueError("Value clipping must be positive or disabled")
        for value in (self.epochs, self.decisions_per_window, self.windows_per_minibatch):
            if type(value) is not int or value < 1:
                raise ValueError("PPO iteration/window sizes must be positive integers")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("Advantage normalization must be explicit")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AdvantageTargets:
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


def gae(rewards, values, successor_values, *, gae_lambda=1.0):
    """Undiscounted same-player GAE; caller has verified the final boundary.

    Natural final successor value is zero; a verified cutoff supplies its own
    held-decision behavior value. There is no recurrence beyond either boundary.
    """
    if (not rewards or len(rewards) != len(values) or len(values) != len(successor_values)
            or type(gae_lambda) not in (int, float) or not 0 <= gae_lambda <= 1):
        raise ValueError("GAE requires aligned nonempty vectors and lambda in [0,1]")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for vector in (rewards, values, successor_values) for value in vector):
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
    start_seq: int
    end_seq: int


@dataclass(frozen=True)
class EvaluatedDecision:
    decision_seq: int
    log_prob: torch.Tensor
    value: torch.Tensor


def _check_window(inputs, decisions, start_seq):
    if not decisions or type(start_seq) is not int or not 0 <= start_seq <= decisions[0].decision_seq:
        raise ValueError("Invalid recurrent loss-window boundary")
    sequences = tuple(record.decision_seq for record in decisions)
    if sequences != tuple(sorted(set(sequences))) or sequences[-1] >= len(inputs):
        raise ValueError("Loss decisions must be ordered within the complete input prefix")


def evaluate_window(model, inputs, decisions, *, start_seq):
    """Replay a complete current-weight prefix, then a differentiable window.

    Ragged sequences avoid padding. ``decisions`` explicitly supplies the loss
    mask; every intervening frame is still forwarded, including other decisions.
    ``start_seq`` denotes the first differentiable frame, not a memory reset.
    """
    _check_window(inputs, decisions, start_seq)
    state = model.prefix_states(inputs, (start_seq,))[start_seq]
    return _evaluate_from_state(model, inputs, decisions, start_seq, state)


def _evaluate_from_state(model, inputs, decisions, start_seq, state):
    # Only the caller's current minibatch may supply this detached state. It is
    # never persisted in a rollout or reused after optimizer.step().
    state = state.detach()
    wanted = {record.decision_seq: record for record in decisions}
    result = []
    for seq in range(start_seq, decisions[-1].decision_seq + 1):
        output = model.forward_step(inputs[seq], state)
        state = output.state
        if seq in wanted:
            record = wanted[seq]
            if output.decision is None or output.decision.distribution.support != record.support:
                raise ValueError("Current policy support differs from the recorded behavior support")
            result.append(EvaluatedDecision(seq, output.decision.distribution.log_prob(record.action), output.decision.value))
    return tuple(result)


def clipped_losses(log_probs, values, old_log_probs, old_values, advantages, returns, config):
    """All vectors contain only loss-enabled samples; no padded values enter means."""
    vectors = (log_probs, values, old_log_probs, old_values, advantages, returns)
    if any(value.ndim != 1 or value.shape != log_probs.shape for value in vectors) or log_probs.numel() == 0:
        raise ValueError("PPO loss requires aligned nonempty vectors")
    if any(not torch.isfinite(value).all().item() for value in vectors):
        raise ValueError("Nonfinite PPO input")
    old_log_probs, old_values, advantages, returns = (value.detach() for value in (old_log_probs, old_values, advantages, returns))
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


def _admit(rollouts, model, *, native=False):
    from .contextual_model import weights_sha256
    from .contextual_rollout import ContextualRollout
    if type(rollouts) is not tuple or not rollouts or any(type(item) is not ContextualRollout for item in rollouts):
        raise ValueError("PPO requires immutable typed contextual rollouts")
    current = weights_sha256(model)
    identities = set()
    for item in rollouts:
        purpose, origin = ("training", "native") if native else ("synthetic", "synthetic")
        if (item.purpose != purpose or item.origin != origin or item.role != "learner"
                or not item.optimization_eligible or item.boundary not in ("natural", "cutoff")):
            raise ValueError("Optimizer updates admit only eligible " + purpose + " learner trajectories")
        if not item.samples or any(not sample.eligible for sample in item.samples):
            raise ValueError("PPO cannot skip an ineligible sample or bridge a trajectory gap")
        identities.add((item.behavior_checkpoint, item.weights_sha256))
        if item.weights_sha256 != current:
            raise ValueError("Each PPO update must start from its frozen behavior weights")
        for index, sample in enumerate(item.samples):
            if sample.reward not in (-1, 0, 1) or (sample.bootstrap_value is not None and type(sample.bootstrap_value) not in (float, int)):
                raise ValueError("Unscored transitions cannot enter PPO")
            if index + 1 < len(item.samples):
                next_decision = item.samples[index + 1].decision
                if (sample.reward != 0 or sample.successor_seq != next_decision.decision_seq
                        or sample.bootstrap_value != next_decision.value):
                    raise ValueError("PPO successors must be the next same-player sample")
            elif item.boundary == "natural":
                if sample.reward not in (-1, 1) or sample.successor_seq is not None or sample.bootstrap_value is not None:
                    raise ValueError("Natural PPO boundary requires terminal-only reward and zero bootstrap")
            elif sample.reward != 0 or sample.successor_seq is None or sample.bootstrap_value is None:
                raise ValueError("PPO cutoff requires a real own-decision bootstrap")
    if len(identities) != 1:
        raise ValueError("One PPO update cannot mix behavior checkpoints")
    if len({item.opponent_pool for item in rollouts}) != 1:
        raise ValueError("One PPO update must retain its fixed opponent-pool distribution")


def _verify_behavior(model, rollouts, inputs):
    """Recompute frozen old probabilities/values before the first parameter update.

    The modules ``_can_cache_prefix`` admits behave identically in training and
    evaluation mode, so the check then runs them in evaluation mode, where the
    exact inference memos apply, and restores every module's mode afterwards.
    It starts from empty memos and discards them afterwards: nothing the
    collector or the policy reconstruction cached is reused, so the check stays
    an independent recomputation of every stored probability and value.
    """
    modes = [(module, module.training) for module in model.modules()] if model._can_cache_prefix() else None
    if modes is not None:
        model.eval()
    try:
        _drop_inference_memos(model)
        _verify_behavior_frames(model, rollouts, inputs)
    finally:
        _drop_inference_memos(model)
        for module, mode in modes or ():
            module.training = mode


def _drop_inference_memos(model):
    for name in ("_leaf_memo_state", "_observation_memo_state"):
        model.__dict__.pop(name, None)


def _verify_behavior_frames(model, rollouts, inputs):
    with torch.no_grad():
        for item, encoded in zip(rollouts, inputs):
            records = {record.decision_seq: record for record in item.decisions}
            needed_values = {sample.successor_seq: sample.bootstrap_value for sample in item.samples
                             if sample.successor_seq is not None}
            state = model.initial_state()
            for seq, frame in enumerate(encoded):
                output = model.forward_step(frame, state)
                state = output.state
                if seq in records:
                    old = records[seq]
                    if output.decision is None or output.decision.distribution.support != old.support:
                        raise ValueError("Behavior support cannot be reconstructed")
                    evaluated = output.decision.distribution.evaluate(old.action)
                    log_prob = evaluated.log_prob.item()
                    if not math.isclose(log_prob, old.log_prob, rel_tol=2e-5, abs_tol=2e-5):
                        raise ValueError("Stored behavior probability differs from frozen policy")
                    if len(evaluated.conditionals) != len(old.conditionals) or any(
                            (actual.name, actual.selected_index, actual.support_size) !=
                            (saved.name, saved.selected_index, saved.support_size) or not math.isclose(
                                actual.log_prob.item(), saved.log_prob, rel_tol=2e-5, abs_tol=2e-5)
                            for actual, saved in zip(evaluated.conditionals, old.conditionals)):
                        raise ValueError("Stored conditional probability/support differs from frozen policy")
                    if not math.isclose(output.decision.value.item(), old.value, rel_tol=2e-5, abs_tol=2e-5):
                        raise ValueError("Stored behavior value differs from frozen critic")
                if seq in needed_values and (output.decision is None or not math.isclose(
                        output.decision.value.item(), needed_values[seq], rel_tol=2e-5, abs_tol=2e-5)):
                    raise ValueError("Bootstrap value differs from the actual own decision under behavior weights")


# Separate numerical experiments, never defaults. Only a caller that binds the choice into
# its immutable plan (the benchmark harness) sets this; no environment variable can.
EXPERIMENTS = frozenset()
# Exact, opt-in: forked helpers compute prefix frame encodings bit for bit (``prefix_pool``).
# Zero keeps every encoding in this process. Set the same way as ``EXPERIMENTS``.
PREFIX_HELPERS = 0


@dataclass(frozen=True)
class PPOReport:
    samples: int
    optimizer_steps: int
    policy_loss: float
    value_loss: float
    approximate_kl: float
    clip_fraction: float
    entropy: None = None
    purpose: str = "synthetic_optimizer_test"
    # Wall-clock diagnostics do not affect deterministic report equivalence.
    timings_seconds: dict[str, float] | None = field(default=None, compare=False)


def ppo_update(model, optimizer, rollouts, *, config=PPOConfig(), generator):
    """Run synthetic optimizer validation. Native/diagnostic data always rejects."""
    _admit(rollouts, model)
    return _update(model, optimizer, rollouts, config=config, generator=generator)


def _update(model, optimizer, rollouts, **options):
    pools = []
    try:
        return _update_kernel(model, optimizer, rollouts, pools=pools, **options)
    finally:
        # Prefix helpers end with the update, whatever its outcome.
        for pool in pools:
            pool.close()
        # The experiment's shared encodings never outlive their minibatch graph.
        model.__dict__.pop("_graph_memo_state", None)


def _update_kernel(model, optimizer, rollouts, *, config, generator, before_step=None, after_step=None, max_optimizer_steps=None,
                   pools=None):
    """Shared numerical kernel; callers must establish admission before entering.

    The public default entry point remains synthetic-only. The separately gated
    gameplay runner verifies native artifacts and replay before using this kernel.
    """
    from .contextual_model import encode_history, frame_keys_until
    started = perf_counter()
    timings = dict(history_encoding=0.0, behavior_verification=0.0, prefix_reconstruction=0.0,
                   loss_forward=0.0, backward_and_clip=0.0, optimizer_step=0.0)
    if type(config) is not PPOConfig or type(generator) is not torch.Generator or generator.device.type != "cpu":
        raise ValueError("Expected explicit PPO config and CPU minibatch RNG")
    if max_optimizer_steps is not None and (type(max_optimizer_steps) is not int or max_optimizer_steps < 1):
        raise ValueError("Optimizer-step cap must be a positive integer")
    owned = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(optimized) != owned or len(optimized) != len(owned):
        raise ValueError("Optimizer must own exactly the current trainable model parameters")
    phase = perf_counter()
    inputs = tuple(encode_history(item.history) for item in rollouts)
    timings["history_encoding"] = perf_counter() - phase
    phase = perf_counter()
    _verify_behavior(model, rollouts, inputs)
    timings["behavior_verification"] = perf_counter() - phase
    targets = tuple(gae(tuple(s.reward for s in item.samples), tuple(s.decision.value for s in item.samples),
                        tuple(0.0 if s.bootstrap_value is None else s.bootstrap_value for s in item.samples),
                        gae_lambda=config.gae_lambda) for item in rollouts)
    advantages = [value for target in targets for value in target.advantages]
    if config.normalize_advantages:
        mean = sum(advantages) / len(advantages)
        scale = math.sqrt(sum((value - mean) ** 2 for value in advantages) / len(advantages)) + 1e-8
        advantages = [(value - mean) / scale for value in advantages]
    offsets, offset = [], 0
    windows = []
    for rollout_index, item in enumerate(rollouts):
        offsets.append(offset)
        offset += len(item.samples)
        for start in range(0, len(item.samples), config.decisions_per_window):
            indices = tuple(range(start, min(start + config.decisions_per_window, len(item.samples))))
            # Include all observations/results since the preceding decision in
            # this window's graph; detach only before that history interval.
            begin = 0 if start == 0 else item.samples[start - 1].decision.decision_seq + 1
            windows.append(LossWindow(rollout_index, indices, begin, item.samples[indices[-1]].decision.decision_seq))
    planned_steps = config.epochs * math.ceil(len(windows) / config.windows_per_minibatch)
    if max_optimizer_steps is not None and planned_steps > max_optimizer_steps:
        raise ValueError("Complete PPO update exceeds its predeclared optimizer-step cap")
    totals = [0.0] * 4
    weighted_count = steps = 0
    # Separate numerical experiment, never a default: exact encodings shared within each
    # minibatch graph change parameter gradients in floating-point summation order only.
    shared = "shared_graph_encodings" in EXPERIMENTS
    if shared:
        timings["experiment_shared_graph_encodings"] = 1.0
    # Memo keys of the unchanged histories, computed once per update instead of once per minibatch.
    # A caller's own replacement of prefix_states keeps the original call.
    prefix_keys = None if "prefix_states" in model.__dict__ else [[] for _ in inputs]
    pool = None
    if PREFIX_HELPERS and pools is not None and prefix_keys is not None and model._can_cache_prefix():
        from .prefix_pool import PrefixPool
        phase = perf_counter()
        pool = PrefixPool(model, inputs, PREFIX_HELPERS)
        pools.append(pool)
        timings["prefix_helpers"] = float(PREFIX_HELPERS)
        timings["prefix_helper_start"] = perf_counter() - phase
    for _ in range(config.epochs):
        order = torch.randperm(len(windows), generator=generator).tolist()
        for batch_start in range(0, len(order), config.windows_per_minibatch):
            selected = [windows[index] for index in order[batch_start:batch_start + config.windows_per_minibatch]]
            # All selected windows see identical parameters. Reconstruct each
            # episode once up to its furthest boundary, then retain the original
            # selected-window order and separate detached autograd graphs.
            sharing = model.shared_graph_encodings() if shared else nullcontext()
            sharing.__enter__()
            phase = perf_counter()
            boundaries = {}
            if model._can_cache_prefix():
                for window in selected:
                    boundaries.setdefault(window.rollout_index, []).append(window.start_seq)
            prefixes = {}
            for index, stops in boundaries.items():
                if prefix_keys is None:
                    prefixes[index] = model.prefix_states(inputs[index], stops)
                    continue
                keys = frame_keys_until(prefix_keys[index], inputs[index], max(stops))
                vectors = None if pool is None else pool.vectors(index, max(stops), keys)
                prefixes[index] = model.prefix_states(inputs[index], stops, vectors=vectors, keys=keys)
            timings["prefix_reconstruction"] += perf_counter() - phase
            phase = perf_counter()
            new_log_probs, new_values, old_log_probs, old_values, batch_advantages, returns = [], [], [], [], [], []
            for window in selected:
                item = rollouts[window.rollout_index]
                records = tuple(item.samples[index].decision for index in window.sample_indices)
                _check_window(inputs[window.rollout_index], records, window.start_seq)
                if not prefixes:
                    # Hooks/custom or stochastic modules retain the original
                    # prefix/window execution order, without shared replay.
                    timings["loss_forward"] += perf_counter() - phase
                    phase = perf_counter()
                    state = model.prefix_states(inputs[window.rollout_index], (window.start_seq,))[window.start_seq]
                    timings["prefix_reconstruction"] += perf_counter() - phase
                    phase = perf_counter()
                else:
                    state = prefixes[window.rollout_index][window.start_seq]
                evaluated = _evaluate_from_state(model, inputs[window.rollout_index], records, window.start_seq,
                                                 state)
                for index, current, old in zip(window.sample_indices, evaluated, records):
                    new_log_probs.append(current.log_prob)
                    new_values.append(current.value)
                    old_log_probs.append(old.log_prob)
                    old_values.append(old.value)
                    batch_advantages.append(advantages[offsets[window.rollout_index] + index])
                    returns.append(targets[window.rollout_index].returns[index])
            log_probs = torch.stack(new_log_probs)
            values = torch.stack(new_values)
            make = log_probs.new_tensor
            loss, policy, value, kl, clipped = clipped_losses(log_probs, values, make(old_log_probs), make(old_values),
                                                           make(batch_advantages), make(returns), config)
            # Discard the minibatch's state lookup before parameters change.
            del prefixes, state
            timings["loss_forward"] += perf_counter() - phase
            if before_step is not None:
                before_step()
            phase = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            timings["backward_and_clip"] += perf_counter() - phase
            sharing.__exit__(None, None, None)
            if before_step is not None:
                before_step()
            phase = perf_counter()
            optimizer.step()
            timings["optimizer_step"] += perf_counter() - phase
            if after_step is not None:
                after_step()
            count = len(new_log_probs)
            for index, metric in enumerate((policy, value, kl, clipped)):
                totals[index] += metric.detach().item() * count
            weighted_count += count
            steps += 1
    timings["total"] = perf_counter() - started
    return PPOReport(sum(len(item.samples) for item in rollouts), steps, *(value / weighted_count for value in totals),
                     timings_seconds=timings)
