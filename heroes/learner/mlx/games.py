"""One PPO update on Apple GPUs in which every optimizer step uses whole games.

A step takes all loss windows of its games. For each of those games one pass
without gradients gives the recurrent state before every frame with the
current weights. The windows are then differentiated from those states, a few
hundred at a time so the GPU memory stays bounded, and their gradients are
summed before the one Adam step. Nothing computed with earlier weights is ever
used, as in ``update.py``, but a game is replayed once per step it is part of
instead of once per group of windows: an epoch costs about three passes over
the batch, whatever the length of its games.

Plans come from ``heroes.learner.compact``: the recipe as arrays, planned a
segment of consecutive windows at a time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time

SEGMENT_WINDOWS = 256  # consecutive loss windows of a game that are differentiated in one pass
# Plans kept in memory between passes; beyond it they are built again, on the helper thread while the GPU works,
# so a smaller budget costs memory nothing and time next to nothing. HEROES_HOST_PLAN_GB sets it for a run that
# shares the machine: an update on 9 million samples holds 66 GB with the default and about 40 GB with 0.
HOST_PLAN_BYTES = int(float(os.environ.get("HEROES_HOST_PLAN_GB", "24")) * 1024**3)


def _bytes(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes(item) for item in value)
    return 0


def segments(compact, starts, ends, firsts, names, size):
    """The weight-independent part of a game's loss windows, ``size`` consecutive windows at a time.

    Window ``k`` is the frames ``[starts[k], ends[k])`` and trains the game's
    decisions ``[firsts[k], firsts[k + 1])``; the windows follow one another
    without a gap. Each segment holds the plan of its frames, where every
    window lies in it, and which state of the whole-game pass each window
    starts from: that pass returns the state before each frame the model reads,
    in the order of the segments. ``names`` are the model's parameters in order.
    """
    import numpy as np
    from .plans import _active_parameters

    result, before = [], 0
    for left in range(0, len(starts), size):
        right = min(left + size, len(starts))
        plan = compact.plan(int(starts[left]), int(ends[right - 1]), int(firsts[left]), int(firsts[right]))
        jobs = plan.pop("jobs")
        first = np.searchsorted(jobs, starts[left:right])
        lengths = (np.searchsorted(jobs, ends[left:right]) - first).astype(np.int32)
        if int(lengths.sum()) != len(jobs) or (lengths < 1).any():
            raise ValueError("Loss windows do not cover their frames once each")
        longest = int(lengths.max())
        row = np.repeat(np.arange(right - left), lengths)
        column = np.arange(len(jobs)) - np.repeat(first, lengths)
        # Row ``len(jobs)`` of the frame features is a row of zeros: the padding of shorter windows.
        frames = np.full((right - left, longest), len(jobs), dtype=np.int32)
        frames[row, column] = np.arange(len(jobs), dtype=np.int32)
        result.append({
            "plan": plan, "frames": frames, "lengths": lengths,
            "position": (row * longest + column).astype(np.int32),  # of every frame among the padded window frames
            "initial": (before + first).astype(np.int32), "count": len(jobs),
            "samples": (int(firsts[left]), int(firsts[right])),
            "active_parameters": _active_parameters(plan, names),
        })
        before += len(jobs)
    return result


def _device(mx, value):
    """The arrays of a host plan on the GPU."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return mx.array(value)
    if isinstance(value, dict):
        return {key: _device(mx, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_device(mx, item) for item in value]
    return value


SCAN_FRAMES = 3_000_000  # frames of whole-game passes that run side by side in one kernel call


def _scan(backend, frozen, games):
    """For each game the state before each of its frames with the weights ``frozen``; row 0 is the initial state.

    ``games`` are pairs of a game's segment plans and its number of frames.
    The recurrences of several games run side by side, as many as fit
    ``SCAN_FRAMES``, so a step waits for its longest game and not for all of
    them one after another.
    """
    import numpy as np
    mx = backend.mx
    weights = [frozen["gru." + key] for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    initial = mx.zeros((1, 64), dtype=mx.float32)
    vectors = [mx.concatenate([backend.encode(frozen, plan)[0] for plan in plans], axis=0)[:count]
               for plans, count in games]
    result, first = [None] * len(games), 0
    while first < len(games):
        last, frames = first + 1, games[first][1]
        while last < len(games) and frames + games[last][1] <= SCAN_FRAMES:
            frames += games[last][1]
            last += 1
        if last == first + 1:
            states = backend.optimized.gru_prefix(vectors[first][None], initial, *weights)[0]
            result[first] = mx.concatenate((initial, states), axis=0)
        else:
            counts = [games[index][1] for index in range(first, last)]
            offsets = np.cumsum([0] + counts[:-1])
            states = backend.optimized.gru_prefixes(
                mx.concatenate(vectors[first:last], axis=0), mx.array(offsets.astype(np.int32)),
                mx.array(np.asarray(counts, dtype=np.int32)), *weights)
            for index, offset, count in zip(range(first, last), offsets.tolist(), counts):
                result[index] = mx.concatenate((initial, states[offset:offset + count]), axis=0)
        first = last
    return result


def _forward(backend, p, plan, constants, states, value_trunk=1.0):
    """Log-probabilities and values of a segment's decisions, each window started from its state in ``states``."""
    mx = backend.mx
    frames, lengths, position, initial = constants[:4]
    vectors, table, cells = backend.encode(p, plan)
    padded = mx.concatenate((vectors, mx.zeros((1, vectors.shape[1]), dtype=mx.float32)), axis=0)
    sequences = backend.recurrent(p, padded[frames], "gru", mx.stop_gradient(states[initial]), lengths)
    return backend.score(p, plan, sequences.reshape(-1, 64)[position], table, cells, value_trunk)


def _window_arrays(windows, decisions):
    """Where the loss windows of one game start and end and which decisions they train, as arrays."""
    import numpy as np
    return (np.asarray([window.start_seq for window in windows]),
            np.asarray([window.end_seq + 1 for window in windows]),
            np.asarray([window.sample_indices[0] for window in windows] + [decisions]))


def evaluate(model, trajectories, *, config):
    """Judge ``model``'s weights on games without taking a step.

    For each game: its samples and, summed over them, (ratio - 1) x advantage,
    (ratio - 1) - log ratio, the decisions whose ratio left the clip range and
    half the squared error of the value against the return, where the ratio is
    the new probability of the recorded action over the one recorded while
    playing and advantage and return are the game's own, as ``prepare`` gives
    them for it alone under ``config``.
    """
    import numpy as np
    from ..compact import Compact
    from ..ppo import prepare
    from .model import MLXBackend
    from .update import _host, _released, _returned

    parameters = dict(model.named_parameters())
    manifest = {"parameter_order": list(parameters), "parameters": {name: _host(p) for name, p in parameters.items()},
                "model_config": model.config.to_dict(), "ppo": config.to_dict(),
                "optimizer": {"config": {"lr": config.learning_rate, "betas": (0.9, 0.999), "eps": 1e-8,
                                         "weight_decay": 0, "amsgrad": False}, "state": {}},
                "history_plans": []}
    names, result = list(parameters), []
    backend = MLXBackend(manifest, array_loader=lambda value: value, plan_loader=None)
    try:
        mx = backend.mx
        frozen = {key: mx.stop_gradient(value) for key, value in backend.p.items()}
        for rollout in trajectories:
            batch = prepare([rollout], config)
            compact = rollout.compact
            if compact is None:
                compact = Compact.from_history(rollout.recipe, rollout.decisions)
            parts = segments(compact, *_window_arrays(batch.windows, len(rollout.decisions)), names, SEGMENT_WINDOWS)
            plans = [_device(mx, part["plan"]) for part in parts]
            states = _scan(backend, frozen, [(plans, sum(part["count"] for part in parts))])[0]
            logs, values = [], []
            for part, plan in zip(parts, plans):
                constants = tuple(mx.array(part[key]) for key in ("frames", "lengths", "position", "initial"))
                found = _forward(backend, frozen, plan, constants, states)
                mx.eval(found)
                logs.append(np.array(found[0], dtype=np.float64))
                values.append(np.array(found[1], dtype=np.float64))
            logs, values = np.concatenate(logs), np.concatenate(values)
            log_ratio = logs - np.asarray(rollout.log_probs, dtype=np.float32).astype(np.float64)
            ratio = np.exp(log_ratio)
            advantages = np.asarray(batch.advantages, dtype=np.float32).astype(np.float64)
            returns = np.asarray(batch.targets[0].returns, dtype=np.float32).astype(np.float64)
            result.append([float(logs.size), float(((ratio - 1) * advantages).sum()),
                           float(((ratio - 1) - log_ratio).sum()),
                           float((np.abs(ratio - 1) > config.clip_ratio).sum()),
                           float(0.5 * ((values - returns) ** 2).sum())])
            del plans, states
    finally:
        mx = _released(backend)
        del backend
        _returned(mx)
    return result


def _grouped(games, frames, limit):
    """``games`` in order, cut into groups whose frames add up to at most ``limit``; a game alone may exceed it."""
    groups, total = [], 0
    for game in games:
        if not groups or total + frames[game] > limit:
            groups.append([])
            total = 0
        groups[-1].append(game)
        total += frames[game]
    return groups


def recent_kl(history, batch_samples):
    """The mean KL over the latest steps' measurements that together cover half of the batch's samples.

    ``history`` holds a row per step, measured before the step: its samples and, fourth, its mean KL; the last
    row is the step about to be taken. A step on ten trajectories shows 0.005 or 0.013 depending on which ten
    they are, so one step's reading cannot decide whether another pass is made; with every trajectory in each
    step the last reading covers the whole batch and stands alone.
    """
    samples = total = 0.0
    for row in reversed(history):
        samples += row[0]
        total += row[0] * row[3]
        if 2 * samples >= batch_samples:
            break
    return total / samples


def moved_enough(config, taken, kl, first_of_epoch=True):
    """Whether the update ends before its next step: its games show a mean KL above the configured target.

    ``taken`` are the steps taken so far. The update ends only between epochs,
    before an epoch's first step: every game of the batch is then trained on
    in every epoch that was begun, and at least once, because before the first
    epoch the policy is the one that played, whatever the estimate says.
    """
    return config.target_kl is not None and taken > 0 and first_of_epoch and kl > config.target_kl


def update(model, optimizer, trajectories, *, config, generator, progress=None):
    """Run the whole update, then copy FP32 weights, Adam state and the shuffle RNG back."""
    import numpy as np
    import torch
    from ..compact import Compact
    from ..ppo import PPOReport, prepare, set_momentum
    from .model import INITIAL_LOG_PROBABILITY_LIMIT, INITIAL_VALUE_LIMIT, MLXBackend
    from .update import _given_back, _manifest, _released, _require, _returned, _taken

    started = time.perf_counter()
    set_momentum(optimizer, config)
    rollouts, histories, parameters, group, adam = _taken(model, optimizer, trajectories)
    batch = prepare(rollouts, config)
    planned_steps = batch.planned_steps(config)
    schedule = torch.Generator(device="cpu")
    schedule.set_state(generator.get_state())
    manifest = _manifest(model, parameters, group, adam, config, histories)
    names = list(parameters)
    firsts = [0]
    for starts in batch.starts:
        firsts.append(firsts[-1] + len(starts))
    seconds = {"plans": 0.0, "upload": 0.0, "scan": 0.0, "windows": 0.0, "step": 0.0}
    frame_counts = [history.frame_count for history in histories]  # at least the frames the model reads of each game
    kept, kept_bytes = {}, 0  # game -> its segments, while they fit

    def planned(index):
        """Everything of game ``index`` that no weight enters."""
        nonlocal kept_bytes
        if index in kept:
            return kept[index]
        plan_started = time.perf_counter()
        rollout = rollouts[index]
        compact = rollout.compact
        if compact is None:
            compact = Compact.from_history(rollout.recipe, rollout.decisions)
        parts = segments(compact, *_window_arrays(batch.windows[firsts[index]:firsts[index + 1]],
                                                  len(rollout.decisions)), names, SEGMENT_WINDOWS)
        offset = batch.offsets[index]
        targets = {key: np.asarray(values, dtype=np.float32) for key, values in (
            ("old_log_probs", rollout.log_probs), ("old_values", rollout.values),
            ("advantages", batch.advantages[offset:offset + len(rollout.decisions)]),
            ("returns", batch.targets[index].returns))}
        for part in parts:
            low, high = part.pop("samples")
            part.update({key: values[low:high] for key, values in targets.items()})
        size = _bytes(parts)
        if kept_bytes + size <= HOST_PLAN_BYTES:
            kept[index] = parts
            kept_bytes += size
        seconds["plans"] += time.perf_counter() - plan_started
        return parts

    backend = None
    weighted = [0.0] * 4
    count_seen = completed = 0
    history, stopped = [], False
    planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-plans")
    try:
        backend = MLXBackend(manifest, array_loader=lambda value: value, plan_loader=None)
        mx = backend.mx
        clip, value_clip = config.clip_ratio, config.value_clip
        def losses(p, plan, constants, states, scale):
            old_logs, old_values, advantages, returns = constants[4:]
            logs, values = _forward(backend, p, plan, constants, states, config.value_trunk)
            log_ratio = logs - old_logs
            ratio = mx.exp(log_ratio)
            clipped = mx.clip(ratio, 1 - clip, 1 + clip)
            policy = -mx.sum(mx.minimum(ratio * advantages, clipped * advantages))
            squared = (values - returns) ** 2
            if value_clip is not None:
                limited = old_values + mx.clip(values - old_values, -value_clip, value_clip)
                squared = mx.maximum(squared, (limited - returns) ** 2)
            value = 0.5 * mx.sum(squared)
            kl = mx.sum((ratio - 1) - log_ratio)
            fraction = mx.sum((mx.abs(ratio - 1) > clip).astype(mx.float32))
            return (policy + config.value_coefficient * value) * scale, policy, value, kl, fraction, logs, values

        prepared = time.perf_counter()
        for _ in range(config.epochs):
            minibatches = batch.game_minibatches(config, schedule)
            order = [game for selected in minibatches for game in selected]
            # Plans hold nothing that depends on the weights: the next game's is built while the GPU works.
            ahead = {order[0]: planner.submit(planned, order[0])}
            cursor = 0
            for selected in minibatches:
                samples = sum(len(rollouts[game].decisions) for game in selected)
                gradients, active = {}, set()
                sums = [mx.zeros(()) for _ in range(4)]
                # Largest |log-probability| and |value| difference from play, and the summed squared value difference.
                differences = [mx.zeros(()) for _ in range(3)] if backend.initial_check is None else None
                frozen = {key: mx.stop_gradient(value) for key, value in backend.p.items()}
                # Games are taken a few at a time: as many as the whole-game pass runs side by side.
                for group in _grouped(selected, frame_counts, SCAN_FRAMES):
                    loaded = []
                    for game in group:
                        cursor += 1
                        if cursor < len(order) and order[cursor] not in ahead:
                            ahead[order[cursor]] = planner.submit(planned, order[cursor])
                        parts = ahead.pop(game).result()
                        stamp = time.perf_counter()
                        plans = [_device(mx, part["plan"]) for part in parts]
                        constants = [tuple(mx.array(part[key]) for key in (
                            "frames", "lengths", "position", "initial",
                            "old_log_probs", "old_values", "advantages", "returns")) for part in parts]
                        seconds["upload"] += time.perf_counter() - stamp
                        loaded.append((parts, plans, constants))
                    stamp = time.perf_counter()
                    scanned = _scan(backend, frozen, [(plans, sum(part["count"] for part in parts))
                                                      for parts, plans, _ in loaded])
                    mx.eval(scanned)
                    seconds["scan"] += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    for (parts, plans, constants), states in zip(loaded, scanned):
                        for part, plan, constant in zip(parts, plans, constants):
                            values, found = mx.value_and_grad(
                                lambda p: losses(p, plan, constant, states, 1.0 / samples))(backend.p)
                            for name in part["active_parameters"]:
                                gradients[name] = gradients[name] + found[name] if name in gradients else found[name]
                            active.update(part["active_parameters"])
                            sums = [total + value for total, value in zip(sums, values[1:5])]
                            waiting = [sums, list(gradients.values())]
                            if differences is not None:
                                delta_logs, delta_values = values[5] - constant[4], values[6] - constant[5]
                                differences = [mx.maximum(differences[0], mx.max(mx.abs(delta_logs))),
                                               mx.maximum(differences[1], mx.max(mx.abs(delta_values))),
                                               differences[2] + mx.sum(delta_values ** 2)]
                                waiting.append(differences)
                            # Evaluated as it is built: the GPU works on this segment while the next is put together.
                            mx.async_eval(*waiting)
                    del loaded, scanned
                    mx.eval(sums)
                    seconds["windows"] += time.perf_counter() - stamp
                stamp = time.perf_counter()
                ordered = [name for name in names if name in active]
                backend._check_active(ordered)
                norm = mx.sqrt(sum(mx.sum(gradients[name] ** 2) for name in ordered))
                mx.eval(norm, sums, differences or [])
                if not backend.finite([norm, *sums]):
                    raise FloatingPointError("Nonfinite MLX PPO loss, predictions or gradient norm")
                if differences is not None:
                    # Every window of the first step was computed with the weights that played.
                    largest_log, largest_value, squared = (float(item.item()) for item in differences)
                    backend.initial_check = {
                        "samples": samples, "maximum_log_probability_difference": largest_log,
                        "maximum_value_difference": largest_value, "value_rmse": (squared / samples) ** 0.5,
                        "maximum_log_probability_limit": INITIAL_LOG_PROBABILITY_LIMIT,
                        "maximum_value_limit": INITIAL_VALUE_LIMIT,
                        "reference": "recorded collection behavior on every game of the first step before any update",
                        "scope": "initial FP32 port sanity, not learning-quality equivalence",
                        "passed": largest_log <= INITIAL_LOG_PROBABILITY_LIMIT and largest_value <= INITIAL_VALUE_LIMIT,
                    }
                    if not backend.initial_check["passed"]:
                        raise ValueError("Initial MLX prediction sanity limits exceeded; no optimizer step performed: "
                                         f"{backend.initial_check}")
                measured = [float(item.item()) for item in sums]
                history.append([samples, *(value / samples for value in measured), float(norm.item())])
                if moved_enough(config, completed, recent_kl(history, batch.samples), selected is minibatches[0]):
                    # The policy has moved as far as it may: this epoch and the rest are not taken.
                    stopped = True
                    break
                backend.apply(gradients, ordered, norm)
                seconds["step"] += time.perf_counter() - stamp
                for index in range(4):
                    weighted[index] += measured[index]
                count_seen += samples
                completed += 1
                if progress is not None:
                    progress(completed, planned_steps)
            if stopped:
                break
        mx.synchronize()
        updated = time.perf_counter()
        _require(stopped or (completed == planned_steps and count_seen == batch.samples * config.epochs),
                 "MLX update skipped samples or optimizer steps")
        # Whole epochs only: every sample of the batch was trained on as often as every other, at least once.
        _require(count_seen >= batch.samples and count_seen % batch.samples == 0,
                 "MLX update ended inside an epoch")
        _given_back(backend, optimizer, parameters)
        generator.set_state(schedule.get_state())
        restored = time.perf_counter()
        validation_seconds, initial_check = backend.state_validation_seconds, backend.initial_check
    finally:
        planner.shutdown(wait=True, cancel_futures=True)
        kept.clear()
        if backend is not None:
            mx = _released(backend)
            del backend
            _returned(mx)
    finished = time.perf_counter()
    return PPOReport(batch.samples, completed, *(value / count_seen for value in weighted), timings_seconds={
        "preparation": prepared - started, "update": updated - prepared,
        "plan_build": seconds["plans"], "plan_upload": seconds["upload"], "game_passes": seconds["scan"],
        "window_passes": seconds["windows"], "optimizer_steps": seconds["step"], "cpu_restore": restored - updated,
        "state_validation": validation_seconds, "cleanup": finished - restored,
        "full_update": finished - started,
    }, initial_check=initial_check, steps=history)
