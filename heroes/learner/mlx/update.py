"""One PPO update on Apple GPUs through MLX.

The torch model and its Adam optimizer stay the source of truth: weights and
optimizer state are copied into MLX, every optimizer step runs there, and the
result is copied back only after the whole update finished and is finite.
Weight-independent plans are cached, and the plan of a minibatch is built on a
helper thread while the GPU runs the steps before it. The recurrent state
before a loss window is recomputed with the current weights at every step, by
one replay of each game the minibatch draws on.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import time

PLANS_AHEAD = 2  # minibatch plans a helper thread builds while the GPU runs earlier steps


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _host(value):
    """Convert tensor leaves once; retain finalized CPU metadata without per-int recursion."""
    import numpy as np
    import torch
    from .plans import _CPU_PLAN_METADATA_KEYS, _cpu_plan_metadata

    def convert(item):
        if isinstance(item, torch.Tensor):
            _require(item.device.type == "cpu", "MLX plans require CPU constants")
            item = item.detach().numpy()
        if isinstance(item, np.ndarray):
            if item.dtype == np.dtype("int64"):
                _require(not item.size or (item.min() >= -(2**31) and item.max() < 2**31),
                         "MLX plan index exceeds int32")
                item = item.astype(np.int32)
            _require(item.dtype.name in ("float32", "int32", "uint8", "bool"),
                     "Unsupported MLX plan dtype: " + str(item.dtype))
            return np.asarray(item, order="C")
        if isinstance(item, dict):
            return {key: _cpu_plan_metadata(key, child) if key in _CPU_PLAN_METADATA_KEYS
                    and isinstance(child, (list, tuple)) else convert(child)
                    for key, child in item.items() if key != "_arenas"}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    return convert(value)


def _taken(model, optimizer, trajectories):
    """Check what MLX is given and copy the Adam state out of torch.

    Returns the trajectories, their recipes, the named parameters, the
    optimizer's one parameter group and the Adam state as host arrays. The
    torch objects stay unchanged until ``_given_back``.
    """
    import torch
    _require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
             "MLX requires one plain Adam parameter group")
    rollouts = tuple(trajectories)
    histories = tuple(item.recipe if item.recipe is not None else item.compact for item in rollouts)
    _require(rollouts and all(history is not None for history in histories),
             "MLX requires trajectories with compiled recipes")
    parameters = dict(model.named_parameters())
    group = optimizer.param_groups[0]
    _require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                 and parameter.requires_grad for parameter in parameters.values()),
             "MLX requires trainable CPU FP32 parameters")
    _require({id(p) for p in group["params"]} == {id(p) for p in parameters.values()}
             and len(group["params"]) == len(parameters), "The optimizer does not own exactly the model parameters")
    _require(not group.get("amsgrad", False) and not group.get("maximize", False)
             and group.get("weight_decay", 0) == 0, "MLX implements plain Adam only")
    for rollout, history in zip(rollouts, histories):
        _require(rollout.decisions[-1] < history.frame_count, "A trained decision lies outside its recipe")

    # Weights and Adam state are copied into MLX. The torch objects stay
    # unchanged until every step succeeded and the result is finite.
    adam = {}
    for name, parameter in parameters.items():
        state = optimizer.state.get(parameter)
        if not state:
            adam[name] = None
            continue
        step = state["step"]
        step = float(step.item()) if isinstance(step, torch.Tensor) else float(step)
        _require(step.is_integer() and step >= 0, "Invalid restored Adam step")
        _require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam state fields")
        adam[name] = {"step": int(step), "exp_avg": _host(state["exp_avg"]),
                      "exp_avg_sq": _host(state["exp_avg_sq"])}
    return rollouts, histories, parameters, group, adam


def _manifest(model, parameters, group, adam, config, histories):
    return {"parameter_order": list(parameters), "parameters": {name: _host(p) for name, p in parameters.items()},
            "model_config": model.config.to_dict(), "ppo": config.to_dict(),
            "optimizer": {"config": {key: group[key] for key in
                          ("lr", "betas", "eps", "weight_decay", "amsgrad")}, "state": adam},
            "history_plans": [{"path": str(index), "history_index": index} for index in range(len(histories))]}


def _given_back(backend, optimizer, parameters):
    """Copy the finished update's weights and Adam state into torch; nothing is written unless all is finite."""
    import numpy as np
    import torch
    restored_parameters = {name: torch.from_numpy(np.array(value, copy=True)) for name, value in backend.p.items()}
    restored_adam = {}
    for name, parameter in parameters.items():
        saved = backend.adam[name]
        if not optimizer.state.get(parameter) and saved["step"] == 0:
            continue
        existing_step = optimizer.state.get(parameter, {}).get("step")
        step_dtype = existing_step.dtype if isinstance(existing_step, torch.Tensor) else torch.float32
        restored_adam[parameter] = {
            "step": torch.tensor(float(saved["step"]), dtype=step_dtype),
            "exp_avg": torch.from_numpy(np.array(saved["exp_avg"], copy=True)),
            "exp_avg_sq": torch.from_numpy(np.array(saved["exp_avg_sq"], copy=True)),
        }
    _require(all(torch.isfinite(p).all().item() for p in restored_parameters.values())
             and all(torch.isfinite(v).all().item() for state in restored_adam.values() for v in state.values()),
             "Nonfinite weights or optimizer state after the MLX update")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(restored_parameters[name])
    optimizer.state.clear()
    optimizer.state.update(restored_adam)
    optimizer.zero_grad(set_to_none=True)


def _released(backend):
    """Drop the backend's GPU arrays. The caller deletes the backend and then calls ``_returned``."""
    backend.plan_cache.clear()
    backend.p.clear()
    backend.adam.clear()
    return backend.mx


def _returned(mx):
    """Give MLX's cached GPU memory back to the system."""
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def update(model, optimizer, trajectories, *, config, generator, progress=None):
    """Run the whole update, then copy FP32 weights, Adam state and the shuffle RNG back."""
    import numpy as np
    import torch
    from ..ppo import PPOReport, prepare, set_momentum
    from .model import MLXBackend
    from .plans import _prefix_plan, _joined_plan, _active_parameters

    started = time.perf_counter()
    set_momentum(optimizer, config)
    if config.games_per_minibatch is not None:
        from .games import update as update_by_games
        return update_by_games(model, optimizer, trajectories, config=config, generator=generator, progress=progress)
    rollouts, histories, parameters, group, adam = _taken(model, optimizer, trajectories)

    batch = prepare(rollouts, config)
    targets, advantages, offsets, windows, offset = batch.targets, batch.advantages, batch.offsets, batch.windows, batch.samples
    planned_steps = batch.planned_steps(config)
    schedule = torch.Generator(device="cpu")
    schedule.set_state(generator.get_state())
    history_jobs = [[i for i, frame in enumerate(history.frames) if frame["root"] >= 0]
                    for history in histories]
    manifest = _manifest(model, parameters, group, adam, config, histories)
    plan_seconds = 0.0

    def load_history(reference):
        nonlocal plan_seconds
        plan_started = time.perf_counter()
        result = _host(_prefix_plan(histories[reference["history_index"]]))
        plan_seconds += time.perf_counter() - plan_started
        return result

    def prefix(index, stops):
        """What to replay of game ``index`` for the states before the frames ``stops`` (ascending)."""
        jobs = history_jobs[index]
        return {"history": index, "frame_limit": bisect_left(jobs, stops[-1]),
                "states": [{"stop": stop, "state_index": bisect_left(jobs, stop) - 1} for stop in stops]}

    names = list(parameters)

    def assemble(selected):
        """The part of a minibatch that no weight enters: its plan and its targets, and the seconds this took."""
        build_started = time.perf_counter()
        joined, sequences = _joined_plan(histories, selected, rollouts)
        old_logs, old_values, batch_advantages, returns = [], [], [], []
        for window in selected:
            for index in window.sample_indices:
                old_logs.append(rollouts[window.rollout_index].log_probs[index])
                old_values.append(rollouts[window.rollout_index].values[index])
                batch_advantages.append(advantages[offsets[window.rollout_index] + index])
                returns.append(targets[window.rollout_index].returns[index])
        step = {"plan": _host(joined), "sequences": sequences, "active_parameters": _active_parameters(joined, names)}
        step.update({key: np.asarray(value, dtype=np.float32) for key, value in
                     (("old_log_probs", old_logs), ("old_values", old_values),
                      ("advantages", batch_advantages), ("returns", returns))})
        return step, time.perf_counter() - build_started

    backend = None
    prepared_prefixes = None
    weighted = [0.0] * 4
    count_seen = completed = 0
    planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-plans")
    try:
        backend = MLXBackend(manifest, array_loader=lambda value: value, plan_loader=load_history)
        prepared = time.perf_counter()
        for _ in range(config.epochs):
            minibatches = [[windows[index] for index in selected] for selected in batch.minibatches(config, schedule)]
            # The schedule of an epoch is known, so the plans of the next minibatches are
            # built while the GPU is busy with this one. They hold nothing that depends on
            # the weights, and they are consumed in order.
            ahead = deque(planner.submit(assemble, selected) for selected in minibatches[:PLANS_AHEAD])
            for number, selected in enumerate(minibatches):
                if number + PLANS_AHEAD < len(minibatches):
                    ahead.append(planner.submit(assemble, minibatches[number + PLANS_AHEAD]))
                plan_started = time.perf_counter()
                boundaries = {}
                for window in selected:
                    boundaries.setdefault(window.rollout_index, []).append(window.start_seq)
                prefixes = [prefix(index, sorted(set(stops))) for index, stops in boundaries.items()]
                plan_seconds += time.perf_counter() - plan_started
                # A replay runs on the GPU while the CPU does the rest. Its states are used once and dropped.
                prepared_prefixes = backend.prefix_states(prefixes, synchronize=False)
                step, seconds = ahead.popleft().result()
                plan_seconds += seconds
                step["prefixes"] = prefixes
                metrics, count = backend.update(step, prefixes=prepared_prefixes)
                for index, value in enumerate(metrics):
                    weighted[index] += value * count
                count_seen += count
                completed += 1
                if progress is not None:
                    progress(completed, planned_steps)
                del step
                prepared_prefixes.clear()
                prepared_prefixes = None
        backend.mx.synchronize()
        updated = time.perf_counter()
        _require(completed == planned_steps and count_seen == offset * config.epochs,
                 "MLX update skipped samples or optimizer steps")
        _given_back(backend, optimizer, parameters)
        generator.set_state(schedule.get_state())
        restored = time.perf_counter()
        validation_seconds, initial_check = backend.state_validation_seconds, backend.initial_check
    finally:
        planner.shutdown(wait=True, cancel_futures=True)
        if prepared_prefixes is not None:
            prepared_prefixes.clear()
        for history in histories:
            history.plans.pop((0, history.frame_count, ()), None)
        if backend is not None:
            mx = _released(backend)
            del backend
            _returned(mx)
    finished = time.perf_counter()
    return PPOReport(offset, completed, *(value / count_seen for value in weighted), timings_seconds={
        "preparation": prepared - started, "update": updated - prepared,
        "plan_build": plan_seconds, "cpu_restore": restored - updated,
        "state_validation": validation_seconds, "cleanup": finished - restored,
        "full_update": finished - started,
    }, initial_check=initial_check)
