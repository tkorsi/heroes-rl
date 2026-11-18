"""One PPO update on Apple GPUs through MLX.

The torch model and its Adam optimizer stay the source of truth: weights and
optimizer state are copied into MLX, every optimizer step runs there, and the
result is copied back only after the whole update finished and is finite.
Only weight-independent plans are cached; every recurrent prefix is recomputed
with the current weights.
"""
from __future__ import annotations

from bisect import bisect_left
import gc
import time


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


def update(model, optimizer, trajectories, *, config, generator, progress=None):
    """Run the whole update, then copy FP32 weights, Adam state and the shuffle RNG back."""
    import numpy as np
    import torch
    from ..ppo import PPOReport, prepare
    from .model import MLXBackend
    from .plans import _prefix_plan, _joined_plan, _active_parameters

    started = time.perf_counter()
    _require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
             "MLX requires one plain Adam parameter group")
    rollouts = tuple(trajectories)
    histories = tuple(item.recipe for item in rollouts)
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

    batch = prepare(rollouts, config)
    targets, advantages, offsets, windows, offset = batch.targets, batch.advantages, batch.offsets, batch.windows, batch.samples
    planned_steps = batch.planned_steps(config)
    schedule = torch.Generator(device="cpu")
    schedule.set_state(generator.get_state())
    references = [{"path": str(index), "history_index": index} for index in range(len(histories))]
    history_jobs = [[i for i, frame in enumerate(history.frames) if frame["root"] >= 0]
                    for history in histories]
    manifest = {"parameter_order": list(parameters), "parameters": {name: _host(p) for name, p in parameters.items()},
                "model_config": model.config.to_dict(), "ppo": config.to_dict(),
                "optimizer": {"config": {key: group[key] for key in
                              ("lr", "betas", "eps", "weight_decay", "amsgrad")}, "state": adam},
                "history_plans": references}
    plan_seconds = 0.0

    def load_history(reference):
        nonlocal plan_seconds
        plan_started = time.perf_counter()
        result = _host(_prefix_plan(histories[reference["history_index"]]))
        plan_seconds += time.perf_counter() - plan_started
        return result

    backend = None
    prepared_prefixes = None
    weighted = [0.0] * 4
    count_seen = completed = 0
    try:
        backend = MLXBackend(manifest, array_loader=lambda value: value, plan_loader=load_history)
        prepared = time.perf_counter()
        for _ in range(config.epochs):
            order = torch.randperm(len(windows), generator=schedule).tolist()
            for left in range(0, len(order), config.windows_per_minibatch):
                plan_started = time.perf_counter()
                selected = [windows[index] for index in order[left:left + config.windows_per_minibatch]]
                boundaries = {}
                for window in selected:
                    boundaries.setdefault(window.rollout_index, []).append(window.start_seq)
                prefixes = []
                for index, stops in boundaries.items():
                    jobs = history_jobs[index]
                    prefixes.append({"history": index, "frame_limit": bisect_left(jobs, max(stops)),
                                     "states": [{"stop": stop, "state_index": bisect_left(jobs, stop) - 1}
                                                for stop in sorted(set(stops))]})
                plan_seconds += time.perf_counter() - plan_started
                # The prefix graph runs on the GPU while the CPU builds this
                # minibatch's loss-window plan. It is used once and then dropped.
                prepared_prefixes = backend.prefix_states(prefixes, synchronize=False)
                plan_started = time.perf_counter()
                joined, sequences = _joined_plan(histories, selected, rollouts)
                old_logs, old_values, batch_advantages, returns = [], [], [], []
                for window in selected:
                    for index in window.sample_indices:
                        old_logs.append(rollouts[window.rollout_index].log_probs[index])
                        old_values.append(rollouts[window.rollout_index].values[index])
                        batch_advantages.append(advantages[offsets[window.rollout_index] + index])
                        returns.append(targets[window.rollout_index].returns[index])
                step = {"plan": _host(joined), "sequences": sequences, "prefixes": prefixes,
                        "active_parameters": _active_parameters(joined, list(parameters))}
                step.update({key: np.asarray(value, dtype=np.float32) for key, value in
                             (("old_log_probs", old_logs), ("old_values", old_values),
                              ("advantages", batch_advantages), ("returns", returns))})
                plan_seconds += time.perf_counter() - plan_started
                metrics, count = backend.update(step, prefixes=prepared_prefixes)
                for index, value in enumerate(metrics):
                    weighted[index] += value * count
                count_seen += count
                completed += 1
                if progress is not None:
                    progress(completed, planned_steps)
                del step, joined
                prepared_prefixes.clear()
                prepared_prefixes = None
        backend.mx.synchronize()
        updated = time.perf_counter()
        _require(completed == planned_steps and count_seen == offset * config.epochs,
                 "MLX update skipped samples or optimizer steps")
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
        generator.set_state(schedule.get_state())
        restored = time.perf_counter()
        validation_seconds = backend.state_validation_seconds
    finally:
        if prepared_prefixes is not None:
            prepared_prefixes.clear()
        for history in histories:
            history.plans.pop((0, history.frame_count, ()), None)
        if backend is not None:
            backend.plan_cache.clear()
            backend.p.clear()
            backend.adam.clear()
            mx = backend.mx
            del backend
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
    finished = time.perf_counter()
    return PPOReport(offset, completed, *(value / count_seen for value in weighted), timings_seconds={
        "preparation": prepared - started, "update": updated - prepared,
        "plan_build": plan_seconds, "cpu_restore": restored - updated,
        "state_validation": validation_seconds, "cleanup": finished - restored,
        "full_update": finished - started,
    })
