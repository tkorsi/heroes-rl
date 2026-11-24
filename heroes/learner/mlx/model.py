"""Canonical FP32 MLX network and PPO math shared by training and comparisons.

Construction imports MLX lazily. Immutable plans may be cached; learned features
and recurrent prefixes are recomputed at current weights for each optimizer step.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
import math
import time

PREFIX_CONSTANT_CACHE_BYTES = 4 * 1024**3
INITIAL_LOG_PROBABILITY_LIMIT = 0.02
INITIAL_VALUE_LIMIT = 0.002


def plan_bytes(host):
    """The bytes a host plan takes on the GPU: four for every element of every array.

    ``_convert`` creates an FP32 or int32 array for every occurrence, including
    uint8 and bool inputs and repeated host-array aliases. The planner's integer
    sequences stay on the CPU and hold no array. A long game has hundreds of
    thousands of them, and walking them at every miss of the plan cache cost
    more than converting the plan, so they are skipped.
    """
    import numpy as np
    from .plans import _CPU_PLAN_METADATA_KEYS

    def size(value):
        if isinstance(value, np.ndarray):
            return value.size * 4
        if isinstance(value, dict):
            return sum(size(x) for key, x in value.items()
                       if not (key in _CPU_PLAN_METADATA_KEYS and isinstance(x, (list, tuple))))
        if isinstance(value, (list, tuple)):
            return sum(size(x) for x in value)
        return 0

    return size(host)


class MLXModel:
    """Parameter-agnostic neural operations; no Adam or checkpoint construction."""

    def __init__(self):
        import numpy as np
        import mlx.core as mx
        from .kernels import OptimizedOps
        if not mx.metal.is_available():
            raise RuntimeError("MLX requires an Apple GPU")
        mx.set_default_device(mx.gpu)
        self.mx, self.np = mx, np
        self.optimized = OptimizedOps()

    def _array(self, value):
        if value.dtype.kind not in "ifbu":
            raise ValueError("Portable plans support only integer and FP32 tensors")
        return self.mx.array(value, dtype=self.mx.float32 if value.dtype.kind == "f" else self.mx.int32)

    def _convert(self, value):
        from .plans import _CPU_PLAN_METADATA_KEYS, _cpu_plan_metadata

        def convert(item):
            if isinstance(item, self.np.ndarray):
                return self._array(item)
            if isinstance(item, dict):
                return {key: _cpu_plan_metadata(key, child) if key in _CPU_PLAN_METADATA_KEYS
                        and isinstance(child, (list, tuple)) else convert(child)
                        for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [convert(child) for child in item]
            return item

        return convert(value)

    def _prepare_plan(self, plan):
        from .plans import _compact_groups
        # Inference and legacy portable bundles may retain old unused fields.
        plan = {**plan, "groups": _compact_groups(plan["groups"])}
        counts = [([int(x) for x in group[2]["counts"]] if group[0] == "pooled" else None)
                  for group in plan["groups"]]
        converted = self._convert(plan)
        converted["_pool_counts"] = counts
        return converted

    def linear(self, p, x, name):
        return x @ p[name + ".weight"].T + p[name + ".bias"]

    def embedding(self, p, name):
        # PyTorch padding_idx=0 freezes row zero; it remains in restored Adam.
        value = p[name + ".weight"]
        return self.mx.concatenate((self.mx.stop_gradient(value[:1]), value[1:]), axis=0)

    def recurrent(self, p, x, name, state=None, lengths=None):
        suffix = "" if name == "gru" else "_l0"
        width = 64 if name == "gru" else 32
        if state is None:
            state = self.mx.zeros((x.shape[0], width))
        return self.optimized.gru(x, state, *[
            p[name + "." + key + suffix]
            for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")], lengths=lengths)

    def tree(self, p, plan, frame_limit=None):
        return self.optimized.tree(p, plan, frame_limit)

    def encode(self, p, plan, frame_limit=None):
        mx = self.mx
        table = self.tree(p, plan, frame_limit)
        indices = plan["roots"] if frame_limit is None else plan["roots"][:frame_limit]
        roots = table[indices]
        glob, own = mx.zeros_like(roots), mx.zeros_like(roots)
        cells = None
        maps = plan["maps"]
        if maps is not None and frame_limit is not None:
            count = bisect_left(maps["cpu_positions"], frame_limit)
            if not count:
                maps = None
            else:
                unique = max(maps["cpu_reuse"][:count]) + 1
                maps = {**maps, "count": unique,
                        **{k: maps[k][:count] for k in ("positions", "own", "reuse")}}
                if maps.get("delta") is None:
                    end = maps["attribute_ends"][unique - 1]
                    maps.update({k: maps[k][:unique] for k in ("terrain", "road", "flags")})
                    maps.update({k: maps[k][:end] for k in ("attributes", "indices")})
        if maps is not None:
            if maps.get("delta") is not None:
                from .sparse_map import encode as sparse_map
                global_features = sparse_map(self, p, maps)
            else:
                n, width, height = maps["count"], maps["width"], maps["height"]
                size = width * height
                objects = mx.zeros((n * size, 16), dtype=mx.float32)
                encoded = mx.tanh(self.linear(p, maps["attributes"], "object_encoder.0"))
                objects = objects.at[maps["indices"]].add(encoded)
                grid = mx.concatenate((self.embedding(p, "terrain")[maps["terrain"]],
                                       self.embedding(p, "road")[maps["road"]],
                                       objects.reshape(n, size, 16), maps["flags"]), axis=-1)
                array = grid.reshape(n, height, width, 33)
                for name in ("map_cnn.0", "map_cnn.2"):
                    weight = mx.transpose(p[name + ".weight"], (0, 2, 3, 1))
                    array = mx.tanh(mx.conv2d(array, weight, padding=1) + p[name + ".bias"])
                # Match PyTorch adaptive pooling boundaries for any allowed map size,
                # including overlapping bins when a dimension is not divisible by 4.
                pooled = []
                for y in range(4):
                    row = []
                    for x in range(4):
                        tile = array[:, y * height // 4:math.ceil((y + 1) * height / 4),
                                     x * width // 4:math.ceil((x + 1) * width / 4), :]
                        row.append(mx.mean(tile, axis=(1, 2)))
                    pooled.append(mx.stack(row, axis=1))
                pooled = mx.stack(pooled, axis=1)
                nchw = mx.transpose(pooled, (0, 3, 1, 2)).reshape(n, 256)
                global_features = mx.tanh(self.linear(p, nchw, "map_global.2"))
                cells = array.reshape(n, size, 16)
            glob = glob.at[maps["positions"]].add(global_features[maps["reuse"]])
            own = own.at[maps["positions"]].add(table[maps["own"]])
        return mx.concatenate((roots, glob, own), axis=-1), table, cells

    def score(self, p, plan, states, table, cells, value_trunk=1.0):
        mx = self.mx
        decision_states = states[plan["decision_states"]]
        # ``value_trunk`` of the value's gradient goes on into the states; see ``PPOConfig.value_trunk``.
        read = (decision_states if value_trunk == 1 else mx.stop_gradient(decision_states) if value_trunk == 0
                else value_trunk * decision_states + (1 - value_trunk) * mx.stop_gradient(decision_states))
        values = self.linear(p, read, "value").squeeze(-1)
        query = mx.tanh(self.linear(p, decision_states, "policy_query"))
        logs = mx.zeros((plan["decision_count"],), dtype=mx.float32)
        for kind, values_by_head in plan["heads"]:
            d = values_by_head
            if kind == "categorical":
                logits = mx.sum(table[d["roots"]] * query[d["decision"]][:, None, :], axis=-1) / math.sqrt(32)
                if "window_memory" in d:
                    logits = logits + d["window_memory"] * p["window_memory.weight"]
            elif kind == "quantity":
                inputs = mx.concatenate((states[d["state"]], table[d["roots"]], d["numeric"]), axis=-1)
                logits = self.linear(p, mx.tanh(self.linear(p, inputs, "quantity.0")), "quantity.2") + d["offsets"]
            else:
                n = d["positions"].shape[1]
                inputs = mx.concatenate((mx.broadcast_to(states[d["state"]][:, None, :], (d["state"].shape[0], n, 64)),
                                         mx.broadcast_to(table[d["roots"]][:, None, :], (d["state"].shape[0], n, 32)),
                                         cells[d["cells"]], d["positions"]), axis=-1)
                logits = self.linear(p, mx.tanh(self.linear(p, inputs, "coordinate.0")), "coordinate.2").squeeze(-1)
                logits = mx.take_along_axis(logits, d["indices"], axis=1)
            logits = logits + d["mask"]
            selected = mx.take_along_axis(logits - mx.logsumexp(logits, axis=1, keepdims=True), d["selected"], axis=1).squeeze(1)
            logs = logs.at[d["decision"]].add(selected)
        return logs, values

    def finite(self, values):
        mx = self.mx
        return bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in values])).item())


class MLXBackend(MLXModel):
    """Adam working state for one complete PPO update."""

    def __init__(self, manifest, *, array_loader, plan_loader):
        super().__init__()
        mx = self.mx
        info = mx.device_info(mx.gpu)
        recommended = info.get("max_recommended_working_set_size")
        if not isinstance(recommended, int) or recommended <= 0:
            raise RuntimeError("MLX did not expose its recommended working-set size")
        self.memory_limit = int(recommended * 0.25)
        mx.set_memory_limit(self.memory_limit)
        mx.set_cache_limit(PREFIX_CONSTANT_CACHE_BYTES)
        mx.reset_peak_memory()
        self._load_plan = plan_loader
        self.state_validation_seconds = 0.0
        self.manifest = manifest
        self.order = list(manifest["parameter_order"])
        self.config = manifest["ppo"]
        self.adam_config = manifest["optimizer"]["config"]
        if self.adam_config.get("weight_decay", 0) != 0 or self.adam_config.get("amsgrad", False):
            raise ValueError("MLX preserves the existing plain Adam contract")
        if manifest["model_config"]["feature_size"] != 32 or manifest["model_config"]["hidden_size"] != 64:
            raise ValueError("MLX requires the current feature32/hidden64 model")
        self.p = {name: self._array(array_loader(manifest["parameters"][name]))
                  for name in self.order}
        self.adam = {}
        for name in self.order:
            state = manifest["optimizer"]["state"].get(name)
            if state is None:
                self.adam[name] = {"step": 0, "exp_avg": mx.zeros_like(self.p[name]),
                                   "exp_avg_sq": mx.zeros_like(self.p[name])}
            else:
                self.adam[name] = {
                    "step": int(state["step"]),
                    "exp_avg": self._array(array_loader(state["exp_avg"])),
                    "exp_avg_sq": self._array(array_loader(state["exp_avg_sq"])),
                }
        self.plan_cache = OrderedDict()
        self.plan_cache_bytes = 0
        self.initial_check = None
        mx.eval(self.p, self._moments())
        if not self.finite(list(self.p.values()) + self._moments()):
            raise FloatingPointError("Nonfinite initial MLX model or Adam state")
        mx.synchronize()

    def plan(self, reference):
        key = reference["path"]
        if key in self.plan_cache:
            self.plan_cache.move_to_end(key)
            return self.plan_cache[key][0]
        host = self._load_plan(reference)
        count = plan_bytes(host)
        while self.plan_cache and self.plan_cache_bytes + count > PREFIX_CONSTANT_CACHE_BYTES:
            _, (_, removed) = self.plan_cache.popitem(last=False)
            self.plan_cache_bytes -= removed
        plan = self._prepare_plan(host)
        if count <= PREFIX_CONSTANT_CACHE_BYTES:
            self.plan_cache[key] = (plan, count)
            self.plan_cache_bytes += count
        return plan

    def _replay(self, parameters, spec):
        """The initial state and, for a nonempty prefix, the state after each of its frames."""
        mx = self.mx
        count = spec["frame_limit"]
        initial = mx.zeros((1, 64), dtype=mx.float32)
        if not count:
            return initial, None
        plan = self.plan(self.manifest["history_plans"][spec["history"]])
        vectors, _, _ = self.encode(parameters, plan, count)
        return initial, self.optimized.gru_prefix(vectors[None, :count], initial, *[
            parameters["gru." + key] for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        ])[0]

    def prefix_states(self, prefixes, *, synchronize=True):
        mx = self.mx
        parameters = {key: mx.stop_gradient(value) for key, value in self.p.items()}
        result = {}
        for spec in prefixes:
            initial, states = self._replay(parameters, spec)
            for entry in spec["states"]:
                index = entry["state_index"]
                value = initial if index < 0 else states[index:index + 1]
                result[(spec["history"], entry["stop"])] = mx.stop_gradient(value)
        if synchronize:
            mx.eval(list(result.values()))
        else:
            mx.async_eval(list(result.values()))
        return result

    def forward(self, p, step, plan, prefixes):
        mx = self.mx
        vectors, table, cells = self.encode(p, plan)
        states = []
        for segment in step["sequences"]:
            initial = mx.stop_gradient(prefixes[(segment["history"], segment["start"])])
            sequence = self.recurrent(p, vectors[None, segment["left"]:segment["right"]], "gru", initial)
            states.append(sequence[0])
        return self.score(p, plan, mx.concatenate(states, axis=0), table, cells, self.config.get("value_trunk", 1.0))

    def losses(self, p, step, plan, prefixes, targets):
        mx = self.mx
        logs, values = self.forward(p, step, plan, prefixes)
        old_logs, old_values, advantages, returns = targets
        log_ratio = logs - old_logs
        ratio = mx.exp(log_ratio)
        clipped = mx.clip(ratio, 1 - self.config["clip_ratio"], 1 + self.config["clip_ratio"])
        policy = -mx.mean(mx.minimum(ratio * advantages, clipped * advantages))
        squared = (values - returns) ** 2
        if self.config["value_clip"] is not None:
            clip = self.config["value_clip"]
            limited = old_values + mx.clip(values - old_values, -clip, clip)
            squared = mx.maximum(squared, (limited - returns) ** 2)
        value = 0.5 * mx.mean(squared)
        total = policy + self.config["value_coefficient"] * value
        kl = mx.mean((ratio - 1) - log_ratio)
        fraction = mx.mean((mx.abs(ratio - 1) > self.config["clip_ratio"]).astype(mx.float32))
        return total, policy, value, kl, fraction, logs, values

    def _moments(self):
        return [value for state in self.adam.values() for key, value in state.items() if key != "step"]

    def _check_active(self, active):
        if not active or len(set(active)) != len(active) or any(name not in self.p for name in active):
            raise ValueError("Invalid minibatch active-parameter declaration")

    def check_initial(self, delta_logs, delta_values):
        """Before the first step: what MLX computes must be what was recorded while playing."""
        mx = self.mx
        self.initial_check = {
            "samples": int(delta_logs.size),
            "maximum_log_probability_difference": float(mx.max(mx.abs(delta_logs)).item()),
            "maximum_value_difference": float(mx.max(mx.abs(delta_values)).item()),
            "value_rmse": float(mx.sqrt(mx.mean(delta_values ** 2)).item()),
            "maximum_log_probability_limit": INITIAL_LOG_PROBABILITY_LIMIT,
            "maximum_value_limit": INITIAL_VALUE_LIMIT,
            "reference": "recorded collection behavior on the first scheduled minibatch before any update",
            "scope": "initial FP32 port sanity, not learning-quality equivalence",
        }
        self.initial_check["passed"] = (
            self.initial_check["maximum_log_probability_difference"] <= INITIAL_LOG_PROBABILITY_LIMIT
            and self.initial_check["maximum_value_difference"] <= INITIAL_VALUE_LIMIT
        )
        if not self.initial_check["passed"]:
            raise ValueError("Initial MLX prediction sanity limits exceeded; no optimizer step performed")

    def apply(self, gradients, active, norm):
        """One Adam step on the ``active`` parameters, their gradients clipped to the configured norm."""
        mx = self.mx
        scale = mx.minimum(1.0, self.config["max_grad_norm"] / (norm + 1e-6))
        beta1, beta2 = self.adam_config["betas"]
        lr, epsilon = self.adam_config["lr"], self.adam_config["eps"]
        updated = dict(self.p)
        for name in active:
            gradient, state = gradients[name] * scale, self.adam[name]
            number = state["step"] + 1
            m = beta1 * state["exp_avg"] + (1 - beta1) * gradient
            v = beta2 * state["exp_avg_sq"] + (1 - beta2) * gradient**2
            denominator = mx.sqrt(v) / math.sqrt(1 - beta2**number) + epsilon
            updated[name] = self.p[name] - (lr / (1 - beta1**number)) * m / denominator
            self.adam[name] = {"step": number, "exp_avg": m, "exp_avg_sq": v}
        # Parameters outside active_parameters preserve their weights, moments
        # and counters, matching torch.optim.Adam's grad=None behavior.
        self.p = updated
        # Only what this step wrote can have become nonfinite; everything else was
        # checked when it was written. The check is evaluated with the step itself.
        written = [value for name in active
                   for value in (updated[name], self.adam[name]["exp_avg"], self.adam[name]["exp_avg_sq"])]
        state_is_finite = mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in written]))
        mx.eval(self.p, self._moments(), state_is_finite)
        validation_started = time.perf_counter()
        state_is_finite = bool(state_is_finite.item())
        self.state_validation_seconds += time.perf_counter() - validation_started
        if not state_is_finite:
            raise FloatingPointError("Nonfinite MLX model or Adam state after update")
        if mx.get_peak_memory() > self.memory_limit:
            raise MemoryError("MLX peak memory exceeded the declared reserve")

    def update(self, step, *, prefixes=None):
        mx = self.mx
        if prefixes is None:
            prefixes = self.prefix_states(step["prefixes"])
        elif set(prefixes) != {(spec["history"], entry["stop"])
                              for spec in step["prefixes"] for entry in spec["states"]}:
            raise ValueError("Prepared prefixes differ from this minibatch's boundaries")
        plan = self._prepare_plan(step["plan"])
        targets = tuple(self._array(step[key]) for key in ("old_log_probs", "old_values", "advantages", "returns"))
        # Only the parameter tree is transformed. Metadata, NumPy-backed step
        # records and tuple-keyed prefix lookups remain ordinary Python values.
        values, gradients = mx.value_and_grad(
            lambda parameters: self.losses(parameters, step, plan, prefixes, targets)
        )(self.p)
        active = step["active_parameters"]
        self._check_active(active)
        norm = mx.sqrt(sum(mx.sum(gradients[name] ** 2) for name in active))
        mx.eval(values, norm)
        if not self.finite([norm, *values]):
            raise FloatingPointError("Nonfinite MLX PPO loss, predictions or gradient norm")
        if self.initial_check is None:
            self.check_initial(values[-2] - targets[0], values[-1] - targets[1])
        self.apply(gradients, active, norm)
        return [float(value.item()) for value in values[1:5]], int(values[-2].size)
