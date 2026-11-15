"""MLX neural inference with the existing typed policy and CPU RNG boundary.

The campaign selects this backend explicitly from its bound plan. Checkpoint
weights, recurrent-state records and torch.Generator sampling retain their
existing formats; tree/map/GRU and policy/value network computations use MLX.
No games, numerical work or device initialization occur on import.
"""
from __future__ import annotations

from collections import OrderedDict
from math import ceil, log1p, sqrt
from weakref import WeakKeyDictionary, ref

import torch

from .contextual_model import (
    ConditionalProbability, ContextualDistribution, PolicyValue,
    StepOutput, WeightsSnapshot, _exact_tree, _record_key, observation_semantics,
)

_BACKEND = "cpu"
_CONTEXTS = WeakKeyDictionary()
_TREE_CACHE_ENTRIES = 4096
_MAP_CACHE_ENTRIES = 8


def set_policy_backend(backend):
    """Select the process's policy implementation from an already bound plan."""
    if backend not in ("cpu", "mlx"):
        raise ValueError("Policy backend must be cpu or mlx")
    global _BACKEND
    _BACKEND = backend
    _CONTEXTS.clear()


def get_policy_backend():
    return _BACKEND


class _Inference:
    def __init__(self, model):
        import mlx.core as mx
        import numpy as np
        from .mlx_model import MLXModel

        if not mx.metal.is_available():
            raise RuntimeError("MLX policy requires an Apple GPU")
        if (model.config.feature_size, model.config.hidden_size) != (32, 64):
            raise ValueError("MLX policy requires feature32/hidden64")
        info = mx.device_info(mx.gpu)
        limit = info.get("max_recommended_working_set_size")
        if not isinstance(limit, int) or limit <= 0:
            raise RuntimeError("MLX GPU did not report its working-set budget")
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(int(limit * .25))
        mx.set_cache_limit(256 * 1024**2)
        self.mx, self.np = mx, np
        self.model = ref(model)
        self.snapshot = WeightsSnapshot(model)
        self.p = {name: mx.array(value.detach().numpy().copy(), dtype=mx.float32)
                  for name, value in model.named_parameters()}
        self.math = MLXModel()
        self.trees, self.maps = OrderedDict(), OrderedDict()
        mx.eval(self.p)

    def ensure_unchanged(self):
        model = self.model()
        if model is None or not self.snapshot.unchanged(model):
            raise ValueError("MLX behavior weights changed during a decision")

    def array(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().numpy()
        return self.mx.array(value, dtype=self.mx.float32)

    def cpu(self, value):
        return torch.from_numpy(self.np.array(value, dtype=self.np.float32, copy=True))

    def compact(self, value):
        # Materialize only this small result; memo entries must not retain a
        # complete semantic-table allocation or its lazy forward graph.
        return self.mx.array(self.np.array(value, dtype=self.np.float32, copy=True))

    def linear(self, value, name):
        return self.math.linear(self.p, value, name)

    def recurrent(self, vector, state):
        return self.math.recurrent(self.p, vector[None, None, :], "gru", state[None, :])[0, 0]

    def feature(self, semantic):
        cacheable = _exact_tree(semantic)
        if cacheable and semantic in self.trees:
            self.trees.move_to_end(semantic)
            return self.trees[semantic]
        from .native_recipes import _Arena
        from .metal_backend import _History
        from .mlx_backend import _host
        nodes = []
        root = _Arena(nodes).add(semantic)
        history = _History(nodes, [{"root": root, "map": None, "conditionals": []}])
        plan = self.math._prepare_plan(_host(history.plan(0, 1, device=False)))
        result = self.compact(self.math.tree(self.p, plan)[plan["roots"][0]])
        if cacheable:
            if len(self.trees) >= _TREE_CACHE_ENTRIES:
                self.trees.popitem(last=False)
            self.trees[semantic] = result
        return result

    def observation(self, observation):
        key = _record_key(observation)
        if key in self.maps:
            self.maps.move_to_end(key)
            return self.maps[key]
        mx, np = self.mx, self.np
        model = self.model()
        width, height, _ = observation.size
        if width > model.config.max_map_side or height > model.config.max_map_side:
            raise ValueError("Map exceeds declared model configuration")
        size = width * height
        terrain, road = np.zeros(size, np.int32), np.zeros(size, np.int32)
        flags = np.zeros((size, 5), np.float32)
        for tile in observation.tiles:
            index = tile.position[1] * width + tile.position[0]
            terrain[index], road[index] = tile.terrain + 1, tile.road + 1
            flags[index, :3] = 1., float(tile.blocked), float(tile.visitable)
        for hero in observation.heroes:
            flags[hero.position[1] * width + hero.position[0], 3] += 1.
        for town in observation.towns:
            flags[town.position[1] * width + town.position[0], 4] += 1.
        objects = sorted(observation.objects, key=lambda obj: (obj.position, obj.kind, obj.subtype, obj.owner))
        grid_objects = mx.zeros((size, 16))
        if objects:
            attributes = [[float(value >> bit & 1) for value in (obj.kind, obj.subtype) for bit in range(32)]
                          + [float(obj.owner == owner) for owner in ("self", "opponent", "neutral")] for obj in objects]
            encoded = mx.tanh(self.linear(mx.array(attributes, dtype=mx.float32), "object_encoder.0"))
            groups = {}
            for index, obj in enumerate(objects):
                groups.setdefault(obj.position[1] * width + obj.position[0], []).append(index)
            # One writer per cell. Atomic additions for collocated objects can
            # vary between runs and violate exact diagnostic reconstruction.
            positions, sums = [], []
            for position, indices in groups.items():
                value = encoded[indices[0]]
                for index in indices[1:]:
                    value = value + encoded[index]
                positions.append(position)
                sums.append(value)
            grid_objects = grid_objects.at[mx.array(positions, dtype=mx.int32)].add(mx.stack(sums))
        grid = mx.concatenate((self.p["terrain.weight"][mx.array(terrain)],
                               self.p["road.weight"][mx.array(road)], grid_objects, mx.array(flags)), axis=-1)
        cells = grid.reshape(1, height, width, 33)
        for name in ("map_cnn.0", "map_cnn.2"):
            cells = mx.tanh(mx.conv2d(cells, self.p[name + ".weight"].transpose(0, 2, 3, 1), padding=1)
                            + self.p[name + ".bias"])
        bins = [cells[:, y * height // 4:ceil((y + 1) * height / 4),
                      x * width // 4:ceil((x + 1) * width / 4)].mean(axis=(1, 2))
                for y in range(4) for x in range(4)]
        pooled = mx.stack(bins, axis=1).transpose(0, 2, 1).reshape(256)
        global_map = mx.tanh(self.linear(pooled, "map_global.2"))
        result = (self.compact(global_map), self.feature(observation_semantics(observation)),
                  self.compact(cells.reshape(size, 16)))
        if len(self.maps) >= _MAP_CACHE_ENTRIES:
            self.maps.popitem(last=False)
        self.maps[key] = result
        return result

    def step(self, model, frame, state):
        model._check_step(frame, state)
        if frame.semantics is None:
            return StepOutput(state, None)
        mx = self.mx
        if frame.observation is None:
            glob = own = mx.zeros((32,))
            cells = None
        else:
            glob, own, cells = self.observation(frame.observation)
        vector = mx.concatenate((self.feature(frame.semantics), glob, own))
        hidden = self.recurrent(vector, self.array(state))
        cpu_state = self.cpu(hidden)
        decision = None
        if frame.frame.kind == "decision":
            distribution = _Distribution(model, cpu_state, frame.frame.observation, frame.frame.offers,
                                         cells, {}, frame.guard, backend=self, hidden=hidden)
            decision = PolicyValue(self.cpu(self.linear(hidden, "value").squeeze()), distribution)
        return StepOutput(cpu_state, decision)


class _Distribution(ContextualDistribution):
    def __init__(self, *args, backend, hidden, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend, self.hidden = backend, hidden

    def _walk(self, action=None, generator=None):
        self.backend.ensure_unchanged()
        return super()._walk(action=action, generator=generator)

    def _features(self, semantic):
        return self.backend.feature(semantic)

    def _logits(self, semantics):
        b, mx = self.backend, self.backend.mx
        candidates = mx.stack([self._features(item) for item in semantics])
        return candidates @ mx.tanh(b.linear(self.hidden, "policy_query")) / sqrt(self.model.config.feature_size)

    def _quantity_logits(self, features, low, high, maximum, depth):
        b, mx = self.backend, self.backend.mx
        numeric = mx.array((low / maximum, high / maximum, log1p(maximum), depth / 31), dtype=mx.float32)
        value = mx.tanh(b.linear(mx.concatenate((self.hidden, features, numeric)), "quantity.0"))
        middle = (low + high) // 2
        return b.linear(value, "quantity.2") + mx.array((log1p(middle - low), log1p(high - middle - 1)), dtype=mx.float32)

    def _coordinate_logits(self, semantic, subject, width, height):
        b, mx = self.backend, self.backend.mx
        positions = b.array(super()._coordinate_positions(subject, width, height))
        state = mx.broadcast_to(self.hidden, (width * height, 64))
        target = mx.broadcast_to(self._features(semantic), (width * height, 32))
        value = mx.tanh(b.linear(mx.concatenate((state, target, self.cells, positions), axis=-1), "coordinate.0"))
        return b.linear(value, "coordinate.2").squeeze(-1)

    def _coordinate_subset(self, logits, coordinates):
        return logits[self.backend.mx.array(list(coordinates), dtype=self.backend.mx.int32)]

    def _choose(self, name, logits, selected, generator, trace):
        b, mx = self.backend, self.backend.mx
        log_probs = b.cpu(logits - mx.logsumexp(logits, axis=0))
        if selected is None:
            selected = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
        trace.append(ConditionalProbability(name, selected, log_probs.numel(), log_probs[selected]))
        return selected


def forward_step(model, frame, state):
    """Run neural math in MLX while preserving the checked session interface."""
    if torch.is_grad_enabled() or model.training or model.device.type != "cpu":
        raise ValueError("MLX policy requires no-grad inference on a CPU checkpoint model")
    if (not model._can_cache_prefix() or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                                            for parameter in model.parameters())):
        raise ValueError("MLX policy requires the plain FP32 model without custom execution")
    backend = _CONTEXTS.get(model)
    if backend is None or not backend.snapshot.unchanged(model):
        backend = _Inference(model)
        _CONTEXTS[model] = backend
    return backend.step(model, frame, state)
