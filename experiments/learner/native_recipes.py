"""Compile checked visible histories into weight-independent native recipes.

Action enumeration uses the ordinary distribution's selected-action walk. Only
its tensor-scoring primitives are replaced while compiling; there is one policy
support definition, including the disclosed visible interaction guard.
"""
from __future__ import annotations

from math import copysign, log1p
import sys

import torch

from .contextual_model import (
    ConditionalProbability, ContextualActorCritic, ContextualDistribution,
    _operation_modes, _parameter_hooks, _saved_tensor_hooks, _text,
    observation_semantics,
)


class _Arena:
    def __init__(self, nodes):
        self.nodes = nodes
        # A new equality cache per frame preserves signed-zero first-use and
        # Python's existing numeric equality rules within that frame only.
        self.ids = {}

    def add(self, tree):
        if tree in self.ids:
            return self.ids[tree]
        kind = tree[0]
        if kind == "text":
            node = (kind, tree[1].encode("utf-8"), [])
        elif kind == "number":
            value = tree[1]
            node = (kind, [copysign(log1p(abs(value)), value), float(value == 0)], [])
        else:
            tag, children = tree[1:]
            children = [self.add(child) for child in children]
            node = (kind, self.add(_text(tag)), children)
        index = len(self.nodes)
        self.nodes.append(node)
        self.ids[tree] = index
        return index


class _RecipeDistribution(ContextualDistribution):
    def __init__(self, model, frame, arena):
        super().__init__(model, model.initial_state(), frame.frame.observation,
                         frame.frame.offers, None, {}, frame.guard)
        self.arena = arena
        self.recipes = []

    def _features(self, semantic):
        return self.arena.add(semantic)

    def _logits(self, semantics):
        return {"kind": "categorical", "roots": [self._features(item) for item in semantics]}

    def _quantity_logits(self, features, low, high, maximum, depth):
        middle = (low + high) // 2
        return {"kind": "quantity", "root": features,
                "numeric": self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31)),
                "offsets": self.model._float((log1p(middle - low), log1p(high - middle - 1)))}

    def _coordinate_logits(self, semantic, width, height):
        return {"kind": "coordinate", "root": self._features(semantic),
                "positions": self.model._float([(x / max(width - 1, 1), y / max(height - 1, 1))
                                                for y in range(height) for x in range(width)]),
                "indices": None}

    def _coordinate_subset(self, logits, coordinates):
        return {**logits, "indices": self.model._long(coordinates)}

    def _choose(self, name, logits, selected, generator, trace):
        if generator is not None or type(selected) is not int:
            raise ValueError("Native recipes describe recorded actions; they never sample a policy")
        kind = logits["kind"]
        size = (len(logits["roots"]) if kind == "categorical" else 2 if kind == "quantity"
                else len(logits["indices"]) if logits["indices"] is not None else len(logits["positions"]))
        if not 0 <= selected < size:
            raise ValueError("Recorded action is outside the compiled conditional support")
        self.recipes.append({**logits, "selected": selected})
        trace.append(ConditionalProbability(name, selected, size, self.state.new_zeros(())))
        return selected


def _map_recipe(model, observation, arena):
    width, height, _ = observation.size
    if width > model.config.max_map_side or height > model.config.max_map_side:
        raise ValueError("Map exceeds declared model configuration")
    terrain, road = [0] * (width * height), [0] * (width * height)
    flags = [[0.] * 5 for _ in terrain]
    for tile in observation.tiles:
        index = tile.position[1] * width + tile.position[0]
        terrain[index], road[index] = tile.terrain + 1, tile.road + 1
        flags[index][:3] = (1., float(tile.blocked), float(tile.visitable))
    for hero in observation.heroes:
        flags[hero.position[1] * width + hero.position[0]][3] += 1.
    for town in observation.towns:
        flags[town.position[1] * width + town.position[0]][4] += 1.
    objects = sorted(observation.objects, key=lambda obj: (obj.position, obj.kind, obj.subtype, obj.owner))
    attributes = [[float(value >> bit & 1) for value in (obj.kind, obj.subtype) for bit in range(32)]
                  + [float(obj.owner == owner) for owner in ("self", "opponent", "neutral")] for obj in objects]
    return {"width": width, "height": height, "terrain": model._long(terrain), "road": model._long(road),
            "flags": model._float(flags), "object_attributes": model._float(attributes).reshape(-1, 67),
            "object_indices": model._long([obj.position[1] * width + obj.position[0] for obj in objects]),
            "own": arena.add(observation_semantics(observation))}


def require_native_model(model):
    """Reject custom execution that a native tensor kernel would bypass."""
    compiled = sys.modules.get('torch._dynamo.compiled_autograd')
    if compiled is not None and any(getattr(compiled, name, False) for name in
            ('compiled_autograd_enabled', 'compiled_autograd_enabled_force_eager', 'in_compiled_autograd_region')):
        raise ValueError("Native learner requires ordinary eager autograd execution")
    if (type(model) is not ContextualActorCritic
            or any(name in model.__dict__ for name, value in vars(ContextualActorCritic).items() if callable(value))
            or any(name in model.__dict__ for name in ('modules', 'named_modules', 'parameters', 'named_parameters'))
            or not model._can_cache_prefix()
            or _operation_modes() or _saved_tensor_hooks() or torch.is_anomaly_enabled()
            or torch.nn.modules.module._global_backward_hooks
            or torch.nn.modules.module._global_backward_pre_hooks
            or any(module._backward_hooks or module._backward_pre_hooks for module in model.modules())
            or any(_parameter_hooks(param) for param in model.parameters())
            or any(type(param) is not torch.nn.Parameter or param.device.type != "cpu" or param.dtype != torch.float32 or not param.requires_grad
                   for param in model.parameters())
            or any(name in model.__dict__ for name in ("initial_state", "_float", "_long", "prefix_states"))
            or any(name in model.__dict__ for name in ("_graph_memo_state", "_leaf_replay_state"))):
        raise ValueError("Native learner requires the unmodified CPU float32 model without hooks or operation modes")
    # Parameter names alone cannot detect a removed or reordered activation.
    # Describe the canonical topology without constructing a model or using RNG.
    nn = torch.nn
    topology = {'': ContextualActorCritic, 'byte': nn.Embedding, 'text_gru': nn.GRU}
    groups = {'number': (nn.Linear, nn.Tanh), 'tree': (nn.Linear, nn.Tanh),
              'object_encoder': (nn.Linear, nn.Tanh),
              'map_cnn': (nn.Conv2d, nn.Tanh, nn.Conv2d, nn.Tanh),
              'map_global': (nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear, nn.Tanh),
              'coordinate': (nn.Linear, nn.Tanh, nn.Linear),
              'quantity': (nn.Linear, nn.Tanh, nn.Linear)}
    topology.update(ordered=nn.GRU, terrain=nn.Embedding, road=nn.Embedding,
                    gru=nn.GRUCell, policy_query=nn.Linear, value=nn.Linear)
    for name, children in groups.items():
        topology[name] = nn.Sequential
        topology.update({f'{name}.{index}': kind for index, kind in enumerate(children)})
    if {name: type(module) for name, module in model.named_modules()} != topology:
        raise ValueError("Native learner requires the exact canonical module topology")
    for name, children in groups.items():
        # Sequential uses insertion order even when its numeric keys are intact.
        actual = [(key, type(child)) for key, child in getattr(model, name)._modules.items()]
        if actual != [(str(index), kind) for index, kind in enumerate(children)]:
            raise ValueError("Native learner cannot bypass reordered Sequential children")
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            valid = (module.padding_idx == 0 and module.max_norm is None and module.norm_type == 2
                     and not module.scale_grad_by_freq and not module.sparse)
        elif isinstance(module, torch.nn.GRU):
            valid = (module.num_layers == 1 and module.bias and module.batch_first
                     and not module.bidirectional and module.dropout == 0)
            names = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
            valid = (valid and module._flat_weights_names == names and len(module._flat_weights) == len(names)
                     and all(value is getattr(module, name) for name, value in zip(names, module._flat_weights)))
        elif isinstance(module, torch.nn.GRUCell):
            valid = module.bias
        elif isinstance(module, torch.nn.Conv2d):
            valid = (module.stride == (1, 1) and module.padding == (1, 1) and module.dilation == (1, 1)
                     and module.groups == 1 and module.padding_mode == "zeros" and module.kernel_size == (3, 3))
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            valid = module.output_size == (4, 4)
        elif isinstance(module, torch.nn.Flatten):
            valid = module.start_dim == 1 and module.end_dim == -1
        elif isinstance(module, torch.nn.Linear):
            valid = module.bias is not None
        else:
            valid = True
        if not valid:
            raise ValueError("Native learner cannot bypass modified module attributes")
    if torch.get_num_threads() != 1:
        raise ValueError("Native learner parity requires one ATen intra-op thread per native worker")


def compile_history(model, inputs, records, backend):
    """Retain every chronological frame and exact selected-action support."""
    require_native_model(model)
    by_seq = {record.decision_seq: record for record in records}
    if len(by_seq) != len(records):
        raise ValueError("Duplicate native decision recipe")
    nodes, frames = [], []
    for seq, frame in enumerate(inputs):
        model._check_step(frame, model.initial_state())
        arena = _Arena(nodes)
        if frame.semantics is None:
            if seq in by_seq:
                raise ValueError("Loss decision has no recurrent input")
            frames.append({"root": -1, "map": None, "conditionals": None})
            continue
        # Preserve the order in which the reference populates its tree cache.
        map_recipe = None if frame.observation is None else _map_recipe(model, frame.observation, arena)
        root = arena.add(frame.semantics)
        conditionals = None
        if seq in by_seq:
            if frame.frame.kind != "decision":
                raise ValueError("Native loss recipe is not a decision frame")
            distribution = _RecipeDistribution(model, frame, arena)
            record = by_seq[seq]
            if distribution.support != record.support:
                raise ValueError("Native recipe support differs from recorded behavior")
            evaluated = distribution.evaluate(record.action)
            if tuple((v.name, v.selected_index, v.support_size) for v in evaluated.conditionals) != tuple(
                    (v.name, v.selected_index, v.support_size) for v in record.conditionals):
                raise ValueError("Native conditional recipe differs from recorded behavior")
            conditionals = distribution.recipes
        frames.append({"root": root, "map": map_recipe, "conditionals": conditionals})
    if any(type(seq) is not int or not 0 <= seq < len(inputs) for seq in by_seq):
        raise ValueError("Native decision lies outside the complete history")
    return backend.NativeHistory(nodes, frames)
