"""Compile one game's frames into a weight-independent recipe for the learner.

The recipe lists which semantic trees every frame reads and, for each trained
decision, which conditional choices the recorded action took. Action
enumeration reuses the policy distribution's own walk with its tensor scoring
replaced, so there is exactly one definition of the action space.
"""
from __future__ import annotations

from math import copysign, log1p
from ..model.policy import ConditionalProbability, ContextualDistribution, _text, observation_semantics
from .plan import History


class _Arena:
    def __init__(self, nodes, shared=None):
        self.nodes = nodes
        self.shared = {} if shared is None else shared
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
        # Share immutable recipes across frames only after the frame-local
        # equality choice above. Keep original numeric identity and signed zero:
        # different inputs may round to the same preprocessed double features.
        key = ((kind, type(tree[1]), tree[1], copysign(1., tree[1])) if kind == "number"
               else (kind, node[1], tuple(node[2])))
        index = self.shared.get(key)
        if index is None:
            index = len(self.nodes)
            self.nodes.append(node)
            self.shared[key] = index
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

    def _coordinate_logits(self, semantic, subject, width, height):
        return {"kind": "coordinate", "root": self._features(semantic),
                "positions": self._coordinate_positions(subject, width, height),
                "indices": None}

    def _coordinate_subset(self, logits, coordinates):
        return {**logits, "indices": self.model._long(coordinates)}

    def _choose(self, name, logits, selected, generator, trace):
        if generator is not None or type(selected) is not int:
            raise ValueError("Recipes describe recorded actions; they never sample")
        kind = logits["kind"]
        size = (len(logits["roots"]) if kind == "categorical" else 2 if kind == "quantity"
                else len(logits["indices"]) if logits["indices"] is not None else len(logits["positions"]))
        if not 0 <= selected < size:
            raise ValueError("Recorded action is outside the compiled conditional support")
        self.recipes.append({**logits, "selected": selected})
        trace.append(ConditionalProbability(name, selected, size, self.state.new_zeros(())))
        return selected


def _map_recipe(model, observation, arena, cache=None):
    key = (observation.size, observation.tiles, tuple(hero.position for hero in observation.heroes),
           tuple(town.position for town in observation.towns),
           tuple((obj.position, obj.kind, obj.subtype, obj.owner) for obj in observation.objects))
    if cache is not None and key in cache:
        return {**cache[key], 'own': arena.add(observation_semantics(observation))}
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
    result = {"width": width, "height": height, "terrain": model._long(terrain), "road": model._long(road),
            "flags": model._float(flags), "object_attributes": model._float(attributes).reshape(-1, 67),
            "object_indices": model._long([obj.position[1] * width + obj.position[0] for obj in objects]),
            "own": arena.add(observation_semantics(observation))}
    if cache is not None: cache[key] = {k: v for k, v in result.items() if k != 'own'}
    return result


def compile_history(model, inputs, actions):
    """Build the recipe for every frame of ``inputs`` (an iterable of encoded
    frames) and the conditionals of each decision in ``actions`` {seq: Action}."""
    nodes, frames, shared, map_cache = [], [], {}, {}
    pending = set(actions)
    last = None  # the observation compiled last, its map recipe and the arena's trees after it
    for seq, frame in enumerate(inputs):
        arena = _Arena(nodes, shared)
        if frame.semantics is None:
            frames.append({"root": -1, "map": None, "conditionals": None})
            continue
        # Map first, then the frame, then its conditionals: node order is part of the recipe.
        if frame.observation is None:
            map_recipe = None
        elif last is not None and last[0] is frame.observation:
            # A decision shares the observation of the view before it. Compiling it
            # again would add no node and leave the arena with exactly these trees.
            map_recipe, arena.ids = dict(last[1]), dict(last[2])
        else:
            map_recipe = _map_recipe(model, frame.observation, arena, map_cache)
            last = (frame.observation, map_recipe, dict(arena.ids))
        root = arena.add(frame.semantics)
        conditionals = None
        if seq in actions:
            distribution = _RecipeDistribution(model, frame, arena)
            distribution.evaluate(actions[seq])
            conditionals = distribution.recipes
            pending.discard(seq)
        frames.append({"root": root, "map": map_recipe, "conditionals": conditionals})
    if pending:
        raise ValueError("A trained decision is not a decision frame of this history")
    return History(nodes, frames)
