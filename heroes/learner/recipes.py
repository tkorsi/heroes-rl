"""Compile one game's frames into a weight-independent recipe for the learner.

The recipe lists which semantic trees every frame reads and, for each trained
decision, which conditional choices the recorded action took. Action
enumeration reuses the policy distribution's own walk with its tensor scoring
replaced, so there is exactly one definition of the action space.
"""
from __future__ import annotations

from math import copysign, log1p
from ..model.policy import (ConditionalProbability, ContextualDistribution, _Kept, _exact_tree, _text, ground, map_key,
                            object_row, observation_semantics)
from .plan import History


_KNOWN_LIMIT = 16384  # exact trees whose node is remembered across frames; a miss only walks the tree again
_KEPT_LIMIT = 4096  # kept trees whose node (and subtree ids) are remembered by identity; a miss only walks again


class _Arena:
    """The trees of one frame and the nodes they share with every other frame of the game.

    ``add`` gives the node of a top-level tree. While that tree and every tree
    this frame added before are exact (see ``policy._exact_tree``), equality
    is identity of content, the node of a tree is a function of the tree alone
    once it is among the shared nodes, and a tree met before is not entered
    again (``known``): the frame's own equality cache ``ids`` is not filled,
    the root is only noted. A tree that is not exact must meet the ``ids`` the
    plain walk would have filled, so ``_settle`` walks the noted roots first.
    """
    lazy = True  # tests switch it off to compare with the plain walk

    def __init__(self, nodes, shared=None, kept=None, known=None):
        self.nodes = nodes
        self.shared = {} if shared is None else shared
        # A new equality cache per frame preserves signed-zero first-use and
        # Python's existing numeric equality rules within that frame only.
        self.ids = {}
        self.kept = {} if kept is None else kept  # id of a kept tree -> (the tree, its node, the ids of its subtrees)
        self.exact = True  # until this frame adds a leaf that equals another without being it
        # Across frames: exact tree -> its node, and id of a kept exact tree -> (the tree, its node).
        self.known = ({}, {}) if known is None else known
        self.unsettled = []  # exact roots of this frame whose subtrees ``ids`` does not hold yet

    def add(self, tree):
        """The node of the top-level ``tree``."""
        if self.lazy and self.exact and _exact_tree(tree):
            self.unsettled.append(tree)
            return self._add_exact(tree)
        self._settle()
        return self._add_any(tree)

    def _settle(self):
        """Fill ``ids`` with the subtrees of the exact roots met so far; no node is new."""
        unsettled, self.unsettled = self.unsettled, []
        for tree in unsettled:
            self._add_any(tree)

    def state(self):
        """What a later frame that shares this frame's trees starts from; see ``restore``."""
        return dict(self.ids), tuple(self.unsettled), self.exact

    def restore(self, state):
        self.ids, self.unsettled, self.exact = dict(state[0]), list(state[1]), state[2]

    def _add_exact(self, tree):
        """The node of an exact tree: the nodes the plain walk gives, made in the order it makes them."""
        trees, kept = self.known
        if type(tree) is _Kept:
            known = kept.get(id(tree))
            if known is not None:
                return known[1]
        index = trees.get(tree)
        if index is None:
            kind = tree[0]
            if kind == "text":
                node = (kind, tree[1].encode("utf-8"), [])
            elif kind == "number":
                value = tree[1]
                node = (kind, [copysign(log1p(abs(value)), value), float(value == 0)], [])
            else:
                tag, children = tree[1:]
                children = [self._add_exact(child) for child in children]
                node = (kind, self._add_exact(_text(tag)), children)
            key = ((kind, type(tree[1]), tree[1], copysign(1., tree[1])) if kind == "number"
                   else (kind, node[1], tuple(node[2])))
            index = self.shared.get(key)
            if index is None:
                index = len(self.nodes)
                self.nodes.append(node)
                self.shared[key] = index
            if len(trees) >= _KNOWN_LIMIT:
                trees.clear()  # bounded memory; frame roots are met once and would only pile up
            trees[tree] = index
        if type(tree) is _Kept:
            if len(kept) >= _KEPT_LIMIT:
                kept.clear()  # bounded memory: every entry keeps its tree alive for the rest of the game
            kept[id(tree)] = (tree, index)  # holds the tree, so its id names nothing else
        return index

    def _add_any(self, tree):
        """The node of ``tree``; the frame's ids gain every subtree they did not hold.

        A kept tree is walked once per game. While equality is identity for
        everything this frame added and for the tree, each of its subtrees
        gets the node it got before, a function of the subtree alone once it
        is among the shared nodes, and no node is new: the later meeting adds
        those same ids at once.
        """
        if type(tree) is not _Kept or not tree.exact or not self.exact:
            return self._add(tree)
        visit = self.kept.get(id(tree))
        if visit is None:
            index = self._add(tree)
            ids, stack = {}, [tree]
            while stack:
                item = stack.pop()
                if item in ids:
                    continue
                inner = self.kept.get(id(item)) if type(item) is _Kept else None
                if inner is not None:
                    ids.update(inner[2])
                    continue
                ids[item] = self.ids[item]
                if item[0] not in ("text", "number"):
                    label = _text(item[1])
                    ids[label] = self.ids[label]
                    stack.extend(item[2])
            if len(self.kept) >= _KEPT_LIMIT:
                self.kept.clear()  # bounded memory, as above
            self.kept[id(tree)] = (tree, index, ids)  # holds the tree, so its id names nothing else
            return index
        self.ids.update(visit[2])
        return visit[1]

    def _add(self, tree):
        if tree in self.ids:
            return self.ids[tree]
        kind = tree[0]
        if kind == "text":
            self.exact = self.exact and type(tree[1]) is str
            node = (kind, tree[1].encode("utf-8"), [])
        elif kind == "number":
            value = tree[1]
            self.exact = self.exact and type(value) is float and (value > 0 or value < 0 or copysign(1., value) > 0)
            node = (kind, [copysign(log1p(abs(value)), value), float(value == 0)], [])
        else:
            tag, children = tree[1:]
            children = [self._add_any(child) for child in children]
            node = (kind, self._add_any(_text(tag)), children)
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


def _small_map(width, height, terrain, road, flags, attributes, indices):
    """The map's parts as arrays of the smallest type that holds them exactly.

    ``Compact`` stores a game's maps this way; a recipe recorded for it never
    holds them as 64-bit tensors, which were most of an actor's memory.
    """
    import numpy as np

    def small(values, wide, shape):
        array = np.asarray(values, dtype=wide).reshape(shape)
        narrow = array.astype(np.uint8)
        return narrow if np.array_equal(narrow, array) else array

    return {"width": width, "height": height, "terrain": small(terrain, np.int64, (-1,)),
            "road": small(road, np.int64, (-1,)), "flags": small(flags, np.float32, (-1, 5)),
            "object_attributes": small(attributes, np.float32, (-1, 67)),
            "object_indices": np.asarray(indices, dtype=np.int32)}


_TILES, _FLAGS, _OBJECTS = "tiles", "flags", "objects"  # the map cache's own tables; a map's key is a tuple


def _map_recipe(model, observation, arena, cache=None, small=False):
    """The map a frame's observation shows, as the learner reads it, and the node of its own part.

    Observations with equal ``map_key`` get the very same parts, which is how
    the learner knows that frames share a map. A map differs from the one
    before it mostly in where the heroes stand, so its parts are kept apart:
    terrain and roads by the tiles, the flags by tiles, heroes and towns, the
    objects by themselves. What the tiles say comes from ``policy.ground``,
    which keeps it from view to view and knows two equal tile tuples by the
    same bytes: no tile is read here.
    """
    if cache is None:
        cache = {}
    tokens = cache.get(_TILES)
    if tokens is None:
        tokens = cache[_TILES] = {}  # the content of a tile tuple -> its number
        cache[_FLAGS], cache[_OBJECTS] = {}, {}
    seen = ground(observation.tiles, observation.size)
    token = tokens.get(seen[5])
    if token is None:
        token = tokens[seen[5]] = len(tokens)
    heroes = tuple(hero.position for hero in observation.heroes)
    towns = tuple(town.position for town in observation.towns)
    described = tuple((obj.position, obj.kind, obj.subtype, obj.owner) for obj in observation.objects)
    key = (observation.size, token, heroes, towns, described)  # ``map_key`` with the tiles by their number
    parts = cache.get(key)
    if parts is None:
        width, height, _ = observation.size
        if width > model.config.max_map_side or height > model.config.max_map_side:
            raise ValueError("Map exceeds declared model configuration")
        flags_key = (observation.size, token, heroes, towns)
        grounds = cache[_FLAGS].get((observation.size, token))
        flags = cache[_FLAGS].get(flags_key)
        if grounds is None:
            if small:
                made = _small_map(width, height, seen[2], seen[3], [], [], [])
                grounds = (made["terrain"], made["road"])
            else:
                grounds = (model._long(seen[2].tolist()), model._long(seen[3].tolist()))
            cache[_FLAGS][(observation.size, token)] = grounds
        if flags is None:
            rows = seen[4].copy()
            for position in heroes:
                rows[position[1] * width + position[0], 3] += 1.
            for position in towns:
                rows[position[1] * width + position[0], 4] += 1.
            flags = cache[_FLAGS][flags_key] = (_small_map(width, height, [], [], rows, [], [])["flags"] if small
                                               else model._float(rows.tolist()))
        things = cache[_OBJECTS].get((width, described))
        if things is None:
            objects = sorted(described)  # by position, kind, subtype, owner
            attributes = [object_row(kind, subtype, owner) for _, kind, subtype, owner in objects]
            indices = [position[1] * width + position[0] for position, _, _, _ in objects]
            if small:
                made = _small_map(width, height, [], [], [], attributes, indices)
                things = (made["object_attributes"], made["object_indices"])
            else:
                things = (model._float(attributes).reshape(-1, 67), model._long(indices))
            cache[_OBJECTS][(width, described)] = things
        parts = cache[key] = {"width": width, "height": height, "terrain": grounds[0], "road": grounds[1],
                              "flags": flags, "object_attributes": things[0], "object_indices": things[1]}
    return {**parts, "own": arena.add(observation_semantics(observation))}


def _frame_recipe(model, frame, arena, state, small=False):
    """The map recipe and the root of one frame the model reads; ``state`` carries what frames share.

    Map first, then the frame: node order is part of the recipe. ``state`` is
    ``[the map cache, (the observation compiled last, its map recipe, the
    arena's state after it)]``. With ``small`` the map's parts are small
    arrays instead of tensors; see ``_small_map``.
    """
    last = state[1]
    if frame.observation is None:
        map_recipe = None
    elif last is not None and last[0] is frame.observation:
        # A decision shares the observation of the view before it. Compiling it
        # again would add no node and leave the arena with exactly these trees.
        map_recipe = dict(last[1])
        arena.restore(last[2])
    else:
        map_recipe = _map_recipe(model, frame.observation, arena, state[0], small)
        state[1] = (frame.observation, map_recipe, arena.state())
    return map_recipe, arena.add(frame.semantics)


class Recorder:
    """Compiles one seat's recipe while its game is played.

    ``feed`` takes the seat's encoded frames in order, ``sampled`` what the
    policy's walk read when it chose an action, and ``finish`` returns the
    ``History`` that ``compile_history`` builds from the finished game, node
    for node. Only a trained decision has conditionals, and whether a decision
    is trained is known only later, so the frames after a decision wait until
    the seat's next decision arrives: by then the action's result has been
    seen or it has not. ``finish`` settles the last decision from the finished
    trajectory and returns None if what was recorded on the way differs from
    it; the caller then compiles the game the slow way. With ``small_maps``
    the recipe holds its maps as small arrays, which is all ``Compact`` needs
    of them; the window update reads tensors and needs the recipe without it.
    """

    def __init__(self, model, small_maps=False):
        self.model, self.small_maps = model, small_maps
        self.nodes, self.frames, self.shared, self.kept, self.known = [], [], {}, {}, ({}, {})
        self.state = [{}, None]
        self.waiting = []  # frames met after the open decision
        self.open = None  # (frame number of the decision whose fate is not known yet, its arena)
        self.results = set()  # decisions whose result is among the waiting frames
        self.recordings = {}  # decision frame -> Recording of its sampled action
        self.trained = []  # decisions that got conditionals
        self.fed = 0
        self.failed = False  # recording met something it cannot compile; ``finish`` then returns None

    def sampled(self, seq, recording):
        self.recordings[seq] = recording

    def feed(self, item):
        """The next encoded frame of the seat."""
        self.fed += 1
        if self.failed:
            return
        try:
            if self.open is None:
                self._frame(item)
            elif item.frame.kind == "decision":
                # The policy is about to answer the next decision: the open one is trained if its result came.
                self._close(self.open[0] in self.results)
                self._frame(item)
            else:
                if item.frame.kind == "result":
                    self.results.add(item.frame.decision_seq)
                self.waiting.append(item)
        except Exception:  # the game goes on; it is compiled after its end instead
            self.failed = True

    def _frame(self, item):
        arena = _Arena(self.nodes, self.shared, self.kept, self.known)
        if item.semantics is None:
            self.frames.append({"root": -1, "map": None, "conditionals": None})
            return
        map_recipe, root = _frame_recipe(self.model, item, arena, self.state, self.small_maps)
        self.frames.append({"root": root, "map": map_recipe, "conditionals": None})
        if item.frame.kind == "decision":
            self.open = (len(self.frames) - 1, arena)

    def _close(self, trained):
        """Settle the open decision, then compile the frames that waited for it."""
        seq, arena = self.open
        recording = self.recordings.pop(seq, None)
        if trained:
            if recording is None:
                raise ValueError("A trained decision was not sampled by this policy")
            nodes = [arena.add(tree) for tree in recording.trees]  # in the order the walk asked for them
            conditionals = []
            for step in recording.steps:
                if step["kind"] == "categorical":
                    conditionals.append({"kind": "categorical", "roots": [nodes[index] for index in step["roots"]],
                                         "selected": step["selected"]})
                else:
                    conditionals.append({**step, "root": nodes[step["root"]]})
            self.frames[seq]["conditionals"] = conditionals
            self.trained.append(seq)
        self.open = None
        waiting, self.waiting = self.waiting, []
        self.results.clear()
        for position, item in enumerate(waiting):
            self._frame(item)
            if self.open is not None:  # only when ``finish`` feeds frames the policy never saw
                self.waiting = waiting[position + 1:]
                break

    def finish(self, rest, decisions):
        """Close the recipe of a finished game whose trained decisions are ``decisions``.

        ``rest`` are the encoded frames the policy never observed, in order.
        Returns the ``History``, or None when the decisions that were compiled
        while playing are not exactly ``decisions``.
        """
        if self.failed:
            return None
        wanted = set(decisions)
        self.waiting.extend(rest)
        self.fed += len(rest)
        try:
            while self.open is not None or self.waiting:
                if self.open is not None:
                    self._close(self.open[0] in wanted)
                else:
                    waiting, self.waiting = self.waiting, []
                    for position, item in enumerate(waiting):
                        self._frame(item)
                        if self.open is not None:
                            self.waiting = waiting[position + 1:]
                            break
        except Exception:
            return None
        if self.trained != list(decisions) or len(self.frames) != self.fed:
            return None
        return History(self.nodes, self.frames)


def compile_history(model, inputs, actions):
    """Build the recipe for every frame of ``inputs`` (an iterable of encoded
    frames) and the conditionals of each decision in ``actions`` {seq: Action}."""
    nodes, frames, shared, kept, known = [], [], {}, {}, ({}, {})
    pending = set(actions)
    state = [{}, None]  # the map cache; the observation compiled last, its map recipe and the arena's state after it
    for seq, frame in enumerate(inputs):
        arena = _Arena(nodes, shared, kept, known)
        if frame.semantics is None:
            frames.append({"root": -1, "map": None, "conditionals": None})
            continue
        # Map first, then the frame, then its conditionals: node order is part of the recipe.
        map_recipe, root = _frame_recipe(model, frame, arena, state)
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
