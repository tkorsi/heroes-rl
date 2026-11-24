"""A game's recipe as flat arrays, and plans built from them without Python loops.

``heroes.learner.plan.History`` keeps a recipe as Python lists and dictionaries
and builds a plan node by node. That is the reference. Here the same recipe is
held as arrays (``Compact``) and ``Compact.plan`` builds the plan of a range of
frames with array operations: the same nodes, maps and heads in the form the
MLX model reads, with the rows of a group in another order, which changes no
number. The whole-game update plans every game a few hundred loss windows at a
time, so this is what its time on the CPU depends on.
"""
from __future__ import annotations

import numpy as np

NUMBER, TEXT, POOLED, ORDERED = 0, 1, 2, 3
_KINDS = {"number": NUMBER, "text": TEXT, "node": POOLED, "ordered": ORDERED}
CATEGORICAL, QUANTITY, COORDINATE = 0, 1, 2
_HEADS = {"categorical": CATEGORICAL, "quantity": QUANTITY, "coordinate": COORDINATE}
_MAP_PARTS = ("terrain", "road", "flags", "object_attributes", "object_indices")


def _ranges(first, picked):
    """For the rows ``picked`` of a ragged array with offsets ``first``: their lengths and their flat positions."""
    starts = first[picked]
    counts = (first[picked + 1] - starts).astype(np.int64)
    total = int(counts.sum())
    if not total:
        return counts, np.zeros(0, dtype=np.int64)
    ends = np.cumsum(counts)
    flat = np.arange(total, dtype=np.int64) + np.repeat(starts - (ends - counts), counts)
    return counts, flat


def _padded(values, counts, fill, width=None):
    """A ragged list given flat as ``values`` with row lengths ``counts``, as a matrix padded with ``fill``."""
    rows = len(counts)
    width = int(counts.max()) if width is None and rows else (width or 0)
    result = np.full((rows, width), fill, dtype=values.dtype)
    if len(values):
        ends = np.cumsum(counts)
        column = np.arange(len(values), dtype=np.int64) - np.repeat(ends - counts, counts)
        result[np.repeat(np.arange(rows), counts), column] = values
    return result


def _small(values):
    """Whole numbers from 0 to 255 as bytes, anything else as it is: maps are most of a recipe's size."""
    if values.size and (values.min() < 0 or values.max() > 255 or not np.array_equal(values, values.astype(np.uint8))):
        return np.ascontiguousarray(values)
    return values.astype(np.uint8)


def _array(values):
    """A map part as an array: a recorded recipe holds arrays, a compiled one tensors."""
    return values if isinstance(values, np.ndarray) else values.numpy()


def _distinct_rows(rows):
    """The distinct rows of a float32 matrix and, for every row, which of them it is."""
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if not len(rows):
        return rows, np.zeros(0, dtype=np.int32)
    # Rows are compared as bytes: equal bytes are equal rows, and a row is 67 numbers of four bytes.
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return np.ascontiguousarray(rows[first]), inverse.reshape(-1).astype(np.int32)


class Compact:
    """One game's weight-independent recipe as arrays.

    Nodes: ``kind``, and by kind ``tag`` is the row of a number, the row of a
    text, or the node of an inner node's tag; ``first`` and ``children`` list
    the children; ``depth`` is 0 for a leaf and one more than the deepest part
    otherwise. Frames: ``root`` (-1 for a frame the model skips), ``map`` (the
    frame's map among the game's distinct maps, or -1) and ``own``. Maps:
    ``terrain``, ``road``, ``flags`` per distinct map and its objects as rows
    of ``descriptors`` at cells. Conditionals: for each trained decision, in
    order, the steps its action took.
    """

    @classmethod
    def from_history(cls, history, decisions):
        """The arrays of a compiled ``History`` whose trained decisions are the frames ``decisions``."""
        self = cls()
        nodes = history.nodes
        count = len(nodes)
        kind = np.empty(count, dtype=np.int8)
        tag = np.empty(count, dtype=np.int32)
        lengths = np.zeros(count, dtype=np.int64)
        numbers, texts, children = [], [], []
        depth = []  # a node is added after its parts, so their depths are already there
        deepest = depth.__getitem__
        for index, (name, value, parts) in enumerate(nodes):
            code = kind[index] = _KINDS[name]
            if code == NUMBER:
                tag[index] = len(numbers)
                numbers.append(value)
                depth.append(0)
            elif code == TEXT:
                tag[index] = len(texts)
                texts.append(bytes(value))
                depth.append(0)
            else:
                tag[index] = value
                lengths[index] = len(parts)
                children.extend(parts)
                depth.append(1 + max(depth[value], max(map(deepest, parts), default=0)))
        self.kind, self.tag = kind, tag
        self.depth = np.asarray(depth, dtype=np.int32)
        self.first = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.first[1:])
        self.children = np.asarray(children, dtype=np.int32)
        self.numbers = np.asarray(numbers, dtype=np.float32).reshape(-1, 2)
        longest = max((len(text) for text in texts), default=0)
        self.text_bytes = np.zeros((len(texts), longest), dtype=np.int32)  # byte + 1; 0 pads
        for row, text in enumerate(texts):
            self.text_bytes[row, :len(text)] = np.frombuffer(text, dtype=np.uint8).astype(np.int32) + 1
        self.text_length = np.asarray([len(text) for text in texts], dtype=np.int32)

        frames = history.frames
        self.root = np.asarray([frame["root"] for frame in frames], dtype=np.int32)
        self.map = np.full(len(frames), -1, dtype=np.int32)
        self.own = np.full(len(frames), -1, dtype=np.int32)
        seen, distinct = {}, []
        for index, frame in enumerate(frames):
            recipe = frame["map"]
            if recipe is None:
                continue
            key = tuple(id(recipe[name]) for name in _MAP_PARTS)  # frames that share the arrays share the map
            found = seen.get(key)
            if found is None:
                found = seen[key] = len(distinct)
                distinct.append(recipe)
            self.map[index], self.own[index] = found, recipe["own"]
        self.width = distinct[0]["width"] if distinct else 0
        self.height = distinct[0]["height"] if distinct else 0
        if any((recipe["width"], recipe["height"]) != (self.width, self.height) for recipe in distinct):
            raise ValueError("The maps of one game must have one size")
        size = self.width * self.height
        self.terrain = _small(np.stack([_array(recipe["terrain"]) for recipe in distinct])
                              if distinct else np.zeros((0, size), dtype=np.int64))
        self.road = _small(np.stack([_array(recipe["road"]) for recipe in distinct])
                           if distinct else np.zeros((0, size), dtype=np.int64))
        self.flags = _small(np.stack([_array(recipe["flags"]) for recipe in distinct])
                            if distinct else np.zeros((0, size, 5), dtype=np.float32))
        object_counts = np.asarray([len(recipe["object_indices"]) for recipe in distinct], dtype=np.int64)
        self.object_first = np.zeros(len(distinct) + 1, dtype=np.int64)
        np.cumsum(object_counts, out=self.object_first[1:])
        self.object_cells = (np.concatenate([_array(recipe["object_indices"]) for recipe in distinct]).astype(np.int32)
                             if distinct else np.zeros(0, dtype=np.int32))
        attributes = (np.concatenate([_array(recipe["object_attributes"]).reshape(-1, 67) for recipe in distinct])
                      if distinct else np.zeros((0, 67), dtype=np.float32))
        # Most objects stay as they are from map to map: keep each distinct row once.
        self.descriptors, self.object_rows = _distinct_rows(attributes)

        self.decisions = np.asarray(decisions, dtype=np.int32)
        steps = np.zeros(len(decisions), dtype=np.int64)
        head, selected, roots, extra = [], [], [], []
        choice_counts, choices = [], []
        repeated_choices = []  # flat indices whose candidate window was previously visited without effect
        numeric, offsets = [], []
        positions, cell_counts, cells = [], [], []
        for row, frame_index in enumerate(decisions):
            conditionals = frames[frame_index]["conditionals"]
            if conditionals is None:
                raise ValueError("A trained decision has no compiled conditionals")
            steps[row] = len(conditionals)
            for conditional in conditionals:
                code = _HEADS[conditional["kind"]]
                head.append(code)
                selected.append(conditional["selected"])
                if code == CATEGORICAL:
                    roots.append(-1)
                    extra.append(len(choice_counts))
                    choice_counts.append(len(conditional["roots"]))
                    if "window_memory" in conditional:
                        base = len(choices)
                        repeated_choices.extend(base + i for i, flag in enumerate(conditional["window_memory"]) if flag)
                    choices.extend(conditional["roots"])
                elif code == QUANTITY:
                    roots.append(conditional["root"])
                    extra.append(len(numeric))
                    numeric.append(conditional["numeric"].numpy())
                    offsets.append(conditional["offsets"].numpy())
                else:
                    roots.append(conditional["root"])
                    extra.append(len(positions))
                    grid = conditional["positions"].numpy()
                    positions.append(grid)
                    indices = conditional["indices"]
                    indices = np.arange(len(grid)) if indices is None else indices.numpy()
                    cell_counts.append(len(indices))
                    cells.append(indices)
        self.step_first = np.zeros(len(decisions) + 1, dtype=np.int64)
        np.cumsum(steps, out=self.step_first[1:])
        self.step_head = np.asarray(head, dtype=np.int8)
        self.step_selected = np.asarray(selected, dtype=np.int32)
        self.step_root = np.asarray(roots, dtype=np.int32)
        self.step_extra = np.asarray(extra, dtype=np.int64)  # row among the steps of its own head
        self.choice_first = np.zeros(len(choice_counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(choice_counts, dtype=np.int64), out=self.choice_first[1:])
        self.choices = np.asarray(choices, dtype=np.int32)
        self.window_memory = None
        if repeated_choices:
            self.window_memory = np.zeros(len(choices), dtype=bool)
            self.window_memory[repeated_choices] = True
        self.numeric = np.asarray(numeric, dtype=np.float32).reshape(-1, 4)
        self.offsets = np.asarray(offsets, dtype=np.float32).reshape(-1, 2)
        self.positions = (np.stack(positions).astype(np.float32) if positions
                          else np.zeros((0, size, 2), dtype=np.float32))
        self.cell_first = np.zeros(len(cell_counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(cell_counts, dtype=np.int64), out=self.cell_first[1:])
        self.cells = (np.concatenate(cells).astype(np.int32) if cells else np.zeros(0, dtype=np.int32))
        return self

    @property
    def frame_count(self):
        return len(self.root)

    def jobs(self, start=0, end=None):
        """The frames of ``[start, end)`` the model reads, ascending."""
        end = self.frame_count if end is None else end
        return (np.flatnonzero(self.root[start:end] >= 0) + start).astype(np.int32)

    def _closure(self, seeds):
        """Every node the nodes ``seeds`` are built from, themselves included, as a mask over all nodes."""
        needed = np.zeros(len(self.kind), dtype=bool)
        frontier = np.unique(seeds)
        while len(frontier):
            needed[frontier] = True
            inner = frontier[self.kind[frontier] >= POOLED]
            _, flat = _ranges(self.first, inner)
            parts = np.unique(np.concatenate((self.tag[inner], self.children[flat])))
            frontier = parts[~needed[parts]]
        return needed

    def plan(self, start, end, low, high):
        """The plan of the frames ``[start, end)`` with the trained decisions ``[low, high)`` of the game.

        The result is what ``History.plan`` gives for that range after
        ``plans._portable_plan`` and conversion to host arrays: every array is
        int32 or float32. ``jobs`` are the frames of the range the model reads.
        """
        jobs = self.jobs(start, end)
        decisions = self.decisions[low:high]
        steps = np.arange(self.step_first[low], self.step_first[high], dtype=np.int64)
        head, extra = self.step_head[steps], self.step_extra[steps]
        step_decision = (np.repeat(np.arange(high - low), np.diff(self.step_first[low:high + 1]))).astype(np.int32)
        categorical = np.flatnonzero(head == CATEGORICAL)
        choice_counts, choice_flat = _ranges(self.choice_first, extra[categorical])
        mapped = jobs[self.map[jobs] >= 0]
        seeds = np.concatenate((self.root[jobs], self.own[mapped], self.choices[choice_flat],
                                self.step_root[steps][head != CATEGORICAL]))
        needed = self._closure(seeds)
        ids = np.flatnonzero(needed)
        count = len(ids)
        local = np.full(len(needed) + 1, count, dtype=np.int32)  # the last entry stands for "no node"
        local[ids] = np.arange(count, dtype=np.int32)

        groups = []
        kind, depth = self.kind[ids], self.depth[ids]
        sizes = self.first[ids + 1] - self.first[ids]  # children of an inner node, bytes of a text
        sizes[kind == TEXT] = self.text_length[self.tag[ids[kind == TEXT]]]
        for level in range(int(depth.max()) + 1 if count else 0):
            at_level = depth == level
            rows = ids[at_level & (kind == NUMBER)]
            if len(rows):
                groups.append(("number", local[rows], {"values": self.numbers[self.tag[rows]]}, len(rows)))
            rows = ids[at_level & (kind == TEXT) & (sizes == 0)]
            if len(rows):
                groups.append(("text", local[rows], {}, len(rows)))
            rows = ids[at_level & (kind == TEXT) & (sizes > 0)]
            if len(rows):
                lengths = self.text_length[self.tag[rows]]
                groups.append(("text", local[rows], {
                    "ids": np.ascontiguousarray(self.text_bytes[self.tag[rows], :int(lengths.max())]),
                    "lengths": (lengths - 1).astype(np.int32)}, len(rows)))
            rows = ids[at_level & (kind == ORDERED) & (sizes == 0)]
            if len(rows):
                groups.append(("ordered", local[rows], {"tags": local[self.tag[rows]]}, len(rows)))
            rows = ids[at_level & (kind == ORDERED) & (sizes > 0)]
            if len(rows):
                counts, flat = _ranges(self.first, rows)
                groups.append(("ordered", local[rows], {
                    "tags": local[self.tag[rows]], "children": _padded(local[self.children[flat]], counts, count),
                    "lengths": (counts - 1).astype(np.int32)}, len(rows)))
            rows = ids[at_level & (kind == POOLED)]
            if len(rows):
                counts, flat = _ranges(self.first, rows)
                groups.append(("pooled", local[rows], {
                    "tags": local[self.tag[rows]], "flat": local[self.children[flat]],
                    "offsets": (np.cumsum(counts) - counts).astype(np.int32), "counts": counts.astype(np.int32),
                    "scale": np.sqrt(counts.astype(np.float64)).astype(np.float32)}, len(rows)))

        maps, reuse = None, np.zeros(0, dtype=np.int32)
        size = self.width * self.height
        if len(mapped):
            distinct, reuse = np.unique(self.map[mapped], return_inverse=True)
            reuse = reuse.reshape(-1).astype(np.int32)
            object_counts, object_flat = _ranges(self.object_first, distinct)
            maps = {
                "width": self.width, "height": self.height, "count": len(distinct),
                "positions": np.searchsorted(jobs, mapped).astype(np.int32),
                "own": local[self.own[mapped]], "reuse": reuse,
                "terrain": self.terrain[distinct].astype(np.int32), "road": self.road[distinct].astype(np.int32),
                "flags": self.flags[distinct].astype(np.float32),
                "attributes": self.descriptors[self.object_rows[object_flat]],
                "indices": (self.object_cells[object_flat]
                            + np.repeat(np.arange(len(distinct), dtype=np.int64) * size, object_counts)).astype(np.int32),
            }

        decision_jobs = np.searchsorted(jobs, decisions).astype(np.int32)
        if len(decisions) and not np.array_equal(jobs[decision_jobs], decisions):
            raise ValueError("A trained decision is not a frame of this range")
        heads = []
        selected, state = self.step_selected[steps], decision_jobs[step_decision]
        if len(categorical):
            width = int(choice_counts.max())
            data = {
                "decision": step_decision[categorical], "state": state[categorical],
                "selected": selected[categorical].reshape(-1, 1),
                "mask": np.where(np.arange(width) < choice_counts[:, None], 0.0, -np.inf).astype(np.float32),
                "roots": _padded(local[self.choices[choice_flat]], choice_counts, count, width)}
            if getattr(self, "window_memory", None) is not None:
                flags = self.window_memory[choice_flat]
                if flags.any():
                    data["window_memory"] = _padded(flags, choice_counts, False, width)
            heads.append(("categorical", data))
        coordinate = np.flatnonzero(head == COORDINATE)
        if len(coordinate):
            # The map of a decision's own frame gives the cells its coordinates are scored on.
            frame_map = np.full(len(jobs), -1, dtype=np.int32)
            frame_map[np.searchsorted(jobs, mapped)] = reuse
            cell_counts, _ = _ranges(self.cell_first, extra[coordinate])
            # ``(n - 1).bit_length()``: the power of two that holds n choices.
            buckets = np.asarray([(int(n) - 1).bit_length() for n in cell_counts], dtype=np.int64)
            for bucket in np.unique(buckets):
                picked = coordinate[buckets == bucket]
                counts, flat = _ranges(self.cell_first, extra[picked])
                width = 2 ** int(bucket)
                cells = frame_map[state[picked]]
                if (cells < 0).any():
                    raise ValueError("A coordinate decision has no map")
                heads.append(("coordinate", {
                    "decision": step_decision[picked], "state": state[picked],
                    "selected": selected[picked].reshape(-1, 1),
                    "mask": np.where(np.arange(width) < counts[:, None], 0.0, -np.inf).astype(np.float32),
                    "roots": local[self.step_root[steps][picked]], "cells": cells,
                    "positions": self.positions[extra[picked]],
                    "indices": _padded(self.cells[flat], counts, 0, width)}))
        quantity = np.flatnonzero(head == QUANTITY)
        if len(quantity):
            heads.append(("quantity", {
                "decision": step_decision[quantity], "state": state[quantity],
                "selected": selected[quantity].reshape(-1, 1),
                "mask": np.zeros((len(quantity), 2), dtype=np.float32),
                "roots": local[self.step_root[steps][quantity]],
                "numeric": self.numeric[extra[quantity]], "offsets": self.offsets[extra[quantity]]}))
        return {"n": count, "groups": groups, "jobs": jobs, "roots": local[self.root[jobs]], "maps": maps,
                "heads": heads, "decision_states": decision_jobs, "decision_count": len(decisions)}
