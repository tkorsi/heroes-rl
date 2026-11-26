"""The policy while it plays, with its arithmetic in ``fastnet.c`` instead of torch.

An actor asks the network for one frame at a time. Each request is a few
small operations, and through a tensor library every one of them costs more
in dispatch than in arithmetic: about half a millisecond per decision, a
quarter of an actor's time. ``FastPolicy`` is the acting half of
``ContextualActorCritic`` over a frozen copy of its weights: the same trees,
the same memo, the same walk through an action's choices, with the numbers
computed by plain loops and BLAS calls in C.

What it computes is what the torch model computes, bit for bit: the same
Accelerate routines on the same layouts, torch's own ``tanh`` and ``exp``
(the SLEEF functions exported by the ``libtorch_cpu`` of this process), and
torch's order of operations in between. Sampling draws through
``torch.multinomial`` from the same generator. A game played this way is
therefore the game torch would have played: the same actions, the same
recorded log-probabilities and values, the same digest of everything the
learner reads. ``tests/test_fast.py`` holds every operation to exact equality.

The model is used for acting only: no gradients, no training mode. Anything
else (the learner, the reference update, tests of the model itself) uses
``ContextualActorCritic``. Where the kernel cannot be built, or torch's
library does not export its two functions, actors play through torch as
before; ``HEROES_FASTNET=0`` asks for that.
"""
from __future__ import annotations

import ctypes
import hashlib
from math import copysign, log1p, sqrt
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
import torch

from . import native as _native
from . import policy as _policy
from .policy import (ContextualDistribution, PolicyValue, StepOutput, _exact_tree, _Kept, map_key,
                     observation_semantics)

_SOURCE = Path(__file__).with_name("fastnet.c")
# The weights in the order of the enum in fastnet.c.
_PARAMETERS = (
    "byte.weight", "text_gru.weight_ih_l0", "text_gru.weight_hh_l0", "text_gru.bias_ih_l0", "text_gru.bias_hh_l0",
    "number.0.weight", "number.0.bias", "tree.0.weight", "tree.0.bias",
    "ordered.weight_ih_l0", "ordered.weight_hh_l0", "ordered.bias_ih_l0", "ordered.bias_hh_l0",
    "terrain.weight", "road.weight", "object_encoder.0.weight", "object_encoder.0.bias",
    "map_cnn.0.weight", "map_cnn.0.bias", "map_cnn.2.weight", "map_cnn.2.bias", "map_global.2.weight", "map_global.2.bias",
    "gru.weight_ih", "gru.weight_hh", "gru.bias_ih", "gru.bias_hh", "policy_query.weight", "policy_query.bias",
    "value.weight", "value.bias", "coordinate.0.weight", "coordinate.0.bias", "coordinate.2.weight", "coordinate.2.bias",
    "quantity.0.weight", "quantity.0.bias", "quantity.2.weight", "quantity.2.bias")
ARENA_SLOTS = 1 << 21  # 32 numbers each: 256 MB at most, given back to the next game's trees when it fills


def enabled():
    return os.environ.get("HEROES_FASTNET", "1") != "0"


def _build_folder():
    return Path(os.environ.get("HEROES_ARTIFACTS", "~/Projects/Heroes-artifacts")).expanduser() / "build"


_LIBRARY = None
_ROOT = None  # fastnet.c's walk over a semantic tree, where it was built against this interpreter
_CALLS = None  # (used, step, value, query, logits): fastnet.c's per-frame calls as functions of this interpreter
_PYTHON_WALK = _policy._walk_exact  # the walk the C one stands in for; tests compare the two


def _built(source, folder):
    """The library of this source for this interpreter, compiled if it is not there yet."""
    headers = sysconfig.get_paths().get("include") or ""
    with_walks = (Path(headers) / "Python.h").is_file()
    stamp = hashlib.sha256(source + (f"|{sys.version}|{headers}".encode() if with_walks else b"")).hexdigest()[:16]
    target = folder / f"fastnet-{stamp}.dylib"
    if not target.exists():
        folder.mkdir(parents=True, exist_ok=True)
        # Several actors may arrive here at once: each builds its own file, the first rename wins.
        handle, name = tempfile.mkstemp(prefix="fastnet-", suffix=".dylib", dir=folder)
        os.close(handle)
        # No product is fused into a sum: torch rounds after every operation, and so does this.
        command = ["cc", "-O3", "-ffp-contract=off", "-DACCELERATE_NEW_LAPACK", "-shared", "-fPIC",
                   "-framework", "Accelerate", str(_SOURCE), "-o", name]
        if with_walks:  # the tree walks are written against the interpreter's objects, found when it loads them
            command[4:4] = ["-DFASTNET_PYTHON", f"-I{headers}", "-undefined", "dynamic_lookup"]
        subprocess.run(command, check=True, capture_output=True)
        os.replace(name, target)
    return target


def library():
    """The compiled ``fastnet.c``, built once per version of the source; None when it cannot be built here."""
    global _LIBRARY, _ROOT, _CALLS
    if _LIBRARY is not None:
        return _LIBRARY or None
    try:
        target = _built(_SOURCE.read_bytes(), _build_folder())
        lib = ctypes.PyDLL(str(target))  # calls keep the interpreter lock: the arena has one user at a time
        # torch's own exp and tanh of four numbers, from the library this process computes with.
        theirs = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.dylib"))
        lib.fn_math.restype, lib.fn_math.argtypes = ctypes.c_int, (ctypes.c_void_p, ctypes.c_void_p)
        if lib.fn_math(ctypes.cast(theirs.Sleef_expf4_u10, ctypes.c_void_p),
                       ctypes.cast(theirs.Sleef_tanhf4_u10, ctypes.c_void_p)):
            raise OSError("torch's exp and tanh were not found")
    except Exception:
        _LIBRARY = False
        return None
    pointer, number, real = ctypes.c_void_p, ctypes.c_long, ctypes.c_float
    for name, result, arguments in (
            ("fn_create", pointer, (number,)), ("fn_destroy", None, (pointer,)),
            ("fn_set", ctypes.c_int, (pointer, ctypes.c_int, pointer, number)), ("fn_ready", ctypes.c_int, (pointer,)),
            ("fn_reset", number, (pointer,)), ("fn_used", number, (pointer,)), ("fn_read", None, (pointer, number, pointer)),
            ("fn_number", number, (pointer, real, real)), ("fn_text", number, (pointer, ctypes.c_char_p, number)),
            ("fn_node", number, (pointer, number, pointer, number, ctypes.c_int)),
            ("fn_step", None, (pointer, pointer, pointer, number, number, number)),
            ("fn_cell", None, (pointer, pointer, pointer, pointer)),
            ("fn_value", real, (pointer, pointer)), ("fn_query", None, (pointer, pointer, pointer)),
            ("fn_logits", ctypes.c_int, (pointer, pointer, pointer, number, pointer)),
            ("fn_quantity", None, (pointer, pointer, number, pointer, pointer)),
            ("fn_coordinate", ctypes.c_int, (pointer, pointer, number, pointer, pointer, number, pointer)),
            ("fn_map", number, (pointer, pointer, pointer, pointer, pointer, pointer, number, number, number, pointer))):
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    if hasattr(lib, "fn_python"):
        # The two walks over semantic trees, as functions of this interpreter. The check that a tree is exact
        # also serves the model and the recorder, which ask it of the same trees (policy._exact_tree).
        lib.fn_python.restype = ctypes.py_object
        lib.fn_python.argtypes = (ctypes.py_object, ctypes.c_long, ctypes.c_long, ctypes.c_long)
        walk, _ROOT, *calls = lib.fn_python(_Kept, _policy._LEAF_MEMO_LIMIT, _policy._NODE_MEMO_LIMIT, _policy._KEPT_LIMIT)
        _CALLS = tuple(calls)
        _policy._walk_exact = walk
    _LIBRARY = lib
    return lib


_ROWS = 4096  # states cut from one block of zeros
_BLOCK = [None, 0, 0]  # the block states are cut from now, its address, the next row


class State:
    """A recurrent state: 64 numbers of the caller's own, with their address for the C calls.

    Every frame makes one. A state is a row of a block of zeros made once for 4,096 of them, its
    address the block's plus the row's offset: no array and no ctypes object of its own for each
    frame. A row is handed out once and the block stays alive while any of its rows does.
    """
    __slots__ = ("values", "address")

    def __init__(self):
        block, base, row = _BLOCK
        if row == _ROWS or block is None:
            block = np.zeros((_ROWS, 64), dtype=np.float32)
            base, row = block.ctypes.data, 0
            _BLOCK[0], _BLOCK[1] = block, base
        _BLOCK[2] = row + 1
        self.values = block[row]
        self.address = base + 256 * row

    def tensor(self):
        return torch.from_numpy(self.values.copy())


class _Quiet:
    """What the action walk asks of its model besides numbers: no memo of torch tensors to revalidate."""

    def __init__(self, model):
        self.config, self._float = model.config, model._float
        self.window_memory = model.window_memory

    def _memo(self):
        return None

    def _settle(self, cache):
        pass


class FastPolicy:
    """Acting with the weights ``model`` holds now; build a new one after the weights change."""

    def __init__(self, model):
        lib = library()
        if lib is None:
            raise RuntimeError("fastnet.c could not be built on this machine")
        self.model, self.config, self.lib = model, model.config, lib
        self.quiet = _Quiet(model)
        self.root = _ROOT
        # The calls of every frame and choice: fastnet.c's own functions, through ctypes where it was not built with Python.h.
        self.used, self.step, self.value, self.query, self.c_logits = _CALLS or (
            lib.fn_used, lib.fn_step, lib.fn_value, lib.fn_query, lib.fn_logits)
        self.net = lib.fn_create(ARENA_SLOTS)
        if not self.net:
            raise MemoryError("No memory for the arena of tree encodings")
        state = model.state_dict()
        for index, name in enumerate(_PARAMETERS):
            values = np.ascontiguousarray(state[name].detach().to(torch.float32).cpu().numpy()).reshape(-1)
            if lib.fn_set(self.net, index, values.ctypes.data, values.size):
                raise MemoryError("No memory for the weights")
        if lib.fn_ready(self.net):
            raise ValueError("The weights do not have the shapes fastnet.c is written for")
        self.children = (ctypes.c_int * 4096)()
        self.children_address = ctypes.addressof(self.children)
        self.logits = np.zeros(4096, dtype=np.float32)
        self.logits_address = self.logits.ctypes.data  # kept with the array: a ctypes view is made once, not per choice
        self._begin()

    def __del__(self):
        net, self.net = getattr(self, "net", None), None
        if net:
            self.lib.fn_destroy(net)

    def _begin(self):
        """Empty memos and an empty arena: every slot number handed out before is forgotten with them."""
        self.lib.fn_reset(self.net)
        self.text, self.number, self.node, self.exact, self.frame = {}, {}, {}, {}, {}
        self.map = self.observation = self.things = None
        self.features = {}  # id of an observation -> (it, its features): slots of this arena, forgotten with it
        self.epoch = getattr(self, "epoch", 0) + 1  # native slots name this arena's contents until the next reset

    def initial_state(self):
        return State()

    # -- trees ---------------------------------------------------------------------
    def _slot(self, index):
        if index < 0:
            raise MemoryError("The arena of tree encodings is full")
        return index

    def _leaf(self, kind, value):
        if kind == "text":
            raw = value.encode("utf-8")
            return self._slot(self.lib.fn_text(self.net, raw, len(raw)))
        return self._slot(self.lib.fn_number(self.net, copysign(log1p(abs(value)), value), float(value == 0)))

    def _join(self, tag, encodings, ordered):
        count = len(encodings)
        children = self.children
        if count > len(children):
            children = self.children = (ctypes.c_int * (2 * count))()
            self.children_address = ctypes.addressof(children)
        for position, encoding in enumerate(encodings):
            children[position] = encoding
        return self._slot(self.lib.fn_node(self.net, tag, self.children_address, count, ordered))

    def _encode_exact(self, tree):
        """The slot of an exact tree's encoding: ``ContextualActorCritic._encode_exact`` with slots for tensors."""
        if type(tree) is _Kept:
            known = self.exact.get(id(tree))
            if known is not None:
                return known[1]
        kind = tree[0]
        if kind == "text" or kind == "number":
            entries = self.text if kind == "text" else self.number
            key = tree[1]  # a string, or a float that is neither a negative zero nor not a number: the tree is exact
            result = entries.get(key)
            if result is None:
                result = self._leaf(kind, tree[1])
                if len(entries) >= _policy._LEAF_MEMO_LIMIT:
                    entries.clear()  # bounded memory; a miss only computes again
                entries[key] = result
            return result
        nodes = self.node
        result = nodes.get(tree)
        if result is None:
            tag, children = tree[1:]
            encodings = [self._encode_exact(child) for child in children]
            result = self._join(self._encode_exact(("text", tag)), encodings, kind == "ordered")
            if len(nodes) >= _policy._NODE_MEMO_LIMIT:
                nodes.clear()  # bounded memory; a miss only computes again
            nodes[tree] = result
        if type(tree) is _Kept:
            exact = self.exact
            if len(exact) >= _policy._KEPT_LIMIT:
                exact.clear()  # bounded memory; a miss only looks the tree up again
            exact[id(tree)] = (tree, result)  # holds the tree, so its id names nothing else
        return result

    def _encode_anew(self, tree):
        """A tree in which equal is not identical (a negative zero, a number that is not one): no memo is used."""
        kind = tree[0]
        if kind == "text" or kind == "number":
            return self._leaf(kind, tree[1])
        tag, children = tree[1:]
        return self._join(self._leaf("text", tag), [self._encode_anew(child) for child in children], kind == "ordered")

    def _encode_root(self, tree):
        root = self.root
        if root is not None:  # the same walk in C: the slot of an exact tree, None for one that is not
            slot = root(self.net, tree, self.text, self.number, self.node, self.exact)
            return self._encode_anew(tree) if slot is None else slot
        return self._encode_exact(tree) if _exact_tree(tree) else self._encode_anew(tree)

    # -- the map ---------------------------------------------------------------------
    def _encode_map(self, observation):
        width, height, _ = observation.size
        if width > self.config.max_map_side or height > self.config.max_map_side:
            raise ValueError("Map exceeds declared model configuration")
        count = width * height
        ground = _policy.ground(observation.tiles, observation.size)  # kept from view to view; its arrays are read only
        flags = ground[4].copy()
        for hero in observation.heroes:
            flags[hero.position[1] * width + hero.position[0], 3] += 1.0
        for town in observation.towns:
            flags[town.position[1] * width + town.position[0], 4] += 1.0
        described = (observation.size, tuple((obj.position, obj.kind, obj.subtype, obj.owner) for obj in observation.objects))
        things = self.things
        if things is None or things[0] != described:
            objects = sorted(described[1])  # by position, kind, subtype, owner
            attributes = np.array([_policy.object_row(kind, subtype, owner) for _, kind, subtype, owner in objects],
                                  dtype=np.float32).reshape(-1, 67)
            indices = np.array([position[1] * width + position[0] for position, _, _, _ in objects], dtype=np.int32)
            things = self.things = (described, attributes, indices)
        cells = np.empty((count, 16), dtype=np.float32)
        summary = self._slot(self.lib.fn_map(
            self.net, ground[2].ctypes.data, ground[3].ctypes.data, flags.ctypes.data, things[1].ctypes.data,
            things[2].ctypes.data, len(things[2]), width, height, cells.ctypes.data))
        return summary, cells

    def _observation_features(self, observation):
        """The map's summary, the seat's own view and the map's cells; the observation encoded last is not encoded again."""
        last = self.observation
        if last is not None and last[0] is observation:
            return last[1]
        known = self.features.get(id(observation))
        if known is not None and known[0] is observation:
            self.observation = known
            return known[1]
        key, known = map_key(observation), self.map
        if known is not None and known[0] == key:
            summary, cells = known[1]
        else:
            summary, cells = self._encode_map(observation)
            self.map = (key, (summary, cells))
        trees = _native.module()
        root = (trees.observation_slot(self.net, self.epoch, observation) if trees is not None
                else self._encode_root(observation_semantics(observation)))
        found = (summary, root, cells)
        self.observation = (observation, found)
        if len(self.features) >= 2048:
            self.features.clear()  # bounded memory; a miss only encodes again
        self.features[id(observation)] = self.observation  # holds the observation, so its id names nothing else
        return found

    # -- a frame ---------------------------------------------------------------------
    def forward_step(self, frame, state):
        if frame.semantics is None:
            return StepOutput(state, None)
        net = self.net
        if self.used(net) > ARENA_SLOTS - 65536:
            self._begin()  # nothing outside this object holds a slot number between two frames
        tree, cells = frame.semantics, None
        if frame.observation is None and type(tree) is _Kept:
            # A frame without an observation whose payload was built once: its input is a function of the payload.
            known = self.frame.get(id(tree))
            if known is None:
                root = self._encode_root(tree)
                if len(self.frame) >= _policy._KEPT_LIMIT:
                    self.frame.clear()  # bounded memory; a miss only looks the payload up again
                self.frame[id(tree)] = (tree, root)  # holds the tree, so its id names nothing else
            else:
                root = known[1]
            summary = own = 0
        elif frame.observation is None:
            root, summary, own = self._encode_root(tree), 0, 0
        else:
            summary, own, cells = self._observation_features(frame.observation)
            root = self._encode_root(tree)
        after = State()
        self.step(net, state.address, after.address, root, summary, own)
        decision = None
        if frame.frame.kind == "decision":
            distribution = FastDistribution(self, after, frame.frame.observation, frame.frame.offers,
                                            cells, frame.guard, frame.window_memory)
            decision = PolicyValue(self.value(net, after.address), distribution)
        return StepOutput(after, decision)


class FastDistribution(ContextualDistribution):
    """The action walk of ``ContextualDistribution`` with its features and logits from ``FastPolicy``."""

    def __init__(self, policy, state, observation, offers, cells, guard=None, window_memory=frozenset()):
        super().__init__(policy.quiet, state, observation, offers, cells, {}, guard, window_memory)
        self.policy = policy
        self.query_address = None  # of the query array, once the first categorical choice computed it

    def _features(self, semantic):
        result = self.policy._encode_root(semantic)
        recording = self.recording
        if recording is not None:
            recording.feature[result] = len(recording.trees)  # a slot names one tree's encoding
            recording.trees.append(semantic)
        return result

    def _tensor(self, count):
        return torch.from_numpy(self.policy.logits[:count].copy())

    def _logits(self, semantics):
        policy, recording = self.policy, self.recording
        first = len(recording.trees) if recording is not None else 0
        candidates = [self._features(item) for item in semantics]
        if recording is not None:
            recording.pending = {"kind": "categorical", "roots": range(first, len(recording.trees))}
        address = self.query_address
        if address is None:
            query = self.query = np.empty(32, dtype=np.float32)
            address = self.query_address = query.ctypes.data
            policy.query(policy.net, self.state.address, address)
        count = len(candidates)
        if count > len(policy.logits):
            policy.logits = np.zeros(2 * count, dtype=np.float32)
            policy.logits_address = policy.logits.ctypes.data
        children = (ctypes.c_int * count)(*candidates)
        if policy.c_logits(policy.net, address, ctypes.addressof(children), count, policy.logits_address):
            raise MemoryError("No memory for the candidates of a choice")
        return self._tensor(count)

    def _quantity_logits(self, features, low, high, maximum, depth):
        policy = self.policy
        numeric = self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31))
        middle = (low + high) // 2
        offsets = self.model._float((log1p(middle - low), log1p(high - middle - 1)))
        if self.recording is not None:
            self.recording.pending = {"kind": "quantity", "root": self.recording.feature[features],
                                      "numeric": numeric, "offsets": offsets}
        given = numeric.numpy()
        policy.lib.fn_quantity(policy.net, self.state.address, features, given.ctypes.data, policy.logits_address)
        return self._tensor(2) + offsets

    def _coordinate_logits(self, semantic, subject, width, height):
        policy = self.policy
        positions = self._coordinate_positions(subject, width, height)
        target = self._features(semantic)
        if self.recording is not None:
            self.recording.pending = {"kind": "coordinate", "root": len(self.recording.trees) - 1,
                                      "positions": positions, "indices": None}
        count = width * height
        if count > len(policy.logits):
            policy.logits = np.zeros(2 * count, dtype=np.float32)
            policy.logits_address = policy.logits.ctypes.data
        given = positions.numpy()
        if policy.lib.fn_coordinate(policy.net, self.state.address, target, self.cells.ctypes.data, given.ctypes.data,
                                    count, policy.logits_address):
            raise MemoryError("No memory for the coordinate head")
        return self._tensor(count)

    def settled_cache(self):
        raise ValueError("A policy that acts through fastnet.c keeps no tree cache")


def _same(found, expected):
    found = np.ascontiguousarray(found, dtype=np.float32).reshape(-1)
    expected = np.ascontiguousarray(expected.detach().to(torch.float32).cpu().numpy()).reshape(-1)
    return found.shape == expected.shape and np.array_equal(found.view(np.uint32), expected.view(np.uint32))


def proven(twin, model):
    """Whether ``twin`` gives, on a fixed set of inputs, what ``model`` gives in every bit.

    A few leaves, sets on both sides of the sixteen rows torch sums at a
    time, an ordered list, a map with objects, the recurrent step and every
    head. It takes milliseconds and is asked once for every twin: on another
    machine, another build of torch or of the system's BLAS the arithmetic
    here might round elsewhere than torch's, and a twin that does is not used.
    """
    lib, net, config = twin.lib, twin.net, model.config
    generator = torch.Generator().manual_seed(20261122)

    def read(index):
        out = np.zeros(32, dtype=np.float32)
        lib.fn_read(net, index, out.ctypes.data)
        return out

    with torch.no_grad():
        leaves = [("number", 0.0), ("number", -3.5), ("number", 123456.0), ("text", ""), ("text", "pikeman"),
                  ("text", "a text that is longer than sixty-four bytes, so that it is embedded outside the small buffer")]
        trees = leaves + [
            ("set", "army", tuple(("set", "stack", (("number", float(n)), ("text", "x" * n))) for n in range(1, 4))),
            ("set", "many", tuple(("number", float(n)) for n in range(15))),
            ("set", "many", tuple(("number", float(n)) for n in range(16))),
            ("set", "many", tuple(("number", float(n)) for n in range(40))),
            ("ordered", "list", tuple(("text", word) for word in ("you", "gain", "5", "wood"))),
            ("ordered", "none", ()),
        ]
        slots = []
        for tree in trees:
            slots.append(twin._encode_anew(tree))
            expected = model._encode_root(tree, {})
            # without the memos, and through them as a seat's frames go (twice: computed, then found)
            if not all(_same(read(slot), expected) for slot in (slots[-1], twin._encode_root(tree), twin._encode_root(tree))):
                return False
        # The map: a ground of 9 by 7 cells, some of them not seen, with objects, two of them on one cell.
        width, height = 9, 7
        count = width * height
        terrain = torch.randint(0, 12, (count,), generator=generator)
        road = torch.randint(0, 4, (count,), generator=generator)
        flags = (torch.rand(count, 5, generator=generator) < 0.3).to(torch.float32)
        attributes = (torch.rand(6, 67, generator=generator) < 0.2).to(torch.float32)
        indices = torch.tensor([3, 3, 10, 17, 40, 62])
        grid = torch.cat((model.terrain(terrain), model.road(road),
                          model.byte.weight.new_zeros(count, 16).index_add(0, indices, model.object_encoder(attributes)),
                          flags), -1)
        features = model.map_cnn(grid.reshape(height, width, 33).permute(2, 0, 1).unsqueeze(0))
        summary, expected_cells = model.map_global(features)[0], features[0].permute(1, 2, 0).reshape(-1, 16)
        cells = np.empty((count, 16), dtype=np.float32)
        given = [np.ascontiguousarray(value.numpy()) for value in (terrain.to(torch.int32), road.to(torch.int32), flags,
                                                                   attributes, indices.to(torch.int32))]
        found = lib.fn_map(net, *(value.ctypes.data for value in given), len(indices), width, height, cells.ctypes.data)
        if found < 0 or not _same(read(found), summary) or not _same(cells, expected_cells):
            return False
        # The recurrent step from a state that is not zero, then every head on what it gives.
        state = State()
        state.values[:] = torch.tanh(torch.randn(config.hidden_size, generator=generator)).numpy()
        before = torch.from_numpy(state.values.copy())
        after = State()
        lib.fn_step(net, state.address, after.address, slots[6], found, slots[4])
        vector = torch.cat([torch.from_numpy(read(index)) for index in (slots[6], found, slots[4])])
        expected = model.gru(vector, before)
        if not _same(after.values, expected):
            return False
        if lib.fn_value(net, after.address) != float(model.value(expected)):
            return False
        query = np.empty(32, dtype=np.float32)
        lib.fn_query(net, after.address, query.ctypes.data)
        wanted = torch.tanh(model.policy_query(expected))
        if not _same(query, wanted):
            return False
        candidates = (ctypes.c_int * len(slots))(*slots)
        logits = np.empty(len(slots), dtype=np.float32)
        if lib.fn_logits(net, query.ctypes.data, ctypes.addressof(candidates), len(slots), logits.ctypes.data):
            return False
        stacked = torch.stack([torch.from_numpy(read(index)) for index in slots])
        if not _same(logits, stacked.mv(wanted) / sqrt(config.feature_size)):
            return False
        numeric = torch.tensor([0.25, 0.75, log1p(40), 3 / 31])
        out = np.empty(2, dtype=np.float32)
        lib.fn_quantity(net, after.address, slots[7], numeric.numpy().ctypes.data, out.ctypes.data)
        if not _same(out, model.quantity(torch.cat((expected, torch.from_numpy(read(slots[7])), numeric)))):
            return False
        positions = torch.rand(count, 2, generator=generator)
        out = np.empty(count, dtype=np.float32)
        if lib.fn_coordinate(net, after.address, slots[8], cells.ctypes.data, positions.numpy().ctypes.data, count,
                             out.ctypes.data):
            return False
        target = torch.from_numpy(read(slots[8]))
        rows = torch.cat((expected.expand(count, -1), target.expand(count, -1), expected_cells, positions), -1)
        if not _same(out, model.coordinate(rows).squeeze(-1)):
            return False
    return True


def acting(model):
    """What a seat plays with: ``FastPolicy`` over a frozen model where it can be had, else the model itself.

    A twin is used only after it has shown itself equal to the model in every
    bit on the inputs of ``proven``; one that has not leaves the model to torch.
    """
    if not enabled() or not model.__dict__.get("_frozen") or library() is None:
        return model
    twin = model.__dict__.get("_fast")
    if twin is None:
        twin = FastPolicy(model)
        if proven(twin, model):
            twin._begin()  # the proof's trees are not the game's
        else:
            twin = False
        model.__dict__["_fast"] = twin
    return twin or model
