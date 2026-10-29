"""Randomly initialized contextual actor-critic; no collection or optimization.

The codec consumes strict player-local chronology. Routing references resolve only
against the supplied current/earlier permitted snapshot; their spelling is never
encoded. Current map grids are rebuilt from observations. Only the learned GRU
retains earlier information: there is no accumulated last-seen world cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from math import copysign, log1p, sqrt
from operator import attrgetter
import signal
from sys import getsizeof
import threading
import weakref

import torch
from torch import Tensor, nn

from . import contextual as c
from . import interaction_guard as interaction_guard

_DEFAULT_GUARD = object()

MODEL_SCHEMA = "heroes-context-model-v1"
DISTRIBUTION_SCHEMA = "heroes-context-policy-v1"
ARMIES_MODEL_SCHEMA = "heroes-context-model-v2"
ARMIES_DISTRIBUTION_SCHEMA = "heroes-context-policy-v2"
HEROES_MODEL_SCHEMA = "heroes-context-model-v3"
HEROES_DISTRIBUTION_SCHEMA = "heroes-context-policy-v3"
MANAGEMENT_MODEL_SCHEMA = "heroes-context-model-v4"
MANAGEMENT_DISTRIBUTION_SCHEMA = "heroes-context-policy-v4"
DIALOGS_MODEL_SCHEMA = "heroes-context-model-v5"
DIALOGS_DISTRIBUTION_SCHEMA = "heroes-context-policy-v5"
EXPANDED_MODEL_SCHEMA = "heroes-context-model-v6"
EXPANDED_DISTRIBUTION_SCHEMA = "heroes-context-policy-v6"
_ACTIONS = ("move_to", "open_hero", "open_town", "end_turn", "close", "open_panel",
            "open_building", "construct", "back", "select_recruit", "recruit", "transfer_stack", "select_hero", "hire_hero", "split_stack", "merge_stacks", "swap_stacks",
            "open_stack", "select_upgrade", "upgrade_stack", "swap_garrison", "choose_dialog")
# Verification, command identifiers and technical boundaries are not observations.
_METADATA = frozenset(("issued", "cutoff", "cutoff_finalized", "finalized", "failure", "superseded"))


@dataclass(frozen=True)
class ModelConfig:
    schema: str = MODEL_SCHEMA
    hidden_size: int = 64
    feature_size: int = 32
    max_map_side: int = 256

    def __post_init__(self):
        if self.schema not in (MODEL_SCHEMA, ARMIES_MODEL_SCHEMA, HEROES_MODEL_SCHEMA, MANAGEMENT_MODEL_SCHEMA, DIALOGS_MODEL_SCHEMA, EXPANDED_MODEL_SCHEMA):
            raise ValueError("Unsupported contextual model schema")
        for name in ("hidden_size", "feature_size", "max_map_side"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer model dimensions required")
        if self.max_map_side > 256:
            raise ValueError("This bounded model supports map sides through 256")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Exact versioned model configuration required")
        return cls(**value)


def model_schema_for(profile_or_schema):
    profile = c.profile_for(profile_or_schema)
    if profile.capabilities:
        return EXPANDED_MODEL_SCHEMA
    if profile.dialogs:
        return DIALOGS_MODEL_SCHEMA
    if profile.army_management:
        return MANAGEMENT_MODEL_SCHEMA
    if profile.hero_recruitment:
        return HEROES_MODEL_SCHEMA
    return ARMIES_MODEL_SCHEMA if profile.army_transfer else MODEL_SCHEMA


def distribution_schema_for(profile_or_schema, guard_version=None):
    if guard_version is not None:
        if guard_version != interaction_guard.VERSION:
            raise ValueError("Unsupported contextual interaction guard")
        return distribution_schema_for(profile_or_schema) + "+" + guard_version
    profile = c.profile_for(profile_or_schema)
    if profile.capabilities:
        return EXPANDED_DISTRIBUTION_SCHEMA
    if profile.dialogs:
        return DIALOGS_DISTRIBUTION_SCHEMA
    if profile.army_management:
        return MANAGEMENT_DISTRIBUTION_SCHEMA
    if profile.hero_recruitment:
        return HEROES_DISTRIBUTION_SCHEMA
    return ARMIES_DISTRIBUTION_SCHEMA if profile.army_transfer else DISTRIBUTION_SCHEMA


# Length-prefixed JSON header of each (name, dtype, shape): a pure function, encoded once.
_DIGEST_HEADERS = {}


def _weight_entries(model):
    """(name, dtype, shape, exact uint8 view) of every state_dict tensor, sorted by name."""
    entries = []
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("Weights digest requires a contiguous CPU tensor")
        entries.append((name, value.dtype, tuple(value.shape), value.reshape(-1).view(torch.uint8)))
    return entries


def _config_text(model):
    return json.dumps(model.config.to_dict(), sort_keys=True, separators=(",", ":"))


def _entries_sha256(config, entries):
    digest = hashlib.sha256(config.encode())
    for name, dtype, shape, flat in entries:
        key = (name, dtype, shape)
        header = _DIGEST_HEADERS.get(key)
        if header is None:
            text = json.dumps((name, str(dtype), shape), separators=(",", ":")).encode()
            header = _DIGEST_HEADERS[key] = len(text).to_bytes(8, "big") + text
        # The exact bytes of the uint8 view, copied once by torch into Python-owned
        # memory: no Python integer per byte, no pointer reads, NumPy or pickle.
        data = bytearray(flat.numel())
        if data:
            torch.frombuffer(data, dtype=torch.uint8).copy_(flat)
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def weights_sha256(model):
    """Bind architecture/config and named tensors without NumPy or pickle."""
    return _entries_sha256(_config_text(model), _weight_entries(model))


_MODULE_STATE_DICT, _MODULE_SAVE_STATE, _MODULE_EXTRA_STATE = (
    nn.Module.state_dict, nn.Module._save_to_state_dict, nn.Module.get_extra_state)
_PLAIN_TENSORS = (Tensor, nn.Parameter)


def _plain_state(model):
    """``model.state_dict()`` before detaching, when that call is plain Module code; else None.

    Plain means every module's class keeps Module.state_dict, _save_to_state_dict
    and get_extra_state, no instance overrides the first two, no module has
    state-dict hooks and every stored value is an exact Tensor or Parameter.
    state_dict() then makes, in this same order, exactly these assignments of
    detached aliases of these very tensors (same dtype, shape, strides, device
    and bits), so the name to tensor mapping equals its result.
    """
    state, stack = {}, [("", model)]
    while stack:
        prefix, module = stack.pop()
        cls, attributes = type(module), module.__dict__
        try:
            if (cls.state_dict is not _MODULE_STATE_DICT or cls._save_to_state_dict is not _MODULE_SAVE_STATE
                    or getattr(cls, "get_extra_state", _MODULE_EXTRA_STATE) is not _MODULE_EXTRA_STATE
                    or "state_dict" in attributes or "_save_to_state_dict" in attributes
                    or attributes["_state_dict_pre_hooks"] or attributes["_state_dict_hooks"]):
                return None
            parameters, buffers = attributes["_parameters"], attributes["_buffers"]
            persistent, children = attributes["_non_persistent_buffers_set"], attributes["_modules"]
        except KeyError:
            return None
        for name, value in parameters.items():
            if value is not None:
                if type(value) not in _PLAIN_TENSORS:
                    return None
                state[prefix + name] = value
        for name, value in buffers.items():
            if value is not None and name not in persistent:
                if type(value) not in _PLAIN_TENSORS:
                    return None
                state[prefix + name] = value
        stack.extend((prefix + name + ".", child) for name, child in reversed(children.items()) if child is not None)
    return state


_WORD_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_FLATTEN = getattr(torch._C._nn, "flatten_dense_tensors", None)  # torch.cat of view(-1)s, in one call


def _joined_words(tensors, fresh):
    """The concatenated bits of nonempty contiguous CPU tensors of one plain dtype, as integer words.

    Words are int64 when the byte count allows, else integers of the element
    width; either way, for lists of equal dtypes and sizes, equal words mean
    exactly equal bits, signed zeros and NaN payloads included. With ``fresh``
    the result never aliases the tensors (flattening one tensor is a view).
    """
    with torch.no_grad():
        if _FLATTEN is not None and (len(tensors) > 1 or not fresh):
            joined = _FLATTEN(tensors)
        else:
            joined = torch.cat([tensor.view(-1) for tensor in tensors])
    size = tensors[0].element_size()
    if joined.numel() * size % 8:
        return joined.view(_WORD_DTYPES[size])
    if joined.storage_offset() * size % 8:
        joined = joined.clone()  # a view of one tensor can start inside a word; its copy cannot
    return joined.view(torch.int64)


def _typed_words(tensors):
    """A fresh copy of the bits of contiguous CPU tensors of one dtype as integer words, else None."""
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if tensors[0].element_size() not in _WORD_DTYPES or dtype.is_complex:
        return None
    for tensor in tensors:
        if tensor.dtype != dtype or tensor.device.type != "cpu" or not tensor.is_contiguous():
            return None
    return _joined_words(tensors, True)


class WeightsSnapshot:
    """A private copy of exactly the bytes one weights_sha256 value digests.

    ``digest`` is weights_sha256 of the copied bytes. ``matches(model)`` is True
    only when the model's config text, state_dict names, dtypes and shapes
    equal the copy's and every tensor is a contiguous CPU tensor whose bytes
    equal the copy's; weights_sha256(model) is then ``digest``, because it is a
    function of exactly those inputs. ``unchanged`` computes the digest itself
    whenever ``matches`` is False, so it decides exactly as comparing
    weights_sha256(model) with ``digest`` would.
    """

    def __init__(self, model):
        self.config = _config_text(model)
        entries = [(name, dtype, shape, flat.clone()) for name, dtype, shape, flat in _weight_entries(model)]
        self.layout = tuple((name, dtype, shape) for name, dtype, shape, _ in entries)
        # Zero padding to whole int64 words: torch.equal compares int64 words
        # (bit patterns, as for uint8) many times faster than single bytes.
        self.pad = torch.zeros(-sum(flat.numel() for *_, flat in entries) % 8, dtype=torch.uint8)
        self.words = torch.cat([*(flat for *_, flat in entries), self.pad]).view(torch.int64)
        # The same bytes as integer words of the tensors' own width, when they all share one dtype.
        self.typed = None
        if len({dtype for _, dtype, _ in self.layout}) == 1:
            try:
                self.typed = _typed_words([flat.view(dtype).view(shape) for _, dtype, shape, flat in entries])
            except RuntimeError:  # a dtype without a plain view keeps the byte comparison
                pass
        self.digest = _entries_sha256(self.config, entries)

    def matches(self, model):
        if _config_text(model) != self.config:
            return False
        state = _plain_state(model)
        items = sorted((model.state_dict() if state is None else state).items())
        if len(items) != len(self.layout):
            return False
        tensors = []
        for (name, tensor), (known, dtype, shape) in zip(items, self.layout):
            if (name != known or tensor.dtype != dtype or tensor.shape != shape
                    or tensor.device.type != "cpu" or not tensor.is_contiguous()):
                return False
            tensors.append(tensor)
        if self.typed is not None:
            # Same names, one dtype and the same shapes: equal words are exactly equal bytes.
            return torch.equal(_joined_words(tensors, False), self.typed)
        with torch.no_grad():
            flats = [tensor.reshape(-1).view(torch.uint8) for tensor in tensors]
            return torch.equal(torch.cat([*flats, self.pad]).view(torch.int64), self.words)

    def unchanged(self, model):
        return self.matches(model) or weights_sha256(model) == self.digest


# Key of the per-frame cache entry that holds the exact leaf memo; no tree equals it.
_LEAF_MEMO = object()
_LEAF_MEMO_LIMIT = 65536
_NODE_MEMO_LIMIT = 16384  # internal-node entries keep their (larger) trees alive
_OBSERVATION_MEMO_BYTES = 64 << 20  # observation entries hold map cells and tree caches, so bound their bytes
_OBSERVATION_TENSOR_BYTES = 512  # a CPU tensor beyond its data: object, implementation, storage (measured ~450)
# Key of the per-frame cache entry that is False once the cache may hold a tree
# equal to, but not identical with, a later one (see _encode_root).
_EXACT_CACHE = object()
# Key of the per-frame cache entry that holds the experiment-only encodings
# shared by every frame of one autograd graph (see shared_graph_encodings).
_GRAPH_MEMO = object()
# Key of the per-frame cache entry that holds the text leaves of one autograd graph whose
# forward is computed once and whose backward is replayed per occurrence (see replayed_text_leaves).
_LEAF_REPLAY = object()
# The replay (leaf, contribution lists) that this thread is running, for the leaves' node hooks.
_REPLAY_CAPTURE = threading.local()
_TOP_SAVED_TENSOR_HOOKS = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
_CURRENT_GRAPH_TASK = getattr(torch._C, "_current_graph_task_id", None)
_FUNCTION_MODE = getattr(torch._C, "_is_torch_function_mode_enabled", None)
_DISPATCH_STACK = getattr(torch._C, "_len_torch_dispatch_stack", None)
# Signals a replay thread never takes, so that process-directed ones reach the main thread, whose masked
# sections (prefix helpers, background checks) defer them. Faults stay with the thread that causes them.
_REPLAY_THREAD_BLOCKED = signal.valid_signals() - {getattr(signal, name) for name in (
    "SIGKILL", "SIGSTOP", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL", "SIGTRAP", "SIGABRT", "SIGSYS") if hasattr(signal, name)}
# Every submodule that a tree encoding reads (with ``config``): leaves and internal nodes.
_LEAF_MODULES = ("byte", "text_gru", "number", "tree", "ordered")
# The other submodules an observation encoding reads.
_MAP_MODULES = ("terrain", "road", "object_encoder", "map_cnn", "map_global")
_BIT_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bit_state(tensors):
    """Each tensor's (dtype, device, shape), and all their exact bits as CPU integer words.

    The words are the tensors' logical-order elements as integers of their own
    width when all are contiguous CPU tensors of one dtype, else their bytes,
    concatenated and zero padded to whole int64 words; the last meta entry
    names which. For equal meta, two states' words are equal exactly when every
    tensor's bits are, signed zeros and NaN payloads included; torch.equal
    compares integer words much faster than single bytes.
    """
    meta = tuple((value.dtype, value.device, value.shape) for value in tensors)
    words = _typed_words(tensors)
    if words is None:
        flats = [value.detach().reshape(-1).view(torch.uint8).to("cpu") for value in tensors]
        flats.append(torch.zeros(-sum(flat.numel() for flat in flats) % 8, dtype=torch.uint8))
        words = torch.cat(flats).view(torch.int64)
        return (*meta, "bytes"), words
    return (*meta, "typed"), words


def _bits(tensor):
    """Integer view of a tensor's exact bits: signed zeros and NaN payloads differ."""
    dtype = _BIT_DTYPES.get(tensor.element_size())
    return tensor.view(dtype) if dtype is not None else tensor.reshape(-1).view(torch.uint8)


def _start_replay_thread(threads):
    """Before a replay thread's first kernel: block process signals and take the caller's intra-op thread count."""
    signal.pthread_sigmask(signal.SIG_BLOCK, _REPLAY_THREAD_BLOCKED)
    torch.get_num_threads()  # the thread's own lazy initialization first, so that it cannot replace the count later
    torch.set_num_threads(threads)


def _saved_tensor_hooks():
    """True when saved-tensor default hooks may be active (and when this torch cannot tell)."""
    return _TOP_SAVED_TENSOR_HOOKS is None or _TOP_SAVED_TENSOR_HOOKS(True) is not None


def _operation_modes():
    """True when a torch function or dispatch mode may intercept operations (and when this torch cannot tell)."""
    return _FUNCTION_MODE is None or _DISPATCH_STACK is None or _FUNCTION_MODE() or _DISPATCH_STACK() > 0


def _text_replay_blocked(byte, gru):
    """True when a text leaf must be computed as in the reference: a mode or hook the replay would bypass."""
    module = nn.modules.module
    return bool(_saved_tensor_hooks() or _operation_modes() or torch._C._is_any_autocast_enabled() or torch.is_anomaly_enabled()
                or module._global_forward_hooks or module._global_forward_pre_hooks
                or module._global_backward_hooks or module._global_backward_pre_hooks
                or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
                       or "forward" in m.__dict__ for m in (byte, gru)))


def _parameter_state(params):
    """What a replayed leaf's aliases depend on: each parameter's data, layout, version and trainability."""
    return tuple((param.data_ptr(), param.dtype, param.shape, param.stride(), param._version, param.requires_grad)
                 for param in params)


def _parameter_hooks(param):
    return bool(param._backward_hooks or getattr(param, "_post_accumulate_grad_hooks", None))


def _leaf_key(kind, value):
    """Exact memo key, or None to bypass: 1, 1.0 and True stay distinct, as do 0.0 and -0.0."""
    if kind == "text":
        return value if type(value) is str else None
    if type(value) is float:
        return None if value != value else (float, value, copysign(1.0, value))
    return (type(value), value) if type(value) in (int, bool) else None


class _ReplayedText:
    """One text leaf's reference graph, built once per autograd graph on detached parameter aliases.

    ``value`` is the encoding ``text_gru(byte(...))`` returns: the same operations on the same
    inputs and parameters. ``replay(grad)`` runs this leaf's own backward graph again for one
    occurrence and returns every gradient contribution it delivers to the leaf parameters,
    each separately and in the order the reference chain delivers them, so that each
    parameter's gradient sums exactly the same terms in exactly the same order.
    """

    def __init__(self, model, raw):
        byte, gru = model.byte, model.text_gru
        self.params = (byte.weight, *(getattr(gru, name) for name in gru._flat_weights_names))
        self.alias = [param.detach().requires_grad_() for param in self.params]
        self.state = _parameter_state(self.params)
        index = model._long([value + 1 for value in raw])
        with torch.enable_grad():
            embedding = nn.functional.embedding(index, self.alias[0], byte.padding_idx, byte.max_norm, byte.norm_type,
                                                byte.scale_grad_by_freq, byte.sparse).unsqueeze(0)
            hx = torch.zeros(gru.num_layers, 1, gru.hidden_size, dtype=embedding.dtype, device=embedding.device)
            self.out = torch._VF.gru(embedding, hx, self.alias[1:], gru.bias, gru.num_layers, gru.dropout,
                                     gru.training, gru.bidirectional, gru.batch_first)[1][0, 0]
        self.value = self.out.detach()
        # Every edge from this graph to an alias is one contribution; hooks record them as they run.
        slot = {id(alias): position for position, alias in enumerate(self.alias)}
        self.counts, seen, pending = [0] * len(self.alias), set(), [self.out.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            targets = []
            for edge, (child, _) in enumerate(node.next_functions):
                if child is not None and type(child).__name__ == "AccumulateGrad" and id(child.variable) in slot:
                    targets.append((edge, slot[id(child.variable)]))
                    self.counts[slot[id(child.variable)]] += 1
                elif child is not None:
                    pending.append(child)
            if targets:
                node.register_hook(self._recorder(targets))
        self.arguments = tuple(param for param, count in zip(self.params, self.counts) for _ in range(count))

    def _recorder(self, targets):
        owner = weakref.ref(self)  # no reference cycle through the graph's hook storage, which the collector cannot see

        def record(grad_inputs, grad_outputs):
            leaf, captured = getattr(_REPLAY_CAPTURE, "current", (None, None))
            if leaf is None or leaf is not owner():
                raise RuntimeError("A text leaf graph ran outside its own replay")
            grad_inputs = list(grad_inputs)
            for edge, position in targets:
                captured[position].append(grad_inputs[edge])
                grad_inputs[edge] = None  # returned by the replay itself: its own engine sums none of them
            return tuple(grad_inputs)
        return record

    def replay(self, grad):
        """This occurrence's contributions, in order; any thread may replay, each records its own."""
        captured = [[] for _ in self.alias]
        outer = getattr(_REPLAY_CAPTURE, "current", None)
        _REPLAY_CAPTURE.current = (self, captured)
        try:
            torch.autograd.grad(self.out, self.alias, grad, retain_graph=True, allow_unused=True)
        finally:
            _REPLAY_CAPTURE.current = outer
        if [len(values) for values in captured] != self.counts:
            raise RuntimeError("A replayed text leaf delivered other gradient contributions than its graph has")
        return [value for values in captured for value in values]

    def matches(self, params):
        """True when ``params`` are this leaf's parameter objects, trainable and unchanged (data, layout, version)."""
        return (all(a is b for a, b in zip(self.params, params)) and all(param.requires_grad for param in params)
                and _parameter_state(params) == self.state)

    def release(self):
        self.out = self.alias = self.value = None


class _DeferredTextReplays:
    """Occurrence replays computed by worker threads while one bracketed backward continues.

    Only ``ContextualActorCritic.backward_with_text_replays`` defers: ``begin`` before its
    backward, ``finish`` after it (``fail`` if it raises). Only occurrences reached by that
    backward's own graph task defer (a pre-hook records the task on a fresh view of the loss, a
    node that no other backward can reach and no hook is registered on); every other
    backward or ``autograd.grad`` in the context, nested in a hook or on another thread included,
    replays immediately, inside the engine. A bracket defers only when the text leaf parameters
    are trainable, without gradients and without hooks, so that they receive gradient from
    replayed occurrences alone, and only when nothing the workers would not see is active (an
    operation, autocast, anomaly or saved-tensor mode, module hooks, another intra-op thread
    count); ``finish`` checks the parameters again before it installs anything. Occurrences are queued in the order the engine
    reaches them; their contributions are added to each parameter's sum strictly in that order,
    each occurrence's in its replay order, exactly the sequence of additions the engine makes
    in the immediate replay and in the reference, and ``finish`` installs the sums as the
    gradients. A replay's values do not depend on the thread that computes it: the same
    operations run on the same operands, one thread per operation, and each worker takes the
    main thread's intra-op thread count before its first kernel (and leaves process signals to
    the main thread); every replay, on a worker or on the main thread, raises if its thread's
    count is not the bracket's, and a count changed during the backward after an occurrence
    was queued sends the later occurrences to the engine, so ``finish`` raises. Replays are
    folded as soon as all earlier ones are; once the backward is done, ``finish`` also replays
    on the main thread the earliest queued occurrence that no worker has started whenever the
    front one is still running. The sums are installed all or none. After a failure the object
    refuses further brackets.
    """

    def __init__(self, params, workers, modules):
        self.params, self.modules, self.threads = params, modules, torch.get_num_threads()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="text-replay",
                                       initializer=_start_replay_thread, initargs=(self.threads,))
        self.pending = []  # (leaf, grad, future) from ``head`` on, in engine order
        self.head = 0
        self.sums = [None] * len(params)
        self.state = self.task = self.hook = None
        self.active = self.broken = False

    def can_defer(self):
        if self.broken:
            raise RuntimeError("Deferred text leaf replays failed earlier in this context")
        return (_CURRENT_GRAPH_TASK is not None and torch.get_num_threads() == self.threads
                and not _text_replay_blocked(*self.modules)
                and all(param.requires_grad and param.grad is None and not _parameter_hooks(param) for param in self.params))

    def begin(self, root):
        if self.active or self.pending or any(total is not None for total in self.sums):
            raise RuntimeError("Deferred text leaf replays are already in progress")
        self.state = _parameter_state(self.params)
        self.hook = root.register_prehook(self._enter)
        self.active = True

    def _enter(self, grad_outputs):
        if self.task is None:  # the bracket's backward starts at the loss's node
            self.task = _CURRENT_GRAPH_TASK()

    def defers(self):
        """True for an occurrence reached by the bracket's own backward on the bracket's thread count.

        The engine runs every node of a backward under the thread-local state the backward started
        with (operation modes, autocast, saved-tensor hooks), so what ``can_defer`` saw holds for
        each occurrence; the intra-op thread count is not part of that state and is checked again.
        """
        return (self.active and self.task is not None and _CURRENT_GRAPH_TASK() == self.task
                and torch.get_num_threads() == self.threads)

    def add(self, leaf, grad):
        self.pending.append((leaf, grad, self.pool.submit(self._replay, leaf, grad)))
        while self.head < len(self.pending) and self.pending[self.head][2].done():
            self._fold_head(self.pending[self.head][2].result())

    def _replay(self, leaf, grad):
        """One occurrence's replay, on any thread, only on the bracket's intra-op thread count."""
        if torch.get_num_threads() != self.threads:
            raise RuntimeError("A deferred text leaf replay would run on another intra-op thread count than its bracket")
        return leaf.replay(grad)

    def _fold_head(self, values):
        leaf = self.pending[self.head][0]
        self.pending[self.head] = None  # fold once and keep nothing of it
        self.head += 1
        values = iter(values)
        with torch.no_grad():
            for position, (param, count) in enumerate(zip(leaf.params, leaf.counts)):
                if param is not self.params[position]:
                    raise RuntimeError("Text leaf parameters changed inside a replay context")
                for _ in range(count):
                    value = next(values)
                    if value is None:
                        continue  # the engine adds nothing for an undefined gradient
                    if self.sums[position] is None:
                        self.sums[position] = torch.empty_like(param).copy_(value)
                    else:
                        self.sums[position].add_(value)

    def finish(self):
        """After the backward: fold every queued replay in order and install the sums as the gradients."""
        try:
            if not self.active or self.broken:
                raise RuntimeError("No deferred text leaf replays in progress")
            taken, cursor = {}, self.head
            while self.head < len(self.pending):
                future = self.pending[self.head][2]
                if self.head in taken:
                    self._fold_head(taken.pop(self.head))
                    continue
                if not future.done():
                    # Replay here the earliest occurrence no worker has started (workers take them in order).
                    cursor = max(cursor, self.head)
                    while cursor < len(self.pending) and not self.pending[cursor][2].cancel():
                        cursor += 1
                    if cursor < len(self.pending):
                        leaf, grad, _ = self.pending[cursor]
                        taken[cursor] = self._replay(leaf, grad)
                        cursor += 1
                        continue
                self._fold_head(future.result())
            sums = self.sums
            for param, state, total in zip(self.params, self.state, sums):
                if total is not None and (param.grad is not None or _parameter_hooks(param)
                                          or _parameter_state((param,))[0] != state):
                    raise RuntimeError("A text leaf parameter changed or received gradient outside its replayed "
                                       "occurrences during the backward")
            installed = []
            try:
                for param, total in zip(self.params, sums):
                    if total is not None:
                        param.grad = total
                        installed.append(param)
            except BaseException:
                for param in installed:  # all or nothing
                    param.grad = None
                raise
            self._reset()
        except BaseException:
            self.fail()
            raise

    def fail(self):
        """The bracket failed: drop every queued replay and partial sum, and refuse further brackets."""
        self.broken = True
        for entry in self.pending[self.head:]:
            if entry is not None:
                entry[2].cancel()
        self._reset()

    def _reset(self):
        hook, self.hook = self.hook, None
        if hook is not None:
            hook.remove()
        self.pending, self.head, self.sums, self.state, self.task = [], 0, [None] * len(self.params), None, None
        self.active = False

    def close(self):
        """Stop the workers; True if a bracket was left unfinished. The object is not used again."""
        unfinished = self.active or bool(self.pending) or any(total is not None for total in self.sums)
        self.broken = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._reset()
        return unfinished


class _TextOccurrence(torch.autograd.Function):
    """One occurrence of a replayed text leaf: the shared value forward, its own replayed backward.

    Each parameter is an input once per contribution, so autograd adds the contributions to the
    parameter's gradient one by one, where the reference chain of this occurrence would. Inside a
    bracketed backward with deferred replays, the occurrence delivers nothing to the engine and
    queues its replay instead.
    """

    @staticmethod
    def forward(ctx, leaf, deferred, *parameters):
        if leaf.value is None:
            raise RuntimeError("A released text leaf cannot be replayed")
        ctx.set_materialize_grads(False)
        ctx.leaf, ctx.deferred = leaf, deferred
        return leaf.value.clone()

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("Replayed text leaves do not support higher-order gradients")
        if ctx.leaf is None:
            raise RuntimeError("A replayed text leaf occurrence supports a single backward")
        (leaf, deferred), ctx.leaf, ctx.deferred = (ctx.leaf, ctx.deferred), None, None
        if grad is None:
            # The reference chain receives no gradient and delivers none.
            return (None, None) + (None,) * len(leaf.arguments)
        if deferred is not None and deferred.defers():
            deferred.add(leaf, grad)
            return (None, None) + (None,) * len(leaf.arguments)
        return (None, None, *leaf.replay(grad))


def _exact_tree(tree):
    """True when equality is identity for every subtree of this tree.

    Tree equality treats 0.0 and -0.0 as equal and NaN as unequal, and 1 equals
    1.0 and True. Trees whose numbers are floats other than -0.0 and NaN and
    whose texts and tags are strings have no such distinctions: equal subtrees
    are exactly identical and encode to identical tensors.
    """
    stack = [tree]
    while stack:
        item = stack.pop()
        if type(item) is not tuple:
            return False
        kind = item[0] if item else None
        if kind == "number":
            if len(item) != 2 or type(item[1]) is not float or item[1] != item[1]:
                return False
            if item[1] == 0.0 and copysign(1.0, item[1]) < 0:
                return False
        elif kind == "text":
            if len(item) != 2 or type(item[1]) is not str:
                return False
        elif kind in ("node", "ordered"):
            if len(item) != 3 or type(item[1]) is not str or type(item[2]) is not tuple:
                return False
            stack.extend(item[2])
        else:
            return False
    return True


# Tokens of _record_key; no int, str or None equals them.
_TRUE, _FALSE, _TUPLE = object(), object(), object()
_RECORD_FIELDS = {}


def _record_fields(kind):
    """A getter of every field of a contract record class as a tuple, or None for any other type."""
    getter = _RECORD_FIELDS.get(kind)
    if getter is None and kind not in _RECORD_FIELDS:
        if isinstance(kind, type) and is_dataclass(kind) and kind.__module__ == c.__name__:
            names = tuple(field.name for field in fields(kind))
            getter = attrgetter(*names) if len(names) > 1 else lambda item, names=names: tuple(getattr(item, name) for name in names)
        _RECORD_FIELDS[kind] = getter
    return getter


def _record_key(value):
    """An exact, injective key of a value of contract records, tuples, int, str, bool and None, or None.

    The key lists every record's class and field values, every tuple's length
    and elements, and each bool as its own token, so 1 and True stay distinct.
    Equal keys therefore mean values of identical types and contents, which
    every encoding reads identically. Anything else (floats, other types) has
    no key and is never memoized.
    """
    out, stack = [], [value]
    append, extend, pop = out.append, stack.extend, stack.pop
    while stack:
        item = pop()
        kind = type(item)
        if kind is int or kind is str or item is None:
            append(item)
        elif kind is tuple:
            append(_TUPLE); append(len(item)); extend(item)
        elif kind is bool:
            append(_TRUE if item else _FALSE)
        else:
            getter = _record_fields(kind)
            if getter is None:
                return None
            append(kind); extend(getter(item))
    return tuple(out)


def _observation_entry_bytes(store, key, entry):
    """An upper estimate of the memory one observation memo entry adds to ``store``.

    It charges the record key with its strings and uncached integers, the entry
    and its cache dict, every tree object the cache keys keep alive, once each,
    and each tensor the store does not hold yet, with its data. The trees are
    walked because a child equal to an earlier key is a different object that
    only its parent keeps, with its whole subtree. A tree shared with an earlier
    entry is charged again; a charged tensor stays alive in the entries until
    they are cleared, which also forgets the charges.
    """
    charged, added = store["charged"], entry[3]
    size = getsizeof(key) + getsizeof(entry) + getsizeof(added) + sum(
        getsizeof(item) for item in key if type(item) is str or (type(item) is int and not -5 <= item <= 256))
    seen, stack = set(), [tree for tree in added if type(tree) is tuple]
    while stack:
        tree = stack.pop()
        if id(tree) not in seen:
            seen.add(id(tree))
            size += getsizeof(tree) + getsizeof(tree[1])
            if len(tree) > 2:
                size += getsizeof(tree[2])
                stack.extend(tree[2])
    for tensor in (*entry[:3], *added.values()):
        if isinstance(tensor, Tensor) and id(tensor) not in charged:
            charged.add(id(tensor))
            size += tensor.element_size() * tensor.numel() + _OBSERVATION_TENSOR_BYTES
    return size


def _has_army_operation(armies, kind):
    """bool(c.army_operation_choices(armies, kind)), stopping at the first choice."""
    return any(c.army_operation_maximum(armies, kind, source.ref, destination.ref) > 0
               for first in armies for source in first.slots for second in armies for destination in second.slots)


def _node(tag, *children):
    return ("node", tag, tuple(children))


def _number(value):
    return ("number", float(value))


def _text(value):
    return ("text", value)


def _set(tag, children):
    # No opaque-ref order, hash order or engine allocation order becomes a feature.
    return _node(tag, *sorted(children, key=repr))


def _subject(observation, ref):
    if ref is None or observation is None:
        return _node("unresolved_reference")
    for entry in (*observation.heroes, *observation.towns, *observation.objects):
        if entry.handle == ref:
            return _semantic(entry)
    return _node("unresolved_reference")


def _selection(observation, ref):
    if observation is not None and observation.window is not None and observation.window.capability_panel is not None:
        return next((_capability_choice(choice, observation) for choice in observation.window.capability_panel.choices
                     if choice.ref == ref), _node("unresolved_selection"))
    if observation is not None and observation.window is not None and observation.window.dialog_panel is not None:
        panel = observation.window.dialog_panel
        return next((_dialog_choice(panel, item) for item in panel.choices if item.ref == ref), _node("unresolved_selection"))
    if observation is not None and observation.window is not None and observation.window.stack_panel is not None:
        return next((_semantic(item) for item in observation.window.stack_panel.offers if item.ref == ref), _node("unresolved_selection"))
    if observation is None or observation.window is None or observation.window.town_panel is None:
        return _node("unresolved_selection")
    panel = observation.window.town_panel
    for entry in (*panel.controls, *panel.buildings,
                  *((panel.building,) if panel.building else ()),
                  *(panel.recruitment.offers if panel.recruitment else ()),
                  *(panel.tavern.offers if panel.tavern else ())):
        if entry.ref == ref:
            return _semantic(entry)
    return _node("unresolved_selection")


def _action(action, observation):
    # The binding is the held decision's snapshot, never a newly queried subject.
    children = [_text(action.kind), _node("decision_subject", _subject(observation, action.subject))]
    for key in ("x", "y", "count"):
        value = getattr(action, key)
        children.append(_node(key, _semantic(value)))
    children.extend((_node("panel", _semantic(action.panel)),
                     _node("decision_selection", _army_slot(observation, action.selection, receiving=False)
                           if action.kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks", "open_stack") else _selection(observation, action.selection))))
    if action.destination is not None:
        if action.kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            children.append(_node("destination", _army_slot(observation, action.destination, receiving=True)))
        else:
            panel = observation.window.town_panel.recruitment
            if panel is None or panel.destination.ref != action.destination:
                raise ValueError("Action destination has no displayed semantic binding")
            children.append(_node("destination", _semantic(panel.destination)))
    return _node("action", *children)


def _army_slot(observation, ref, *, receiving):
    window = observation.window if observation else None
    if window is not None and window.armies is not None:
        army, slot = c.army_slot(window.armies, ref)
    else:
        panel = window.town_panel if window else None
        exchange = panel.armies if panel else None
        army = (exchange.destination if receiving else exchange.source) if exchange else None
        slot = next((slot for slot in army.slots if slot.ref == ref), None) if army else None
    if slot is None:
        raise ValueError("Army slot has no current displayed semantic binding")
    return _node("army_slot", _node("role", _text(army.role)), _node("owner_subject", _subject(observation, army.subject)),
                 _node("requires_nonempty", _semantic(army.requires_nonempty)), _semantic(slot))


def _dialog_choice(panel, choice):
    # The ordinal only routes to an already displayed component. Its magnitude
    # and the opaque choice/query references never become learned features.
    component = panel.components[choice.component_index] if choice.kind == "select_component" else None
    return _node("dialog_choice", _text(choice.kind), _semantic(component))


def _capability_choice(choice, observation):
    return _node("capability_choice", _semantic(choice),
                 _node("actor", _subject(observation, choice.subject)),
                 _node("recipient", _subject(observation, choice.destination))
                 if choice.destination is not None else _node("no_recipient"))


def _window_semantic(observation):
    """Bind expanded inventory ownership without learning opaque references."""
    window = observation.window
    ordinary = _semantic(window)
    if window is None or window.capability_panel is None:
        return ordinary
    panel = window.capability_panel
    items = [_node("owned_item", _semantic(item), _subject(observation, item.subject)) for item in panel.items]
    choices = [_capability_choice(choice, observation) for choice in panel.choices]
    return _node("capability_window", ordinary, _set("bound_items", items), _set("bound_choices", choices),
                 _node("inspected_object", _subject(observation, panel.inspected_subject)))


def _semantic(value):
    """Lossless field selection before learned compression; no reference IDs.

    Static definition numbers/text identifiers are categorical bytes learned from
    random initialization, not ordinal utility or pretrained text embeddings.
    Ordered sequences use a separate recurrent reduction (notice substitutions).
    """
    if value is None:
        return _node("absent")
    if type(value) is bool:
        return _text("true" if value else "false")
    if type(value) in (float, int):
        return _number(value)
    if type(value) is str:
        return _text(value)
    if type(value) is tuple:
        return ("ordered", "tuple", tuple(_semantic(item) for item in value))
    if type(value) is c.RecruitmentPanel:
        cards = [_node("card", _semantic(card), _semantic(card.ref == value.selected_offer_ref)) for card in value.offers]
        return _node("recruitment", _set("cards", cards), _node("stock", _number(value.stock)),
                     _node("unit_cost", _semantic(value.unit_cost)), _node("slider_max", _number(value.slider_max)),
                     _semantic(value.destination))
    if type(value) is c.TavernPanel:
        cards = [_node("card", _semantic(card), _semantic(card.ref == value.selected_offer_ref)) for card in value.offers]
        return _node("tavern", _set("cards", cards), _node("selected_detail", _semantic(value.selected_detail)),
                     _node("gold_cost", _number(value.gold_cost)), _node("recruit_status", _text(value.recruit_status)))
    if type(value) is c.StackPanel:
        cards = [_node("upgrade", _semantic(card), _semantic(card.ref == value.selected_upgrade_ref)) for card in value.offers]
        return _node("stack_panel", _semantic(value.parent_context), _semantic(value.source), _set("offers", cards),
                     _node("total_cost", _semantic(value.total_cost)), _node("affordable", _semantic(value.affordable)))
    if type(value) is c.DialogPanel:
        return _node("dialog_panel", _text(value.family), _semantic(value.text),
                     _set("components", [_semantic(item) for item in value.components]),
                     _node("selection", _semantic(value.selection)), _node("cancel", _semantic(value.cancel)),
                     _set("choices", [_dialog_choice(value, choice) for choice in value.choices]))
    if type(value) is c.DialogChoice:
        raise ValueError("Dialog choice requires its displayed component binding")
    if type(value) is c.Notice:
        return _node("notice", _text(value.display_mode),
                     ("ordered", "notice_text", tuple(_semantic(token) for token in value.text)),
                     _set("components", [_semantic(component) for component in value.components]))
    if is_dataclass(value):
        children = []
        for field in fields(value):
            name, item = field.name, getattr(value, field.name)
            # Additive profile fields must not add even an "absent" node to
            # old observations: archived values/probabilities remain exact.
            if item is None and (type(value) is c.TownPanel and name in ("armies", "tavern")
                                 or type(value) is c.Hero and name == "in_garrison"
                                 or type(value) is c.Window and name in ("armies", "stack_panel", "dialog_panel", "capability_panel", "capability_controls")
                                 or type(value) is c.VisibleEvent and name in ("partner", "dialog", "capability")
                                 or type(value) is c.NoticeComponent and name == "skill"):
                continue
            if type(value) is c.TownPanel and name in ("armies", "tavern") and item is None:
                continue
            if name in ("handle", "ref", "control_ref", "selected_offer_ref", "selected_upgrade_ref", "subject", "selection", "partner", "schema", "inspected_subject"):
                continue
            if type(value) is c.CapabilityChoice and name == "destination":
                continue  # requires its displayed recipient binding, never handle bytes
            if (name in ("creature", "terrain", "road", "kind", "subtype", "object_kind", "object_subtype", "faction", "portrait", "hero_class")
                    or type(value) in (c.CapabilityChoice, c.CapabilityItem, c.CapabilityPanel)
                    and name in ("id", "artifact", "spell", "skill", "slot", "destination_slot", "resource_from", "resource_to", "primary_skill")) and type(item) is int:
                encoded = _text(f"category:{item}")
            elif type(value) is c.CapabilityItem and value.kind == "world_object" and name == "amount" and type(item) is int:
                encoded = _text(f"category:{item}")  # delivered object subtype, not an amount
            elif name in ("children", "controls", "buildings", "army", "items", "choices", "capability_controls") and type(item) is tuple:
                encoded = _set(name, [_semantic(entry) for entry in item])
            else:
                encoded = _semantic(item)
            children.append(_node(name, encoded))
        return _node(type(value).__name__, *children)
    raise ValueError("Unsupported model semantic payload")


@dataclass(frozen=True)
class ContextualInput:
    frame: c.Frame
    semantics: tuple | None
    # These are the frame's CURRENT observation, never an accumulated map cache.
    observation: c.Observation | None = None
    guard: interaction_guard.GuardSupport | None = None


def frame_key(frame):
    """Exact prefix memo key of a frame with semantics: repr keeps number types and signed zeros."""
    if frame.guard is None:
        return repr((frame.semantics, frame.observation))
    return repr((frame.semantics, frame.observation, frame.guard))


def frame_keys_until(keys, frames, stop):
    """Extend ``keys`` (one entry per frame, None without semantics) to ``stop`` frames; return it."""
    for seq in range(len(keys), stop):
        frame = frames[seq]
        keys.append(None if frame.semantics is None else frame_key(frame))
    return keys


class ContextualEncoder:
    """Chronology checker plus temporary routing bindings, with no neural state."""

    def __init__(self, episode, player, schema=c.SCHEMA, *, guard_version=_DEFAULT_GUARD):
        self.history = c.PlayerHistory(episode, player, schema=schema)
        self.observation = None
        self.guard_tracker = (interaction_guard.Tracker() if guard_version is _DEFAULT_GUARD
                              else interaction_guard.Tracker(guard_version))

    def encode(self, frame):
        candidate = copy(self.history)
        candidate._frames = list(self.history.frames)
        candidate.dialog_refs = set(self.history.dialog_refs)
        candidate.capability_refs = set(self.history.capability_refs)
        candidate.append(frame)
        basis = self.history.decision.observation if self.history.decision is not None else self.observation
        kind = frame.kind
        payload = None
        if kind not in _METADATA:
            children = [_node("day", _number(frame.day)), _text(kind)]
            if frame.action is not None:
                children.append(_action(frame.action, basis))
            if frame.event is not None:
                children.append(_semantic(frame.event))
                if frame.event.subject is not None:
                    children.append(_node("prior_observation_subject", _subject(self.observation, frame.event.subject)))
                if frame.event.partner is not None:
                    children.append(_node("prior_observation_partner", _subject(self.observation, frame.event.partner)))
            if frame.result is not None:
                children.append(_semantic(frame.result))
            if frame.context is not None:
                children.append(_semantic(frame.context))
                children.append(_node("prior_observation_subject", _subject(self.observation, frame.context.subject)))
                if frame.context.kind == "exchange":
                    children.append(_node("prior_observation_partner", _subject(self.observation, frame.context.selection)))
            if frame.accepted is not None:
                children.append(_node("accepted", _semantic(frame.accepted)))
            if frame.outcome is not None:
                children.append(_node("observed_outcome", _text(frame.outcome)))
            # Consumer layer, revisions, sequence numbers, evidence and command
            # counters are deliberately absent. A consumed-action marker remains.
            payload = _node("frame", *children)
        tracker = self.guard_tracker if self.guard_tracker.version is None else self.guard_tracker.copy()
        guard = tracker.observe(frame)
        result = ContextualInput(frame, payload, frame.observation, guard)
        self.guard_tracker = tracker
        self.history = candidate
        if frame.observation is not None:
            self.observation = frame.observation
        return result


def encode_history(frames, *, guard_version=_DEFAULT_GUARD):
    frames = tuple(frames)
    if not frames:
        return ()
    first = frames[0]
    encoder = ContextualEncoder(first.episode, first.player, first.schema, guard_version=guard_version)
    return tuple(encoder.encode(frame) for frame in frames)


@dataclass(frozen=True)
class DecisionSupport:
    action_schema: str
    context: c.Context
    offers: c.Offers
    distribution_schema: str = DISTRIBUTION_SCHEMA
    guard: interaction_guard.GuardSupport | None = None

    def __post_init__(self):
        if self.guard is not None and type(self.guard) is not interaction_guard.GuardSupport:
            raise ValueError("Exact immutable interaction support required")
        if (self.distribution_schema != distribution_schema_for(self.action_schema, self.guard.version if self.guard else None) or type(self.offers) is not c.Offers
                or self.action_schema != self.offers.schema or self.context != self.offers.context):
            raise ValueError("Exact contextual policy support required")


def decision_support(frame, guard=None):
    """Bind native offers and the independently reconstructed behavior guard."""
    if type(frame) is not c.Frame or frame.kind != "decision":
        raise ValueError("A contextual decision frame is required")
    return DecisionSupport(frame.offers.schema, frame.offers.context, frame.offers,
                           distribution_schema_for(frame.offers.schema, guard.version if guard else None), guard)


@dataclass(frozen=True)
class ConditionalProbability:
    name: str
    selected_index: int
    support_size: int
    log_prob: Tensor


ConditionalLogProbability = ConditionalProbability


@dataclass(frozen=True)
class ActionEvaluation:
    log_prob: Tensor
    conditionals: tuple[ConditionalProbability, ...]


@dataclass(frozen=True)
class PolicyValue:
    value: Tensor
    distribution: ContextualDistribution


@dataclass(frozen=True)
class StepOutput:
    state: Tensor
    decision: PolicyValue | None


def observation_semantics(observation):
    """The visible semantic tree shared by Python and native tensor execution."""
    own = _node("observation", _node("size", _semantic(observation.size)),
                _node("resources", _semantic(observation.resources)),
                _set("heroes", [_semantic(hero) for hero in observation.heroes]),
                _set("towns", [_semantic(town) for town in observation.towns]),
                _window_semantic(observation),
                _node("current_window_subject", _subject(observation, observation.context.subject)))
    if observation.window is not None and observation.window.armies is not None:
        window = observation.window
        bindings = [_node("displayed_army", _semantic(army), _subject(observation, army.subject)) for army in window.armies]
        if window.stack_panel is not None:
            parent = window.stack_panel.parent_context
            bindings.append(_node("stack_parent", _subject(observation, parent.subject),
                                  _subject(observation, parent.selection) if parent.kind == "exchange" else _node("absent")))
        if window.context.kind == "exchange":
            bindings.append(_node("exchange_partner", _subject(observation, window.context.selection)))
        own = _node("management_observation", own, _set("army_bindings", bindings))
    return own


class ContextualActorCritic(nn.Module):
    """Current-view spatial encoder and typed semantics share one learned GRU.

    No object/choice routing handle, color, episode, revision or proof is encoded.
    The tree encoder binds fields within entities before symmetric pooling;
    notice operations retain order. Text bytes have randomly initialized learned
    embeddings; there are no language-model weights or built-in strategy scores.
    """

    def __init__(self, config=ModelConfig()):
        super().__init__()
        if type(config) is not ModelConfig:
            raise ValueError("Expected ModelConfig")
        self.config = config
        f, h = config.feature_size, config.hidden_size
        self.byte = nn.Embedding(257, 8, padding_idx=0)
        self.text_gru = nn.GRU(8, f, batch_first=True)
        self.number = nn.Sequential(nn.Linear(2, f), nn.Tanh())
        self.tree = nn.Sequential(nn.Linear(2 * f, f), nn.Tanh())
        self.ordered = nn.GRU(f, f, batch_first=True)
        self.terrain = nn.Embedding(257, 8, padding_idx=0)
        self.road = nn.Embedding(257, 4, padding_idx=0)
        self.object_encoder = nn.Sequential(nn.Linear(32 * 2 + 3, 16), nn.Tanh())
        self.map_cnn = nn.Sequential(nn.Conv2d(33, 16, 3, padding=1), nn.Tanh(),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.Tanh())
        self.map_global = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, f), nn.Tanh())
        self.gru = nn.GRUCell(3 * f, h)
        self.policy_query = nn.Linear(h, f)
        self.value = nn.Linear(h, 1)
        self.coordinate = nn.Sequential(nn.Linear(h + f + 18, f), nn.Tanh(), nn.Linear(f, 1))
        self.quantity = nn.Sequential(nn.Linear(h + f + 4, f), nn.Tanh(), nn.Linear(f, 2))

    @property
    def device(self):
        return self.byte.weight.device

    def initial_state(self):
        return self.byte.weight.new_zeros(self.config.hidden_size)

    def _float(self, values):
        return self.byte.weight.new_tensor(values)

    def _long(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_leaf_memo_state", None)  # copies and pickles start without a memo
        state.pop("_observation_memo_state", None)
        state.pop("_graph_memo_state", None)
        state.pop("_leaf_replay_state", None)
        state.pop("_text_replay_deferred", None)
        return state

    @contextmanager
    def replayed_text_leaves(self, workers=0):
        """Exact: encode each distinct text leaf once per autograd graph, replay its backward per occurrence.

        Under autograd, the first occurrence of a text (by its exact string) builds the
        leaf's reference graph, ``text_gru(byte(bytes))``, on detached aliases of the leaf
        parameters; every occurrence, the first included, is one autograd node whose
        forward value is that graph's output and whose backward runs that graph again
        with the occurrence's own incoming gradient. A leaf's graph reads only constant
        byte indices and the leaf parameters, so in the reference each occurrence's chain
        runs as one uninterrupted sequence when its output's gradient is complete; the
        replay returns the same contributions, computed by the same operations on the
        same operands, to the same parameters in the same order (each parameter is an
        input once per contribution). Losses, values, log-probabilities, gradients and
        therefore optimizer states are bit-identical to separate recomputation; only the
        repeated forward work is saved. Within a frame the reference cache is unchanged.
        Callers must leave the context before any parameter change: an occurrence after a
        parameter is replaced (also through ``.data``), frozen or changed in place raises. A write
        in place through ``.data`` bypasses version counters and is not detected, as for any
        tensor autograd saves. Each occurrence supports one backward. Yields False, replaying
        nothing, unless the model and leaf modules are the plain modules this argument covers,
        with distinct, trainable leaf parameters and without module, saved-tensor, anomaly,
        autocast or operation modes or hooks; occurrences under any of these, or after the
        context, are computed separately as in the reference.

        With ``workers`` > 0, ``backward_with_text_replays(loss)`` runs occurrences' replays on
        that many threads while its backward continues (``_DeferredTextReplays``) and installs
        their gradients before it returns; any other backward replays inside the engine. The
        gradients are the same bits either way.
        """
        byte, gru = self.byte, self.text_gru
        module = nn.modules.module
        plain = (self._can_cache_prefix() and type(byte) is nn.Embedding and type(gru) is nn.GRU
                 and byte.padding_idx == 0 and byte.max_norm is None and not byte.scale_grad_by_freq and not byte.sparse
                 and gru.num_layers == 1 and gru.bias and gru.batch_first and not gru.bidirectional and gru.dropout == 0
                 and gru.proj_size == 0
                 and not (module._global_backward_hooks or module._global_backward_pre_hooks)
                 and not any(m._backward_hooks or m._backward_pre_hooks for m in self.modules()))
        params = (byte.weight, *(getattr(gru, name) for name in gru._flat_weights_names)) if plain else ()
        if (not plain or len({id(param) for param in params}) != len(params)
                or not all(param.requires_grad for param in params) or _text_replay_blocked(byte, gru)):
            yield False
            return
        deferred = None
        if workers > 0:
            deferred = self.__dict__["_text_replay_deferred"] = _DeferredTextReplays(params, workers, (byte, gru))
        state = self.__dict__["_leaf_replay_state"] = {}
        unfinished = False
        try:
            yield True
        finally:
            if self.__dict__.get("_leaf_replay_state") is state:
                self.__dict__.pop("_leaf_replay_state", None)
            if self.__dict__.get("_text_replay_deferred") is deferred:
                self.__dict__.pop("_text_replay_deferred", None)
            try:
                if deferred is not None:
                    unfinished = deferred.close()
            finally:
                for leaf in state.values():  # only after the workers stopped, but whatever stopping them raised
                    leaf.release()
        if unfinished:
            raise RuntimeError("Deferred text leaf replays were not finished before leaving the context")

    def backward_with_text_replays(self, loss):
        """``loss.backward()``, with text leaf replays on the context's threads when it has them.

        Replays are deferred only when the leaf parameters are trainable, have no gradient and no
        hooks, so that they receive gradient from replayed occurrences alone, and nothing active
        would escape the workers; otherwise, as for any other backward, each occurrence replays
        inside the engine. Deferred replays' sums are installed as the gradients before this
        returns, the same bits as the engine's. Hooks registered directly on a parameter's
        gradient accumulator node are not visible from Python and, under deferral, run without
        a gradient; the runner registers none.
        """
        deferred = self.__dict__.get("_text_replay_deferred")
        if deferred is None or loss.grad_fn is None or not deferred.can_defer():
            loss.backward()
            return
        with torch.enable_grad():
            root = loss.view_as(loss)  # the backward's first node; nothing else reaches it or hooks it
        try:
            deferred.begin(root.grad_fn)
            root.backward()
            deferred.finish()
        except BaseException:
            deferred.fail()
            raise

    @contextmanager
    def shared_graph_encodings(self):
        """Experiment only: share exact tree encodings across every frame of one autograd graph.

        Under autograd, a text or number leaf (by ``_leaf_key``) or an exact
        internal tree (``_exact_tree``) is encoded once per context and the
        same tensor, with its graph, serves every later frame. Forward values,
        losses, log-probabilities and values are bit-identical to separate
        recomputation, because each reuse is the tensor the same operations on
        the same inputs and parameters would return. Gradients are not: every
        reuse sums its incoming gradients before backpropagating them once, so
        parameter gradients differ from the reference in floating-point
        summation order. Callers must leave the context before any parameter
        change. Yields False, sharing nothing, outside ``_can_cache_prefix``.
        """
        if not self._can_cache_prefix():
            yield False
            return
        self.__dict__["_graph_memo_state"] = {"text": {}, "number": {}, "node": {}}
        try:
            yield True
        finally:
            self.__dict__.pop("_graph_memo_state", None)

    def _walk_modules(self, names):
        """(layout, references, tensors) of these submodules in eval mode without hooks, else None.

        The layout names every module's identity, class, parameter, buffer and
        submodule names and flat weight identities; holding the references keeps
        those identities unique while a memo keeps the layout.
        """
        layout, tensors, referenced, modules = [], [], [], [self._modules.get(name) for name in names]
        while modules:
            module = modules.pop()
            if (not isinstance(module, nn.Module) or module.training or module._forward_hooks
                    or module._forward_pre_hooks or "forward" in module.__dict__):
                return None
            flat = tuple(module.__dict__.get("_flat_weights", ()))
            referenced.append((module, flat))
            layout.append((id(module), type(module), tuple(module._parameters), tuple(module._buffers),
                           tuple(map(id, flat)), tuple(module._modules)))
            tensors.extend(value for value in (*module._parameters.values(), *module._buffers.values()) if value is not None)
            modules.extend(module._modules.values())
        return layout, referenced, tensors

    def _leaf_memo(self):
        """Inference-only memo of text/number leaf and internal-node encodings, or None.

        A leaf encoding is a pure function of its exact value, ``config`` and
        the leaf submodules, so a hit is the tensor recomputation would return.
        An internal node of an exact tree (see ``_exact_tree``) is likewise a pure
        function of that tree, ``config`` and the tree submodules. It is
        computation reuse only, never player memory or input. It exists
        only in eval mode with autograd and autocast off and no forward hooks; it is valid
        only while the tree submodules (identity, class, training flag,
        parameter/buffer/submodule layout) and every parameter and buffer's
        dtype, device, shape and bits equal the snapshot taken at creation,
        under the same config, inference mode and thread count. Any difference,
        however made, starts an empty memo. Entries made under inference mode
        are used only under inference mode, and never under autograd.
        """
        if (torch.is_grad_enabled() or self.training or torch._C._is_any_autocast_enabled()
                or nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks):
            return None
        walked = self._walk_modules(_LEAF_MODULES)
        if walked is None:
            return None
        layout, referenced, tensors = walked
        context = (self.config, torch.is_inference_mode_enabled(), torch.get_num_threads(), layout)
        memo = self.__dict__.get("_leaf_memo_state")
        state = _bit_state(tensors)
        if memo is None or memo[0] != context or memo[2][0] != state[0] or not torch.equal(memo[2][1], state[1]):
            # Referencing the modules and flat weights keeps every id in ``context`` unique.
            memo = (context, referenced, state, {"text": {}, "number": {}, "node": {}})
            self.__dict__["_leaf_memo_state"] = memo
        return memo[3]

    def _encode_root(self, tree, cache, final=False):
        """Encode one top-level tree with this frame's cache, as ``_encode_tree`` does.

        Internal nodes come from the inference memo only while this tree and
        every earlier root of the cache are exact. Then no lookup can match an
        equal but different tree, so every subtree's cached value is the exact
        encoding of that very subtree, the value a memo entry holds. Hits still
        visit their subtrees, so the cache gains exactly the entries the
        reference computation would add, and a later inexact root (signed zero,
        NaN) meets the reference cache. Once a root is inexact, or a root was
        encoded without the memo, the cache no longer uses it.
        """
        memo = cache.get(_LEAF_MEMO)
        nodes = memo.get("node") if memo is not None and not torch.is_grad_enabled() else None
        graph = cache.get(_GRAPH_MEMO) if nodes is None and torch.is_grad_enabled() else None
        if graph is not None:
            nodes = graph["node"]
        if nodes is not None and cache.get(_EXACT_CACHE, True) and _exact_tree(tree):
            cache[_EXACT_CACHE] = True
            # The last root of a cache that nothing reads afterwards needs no visit.
            hit = nodes.get(tree) if final else None
            return hit if hit is not None else self._encode_tree(tree, cache, nodes)
        cache[_EXACT_CACHE] = False
        return self._encode_tree(tree, cache)

    def _encode_tree(self, tree, cache, nodes=None):
        if tree in cache:
            return cache[tree]
        kind = tree[0]
        memo = cache.get(_LEAF_MEMO) if kind in ("text", "number") else None
        if kind in ("text", "number") and torch.is_grad_enabled() and _GRAPH_MEMO in cache:
            memo = cache[_GRAPH_MEMO]
            key = _leaf_key(kind, tree[1])
        else:
            key = None if memo is None or torch.is_grad_enabled() else _leaf_key(kind, tree[1])
        if key is not None:
            entries = memo[kind]
            result = entries.get(key)
            if result is None:
                result = self._encode_tree(tree, {})
                if len(entries) >= _LEAF_MEMO_LIMIT:
                    entries.clear()  # bounded memory; a miss only recomputes
                entries[key] = result
            cache[tree] = result
            return result
        if kind == "text":
            raw = tree[1].encode("utf-8")
            replay = cache.get(_LEAF_REPLAY) if raw and type(tree[1]) is str and torch.is_grad_enabled() else None
            if replay is not None and (replay is not self.__dict__.get("_leaf_replay_state")
                                       or _text_replay_blocked(self.byte, self.text_gru)):
                replay = None  # after its context, or under a mode or hook it would bypass: computed as in the reference
            if not raw:
                result = self.byte.weight.new_zeros(self.config.feature_size)
            elif replay is not None:
                leaf = replay.get(tree[1])
                if leaf is None:
                    leaf = replay[tree[1]] = _ReplayedText(self, raw)
                gru = self.text_gru
                if not leaf.matches((self.byte.weight, *(getattr(gru, name) for name in gru._flat_weights_names))):
                    raise RuntimeError("Text leaf parameters changed inside a replay context")
                result = _TextOccurrence.apply(leaf, self.__dict__.get("_text_replay_deferred"), *leaf.arguments)
            else:
                embedding = self.byte(self._long([value + 1 for value in raw])).unsqueeze(0)
                result = self.text_gru(embedding)[1][0, 0]
        elif kind == "number":
            value = tree[1]
            result = self.number(self._float((copysign(log1p(abs(value)), value), float(value == 0))))
        else:
            tag, children = tree[1:]
            result = nodes.get(tree) if nodes is not None else None
            if result is not None:
                # Visit the subtrees as the reference does, adding exactly its cache entries.
                for child in children:
                    self._encode_tree(child, cache, nodes)
                self._encode_tree(_text(tag), cache, nodes)
                cache[tree] = result
                return result
            encodings = [self._encode_tree(child, cache, nodes) for child in children]
            if not encodings:
                pooled = self.byte.weight.new_zeros(self.config.feature_size)
            elif kind == "ordered":
                pooled = self.ordered(torch.stack(encodings).unsqueeze(0))[1][0, 0]
            else:
                pooled = torch.stack(encodings).sum(0) / sqrt(len(encodings))
            result = self.tree(torch.cat((self._encode_tree(_text(tag), cache, nodes), pooled)))
            if nodes is not None:
                if len(nodes) >= _NODE_MEMO_LIMIT:
                    nodes.clear()  # bounded memory; a miss only recomputes
                nodes[tree] = result
        cache[tree] = result
        return result

    def _encode_observation(self, observation, cache):
        width, height, _ = observation.size
        if width > self.config.max_map_side or height > self.config.max_map_side:
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
        object_grid = self.byte.weight.new_zeros(width * height, 16)
        if objects:
            # Static content IDs are categorical bit patterns, not scalar ranks.
            attributes = [[float(value >> bit & 1) for value in (obj.kind, obj.subtype) for bit in range(32)]
                          + [float(obj.owner == owner) for owner in ("self", "opponent", "neutral")] for obj in objects]
            indices = self._long([obj.position[1] * width + obj.position[0] for obj in objects])
            object_grid = object_grid.index_add(0, indices, self.object_encoder(self._float(attributes)))
        grid = torch.cat((self.terrain(self._long(terrain)), self.road(self._long(road)), object_grid, self._float(flags)), -1)
        cells = self.map_cnn(grid.reshape(height, width, 33).permute(2, 0, 1).unsqueeze(0))
        global_map = self.map_global(cells)[0]
        return global_map, self._encode_root(observation_semantics(observation), cache), cells[0].permute(1, 2, 0).reshape(-1, 16)

    def _check_step(self, frame, state):
        if type(frame) is not ContextualInput:
            raise ValueError("Use the checked contextual history codec")
        if frame.guard is not None:
            decision_support(frame.frame, frame.guard)
        if self.config.schema != model_schema_for(frame.frame.schema):
            raise ValueError("Model version does not match this explicit history profile")
        if (not isinstance(state, Tensor) or state.shape != (self.config.hidden_size,)
                or state.device != self.device or state.dtype != self.byte.weight.dtype):
            raise ValueError("State must match model dimensions, dtype and device")

    def _frame_features(self, frame, cache, final=False):
        """State-independent features; callers retain their own per-frame tree cache.

        ``final`` means nothing reads the cache after these features.
        """
        zero = self.byte.weight.new_zeros(self.config.feature_size)
        if frame.observation is None:
            map_features = own = zero
            cells = None
        else:
            map_features, own, cells = self._observation_features(frame.observation, cache)
        return torch.cat((self._encode_root(frame.semantics, cache, final), map_features, own)), cells

    def _observation_features(self, observation, cache):
        """``_encode_observation``, reused for an observation with identical contents.

        Only with the inference memo and a cache that holds nothing else yet, so
        the encoding and the tree cache entries it adds are a pure function of
        the observation (exactly keyed by ``_record_key``), ``config`` and the
        tree and map submodules. A hit returns the stored tensors and adds
        exactly the stored cache entries, as the computation would.
        """
        memo = cache.get(_LEAF_MEMO)
        store = (self._observation_memo(memo) if memo is not None and len(cache) == 1
                 and not torch.is_grad_enabled() else None)
        key = _record_key(observation) if store is not None else None
        if key is None:
            return self._encode_observation(observation, cache)
        entries = store["entries"]
        hit = entries.get(key)
        if hit is not None:
            global_map, own, cells, added = hit
            cache.update(added)
            return global_map, own, cells
        global_map, own, cells = self._encode_observation(observation, cache)
        entry = (global_map, own, cells, {k: v for k, v in cache.items() if k is not _LEAF_MEMO})
        size = _observation_entry_bytes(store, key, entry)
        if store["bytes"] + size > _OBSERVATION_MEMO_BYTES:
            entries.clear()  # bounded memory; a miss only recomputes
            store["bytes"], store["charged"] = 0, set()
            size = _observation_entry_bytes(store, key, entry)
        store["bytes"] += size
        entries[key] = entry
        return global_map, own, cells

    def _observation_memo(self, memo):
        """Observation entries that belong to this leaf memo and the current map submodules, or None.

        Like the leaf memo for the tree submodules, they are kept only while the
        map submodules' identity, class, mode, layout and every parameter's
        dtype, device, shape and bits equal the snapshot taken when they were
        started; a new leaf memo (config, inference mode, thread count or tree
        submodules changed) also starts them anew.
        """
        walked = self._walk_modules(_MAP_MODULES)
        if walked is None:
            return None
        layout, referenced, tensors = walked
        state = _bit_state(tensors)
        store = self.__dict__.get("_observation_memo_state")
        if (store is None or store["leaf"] is not memo or store["layout"] != layout or store["state"][0] != state[0]
                or not torch.equal(store["state"][1], state[1])):
            store = {"leaf": memo, "layout": layout, "referenced": referenced, "state": state, "entries": {}, "bytes": 0,
                     "charged": set()}
            self.__dict__["_observation_memo_state"] = store
        return store

    def _can_cache_prefix(self):
        # These exact modules have no stochastic or mutable training behavior.
        # In particular, the text/ordered GRUs have no dropout. Do not elide
        # custom forwards, hooks or autocast operations by reusing their output.
        # Accelerator kernels keep the reference path until separately verified.
        allowed = (ContextualActorCritic, nn.Embedding, nn.GRU, nn.GRUCell, nn.Sequential,
                   nn.Linear, nn.Tanh, nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten)
        if (type(self) is not ContextualActorCritic or self.device.type != "cpu"
                or torch._C._is_any_autocast_enabled()
                or nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks
                or any(name in self.__dict__ for name in
                       ("forward_step", "_frame_features", "_observation_features", "_encode_root", "_encode_tree", "_encode_observation",
                        "_check_step"))):
            return False
        return all(type(module) in allowed and not module._forward_hooks and not module._forward_pre_hooks
                   and "forward" not in module.__dict__
                   and (type(module) is not nn.GRU or module.dropout == 0)
                   for module in self.modules())

    def prefix_states(self, inputs, stops, vectors=None, keys=None):
        """Reconstruct detached states at frame boundaries under the current weights.

        The memo exists only for this synchronous call, before an optimizer step.
        It reuses pure frame/leaf encodings, never recurrent state across weight
        versions. Every chronological GRU transition still executes in order.
        Training mode is unchanged; train-mode leaves use only this local memo,
        not the persistent inference memo. Differentiable windows never see it.
        ``vectors`` may supply, for every frame before the last stop that has
        semantics, the exact encoding this memo would hold (``prefix_pool``
        computes them under the same weights); checks and transitions are unchanged.
        ``keys`` may supply the memo keys (``frame_keys_until``) of the same
        unchanged history, computed once instead of in every call.
        """
        stops = tuple(stops)
        if not stops or any(type(stop) is not int or not 0 <= stop <= len(inputs) for stop in stops):
            raise ValueError("Prefix boundaries must lie within the complete history")
        wanted, result = set(stops), {}
        state = self.initial_state()
        use_cache = self._can_cache_prefix()
        if vectors is not None and not use_cache:
            raise ValueError("Supplied prefix encodings need the exact cached reconstruction path")
        # Internal-node entries obey the same exact-tree rule as the inference memo.
        features, leaves = {}, {"text": {}, "number": {}, "node": {}}
        with torch.no_grad():
            if 0 in wanted:
                result[0] = state.detach()
            for seq in range(max(stops)):
                frame = inputs[seq]
                if not use_cache:
                    state = self.forward_step(frame, state).state
                else:
                    self._check_step(frame, state)
                    if frame.semantics is not None and vectors is not None:
                        state = self.gru(vectors[seq], state)
                    elif frame.semantics is not None:
                        # repr preserves number types and signed zeros. A fresh
                        # tree cache on a miss preserves within-frame evaluation
                        # order and the original cache's equality semantics.
                        key = frame_key(frame) if keys is None else keys[seq]
                        vector = features.get(key)
                        if vector is None:
                            vector, _ = self._frame_features(frame, {_LEAF_MEMO: leaves})
                            if len(features) >= 4096:
                                features.clear()
                            features[key] = vector
                        state = self.gru(vector, state)
                if seq + 1 in wanted:
                    result[seq + 1] = state.detach()
        return result

    def forward_step(self, frame, state):
        self._check_step(frame, state)
        if frame.semantics is None:
            return StepOutput(state, None)
        cache = {}
        memo = self._leaf_memo()
        if memo is not None:
            cache[_LEAF_MEMO] = memo
        graph = self.__dict__.get("_graph_memo_state")
        if graph is not None and torch.is_grad_enabled():
            cache[_GRAPH_MEMO] = graph
        replay = self.__dict__.get("_leaf_replay_state")
        if replay is not None and graph is None and torch.is_grad_enabled():
            cache[_LEAF_REPLAY] = replay
        vector, cells = self._frame_features(frame, cache, final=frame.frame.kind != "decision")
        state = self.gru(vector, state)
        decision = None
        if frame.frame.kind == "decision":
            distribution = ContextualDistribution(self, state, frame.frame.observation, frame.frame.offers, cells, cache, frame.guard)
            decision = PolicyValue(self.value(state).squeeze(-1), distribution)
        return StepOutput(state, decision)


class ContextualDistribution:
    """Typed autoregressive distribution, exact support and chosen-path log P.

    Quantity uses a balanced binary interval tree. At most 31 learned choices
    represent every legal int32 count. There is deliberately no entropy method:
    exact joint count entropy would require visiting every leaf. PPO starts with
    entropy coefficient zero, not a selected-path value mislabeled as entropy.
    """

    def __init__(self, model, state, observation, offers, cells, cache, guard=None):
        self.model, self.state, self.observation, self.offers = model, state, observation, offers
        self.cells, self.cache = cells, cache
        self.guard = guard
        self.support = DecisionSupport(offers.schema, offers.context, offers,
                                       distribution_schema_for(offers.schema, guard.version if guard else None), guard)

    def _features(self, semantic):
        return self.model._encode_root(semantic, self.cache)

    def _logits(self, semantics):
        candidates = torch.stack([self._features(item) for item in semantics])
        return candidates.mv(torch.tanh(self.model.policy_query(self.state))) / sqrt(self.model.config.feature_size)

    def _quantity_logits(self, features, low, high, maximum, depth):
        numeric = self.model._float((low / maximum, high / maximum, log1p(maximum), depth / 31))
        logits = self.model.quantity(torch.cat((self.state, features, numeric)))
        middle = (low + high) // 2
        return logits + self.model._float((log1p(middle - low), log1p(high - middle - 1)))

    def _coordinate_logits(self, semantic, width, height):
        positions = self.model._float([(px / max(width - 1, 1), py / max(height - 1, 1))
                                      for py in range(height) for px in range(width)])
        target = self._features(semantic).expand(width * height, -1)
        state = self.state.expand(width * height, -1)
        return self.model.coordinate(torch.cat((state, target, self.cells, positions), -1)).squeeze(-1)

    def _coordinate_subset(self, logits, coordinates):
        return logits[torch.tensor(coordinates, dtype=torch.long, device=logits.device)]

    def _choose(self, name, logits, selected, generator, trace):
        log_probs = torch.log_softmax(logits, 0)
        if selected is None:
            selected = int(torch.multinomial(log_probs.detach().exp(), 1, generator=generator).item())
        trace.append(ConditionalProbability(name, selected, logits.numel(), log_probs[selected]))
        return selected

    def _kinds(self):
        kinds = self._native_kinds()
        if self.guard is None:
            return kinds
        return tuple(kind for kind in kinds if any(self._permitted(action) for action in self._representatives(kind)))

    def _native_kinds(self):
        if self.offers.capability_panel is not None:
            panel = self.offers.capability_panel
            return (("close_capability",) if not panel.mandatory else ()) + tuple(
                kind for kind in c.CAPABILITY_ACTIONS if any(choice.kind == kind for choice in panel.choices))
        kinds = self._legacy_kinds()
        return kinds + (("open_capability",) if self.offers.capability_controls else ())

    def _legacy_kinds(self):
        offers, context = self.offers, self.offers.context
        if context.kind == "dialog":
            return ("choose_dialog",)
        if context.kind == "map":
            movable = offers.hero_refs if offers.movable_hero_refs is None else offers.movable_hero_refs
            return tuple(kind for kind in _ACTIONS if kind == "end_turn"
                         or kind == "move_to" and movable or kind == "open_hero" and offers.hero_refs
                         or kind == "open_town" and offers.town_refs)
        if context.kind == "stack":
            if context.panel == "overview":
                return ("back",) + (("select_upgrade",) if offers.stack_panel.offers else ())
            return ("back",) + (("upgrade_stack",) if offers.stack_panel.affordable else ())
        if offers.armies:
            kinds = ("close",) + (("open_panel",) if offers.panel and offers.panel.controls else ())
            kinds += tuple(kind for kind in ("split_stack", "merge_stacks", "swap_stacks") if _has_army_operation(offers.armies, kind))
            kinds += (("open_stack",) if any(slot.count for army in offers.armies for slot in army.slots) else ())
            if context.kind == "town" and any(army.role in ("garrison_hero", "visiting_hero") for army in offers.armies):
                kinds += ("swap_garrison",)
            return kinds
        if offers.panel is None:
            return ("close",)
        if context.panel == "overview":
            kinds = ("close", "open_panel") if offers.panel.controls else ("close",)
            return kinds + (("transfer_stack",) if offers.panel.armies and offers.panel.armies.choices() else ())
        if context.panel == "hall":
            return ("open_building", "back") if offers.panel.buildings else ("back",)
        if context.panel == "building":
            return ("construct", "back") if offers.panel.building.status == "allowed" else ("back",)
        if context.panel == "tavern":
            tavern = offers.panel.tavern
            return (("back",) + (("select_hero",) if tavern.offers else ())
                    + (("hire_hero",) if tavern.recruit_status == "allowed" else ()))
        recruitment = offers.panel.recruitment
        return ("back", "select_recruit", "recruit") if recruitment.slider_max else ("back", "select_recruit")

    def _action(self, kind, **values):
        subject = None if kind in ("close", "end_turn") else self.offers.context.subject
        return c.Action(kind, schema=self.offers.schema, **dict({"subject": subject}, **values))

    def _permitted(self, action):
        return self.guard is None or self.guard.permits(action, self.observation, self.offers)

    def _filtered(self, choices, action_for):
        # Guard predicates depend on the action's actors, not its quantity. The
        # count tree therefore retains every native count for an admitted pair.
        if self.guard is None:
            return choices
        return tuple(choice for choice in choices if self._permitted(action_for(choice)))

    def _coordinates(self, subject):
        width, height, _ = self.offers.size
        return (index for index in range(width * height)
                if self._permitted(self._action("move_to", subject=subject, x=index % width, y=index // width)))

    def _subjects(self, kind):
        refs = (self.offers.town_refs if kind == "open_town" else self.offers.movable_hero_refs
                if kind == "move_to" and self.offers.movable_hero_refs is not None else self.offers.hero_refs)
        if self.guard is None:
            return refs
        if kind == "move_to":
            return tuple(ref for ref in refs if ref in self.guard.movable_heroes
                         and next(self._coordinates(ref), None) is not None)
        return self._filtered(refs, lambda ref: self._action(kind, subject=ref))

    def _panel_action(self, control):
        return self._action("open_panel", panel=control.panel,
                            selection=control.ref if control.panel == "recruitment" else None)

    def _pair_action(self, kind, pair):
        source, destination, _ = pair
        return self._action(kind, selection=source, destination=destination,
                            count=1 if kind in ("transfer_stack", "split_stack") else None)

    def _representatives(self, kind):
        """One completion per non-quantity leaf, used before choosing a kind.

        The visible interaction budget is quantity independent. Movement stays
        lazy here, so testing whether a hero has a destination never materializes
        the rectangle or scores a second policy distribution.
        """
        offers = self.offers
        if kind in ("move_to", "open_hero", "open_town"):
            for ref in self._subjects(kind):
                if kind == "move_to":
                    index = next(self._coordinates(ref), None)
                    if index is not None:
                        yield self._action(kind, subject=ref, x=index % offers.size[0], y=index // offers.size[0])
                else:
                    yield self._action(kind, subject=ref)
        elif kind == "open_capability":
            for panel in offers.capability_controls:
                yield self._action(kind, panel=panel)
        elif kind in c.CAPABILITY_ACTIONS:
            for choice in offers.capability_panel.choices:
                if choice.kind == kind:
                    yield self._action(kind, selection=choice.ref, count=1 if kind == "trade_resource" else None)
        elif kind == "choose_dialog":
            for choice in offers.dialog_panel.choices:
                yield self._action(kind, selection=choice.ref)
        elif kind == "open_panel":
            for control in offers.panel.controls:
                yield self._panel_action(control)
        elif kind == "open_stack":
            for army in offers.armies:
                for slot in army.slots:
                    if slot.count:
                        yield self._action(kind, selection=slot.ref)
        elif kind in ("select_upgrade", "open_building", "select_recruit", "select_hero"):
            choices = (offers.stack_panel.offers if kind == "select_upgrade" else
                       offers.panel.buildings if kind == "open_building" else
                       offers.panel.tavern.offers if kind == "select_hero" else offers.panel.recruitment.offers)
            for choice in choices:
                yield self._action(kind, selection=choice.ref)
        elif kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            choices = (offers.panel.armies.choices() if kind == "transfer_stack"
                       else c.army_operation_choices(offers.armies, kind))
            for pair in choices:
                yield self._pair_action(kind, pair)
        elif kind == "recruit":
            recruit = offers.panel.recruitment
            yield self._action(kind, selection=recruit.selected_offer_ref, destination=recruit.destination.ref, count=1)
        elif kind == "construct":
            yield self._action(kind, selection=offers.panel.building.ref)
        elif kind == "hire_hero":
            yield self._action(kind, selection=offers.panel.tavern.selected_offer_ref)
        elif kind == "upgrade_stack":
            yield self._action(kind, selection=offers.stack_panel.selected_upgrade_ref)
        else:
            yield self._action(kind)

    def _walk(self, action=None, generator=None):
        if action is not None and (not self.offers.permits(action) or not self._permitted(action)):
            raise ValueError("Action is outside this exact behavior support")
        memo = self.model._leaf_memo()  # revalidated: weights or modes may differ from forward_step
        if memo is not self.cache.get(_LEAF_MEMO):
            # Values cached under other weights or modes must never seed internal-node entries.
            self.cache[_EXACT_CACHE] = False
        if memo is None:
            self.cache.pop(_LEAF_MEMO, None)
        else:
            self.cache[_LEAF_MEMO] = memo
        trace = []

        def choose(name, logits, selected=None):
            return self._choose(name, logits, selected, generator, trace)

        kinds = self._kinds()
        kind = kinds[choose("kind", self._logits([_node("action_kind", _text(kind)) for kind in kinds]),
                            kinds.index(action.kind) if action else None)]
        subject = panel = selection = destination = x = y = count = None
        if kind == "open_capability":
            controls = self._filtered(self.offers.capability_controls, lambda panel: self._action(kind, panel=panel))
            index = choose("capability_panel", self._logits([_node("capability_panel", _text(item)) for item in controls]),
                           controls.index(action.panel) if action else None)
            subject, panel = self.offers.context.subject, controls[index]
        elif kind == "close_capability":
            subject = self.offers.context.subject
        elif kind in c.CAPABILITY_ACTIONS:
            choices = self._filtered(tuple(item for item in self.offers.capability_panel.choices if item.kind == kind),
                                     lambda item: self._action(kind, selection=item.ref, count=1 if kind == "trade_resource" else None))
            index = choose("capability_choice", self._logits([_capability_choice(item, self.observation) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            choice = choices[index]
            subject, selection = self.offers.context.subject, choice.ref
            if kind == "trade_resource":
                low, high, maximum, depth = 1, choice.max_count, choice.max_count, 0
                features = self._features(_capability_choice(choice, self.observation))
                while low < high:
                    middle = (low + high) // 2
                    logits = self._quantity_logits(features, low, high, maximum, depth)
                    branch = choose(f"count:{low}:{high}", logits, int(action.count > middle) if action else None)
                    if branch: low = middle + 1
                    else: high = middle
                    depth += 1
                count = low
        elif kind == "choose_dialog":
            delivered = self.offers.dialog_panel
            choices = delivered.choices
            index = choose("dialog_choice", self._logits([_dialog_choice(delivered, item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("move_to", "open_hero", "open_town"):
            refs = self._subjects(kind)
            semantics = [_node(kind, _subject(self.observation, ref)) for ref in refs]
            index = choose("subject", self._logits(semantics), refs.index(action.subject) if action else None)
            subject = refs[index]
            if kind == "move_to":
                width, height, _ = self.offers.size
                logits = self._coordinate_logits(semantics[index], width, height)
                if self.guard is None:
                    index = choose("coordinate", logits, action.y * width + action.x if action else None)
                else:
                    coordinates = tuple(self._coordinates(subject))
                    logits = self._coordinate_subset(logits, coordinates)
                    index = coordinates[choose("coordinate", logits,
                                        coordinates.index(action.y * width + action.x) if action else None)]
                x, y = index % width, index // width
        elif kind == "open_panel":
            controls = self.offers.panel.controls
            # Hall and tavern each have one semantic navigation action;
            # duplicate controls must never double-count its probability mass.
            if any(sum(control.panel == panel for control in controls) > 1 for panel in ("hall", "tavern")):
                raise ValueError("Duplicate semantic town panel controls")
            controls = self._filtered(controls, self._panel_action)
            selected = next((i for i, control in enumerate(controls) if action and control.panel == action.panel
                             and (control.panel in ("hall", "tavern") or control.ref == action.selection)), None)
            index = choose("control", self._logits([_semantic(control) for control in controls]), selected)
            control = controls[index]
            subject, panel = self.offers.context.subject, control.panel
            selection = control.ref if panel == "recruitment" else None
        elif kind == "open_stack":
            choices = tuple(slot.ref for army in self.offers.armies for slot in army.slots if slot.count)
            choices = self._filtered(choices, lambda ref: self._action(kind, selection=ref))
            index = choose("selection", self._logits([_army_slot(self.observation, ref, receiving=False) for ref in choices]),
                           choices.index(action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index]
        elif kind == "select_upgrade":
            choices = self._filtered(self.offers.stack_panel.offers, lambda item: self._action(kind, selection=item.ref))
            index = choose("selection", self._logits([_semantic(item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("open_building", "select_recruit", "select_hero"):
            choices = (self.offers.panel.buildings if kind == "open_building" else
                       self.offers.panel.tavern.offers if kind == "select_hero" else self.offers.panel.recruitment.offers)
            choices = self._filtered(choices, lambda item: self._action(kind, selection=item.ref))
            index = choose("selection", self._logits([_semantic(item) for item in choices]),
                           next(i for i, item in enumerate(choices) if item.ref == action.selection) if action else None)
            subject, selection = self.offers.context.subject, choices[index].ref
        elif kind in ("transfer_stack", "split_stack", "merge_stacks", "swap_stacks"):
            exchange = self.offers.panel.armies if kind == "transfer_stack" else None
            choices = exchange.choices() if exchange else c.army_operation_choices(self.offers.armies, kind)
            choices = self._filtered(choices, lambda pair: self._pair_action(kind, pair))
            sources = tuple(dict.fromkeys(source for source, _, _ in choices))
            source_semantics = [_army_slot(self.observation, ref, receiving=False) for ref in sources]
            if kind != "transfer_stack": source_semantics = [_node(kind, item) for item in source_semantics]
            source_index = choose("source_slot", self._logits(source_semantics),
                                  sources.index(action.selection) if action else None)
            selection = sources[source_index]
            destinations = tuple(destination for source, destination, _ in choices if source == selection)
            destination_index = choose("destination_slot", self._logits([
                _node("transfer_pair" if kind == "transfer_stack" else kind + "_pair", _army_slot(self.observation, selection, receiving=False),
                      _army_slot(self.observation, ref, receiving=True)) for ref in destinations]),
                destinations.index(action.destination) if action else None)
            destination = destinations[destination_index]
            subject = self.offers.context.subject
            low, maximum = 1, (exchange.maximum(selection, destination) if exchange else
                               c.army_operation_maximum(self.offers.armies, kind, selection, destination))
            high = maximum if kind in ("transfer_stack", "split_stack") else 1
            features = self._features(_node("transfer_quantity" if kind == "transfer_stack" else kind + "_quantity", _army_slot(self.observation, selection, receiving=False),
                                           _army_slot(self.observation, destination, receiving=True)))
            depth = 0
            while low < high:
                middle = (low + high) // 2
                logits = self._quantity_logits(features, low, high, maximum, depth)
                selected = int(action.count > middle) if action else None
                branch = choose(f"count:{low}:{high}", logits, selected)
                if branch: low = middle + 1
                else: high = middle
                depth += 1
            count = low if kind in ("transfer_stack", "split_stack") else None
        elif kind in ("construct", "back", "recruit", "hire_hero", "swap_garrison", "upgrade_stack"):
            subject = self.offers.context.subject
            if kind == "upgrade_stack":
                selection = self.offers.stack_panel.selected_upgrade_ref
            elif kind == "construct":
                selection = self.offers.panel.building.ref
            elif kind == "hire_hero":
                selection = self.offers.panel.tavern.selected_offer_ref
            elif kind == "recruit":
                recruit = self.offers.panel.recruitment
                selection, destination = recruit.selected_offer_ref, recruit.destination.ref
                low, high, maximum = 1, recruit.slider_max, recruit.slider_max
                chosen = next(card for card in recruit.offers if card.ref == selection)
                features = self._features(_node("quantity", _semantic(chosen), _semantic(recruit.destination)))
                depth = 0
                while low < high:
                    middle = (low + high) // 2
                    # A zero learned logit yields uniform integer mass even when
                    # the two subtree sizes differ. This is support geometry.
                    logits = self._quantity_logits(features, low, high, maximum, depth)
                    selected = int(action.count > middle) if action else None
                    branch = choose(f"count:{low}:{high}", logits, selected)
                    if branch: low = middle + 1
                    else: high = middle
                    depth += 1
                count = low
        selected_action = c.Action(kind, subject, x, y, schema=self.offers.schema,
                                   panel=panel, selection=selection, count=count, destination=destination)
        if not self.offers.permits(selected_action) or not self._permitted(selected_action) or action is not None and selected_action != action:
            raise ValueError("Conditional action construction disagrees with support")
        return selected_action, ActionEvaluation(torch.stack([part.log_prob for part in trace]).sum(), tuple(trace))

    def sample(self, generator=None):
        return self._walk(generator=generator)[0]

    def sample_evaluated(self, generator=None):
        """The sampled action and the evaluation the same walk computed for it.

        Equal to ``evaluate(sample(generator))``: every conditional uses the same
        logits, and only the selected indices come from the generator. Saved
        behavior audits recompute it through ``evaluate``.
        """
        return self._walk(generator=generator)

    def evaluate(self, action):
        return self._walk(action=action)[1]

    def log_prob(self, action):
        return self.evaluate(action).log_prob

    def trace(self, action):
        return self.evaluate(action).conditionals
