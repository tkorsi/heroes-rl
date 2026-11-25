"""``native.cpp``: the semantic trees of ``policy`` as a native DAG, built on demand for this interpreter.

``module()`` is the extension, or None where it cannot be built or ``HEROES_NATIVE=0``. It builds the trees
``policy._semantic`` and ``policy._observation_semantics`` build, from the same contract objects, and encodes them
through fastnet's own functions (``fast.library``).
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

_SOURCE = Path(__file__).with_name("native.cpp")
_MODULE = None  # the extension, False when it cannot be had


def enabled():
    return os.environ.get("HEROES_NATIVE", "1") != "0"


def _built(folder):
    headers = sysconfig.get_paths().get("include") or ""
    source = _SOURCE.read_bytes()
    stamp = hashlib.sha256(source + f"|{sys.version}|{headers}".encode()).hexdigest()[:16]
    target = folder / f"heroes_native-{stamp}{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"
    if not target.exists():
        folder.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix="heroes_native-", suffix=".so", dir=folder)
        os.close(handle)
        subprocess.run(["c++", "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{headers}", "-undefined", "dynamic_lookup",
                        str(_SOURCE), "-o", name], check=True, capture_output=True)
        os.replace(name, target)
    return target


def module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE or None
    _MODULE = False
    if not enabled():
        return None
    from . import fast, policy
    from ..game import contract
    lib = fast.library()
    if lib is None or not (Path(sysconfig.get_paths().get("include") or "") / "Python.h").is_file():
        return None
    path = _built(fast._build_folder())
    spec = importlib.util.spec_from_file_location("heroes_native", path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    address = lambda function: ctypes.cast(function, ctypes.c_void_p).value  # noqa: E731
    extension.configure(policy, contract, address(lib.fn_text), address(lib.fn_number), address(lib.fn_node))
    extension.configure_values(*_schema(contract))
    theirs = ctypes.CDLL(str(Path(fast.torch.__file__).parent / "lib" / "libtorch_cpu.dylib"))
    extension.configure_math(address(theirs.Sleef_expf4_u10), address(theirs.Sleef_logf4_u10))
    from ..engine.protocol import NativeFailure
    extension.set_native_failure(NativeFailure)
    _MODULE = extension
    return extension


_SPEC_KINDS = {"str": 0, "int": 1, "bool": 2, "optional": 3, "tuple": 4, "fixed": 5, "record": 6}


def _schema(contract):
    """The contract's dataclasses for native.cpp: ``(types, specs, missing)``.

    ``types`` lists ``(name, class, [(field, spec, default)])`` with each field's default (or ``missing``);
    ``specs`` lists ``(kind, type index, [inner specs])``, read from the fields' annotations.
    """
    import dataclasses

    classes = [value for value in vars(contract).values()
               if isinstance(value, type) and dataclasses.is_dataclass(value) and value.__module__ == contract.__name__]
    index = {cls.__name__: position for position, cls in enumerate(classes)}
    specs, known = [], {}

    def spec(text):
        text = text.strip()
        if text in known:
            return known[text]
        if text.endswith("| None"):
            entry = (_SPEC_KINDS["optional"], -1, [spec(text[: -len("| None")])])
        elif text.startswith("tuple[") and text.endswith("]"):
            parts, depth, start = [], 0, 6
            for position, char in enumerate(text[6:-1], 6):
                depth += char == "["
                depth -= char == "]"
                if char == "," and depth == 0:
                    parts.append(text[start:position])
                    start = position + 1
            parts.append(text[start:-1])
            parts = [part.strip() for part in parts]
            if len(parts) == 2 and parts[1] == "...":
                entry = (_SPEC_KINDS["tuple"], -1, [spec(parts[0])])
            else:
                entry = (_SPEC_KINDS["fixed"], -1, [spec(part) for part in parts])
        elif text in ("str", "int", "bool"):
            entry = (_SPEC_KINDS[text], -1, [])
        elif text in index:
            entry = (_SPEC_KINDS["record"], index[text], [])
        else:
            raise ValueError(f"native values: unsupported annotation {text!r}")
        known[text] = len(specs)
        specs.append(entry)
        return known[text]

    missing = object()
    types = []
    for cls in classes:
        fields = []
        for item in dataclasses.fields(cls):
            default = (item.default if item.default is not dataclasses.MISSING else item.default_factory()
                       if item.default_factory is not dataclasses.MISSING else missing)
            fields.append((item.name, spec(str(item.type)), default))
        types.append((cls.__name__, cls, fields))
    return types, specs, missing


def compact_of(arrays):
    """``Compact`` from the arrays ``Seat.finish`` gives: name -> (dtype, shape, bytes), or a plain value."""
    import numpy as np
    from ..learner.compact import Compact
    made = Compact()
    for name, value in arrays.items():
        if isinstance(value, tuple):
            dtype, shape, data = value
            value = np.frombuffer(data, dtype=dtype).reshape(shape)
        setattr(made, name, value)
    return made


class SampleFrame:
    """What ``actor.learner_samples`` reads of a frame."""
    __slots__ = ("kind", "seq", "decision_seq", "reason")

    def __init__(self, kind, seq, decision_seq, reason):
        self.kind, self.seq, self.decision_seq, self.reason = kind, seq, decision_seq, reason


_ADDRESSES = ("fn_used", "fn_step", "fn_value", "fn_query", "fn_logits", "fn_quantity", "fn_coordinate", "fn_map")


def seat(policy, model, producer, seed, guard, record):
    """A native ``Seat`` answering ``producer``'s decisions with ``policy`` (the ``FastPolicy`` of ``model``).

    ``record`` makes it record the learner's recipe as ``Compact`` arrays (``Seat.finish``).
    """
    from . import fast
    from ..game import guard as interaction_guard
    lib = fast.library()
    addresses = tuple(ctypes.cast(getattr(lib, name), ctypes.c_void_p).value for name in _ADDRESSES)
    limits = None
    if guard is not None:
        management, windows, decisions = interaction_guard.budgets(guard)
        limits = (management, -1 if windows is None else windows, -1 if decisions is None else decisions)
    weight = float(model.window_memory.weight.detach()[0])
    return module().Seat(producer, policy, policy.net, addresses, seed, limits, weight, model.config.max_map_side, bool(record))
