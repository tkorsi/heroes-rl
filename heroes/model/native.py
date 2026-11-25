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
    _MODULE = extension
    return extension
