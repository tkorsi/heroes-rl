"""Explicit, source- and binary-bound native learner loading; never an implicit fallback.

The extension shares the existing model's tensors. It does not own checkpoint,
optimizer, sampling, admission or official-run state. Building is an explicit
preparation step into a fresh directory outside the source tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-native-learner-build-v1"
CONTRACT = "heroes-native-learner-v1"
FLAGS = ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
SOURCES = ("native/learner/kernels.cpp", "native/learner/text_replay.h", "native/learner/replay_tape.h",
           "experiments/learner/native_backend.py", "tools/build_native_learner.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity():
    import torch
    torch_root = Path(torch.__file__).resolve().parent
    libraries = sorted(path for path in (torch_root / "lib").iterdir()
                       if path.is_file() and (path.name.startswith("libtorch") or path.name.startswith("libc10")))
    if not libraries:
        raise RuntimeError("The installed PyTorch runtime libraries are missing")
    return {
        "sources": {name: sha256(ROOT / name) for name in SOURCES},
        "runtime": {
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "python_soabi": sysconfig.get_config_var("SOABI"), "machine": platform.machine(),
            "system": platform.system(), "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "torch_libraries": {path.name: sha256(path) for path in libraries},
        },
        "compile_flags": FLAGS,
        "contract": CONTRACT,
    }


def build_backend(output_dir, *, jobs=2, verbose=False):
    """Build from scratch and bind the actual loadable library to its sources/ABI."""
    if type(jobs) is not int or not 1 <= jobs <= 16:
        raise ValueError("Native build jobs must lie between one and sixteen")
    output = Path(output_dir).expanduser().absolute()
    if output.exists():
        raise FileExistsError("Native builds require a fresh directory, never stale object reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve().is_relative_to(ROOT):
        raise ValueError("Keep native builds outside the authored source tree")
    output.mkdir(mode=0o700)
    identity = _identity()
    key = hashlib.sha256(_json_bytes(identity)).hexdigest()
    name = "heroes_native_" + key[:24]
    request = {"schema": SCHEMA, "identity": identity, "module_name": name, "jobs": jobs}
    (output / "build-request.json").write_text(json.dumps(request, indent=2) + "\n")
    from torch.utils.cpp_extension import load
    previous = os.environ.get("MAX_JOBS")
    try:
        os.environ["MAX_JOBS"] = str(jobs)
        module = load(name=name, sources=[str(ROOT / "native/learner/kernels.cpp")],
                      extra_cflags=FLAGS, build_directory=str(output), with_cuda=False, verbose=verbose)
    finally:
        if previous is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous
    if module.contract != CONTRACT:
        raise RuntimeError("The rebuilt native library exposes the wrong contract")
    if identity != _identity():
        raise RuntimeError("Native source or PyTorch ABI changed during compilation")
    binary = Path(module.__file__).resolve()
    if not binary.is_relative_to(output.resolve()):
        raise RuntimeError("Native build loaded a library outside its fresh output directory")
    compiler = os.environ.get("CXX", "c++")
    compiler_version = subprocess.run([compiler, "--version"], check=True, text=True,
                                      capture_output=True, timeout=15).stdout
    receipt = {
        **request,
        "binary": {"path": str(binary), "sha256": sha256(binary), "bytes": binary.stat().st_size},
        "compiler": {"command": compiler, "version": compiler_version},
        "loaded_contract": module.contract,
        "official_training_started": False,
    }
    manifest = output / "manifest.json"
    with manifest.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    verified_manifest(manifest, sha256(manifest))
    return {"path": str(manifest), "sha256": sha256(manifest), "bytes": manifest.stat().st_size}


def verified_manifest(manifest_path, expected_sha256=None):
    """Verify current source, installed runtime, and exact built bytes before loading."""
    path = Path(manifest_path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native build manifest must be a regular file")
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Native build manifest digest differs from its bound identity")
    receipt = json.loads(path.read_text())
    if (type(receipt) is not dict or receipt.get("schema") != SCHEMA
            or receipt.get("loaded_contract") != CONTRACT or receipt.get("identity") != _identity()):
        raise ValueError("Native source, build flags or installed PyTorch/Python ABI differs from the build")
    expected_name = "heroes_native_" + hashlib.sha256(_json_bytes(receipt["identity"])).hexdigest()[:24]
    if receipt.get("module_name") != expected_name:
        raise ValueError("Native module name does not bind the current source/ABI")
    binary_info = receipt.get("binary", {})
    binary = Path(binary_info.get("path", ""))
    if (not binary.is_absolute() or binary.is_symlink() or not binary.is_file()
            or binary.resolve().parent != path.resolve().parent
            or binary.stat().st_size != binary_info.get("bytes")
            or sha256(binary) != binary_info.get("sha256")):
        raise ValueError("Native binary differs from its build receipt")
    return receipt


def load_backend(manifest_path, expected_sha256=None):
    """Load only an explicit verified build; no auto-build and no fallback backend."""
    receipt = verified_manifest(manifest_path, expected_sha256)
    name, binary = receipt["module_name"], receipt["binary"]["path"]
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != Path(binary).resolve() or existing.contract != CONTRACT:
            raise RuntimeError("A different native module is already loaded under this build identity")
        return existing
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot construct the native learner module loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "contract", None) != CONTRACT:
        raise RuntimeError("Loaded native learner has the wrong contract")
    sys.modules[name] = module
    return module


# Descriptive alias for diagnostic callers; official plans bind an explicit manifest.
load_native_backend = load_backend
