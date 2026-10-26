"""Finalization and identity rules of the in-process backend; never a learner input.

The executable backend proves a host stop with the SIGTERM/SIGKILL status of
its game process. The embedded backend has no game process: the engine's own
close report proves the stop instead (no exit status existed, every episode
thread joined, the opponent instance was released and no failure was found).
A natural ending still requires the engine's exit status 0 after the terminal
barrier, plus a clean close. builtin_evidence.read_finalization applies these
rules to manifests whose identity names this backend; everything else is the
unchanged one-seat evidence.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
from pathlib import Path
import subprocess

from experiments.headless import contextual_evidence as native

ROOT = Path(__file__).resolve().parents[2]
IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

BACKEND = "embedded-in-process-v1"
PROTOCOL = "heroes-embedded-engine-v2"
CLOSE_SCHEMA = "heroes-embedded-close-v1"
MODULE_NAME = "heroes_embedded_engine"
NATIVE_SESSION = "engine/NativeSession"
ADAPTER_SOURCES = frozenset(("experiments/headless/embedded_env.py", "experiments/headless/embedded_evidence.py"))
CLOSE_FIELDS = ("exit_code", "natural", "host_stopped", "shutdown_deadline", "close_clean", "poisoned",
                "experiment_failed", "channel_failure", "close_error", "decode_error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_embedded(identity):
    backend = identity.get("backend") if type(identity) is dict else None
    return type(backend) is dict and backend.get("kind") == BACKEND


def module_path(binary_directory):
    """The one built extension for this interpreter, beside the executable runtime."""
    folder = Path(binary_directory).resolve().parent / "embedded"
    found = [folder / (MODULE_NAME + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
             if (folder / (MODULE_NAME + suffix)).is_file()]
    require(len(found) >= 1, f"Build {MODULE_NAME} for this Python first (-DENABLE_EMBEDDED_ENGINE=ON): {folder}")
    return found[0]


def native_session_digests(root=None):
    """Authored host sources compiled into the module; ENGINE_RECIPE stays unchanged."""
    root = Path(root or ROOT)
    try:
        listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard", "--",
                                 NATIVE_SESSION], capture_output=True, check=True).stdout
        paths = [root / name for name in listed.decode().split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        paths = [path for path in (root / NATIVE_SESSION).rglob("*")
                 if not any(part.startswith(".") for part in path.relative_to(root).parts)]
    return {str(path.relative_to(root)): native.file_identity(path)["sha256"] for path in sorted(paths) if path.is_file()}


def backend_identity(binary_directory, root=None):
    module = module_path(binary_directory)
    return {"kind": BACKEND, "protocol": PROTOCOL,
            "module": str(module.relative_to(Path(binary_directory).resolve().parent)),
            "module_sha256": native.file_identity(module)["sha256"],
            "native_session_sha256": native_session_digests(root)}


def verify_backend(identity, binary_directory, root=None):
    """The captured module file and host sources must be the ones present now.

    The module is the file the capture names, whatever Python ABI verifies it.
    """
    require(is_embedded(identity), "Not an embedded-backend capture")
    captured = identity["backend"]
    require(type(captured) is dict and set(captured) == {"kind", "protocol", "module", "module_sha256",
            "native_session_sha256"} and captured["protocol"] == PROTOCOL, "Invalid embedded backend identity")
    name = captured["module"]
    require(type(name) is str and Path(name).parent == Path("embedded") and Path(name).name.startswith(MODULE_NAME),
            "Invalid embedded module path")
    module = native.inside(Path(binary_directory).resolve().parent, name)
    require(module.is_file() and native.file_identity(module)["sha256"] == captured["module_sha256"],
            "Captured embedded engine module changed")
    require(captured["native_session_sha256"] == native_session_digests(root),
            "Captured embedded host sources changed")
    return {"embedded_module_unchanged": True, "embedded_host_sources_unchanged": True}


def require_consistent_backend(manifest, final):
    """Every bound field must name the same backend; no mixture selects weaker rules."""
    launch = manifest.get("launch") if type(manifest.get("launch")) is dict else {}
    if is_embedded(manifest.get("identity")):
        require(launch.get("backend") == BACKEND and launch.get("command") is None and "native_close" in final
                and (launch.get("environment") or {}).get("EXPLORER_RL_TRANSPORT") == "in-process",
                "Embedded identity without an embedded launch and close report")
    else:
        require("backend" not in launch and ("command" not in launch or type(launch["command"]) is list)
                and (launch.get("environment") or {}).get("EXPLORER_RL_TRANSPORT") != "in-process"
                and "native_close" not in final, "Executable-backend capture carries embedded launch evidence")


def close_record(report):
    """Compact, JSON-safe copy of the engine's finish() report (messages excluded)."""
    require(type(report) is dict, "Missing embedded close report")
    record = {"schema": CLOSE_SCHEMA, **{key: report.get(key) for key in CLOSE_FIELDS}}
    opponent = report.get("opponent") or {}
    record["opponent"] = {key: opponent.get(key) for key in ("module", "player", "difficulty", "created", "released",
                          "module_reset", "turns", "turn_seconds", "battles", "battle_seconds")}
    record["wait_seconds"] = round(float(report.get("wait_seconds") or 0.0), 6)
    record["close_seconds"] = round(float(report.get("close_seconds") or 0.0), 6)
    return record


def _close(final):
    close = final.get("native_close")
    require(type(close) is dict and close.get("schema") == CLOSE_SCHEMA, "Missing embedded close evidence")
    return close


def _opponent_closed(close):
    """The owned opponent existed, died with its game and its module state was reset."""
    opponent = close.get("opponent") if type(close.get("opponent")) is dict else {}
    return opponent.get("created") is True and opponent.get("released") is True and opponent.get("module_reset") is True


def truncation_closed(final):
    """A host stop before any exit status, with a clean join and no failure."""
    close = _close(final)
    return (final.get("returncode") is None and close.get("exit_code") is None and close.get("natural") is False
            and close.get("host_stopped") is True and close.get("close_clean") is True
            and close.get("poisoned") is False and close.get("experiment_failed") is False
            and close.get("channel_failure") is None and close.get("close_error") is None
            and close.get("decode_error") is None and _opponent_closed(close))


def natural_closed(final):
    """The engine's own exit status 0 after the terminal barrier, then a clean join."""
    close = _close(final)
    return (final.get("returncode") == 0 and type(final.get("returncode")) is int and close.get("exit_code") == 0
            and close.get("natural") is True and close.get("host_stopped") is False
            and close.get("shutdown_deadline") is False and close.get("close_clean") is True
            and _opponent_closed(close))
