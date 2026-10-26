"""Diagnostic evidence for native contextual completion; never a learner input.

A clean exit and two socket EOFs cannot establish native packet drainage. The
server's terminal barrier and the client's post-callback receipt must agree.
Replay separately verifies the request and packet positions against the engine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "heroes-contextual-terminal-barrier-v1"
FILES = tuple(f"engine/terminal-barrier-{side}.jsonl" for side in ("server", "client"))
COMMON = {"schema", "kind", "episode", "interface", "barrier_id", "origin", "request",
          "client_packets", "outcomes"}
REPLAY_FILES = ("engine/replay.jsonl", "engine/contextual-replay.jsonl")
ADAPTER_SOURCES = {"experiments/headless/env.py", "experiments/headless/contextual_env.py",
                   "experiments/headless/contextual_protocol.py", "experiments/headless/contextual_evidence.py",
                   "experiments/learner/contextual.py"}


def _unique(pairs):
    # dict() keeps first insertion order and drops nothing unless a key repeats;
    # only then does the loop name the first repeated key, as it always did.
    result = dict(pairs)
    if len(result) == len(pairs):
        return result
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate evidence field: {key}")
        result[key] = value
    return result


def strict_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite evidence number: {value}")
    return json.loads(data, object_pairs_hook=_unique, parse_constant=invalid)


def file_identity(path):
    digest, size = _sha256_file(path)
    return {"sha256": digest, "bytes": size}


def _sha256_file(path):
    # Also imported as top-level ``contextual_evidence`` (contextual_env puts this directory on
    # sys.path); without the repository root there, hash directly as before.
    try:
        from experiments.headless.digest_scope import sha256_file
    except ImportError:
        data = Path(path).read_bytes()
        return hashlib.sha256(data).hexdigest(), len(data)
    return sha256_file(path)


def inside(root, name):
    """Evidence paths may neither escape the episode nor use symlinks."""
    root, relative = Path(root).resolve(), Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Evidence path must be relative to the episode")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Evidence path must not use symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Evidence path escapes the episode")
    return path


def runtime_identity(directory):
    """All local rules and AI libraries plus verifier, including version aliases."""
    directory = Path(directory).resolve()
    paths = [directory / "vcmireplay", *directory.glob("*.dylib"), *directory.glob("*.so*"),
             *directory.glob("AI/*.dylib"), *directory.glob("AI/*.so*")]
    result = {}
    for path in sorted(set(paths)):
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Native runtime symlink escapes its build directory")
        if path.is_file():
            result[str(path.relative_to(directory))] = file_identity(path)["sha256"]
    suffix = ".dylib" if "libvcmi.dylib" in result else ".so"
    required = {"vcmireplay", "libvcmi" + suffix, "AI/libExplorerRL" + suffix, "AI/libBattleAI" + suffix}
    if not required <= result.keys():
        raise ValueError("Native runtime inventory lacks the verifier, rules or required AI libraries")
    return result


def terminal_barrier(directory, *, episode, interface, outcomes):
    """Validate both receipts, returning their exact file and semantic identity.

    This version supports request-caused endings only. Initial and timer endings
    require an explicit replayable origin before they can receive a reward.
    """
    if (type(episode) is not str or not episode or interface not in
            ("adventure-windows-v1", "adventure-town-v1", "adventure-armies-v1", "adventure-heroes-v1", "adventure-army-management-v1", "adventure-dialogs-v1", "adventure-capabilities-v1") or type(outcomes) is not list
            or outcomes not in (["win", "loss"], ["loss", "win"])):
        raise ValueError("Invalid expected terminal barrier identity")
    records, files = [], {}
    for side, name in zip(("server", "client"), FILES):
        path = inside(directory, name)
        data = path.read_bytes()
        if not data.endswith(b"\n") or len(data.splitlines()) != 1:
            raise ValueError(f"{side} barrier must contain exactly one complete record")
        record = strict_json(data)
        expected = COMMON | ({"callbacks_complete", "observed_count"} if side == "client" else set())
        if type(record) is not dict or set(record) != expected:
            raise ValueError(f"Unexpected or missing {side} barrier fields")
        if (record["schema"] != SCHEMA or record["kind"] != side or record["episode"] != episode
                or record["interface"] != interface or type(record["barrier_id"]) is not int
                or record["barrier_id"] != 1 or record["origin"] != "request"
                or record["outcomes"] != outcomes):
            raise ValueError(f"{side} barrier identity or outcomes mismatch")
        request = record["request"]
        if (type(request) is not dict or set(request) != {"player", "id", "type"}
                or type(request["player"]) is not int
                or not (request["player"] in (0, 1) or (request["player"] == -1 and request["type"] == 198))
                or type(request["id"]) is not int or not 0 <= request["id"] <= 0xffffffff
                or type(request["type"]) is not int or not 1 <= request["type"] <= 0xffff):
            raise ValueError(f"Invalid {side} barrier request")
        count = record["client_packets"]
        if type(count) is not int or not 1 <= count <= 0x7fffffffffffffff:
            raise ValueError(f"Invalid {side} barrier packet count")
        if side == "client" and (record["callbacks_complete"] is not True
                or type(record["observed_count"]) is not int or record["observed_count"] != count):
            raise ValueError("Client barrier has not completed every callback")
        records.append(record)
        files[name] = file_identity(path)
    for key in COMMON - {"kind"}:
        if records[0][key] != records[1][key]:
            raise ValueError(f"Terminal barrier server/client mismatch: {key}")
    return {"schema": SCHEMA, "verified": True, "files": files, "server": records[0], "client": records[1]}


def capture_files(directory):
    """Freeze file identities; recording evidence never implies replay success."""
    names = (*REPLAY_FILES, *FILES)
    existing = {name: file_identity(inside(directory, name)) for name in names
                if inside(directory, name).is_file()}
    missing = [name for name in REPLAY_FILES if name not in existing or not existing[name]["bytes"]]
    return {"requested": True, "schema": "heroes-contextual-replay-v1", "status": "recorded" if not missing else "incomplete",
            "recorded": not missing, "verified": False, "files": existing, "missing": missing}


def terminal_capture_close(directory, barrier):
    """Require the recorder's flushed close after the terminal transaction."""
    path = inside(directory, "engine/contextual-replay.jsonl")
    with path.open("rb") as stream:
        end = stream.seek(0, 2)
        stream.seek(max(0, end - 4096))
        tail = stream.read()
    if not tail.endswith(b"\n"):
        raise ValueError("Contextual terminal capture ends inside a record")
    line = tail[:-1].rsplit(b"\n", 1)[-1]
    record = strict_json(line)
    if (type(record) is not dict or set(record) != {"seq", "kind", "barrier_id", "client_packets"}
            or type(record["seq"]) is not int or record["seq"] < 0 or record["kind"] != "close"
            or type(record["barrier_id"]) is not int or record["barrier_id"] != barrier["client"]["barrier_id"]
            or type(record["client_packets"]) is not int
            or record["client_packets"] != barrier["client"]["client_packets"]):
        raise ValueError("Contextual terminal capture lacks its matching final close")
    return record
