"""Fresh-interpreter native replay jobs, with private completion and owned deadlines.

The coordinator retains every admission capability. A child only writes the
ordinary independent replay reports; its private pipe binds their exact bytes
to this live launch. No training receipt or policy reconstruction is delegated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

from . import contextual_evidence as evidence
from .process_deadline import guarded_popen, close_owned_process, _release_owned_process

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-spawned-native-replay-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _files(folder, names):
    return {name: evidence.file_identity(evidence.inside(folder, name)) for name in names}


class SpawnedReplay:
    """One exact replay in an exec child; never accepts an existing result file."""

    def __init__(self, folder, *, binary, data, original_files, deadline_monotonic, deadline_wall):
        self.folder, self.binary, self.data = (Path(p).resolve() for p in (folder, binary, data))
        self.original_files = dict(original_files)
        self.deadline, self.wall_deadline = deadline_monotonic, deadline_wall
        self.token = secrets.token_hex(32)
        self.process = self.log = None
        self.consumed = False
        self.closed = False
        self.output = self.folder / "training-verification"

    def start(self):
        _require(self.process is None, "A native replay job starts once")
        _require(not self.output.exists(), "Never reuse an existing replay output")
        _require(_files(self.folder, self.original_files) == self.original_files,
                 "Capture changed before spawned verification")
        request = dict(schema=SCHEMA, token=self.token, folder=str(self.folder), binary=str(self.binary),
                       data=str(self.data), original_files=self.original_files,
                       deadline_monotonic=self.deadline, deadline_wall=self.wall_deadline)
        self.log = (self.folder / "spawned-replay.log").open("xb")
        try:
            self.process = guarded_popen([sys.executable, "-m", "experiments.headless.spawned_replay", "worker"],
                deadline_monotonic=self.deadline, deadline_wall=self.wall_deadline,
                evidence_path=self.folder / "spawned-replay-deadline.jsonl", cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
            self.process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
            self.process.stdin.close()
        except BaseException as error:
            try:
                self.stop()
            except BaseException as cleanup_error:
                error.verification_cleanup_verified = False
                error.add_note(f"Spawned verification launch cleanup failed: {cleanup_error}")
            raise
        return self

    def result(self, *, timeout):
        _require(self.process is not None and not self.consumed, "Replay completion is not fresh")
        self.consumed = True
        allowance = min(timeout, self.deadline-time.monotonic(), self.wall_deadline-time.time())
        if allowance <= 0:
            raise TimeoutError("Spawned replay deadline exhausted")
        self.process.wait(timeout=allowance)
        _require(self.process.returncode == 0, f"Spawned replay exited with status {self.process.returncode}")
        message = evidence.strict_json(self.process.stdout.read(65537))
        _require(type(message) is dict and set(message) == {"schema", "token", "report"}
                 and message["schema"] == SCHEMA and message["token"] == self.token,
                 "Replay completion differs from its live launch")
        path = self.output / "builtin-replay-verification.json"
        raw = path.read_bytes()
        _require(message["report"] == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                 "Spawned replay report changed after completion")
        _require(_files(self.folder, self.original_files) == self.original_files,
                 "Capture changed during spawned verification")
        # The caller independently validates this report, native report, history,
        # policy and all input bindings before it can mint a receipt.
        return evidence.strict_json(raw)

    def release(self):
        if self.process is not None:
            _release_owned_process(self.process)

    def stop(self, *, deadline_monotonic=None, deadline_wall=None):
        if self.closed:
            return
        try:
            if self.process is not None:
                close_owned_process(self.process, timeout=5,
                    deadline_monotonic=self.deadline if deadline_monotonic is None else deadline_monotonic,
                    deadline_wall=self.wall_deadline if deadline_wall is None else deadline_wall)
            self.closed = True
        finally:
            if self.process is not None:
                for stream in (self.process.stdin, self.process.stdout):
                    if stream is not None:
                        stream.close()
            if self.log is not None:
                self.log.close()


def _worker():
    from . import background_check, builtin_replay

    request = evidence.strict_json(sys.stdin.buffer.readline(4 * 1024 * 1024))
    _require(type(request) is dict and set(request) == {"schema", "token", "folder", "binary", "data",
             "original_files", "deadline_monotonic", "deadline_wall"} and request["schema"] == SCHEMA
             and type(request["token"]) is str and len(request["token"]) == 64,
             "Malformed spawned replay request")
    folder, binary, data = (Path(request[key]).resolve() for key in ("folder", "binary", "data"))
    _require(_files(folder, request["original_files"]) == request["original_files"],
             "Spawned replay capture differs from its launch")
    remaining = min(request["deadline_monotonic"]-time.monotonic(), request["deadline_wall"]-time.time())
    if remaining <= 0:
        raise TimeoutError("Spawned replay deadline exhausted before execution")
    # Four independent fresh interpreters supply parallelism. No post-Torch
    # fork or hidden per-job history thread is needed inside this worker.
    background_check.FORK = False
    output = folder / "training-verification"
    # The separately armed guardian covers the complete interpreter and its
    # native descendants, including history checks before native replay starts.
    report = builtin_replay.verify_episode(folder, binary=binary, data=data, output=output, timeout=remaining)
    _require(_files(folder, request["original_files"]) == request["original_files"],
             "Spawned replay capture changed during execution")
    raw = (output / "builtin-replay-verification.json").read_bytes()
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _require(canonical(evidence.strict_json(raw)) == canonical(report),
             "Persisted spawned replay report differs from its executed verification")
    print(json.dumps(dict(schema=SCHEMA, token=request["token"],
                         report={"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker",))
    parser.parse_args()
    _worker()


if __name__ == "__main__":
    main()
