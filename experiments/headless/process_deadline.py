"""POSIX process-group deadlines, independent of the collecting Python process.

The wrapper starts a private session, forks its guardian, waits until it is
armed, then execs the requested program without changing the leader PID. The
guardian is a child of that program. It kills its own private process group
when its parent exits, at the earlier absolute clock deadline, or when the
launcher's exclusive ownership pipe closes. Processes
that deliberately leave that group require their own guardian.

Evidence is append-only JSONL. ``exec_attempt`` does not claim a successful
exec, and ``deadline_expired`` records the impending signal, not an outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "heroes-process-deadline-v1"
POLL_SECONDS = 0.05


def _deadline(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive timestamp")
    return float(value)


def _remaining(monotonic, wall):
    # Some platforms pause the monotonic clock during suspend. Wall time also
    # bounds that interval; neither a backward wall jump nor suspend extends it.
    return min(monotonic - time.monotonic(), wall - time.time())


def _event(path, event, **fields):
    record = {"schema": SCHEMA, "event": event, "monotonic": time.monotonic(),
              "wall_time": time.time(), "writer_pid": os.getpid(), **fields}
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        if os.write(fd, data) != len(data):
            raise OSError("Incomplete supervision evidence write")
    finally:
        os.close(fd)


def guarded_popen(command, *, deadline_monotonic, deadline_wall, evidence_path, **kwargs):
    """Start a deadline-supervised command; never restore RNG or launch a shell.

    The earlier same-host monotonic/wall timestamp governs termination: suspend
    cannot extend wall time, and a backward wall-clock adjustment cannot extend
    monotonic time. A forward wall jump can safely stop the process early.
    The returned Popen PID becomes the requested program's PID and private PGID.
    The evidence path must not exist. Each launch, including a failed exec, keeps
    its separate record. The guardian is armed before the program can execute.
    """
    if os.name != "posix" or not hasattr(os, "fork"):
        raise RuntimeError("Independent process deadlines require POSIX fork")
    deadline_monotonic = _deadline(deadline_monotonic, "deadline_monotonic")
    deadline_wall = _deadline(deadline_wall, "deadline_wall")
    if _remaining(deadline_monotonic, deadline_wall) <= 0:
        raise TimeoutError("Process deadline expired before launch")
    if not isinstance(command, (list, tuple)) or not command or any(
            not isinstance(arg, (str, os.PathLike)) for arg in command):
        raise ValueError("Use a nonempty command argument sequence")
    command = [os.fspath(arg) for arg in command]
    if not command[0] or any("\0" in arg for arg in command):
        raise ValueError("Invalid command argument")
    if kwargs.pop("start_new_session", True) is not True:
        raise ValueError("A guarded process must have a private session")
    if kwargs.pop("shell", False) is not False or any(
            name in kwargs for name in ("executable", "preexec_fn", "process_group", "pass_fds")):
        raise ValueError("Do not override the guarded exec or private process group")
    if kwargs.pop("close_fds", True) is not True:
        raise ValueError("The ownership pipe must not leak to other processes")
    path = Path(evidence_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    _event(path, "launch_requested", command=command, deadline_monotonic=deadline_monotonic,
           deadline_wall=deadline_wall)
    owner_read, owner_write = os.pipe()
    wrapper = [sys.executable, "-I", str(Path(__file__).resolve()), "--guarded-exec",
               "--deadline-monotonic", repr(deadline_monotonic), "--deadline-wall", repr(deadline_wall),
               "--evidence", str(path), "--owner-pipe", str(owner_read), "--", *command]
    try:
        process = subprocess.Popen(wrapper, start_new_session=True, close_fds=True,
                                   pass_fds=(owner_read,), **kwargs)
        process._deadline_owner_fd = owner_write
        process._deadline_evidence_path = path
        process._deadline_guarded = True
        return process
    except BaseException as error:
        os.close(owner_write)
        _event(path, "launch_failed", error_type=type(error).__name__)
        raise
    finally:
        os.close(owner_read)


def _guardian(leader, deadline, wall, evidence, ready_fd, owner_fd):
    # Do not hold caller pipes or the native log open after the engine exits.
    for fd in (0, 1, 2):
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        if os.getppid() != leader or os.getpgrp() != leader or os.getsid(0) != leader:
            return 125
        _event(evidence, "guardian_armed", leader_pid=leader, process_group=leader,
               guardian_pid=os.getpid(), deadline_monotonic=deadline, deadline_wall=wall)
        os.write(ready_fd, b"1")
        os.close(ready_fd)
        os.set_blocking(owner_fd, False)
        while os.getppid() == leader:
            try:
                owner_gone = os.read(owner_fd, 1) == b""
            except BlockingIOError:
                owner_gone = False
            remaining = _remaining(deadline, wall)
            if remaining <= 0 or owner_gone:
                # Parentage prevents a reused leader PID from being targeted.
                if os.getppid() == leader and os.getpgrp() == leader and os.getsid(0) == leader:
                    try:
                        _event(evidence, "launcher_exited" if owner_gone else "deadline_expired", leader_pid=leader,
                               process_group=leader, signal="SIGKILL")
                    finally:
                        os.killpg(leader, signal.SIGKILL)
                return 124
            time.sleep(min(POLL_SECONDS, remaining))
        # The guardian itself still anchors this exact group/session, even
        # after its original leader exits. Stop any surviving children too;
        # this does not depend on the external launcher surviving to clean up.
        if os.getpgrp() == leader and os.getsid(0) == leader:
            try:
                _event(evidence, "parent_exited", leader_pid=leader, process_group=leader,
                       cleanup_signal="SIGKILL")
            finally:
                os.killpg(leader, signal.SIGKILL)
        return 0
    finally:
        try:
            os.close(ready_fd)
        except OSError:
            pass


def _guarded_exec(args):
    deadline = _deadline(args.deadline_monotonic, "deadline_monotonic")
    wall = _deadline(args.deadline_wall, "deadline_wall")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Missing guarded command")
    leader = os.getpid()
    if os.getpgrp() != leader or os.getsid(0) != leader:
        raise RuntimeError("Guarded wrapper must be the private session leader")
    _event(args.evidence, "wrapper_started", leader_pid=leader, process_group=leader,
           deadline_monotonic=deadline, deadline_wall=args.deadline_wall)
    if _remaining(deadline, wall) <= 0:
        _event(args.evidence, "expired_before_exec", leader_pid=leader, process_group=leader)
        return 124
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            result = _guardian(leader, deadline, wall, args.evidence, write_fd, args.owner_pipe)
        except BaseException:
            # A guardian that failed to arm closes the handshake pipe. A failure
            # after arming must not leave an unsupervised program running.
            if os.getppid() == leader and os.getpgrp() == leader:
                os.killpg(leader, signal.SIGKILL)
            result = 125
        os._exit(result)
    os.close(write_fd)
    os.close(args.owner_pipe)
    try:
        armed = os.read(read_fd, 1)
    finally:
        os.close(read_fd)
    if armed != b"1":
        _event(args.evidence, "guardian_failed", leader_pid=leader, guardian_pid=child)
        return 125
    if _remaining(deadline, wall) <= 0:
        _event(args.evidence, "expired_before_exec", leader_pid=leader, process_group=leader)
        return 124
    _event(args.evidence, "exec_attempt", leader_pid=leader, process_group=leader,
           guardian_pid=child, command=command)
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError as error:
        _event(args.evidence, "exec_failed", leader_pid=leader, errno=error.errno)
        return 127


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded-exec", action="store_true", required=True)
    parser.add_argument("--deadline-monotonic", type=float, required=True)
    parser.add_argument("--deadline-wall", type=float, required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--owner-pipe", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return _guarded_exec(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(_main())


# The standalone guardian imports only the standard library above.
from .env import VcmiLauncher  # noqa: E402


def group_live_members(group, *, timeout=1.0):
    """Inspect only a recorded private PGID; zombies are no longer executing."""
    if type(group) is not int or group <= 1 or group == os.getpgrp():
        raise ValueError("Expected a separate recorded process group")
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], text=True,
                            capture_output=True, check=True, timeout=timeout)
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and int(fields[1]) == group and not fields[2].startswith("Z"):
            members.append({"pid": int(fields[0]), "process_group": group, "state": fields[2]})
    return tuple(members)


def close_owned_process(process, *, timeout=2.0):
    """Close exclusive ownership, then await guardian cleanup and reap.

    Only the live guardian sends group signals. A historical PID/PGID is never
    a sufficient ownership proof for this external caller to send a signal.
    """
    if getattr(process, "_deadline_guarded", False) is not True:
        raise ValueError("Only processes created by guarded_popen are owned here")
    end = time.monotonic() + _deadline(timeout, "cleanup timeout")
    fd = getattr(process, "_deadline_owner_fd", None)
    if fd is not None:
        os.close(fd)
        process._deadline_owner_fd = None
    process.wait(timeout=max(0.01, end - time.monotonic()))
    while group_live_members(process.pid, timeout=max(0.01, end - time.monotonic())):
        if time.monotonic() >= end:
            raise TimeoutError("Owned process group did not stop within cleanup budget")
        time.sleep(min(POLL_SECONDS, max(0, end - time.monotonic())))


def cleanup_from_evidence(path, *, timeout=2.0):
    """Verify recorded groups stopped after a worker failure or normal close.

    Launcher pipe EOF stops an orphan even if this receipt was not yet written.
    This helper never signals a recorded numeric PID/PGID that may be reused.
    Call only after closing/waiting for the owning worker. An unexpectedly live
    group conservatively fails verification, without harming another process.
    """
    path = Path(path)
    if not path.exists():
        return ()
    groups = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("schema") != SCHEMA:
            raise ValueError("Unexpected supervision evidence schema")
        if row.get("event") == "wrapper_started":
            group = row["process_group"]
            if row["writer_pid"] != row["leader_pid"] or group != row["leader_pid"]:
                raise ValueError("Supervision evidence does not bind its own private leader")
            groups.add(group)
    end = time.monotonic() + _deadline(timeout, "cleanup timeout")
    while any(group_live_members(group, timeout=max(0.01, end - time.monotonic())) for group in groups):
        if time.monotonic() >= end:
            raise TimeoutError("Recorded process groups remain active")
        time.sleep(min(POLL_SECONDS, max(0, end - time.monotonic())))
    return tuple(sorted(groups))


class DeadlineVcmiLauncher(VcmiLauncher):
    """The ordinary launcher with a shared absolute deadline and native receipts."""

    def __init__(self, *args, deadline_monotonic, deadline_wall, evidence_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline_monotonic = _deadline(deadline_monotonic, "deadline_monotonic")
        self.deadline_wall = _deadline(deadline_wall, "deadline_wall")
        self.evidence_path = None if evidence_path is None else Path(evidence_path).resolve()
        self.launched_processes = []
        self.supervision_records = []

    def _spawn_process(self, command, launch_env, log):
        evidence = self.evidence_path or Path(launch_env["VCMI_EXPERIMENT_DIR"]).parent / "native-process-deadline.jsonl"
        process = guarded_popen(command, deadline_monotonic=self.deadline_monotonic,
            deadline_wall=self.deadline_wall, evidence_path=evidence, cwd=self.binary.parent,
            env=launch_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.launched_processes.append(process)
        self.supervision_records.append({"schema": SCHEMA, "leader_pid": process.pid,
            "process_group": process.pid, "deadline_monotonic": self.deadline_monotonic,
            "deadline_wall": self.deadline_wall, "evidence_path": str(evidence.resolve()),
            "wrapper_command": list(process.args), "guardian_arming": "required_before_native_exec"})
        return process

    def launch(self, directory, seed, environment, log):
        process, metadata = super().launch(directory, seed, environment, log)
        return process, {**metadata, "supervision": dict(self.supervision_records[-1])}

    def assert_closed(self):
        for process in self.launched_processes:
            process.poll()
            if group_live_members(process.pid):
                raise RuntimeError(f"Owned native process group is still running: {process.pid}")

    def close_owned_groups(self, timeout=2.0):
        """Close ownership and verify only this launcher's groups stopped."""
        timeout = _deadline(timeout, "cleanup timeout")
        end = time.monotonic() + timeout
        for process in self.launched_processes:
            close_owned_process(process, timeout=max(0.01, end - time.monotonic()))

    terminate_owned_groups = close_owned_groups
