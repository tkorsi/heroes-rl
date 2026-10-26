"""Run one independent verification step in a forked child while the caller does its own.

``BackgroundCheck(function, *args, **kwargs).start()`` forks; the child keeps only its result pipe
and the standard streams (no ownership pipe or other descriptor of the caller), stays in the
caller's process group (every deadline guard that owns the caller also owns it), calls the
function and returns its value, or the exception it raised, pickled. The step reads only the persisted files
its arguments name. ``result`` returns that value, or raises that exception, exactly as an
in-process call would; ``stop`` ends a step that is still running (SIGTERM unwinds the child's
own subprocesses, then SIGKILL) and reaps it. Callers keep every check they apply to the result.
"""
from __future__ import annotations

import os
import pickle
import select
import signal
import time

_CHUNK = 1 << 20


def _terminate(signum, frame):
    raise SystemExit(143)


def _child(write, function, args, kwargs, mask):
    keep = sorted({0, 1, 2, write})
    for low, high in zip(keep, [*keep[1:], os.sysconf("SC_OPEN_MAX")]):
        os.closerange(low + 1, high)
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    # SIGTERM unwinds through ``finally`` blocks and subprocess cleanup, as an exception would.
    signal.signal(signal.SIGTERM, _terminate)
    signal.pthread_sigmask(signal.SIG_SETMASK, mask)  # a stop sent before this point acts now
    try:
        reply = ("ok", function(*args, **kwargs))
    except Exception as error:
        reply = ("error", error)
    try:
        data = pickle.dumps(reply, protocol=5)
    except Exception as error:  # an unpicklable value or exception still fails the step
        data = pickle.dumps(("error", RuntimeError(f"Unpicklable background result: {type(error).__name__}: {error}")))
    view = memoryview(data)
    while view:
        view = view[os.write(write, view[:_CHUNK]):]


# Deferred while a child is forked and recorded, and while a stop kills and reaps it, so that
# a caller's wall-limit alarm or interrupt can neither lose a child nor leave one running.
_DEFERRED = (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)


class BackgroundCheck:
    """A step to run in a forked child: ``start`` it inside the ``try`` whose ``finally`` calls ``stop``."""

    def __init__(self, function, *args, **kwargs):
        self.call = function, args, kwargs
        self.pid, self.read, self.reaped = None, None, True

    def start(self):
        if self.pid is not None:
            raise RuntimeError("A background check starts once")
        function, args, kwargs = self.call
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _DEFERRED)
        try:
            read, write = os.pipe()
            try:
                pid = os.fork()
            except BaseException:
                os.close(read)
                os.close(write)
                raise
            if pid == 0:  # child: never returns into the caller's code
                code = 1
                try:
                    os.close(read)
                    _child(write, function, args, kwargs, mask)
                    code = 0
                finally:
                    os._exit(code)
            os.close(write)
            self.pid, self.read, self.reaped = pid, read, False
        finally:
            # A deferred alarm or interrupt is raised here, inside the caller's ``try``.
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        return self

    def result(self, *, timeout):
        if self.pid is None:
            raise RuntimeError("A background check returns a result only after it started")
        deadline, chunks = time.monotonic() + timeout, []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.read], [], [], remaining)[0]:
                    raise TimeoutError("Background verification step exceeded its time budget")
                chunk = os.read(self.read, _CHUNK)
                if not chunk:
                    break
                chunks.append(chunk)
            _, status = os.waitpid(self.pid, 0)
            self.reaped = True
            if os.waitstatus_to_exitcode(status) != 0 or not chunks:
                raise RuntimeError(f"Background verification step ended with status {os.waitstatus_to_exitcode(status)}")
            kind, value = pickle.loads(b"".join(chunks))
        finally:
            self.stop()
        if kind != "ok":
            raise value
        return value

    def stop(self):
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _DEFERRED)
        try:
            if not self.reaped:
                os.kill(self.pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while os.waitpid(self.pid, os.WNOHANG)[0] != self.pid:
                    if time.monotonic() > deadline or signal.sigpending() & set(_DEFERRED):
                        # Out of time, or the caller is being interrupted: end the child now.
                        os.kill(self.pid, signal.SIGKILL)
                        os.waitpid(self.pid, 0)
                        break
                    time.sleep(0.005)
                self.reaped = True
            if self.read is not None:
                os.close(self.read)
                self.read = None
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
