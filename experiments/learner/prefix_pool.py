"""Exact prefix-reconstruction frame encodings from forked helper processes (opt-in).

Prefix reconstruction (``ContextualActorCritic.prefix_states``) encodes every earlier frame
under the current parameters, then applies the recurrent transitions in chronological order.
A frame's encoding (``_frame_features`` under no_grad, with a fresh tree cache) is a pure
function of the frame, the model configuration and the exact parameter bits; reconstruction
already reuses it for frames with an identical exact key. Helpers forked after the update's
histories are encoded compute disjoint shares of the distinct encodings with the parameter
bits and module modes the parent sends with every request, the same single-thread CPU
kernels and fresh request-local memos, and return them bit for bit. The parent recomputes
the first encoding of every helper's share and requires identical bits, and still checks
every frame and applies every transition itself. Helpers hold no file descriptors but their
own pipe and the standard streams, never touch the optimizer, and end with the update.
"""
from __future__ import annotations

import gc
from multiprocessing.connection import Connection
import os
import pickle
import signal
import socket
import time

import torch

from .contextual_model import _LEAF_MEMO, _bits, frame_keys_until

MAX_HELPERS = 15


def _empty_memo():
    return {"text": {}, "number": {}, "node": {}}


def _encode(model, frames, seqs):
    memo = _empty_memo()
    with torch.no_grad():
        return [model._frame_features(frames[seq], {_LEAF_MEMO: memo})[0] for seq in seqs]


def _same_bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(_bits(left.contiguous()), _bits(right.contiguous())))


def _serve(connection, model, inputs):
    # Keep only this socket and the standard streams: no ownership pipe, lock or file leaks.
    keep = sorted({0, 1, 2, connection.fileno()})
    for low, high in zip(keep, [*keep[1:], os.sysconf("SC_OPEN_MAX")]):
        os.closerange(low + 1, high)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, 0)
    torch.set_num_threads(1)
    tensors = [*model.parameters(), *model.buffers()]
    modules = list(model.modules())
    while True:
        request = pickle.loads(connection.recv_bytes())
        if request is None:
            return
        try:
            modes, values, index, seqs = request
            with torch.no_grad():
                for target, value in zip(tensors, values, strict=True):
                    if target.dtype != value.dtype or target.shape != value.shape:
                        raise ValueError("Helper parameter layout differs")
                    target.copy_(value)
            for module, mode in zip(modules, modes, strict=True):
                module.training = mode
            reply = ("ok", torch.stack(_encode(model, inputs[index], seqs)))
        except Exception as error:  # reported to the parent, which fails the update
            reply = ("error", f"{type(error).__name__}: {error}")
        connection.send_bytes(pickle.dumps(reply))


class PrefixPool:
    """Helpers for one update: exact frame encodings under the parent's current parameters."""

    def __init__(self, model, inputs, helpers):
        if type(helpers) is not int or not 1 <= helpers <= MAX_HELPERS:
            raise ValueError("Prefix helpers must be a small positive integer")
        if not model._can_cache_prefix():
            raise ValueError("Prefix helpers need the exact cached reconstruction path")
        self.model, self.inputs = model, inputs
        self.keys = [[] for _ in inputs]
        self.connections, self.processes = [], []
        # Helpers never collect the parent's objects, so they copy fewer pages.
        gc.freeze()
        try:
            for _ in range(helpers):
                mine, theirs = socket.socketpair()
                pid = os.fork()
                if pid == 0:  # helper: never returns into the parent's code
                    code = 1
                    try:
                        mine.close()
                        _serve(Connection(theirs.detach()), model, inputs)
                        code = 0
                    finally:
                        os._exit(code)
                theirs.close()
                self.connections.append(Connection(mine.detach()))
                self.processes.append(pid)
        except BaseException:
            self.close()
            raise
        finally:
            gc.unfreeze()

    def vectors(self, index, stop, keys=None):
        """{seq: encoding} for every frame before ``stop`` that has semantics.

        ``keys`` is this history's ``frame_keys_until`` list, if the caller keeps one.
        """
        model, frames = self.model, self.inputs[index]
        if not model._can_cache_prefix():
            raise ValueError("Prefix helpers need the exact cached reconstruction path")
        keys = frame_keys_until(self.keys[index] if keys is None else keys, frames, stop)
        first = {}
        for seq in range(stop):
            if keys[seq] is not None:
                first.setdefault(keys[seq], seq)
        distinct = list(first.values())
        # Contiguous shares of about equal key length (a proxy for tree size).
        total, parts = sum(len(keys[seq]) for seq in distinct), len(self.connections) + 1
        shares, share, running = [], [], 0
        for seq in distinct:
            share.append(seq)
            running += len(keys[seq])
            if running * parts >= total * (len(shares) + 1) and len(shares) < parts - 1:
                shares.append(share)
                share = []
        shares.append(share)
        shares += [[] for _ in range(parts - len(shares))]
        values = [value.detach() for value in (*model.parameters(), *model.buffers())]
        modes = [module.training for module in model.modules()]
        pending = []
        for connection, seqs in zip(self.connections, shares[1:]):
            if seqs:
                connection.send_bytes(pickle.dumps((modes, values, index, seqs)))
                pending.append((connection, seqs))
        encoded = dict(zip(shares[0], _encode(model, frames, shares[0])))
        for connection, seqs in pending:
            status, payload = pickle.loads(connection.recv_bytes())
            if status != "ok" or type(payload) is not torch.Tensor or payload.shape[0] != len(seqs):
                raise RuntimeError(f"Prefix helper failed: {payload if status != 'ok' else 'malformed reply'}")
            # Separate allocations, as the serial reconstruction's own encodings are.
            rows = [row.clone() for row in payload]
            if not _same_bits(rows[0], _encode(model, frames, seqs[:1])[0]):
                raise RuntimeError("Prefix helper encoding differs from the parent's")
            encoded.update(zip(seqs, rows))
        return {seq: encoded[first[keys[seq]]] for seq in range(stop) if keys[seq] is not None}

    def close(self):
        """Stop and reap every helper; one that has not exited within five seconds is killed."""
        for connection in self.connections:
            try:
                connection.send_bytes(pickle.dumps(None))
            except OSError:
                pass
        deadline = time.monotonic() + 5
        for pid in self.processes:
            while os.waitpid(pid, os.WNOHANG)[0] != pid:
                if time.monotonic() > deadline:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    break
                time.sleep(0.005)
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []
