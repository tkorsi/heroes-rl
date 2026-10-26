"""SHA-256 of files, each unchanged file hashed at most once inside one explicit check scope.

Outside a scope every call hashes the file's current bytes, exactly as before. Inside ``scope()``
(one identity check) a digest is reused only for the same resolved path whose device, inode,
size, modification and status-change times all equal those observed before and after the
hashing that produced it; any write, replacement, truncation or metadata change gives a new key
and a fresh hash. Each new scope starts empty, so every check reads every file again at least
once. ``prefetch`` hashes paths concurrently (hashlib releases the interpreter lock) into the
current scope; it never decides anything, it only fills entries that the check would compute.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
import threading

_LOCAL = threading.local()
_WORKERS = 4
_PARALLEL_BYTES = 1 << 20


def _stat_key(path):
    real = os.path.realpath(path)
    st = os.stat(real)
    return real, (real, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _hash(path):
    """(hexdigest, bytes hashed) of the file's current contents."""
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _memo():
    return getattr(_LOCAL, "memo", None)


def sha256_file(path):
    """(hexdigest, bytes hashed) of ``path``; reused only within a scope for an unchanged file."""
    memo = _memo()
    if memo is None:
        return _hash(path)
    real, key = _stat_key(path)
    with memo["lock"]:
        hit = memo["entries"].get(key)
    if hit is not None:
        return hit
    result = _hash(real)
    if _stat_key(real)[1] == key and result[1] == key[3]:
        with memo["lock"]:
            memo["entries"][key] = result
    return result


@contextmanager
def scope():
    """Reuse digests of unchanged files until the outermost scope ends (nested scopes share it)."""
    if _memo() is not None:
        yield
        return
    _LOCAL.memo = {"entries": {}, "lock": threading.Lock()}
    try:
        yield
    finally:
        _LOCAL.memo = None


def prefetch(paths):
    """Hash existing regular files concurrently into the current scope; errors are left to the check."""
    memo = _memo()
    if memo is None:
        return
    def fill(path):
        _LOCAL.memo = memo
        try:
            sha256_file(path)
        except OSError:
            pass  # the check itself reads the path and reports any failure
        finally:
            _LOCAL.memo = None
    # Only large files gain from threads; small ones are dominated by interpreter work.
    files = sorted({str(p) for p in paths if os.path.isfile(p) and not os.path.islink(p)
                    and os.path.getsize(p) >= _PARALLEL_BYTES}, key=lambda p: -os.path.getsize(p))
    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(fill, files))
