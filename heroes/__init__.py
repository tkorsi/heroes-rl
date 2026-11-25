"""Heroes III self-play training.

Some modules may be compiled with Cython next to their sources (``tools/build_compiled.py``); Python then imports
the compiled module. A compiled module whose source changed after it was built would run the old code, so it is
refused here: each one is built with the SHA-256 of its source beside it.
"""
from __future__ import annotations


def _check_compiled():
    import hashlib
    from pathlib import Path

    package = Path(__file__).parent
    for marker in package.rglob("*.srchash"):
        source = marker.with_suffix(".py")
        built = list(marker.parent.glob(f"{marker.stem}.cpython-*.so"))
        if not built:
            continue
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != marker.read_text().strip():
            raise ImportError(f"{built[0]} was compiled from another version of {source.name}; "
                              "run tools/build_compiled.py again, or remove the compiled modules with --clean")


_check_compiled()
