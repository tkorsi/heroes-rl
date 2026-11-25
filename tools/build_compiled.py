"""Compile the actor's hot modules with Cython, next to their sources:  python tools/build_compiled.py [--clean]

The compiled modules are the same Python, run without the interpreter's dispatch; Python prefers an extension
module to the source beside it. Each one is built with the SHA-256 of its source written to
``<module>.srchash``, and importing ``heroes`` refuses a compiled module whose source has changed since
(``heroes/__init__.py``): rebuild or ``--clean``. Needs Cython 3 importable (for example from a private folder on
``PYTHONPATH``; it is not a runtime dependency) and a C compiler. ``--clean`` removes every compiled module.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("heroes/game/contract.py", "heroes/game/guard.py", "heroes/game/window_repeats.py", "heroes/model/policy.py",
           "heroes/model/fast.py", "heroes/engine/protocol.py", "heroes/engine/env.py", "heroes/learner/recipes.py",
           "heroes/actor.py")


def compiled(module):
    folder = ROOT / Path(module).parent
    stem = Path(module).stem
    return sorted(folder.glob(f"{stem}.cpython-*.so")) + sorted(folder.glob(f"{stem}.*.pyd"))


def clean():
    for module in MODULES:
        for path in compiled(module):
            path.unlink()
        for suffix in (".c", ".srchash"):
            path = ROOT / Path(module).with_suffix(suffix)
            if path.exists():
                path.unlink()
    shutil.rmtree(ROOT / "build" / "compiled", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true")
    arguments = parser.parse_args()
    clean()
    if arguments.clean:
        return
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    os.chdir(ROOT)
    extensions = [Extension(module[:-3].replace("/", "."), [module],
                            extra_compile_args=["-O2", "-Wno-unreachable-code", "-Wno-unused-function"])
                  for module in MODULES]
    setup(script_args=["build_ext", "--inplace", "-q", "--build-temp", "build/compiled"],
          ext_modules=cythonize(extensions, quiet=True,
                                compiler_directives={"language_level": 3, "binding": True, "embedsignature": False}))
    for module in MODULES:
        source = ROOT / module
        (ROOT / Path(module).with_suffix(".srchash")).write_text(hashlib.sha256(source.read_bytes()).hexdigest() + "\n")
        c_file = ROOT / Path(module).with_suffix(".c")
        if c_file.exists():
            c_file.unlink()
    print(f"compiled {len(MODULES)} modules in {ROOT}", file=sys.stderr)


if __name__ == "__main__":
    main()
