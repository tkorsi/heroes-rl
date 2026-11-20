#!/usr/bin/env python3
"""Copy a built engine out of ``vcmi/build/bin`` into a folder that stands on its own.

    python3 tools/freeze_runtime.py ~/Projects/Heroes-artifacts/runtime/NAME

The build's programs find ``libvcmi.dylib`` through the absolute path of the
build folder. The copies look beside themselves instead, so the runtime keeps
working when the checkout is rebuilt, moved or removed. ``config`` and ``Mods``
are copied as real folders. Nothing is started; the folder must not exist yet.

The runtime is put together beside its final place and renamed into it after
the last file is ready. A run that fails leaves nothing behind, so the same
command works again once the build is repaired, and no half-copied runtime
ever stands where training would look for one.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ("vcmiclient", "vcmiserver", "vcmireplay", "libvcmi.dylib")
PLUGINS = ("BattleAI", "EmptyAI", "ExplorerRL", "Nullkiller", "StupidAI")


def run(*command):
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def freeze(build, runtime):
    """Copy the build in ``build`` to the new folder ``runtime``; returns its ``vcmiclient``."""
    build, runtime = Path(build).resolve(), Path(runtime)
    if runtime.exists() or runtime.is_symlink():
        raise SystemExit(f"{runtime} exists already; a runtime is never changed. Choose a new name.")
    work = runtime.with_name(f"{runtime.name}.partial-{os.getpid()}")
    target = work / "bin"
    work.mkdir(parents=True)
    try:
        (target / "AI").mkdir(parents=True)
        for name in (*PROGRAMS, *(f"AI/lib{plugin}.dylib" for plugin in PLUGINS)):
            copy = target / name
            shutil.copy2(build / name, copy)
            if str(build) in run("otool", "-l", str(copy)):
                run("install_name_tool", "-rpath", str(build),
                    "@loader_path/.." if name.startswith("AI/") else "@loader_path", str(copy))
            run("codesign", "--force", "--sign", "-", str(copy))  # the edit invalidated the build's signature
            if str(build) in run("otool", "-l", str(copy)) + run("otool", "-L", str(copy)):
                raise SystemExit(f"{name} still refers to the build folder")
        for name in ("config", "Mods"):
            shutil.copytree(build / name, target / name, symlinks=False)
        # One step, and the last one. It does not replace a runtime that
        # appeared meanwhile: renaming onto a folder with files in it fails.
        work.rename(runtime)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # nothing is left to remove once the rename succeeded
    return runtime / "bin" / "vcmiclient"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--build", type=Path, default=ROOT / "vcmi" / "build" / "bin", help="the built bin folder")
    arguments = parser.parse_args()
    print(freeze(arguments.build, arguments.target))


if __name__ == "__main__":
    main()
