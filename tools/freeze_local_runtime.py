#!/usr/bin/env python3
"""Freeze an already-built macOS runtime into a new durable directory.

This copies and signs binaries/resources, checks canonical dependency agreement,
and runs client --help to verify actual library loading. It never builds, starts
a game, trains, overwrites an existing runtime, or copies copyrighted game data.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("vcmiclient", "vcmiserver", "vcmireplay", "libvcmi.dylib", *(
    f"AI/lib{name}.dylib" for name in ("BattleAI", "EmptyAI", "ExplorerRL", "Nullkiller", "StupidAI")))
SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(arguments, **kwargs):
    result = subprocess.run([str(arg) for arg in arguments], capture_output=True, text=True,
                            timeout=kwargs.pop("timeout", 120), **kwargs)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed ({result.returncode}): {result.stderr[-3000:]}")
    return result.stdout


def paths(build_bin, destination, game_map, root=ROOT):
    donor, target, game_map = (Path(item).expanduser().absolute() for item in (build_bin, destination, game_map))
    if target.exists() or target.is_symlink():
        raise ValueError("Destination must be a new directory; existing attempts are preserved")
    donor, target, game_map = donor.resolve(), target.resolve(), game_map.resolve()
    if "/.codex/worktrees/" in str(target) or target.is_relative_to(root.resolve()) or target.is_relative_to(donor):
        raise ValueError("Destination must be durable and outside the project/build/worktrees")
    if not target.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    if not donor.is_dir() or not game_map.is_file() or game_map.suffix.lower() not in (".vmap", ".h3m"):
        raise ValueError("A complete build directory and existing real map are required")
    for name in ARTIFACTS:
        if not (donor / name).is_file():
            raise ValueError(f"Missing required runtime artifact: {name}")
    for name in ("config", "Mods"):
        if not (donor / name).is_dir():
            raise ValueError(f"Missing runtime resources: {name}")
    return donor, target, game_map


def rpaths(load_commands):
    return re.findall(r"\bcmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset \d+\)", load_commands)


def libraries(path):
    # otool -L lists a dylib's own install ID before its dependencies. That
    # ID is not a load request (AI IDs need not resolve from their own RPATH).
    own_ids = set(line.strip() for line in run(["otool", "-D", path]).splitlines()[1:])
    return [name for line in run(["otool", "-L", path]).splitlines()[1:]
            if (name := line.strip().split(" (", 1)[0]) not in own_ids]


def source_inputs():
    names = {"dependencies/vcmi.json", "tools/prepare_vcmi.py", "tools/freeze_local_runtime.py"}
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / "engine").rglob("*") if path.is_file())
    return {name: identity(ROOT / name) for name in sorted(names)}


def expand_token(value, loader, executable):
    for token, base in (("@loader_path", loader.parent), ("@executable_path", executable)):
        if value == token or value.startswith(token + "/"):
            return base / value[len(token):].lstrip("/")
    return Path(value) if value.startswith("/") else None


def dependencies(binary_directory):
    """Resolve every non-system load dependency, including transitive dylibs."""
    queue = [(binary_directory / name, ()) for name in ARTIFACTS]
    visited, external, system = set(), {}, set()
    while queue:
        path, inherited = queue.pop()
        path = path.resolve()
        if path in visited:
            continue
        visited.add(path)
        search = tuple(filter(None, (expand_token(value, path, binary_directory)
                                     for value in rpaths(run(["otool", "-l", path]))))) + inherited
        for name in libraries(path):
            if name.startswith(SYSTEM_PREFIXES):
                system.add(name)
                continue  # dyld shared-cache system files need not exist on disk
            if name.startswith("@rpath/"):
                candidates = [base / name[len("@rpath/"):] for base in search]
                resolved = next((item for item in candidates if item.is_file()), None)
            else:
                resolved = expand_token(name, path, binary_directory)
            if resolved is None or not resolved.is_file():
                raise ValueError(f"Unresolved runtime dependency {name} from {path}")
            if "/.codex/worktrees/" in str(resolved) or "/.codex/worktrees/" in str(resolved.resolve()):
                raise ValueError(f"Dependency still points into a managed worktree: {name}")
            if not resolved.resolve().is_relative_to(binary_directory):
                external[str(resolved)] = {"path": str(resolved), "resolved": str(resolved.resolve()), **identity(resolved)}
            queue.append((resolved, search))
    return list(external.values()), sorted(system)


def freeze(args):
    if sys.platform != "darwin":
        raise ValueError("This freezer validates macOS Mach-O runtimes only")
    donor, target, game_map = paths(args.build_bin, args.destination, args.game_map)
    cache = donor.parent / "CMakeCache.txt"
    home = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(), re.MULTILINE)
    if not home or Path(home.group(1)).resolve() != (ROOT / "vcmi").resolve():
        raise ValueError("Build must originate in this project's canonically prepared vcmi checkout")
    check = run([sys.executable, ROOT / "tools/prepare_vcmi.py", "--check"], timeout=300)
    sources = source_inputs()
    source_commit = run(["git", "-C", ROOT, "rev-parse", "HEAD"]).strip()
    dirty = bool(run(["git", "-C", ROOT, "status", "--porcelain"]).strip())
    donor_identities = {name: identity(donor / name) for name in ARTIFACTS}
    target.mkdir(exist_ok=False)
    binary_directory = target / "bin"
    (binary_directory / "AI").mkdir(parents=True)
    (target / "maps").mkdir()
    write_json(target / "freeze-start.json", {"started_utc": utc(), "donor_bin": str(donor),
               "source_commit": source_commit, "source_dirty": dirty, "source_inputs": sources,
               "training_started": False, "gameplay_started": False})
    try:
        (target / "dependency-check.stdout").write_text(check)
        artifacts = []
        for name in ARTIFACTS:
            source, destination = donor / name, binary_directory / name
            shutil.copy2(source, destination)
            replacements = []
            replacement = "@loader_path/.." if name.startswith("AI/") else "@loader_path"
            for old in rpaths(run(["otool", "-l", destination])):
                if old == str(donor):
                    run(["install_name_tool", "-rpath", old, replacement, destination])
                    replacements.append({"old": old, "new": replacement})
            for dependency in libraries(destination):
                if dependency.startswith(str(donor) + "/"):
                    rewritten = "@rpath/" + str(Path(dependency).relative_to(donor))
                    run(["install_name_tool", "-change", dependency, rewritten, destination])
            for own_id in run(["otool", "-D", destination]).splitlines()[1:]:
                if own_id.strip().startswith(str(donor) + "/"):
                    run(["install_name_tool", "-id", "@rpath/" + destination.name, destination])
            run(["codesign", "--force", "--sign", "-", destination])
            run(["codesign", "--verify", "--strict", destination])
            commands = run(["otool", "-l", destination]) + run(["otool", "-L", destination])
            if str(donor) in commands or "/.codex/worktrees/" in commands:
                raise ValueError(f"Non-durable load reference remains in {name}")
            artifacts.append({"path": name, "donor_sha256": donor_identities[name]["sha256"],
                              "frozen_sha256": identity(destination)["sha256"], "bytes": destination.stat().st_size,
                              "rpath_replacements": replacements, "ad_hoc_signature_verified": True})
        resources = []
        for name in ("config", "Mods"):
            shutil.copytree(donor / name, binary_directory / name, symlinks=False)
            for path in sorted((binary_directory / name).rglob("*")):
                if path.is_symlink():
                    raise ValueError("Frozen resource symlink was not dereferenced")
                if path.is_file():
                    resources.append({"path": str(path.relative_to(binary_directory)), **identity(path)})
        frozen_map = target / "maps" / game_map.name
        shutil.copy2(game_map, frozen_map)
        map_manifest = game_map.with_suffix(".manifest.json")
        if map_manifest.is_file():
            shutil.copy2(map_manifest, target / "maps" / map_manifest.name)
        external, system = dependencies(binary_directory)
        sys.path[:0] = [str(ROOT), str(ROOT / "experiments/headless")]
        from experiments.headless.env import VcmiLauncher
        from experiments.headless.contextual_evidence import runtime_identity
        launcher = VcmiLauncher(binary_directory / "vcmiclient", args.game_data, frozen_map)
        launcher.preflight()
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("EXPLORER_RL_", "VCMI_", "DYLD_", "XDG_"))}
        environment["DYLD_PRINT_LIBRARIES"] = "1"
        result = subprocess.run([str(binary_directory / "vcmiclient"), "--help"], cwd=binary_directory,
                                env=environment, capture_output=True, text=True, timeout=30)
        (target / "client-help.stdout").write_text(result.stdout)
        (target / "client-help.stderr").write_text(result.stderr)
        if (result.returncode != 0 or str(binary_directory / "libvcmi.dylib") not in result.stderr
                or str(donor) in result.stderr or "/.codex/worktrees/" in result.stderr):
            raise ValueError("Client help failed or did not load the frozen libvcmi exclusively")
        if sources != source_inputs() or donor_identities != {name: identity(donor / name) for name in ARTIFACTS}:
            raise ValueError("Source or donor binaries changed during freezing")
        if any(identity(item["path"]) != {key: item[key] for key in ("sha256", "bytes")} for item in external):
            raise ValueError("External native dependencies changed during preflight")
        receipt = {"schema": "heroes-local-runtime-freeze-v2", "created_at_utc": utc(),
                   "source_commit": source_commit, "source_dirty": dirty, "source_directory": str(ROOT),
                   "source_inputs": sources, "source_recipe_check": True,
                   "dependency_head": run(["git", "-C", ROOT / "vcmi", "rev-parse", "HEAD"]).strip(),
                   "dependency_check": identity(target / "dependency-check.stdout"), "build_cache": identity(cache),
                   "donor_bin": str(donor), "frozen_bin": str(binary_directory), "artifacts": artifacts,
                   "resources": resources, "external_dependencies": external, "system_dependencies": system,
                   "external_dependencies_bundled": False, "game_data_copied": False,
                   "donor_load_references_removed": True, "resource_symlinks_dereferenced": True,
                   "old_runtime_modified": False, "map": {"path": str(frozen_map), **identity(frozen_map)},
                   "map_manifest": ({"path": str(target / "maps" / map_manifest.name),
                                     **identity(target / "maps" / map_manifest.name)} if map_manifest.is_file() else None),
                   "training_started": False, "gameplay_started": False}
        preflight = {"schema": "heroes-local-runtime-preflight-v2", "launcher_preflight_passed": True,
                     "native": launcher.identity(), "runtime": runtime_identity(binary_directory),
                     "map": receipt["map"], "client_help_exit_code": result.returncode,
                     "actual_client_libvcmi_load": [line for line in result.stderr.splitlines() if "libvcmi" in line],
                     "donor_load_detected": False, "gameplay_started": False, "training_started": False}
        write_json(target / "freeze-receipt.json", receipt)
        write_json(target / "preflight.json", preflight)
        return {"runtime": str(target), "source": source_commit, "source_dirty": dirty,
                "artifacts": len(artifacts), "resources": len(resources), "external_dependencies": len(external),
                "freeze_sha256": identity(target / "freeze-receipt.json")["sha256"],
                "preflight_sha256": identity(target / "preflight.json")["sha256"], "gameplay_started": False}
    except BaseException as error:
        write_json(target / "freeze-failure.json", {"failed_utc": utc(), "error": f"{type(error).__name__}: {error}",
                   "partial_destination_preserved": True, "gameplay_started": False, "training_started": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bin", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--map", dest="game_map", type=Path, required=True)
    parser.add_argument("--game-data", type=Path, default=Path.home() / "Projects/Heroes-artifacts/game-data")
    print(json.dumps(freeze(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
