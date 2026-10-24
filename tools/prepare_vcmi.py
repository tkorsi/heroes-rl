#!/usr/bin/env python3
"""Recreate the ignored VCMI dependency from pinned upstream plus root-owned code.

Run without arguments to prepare it, or --check to verify an existing checkout.
Existing unexpected changes are reported and preserved; this tool never resets them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class PreparationError(RuntimeError):
    pass


def git(repo: Path, *arguments: str, env: dict | None = None, payload: bytes | None = None) -> bytes:
    environment = {**(env if env is not None else os.environ), "GIT_OPTIONAL_LOCKS": "0"}
    result = subprocess.run(["git", "-C", str(repo), *arguments], input=payload,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    if result.returncode:
        raise PreparationError(f"git {' '.join(arguments[:3])}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def within(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PreparationError(f"Expected a relative path inside the project: {relative}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise PreparationError(f"Path escapes its project directory: {relative}")
    return result


def load_lock(root: Path) -> dict:
    config = json.loads((root / "dependencies" / "vcmi.json").read_text())
    if config.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", config.get("commit", "")):
        raise PreparationError("Unsupported dependency lock format or invalid pinned commit")
    for key in ("directory", "patch_directory"):
        within(root, config[key])
    for overlay in config["overlays"]:
        source = within(root, overlay["source"])
        within(Path("/dependency"), overlay["destination"])
        if not source.is_dir() or source.is_symlink():
            raise PreparationError(f"Missing root-owned source directory: {source}")
    if not within(root, config["patch_directory"]).is_dir():
        raise PreparationError("Root-owned patch directory is missing")
    return config


def tree_entries(repo: Path, tree: str, env: dict | None = None) -> dict[str, tuple[str, str]]:
    result = {}
    for entry in git(repo, "ls-tree", "-r", "-z", tree, env=env).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, _kind, object_id = metadata.decode().split()
            result[os.fsdecode(name)] = (mode, object_id)
    return result


def overlay_files(source: Path):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".git" in relative.parts or path.name == ".DS_Store":
            continue
        if path.is_symlink() or path.is_file():
            yield relative, path


def file_data(path: Path) -> tuple[str, bytes]:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return "120000", os.fsencode(os.readlink(path))
    if not stat.S_ISREG(mode):
        raise PreparationError(f"Expected a regular file or symlink: {path}")
    return ("100755" if mode & 0o111 else "100644"), path.read_bytes()


def expected_entries(root: Path, repo: Path, config: dict) -> dict[str, tuple[str, str]]:
    """Build the intended tree without changing the checkout, index, or object store."""
    with tempfile.TemporaryDirectory(prefix="heroes-vcmi-check-") as temporary:
        temp = Path(temporary)
        objects = temp / "objects"
        objects.mkdir()
        repository_objects = Path(os.fsdecode(git(repo, "rev-parse", "--git-path", "objects")).strip())
        if not repository_objects.is_absolute():
            repository_objects = repo / repository_objects
        environment = {**os.environ, "GIT_INDEX_FILE": str(temp / "index"),
                       "GIT_OBJECT_DIRECTORY": str(objects),
                       "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repository_objects.resolve())}
        git(repo, "read-tree", config["commit"], env=environment)
        for patch in sorted(within(root, config["patch_directory"]).glob("*.patch")):
            git(repo, "apply", "--cached", "--binary", str(patch.resolve()), env=environment)
        for overlay in config["overlays"]:
            source = within(root, overlay["source"])
            destination = overlay["destination"]
            git(repo, "rm", "-r", "--cached", "--ignore-unmatch", "--", destination, env=environment)
            for relative, path in overlay_files(source):
                mode, data = file_data(path)
                object_id = git(repo, "hash-object", "-w", "--stdin", env=environment, payload=data).decode().strip()
                name = (Path(destination) / relative).as_posix()
                git(repo, "update-index", "--add", "--cacheinfo", mode, object_id, name, env=environment)
        tree = git(repo, "write-tree", env=environment).decode().strip()
        return tree_entries(repo, tree, env=environment)


def verify_repository(repo: Path, config: dict) -> None:
    if repo.is_symlink() or not repo.is_dir():
        raise PreparationError(f"Expected a real dependency directory: {repo}")
    actual_root = Path(os.fsdecode(git(repo, "rev-parse", "--show-toplevel")).strip()).resolve()
    if actual_root != repo.resolve():
        raise PreparationError(f"Directory is not its own Git checkout: {repo}")
    git(repo, "cat-file", "-e", config["commit"] + "^{commit}")
    git(repo, "merge-base", "--is-ancestor", config["commit"], "HEAD")


def differences(repo: Path, expected: dict[str, tuple[str, str]], require_submodules=True) -> list[str]:
    problems = []
    tracked = {os.fsdecode(name) for name in git(repo, "ls-files", "-z").split(b"\0") if name}
    problems.extend(f"Unexpected tracked file: {name}" for name in sorted(tracked - expected.keys()))
    untracked = {os.fsdecode(name) for name in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if name}
    problems.extend(f"Unrelated untracked file: {name}" for name in sorted(untracked - expected.keys()))
    if git(repo, "diff", "--cached", "--name-only", "-z"):
        problems.append("Dependency has staged changes; preserve or export them before preparation")
    for name, (mode, object_id) in expected.items():
        path = repo / name
        if mode == "160000":
            if not (path / ".git").exists():
                if require_submodules:
                    problems.append(f"Upstream submodule is uninitialized: {name}")
                continue
            try:
                actual_id = git(path, "rev-parse", "HEAD").decode().strip()
                if actual_id != object_id:
                    problems.append(f"Upstream submodule commit differs: {name}")
                if git(path, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
                    problems.append(f"Upstream submodule has local changes: {name}")
            except PreparationError as error:
                problems.append(f"Cannot verify upstream submodule {name}: {error}")
            continue
        if not path.exists() and not path.is_symlink():
            problems.append(f"Expected file is missing: {name}")
            continue
        try:
            actual_mode, data = file_data(path)
            actual_id = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if mode != "120000" and actual_id != object_id:
                # Honor upstream attributes such as gradlew.bat's CRLF checkout format.
                actual_id = git(repo, "hash-object", "--path", name, "--stdin", payload=data).decode().strip()
            if actual_mode != mode or actual_id != object_id:
                problems.append(f"File differs from root-owned recipe: {name}")
        except (OSError, PreparationError) as error:
            problems.append(str(error))
    return problems


def summarize(problems: list[str]) -> str:
    detail = "\n".join(f"  - {problem}" for problem in problems[:30])
    if len(problems) > 30:
        detail += f"\n  ... and {len(problems) - 30} more differences"
    return detail


def apply_recipe(root: Path, repo: Path, config: dict) -> None:
    if config.get("initialize_upstream_submodules", True):
        git(repo, "submodule", "update", "--init", "--recursive")
    for patch in sorted(within(root, config["patch_directory"]).glob("*.patch")):
        git(repo, "apply", "--binary", str(patch.resolve()))
    for overlay in config["overlays"]:
        source = within(root, overlay["source"])
        destination = within(repo, overlay["destination"])
        # Existing files were verified pristine before this function is called.
        for relative, path in overlay_files(source):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            if path.is_symlink():
                target.symlink_to(os.readlink(path))
            else:
                shutil.copy2(path, target)


def prepare(root: Path, check_only: bool = False) -> None:
    config = load_lock(root)
    repo = within(root, config["directory"])
    if repo.exists() or repo.is_symlink():
        verify_repository(repo, config)
        expected = expected_entries(root, repo, config)
        problems = differences(repo, expected)
        if not problems:
            print(f"Verified {repo}: pinned upstream + root patches + owned source match exactly.")
            return
        if check_only:
            raise PreparationError("Dependency does not match the root-owned recipe:\n" + summarize(problems))
        head = git(repo, "rev-parse", "HEAD").decode().strip()
        baseline_problems = differences(repo, tree_entries(repo, config["commit"]), require_submodules=False)
        if head != config["commit"] or baseline_problems:
            raise PreparationError("Refusing to change an existing checkout with unexpected content. "
                                   "Preserve/export its edits into the root project first.\n" + summarize(problems))
        apply_recipe(root, repo, config)
        problems = differences(repo, expected)
        if problems:
            raise PreparationError("Preparation failed verification; files are preserved for inspection:\n" + summarize(problems))
        print(f"Prepared and verified {repo} from the pinned upstream checkout.")
        return
    if check_only:
        raise PreparationError(f"Dependency is missing: {repo}. Run python3 tools/prepare_vcmi.py first.")
    # Assemble a fresh dependency beside its final location, then publish only after validation.
    with tempfile.TemporaryDirectory(prefix=".vcmi-prepare-", dir=repo.parent) as temporary:
        clone = Path(temporary) / "vcmi"
        git(repo.parent, "clone", "--depth", "1", "--branch", config["ref"], "--no-checkout",
            config["repository"], str(clone))
        if git(clone, "rev-parse", "HEAD").decode().strip() != config["commit"]:
            raise PreparationError("Upstream ref no longer matches the pinned commit; refusing preparation")
        git(clone, "checkout", "--detach", config["commit"])
        expected = expected_entries(root, clone, config)
        apply_recipe(root, clone, config)
        problems = differences(clone, expected)
        if problems:
            raise PreparationError("Fresh dependency failed verification:\n" + summarize(problems))
        if repo.exists() or repo.is_symlink():
            raise PreparationError("Dependency destination appeared during preparation; refusing to overwrite it")
        clone.rename(repo)
    print(f"Cloned, prepared and verified {repo} at {config['commit']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; do not modify or fetch the dependency")
    args = parser.parse_args()
    try:
        prepare(ROOT, check_only=args.check)
    except (PreparationError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"VCMI preparation failed: {error}\n")


if __name__ == "__main__":
    main()
