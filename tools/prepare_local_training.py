#!/usr/bin/env python3
"""Prepare or check a local training machine descriptor without launching games.

This records existing immutable checkpoint bytes and an already validated cold
runtime. It does not modify checkpoints, relocate dynamic libraries, start an
agent, or authorize gameplay. The user-invoked campaign owns its separate clock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_ARTIFACT_ROOT = Path.home() / "Projects/Heroes-artifacts"
DEFAULT_INPUT = DEFAULT_ARTIFACT_ROOT / "acceleration-update412-2026-10-26/input"
DEFAULT_RUNTIME = DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-update92-artifact-fix2/bin"
DEFAULT_RUN_ROOT = DEFAULT_ARTIFACT_ROOT / "local-script-training"
DEFAULT_ORIGINAL_MAP = DEFAULT_RUNTIME.parent / "maps/hero-management-v2.vmap"
ORIGINAL_MAP_SHA256 = "ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3"
SEED_NAMESPACES = {"easier8x8": {"engine_base": 20_000_000, "policy_base": 30_000_000},
                   "original": {"engine_base": 40_000_000, "policy_base": 50_000_000}}
ORIGIN_FILES = {
    "freeze": DEFAULT_RUNTIME.parent / "freeze-receipt.json",
    "release": DEFAULT_ARTIFACT_ROOT / "local-curriculum-update146/campaign-v3/release.json",
    "gate": DEFAULT_ARTIFACT_ROOT / "local-curriculum-update146/study4/study-report.json",
}
SCHEMA = "heroes-local-training-machine-v1"
PREREQUISITE_PINS = {
    "local-builtin-revalidation-v2-2026-10-25.json": "3515928087fdbafc4c3e3686ae370ed1be992fbd50faaed70475255c7113a09f",
    "build-terminal-diagnostic-2026-10-25.json": "b67887eb2bb972c07b48563b080b5fbc7471aa52e27b29cf7a61e6edacbbe849",
}
ORIGIN_PINS = {
    "freeze": "3846aef3c80121b6bbbeed0674b6713ab0c84f80b1dfe169f3e41c79615ba7e9",
    "release": "dee510c199047283f95f5193854490dfbc95e953563dcbad80065e24046438ce",
    "gate": "ab1d87a72285441e3b7d9fb55f921e6f29831c885f66411cdd9b95278ce194d4",
}
INPUT_PINS = {
    "plan": ("checkpoint/plan.json", "4ceb55e2f477c8406e90eaa9a2b79c4a13514bd37b984b626e2ea79f24f95e23"),
    "report": ("checkpoint/report.json", "20c070ce2390315176073db880498ad1a7c86ddca2f4451bd82ad2c979327110"),
    "policy": ("checkpoint/after-policy.pt", "1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88"),
    "resume": ("checkpoint/after-resume.pt", "09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321"),
    "initial_checkpoint": ("dependencies/update411-after-policy.pt", "9aeac52025a9bd0cd307303e8b06f732aefa640795ccf9e4fd9d7ba1f0ba65bd"),
}
MAP_PINS = {
    "0": "7d88860234664580c564fc2c71554fd813519ace9904ff368dc30eef3c352019",
    "1": "75303cc2b19c1d0ddc5fbac35315a951dfc9bb6e780b4245508f8a2a78e691f1",
}
# Canonical cold runtime used by the completed local curriculum campaign. Its
# root-owned engine recipe still matches main; PR29/30 binaries are not selected.
RUNTIME_PINS = {
    "vcmiclient": "79e527665872bed791f2a3fed88e5087880964211dcbedebd0ba57917c541540",
    "vcmireplay": "0a00c7b8d01aa004070ee724c120387b4b454966b6eb66ae5aab2dd166ac1b13",
    "libvcmi.dylib": "53370138f2db6a3cefab07c57a6239931876411cf96bdced7d9d6a132204a243",
    "AI/libBattleAI.dylib": "73e9b2b7e714a4440552cd502d8c767df56b787f367cf1a513062bfbb16995b1",
    "AI/libEmptyAI.dylib": "8351b6e36945796fbd3215c537afea120eb59adf83275c589604ab61ac12213c",
    "AI/libExplorerRL.dylib": "27a37adc672fdfa18499d40f72ec4e4be35afe820057987752b6b6b0834abf41",
    "AI/libNullkiller.dylib": "bc9d458abb32b1d9da92bf822dc4b8a9d01bb998386f6f044ab96572bbbfa8db",
    "AI/libStupidAI.dylib": "30e8a58a8754f1cf888f896d2a9a1d1bd466a0bd2181acc978c03a6940d7a205",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file(), "Missing file: " + str(path))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def pinned(path, digest):
    result = binding(path)
    require(result["sha256"] == digest, "Input differs from its accepted identity: " + result["path"])
    return result


def durable(path):
    path = Path(path).expanduser().resolve()
    require("/.codex/worktrees/" not in str(path), "Keep durable outputs outside managed worktrees")
    require(path != ROOT and not path.is_relative_to(ROOT), "Keep durable outputs outside the source checkout")
    return path


def verify_parent(parent):
    from experiments.learner.builtin_preservation import verify_pair
    plan, report = [json.loads(Path(parent[name]["path"]).read_text()) for name in ("plan", "report")]
    require(report["plan_sha256"] == parent["plan"]["sha256"], "Parent plan/report mismatch")
    initial = plan["checkpoint_metadata"]["initial_checkpoint"]
    require(all(initial[key] == parent["initial_checkpoint"][key] for key in ("sha256", "bytes")),
            "Wrong initial checkpoint dependency")
    proof = verify_pair(parent["resume"]["path"], parent["policy"]["path"], plan, report,
        migration=SimpleNamespace(checkpoint_options={"historical_torch_version": plan["source"]["torch"]}))
    require(proof["progress"] == {"updates": 412, "rollouts": 1648, "decisions": 95836}
            and proof["exact_model_adam_rng_progress_pool"], "Checkpoint412 full restoration failed")
    return proof


def current_engine_recipe():
    from experiments.episodes.scoring import engine_recipe_digests
    return engine_recipe_digests(ROOT)


def runtime_origin(runtime_bin, runtime):
    result = {name: pinned(ORIGIN_FILES[name], sha) for name, sha in ORIGIN_PINS.items()}
    freeze, release, gate = [json.loads(Path(result[name]["path"]).read_text())
                             for name in ("freeze", "release", "gate")]
    require(release["runtime"] == runtime and Path(release["runtime_bin"]) == runtime_bin
            and Path(freeze["frozen_bin"]) == runtime_bin, "Runtime differs from its accepted origin")
    require(release["gate"] == result["gate"] and gate["status"] == "finished"
            and gate["owned_cleanup_verified"] and gate["ppo_launched"] is False
            and all(release["validation_outcomes"][seat][outcome] > 0
                    for seat in ("0", "1") for outcome in ("win", "loss")),
            "Accepted cold runtime evidence is incomplete")
    accepted_recipe = {k: v for k, v in release["implementation"].items()
                       if k.startswith("engine/") or k == "dependencies/vcmi.json"}
    require(accepted_recipe and current_engine_recipe() == accepted_recipe,
            "Current engine recipe differs from the accepted cold runtime")
    frozen = {item["path"]: item["frozen_sha256"] for item in freeze["artifacts"]}
    require(all(frozen.get(name) == sha for name, sha in RUNTIME_PINS.items()), "Frozen binary receipt differs")
    external = [pinned(item["path"], item["sha256"]) for item in freeze["external_dependencies"]]
    resources = [pinned(runtime_bin / item["path"], item["sha256"]) for item in freeze["resources"]]
    return {"receipts": result, "engine_recipe": accepted_recipe,
            "external_dependencies": external, "resources": resources,
            "meaning": "Historical cold acceptance only; checkpoint412/current adapter needs fresh launch acceptance."}


def original_map_identity(path):
    from experiments.learner.builtin_v8_migration import map_identity
    map_identity(path)
    return pinned(path, ORIGINAL_MAP_SHA256)


def profile_maps(input_root, profile):
    require(profile in SEED_NAMESPACES, "Unknown explicit map profile")
    if profile == "original":
        result = original_map_identity(DEFAULT_ORIGINAL_MAP)
        return [dict(result), dict(result)]
    return [pinned(input_root / "maps" / ("curriculum-3-" + seat + ".vmap"), MAP_PINS[seat])
            for seat in ("0", "1")]


def build_configuration(input_root, runtime_bin, data, artifact_root, run_root, profile="easier8x8"):
    import torch
    from experiments.headless.contextual_evidence import runtime_identity
    input_root, runtime_bin = [Path(p).expanduser().resolve() for p in (input_root, runtime_bin)]
    require(runtime_bin == DEFAULT_RUNTIME.resolve(),
            "This preparation recipe pins the accepted Mac cold runtime in place; a different build needs a new recipe/gate")
    require(str(torch.__version__) in ("2.9.1", "2.9.1+cpu"), "Use the existing Torch2.9.1 environment")
    parent = {name: pinned(input_root / path, sha) for name, (path, sha) in INPUT_PINS.items()}
    maps = profile_maps(input_root, profile)
    runtime_files = {name: pinned(runtime_bin / name, sha) for name, sha in RUNTIME_PINS.items()}
    data = Path(data).expanduser().resolve()
    require(data.is_dir() and (data / "Data").is_dir(), "Expected an existing VCMI user data directory with Data/")
    proof = verify_parent(parent)
    artifact_root = durable(artifact_root)
    run_root = durable(run_root)
    require(run_root != artifact_root and run_root.is_relative_to(artifact_root),
            "Run output must be a child of the durable artifact root")
    prerequisites = {name: pinned(artifact_root / "inputs/validation" / name, sha)
                     for name, sha in PREREQUISITE_PINS.items()}
    runtime = runtime_identity(runtime_bin)
    require(runtime == {name: sha for name, sha in RUNTIME_PINS.items() if name != "vcmiclient"},
            "Unexpected native module inventory")
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": parent, "maps": maps, "runtime_bin": str(runtime_bin), "runtime": runtime,
        "runtime_files": runtime_files, "runtime_origin": runtime_origin(runtime_bin, runtime),
        "data": str(data), "artifact_root": str(artifact_root), "run_root": str(run_root),
        "prerequisites": prerequisites,
        "python": {"executable": str(Path(sys.executable).absolute()), "version": sys.version,
                   "torch": str(torch.__version__)},
        "checkpoint_restore": proof, "next_sequence": 266, "seed_namespace": dict(SEED_NAMESPACES[profile]),
        "default_profile": profile, "backend": "cold", "gameplay_performed": False,
        "ppo_updates": 0, "training_authorized_by_preparation": False,
        "scope": "Machine-local inputs only. The explicit user launch creates a fresh bounded session and runs current-source acceptance before PPO."}


def validate_config(value):
    """Recheck every input and restore the parent without changing its bytes."""
    require(value.get("schema") == SCHEMA and value.get("backend") == "cold", "Unsupported machine configuration")
    for name, (_, sha) in INPUT_PINS.items():
        require(pinned(value["parent"][name]["path"], sha) == value["parent"][name], "Parent binding changed")
    profile = value.get("default_profile")
    require(profile in SEED_NAMESPACES, "Unknown explicit map profile")
    if profile == "original":
        require(len(value["maps"]) == 2 and value["maps"][0] == value["maps"][1],
                "Original profile requires the same exact map for both seats")
        require(original_map_identity(value["maps"][0]["path"]) == value["maps"][0], "Map binding changed")
    else:
        for seat, sha in MAP_PINS.items():
            require(pinned(value["maps"][int(seat)]["path"], sha) == value["maps"][int(seat)], "Map binding changed")
    for name, sha in RUNTIME_PINS.items():
        require(pinned(Path(value["runtime_bin"]) / name, sha) == value["runtime_files"][name], "Runtime binding changed")
    require(Path(value["runtime_bin"]).resolve() == DEFAULT_RUNTIME.resolve(), "Runtime relocation not accepted")
    input_root = Path(value["parent"]["plan"]["path"]).parent.parent
    expected = build_configuration(input_root, value["runtime_bin"], value["data"],
                                   value["artifact_root"], value["run_root"], profile)
    # Existing easier8x8 descriptors predate the explicit namespace field. Their
    # only permitted implicit namespace remains the original 20M/30M values.
    if profile == "easier8x8" and "seed_namespace" not in value:
        expected.pop("seed_namespace")
    require(set(value) == set(expected), "Machine configuration fields changed")
    for key in expected.keys() - {"created_utc"}:
        require(value[key] == expected[key], "Machine configuration changed: " + key)
    return value


def check_configuration(path):
    return validate_config(json.loads(Path(path).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runtime-bin", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--profile", choices=tuple(SEED_NAMESPACES), default="easier8x8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Check an existing descriptor; never rewrite it")
    args = parser.parse_args()
    output = durable(args.output)
    if args.check:
        value = check_configuration(output)
    else:
        run_root = args.run_root or (DEFAULT_RUN_ROOT / "original" if args.profile == "original" else DEFAULT_RUN_ROOT)
        value = build_configuration(args.input, args.runtime_bin, args.data, args.artifact_root, run_root, args.profile)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"configuration": str(output), "checkpoint": value["checkpoint_restore"]["progress"],
                      "profile": value["default_profile"], "checked": args.check,
                      "gameplay_performed": False, "ppo_updates": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
