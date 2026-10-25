#!/usr/bin/env python3
"""Fresh, inference-only Linux diagnostics with immutable private input bindings.

Preparation checks inputs and writes a plan; it launches no game. The public
loader accepts only the handoff's original checkpoint and its explicit Linux
conversion. Mac schedules and failed historical admissions are never resumed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless.contextual_evidence import strict_json
from tools import inventory_cloud_inputs as inputs
from tools import port_fixed_policy_linux as portability

SCHEMA = "heroes-cloud-linux-diagnostic-v1-plan"
INTERFACE = portability.INTERFACE
LEARNER_SHA256 = "1c290bcb1ca37cb876841645fab5cd08a94f999bbc57a56fb285243a8e46bbae"
OPPONENTS = [["hero-opponent-7176f1a2e81393cb",
              "7176f1a2e81393cbfe709238fcd3c0060871a9ddaec1cf4181812f75fb08fc82"]]
MAP_SHA256 = "852ea0211d20e1cbfb64790fe8076254e98983ba0ae29e854db192c816c76ad1"
CONFIGURATION = {"shared_wall_seconds": 3600, "cleanup_reserve_seconds": 30,
                 "hard_stop_reserve_seconds": 2, "torch_threads": 1}
SCHEDULE = [
    {"episode": offset + seat + 1, "learner_player": seat,
     "engine_seed": engine_seed, "policy_seed": policy_seed, "map_kind": kind,
     "ordinary_choice_limit": choices, "max_total_choices": hard_choices,
     "max_days": days, "case_wall_seconds": seconds}
    for offset, kind, engine_seed, policy_seed, choices, hard_choices, days, seconds in (
        (0, "fixture", 251025301, 251025302, 256, 264, 12, 180),
        (2, "ordinary", 251025401, 251025402, 2040, 2048, 200, 1200))
    for seat in (0, 1)]
VERIFICATION = [
    "exact_original_and_converted_checkpoint_bytes_and_both_models",
    "sole_learner_private_history_and_fresh_episode_memory",
    "independent_persisted_policy_reconstruction_including_intervening_frames",
    "independent_native_replay_and_full_request_accounting",
    "terminal_unapplied_requests_explicitly_checked_never_discarded",
    "natural_terminal_only_scores_cutoffs_and_failures_unscored",
    "post_row_source_runtime_input_identity_and_process_cleanup",
    "durable_completion_receipt_before_supervisor_admission",
]
PLAN_FIELDS = {"schema", "configuration", "schedule", "execution_output", "original_checkpoint", "checkpoint",
    "portability", "input_manifest", "data", "checkpoint_source", "frozen", "source", "implementation",
    "python_runtime", "binary", "replay_binary", "runtime", "native", "maps", "native_opponents", "verification",
    "prepared_check", "training_performed", "optimizer_steps", "training_samples", "seed_scope", "wall_scope",
    "scope", "failure_policy", "private_data", "native_dependencies", "native_loader_environment", "native_resources"}


def require(value, message):
    if not value:
        raise ValueError(message)


def same(left, right):
    """JSON equality that distinguishes bool/int and int/float aliases."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), **inputs.file_binding(path)}


def private_path(path):
    path = Path(path).expanduser().resolve()
    require(not path.is_relative_to(ROOT), "Private inputs and raw execution must remain outside the Git checkout")
    return path


def runtime():
    import torch
    require(platform.system() == "Linux" and platform.machine() == "x86_64"
            and sys.version_info[:2] == (3, 12), "Validated Linux x86_64 CPython 3.12 required")
    require(str(torch.__version__) == "2.9.1+cpu" and torch.version.cuda is None and torch.version.hip is None,
            "Validated CPU Torch 2.9.1+cpu required")
    require(torch.get_num_threads() == 1, "Use exactly one Torch thread")
    packages = sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())
    directory = Path(torch.__file__).resolve().parent
    torch_paths = {Path(torch._C.__file__).resolve(), *(directory / "lib").glob("*.so*")}
    torch_binaries = {str(path.relative_to(directory)): inputs.file_binding(path) for path in sorted(torch_paths)
                      if path.is_file()}
    require(torch_binaries and "lib/libtorch_cpu.so" in torch_binaries, "CPU Torch runtime libraries are missing")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "executable_identity": inputs.file_binding(sys.executable), "torch_binaries": torch_binaries,
            "torch": str(torch.__version__), "torch_threads": 1,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cuda": torch.version.cuda, "hip": torch.version.hip,
            "packages": [list(item) for item in packages]}


def implementation():
    """Bind authored execution code, canonical engine recipe and dependency locks."""
    names = subprocess.check_output(["git", "ls-files", "-z", "--", "engine", "experiments", "tools",
                                     "dependencies", "configs", "validation/probes"], cwd=ROOT).decode().split("\0")
    paths = {ROOT / name for name in names if name}
    # A source-only test can see this file before its first commit. Live prepare
    # separately refuses untracked implementation files and all tracked edits.
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): inputs.file_binding(path)["sha256"] for path in sorted(paths)
            if path.is_file()}


def source_identity():
    from experiments.learner.inference import source_identity as learner_source
    return learner_source(INTERFACE)


def loader_environment():
    return {name: os.environ.get(name) for name in ("LD_LIBRARY_PATH", "LD_PRELOAD")}


def native_dependencies(binary, native_runtime):
    """Bind ldd's actual Linux runtime resolution, including the local Debian prefix."""
    binary = Path(binary).resolve()
    artifacts = {binary, *(binary.parent / name for name in native_runtime)}
    dependencies, resolutions = {}, {}
    for artifact in sorted(artifacts):
        checked = subprocess.run(["ldd", str(artifact)], capture_output=True, text=True, timeout=20, check=True)
        require("not found" not in checked.stdout and not checked.stderr.strip(), "Native dependency resolution failed")
        resolved = []
        for line in checked.stdout.splitlines():
            match = re.search(r"(?:=>\s+)?(/\S+)\s+\(0x[0-9a-fA-F]+\)", line)
            if match:
                path = Path(match.group(1)).resolve(strict=True)
                name = str(path)
                if name not in dependencies:
                    dependencies[name] = inputs.file_binding(path)
                resolved.append(name)
            else:
                require(line.strip().startswith("linux-vdso.so."), "Unknown ldd resolution: " + line.strip())
        require(resolved, "Native object has no resolved runtime dependencies")
        resolutions[str(artifact)] = sorted(set(resolved))
    return {"artifacts": resolutions, "libraries": dict(sorted(dependencies.items()))}


def native_resources(binary):
    """Bind the development runtime's rules/mod files and reject added resource roots.

    VCMIDirsXDG disables system resource directories in this exact development
    layout. The per-game profile supplies only the separately inventoried Data
    and authored map; all shared rules/resources come from config and Mods.
    """
    binary = Path(binary).resolve()
    require(binary == ROOT / "vcmi/build/bin/vcmiclient", "Use the exact prepared development client")
    result = {}
    for name in ("config", "Mods"):
        entry = binary.parent / name
        require([child.name for child in binary.parent.iterdir() if child.name.casefold() == name.casefold()] == [name],
                "Ambiguous development resource root: " + name)
        root = entry.resolve(strict=True)
        require(root == ROOT / "vcmi" / name and root.is_dir(), "Unexpected development resource root: " + name)
        records = []
        def fail(error):
            raise error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
            for child in dirs + files:
                path = Path(directory) / child
                mode = path.lstat().st_mode
                require(not stat.S_ISLNK(mode) and (stat.S_ISDIR(mode) or stat.S_ISREG(mode)),
                        "Native resource contains a symlink or nonregular entry")
            for child in files:
                path = Path(directory) / child
                records.append({"path": path.relative_to(root).as_posix(), **inputs.file_binding(path)})
        require(records, "Native resource root is empty: " + name)
        result[name] = {"resolved": str(root), "symlink": os.readlink(entry) if entry.is_symlink() else None,
                        "files": sorted(records, key=lambda row: row["path"])}
    # These optional filesystem.json roots must come only from the private
    # per-episode profile. Unexpected build-local contents could shadow them.
    for name in ("Data", "Sprites", "Sounds", "Mp3", "Video", "Maps", "scripts"):
        matches = [entry.name for entry in binary.parent.iterdir() if entry.name.casefold() == name.casefold()]
        require(not matches, "Unexpected development resource directory: " + name)
        result[name] = {"present": False}
    return result


def prepared_check():
    return subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, check=True).stdout.strip()


def require_clean_source(source):
    require(source["tracked_worktree_dirty"] is False, "Commit reviewed source before preparing a live Linux plan")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
        "engine", "experiments", "tools", "dependencies", "configs", "validation/probes"], cwd=ROOT, text=True)
    require(not untracked.strip(), "Commit new implementation files before preparing a live Linux plan")


def _compatible_source(saved, current, *, original=False):
    require(type(saved) is dict and set(saved) == set(current), "Checkpoint source metadata fields differ")
    require(type(saved["git_commit"]) is str and re.fullmatch("[0-9a-f]{40}", saved["git_commit"])
            and type(saved["tracked_worktree_dirty"]) is bool, "Checkpoint source commit or dirty marker is malformed")
    require(saved["interface"] == current["interface"] == INTERFACE, "Checkpoint interface changed")
    if original:
        require(saved["torch"] == "2.9.1" and type(saved["python"]) is str
                and saved["python"].startswith("3.13.5 "), "Original runtime provenance differs")
    else:
        require(saved["torch"] == current["torch"] and saved["python"] == current["python"],
                "Converted checkpoint runtime differs")
        require(same(saved["source_sha256"], current["source_sha256"]),
                "Converted checkpoint source inventory differs")
        require(subprocess.run(["git", "merge-base", "--is-ancestor", saved["git_commit"], current["git_commit"]],
                               cwd=ROOT, capture_output=True, timeout=10).returncode == 0,
                "Converted checkpoint source commit is outside this source lineage")
    require(type(saved["source_sha256"]) is dict and saved["source_sha256"], "Missing checkpoint source inventory")
    for name, sha in saved["source_sha256"].items():
        require(type(name) is str and type(sha) is str and re.fullmatch("[0-9a-f]{64}", sha)
                and current["source_sha256"].get(name) == sha, "Original learner source changed: " + str(name))


def _load_fixed_inputs(original_path, converted_path, report_path, *, expected=None):
    """Internal fixture entry; public preparation never accepts replacement identities."""
    import torch
    from experiments.learner.checkpoints import load_policy_checkpoint
    from validation.probes.validate_hero_policy import frozen_fingerprints
    expected = expected or {"original_sha256": portability.ORIGINAL_SHA256, "original_bytes": portability.ORIGINAL_BYTES,
                            "learner": LEARNER_SHA256, "opponents": OPPONENTS}
    runtime()
    original_binding, converted_binding, report_binding = map(binding, (original_path, converted_path, report_path))
    require(original_binding["sha256"] == expected["original_sha256"]
            and original_binding["bytes"] == expected["original_bytes"], "Original checkpoint size or SHA-256 differs")
    require(original_binding["path"] != converted_binding["path"], "The original and converted checkpoints must be separate")
    report = strict_json(Path(report_path).read_bytes())
    expected_report_fields = {"schema", "purpose", "original", "converted", "source_runtime", "target_runtime",
        "learner_weights_sha256", "opponents", "exact_learner_and_opponent_tensors", "standard_loader_passed",
        "original_preserved", "optimizer_updates", "gameplay_performed", "cross_platform_behavior_equivalence",
        "linux_gameplay_and_native_replay"}
    require(type(report) is dict and set(report) == expected_report_fields, "Portability report fields differ")
    declarations = {"schema": portability.SCHEMA, "purpose": "frozen_inference_only",
        "original": {"sha256": expected["original_sha256"], "bytes": expected["original_bytes"]},
        "converted": {"filename": Path(converted_path).name, "sha256": converted_binding["sha256"], "bytes": converted_binding["bytes"]},
        "learner_weights_sha256": expected["learner"], "opponents": expected["opponents"],
        "exact_learner_and_opponent_tensors": True, "standard_loader_passed": True, "original_preserved": True,
        "optimizer_updates": 0, "gameplay_performed": False,
        "cross_platform_behavior_equivalence": "not_established", "linux_gameplay_and_native_replay": "not_run"}
    require(all(same(report.get(key), value) for key, value in declarations.items()), "Portability report binding or declarations differ")
    require(report["target_runtime"] == {"torch": str(torch.__version__), "python": sys.version,
                                         "platform": platform.platform()}, "Portability target runtime differs")
    original = torch.load(original_path, map_location="cpu", weights_only=True)
    converted = torch.load(converted_path, map_location="cpu", weights_only=True)
    current = source_identity()
    require(type(original) is dict and type(converted) is dict and original.keys() == converted.keys(),
            "Original and converted envelope fields differ")
    require(original["kind"] == "policy" and original["provenance"] == "random_initialization"
            and original["resume"] is None and original["torch_version"] == "2.9.1", "Original is not the frozen policy")
    _compatible_source(original["source_metadata"], current, original=True)
    require(report["source_runtime"] == {key: original["source_metadata"][key] for key in ("torch", "python")},
            "Portability source runtime differs")
    require(converted["checkpoint"] == original["checkpoint"] + "-linux-cpu-v1"
            and converted["torch_version"] == str(torch.__version__), "Converted artifact identity differs")
    metadata = deepcopy(converted["source_metadata"])
    provenance = metadata.pop("portability", None)
    require(same(provenance, {"schema": portability.SCHEMA, "original_sha256": expected["original_sha256"],
        "original_bytes": expected["original_bytes"], "original_checkpoint": original["checkpoint"],
        "original_source_metadata": original["source_metadata"], "purpose": "frozen_inference_only",
        "optimizer_updates": 0, "cross_platform_behavior_equivalence": "not_established"}),
        "Converted original provenance differs")
    _compatible_source(metadata, current)
    for name in original.keys() - {"checkpoint", "torch_version", "source_metadata"}:
        portability.assert_exact_state(original[name], converted[name])
    loaded = load_policy_checkpoint(converted_path, expected_source_metadata=converted["source_metadata"],
        interface=INTERFACE, allowed_provenance=("random_initialization",))
    loaded.model.eval().requires_grad_(False)
    fixed = frozen_fingerprints(loaded, converted_path)
    require(fixed["learner_weights_sha256"] == expected["learner"] and same(fixed["opponents"], expected["opponents"]),
            "Frozen learner or archived opponent identities differ")
    portability.assert_exact_state(original["model_state"], dict(loaded.model.state_dict()))
    require(loaded.opponent_pool is not None, "Archived opponent pool missing")
    for entry in original["opponent_pool"]:
        portability.assert_exact_state(entry["model_state"], dict(loaded.opponent_pool.get(entry["checkpoint"]).model.state_dict()))
    require(all(binding(path) == before for path, before in zip((original_path, converted_path, report_path),
        (original_binding, converted_binding, report_binding))), "Checkpoint or portability report changed while loading")
    return loaded, fixed


def load_fixed(plan):
    for name in ("original_checkpoint", "checkpoint", "portability"):
        private_path(plan[name]["path"])
        require(binding(plan[name]["path"]) == plan[name], "Bound private artifact changed: " + name)
    loaded, fixed = _load_fixed_inputs(plan["original_checkpoint"]["path"], plan["checkpoint"]["path"], plan["portability"]["path"])
    require(same(loaded.source_metadata, plan["checkpoint_source"]) and same(fixed, plan["frozen"]),
            "Prepared frozen checkpoint identity changed")
    return loaded, fixed


def validate_inventory(data, original, manifest_path, *, full_data=True):
    private_path(Path(data) / "Data")
    document = strict_json(Path(manifest_path).read_bytes())
    require(type(document) is dict and set(document) == {"schema", "provenance", "data_files", "checkpoint",
        "game_data_validation", "gameplay_performed"}, "Private input inventory fields differ")
    require(document["schema"] == "heroes-cloud-private-inputs-v1" and type(document["provenance"]) is str
            and document["provenance"].strip() and document["game_data_validation"] == "byte_inventory_only"
            and document["gameplay_performed"] is False, "Private input inventory declarations differ")
    require(same(document["checkpoint"], {"filename": "fixed-policy.pt", "bytes": original["bytes"],
                                         "sha256": original["sha256"]}), "Inventory checkpoint binding differs")
    records = document["data_files"]
    require(type(records) is list and records, "Private Data inventory is empty")
    names = []
    for record in records:
        require(type(record) is dict and set(record) == {"path", "bytes", "sha256"}, "Malformed Data inventory row")
        name = record["path"]
        require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts
                and name == Path(name).as_posix(), "Invalid private Data relative path")
        require(type(record["bytes"]) is int and record["bytes"] >= 0 and type(record["sha256"]) is str
                and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "Invalid private Data file identity")
        names.append(name)
    require(names == sorted(set(names)), "Private Data rows must be unique and sorted")
    if full_data:
        require(same(inputs.inventory_data(data), records), "Private Data differs from its inventory")
    return {"files": len(records), "bytes": sum(row["bytes"] for row in records),
            "inventory_sha256": binding(manifest_path)["sha256"], "validation": "byte_inventory_only"}


def map_identity(path, kind):
    if kind == "ordinary":
        from validation.probes.validate_hero_policy import hero_map_identity
        result = {**binding(path), **hero_map_identity(path)}
        require(result["sha256"] == MAP_SHA256, "Ordinary map differs from the existing evaluator")
        return result
    require(kind == "fixture", "Unknown map kind")
    from validation.probes.validate_builtin_boundaries import fixture_identity
    value = fixture_identity(path)
    return {**value["map"], "fixture": True, "kind": value["kind"], "manifest": value["manifest"],
            "semantics_verified": True, "ordinary_cohort_member": False,
            "choice_policy": "unchanged_frozen_neural_policy_no_fixture_action_script"}


def _validate_document(plan, path):
    require(type(plan) is dict and set(plan) == PLAN_FIELDS and plan.get("schema") == SCHEMA,
            "Expected a fresh Linux diagnostic plan with every declared field")
    require(same(plan.get("configuration"), CONFIGURATION) and same(plan.get("schedule"), SCHEDULE),
            "Fixed Linux schedule or shared limits changed")
    require(same(plan.get("verification"), VERIFICATION) and plan.get("training_performed") is False
            and type(plan.get("optimizer_steps")) is int and plan["optimizer_steps"] == 0
            and type(plan.get("training_samples")) is int and plan["training_samples"] == 0,
            "Linux validation scope changed")
    require(plan.get("execution_output") == str(Path(path).resolve().parent / "execution"), "Execution output differs from the fresh plan")
    private_path(path)
    private_path(plan["execution_output"])
    require(set(plan["maps"]) == {"fixture", "ordinary"}, "Both separate map scopes are required")
    for name in ("original_checkpoint", "checkpoint", "portability", "input_manifest"):
        private_path(plan[name]["path"])
    private_path(plan["data"])
    require(Path(plan["replay_binary"]["path"]).resolve() == Path(plan["binary"]).resolve().parent / "vcmireplay"
            and plan["replay_binary"]["sha256"] == plan["runtime"].get("vcmireplay"),
            "Replay binary must be the bound native runtime verifier")


def read_plan(path, sha):
    path = Path(path).expanduser().resolve(strict=True)
    require(binding(path)["sha256"] == sha, "Prepared Linux plan SHA-256 changed")
    plan = strict_json(path.read_bytes())
    _validate_document(plan, path)
    return plan


def check_identity(plan, full_data=True):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.evaluate_builtin_opponent import native_opponent
    require(type(full_data) is bool, "Full Data validation must be explicit")
    require(same(source_identity(), plan["source"]) and same(implementation(), plan["implementation"]),
            "Authored source changed after preparation")
    require_clean_source(plan["source"])
    require(same(runtime(), plan["python_runtime"]), "Python or package runtime changed")
    for name in ("original_checkpoint", "checkpoint", "portability", "input_manifest", "replay_binary"):
        require(binding(plan[name]["path"]) == plan[name], "Bound artifact changed: " + name)
    require(same(runtime_identity(Path(plan["binary"]).parent), plan["runtime"]), "Native libraries or replay changed")
    require(same(loader_environment(), plan["native_loader_environment"])
            and same(native_dependencies(plan["binary"], plan["runtime"]), plan["native_dependencies"]),
            "Resolved native dependencies or loader environment changed")
    require(same(native_resources(plan["binary"]), plan["native_resources"]), "Native rules or mod resources changed")
    require(prepared_check() == plan["prepared_check"], "Prepared canonical engine check changed")
    require(same([native_opponent(plan["runtime"], seat) for seat in (0, 1)], plan["native_opponents"]),
            "Native opponent identity changed")
    for kind, identity in plan["maps"].items():
        require(same(map_identity(identity["path"], kind), identity), "Diagnostic map changed: " + kind)
        launcher = VcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(identity["path"]))
        launcher.preflight()
        require(same(launcher.identity(), plan["native"][kind]), "Native client or map changed: " + kind)
    require(same(validate_inventory(plan["data"], plan["original_checkpoint"], plan["input_manifest"]["path"],
                                   full_data=full_data), plan["private_data"]), "Private input inventory changed")


def prepare(args):
    import torch
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.evaluate_builtin_opponent import native_opponent
    torch.set_num_threads(1)
    output = private_path(args.output)
    require(not output.exists(), "Preserve every previous plan and attempt; output already exists")
    paths = {name: private_path(getattr(args, name)) for name in
             ("original_checkpoint", "checkpoint", "portability", "input_manifest", "data")}
    source = source_identity()
    require_clean_source(source)
    loaded, fixed = _load_fixed_inputs(paths["original_checkpoint"], paths["checkpoint"], paths["portability"])
    native_runtime = runtime_identity(Path(args.binary).resolve().parent)
    maps = {kind: map_identity(path, kind) for kind, path in (("fixture", args.fixture_map), ("ordinary", args.map))}
    natives = {}
    for kind, identity in maps.items():
        launcher = VcmiLauncher(Path(args.binary), paths["data"], Path(identity["path"]))
        launcher.preflight()
        natives[kind] = launcher.identity()
    checked = prepared_check()
    plan = {"schema": SCHEMA, "configuration": deepcopy(CONFIGURATION), "schedule": deepcopy(SCHEDULE),
        "execution_output": str(output / "execution"),
        **{name: binding(paths[name]) for name in ("original_checkpoint", "checkpoint", "portability", "input_manifest")},
        "data": str(paths["data"]), "checkpoint_source": loaded.source_metadata, "frozen": fixed,
        "source": source, "implementation": implementation(), "python_runtime": runtime(),
        "binary": str(Path(args.binary).resolve()), "replay_binary": binding(args.replay_binary),
        "runtime": native_runtime, "native": natives, "maps": maps,
        "native_dependencies": native_dependencies(args.binary, native_runtime),
        "native_loader_environment": loader_environment(),
        "native_resources": native_resources(args.binary),
        "native_opponents": [native_opponent(native_runtime, seat) for seat in (0, 1)],
        "verification": list(VERIFICATION), "prepared_check": checked,
        "training_performed": False, "optimizer_steps": 0, "training_samples": 0,
        "seed_scope": "Fresh declared Linux diagnostic seeds; no stopped Mac schedule is resumed and no held-out statistical claim is made.",
        "wall_scope": "One shared 3600-second maximum covers all four rows, collection, reconstruction, replay and cleanup; row ceilings cannot extend it.",
        "scope": "Two townless fixture cases and two ordinary-map diagnostics; fixture evidence remains separate from ordinary games.",
        "failure_policy": "Preserve every failed, cutoff and not-started row. No retries, replacements or extension to obtain a natural terminal."}
    plan["private_data"] = validate_inventory(paths["data"], plan["original_checkpoint"], paths["input_manifest"])
    _validate_document(plan, output / "plan.json")
    check_identity(plan)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    descriptor = os.open(output / "plan.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"plan": str(output / "plan.json"), "sha256": binding(output / "plan.json")["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "original-checkpoint", "checkpoint", "portability", "input-manifest", "data", "binary", "replay-binary", "map", "fixture-map"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
