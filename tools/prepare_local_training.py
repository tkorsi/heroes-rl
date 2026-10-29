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
from tools.training_environment import NEXT_LAUNCH_ENVIRONMENT, verify_environment

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
RUNTIME_ACCEPTANCE_SCHEMA = "heroes-local-training-runtime-acceptance-v1"
# This opt-in release stays unusable until its completed acceptance descriptor
# has been reviewed and its immutable hash is pinned here. A supplied pathname
# or a newly built binary is never sufficient authorization to select a runtime.
ACCEPTED_RUNTIME_RELEASES = {
    "cpu-next-launch-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-terminal-replay-repair2/cpu-next-launch-acceptance-v2.json",
        "sha256": None,
        "python_environment": NEXT_LAUNCH_ENVIRONMENT,
    },
    "terminal-replay-1392-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-terminal-replay-repair2/accepted-runtime-v2.json",
        "sha256": "9f9292bc8b5e959d7f7ed1152dc50b9d3404b5ed67b73ec0a11f2ec8331d313d",
    },
    "terminal-replay-cloud1335-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-terminal-replay-repair1/cloud1335-acceptance-v1.json",
        "sha256": "fe195632dedb22392e60049c1c12e0a488001fe617c12c59b20509f84f9d85b0",
    },
    "terminal-replay-parallel-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-terminal-replay-repair1/parallel-collection-acceptance-v1.json",
        "sha256": "f1b730d6d2d05771b694db738e337e0e6abe0080da0534d62dd09a704c6c30c7",
    },
    "garrison-dialogs-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-garrison-repair1/accepted-runtime.json",
        "sha256": "772444aa025f27208226af18796262cf807d4646c047f4e9f5504cf046ac3e98",
    },
    "terminal-replay-v1": {
        "descriptor": DEFAULT_ARTIFACT_ROOT / "runtime/local-v8-terminal-replay-repair1/accepted-runtime.json",
        "sha256": "fffce8b1ee74c010f235b97a88547adc6ada3dce65251ac959a4b6dbe6e032b8",
    },
}
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
# original engine recipe must still match to use this historical acceptance.
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


def accepted_runtime_descriptor(release_id):
    entry = ACCEPTED_RUNTIME_RELEASES.get(release_id)
    require(entry is not None and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
            "Runtime release has no reviewed acceptance pin: " + str(release_id))
    receipt = pinned(entry["descriptor"], entry["sha256"])
    descriptor = json.loads(Path(receipt["path"]).read_text())
    require(descriptor.get("schema") == RUNTIME_ACCEPTANCE_SCHEMA and descriptor.get("release_id") == release_id
            and descriptor.get("status") == "accepted"
            and descriptor.get("change_classification") in
                ("adapter-support-extension", "replay-verifier-support-extension"),
            "Unsupported or incomplete runtime acceptance descriptor")
    require(descriptor.get("python_environment") == entry.get("python_environment"),
            "Runtime release changed its reviewed Python environment binding")
    return descriptor, {"id": release_id, "descriptor": receipt}


def candidate_runtime_origin(runtime_bin, runtime, descriptor):
    """Check source/freeze/native evidence for isolated candidate diagnostics only.

    The candidate descriptor contains runtime_bin, runtime_files (name->SHA256),
    engine_recipe, freeze/preflight (file bindings), native and regression
    (plan/report/shutdown bindings),
    but grants no machine configuration acceptance. Official preparation also
    requires a reviewed descriptor pin and independent frozen-policy evidence.
    """
    from validation.probes.validate_garrison_dialogs import implementation_identity
    from validation.probes import validate_capabilities as capability
    from tools.validate_garrison_regressions import implementation_identity as regression_identity
    from experiments.explorer.create_garrison_scenarios import CASES, COLORS

    def checked(item):
        require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}, "Incomplete acceptance file binding")
        require(pinned(item["path"], item["sha256"]) == item, "Acceptance evidence binding changed")
        return json.loads(Path(item["path"]).read_text())

    require(Path(descriptor["runtime_bin"]).resolve() == runtime_bin, "Accepted runtime relocation is not permitted")
    pins = descriptor["runtime_files"]
    require(set(pins) == set(RUNTIME_PINS) and runtime == {name: sha for name, sha in pins.items() if name != "vcmiclient"},
            "Accepted runtime module inventory differs")
    for name, sha in pins.items():
        pinned(runtime_bin / name, sha)
    recipe = descriptor["engine_recipe"]
    require(recipe and current_engine_recipe() == recipe, "Current engine recipe differs from accepted runtime")
    freeze = checked(descriptor["freeze"])
    require(freeze.get("schema") == "heroes-local-runtime-freeze-v2" and freeze.get("source_recipe_check") is True
            and freeze.get("donor_load_references_removed") is True and freeze.get("old_runtime_modified") is False
            and freeze.get("training_started") is False and freeze.get("gameplay_started") is False
            and Path(freeze["frozen_bin"]).resolve() == runtime_bin,
            "Accepted runtime freeze is incomplete")
    artifacts = freeze["artifacts"]
    frozen = {item["path"]: item["frozen_sha256"] for item in artifacts}
    require(len(artifacts) == len(pins) + 1 and set(frozen) == set(pins) | {"vcmiserver"}
            and {name: sha for name, sha in frozen.items() if name != "vcmiserver"} == pins,
            "Accepted runtime freeze does not bind exact binary files")
    server = pinned(runtime_bin / "vcmiserver", frozen["vcmiserver"])
    require(server["bytes"] == next(item["bytes"] for item in artifacts if item["path"] == "vcmiserver"),
            "Accepted frozen server identity differs")
    require(all(freeze["source_inputs"].get(name, {}).get("sha256") == sha for name, sha in recipe.items()),
            "Frozen source recipe does not match accepted engine source")
    dependency = pinned(Path(descriptor["freeze"]["path"]).parent / "dependency-check.stdout",
                        freeze["dependency_check"]["sha256"])
    require({key: dependency[key] for key in ("sha256", "bytes")} == freeze["dependency_check"],
            "Frozen dependency check binding changed")
    preflight = checked(descriptor["preflight"])
    native = preflight.get("native", {})
    require(preflight.get("schema") == "heroes-local-runtime-preflight-v2"
            and preflight.get("launcher_preflight_passed") is True and preflight.get("client_help_exit_code") == 0
            and preflight.get("donor_load_detected") is False and preflight.get("gameplay_started") is False
            and preflight.get("training_started") is False and preflight.get("runtime") == runtime
            and native.get("binary") == str(runtime_bin / "vcmiclient")
            and native.get("binary_sha256") == pins["vcmiclient"]
            and native.get("explorer_rl_sha256") == pins["AI/libExplorerRL.dylib"]
            and preflight.get("map") == freeze.get("map")
            and native.get("map", {}).get("sha256") == freeze.get("map", {}).get("sha256")
            and any(str(runtime_bin / "libvcmi.dylib") in line
                    for line in preflight.get("actual_client_libvcmi_load", [])),
            "Frozen runtime loader preflight is incomplete or mismatched")
    game_map = freeze["map"]
    require(pinned(game_map["path"], game_map["sha256"]) == game_map, "Frozen preflight map binding changed")
    external = [pinned(item["path"], item["sha256"]) for item in freeze["external_dependencies"]]
    recorded_resources = [item["path"] for item in freeze["resources"]]
    actual_resources = set()
    for name in ("config", "Mods"):
        folder = runtime_bin / name
        require(folder.is_dir() and not folder.is_symlink(), "Frozen resource directory was replaced")
        for path in folder.rglob("*"):
            require(not path.is_symlink(), "Frozen resource symlink was introduced")
            if path.is_file():
                actual_resources.add(path.relative_to(runtime_bin).as_posix())
    require(len(recorded_resources) == len(set(recorded_resources))
            and set(recorded_resources) == actual_resources, "Frozen runtime resource inventory changed")
    resources = []
    for item in freeze["resources"]:
        path = (runtime_bin / item["path"]).resolve()
        require(path.is_relative_to(runtime_bin), "Accepted runtime resource escapes the frozen directory")
        resources.append(pinned(path, item["sha256"]))
    evidence = descriptor["native"]
    require(set(evidence) == {"plan", "report", "shutdown"}, "Incomplete native acceptance bindings")
    plan, report, shutdown = [checked(evidence[name]) for name in ("plan", "report", "shutdown")]
    require(plan.get("schema") == "heroes-garrison-diagnostic-plan-v1"
            and plan.get("implementation") == implementation_identity()
            and plan.get("native_runtime") == runtime and plan.get("binary_sha256") == pins["vcmiclient"]
            and plan.get("replay_binary_sha256") == pins["vcmireplay"],
            "Native acceptance source/runtime differs from the current release")
    schedule = plan["schedule"]
    required = {(case, player) for case in CASES for player in COLORS}
    require(len(schedule) == len(required) and {(row["case"], row["player"]) for row in schedule} == required
            and len({row["seed"] for row in schedule}) == len(schedule), "Native acceptance coverage is incomplete or reused")
    require(report.get("schema") == "heroes-garrison-native-validation-v1"
            and report.get("plan_sha256") == evidence["plan"]["sha256"] and report.get("status") == "passed"
            and report.get("not_started") == 0 and len(report.get("cases", [])) == len(schedule),
            "Native acceptance did not pass its complete schedule")
    for intended, observed in zip(schedule, report["cases"]):
        require(all(observed.get(key) == intended[key] for key in ("case", "player", "seed"))
                and observed.get("status") == "passed" and observed.get("replay_verified") is True
                and observed.get("mechanics", {}).get("history_audit_passed") is True
                and observed.get("mechanics", {}).get("probe_status") == "passed",
                "Native garrison case lacks exact history and replay acceptance")
    require(all(item.get("training_performed") is False and item.get("optimizer_steps") == 0
                for item in (plan, report, shutdown))
            and plan.get("training_input") is False and report.get("training_input") is False
            and shutdown.get("owned_cleanup_verified") is True and shutdown.get("within_hard_deadline") is True
            and shutdown.get("returncode") == 0 and shutdown.get("errors") == [],
            "Native acceptance training exclusion or owned cleanup is incomplete")

    regression = descriptor.get("regression")
    require(type(regression) is dict and set(regression) == {"plan", "report", "shutdown"},
            "Complete legacy capability regression bindings are required")
    old_plan, old_report, old_shutdown = [checked(regression[name]) for name in ("plan", "report", "shutdown")]
    require(old_plan.get("schema") == "heroes-capability-diagnostic-plan-v1"
            and old_plan.get("interface") == capability.INTERFACE
            and old_plan.get("implementation") == regression_identity()
            and old_plan.get("native_runtime") == runtime
            and old_plan.get("binary_sha256") == pins["vcmiclient"]
            and old_plan.get("replay_binary_sha256") == pins["vcmireplay"],
            "Legacy capability regression source/runtime differs from the current release")
    old_schedule = old_plan.get("schedule")
    require(type(old_schedule) is list and len(old_schedule) == 28
            and type(old_schedule[0]) is dict and type(old_schedule[0].get("seed")) is int
            and 1 <= old_schedule[0]["seed"] <= 2147480000,
            "Legacy capability regression requires its complete 28-case schedule")
    # Clear/blocked Dimension Door deliberately share an engine seed so their
    # only difference is hidden landing occupancy. All other schedule fields
    # and seed offsets must equal the original fixed acceptance harness.
    expected = capability.schedule(tuple(capability.CASES), capability.fixtures.COLORS,
                                   old_schedule[0]["seed"], 1)
    require(old_schedule == expected, "Legacy capability regression schedule changed or reused an unintended seed")
    require(old_report.get("schema") == "heroes-capability-native-validation-v1"
            and old_report.get("plan_sha256") == regression["plan"]["sha256"]
            and old_report.get("status") == "passed" and old_report.get("not_started") == 0
            and type(old_report.get("cases")) is list and len(old_report["cases"]) == len(expected),
            "Legacy capability regression did not pass all 28 cases")
    for intended, observed in zip(expected, old_report["cases"]):
        mechanics = observed.get("mechanics", {})
        require(all(observed.get(key) == value for key, value in intended.items())
                and observed.get("status") == "passed" and observed.get("replay_verified") is True
                and mechanics.get("history_audit_passed") is True and mechanics.get("probe_status") == "passed"
                and mechanics.get("episode_status") in ("completed", "truncated")
                and mechanics.get("training_performed") is False
                and all(check.get("status") == "passed" for check in mechanics.get("checks", [])),
                "Legacy capability regression case lacks complete excluded history/replay acceptance")
    require(all(item.get("training_performed") is False and item.get("training_input") is False
                and item.get("optimizer_steps") == 0 for item in (old_plan, old_report, old_shutdown))
            and old_shutdown.get("owned_cleanup_verified") is True
            and old_shutdown.get("within_hard_deadline") is True and old_shutdown.get("returncode") == 0
            and old_shutdown.get("errors") == [] and old_shutdown.get("cleanup_errors") == [],
            "Legacy capability regression training exclusion or owned cleanup is incomplete")
    return {"receipts": {"freeze": descriptor["freeze"], "preflight": descriptor["preflight"], **evidence},
            "engine_recipe": recipe, "server": server, "dependency_check": dependency,
            "external_dependencies": external, "resources": resources, "regression": regression}


def accepted_runtime_origin(runtime_bin, runtime, descriptor):
    origin = candidate_runtime_origin(runtime_bin, runtime, descriptor)
    if descriptor.get("release_id") == "cpu-next-launch-v1":
        from tools.accept_cpu_next_launch import pipeline_policy_acceptance
        policy = pipeline_policy_acceptance(descriptor, runtime)
        return {**origin, "pipeline_policy": policy,
                "terminal_replay": terminal_replay_acceptance(descriptor, runtime),
                "meaning": "Accepted CPU pipeline and separate Python environment; explicit launch owns fresh acceptance and the original continuation ledger."}
    evidence = descriptor["native"]
    plan = json.loads(Path(evidence["plan"]["path"]).read_text())
    bound = descriptor["policy_reconstruction"]
    require(pinned(bound["path"], bound["sha256"]) == bound, "Policy reconstruction binding changed")
    policy = json.loads(Path(bound["path"]).read_text())
    require(policy.get("schema") == "heroes-garrison-policy-reconstruction-v1" and policy.get("status") == "passed"
            and policy.get("implementation") == plan["implementation"] and policy.get("runtime") == runtime
            and policy.get("native_report_sha256") == evidence["report"]["sha256"]
            and type(policy.get("decisions_checked")) is int and policy["decisions_checked"] > 0
            and policy.get("mismatches") == 0 and policy.get("parameters_unchanged") is True
            and policy.get("exact_model_adam_rng_progress_pool") is True
            and policy.get("training_performed") is False and policy.get("optimizer_steps") == 0,
            "Frozen policy reconstruction/restore acceptance is incomplete")
    require(set(policy.get("checkpoint", {})) == {"plan", "report", "policy", "resume", "initial_checkpoint"}
            and policy.get("evidence"), "Policy acceptance lacks immutable full-checkpoint/evidence bindings")
    for item in [*policy["checkpoint"].values(), *policy["evidence"]]:
        require(pinned(item["path"], item["sha256"]) == item, "Policy acceptance file binding changed")
    attempts = policy.get("attempts", [])
    require(len(attempts) == 8 and sorted(row.get("learner_player") for row in attempts) == [0] * 4 + [1] * 4
            and len({row["engine_seed"] for row in attempts}) == len({row["policy_seed"] for row in attempts}) == 8,
            "Policy acceptance requires eight distinct paired-seat evaluation attempts")
    require(all(row.get("phase") == "evaluation" and row.get("status") == "passed"
                and all(row.get(key) is True for key in ("replay_verified", "identity_verified", "cleanup_verified",
                                                        "completion_receipt", "weights_unchanged"))
                and row.get("training_samples") == 0 and row.get("optimizer_steps") == 0
                and row.get("reconstruction", {}).get("passed") is True for row in attempts),
            "Policy acceptance lacks complete replay/reconstruction without learning")
    terminal = {}
    meaning = "Accepted garrison adapter support extension; explicit launch still owns fresh acceptance and its original continuation ledger."
    if descriptor.get("change_classification") == "replay-verifier-support-extension":
        terminal = {"terminal_replay": terminal_replay_acceptance(descriptor, runtime)}
        meaning = "Accepted runtime repair; explicit launch still owns fresh acceptance and its original continuation ledger."
    return {**origin, "receipts": {**origin["receipts"], "policy_reconstruction": bound},
            **terminal, "meaning": meaning}


def terminal_replay_acceptance(descriptor, runtime):
    """Bind the failed capture's excluded replay to the accepted native verifier."""
    from experiments.headless import builtin_replay
    bound = descriptor.get("terminal_replay")
    require(type(bound) is dict and set(bound) == {"path", "sha256", "bytes"},
            "Terminal replay repair requires the original failure's diagnostic evidence")
    require(pinned(bound["path"], bound["sha256"]) == bound, "Terminal replay diagnostic binding changed")
    proof = json.loads(Path(bound["path"]).read_text())
    identity = proof.get("identity_check", {})
    require(proof.get("schema") == "heroes-contextual-builtin-replay-diagnostic-v1"
            and proof.get("diagnostic_verified") is True and proof.get("verified") is False
            and proof.get("original_cohort_result_unchanged") is True
            and proof.get("original_cohort_admission") is False
            and proof.get("optimization_eligible") is False and proof.get("training_performed") is False
            and proof.get("training_samples") == 0
            and identity.get("mode") == "diagnostic_verifier_only"
            and all(identity.get(key) is True for key in ("client_unchanged", "adapter_sources_unchanged",
                                                          "nonverifier_runtime_unchanged", "nonverifier_recipe_unchanged"))
            and set(identity.get("runtime_changes", {})) == {"vcmireplay"}
            and identity["runtime_changes"]["vcmireplay"]["current"] == runtime["vcmireplay"],
            "Terminal replay repair lacks exact excluded native replay with unchanged original evidence")
    accounting = proof.get("wire", {}).get("request_accounting", {})
    tail = accounting.get("terminal_unapplied", [])
    town_swap = descriptor.get("release_id") in ("terminal-replay-1392-v1", "cpu-next-launch-v1")
    require(not town_swap or descriptor.get("checkpoint_update") == 1392,
            "Town-swap replay acceptance must bind checkpoint1392")
    expected_type = 189 if town_swap else 190
    require(len(tail) == 1 and tail[0].get("pack_type") == expected_type,
            "Terminal replay repair did not verify its required original request type")
    history = proof.get("history", {})
    require(history.get("schema") == "heroes-builtin-history-reconstruction-v1"
            and history.get("verified") is True and history.get("boundary") == "natural"
            and history.get("opponent_history") is None
            and proof.get("effective_engine", {}).get("verified") is True
            and proof.get("wire", {}).get("request_order_verified") is True
            and proof.get("prefix") == {"server_partial_bytes": 0, "server_unpaired_inputs": 0,
                                        "contextual_partial_bytes": 0},
            "Terminal replay repair lacks complete history, effective settings and captured inputs")
    native = proof.get("native", {})
    require(native.get("returncode") == 0 and native.get("verified") is True,
            "Terminal replay repair lacks a successful native verifier result")
    # Recheck complete native coverage and the report's accounting/barrier
    # consistency. The pinned diagnostic above binds the preserved originals.
    barrier = native.get("summary", {}).get("terminal_barrier")
    builtin_replay.validate_native_report(native, proof["wire"],
        {"terminal_barrier": {"client": barrier}}, cutoff=False)
    return bound


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


def build_configuration(input_root, runtime_bin, data, artifact_root, run_root, profile="easier8x8", *, runtime_release=None):
    import torch
    from experiments.headless.contextual_evidence import runtime_identity
    input_root, runtime_bin = [Path(p).expanduser().resolve() for p in (input_root, runtime_bin)]
    descriptor, release_binding = accepted_runtime_descriptor(runtime_release) if runtime_release else (None, None)
    require(runtime_bin == (Path(descriptor["runtime_bin"]).resolve() if descriptor else DEFAULT_RUNTIME.resolve()),
            "This preparation recipe pins its accepted Mac runtime in place; a different build needs a new recipe/gate")
    require(str(torch.__version__) in ("2.9.1", "2.9.1+cpu"), "Use the existing Torch2.9.1 environment")
    python = {"executable": str(Path(sys.executable).absolute()), "version": sys.version,
              "torch": str(torch.__version__)}
    environment = descriptor.get("python_environment") if descriptor else None
    if environment is not None:
        verify_environment(environment, current_python=python)
    parent = {name: pinned(input_root / path, sha) for name, (path, sha) in INPUT_PINS.items()}
    maps = profile_maps(input_root, profile)
    pins = descriptor["runtime_files"] if descriptor else RUNTIME_PINS
    runtime_files = {name: pinned(runtime_bin / name, sha) for name, sha in pins.items()}
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
    require(runtime == {name: sha for name, sha in pins.items() if name != "vcmiclient"},
            "Unexpected native module inventory")
    origin = accepted_runtime_origin(runtime_bin, runtime, descriptor) if descriptor else runtime_origin(runtime_bin, runtime)
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": parent, "maps": maps, "runtime_bin": str(runtime_bin), "runtime": runtime,
        "runtime_files": runtime_files, "runtime_origin": origin,
        **({"runtime_release": release_binding} if release_binding else {}),
        "data": str(data), "artifact_root": str(artifact_root), "run_root": str(run_root),
        "prerequisites": prerequisites,
        "python": python,
        **({"python_environment": environment} if environment is not None else {}),
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
    release_id = value.get("runtime_release", {}).get("id")
    descriptor, _ = accepted_runtime_descriptor(release_id) if release_id else (None, None)
    pins = descriptor["runtime_files"] if descriptor else RUNTIME_PINS
    for name, sha in pins.items():
        require(pinned(Path(value["runtime_bin"]) / name, sha) == value["runtime_files"][name], "Runtime binding changed")
    expected_runtime = Path(descriptor["runtime_bin"]) if descriptor else DEFAULT_RUNTIME
    require(Path(value["runtime_bin"]).resolve() == expected_runtime.resolve(), "Runtime relocation not accepted")
    input_root = Path(value["parent"]["plan"]["path"]).parent.parent
    expected = build_configuration(input_root, value["runtime_bin"], value["data"],
                                   value["artifact_root"], value["run_root"], profile, runtime_release=release_id)
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
    parser.add_argument("--runtime-bin", type=Path, help="Must match the selected accepted runtime in place")
    parser.add_argument("--runtime-release", choices=tuple(ACCEPTED_RUNTIME_RELEASES),
                        help="Explicit reviewed runtime support extension; default retains historical acceptance")
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
        runtime = args.runtime_bin or (Path(accepted_runtime_descriptor(args.runtime_release)[0]["runtime_bin"])
                                      if args.runtime_release else DEFAULT_RUNTIME)
        value = build_configuration(args.input, runtime, args.data, args.artifact_root, run_root, args.profile,
                                    runtime_release=args.runtime_release)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"configuration": str(output), "checkpoint": value["checkpoint_restore"]["progress"],
                      "profile": value["default_profile"], "checked": args.check,
                      "gameplay_performed": False, "ppo_updates": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
