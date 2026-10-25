"""Exact between-batch continuation from a completed native-opponent run.

This module loads no histories and performs no model forward pass or update.
The archived neural pool is serialization state, never an executed opponent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import subprocess

import torch

from experiments.headless.contextual_evidence import runtime_identity, strict_json
from . import checkpoints as cp

ROOT = Path(__file__).resolve().parents[2]
INTERFACE = "adventure-dialogs-v1"
PILOT = "heroes-builtin-ppo-pilot-v1"
CONTINUATION = "heroes-builtin-training-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def _bound(value):
    require(type(value) is dict and set(value) == {"path", "sha256", "bytes"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and type(value["bytes"]) is int and value["bytes"] > 0,
            "Expected an exact absolute artifact binding")
    require(binding(value["path"]) == value, "Bound parent artifact changed: " + value["path"])


def _exact(left, right):
    if isinstance(left, torch.Tensor):
        require(type(right) is type(left) and left.dtype == right.dtype and left.device == right.device
                and torch.equal(left, right), "Restored tensor differs")
    elif isinstance(left, dict):
        require(type(right) is type(left) and left.keys() == right.keys(), "Restored mapping differs")
        for key in left:
            _exact(left[key], right[key])
    elif type(left) in (list, tuple):
        require(type(right) is type(left) and len(left) == len(right), "Restored sequence differs")
        for a, b in zip(left, right):
            _exact(a, b)
    else:
        require(type(left) is type(right) and left == right, "Restored scalar differs")


def _committed_source(source):
    require(type(source) is dict and source.get("tracked_worktree_dirty") is False
            and type(source.get("git_commit")) is str and len(source["git_commit"]) == 40
            and all(v in "0123456789abcdef" for v in source["git_commit"])
            and type(source.get("source_sha256")) is dict and source["source_sha256"],
            "Parent and current learner sources must identify clean committed files")
    for name, digest in source["source_sha256"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and type(digest) is str,
                "Unsafe committed source binding")
        saved = subprocess.run(["git", "show", source["git_commit"] + ":" + name], cwd=ROOT,
                               capture_output=True, check=False)
        require(saved.returncode == 0 and hashlib.sha256(saved.stdout).hexdigest() == digest,
                "Source hash is not committed at its declared revision: " + name)


def transition_manifest():
    path = ROOT / "configs/builtin-resume-transitions-v1.json"
    if not path.exists():
        return {"source_changes": {}, "runtime_changes": []}
    saved = subprocess.check_output(["git", "show", "HEAD:configs/builtin-resume-transitions-v1.json"], cwd=ROOT)
    require(saved == path.read_bytes(), "Resume transition manifest is not exactly committed")
    value = strict_json(saved)
    require(value.get("schema") == "heroes-reviewed-resume-transitions-v1", "Invalid resume transition manifest")
    require(set(value["source_changes"]) <= {"experiments/learner/builtin_training_resume.py",
            "experiments/learner/builtin_training.py"}, "Transition cannot change policy, features or PPO math")
    return value


def compatible_runtime(parent_runtime, parent_native, current_runtime, current_native):
    if parent_runtime == current_runtime and parent_native == current_native:
        return {"unchanged": True}
    differences = {k for k in set(parent_runtime) | set(current_runtime)
                   if parent_runtime.get(k) != current_runtime.get(k)}
    require(differences == {"vcmireplay"} and
            {k:v for k,v in parent_native.items() if k != "binary"} ==
            {k:v for k,v in current_native.items() if k != "binary"},
            "Only an exact reviewed verifier replacement and binary relocation may continue")
    transition = {"from_runtime": parent_runtime, "to_runtime": current_runtime,
                  "from_native": parent_native, "to_native": current_native}
    require(transition in transition_manifest()["runtime_changes"], "Unreviewed native verifier transition")
    return {"unchanged": False, "verified_transition": transition}


def compatible_source(parent, current, migration=None):
    """Every existing learner file stays exact; committed additive files may appear."""
    _committed_source(parent)
    _committed_source(current)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            == current["git_commit"], "Current source revision is not the checkout HEAD")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                       cwd=ROOT, text=True).strip(), "Current tracked checkout is uncommitted")
    if migration:
        migration.source(parent, current)
    else:
        for key in ("torch", "python", "interface"):
            require(parent.get(key) == current.get(key), "Parent runtime/interface changed: " + key)
    from experiments.headless.builtin_evidence import profile_for
    profile_for(current.get("interface"))
    require(parent.get("interface") == current.get("interface"),
            "Schema transition requires a separately verified checkpoint migration")
    approved_changes = {}
    for name, digest in parent["source_sha256"].items():
        actual = current["source_sha256"].get(name)
        if actual != digest:
            changes = migration.value["source_changes"] if migration else transition_manifest()["source_changes"]
            require(changes.get(name) == {"from": digest, "to": actual},
                    "Existing learner source changed without exact review: " + name)
            approved_changes[name] = {"from": digest, "to": actual}
    for name, digest in current["source_sha256"].items():
        require(binding(ROOT / name)["sha256"] == digest, "Current source bytes changed: " + name)
    return {"parent_commit": parent["git_commit"], "current_commit": current["git_commit"],
            "existing_sources_unchanged": not approved_changes, "reviewed_source_changes": approved_changes,
            "added_sources": {name: digest for name, digest in current["source_sha256"].items()
                              if name not in parent["source_sha256"]}}


def _metadata(plan, plan_sha256):
    if plan["schema"] == PILOT + "-plan":
        opponents = plan["native_opponents"]
        require(type(opponents) is list and len(opponents) == 2, "Pilot native opponent identities missing")
        identity = {k: v for k, v in opponents[0].items() if k != "player"}
        require(identity == {k: v for k, v in opponents[1].items() if k != "player"},
                "Pilot opponent modules differ between seats")
        return {**plan["source"], "plan_sha256": plan_sha256, "initial_checkpoint": plan["checkpoint"],
                "native_opponent": identity, "archived_neural_pool_executed": False}
    template = plan.get("checkpoint_metadata")
    require(type(template) is dict and "plan_sha256" not in template
            and all(template.get(k) == v for k, v in plan["source"].items()),
            "Continuation checkpoint metadata must retain source and omit its circular plan hash")
    parent = plan["parent"]
    require(template.get("parent_plan_sha256") == parent["plan"]["sha256"]
            and template.get("parent_report_sha256") == parent["report"]["sha256"]
            and template.get("initial_checkpoint") == parent["policy"],
            "Continuation metadata lost its immediate parent linkage")
    return {**template, "plan_sha256": plan_sha256}


def _completion(plan, report):
    require(plan["schema"] in (PILOT + "-plan", CONTINUATION + "-plan")
            and report.get("schema") == plan["schema"].removesuffix("-plan") + "-result",
            "Unsupported parent training plan/result")
    require(report.get("status") == "finished" and report.get("update_status") == "checkpoint_verified"
            and report.get("worker_complete") is True and report.get("supervisor_cleanup_verified") is True
            and report.get("training_performed") is True and report.get("within_shared_budget") is True,
            "Parent update is incomplete, uncommitted, interrupted or unverified")
    for name in ("optimizer_updates", "optimizer_steps", "training_samples"):
        require(type(report.get(name)) is int and report[name] > 0, "Missing exact committed update counters")
    require(report["optimizer_updates"] == 1, "Parent must contain one completed batched update")
    schedule, rows = plan["schedule"], report.get("attempts")
    require(type(schedule) is list and schedule and type(rows) is list and len(rows) == len(schedule),
            "Parent report lost scheduled attempts")
    training = []
    for expected, row in zip(schedule, rows):
        require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in expected.items()),
                "Parent attempt differs from its fixed schedule")
        require(row.get("status") == "passed" and row.get("completion_receipt") is True
                and row.get("cleanup_verified") is True and row.get("identity_verified") is True
                and row.get("replay_verified") is True, "Parent contains an incomplete or unverified attempt")
        require(type(row.get("training_samples")) is int and row["training_samples"] >= 0,
                "Parent sample counters are missing")
        if row["phase"] == "training":
            require(row["training_samples"] > 0, "Completed training attempt lacks samples")
            training.append(row)
        else:
            require(row["training_samples"] == 0, "Evaluation data cannot supply continuation progress")
    require(sum(row["training_samples"] for row in training) == report["training_samples"],
            "Parent training sample total differs")
    previous, prior_steps = cp.TrainingProgress(), 0
    if plan["schema"] == CONTINUATION + "-plan":
        require(all(row["phase"] == "training" for row in rows) and report.get("parent") == plan.get("parent"),
                "Continuation batch must retain its training-only parent link")
        previous = cp.TrainingProgress(**plan["parent"]["progress"])
        prior_steps = plan["parent"]["optimizer_steps"]
        require(type(prior_steps) is int and prior_steps > 0, "Missing parent Adam step progress")
    progress = cp.TrainingProgress(previous.updates + report["optimizer_updates"],
                                   previous.rollouts + len(training), previous.decisions + report["training_samples"])
    steps = prior_steps + report["optimizer_steps"]
    if plan["schema"] == CONTINUATION + "-plan":
        require(report.get("cumulative_progress") == vars(progress), "Cumulative training progress differs")
        sampler = {"next_schedule_index": len(schedule), "optimizer_steps": steps,
                   "batch_optimizer_steps": report["optimizer_steps"]}
    else:
        sampler = {"next_schedule_index": max(i for i, row in enumerate(schedule) if row["phase"] == "training") + 1,
                   "optimizer_steps": steps}
    return progress, steps, sampler


@dataclass(frozen=True)
class ParentState:
    model: object
    optimizer: object
    generator: object
    pool: cp.FrozenOpponentPool
    progress: cp.TrainingProgress
    sampler_state: dict
    metadata: dict
    checkpoint: str
    weights_sha256: str
    bindings: dict
    optimizer_steps: int


def load_parent(plan_path, report_path, current_source, *, current_runtime, current_native,
                expected_plan_sha256, expected_report_sha256, migration=None):
    """Restore exact committed parent state; reject drift before returning anything.

    Success restores saved global Torch/Python and named CPU minibatch RNGs.
    Failure preserves the caller's RNGs. The returned model is frozen for fresh
    collection; its Adam instance still owns every parameter and is never reset.
    """
    caller_torch, caller_python = torch.get_rng_state().clone(), random.getstate()
    try:
        plan_binding, report_binding = binding(plan_path), binding(report_path)
        require(plan_binding["sha256"] == expected_plan_sha256 and report_binding["sha256"] == expected_report_sha256,
                "Parent plan/report hash differs from the new plan's explicit binding")
        plan, report = strict_json(Path(plan_binding["path"]).read_bytes()), strict_json(Path(report_binding["path"]).read_bytes())
        require(report.get("plan_sha256") == expected_plan_sha256
                and Path(report_binding["path"]) == (migration.resolve(plan["execution_output"]) if migration else Path(plan["execution_output"])) / "report.json",
                "Parent completion report belongs to another plan/output")
        progress, steps, sampler = _completion(plan, report)
        source_proof = compatible_source(plan["source"], current_source, migration)
        runtime_proof = (migration.runtime if migration else compatible_runtime)(
            plan["runtime"], plan["native"], current_runtime, current_native)
        resolve = migration.resolve if migration else Path
        artifact = migration.artifact if migration else lambda value: (_bound(value) or value)
        require(runtime_identity(Path(current_native["binary"]).parent) == current_runtime
                and binding(current_native["binary"])["sha256"] == current_native["binary_sha256"]
                and binding(resolve(plan["game_map"]))["sha256"] == current_native["map"]["sha256"],
                "Actual native runtime or map changed")
        metadata = _metadata(plan, expected_plan_sha256)
        artifact(metadata["initial_checkpoint"])
        saved = report["after_checkpoint"]
        require(saved.get("exact_model_adam_rng_progress_pool") is True,
                "Parent lacks an exact checkpoint completion receipt")
        if migration:
            migration.parent(plan_binding, report_binding, saved, progress, steps)
        for kind in ("policy", "resume"):
            artifact(saved[kind])
            require(Path(saved[kind]["path"]).parent == Path(plan["execution_output"]),
                    "Parent checkpoint is outside its completed execution")
        options = migration.checkpoint_options if migration else {}
        policy = cp.load_policy_checkpoint(resolve(saved["policy"]["path"]), expected_source_metadata=metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=plan["source"]["interface"], **options)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **plan["configuration"]["adam"])
        generator = torch.Generator(device="cpu")
        resumed = cp.load_checkpoint(resolve(saved["resume"]["path"]), model, optimizer,
            expected_ppo_config=plan["configuration"]["ppo"], expected_source_metadata=metadata,
            generators={"minibatch": generator}, interface=plan["source"]["interface"],
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), at_rollout_boundary=True, **options)
        raw = torch.load(resolve(saved["resume"]["path"]), map_location="cpu", weights_only=True)
        require(resumed.checkpoint == policy.checkpoint and resumed.progress == progress
                and resumed.sampler_state == sampler and policy.weights_sha256 == saved["weights_sha256"]
                and cp.model_weight_digest(model) == saved["weights_sha256"], "Parent checkpoint pair/progress differs")
        _exact(dict(model.state_dict()), raw["model_state"])
        _exact(optimizer.state_dict(), raw["resume"]["optimizer_state"])
        _exact(resumed.opponent_pool.state_dict(), policy.opponent_pool.state_dict())
        _exact(torch.get_rng_state(), raw["rng"]["torch_cpu"])
        _exact(random.getstate(), raw["rng"]["python"])
        _exact(generator.get_state(), raw["rng"]["torch_generators"]["minibatch"])
        require(optimizer.state and any(value["step"].item() > 0 for value in optimizer.state.values()),
                "Parent Adam moments are absent or reset")
        require(binding(plan_path) == plan_binding and binding(report_path) == report_binding,
                "Parent plan/report changed while restoring")
        for kind in ("policy", "resume"):
            artifact(saved[kind])
        model.eval().requires_grad_(False)
        links = {"plan": plan_binding, "report": report_binding, "policy": artifact(saved["policy"]), "resume": artifact(saved["resume"]),
                 "progress": vars(progress), "optimizer_steps": steps, "source_proof": source_proof, "runtime_proof": runtime_proof}
        return ParentState(model, optimizer, generator, resumed.opponent_pool, progress, sampler,
                           metadata, resumed.checkpoint, saved["weights_sha256"], links, steps)
    except BaseException:
        torch.set_rng_state(caller_torch)
        random.setstate(caller_python)
        raise
