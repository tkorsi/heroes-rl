"""One authorized v7 PPO update against fixed Nullkiller Easy on CPU.

Preparation freezes the exact schedule and runtime. One guarded worker keeps
fresh admission receipts in memory through the update. Diagnostic files cannot
be resumed as training inputs. There is no retry or execution-resume command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .evaluate_builtin_opponent import (ROOT, INTERFACE, COLORS, binding, digest,
    write, require, native_opponent, load_fixed, require_choice_budget)

CONFIG = ROOT / "configs/builtin-ppo-pilot-v1.json"
SCHEMA = "heroes-builtin-ppo-pilot-v1"
CHECKPOINT_SHA = "4043b58bb3dc4eb6169514ca1ed7c1be0b373a5139efe04b95dfeda1832232c1"
AUTHORIZATION = {
    "source_thread": "01a16d9e-5f8d-7cb0-a939-3ff005a741e4",
    "verified_direct_human_instruction": True,
    "date": "2026-10-25",
    "purpose": "New recovery experiment from the exact surviving original baseline after loss of the trained pilot artifacts; not a reconstruction of the historical pilot.",
    "continuation": "Successive bounded local batches and held-out evaluation authorized after this verified recovery batch.",
    "durability": "Canonical run artifacts live outside managed worktrees in /Users/orsi/Projects/Heroes-artifacts.",
}


def config():
    cfg = json.loads(CONFIG.read_text())
    expected = {
        "schema": "heroes-builtin-ppo-pilot-config-v1", "interface": INTERFACE,
        "shared_wall_seconds": 3600,
        "phase_deadlines_seconds": {"before": 900, "training": 2100, "update": 3000, "after": 3540},
        "attempt_wall_seconds": 240, "cleanup_reserve_seconds": 30, "hard_stop_reserve_seconds": 2,
        "ordinary_choice_limit": 248, "reserved_dialog_choices": 8, "max_total_choices": 256,
        "max_days": 30, "max_updates": 1, "allow_verified_training_prefix": True,
        "max_optimizer_steps": 32, "torch_threads": 1,
        "evaluation_engine_seed": 160101, "evaluation_policy_seeds": [160201, 160202],
        "training_engine_seeds": [160111, 160112], "training_policy_seeds": [160211, 160212, 160213, 160214],
        "torch_seed": 160401, "python_seed": 160402, "minibatch_seed": 160403,
        "adam": {"lr": 0.0003},
        "ppo": {"gamma": 1.0, "gae_lambda": 1.0, "clip_ratio": 0.2, "value_clip": 0.2,
            "value_coefficient": 0.5, "entropy_coefficient": 0.0, "max_grad_norm": 0.5,
            "epochs": 1, "decisions_per_window": 16, "windows_per_minibatch": 2,
            "normalize_advantages": True},
    }
    require(json.dumps(cfg, sort_keys=True) == json.dumps(expected, sort_keys=True),
            "The versioned pilot scope cannot be expanded by invocation flags")
    return cfg


def schedule(cfg):
    rows = []
    for phase in ("before", "training", "after"):
        seeds = cfg["training_engine_seeds"] if phase == "training" else [cfg["evaluation_engine_seed"]]
        policies = cfg["training_policy_seeds"] if phase == "training" else cfg["evaluation_policy_seeds"]
        for i, seed in enumerate(seeds):
            for seat in (0, 1):
                rows.append({"episode": len(rows) + 1, "phase": phase, "engine_seed": seed,
                    "learner_player": seat, "policy_seed": policies[2 * i + seat]})
    return rows


def implementation():
    from .first_run import implementation_identity
    result = implementation_identity()
    result[str(CONFIG.relative_to(ROOT))] = digest(CONFIG)
    return result


def starting_source_compatibility(saved, current):
    """An explicit checkpoint-admission extension, with unchanged model math."""
    for key in ("torch", "python", "interface"):
        require(saved[key] == current[key], "Starting runtime/interface changed: " + key)
    changes = {}
    for name, previous in saved["source_sha256"].items():
        actual = current["source_sha256"].get(name)
        if previous != actual:
            require(name == "experiments/learner/checkpoints.py" and actual is not None,
                    "Unreviewed starting learner source change: " + name)
            changes[name] = {"original": previous, "current": actual,
                "reason": "Additive v7 native-opponent optimizer provenance; original policy/model path retained"}
    return changes


def prerequisites():
    """Bind established natural endings and the separately repaired verifier."""
    from .builtin_preservation import DURABLE_ROOT
    paths = [DURABLE_ROOT / "inputs/validation/local-builtin-revalidation-v2-2026-10-25.json",
             DURABLE_ROOT / "inputs/validation/build-terminal-diagnostic-2026-10-25.json"]
    expected = ["3515928087fdbafc4c3e3686ae370ed1be992fbd50faaed70475255c7113a09f",
                "b67887eb2bb972c07b48563b080b5fbc7471aa52e27b29cf7a61e6edacbbe849"]
    require([digest(p) for p in paths] == expected, "Reviewed natural-outcome evidence changed")
    return [binding(p) for p in paths]


def prepare(args):
    import torch
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    from .inference import source_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    cfg = config()
    from .builtin_preservation import durable_path
    output = durable_path(args.output)
    durable_path(args.map)
    durable_path(args.checkpoint)
    durable_path(args.binary)
    durable_path(args.replay_binary)
    require(not output.exists(), "Use a new preparation folder")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit and review the complete implementation before freezing the plan")
    torch.set_num_threads(1)
    loaded, fixed = load_fixed(args.checkpoint.resolve())
    source = source_identity(INTERFACE)
    compatibility = starting_source_compatibility(loaded.source_metadata, source)
    launcher = VcmiLauncher(args.binary.resolve(), args.data.resolve(), args.map.resolve())
    launcher.preflight()
    native = launcher.identity()
    runtime = runtime_identity(args.binary.resolve().parent)
    scenario = hero_map_identity(args.map)
    require(scenario["sha256"] == "852ea0211d20e1cbfb64790fe8076254e98983ba0ae29e854db192c816c76ad1",
            "Use the unchanged ordinary v7 scenario")
    checked = subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    plan = {"schema": SCHEMA + "-plan", "configuration": cfg, "schedule": schedule(cfg),
        "execution_output": str(output / "execution"), "checkpoint": binding(args.checkpoint),
        "checkpoint_source": loaded.source_metadata, "frozen": fixed, "source": source,
        "starting_source_compatibility": compatibility, "implementation": implementation(),
        "native": native, "runtime": runtime, "map": scenario,
        "binary": str(args.binary.resolve()), "data": str(args.data.resolve()), "game_map": str(args.map.resolve()),
        "replay_binary": binding(args.replay_binary),
        "native_opponents": [native_opponent(runtime, seat) for seat in (0, 1)],
        "prerequisites": prerequisites(), "authorization": AUTHORIZATION, "prepared_check": checked.stdout.strip(),
        "failure_policy": "No retries or replacements. Any attempted-game correctness/finalization failure aborts the pilot. A phase deadline before a new training attempt permits one update from its verified fresh prefix; no failed capture is admitted.",
        "wall_scope": "One shared 3600-second clock includes initialization, all games, audits/replay, one update, checkpoint verification and cleanup. After evaluations have the final 600 seconds, with the last 60 seconds reserved for cleanup/reporting.",
        "comparison_scope": "Two matched engine/policy-seed and seat pairs; separate training seeds. Nullkiller randomness is not fully seed-reproducible. This tiny sample cannot establish strategic improvement.",
        "replay_scope": "Regenerate engine effects from captured native-opponent commands; do not claim to regenerate Nullkiller decisions.",
        "archived_neural_pool_executed": False}
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "sha256": digest(output / "plan.json")}


def read_plan(path, sha):
    path = Path(path).resolve()
    require(digest(path) == sha, "Immutable plan hash changed")
    plan = json.loads(path.read_text())
    cfg = config()
    require(plan["schema"] == SCHEMA + "-plan" and plan["authorization"] == AUTHORIZATION
            and json.dumps(plan["configuration"], sort_keys=True) == json.dumps(cfg, sort_keys=True)
            and json.dumps(plan["schedule"], sort_keys=True) == json.dumps(schedule(cfg), sort_keys=True)
            and plan["execution_output"] == str(path.parent / "execution"), "Plan scope or schedule changed")
    require(plan["checkpoint"]["sha256"] == CHECKPOINT_SHA, "Starting checkpoint changed")
    return plan


def check_identity(plan):
    if plan.get("scope") == "heroes-update146-curriculum-v2":
        from .curriculum_training import check_identity as check_curriculum
        return check_curriculum(plan)
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    from .inference import source_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    from experiments.headless.builtin_evidence import profile_for
    interface = profile_for(plan["source"]["interface"]).environment
    if interface == "adventure-capabilities-v1":
        from .builtin_v8_migration import map_identity as hero_map_identity
    require(source_identity(interface) == plan["source"] and implementation() == plan["implementation"],
            "Prepared source or implementation changed")
    require(prerequisites() == plan["prerequisites"] and binding(plan["checkpoint"]["path"]) == plan["checkpoint"]
            and binding(plan["replay_binary"]["path"]) == plan["replay_binary"], "Bound evidence/artifact changed")
    launcher = VcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]))
    require(launcher.identity() == plan["native"] and runtime_identity(Path(plan["binary"]).parent) == plan["runtime"]
            and hero_map_identity(plan["game_map"]) == plan["map"], "Native runtime or scenario changed")


class Budget:
    def __init__(self, deadline, wall_deadline):
        self.deadline, self.wall_deadline = deadline, wall_deadline

    def remaining(self, reserve=0):
        seconds = min(self.deadline - time.monotonic(), self.wall_deadline - time.time()) - reserve
        if seconds <= 0:
            raise TimeoutError("Shared pilot/phase budget exhausted")
        return seconds

    def capped(self, seconds):
        return Budget(min(self.deadline, time.monotonic() + seconds), min(self.wall_deadline, time.time() + seconds))


def empty_attempt(row):
    return {**row, "status": "not_started", "reason": "pending_schedule", "outcomes": dict.fromkeys(COLORS),
        "rewards": dict.fromkeys(COLORS), "training_samples": 0, "optimizer_steps": 0, "completion_receipt": False}


def run_attempt(plan, plan_path, sha, index, model, checkpoint, budget, save):
    from experiments.headless.env import Limits
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.headless.builtin_replay import verify_episode
    from .builtin_rollout import BuiltinCollector, audit_persisted_builtin_policy, retain_failed_builtin_evidence
    from .builtin_gameplay import TrainingBinding, TrainingCollector, verify_training_capture
    from .builtin_behavior import analyze_episode
    from .contextual_model import weights_sha256
    from validation.probes.validate_hero_policy import wall_limit
    from experiments.headless.builtin_evidence import profile_for
    interface = profile_for(plan["source"]["interface"]).environment
    cfg, row = plan["configuration"], plan["schedule"][index]
    output, seat = Path(plan["execution_output"]), row["learner_player"]
    folder = output / f"episode-{row['episode']:06d}"
    attempt = {**empty_attempt(row), "status": "running", "reason": None, "stage": "initialization"}
    env, launcher, collector, receipt = None, None, None, None
    started, before = time.monotonic(), weights_sha256(model)
    def progress(stage):
        attempt.update(stage=stage, wall_seconds=time.monotonic() - started)
        save(attempt)
    try:
        with wall_limit(budget.remaining(cfg["cleanup_reserve_seconds"])):
            check_identity(plan)
            opponent = native_opponent(plan["runtime"], seat)
            launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]),
                deadline_monotonic=budget.deadline - 2, deadline_wall=budget.wall_deadline - 2,
                evidence_path=output / f"native-deadline-{row['episode']:02d}.jsonl")
            if row["phase"] == "training":
                bound = TrainingBinding.create(plan_path=plan_path, plan_sha256=sha, schedule_index=index,
                    folder=folder, checkpoint=checkpoint, checkpoint_file=plan["checkpoint"]["path"],
                    weights_sha256=before, opponent_identity=opponent, source_identity=plan["source"],
                    native_identity=plan["native"], max_days=cfg["max_days"],
                    nominal_choices=cfg["ordinary_choice_limit"], max_choices=cfg["max_total_choices"])
                collector = TrainingCollector(model, bound, interface=interface)
            else:
                class BoundedCollector(BuiltinCollector):
                    sampled = 0
                    def act(self, request):
                        require_choice_budget(self.sampled, request.record["day"], cfg)
                        result = super().act(request)
                        self.sampled += 1
                        return result
                collector = BoundedCollector(model, checkpoint, learner_player=seat,
                    opponent_identity=opponent, policy_seed=row["policy_seed"], interface=interface)
            limits = Limits(max_days=cfg["max_days"], max_actions=cfg["max_total_choices"],
                watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30)
            env = BuiltinHeadlessEnv(launcher, output, learner_player=seat, limits=limits, ledger=True,
                replay=True, interface=interface, agents={COLORS[seat]: {"checkpoint": checkpoint,
                    "kind": "recurrent-policy", "evaluation_only": row["phase"] != "training"}})
            progress("collection")
            collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                max_requests=cfg["ordinary_choice_limit"])
            manifest = json.loads((folder / "manifest.json").read_text())
            experience = collected.experience if row["phase"] == "training" else collected
            attempt.update(episode_directory=str(folder), episode_status=manifest["status"],
                termination_reason=manifest["termination_reason"], sampled_choices=len(experience.decisions),
                opponent_identity=opponent, checkpoint=checkpoint, weights_sha256=before,
                environment_limits=asdict(limits), external_player_histories=1)
            require(manifest["status"] in ("completed", "truncated"), "Unscored technical episode failure")
            require(len(experience.decisions) <= cfg["max_total_choices"], "Choice cap exceeded")
            progress("verification")
            if row["phase"] == "training":
                receipt = verify_training_capture(folder, collected, bound, model,
                    replay_binary=Path(plan["replay_binary"]["path"]), data=Path(plan["data"]),
                    remaining=lambda: budget.remaining(cfg["cleanup_reserve_seconds"]))
            else:
                write(folder / "builtin-rollout.json", collected.to_dict())
                reconstructed = audit_persisted_builtin_policy(folder, model, checkpoint, row["policy_seed"], opponent)
                write(folder / "policy-reconstruction.json", reconstructed)
                replay = verify_episode(folder, binary=Path(plan["replay_binary"]["path"]), data=Path(plan["data"]),
                    timeout=budget.remaining(cfg["cleanup_reserve_seconds"]))
                require(replay.get("verified") is True, "Independent native replay failed")
                attempt["reconstruction"] = reconstructed
            require(weights_sha256(model) == before, "Policy changed during collection/verification")
            behavior = analyze_episode(folder)
            write(folder / "behavior.json", behavior)
            attempt.update(status="pending_verification", replay_verified=True,
                behavior=binding(folder / "behavior.json"), weights_unchanged=True)
    except Exception as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        receipt = None
        try:
            retain_failed_builtin_evidence(env, collector, attempt, error, cleanup_seconds=min(10, budget.remaining()))
        except Exception as secondary:
            attempt["failure_retention_error"] = str(secondary)
    finally:
        try:
            progress("cleanup")
        except Exception as error:
            attempt.update(status="failed", progress_error=str(error))
        cleanup_errors = []
        try:
            with wall_limit(min(10, budget.remaining())):
                if env is not None:
                    env.close()
        except Exception as error:
            cleanup_errors.append(f"Environment cleanup: {type(error).__name__}: {error}")
        try:
            with wall_limit(min(10, budget.remaining())):
                if launcher is not None:
                    launcher.close_owned_groups(timeout=min(3, budget.remaining()))
        except Exception as error:
            cleanup_errors.append(f"Native group cleanup: {type(error).__name__}: {error}")
        try:
            with wall_limit(min(10, budget.remaining())):
                check_identity(plan)
            require(not cleanup_errors, "; ".join(cleanup_errors))
            attempt.update(cleanup_verified=True, identity_verified=True)
            if attempt["status"] == "pending_verification":
                attempt.update(status="passed", outcomes=manifest["outcomes_by_player"],
                    rewards=manifest["rewards_by_player"], completion_receipt=True)
        except Exception as error:
            attempt.update(status="failed", cleanup_or_identity_error=f"{type(error).__name__}: {error}")
        if attempt["status"] != "passed":
            receipt = None
            attempt.update(outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS), completion_receipt=False)
        progress("finished")
    return attempt, receipt


def checkpoint_roundtrip(output, model, optimizer, pool, generator, metadata, cfg, progress, sampler, name):
    import torch
    from .checkpoints import (BUILTIN_GAMEPLAY_PROVENANCE, save_checkpoint, load_checkpoint,
        save_policy_checkpoint, load_policy_checkpoint)
    from .contextual_model import ContextualActorCritic, weights_sha256
    from .gameplay import assert_same_state
    from experiments.headless.builtin_evidence import profile_for
    interface = profile_for(cfg["interface"]).environment
    checkpoint = "builtin-pilot-" + name + "-" + weights_sha256(model)[:16]
    before = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
        "torch": torch.get_rng_state().clone(), "python": random.getstate(), "minibatch": generator.get_state().clone()}
    resume_path, policy_path = output / (name + "-resume.pt"), output / (name + "-policy.pt")
    kwargs = {"checkpoint": checkpoint, "source_metadata": metadata, "opponent_pool": pool,
        "generators": {"minibatch": generator}, "interface": interface, "provenance": BUILTIN_GAMEPLAY_PROVENANCE}
    flags = tuple(parameter.requires_grad for parameter in model.parameters())
    try:
        model.requires_grad_(True)
        save_checkpoint(resume_path, model, optimizer, ppo_config=cfg["ppo"], progress=progress,
            sampler_state=sampler, at_rollout_boundary=True, **kwargs)
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
    save_policy_checkpoint(policy_path, model, **kwargs)
    restored = ContextualActorCritic(config=model.config)
    restored_optimizer = torch.optim.Adam(restored.parameters(), **cfg["adam"])
    restored_generator = torch.Generator()
    state = load_checkpoint(resume_path, restored, restored_optimizer, expected_ppo_config=cfg["ppo"],
        expected_source_metadata=metadata, generators={"minibatch": restored_generator},
        interface=interface, allowed_provenance=(BUILTIN_GAMEPLAY_PROVENANCE,), at_rollout_boundary=True)
    assert_same_state(restored.state_dict(), before["model"])
    assert_same_state(restored_optimizer.state_dict(), before["optimizer"])
    require(torch.equal(torch.get_rng_state(), before["torch"]) and random.getstate() == before["python"]
            and torch.equal(restored_generator.get_state(), before["minibatch"]), "Checkpoint RNG restore differed")
    require(state.progress == progress and state.sampler_state == sampler and state.opponent_pool.identities == pool.identities,
            "Checkpoint progress/sampler/archived pool changed")
    loaded = load_policy_checkpoint(policy_path, expected_source_metadata=metadata,
        allowed_provenance=(BUILTIN_GAMEPLAY_PROVENANCE,), interface=interface)
    assert_same_state(loaded.model.state_dict(), restored.state_dict())
    loaded.model.eval().requires_grad_(False)
    return loaded, {"resume": binding(resume_path), "policy": binding(policy_path),
        "weights_sha256": loaded.weights_sha256, "exact_model_adam_rng_progress_pool": True}


def worker(args):
    import torch
    from .builtin_gameplay import native_update, UpdateRolledBack
    from .checkpoints import TrainingProgress
    from .contextual_model import weights_sha256
    from .ppo import PPOConfig
    from validation.probes.validate_hero_policy import wall_limit
    torch.set_num_threads(1)
    plan = read_plan(args.plan, args.sha256)
    output, cfg = Path(plan["execution_output"]), plan["configuration"]
    report = json.loads((output / "report.json").read_text())
    def save():
        report["execution_wall_seconds"] = time.monotonic() - args.started
        write(output / "worker-report.json", report)
    def phase_budget(phase):
        offset = cfg["phase_deadlines_seconds"][phase]
        return Budget(args.started + offset, args.wall_started + offset)
    try:
        check_identity(plan)
        loaded, fixed = load_fixed(plan["checkpoint"]["path"], plan["checkpoint_source"])
        require(fixed == plan["frozen"], "Initial model identity changed")
        torch.manual_seed(cfg["torch_seed"])
        random.seed(cfg["python_seed"])
        generator = torch.Generator().manual_seed(cfg["minibatch_seed"])
        model = loaded.model
        model.eval().requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), **cfg["adam"])
        metadata = {**plan["source"], "plan_sha256": args.sha256, "initial_checkpoint": plan["checkpoint"],
            "native_opponent": {k: v for k, v in plan["native_opponents"][0].items() if k != "player"},
            "archived_neural_pool_executed": False}
        with wall_limit(phase_budget("before").remaining(cfg["cleanup_reserve_seconds"])):
            _, report["before_checkpoint"] = checkpoint_roundtrip(output, model, optimizer, loaded.opponent_pool,
                generator, metadata, cfg, TrainingProgress(), {"next_schedule_index": 0, "optimizer_steps": 0}, "before")
        receipts = []
        for phase, indices in (("before", (0, 1)), ("training", (2, 3, 4, 5))):
            for index in indices:
                budget = phase_budget(phase)
                if budget.remaining() <= 60:
                    for pending in indices:
                        if report["attempts"][pending]["status"] == "not_started":
                            report["attempts"][pending]["reason"] = "phase_budget_reserved"
                    require(phase == "training" and receipts, "No complete pre-evaluation or fresh collection prefix")
                    break
                def save_attempt(row, index=index):
                    report["attempts"][index] = row
                    save()
                attempt, receipt = run_attempt(plan, args.plan, args.sha256, index, model, loaded.checkpoint,
                    budget.capped(cfg["attempt_wall_seconds"]), save_attempt)
                require(attempt["status"] == "passed", "Attempt failed; remaining schedule stopped")
                if phase == "training":
                    require(receipt is not None, "Fresh verified receipt is missing")
                    receipts.append(receipt)
        update_budget = phase_budget("update")
        report["update_status"] = "started"
        save()
        model.train().requires_grad_(True)
        try:
            with wall_limit(update_budget.remaining(cfg["cleanup_reserve_seconds"])):
                result, rollouts = native_update(model, optimizer, tuple(receipts), plan_sha256=args.sha256,
                    config=PPOConfig(**cfg["ppo"]), generator=generator,
                    remaining=lambda: update_budget.remaining(cfg["cleanup_reserve_seconds"]),
                    max_optimizer_steps=cfg["max_optimizer_steps"])
            report.update(update_status="completed", training_performed=True, optimizer_updates=1,
                optimizer_steps=result.optimizer_steps, training_samples=result.samples, ppo=asdict(result))
            for index, rollout in zip((2, 3, 4, 5), rollouts):
                report["attempts"][index]["training_samples"] = len(rollout.samples)
            save()
        except UpdateRolledBack as error:
            report.update(update_status="rolled_back", attempted_optimizer_steps=error.completed_steps,
                training_performed=False, optimizer_updates=0, optimizer_steps=0)
            raise
        model.eval().requires_grad_(False)
        require(weights_sha256(model) != fixed["learner_weights_sha256"], "Update did not change any model weights")
        with wall_limit(update_budget.remaining()):
            after, report["after_checkpoint"] = checkpoint_roundtrip(output, model, optimizer, loaded.opponent_pool,
                generator, metadata, cfg, TrainingProgress(1, len(receipts), result.samples),
                {"next_schedule_index": 6, "optimizer_steps": result.optimizer_steps}, "after")
        report["update_status"] = "checkpoint_verified"
        save()
        for index in (6, 7):
            budget = phase_budget("after")
            require(budget.remaining() > 60, "Post-update execution reserve exhausted")
            def save_attempt(row, index=index):
                report["attempts"][index] = row
                save()
            attempt, _ = run_attempt(plan, args.plan, args.sha256, index, after.model, after.checkpoint,
                budget.capped(cfg["attempt_wall_seconds"]), save_attempt)
            require(attempt["status"] == "passed", "Post-update attempt failed; no replacement")
        require(binding(plan["checkpoint"]["path"]) == plan["checkpoint"], "Original checkpoint changed")
        require(after.opponent_pool.identities == loaded.opponent_pool.identities, "Archived neural pool changed")
        check_identity(plan)
        report.update(status="finished", stop_reason=None)
    except Exception as error:
        report.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}")
        if report.get("update_status") == "started":
            report.update(update_status="interrupted_unknown", optimizer_updates=None,
                optimizer_steps=None, training_performed=None)
    for row in report["attempts"]:
        if row["status"] == "not_started" and row["reason"] == "pending_schedule":
            row["reason"] = report.get("stop_reason") or "phase_budget_reserved"
    report["worker_complete"] = True
    save()
    return report


def read_worker_report(path, plan, sha):
    report = json.loads(Path(path).read_text())
    require(report["plan_sha256"] == sha and len(report["attempts"]) == len(plan["schedule"])
            and all(all(json.dumps(row[k], sort_keys=True) == json.dumps(wanted[k], sort_keys=True) for k in wanted)
                    for row, wanted in zip(report["attempts"], plan["schedule"])), "Worker report identity changed")
    require(all(row.get("completion_receipt") is True and row.get("cleanup_verified") is True
                and row.get("identity_verified") is True and row.get("replay_verified") is True
                for row in report["attempts"] if row["status"] == "passed"),
            "Passed attempt lacks durable post-verification completion")
    return report


def execute(args, *, plan_reader=None, worker_module="experiments.learner.builtin_first_run"):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    plan = (plan_reader or read_plan)(args.plan, args.sha256)
    output, cfg = Path(plan["execution_output"]), plan["configuration"]
    output.mkdir(exist_ok=False)
    started, wall_started = time.monotonic(), time.time()
    budget = Budget(started + cfg["shared_wall_seconds"],
        min(wall_started + cfg["shared_wall_seconds"], plan.get("compute_deadline_unix", float("inf"))))
    report = {"schema": plan["schema"].removesuffix("-plan") + "-result", "plan_sha256": args.sha256, "status": "running",
        "attempts": [empty_attempt(row) for row in plan["schedule"]], "training_performed": False,
        "optimizer_updates": 0, "optimizer_steps": 0, "training_samples": 0, "update_status": "not_started",
        "started_unix": wall_started, "deadline_unix": budget.wall_deadline, "worker_complete": False,
        "wall_scope": plan["wall_scope"], "comparison_scope": plan["comparison_scope"]}
    write(output / "report.json", report)
    command = [sys.executable, "-m", worker_module, "worker",
        "--plan", str(args.plan.resolve()), "--sha256", args.sha256,
        "--started", str(started), "--wall-started", str(wall_started)]
    process, signalled = None, False
    try:
        with (output / "worker.log").open("xb") as log:
            process = guarded_popen(command, deadline_monotonic=budget.deadline - 2,
                deadline_wall=budget.wall_deadline - 2, evidence_path=output / "worker-deadline.jsonl",
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                left = budget.remaining()
                if left <= 30 and not signalled:
                    os.kill(process.pid, signal.SIGTERM)
                    signalled = True
                if left <= 3:
                    break
                time.sleep(min(0.1, left - 3))
        worker_path = output / "worker-report.json"
        if worker_path.exists():
            report = read_worker_report(worker_path, plan, args.sha256)
        require(process.poll() == 0 and report.get("worker_complete") is True, "Worker did not finish durably")
    except Exception as error:
        # A timeout or signal failure can happen before the normal read above.
        # Preserve the last durable update journal instead of reporting zero.
        try:
            if (output / "worker-report.json").exists():
                report = read_worker_report(output / "worker-report.json", plan, args.sha256)
        except Exception as read_error:
            report.update(worker_report_error=f"{type(read_error).__name__}: {read_error}",
                update_status="unverified_worker_state", optimizer_updates=None, optimizer_steps=None,
                training_performed=None)
        report.update(status="stopped", supervisor_error=f"{type(error).__name__}: {error}")
        if report.get("update_status") == "started":
            report.update(update_status="interrupted_unknown", optimizer_updates=None, optimizer_steps=None,
                training_performed=None)
        for row in report["attempts"]:
            if row["status"] in ("running", "pending_verification") or not row.get("completion_receipt") and row["status"] == "passed":
                row.update(status="incomplete", outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS), training_samples=0)
            if row["status"] == "not_started":
                row["reason"] = "worker_interrupted"
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=min(2, budget.remaining()))
            closed = []
            for path in sorted(output.glob("native-deadline-*.jsonl")):
                closed.extend(cleanup_from_evidence(path, timeout=min(1, budget.remaining())))
            report.update(supervisor_cleanup_verified=True, native_groups_closed=closed)
        except Exception as error:
            report.update(status="stopped", supervisor_cleanup_verified=False, cleanup_error=f"{type(error).__name__}: {error}")
        report.update(execution_wall_seconds=time.monotonic() - started,
            execution_wall_clock_seconds=time.time() - wall_started)
        report["within_shared_budget"] = max(report["execution_wall_seconds"], report["execution_wall_clock_seconds"]) <= cfg["shared_wall_seconds"]
        write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("output", "checkpoint", "binary", "replay-binary", "data", "map"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("execute", "worker"):
        entry = sub.add_parser(name)
        entry.add_argument("--plan", type=Path, required=True)
        entry.add_argument("--sha256", required=True)
        if name == "worker":
            entry.add_argument("--started", type=float, required=True)
            entry.add_argument("--wall-started", type=float, required=True)
    args = parser.parse_args()
    result = {"prepare": prepare, "execute": execute, "worker": worker}[args.command](args)
    print(json.dumps(result if args.command == "prepare" else
        {key: result.get(key) for key in ("status", "stop_reason", "update_status", "optimizer_updates", "optimizer_steps", "training_samples", "execution_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
