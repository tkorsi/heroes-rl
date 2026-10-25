"""Bounded v8 mixed-controller startup and terminal replay checks on the ordinary map.

Both seats use the supplied unchanged migrated policy against actual Nullkiller
Easy. No optimizer is constructed, no samples are admitted, and prior failed
startup attempts remain separate. Both native terminal barriers must complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.headless.contextual_evidence import strict_json, runtime_identity
from experiments.learner.evaluate_builtin_opponent import native_opponent
from experiments.learner import checkpoints
from experiments.learner.builtin_v8_migration import map_identity, metadata, SCHEMA as MIGRATION_SCHEMA
from validation.probes.validate_capabilities import capability_implementation_identity
from validation.probes.validate_hero_policy import frozen_fingerprints
from validation.probes.validate_builtin_boundaries import (
    ROOT, COLORS, binding, remaining, require, until, write)

SCHEMA = "heroes-builtin-v8-startup-diagnostic-v1"
INTERFACE = "adventure-capabilities-v1"
CASE_SECONDS = 120
CHOICES = 256
HARD_CHOICES = 264
MAX_DAYS = 365
SCHEDULE = [{"episode": seat + 1, "learner_player": seat,
             "engine_seed": 181902, "policy_seed": 181903} for seat in (0, 1)]


def load_fixed(path):
    path = Path(path).resolve()
    plan_path, report_path = path.parent / "plan.json", path.parent / "report.json"
    plan, report = strict_json(plan_path.read_bytes()), strict_json(report_path.read_bytes())
    receipt = strict_json((path.parent / "migration-receipt.json").read_bytes())
    require(plan["schema"] == MIGRATION_SCHEMA and binding(path) == report["files"]["policy"]
            and binding(plan_path) == receipt["plan"] and binding(report_path) == receipt["report"],
            "Expected exact preserved migrated baseline")
    loaded = checkpoints.load_policy_checkpoint(path, interface=INTERFACE,
        expected_source_metadata=metadata(plan, binding(plan_path)["sha256"]),
        allowed_provenance=(checkpoints.BUILTIN_GAMEPLAY_PROVENANCE,))
    loaded.model.eval().requires_grad_(False)
    return loaded, frozen_fingerprints(loaded, path)


def ordinary_map_identity(path):
    return map_identity(path)


def check(plan):
    require(plan["schema"] == SCHEMA and plan["schedule"] == SCHEDULE,
            "Fixed diagnostic schedule changed")
    require(plan["case_seconds"] == CASE_SECONDS and plan["ordinary_choices"] == CHOICES
            and plan["hard_choices"] == HARD_CHOICES and plan["max_days"] == MAX_DAYS,
            "Fixed diagnostic budgets changed")
    require(ordinary_map_identity(plan["map"]["path"]) == plan["map"], "Fixture changed")
    require(binding(plan["checkpoint"]["path"]) == plan["checkpoint"], "Checkpoint changed")
    require(capability_implementation_identity() == plan["implementation"], "Diagnostic source changed")
    require(runtime_identity(Path(plan["binary"]).parent) == plan["runtime"], "Native runtime changed")


def worker(args):
    import torch
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.builtin_replay import verify_episode
    from experiments.headless.env import Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.learner.builtin_rollout import (
        BuiltinCollector, audit_persisted_builtin_policy, retain_failed_builtin_evidence)
    torch.set_num_threads(1)
    require(binding(args.plan)["sha256"] == args.sha256, "Diagnostic plan changed")
    plan = strict_json(args.plan.read_bytes())
    row = plan["schedule"][args.index]
    output = Path(plan["output"])
    result = {**row, "status": "failed", "ordinary_cohort_member": False,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0,
        "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
    env = collector = launcher = loaded = None
    started = time.monotonic()
    try:
        with until(args.deadline, args.wall_deadline, 10):
            check(plan)
            loaded, fixed = load_fixed(plan["checkpoint"]["path"])
            require(fixed == plan["frozen"], "Frozen model identity changed")
            seat = row["learner_player"]
            opponent = native_opponent(plan["runtime"], seat)
            launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]),
                Path(plan["map"]["path"]), deadline_monotonic=args.deadline,
                deadline_wall=args.wall_deadline,
                evidence_path=output / f"native-deadline-{row['episode']}.jsonl")

            class BoundedCollector(BuiltinCollector):
                sampled = 0

                def act(self, request):
                    require(self.sampled < HARD_CHOICES and request.record["day"] <= MAX_DAYS,
                            "Diagnostic choice/day budget exhausted")
                    action = super().act(request)
                    self.sampled += 1
                    return action

            collector = BoundedCollector(loaded.model, loaded.checkpoint, learner_player=seat,
                opponent_identity=opponent, policy_seed=row["policy_seed"], interface=INTERFACE)
            env = BuiltinHeadlessEnv(launcher, output, learner_player=seat, interface=INTERFACE,
                limits=Limits(max_days=MAX_DAYS, max_actions=HARD_CHOICES, startup_seconds=30,
                    watchdog_seconds=30, shutdown_seconds=10, bridge_timeout=30),
                agents={COLORS[seat]: {"kind": "frozen-policy-v8-startup-diagnostic",
                    "checkpoint": loaded.checkpoint, "evaluation_only": True}})
            collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                              max_requests=CHOICES)
            folder = env.episode_dir
            write(folder / "builtin-rollout.json", collected.to_dict())
            manifest = collected.manifest
            result.update(episode_directory=str(folder), sampled_choices=len(collected.decisions),
                episode_status=manifest["status"], termination_reason=manifest["termination_reason"],
                latest_sampled_day=max((collected.history[d.decision_seq].day for d in collected.decisions), default=None))
            require(len(collected.decisions) > 0, "No actual learner decisions were executed")
            require(manifest["status"] == "completed" and manifest["termination_reason"] == "natural_terminal",
                    "Both native terminal barriers require an actual natural outcome")
            require(manifest["environment"] == INTERFACE
                    and set(manifest["players"]) == {COLORS[seat]}, "Wrong interface or opponent policy stream")
            rebuilt, _ = load_fixed(plan["checkpoint"]["path"])
            reconstruction = audit_persisted_builtin_policy(folder, rebuilt.model, rebuilt.checkpoint,
                                                            row["policy_seed"], opponent)
            write(folder / "policy-reconstruction.json", reconstruction)
            replay = verify_episode(folder, binary=Path(plan["replay_binary"]), data=Path(plan["data"]),
                timeout=remaining(args.deadline, args.wall_deadline) - 10)
            require(replay.get("verified") is True, "Mixed-controller native replay failed")
            require(replay["wire"]["native_opponent_requests"] > 0,
                    "No actual Nullkiller command was recorded")
            require(frozen_fingerprints(loaded, plan["checkpoint"]["path"]) == plan["frozen"], "Weights changed")
            check(plan)
            result.update(status="passed", replay_verified=True, weights_unchanged=True,
                outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                reconstruction=reconstruction, replay=binding(folder / "builtin-verification/builtin-replay-verification.json"),
                manifest=binding(folder / "manifest.json"),
                request_accounting=replay["wire"]["request_accounting"],
                native_opponent_requests=replay["wire"]["native_opponent_requests"],
                native_replay=replay["native"])
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
        if remaining(args.deadline, args.wall_deadline) > 3:
            retain_failed_builtin_evidence(env, collector, result, error,
                cleanup_seconds=remaining(args.deadline, args.wall_deadline) - 2)
    finally:
        if env is not None and env.episode_dir is not None:
            result["episode_directory"] = str(env.episode_dir)
            path = env.episode_dir / "manifest.json"
            if path.exists(): result["manifest"] = binding(path)
        for label, cleanup in (("environment", lambda: env.close() if env else None),
                               ("native", lambda: launcher.close_owned_groups(timeout=1) if launcher else None)):
            try:
                with until(args.deadline, args.wall_deadline, 1): cleanup()
            except Exception as error:
                result.update(status="failed", outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS),
                              **{label + "_cleanup_error": f"{type(error).__name__}: {error}"})
        result.update(finished=True, wall_seconds=time.monotonic() - started)
        write(output / f"case-{row['episode']}.json", result)


def execute(args):
    import torch
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    torch.set_num_threads(1)
    output = args.output.resolve()
    require(not output.exists(), "Preserve original diagnostic attempts; never replace them")
    _, frozen = load_fixed(args.checkpoint)
    plan = {"schema": SCHEMA, "output": str(output), "schedule": SCHEDULE,
        "map": ordinary_map_identity(args.map), "checkpoint": binding(args.checkpoint), "frozen": frozen,
        "implementation": capability_implementation_identity(), "runtime": runtime_identity(args.binary.resolve().parent),
        "binary": str(args.binary.resolve()), "replay_binary": str(args.replay_binary.resolve()),
        "data": str(args.data.resolve()), "case_seconds": CASE_SECONDS, "ordinary_choices": CHOICES,
        "hard_choices": HARD_CHOICES, "max_days": MAX_DAYS, "ordinary_cohort_member": False,
        "scope": "Two fixed ordinary-map v8 startup checks, unchanged migrated policy versus actual Nullkiller Easy; no PPO, sample admission, failed-game replacement or strategic improvement claim.",
        "absolute_deadline_unix": args.deadline_unix,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0}
    check(plan)
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    report = {"schema": SCHEMA + "-result", "plan": binding(output / "plan.json"), "status": "running",
        "attempts": [{**row, "status": "not_started"} for row in SCHEDULE], "ordinary_cohort_member": False,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0}
    write(output / "report.json", report)
    for index, row in enumerate(SCHEDULE):
        wall_deadline = min(time.time() + CASE_SECONDS, plan["absolute_deadline_unix"])
        require(wall_deadline - time.time() > 15, "Original authorization deadline has no diagnostic reserve")
        deadline = time.monotonic() + wall_deadline - time.time()
        command = [sys.executable, "-m", "validation.probes.validate_builtin_v8_startup", "--worker",
            "--plan", str(output / "plan.json"), "--sha256", report["plan"]["sha256"], "--index", str(index),
            "--deadline", str(deadline), "--wall-deadline", str(wall_deadline)]
        process = None
        entry = {**row, "status": "failed", "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
        try:
            with (output / f"worker-{row['episode']}.log").open("xb") as log:
                process = guarded_popen(command, deadline_monotonic=deadline, deadline_wall=wall_deadline,
                    evidence_path=output / f"worker-deadline-{row['episode']}.jsonl", cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                process.wait(timeout=max(.01, remaining(deadline, wall_deadline) - 2))
            entry = strict_json((output / f"case-{row['episode']}.json").read_bytes())
            require(process.returncode == 0 and entry.get("finished") is True
                    and all(entry[key] == value for key, value in row.items()), "Diagnostic worker changed or failed")
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}",
                         outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
        finally:
            try:
                if process is not None: close_owned_process(process, timeout=1)
                entry["native_groups_closed"] = list(cleanup_from_evidence(
                    output / f"native-deadline-{row['episode']}.jsonl", timeout=1))
            except Exception as error:
                entry.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}",
                             outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
            report["attempts"][index] = entry
            write(output / "report.json", report)
        if entry["status"] != "passed": break
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["attempts"]) else "failed"
    report["checkpoint_after"] = binding(args.checkpoint)
    require(report["checkpoint_after"] == plan["checkpoint"], "Checkpoint changed during diagnostics")
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "replay-binary", "map", "checkpoint", "output", "plan"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--sha256")
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--wall-deadline", type=float)
    args = parser.parse_args()
    if args.worker:
        require(all(getattr(args, key) is not None for key in ("plan", "sha256", "index", "deadline", "wall_deadline")), "Incomplete worker arguments")
        worker(args)
    else:
        require(all(getattr(args, key) is not None for key in ("binary", "data", "replay_binary", "map", "checkpoint", "output", "deadline_unix")), "Incomplete diagnostic arguments")
        result = execute(args)
        print(json.dumps({"status": result["status"], "report": str(args.output.resolve() / "report.json")}))
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__": main()
