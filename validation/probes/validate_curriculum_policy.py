"""Bounded initial-state selection using unchanged update146 stochastic actions.

Every predeclared seed is retained. Natural outcomes require native terminal,
policy reconstruction and replay evidence. No optimizer or training admission
is constructed; a passed technical report does not imply the mixed-outcome gate.
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
from experiments.learner.builtin_training_resume import _metadata
from experiments.learner.builtin_v8_migration import map_identity, metadata, SCHEMA as MIGRATION_SCHEMA
from validation.probes.validate_capabilities import capability_implementation_identity
from validation.probes.validate_hero_policy import frozen_fingerprints
from validation.probes.validate_builtin_boundaries import (
    ROOT, COLORS, binding, remaining, require, until, write)

SCHEMA = "heroes-curriculum-policy-diagnostic-v1"
INTERFACE = "adventure-capabilities-v1"
CASE_SECONDS = 120
CHOICES = 4960
HARD_CHOICES = 5120
MAX_DAYS = 365
SCHEDULE = [{"episode": seat + 1, "learner_player": seat,
             "engine_seed": 181902, "policy_seed": 181903} for seat in (0, 1)]


def load_fixed(path):
    path = Path(path).resolve()
    parent = Path('/Users/orsi/Projects/Heroes-artifacts/runs/v8-update92-repair3-continuation-148/plan.json')
    plan = strict_json(parent.read_bytes())
    report = strict_json((Path(plan['execution_output'])/'report.json').read_bytes())
    require(binding(path) == report['after_checkpoint']['policy']
            and binding(path)['sha256'] == '5ca61dfcfe35e36e6075bd0690abad6414ff084b0935b703f571b2a63e1bf023',
            'Only immutable update146 policy is admitted to selection diagnostics')
    loaded = checkpoints.load_policy_checkpoint(path, interface=INTERFACE,
        expected_source_metadata=_metadata(plan, report['plan_sha256']),
        allowed_provenance=(checkpoints.BUILTIN_GAMEPLAY_PROVENANCE,))
    loaded.model.eval().requires_grad_(False)
    return loaded, frozen_fingerprints(loaded, path)


def ordinary_map_identity(path):
    from experiments.explorer.create_curriculum_scenario import verify
    path=Path(path);parts=path.stem.split('-')
    stage=int(parts[-2]);seat=int(parts[-1]);verify(path,stage,seat)
    return dict(binding(path),stage=stage,seat=seat,scenario='curriculum-v1')


def schedule(seat, stage=0, block=0, study=1):
    return [{'episode':i+1,'learner_player':seat,'engine_seed':(610001+1000*stage if study == 1 else 810001)+100*block+i,
             'policy_seed':(620001+1000*stage if study == 1 else 820001)+100*block+i} for i in range(4)]


def check(plan):
    require(binding(__file__) == plan["diagnostic_source"]
            and binding(ROOT/"experiments/explorer/create_curriculum_scenario.py") == plan["generator_source"], "Diagnostic source drift")
    require(plan["schema"] == SCHEMA and plan["schedule"] == schedule(plan["map"]["seat"],plan["map"]["stage"],plan["seed_block"],plan.get("study",1)),
            "Fixed diagnostic schedule changed")
    require(plan["case_seconds"] == CASE_SECONDS and plan["ordinary_choices"] == CHOICES
            and plan["hard_choices"] == HARD_CHOICES and plan["max_days"] == MAX_DAYS,
            "Fixed diagnostic budgets changed")
    require(ordinary_map_identity(plan["map"]["path"]) == plan["map"], "Fixture changed")
    require(binding(plan["checkpoint"]["path"]) == plan["checkpoint"], "Checkpoint changed")
    require(capability_implementation_identity() == plan["implementation"], "Diagnostic source changed")
    require(runtime_identity(Path(plan["binary"]).parent) == plan["runtime"], "Native runtime changed")
    require({key: binding(plan[key]) for key in ("binary", "replay_binary")} == plan["executables"],
            "Native executable bytes changed")


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
                    "Diagnostic gate requires actual natural outcome")
            require(manifest["environment"] == INTERFACE
                    and set(manifest["players"]) == {COLORS[seat]}, "Wrong interface or opponent policy stream")
            rebuilt, _ = load_fixed(plan["checkpoint"]["path"])
            reconstruction = audit_persisted_builtin_policy(folder, rebuilt.model, rebuilt.checkpoint,
                                                            row["policy_seed"], opponent)
            write(folder / "policy-reconstruction.json", reconstruction)
            replay = verify_episode(folder, binary=Path(plan["replay_binary"]), data=Path(plan["data"]),
                timeout=remaining(args.deadline, args.wall_deadline) - 10)
            require(replay.get("verified") is True, "Mixed-controller native replay failed")
            require(replay["wire"]["native_opponent_requests"] >= 0,
                    "Invalid native opponent request accounting")
            # A legitimate first-turn capture can eliminate the opponent before it acts.
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
    plan = {"schema": SCHEMA, "output": str(output), "schedule": schedule(ordinary_map_identity(args.map)["seat"],ordinary_map_identity(args.map)["stage"],args.seed_block,args.study),
        "study": args.study, "seed_block": args.seed_block, "map": ordinary_map_identity(args.map), "checkpoint": binding(args.checkpoint), "frozen": frozen,
        "implementation": capability_implementation_identity(), "diagnostic_source": binding(__file__),
        "generator_source": binding(ROOT/"experiments/explorer/create_curriculum_scenario.py"), "runtime": runtime_identity(args.binary.resolve().parent),
        "binary": str(args.binary.resolve()), "replay_binary": str(args.replay_binary.resolve()),
        "executables": {key: binding(getattr(args, key)) for key in ("binary", "replay_binary")},
        "data": str(args.data.resolve()), "case_seconds": CASE_SECONDS, "ordinary_choices": CHOICES,
        "hard_choices": HARD_CHOICES, "max_days": MAX_DAYS, "ordinary_cohort_member": False,
        "scope": "Four predeclared curriculum selection checks, unchanged update146 policy versus actual Nullkiller Easy; no PPO, sample admission, failed-game replacement or strategic improvement claim.",
        "absolute_deadline_unix": args.deadline_unix,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0}
    check(plan)
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    report = {"schema": SCHEMA + "-result", "plan": binding(output / "plan.json"), "status": "running",
        "attempts": [{**row, "status": "not_started"} for row in plan["schedule"]], "ordinary_cohort_member": False,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0}
    write(output / "report.json", report)
    for index, row in enumerate(plan["schedule"]):
        wall_deadline = min(time.time() + CASE_SECONDS, plan["absolute_deadline_unix"])
        require(wall_deadline - time.time() > 15, "Original authorization deadline has no diagnostic reserve")
        deadline = time.monotonic() + wall_deadline - time.time()
        command = [sys.executable, "-m", "validation.probes.validate_curriculum_policy", "--worker",
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
        # Preserve each failed attempt; later predeclared diagnostic seeds are independent.
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["attempts"]) else "failed"
    report["checkpoint_after"] = binding(args.checkpoint)
    require(report["checkpoint_after"] == plan["checkpoint"], "Checkpoint changed during diagnostics")
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "replay-binary", "map", "checkpoint", "output", "plan"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seed-block", type=int, choices=range(8), default=0)
    parser.add_argument("--study", type=int, choices=(1,2), default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--sha256")
    parser.add_argument("--index", type=int, choices=range(4))
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
