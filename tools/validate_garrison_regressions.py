"""Supervise all 28 existing capability controls after a garrison repair.

Only process supervision and source binding are added. Original scripted
controls, independent history audit and native replay remain unchanged; no
episode is training input and no optimizer or learned checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
from experiments.learner.inference import digest, write_json
from validation.probes import validate_capabilities as capability
from validation.probes import validate_garrison_dialogs as garrison


def implementation_identity():
    identity = garrison.implementation_identity()
    identity["tools/validate_garrison_regressions.py"] = digest(__file__)
    identity["experiments/headless/process_deadline.py"] = digest(ROOT / "experiments/headless/process_deadline.py")
    return dict(sorted(identity.items()))


def check_bindings(plan):
    if plan["implementation"] != implementation_identity():
        raise ValueError("Regression source binding changed")
    binary, replay = Path(plan["binary"]), Path(plan["replay_binary"])
    if (runtime_identity(binary.parent) != plan["runtime"] or digest(binary) != plan["binary_sha256"]
            or digest(replay) != plan["replay_binary_sha256"]):
        raise ValueError("Regression native runtime binding changed")


def run_worker(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError("Regression supervision binding changed")
    plan = json.loads(path.read_text())
    if plan.get("schema") != "heroes-garrison-regression-supervision-v1":
        raise ValueError("Unsupported regression supervision")
    check_bindings(plan)
    if min(plan["absolute_deadline_unix"] - time.time(),
           plan["hard_deadline_monotonic"] - time.monotonic()) <= 10:
        raise TimeoutError("Regression deadline expired before execution")
    output = path.parent
    maps = output / "maps"
    maps.mkdir(exist_ok=False)
    for fixture in dict.fromkeys(capability.CASES.values()):
        for player in capability.fixtures.COLORS:
            target = maps / (capability.fixtures.scenario_id(fixture, player) + ".vmap")
            manifest = capability.fixtures.create_scenario(target, fixture=fixture, player=player)
            write_json(target.with_suffix(".manifest.json"), manifest)

    class GuardedCapabilityRun(garrison.GarrisonRun):
        def __init__(self, name, args, *remaining):
            args.deadline_wall = plan["absolute_deadline_unix"] - 10
            args.deadline_monotonic = plan["hard_deadline_monotonic"] - 10
            super().__init__(name, args, *remaining)

    # The original harness retains all controls/admission/replay checks. Every
    # engine gets its own deadline guardian; replay inherits this worker group.
    previous_run, previous_identity = capability.CapabilityRun, capability.capability_implementation_identity
    capability.CapabilityRun = GuardedCapabilityRun
    capability.capability_implementation_identity = implementation_identity
    try:
        return capability.main([
            "--binary", plan["binary"], "--replay-binary", plan["replay_binary"], "--data", plan["data"],
            "--maps", str(maps), "--output", str(output / "execution"), "--case", "all", "--player", "both",
            "--seed", str(plan["seed"]), "--seed-count", "1", "--case-seconds", str(plan["case_seconds"]),
            "--watchdog", "30", "--startup-seconds", "30"])
    finally:
        capability.CapabilityRun, capability.capability_implementation_identity = previous_run, previous_identity
        check_bindings(plan)


def supervise(args):
    # Never overlap the official local training owner or reuse prior evidence.
    with args.official_lock.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        wall, mono = time.time(), time.monotonic()
        plan = {"schema": "heroes-garrison-regression-supervision-v1", "implementation": implementation_identity(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runtime": runtime_identity(args.binary.parent), "binary": str(args.binary),
                "binary_sha256": digest(args.binary), "replay_binary": str(args.replay_binary),
                "replay_binary_sha256": digest(args.replay_binary), "data": str(args.data),
                "seed": args.seed, "case_seconds": args.case_seconds,
                "schedule": capability.schedule(tuple(capability.CASES), capability.fixtures.COLORS, args.seed, 1),
                "started_unix": wall, "started_monotonic": mono,
                "absolute_deadline_unix": wall + args.wall_seconds,
                "hard_deadline_monotonic": mono + args.wall_seconds, "official_lock": str(args.official_lock),
                "training_performed": False, "training_input": False, "optimizer_steps": 0}
        path = args.output / "supervision.json"
        write_json(path, plan)
        expected_hash = digest(path)
        process, code, errors = None, None, []
        with (args.output / "worker.log").open("w") as log:
            try:
                process = guarded_popen([sys.executable, __file__, "--worker-plan", str(path),
                    "--worker-plan-sha256", expected_hash],
                    deadline_wall=plan["absolute_deadline_unix"] - 5,
                    deadline_monotonic=plan["hard_deadline_monotonic"] - 5,
                    evidence_path=args.output / "deadline.jsonl", cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait(timeout=args.wall_seconds)
            except Exception as error:
                errors.append(f"worker: {type(error).__name__}: {error}")
            finally:
                limits = dict(deadline_wall=plan["absolute_deadline_unix"],
                              deadline_monotonic=plan["hard_deadline_monotonic"])
                operations = [] if process is None else [lambda: close_owned_process(process, timeout=2, **limits)]
                operations.extend(lambda p=p: cleanup_from_evidence(p, timeout=2, **limits)
                                  for p in args.output.rglob("*deadline.jsonl"))
                cleanup_errors = []
                for operation in operations:
                    try:
                        operation()
                    except Exception as error:
                        cleanup_errors.append(f"{type(error).__name__}: {error}")
                try:
                    check_bindings(plan)
                    if digest(path) != expected_hash:
                        raise ValueError("Supervision receipt changed")
                except Exception as error:
                    errors.append(f"identity: {type(error).__name__}: {error}")
                write_json(args.output / "shutdown.json", {"returncode": code, "finished_unix": time.time(),
                    "owned_cleanup_verified": not cleanup_errors, "cleanup_errors": cleanup_errors, "errors": errors,
                    "within_hard_deadline": time.time() <= plan["absolute_deadline_unix"],
                    "training_performed": False, "training_input": False, "optimizer_steps": 0})
        return code if not errors and not cleanup_errors and code is not None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "replay-binary", "data", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--case-seconds", type=int, default=180)
    parser.add_argument("--official-lock", type=Path,
                        default=Path.home() / "Projects/Heroes-artifacts/local-script-training/owner.lock")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_plan is not None:
        if not args.worker_plan_sha256:
            parser.error("Worker requires its bound supervision hash")
        return run_worker(args.worker_plan.resolve(), args.worker_plan_sha256)
    if any(getattr(args, name) is None for name in ("binary", "replay_binary", "data", "output", "seed")):
        parser.error("Provide binary, replay-binary, data, fresh output and a fresh seed namespace")
    if not 1 <= args.seed <= 2147480000 or not 60 <= args.wall_seconds <= 1800 or not 1 <= args.case_seconds <= 180:
        parser.error("Use bounded seeds and at most a 30-minute diagnostic window")
    for name in ("binary", "replay_binary", "data", "output", "official_lock"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
