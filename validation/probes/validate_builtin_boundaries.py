"""Two isolated zero-action mixed-controller cutoff checks, never evaluation games.

Each fixed row starts the supplied exact townless diagnostic fixture, waits for
its learner's first real safe decision, and freezes it without a policy reply.
A private worker guardian bounds its replay and descendants; a second guardian
bounds the native process group. All work shares that row's fixed 90 seconds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless.contextual_evidence import strict_json
from validation.probes.create_boundary_fixture import verify_fixture, SCHEMAS

SCHEMA = "heroes-builtin-initial-cutoff-diagnostic-v1"
CASE_SECONDS = 90
CLEANUP_SECONDS = 10
ENGINE_SEED = 121901
COLORS = ("red", "blue")


def require(value, message):
    if not value:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fixture_identity(path):
    path = Path(path).resolve()
    verify_fixture(path, kind="natural-townless")
    manifest = path.parent / "fixture.json"
    document = strict_json(manifest.read_bytes())
    require(document.get("kind") == "natural-townless" and document.get("schema") == SCHEMAS["natural-townless"]
            and document["fixture"]["sha256"] == binding(path)["sha256"]
            and Path(document["fixture"]["path"]).resolve() == path, "Diagnostic fixture declaration differs")
    return {"map": binding(path), "manifest": binding(manifest), "kind": "natural-townless",
        "semantics_verified": True,
        "probe_plan": "Zero learner actions; freeze the initial real safe own decision. The fixture's historical end_turn evaluator plan is not used.",
        "natural_outcome_validation": False, "ordinary_cohort_member": False}


def remaining(deadline, wall_deadline):
    return min(deadline - time.monotonic(), wall_deadline - time.time())


@contextmanager
def until(deadline, wall_deadline, reserve=0):
    seconds = remaining(deadline, wall_deadline) - reserve
    require(seconds > 0, "Diagnostic case has no remaining wall budget")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Another wall timer is active")
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("Diagnostic case wall budget exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def held_cutoff(env, row):
    """The sole control operation is truncate; no action or replacement is chosen."""
    from experiments.learner import contextual as c
    request = env.reset(episode=row["episode"], seed=row["engine_seed"])
    require(not request.done, "Game ended before its initial safe own decision")
    require(request.player == row["learner_player"] and request.color == COLORS[row["learner_player"]]
            and request.kind == "contextual" and request.reward == 0 and not request.past_horizon,
            "Initial request is not the current learner's ordinary decision")
    decision = c.decode_frame(request.record)
    require(decision.kind == "decision" and decision.observation.context.kind != "dialog"
            and request.frames and c.decode_frame(request.frames[-1]) == decision,
            "No actual initial safe held decision; strategic queries are not answered by this probe")
    end = env.truncate("collection_window")
    require(end.done and end.status == "truncated" and end.termination_reason == "collection_window"
            and end.outcomes == dict.fromkeys(COLORS) and end.rewards == dict.fromkeys(COLORS),
            "Held decision did not produce an unscored production cutoff")
    frames = [c.decode_frame(raw) for raw in env.history(request.color)]
    cutoffs = [frame for frame in frames if frame.kind == "cutoff"]
    require(len(cutoffs) == 1 and frames[-1] == cutoffs[0] and cutoffs[0].decision_seq == decision.seq
            and frames[decision.seq] == decision, "Cutoff does not bind the initial held decision")
    require(not any(frame.kind in ("submitted", "acknowledged", "result") for frame in frames),
            "Zero-action diagnostic unexpectedly contains a learner action")
    declaration = end.manifest["players"][request.color]
    require(set(end.manifest["players"]) == {request.color}
            and all(type(declaration.get(key)) is int and declaration[key] == 0 for key in
                    ("actions_sent", "actions_accepted", "actions_rejected", "responses_undelivered")),
            "Zero-action diagnostic has invented or sent policy responses")
    return end, {"held_decision_seq": decision.seq, "held_decision_day": decision.day,
        "cutoff_seq": cutoffs[0].seq, "own_history_frames": len(frames),
        "visible_events_retained": sum(frame.kind == "event" for frame in frames), "policy_actions": 0}


def worker(args):
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.builtin_replay import verify_episode
    from experiments.headless.env import Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    plan = strict_json(args.plan.read_bytes())
    require(binding(args.plan)["sha256"] == args.sha256, "Diagnostic plan changed")
    row = plan["schedule"][args.index]
    output = Path(plan["output"])
    result = {**row, "status": "failed", "purpose": "diagnostic", "policy_actions": 0,
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0,
        "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
    env = launcher = None
    started = time.monotonic()
    try:
        with until(args.deadline, args.wall_deadline, CLEANUP_SECONDS):
            require(fixture_identity(plan["fixture"]["map"]["path"]) == plan["fixture"], "Fixture changed before launch")
            for item in plan["executables"].values():
                require(binding(item["path"]) == item, "Native executable changed")
            launcher = DeadlineVcmiLauncher(Path(plan["executables"]["binary"]["path"]), Path(plan["data"]),
                Path(plan["fixture"]["map"]["path"]), deadline_monotonic=args.deadline,
                deadline_wall=args.wall_deadline, evidence_path=output / f"native-deadline-{row['episode']}.jsonl")
            env = BuiltinHeadlessEnv(launcher, output, learner_player=row["learner_player"],
                limits=Limits(max_days=2, max_actions=1, startup_seconds=30, watchdog_seconds=30,
                              shutdown_seconds=10, bridge_timeout=30),
                agents={COLORS[row["learner_player"]]: {"kind": "zero-action-boundary-diagnostic", "evaluation_only": True}})
            end, observed = held_cutoff(env, row)
            result.update(observed, episode_directory=str(env.episode_dir), episode_status=end.status)
            bridge = env.episode_dir / "agents" / COLORS[row["learner_player"]] / "bridge.jsonl"
            require(all(strict_json(line)["direction"] == "engine" for line in bridge.read_bytes().splitlines()),
                    "A policy response entered the zero-action bridge")
            replay = verify_episode(env.episode_dir, binary=Path(plan["executables"]["replay_binary"]["path"]),
                data=Path(plan["data"]), timeout=remaining(args.deadline, args.wall_deadline) - CLEANUP_SECONDS)
            require(replay.get("verified") is True and replay["wire"]["responses"] == 0,
                    "Exact mixed native cutoff replay failed")
            result.update(status="passed", replay_verified=True, native_opponent_requests=replay["native_opponent_requests"],
                replay=binding(env.episode_dir / "builtin-verification/builtin-replay-verification.json"),
                effective_engine=replay["effective_engine"], manifest=binding(env.episode_dir / "manifest.json"))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        if env is not None and env.process is not None and env.result is None:
            try:
                with until(args.deadline, args.wall_deadline, 4):
                    env._note_problem("protocol_error", "Zero-action boundary diagnostic failed: " + result["error"])
                    env._finish(None)
            except Exception as secondary:
                result["failure_finalization_error"] = f"{type(secondary).__name__}: {secondary}"
    finally:
        if env is not None and env.episode_dir is not None:
            result["episode_directory"] = str(env.episode_dir)
            path = env.episode_dir / "manifest.json"
            if path.exists():
                result.update(manifest=binding(path), observed_episode_status=strict_json(path.read_bytes())["status"])
        for label, cleanup in (("environment", lambda: env.close() if env else None),
                               ("native", lambda: launcher.close_owned_groups(timeout=1) if launcher else None)):
            try:
                with until(args.deadline, args.wall_deadline, 1):
                    cleanup()
            except Exception as error:
                result.update(status="failed", **{label + "_cleanup_error": f"{type(error).__name__}: {error}"})
        result["wall_seconds"] = time.monotonic() - started
        result["finished"] = True
        write(output / f"case-{row['episode']}.json", result)
    return result


def execute(args):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output = args.output.resolve()
    require(not output.exists(), "Preserve prior diagnostic attempts; no retry or replacement")
    plan = {"schema": SCHEMA, "output": str(output), "fixture": fixture_identity(args.map),
        "executables": {name: binding(getattr(args, name)) for name in ("binary", "replay_binary")},
        "source": binding(__file__), "data": str(args.data.resolve()), "case_seconds": CASE_SECONDS,
        "schedule": [{"episode": seat + 1, "learner_player": seat, "engine_seed": ENGINE_SEED} for seat in (0, 1)],
        "training_performed": False, "training_samples": 0, "policy_actions": 0,
        "scope": "Separate production startup/cutoff preflight. No policy, model, optimizer or strategy; never included in eight-game comparison rates.",
        "replay_scope": "Recorded native opponent commands and regenerated engine effects; not deterministic Nullkiller from seed."}
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    report = {"schema": SCHEMA + "-result", "plan": binding(output / "plan.json"), "status": "running",
        "attempts": [{**row, "status": "not_started"} for row in plan["schedule"]], "training_performed": False,
        "training_samples": 0, "optimizer_steps": 0, "ordinary_cohort_member": False}
    write(output / "report.json", report)
    for index, row in enumerate(plan["schedule"]):
        deadline, wall_deadline = time.monotonic() + CASE_SECONDS, time.time() + CASE_SECONDS
        command = [sys.executable, "-m", "validation.probes.validate_builtin_boundaries", "--worker",
            "--plan", str(output / "plan.json"), "--sha256", report["plan"]["sha256"], "--index", str(index),
            "--deadline", str(deadline), "--wall-deadline", str(wall_deadline)]
        process = None
        entry = {**row, "status": "failed"}
        try:
            with (output / f"worker-{row['episode']}.log").open("xb") as log:
                process = guarded_popen(command, deadline_monotonic=deadline, deadline_wall=wall_deadline,
                    evidence_path=output / f"worker-deadline-{row['episode']}.jsonl", cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                process.wait(timeout=max(0.01, remaining(deadline, wall_deadline) - 2))
            result = strict_json((output / f"case-{row['episode']}.json").read_bytes())
            require(process.returncode == 0 and result.get("finished") is True
                    and all(result[key] == value for key, value in row.items()), "Worker failed or changed its fixed row")
            entry = result
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                if process is not None:
                    close_owned_process(process, timeout=max(0.01, min(1, remaining(deadline, wall_deadline))))
                entry["native_groups_closed"] = list(cleanup_from_evidence(output / f"native-deadline-{row['episode']}.jsonl",
                    timeout=max(0.01, min(1, remaining(deadline, wall_deadline)))))
            except Exception as error:
                entry.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            report["attempts"][index] = entry
            write(output / "report.json", report)
        if entry["status"] != "passed":
            break
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["attempts"]) else "failed"
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "replay-binary", "map", "output", "plan"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sha256")
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--wall-deadline", type=float)
    args = parser.parse_args()
    if args.worker:
        require(all(getattr(args, key) is not None for key in ("plan", "sha256", "index", "deadline", "wall_deadline")), "Incomplete worker arguments")
        worker(args)
    else:
        require(all(getattr(args, key) is not None for key in ("binary", "data", "replay_binary", "map", "output")), "Supply binary, data, replay-binary, map and output")
        result = execute(args)
        print(json.dumps({"status": result["status"], "report": str(args.output.resolve() / "report.json")}))
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
