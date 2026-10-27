"""Eight fixed v7 evaluations against native Nullkiller under a separate hour.

Preparation is read-only with respect to the engine. Execution never constructs
an optimizer. Every scheduled row survives failures; no attempt is replaced.
The shared deadline includes checkpoint loading, collection, reconstruction,
native replay and cleanup. Final offline report assembly is timed separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/builtin-opponent-evaluation-v1.json"
INTERFACE = "adventure-dialogs-v1"
COLORS = ("red", "blue")
SCHEMA = "heroes-builtin-opponent-evaluation-v1"
V2_SCHEMA = "heroes-builtin-opponent-evaluation-v2"
CHECKPOINT_SHA = "4043b58bb3dc4eb6169514ca1ed7c1be0b373a5139efe04b95dfeda1832232c1"
# Only these native, episode-local failures may advance to the next fixed row.
LOCAL_FAILURES = frozenset(("unsupported_interface", "engine_error", "engine_crash",
    "startup_timeout", "watchdog_timeout", "bridge_disconnected", "action_delivery_failed"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    from experiments.headless.digest_scope import sha256_file
    return sha256_file(Path(path))[0]


def write(path, value):
    """Atomic report replacement; an interrupted worker cannot leave half JSON."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temp.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def binding(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def config():
    from experiments.learner.evaluate_long_dialogs import config as neural_config
    value = json.loads(CONFIG.read_text())
    expected = neural_config()
    expected.update(schema="heroes-builtin-opponent-evaluation-config-v1", max_total_choices=2048,
        ordinary_choice_limit=2040, learner_policy_seeds=[p[0] for p in expected["model_policy_seeds"]],
        opponent={"kind": "builtin", "module": "Nullkiller", "difficulty": 0})
    del expected["model_policy_seeds"]
    require(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
            "Versioned comparison scope changed; invocation flags cannot expand it")
    return value


def schedule(cfg):
    return [{"episode": 2 * index + seat + 1, "engine_seed": seed, "learner_player": seat,
             "policy_seed": policy_seed}
            for index, (seed, policy_seed) in enumerate(zip(cfg["engine_seeds"], cfg["learner_policy_seeds"]))
            for seat in (0, 1)]


def implementation(version=1):
    from experiments.learner.first_run import implementation_identity
    result = implementation_identity()
    for path in (CONFIG, ROOT / "experiments/explorer/create_hero_scenario.py"):
        result[str(path.relative_to(ROOT))] = digest(path)
    if version == 2:
        path = ROOT / "configs/long-dialog-evaluation-v1.json"
        result[str(path.relative_to(ROOT))] = digest(path)
    return result


def plan_version(plan):
    from experiments.learner.evaluation_revalidation import version
    return version(plan, SCHEMA)


def result_schema(plan):
    if "schema" not in plan:  # Legacy in-process supervisor callers; read_plan always requires it.
        return SCHEMA + "-result"
    return (SCHEMA if plan_version(plan) == 1 else V2_SCHEMA) + "-result"


def load_fixed(path, metadata=None):
    # The archived neural opponent stays in the checkpoint but is never acted.
    from experiments.learner.evaluate_long_dialogs import load_fixed as original_load
    return original_load(path, metadata)


def reference(plan_path, report_path):
    from experiments.learner import evaluate_long_dialogs as neural
    from experiments.learner.evaluation_revalidation import comparison_identity, equal
    report = json.loads(Path(report_path).read_text())
    require(report.get("status") in ("finished", "stopped"), "Finish the neural cohort before this comparison")
    plan = neural.read_plan(plan_path, report["plan_sha256"])
    require(Path(report_path).resolve() == Path(plan["execution_output"]) / "report.json",
            "Reference report is outside its fixed cohort")
    require(len(report["attempts"]) == 8 and all(
        all(row[k] == wanted[k] for k in wanted) for row, wanted in zip(report["attempts"], plan["schedule"])),
        "Reference must preserve all eight scheduled attempts")
    require(report.get("training_performed") is False and report.get("optimizer_steps") == 0
            and report.get("frozen_after_file_sha256") == plan["checkpoint"]["sha256"],
            "Reference cohort lacks its final frozen-checkpoint declaration")
    result = {"plan": binding(plan_path), "report": binding(report_path), "schedule": plan["schedule"],
            "checkpoint_sha256": plan["checkpoint"]["sha256"], "map_sha256": plan["map"]["sha256"],
            "shared_wall_seconds": plan["configuration"]["shared_wall_seconds"], "status": report["status"]}
    if neural.plan_version(plan) == 2:
        require(report.get("schema") == neural.result_schema(plan), "Reference report version differs")
        require(report.get("training_performed") is False
                and all(type(report.get(k)) is int and report[k] == 0
                        for k in ("optimizer_updates", "optimizer_steps", "training_samples"))
                and equal(report.get("frozen_before"), plan["frozen"]), "Reference frozen evaluation declaration changed")
        for row, scheduled in zip(report["attempts"], plan["schedule"]):
            require(all(equal(row.get(k), v) for k, v in scheduled.items()), "Reference schedule scalar types changed")
            require(row.get("status") in ("passed", "native_failure", "failed", "budget_exhausted", "incomplete", "not_started")
                    and type(row.get("training_samples")) is int and row["training_samples"] == 0
                    and type(row.get("optimizer_steps")) is int and row["optimizer_steps"] == 0,
                    "Reference attempt is unfinished or not evaluation-only")
            if row["status"] != "passed":
                require(row.get("outcomes") == dict.fromkeys(COLORS) and row.get("rewards") == dict.fromkeys(COLORS),
                        "Reference unverified attempt received a score")
            if report["status"] == "finished":
                require(row["status"] in ("passed", "native_failure"), "Finished reference has uncompleted rows")
        result.update(plan_schema=plan["schema"], comparison_identity=comparison_identity(plan))
    return result


def native_opponent(runtime, player):
    from experiments.headless.builtin_evidence import opponent_identity
    libraries = [runtime[name] for name in ("AI/libNullkiller.dylib", "AI/libNullkiller.so") if name in runtime]
    require(len(libraries) == 1, "Exactly one available Nullkiller runtime is required")
    return opponent_identity(player, libraries[0])


def compatible_sources(saved, current):
    """New evaluator files are allowed; every original learner byte stays fixed."""
    for key in ("torch", "python", "interface"):
        require(saved[key] == current[key], "Checkpoint runtime/interface changed: " + key)
    for path, sha in saved["source_sha256"].items():
        require(current["source_sha256"].get(path) == sha, "Checkpoint learner dependency changed: " + path)


def prepare(args):
    import torch
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.inference import source_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    from experiments.learner.evaluation_revalidation import AUTHORIZATION, prepare_provenance, match_reference
    cfg = config()
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Preparation output exists; never replace or retry an earlier cohort")
    revalidation = prepare_provenance(args, reader=read_plan, legacy_schema=SCHEMA,
        configuration=cfg, wanted_schedule=schedule(cfg))
    version = 2 if revalidation is not None else 1
    torch.set_num_threads(1)
    checkpoint = binding(args.checkpoint)
    loaded, fixed = load_fixed(checkpoint["path"])
    compared = reference(args.reference_plan, args.reference_report)
    require(compared["checkpoint_sha256"] == cfg["checkpoint_sha256"]
            and compared["map_sha256"] == cfg["map_sha256"]
            and compared["shared_wall_seconds"] == cfg["shared_wall_seconds"], "Comparison reference identity changed")
    current = source_identity(INTERFACE)
    compatible_sources(loaded.source_metadata, current)
    launcher = VcmiLauncher(args.binary.resolve(), args.data.resolve(), args.map.resolve())
    launcher.preflight()
    scenario = hero_map_identity(args.map)
    require(scenario["sha256"] == cfg["map_sha256"], "Ordinary Tavern-enabled scenario changed")
    checked = subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    plan = {"schema": (SCHEMA if version == 1 else V2_SCHEMA) + "-plan", "configuration": cfg, "schedule": schedule(cfg),
        "execution_output": str(output / "execution"), "checkpoint": checkpoint,
        "reference": compared,
        "native_opponents": [native_opponent(runtime_identity(args.binary.resolve().parent), p) for p in (0, 1)],
        "checkpoint_source": loaded.source_metadata, "checkpoint_provenance": loaded.provenance,
        "source": current, "implementation": implementation(version), "frozen": fixed,
        "binary": str(args.binary.resolve()), "data": str(args.data.resolve()),
        "game_map": str(args.map.resolve()), "replay_binary": binding(args.replay_binary),
        "native": launcher.identity(), "runtime": runtime_identity(args.binary.resolve().parent), "map": scenario,
        "prepared_check": checked.stdout.strip(), "training_performed": False,
        "authorization": {"source_thread": "01a16d9e-5f8d-7cb0-a939-3ff005a741e4",
            "human_turn": "01a170d5-0c38-74d3-8102-ff4d461ca20d",
            "human_message": "01a170d5-0cbf-70e1-a497-919c39b80e4c", "confirmed_before_preparation": True},
        "wall_scope": "One shared 3600-second execution clock includes all launches, collection, reconstruction, native replay and cleanup. Final offline report assembly has a separate measured duration; it cannot launch games or replay.",
        "choice_scope": "2040 ordinary learner choices plus at most 8 reserved real dialog choices, never more than 2048. This approximate half-budget is not identical exposure to 4096 shared neural choices. Native opponent requests are counted separately, never interpreted as policy choices. No technical cutoff continuation is sent.",
        "replay_scope": "Independent native replay consumes captured native-opponent requests, verifies their player and order, and regenerates rules/packets/state. It does not regenerate Nullkiller decisions from an engine seed.",
        "failure_policy": "No retries/replacements. Preserved isolated native failures may advance; global deadline, unknown invariants, identity drift, cleanup or evidence errors abort remaining rows."}
    if version == 2:
        plan.update(revalidation=revalidation, authorization=dict(AUTHORIZATION))
        match_reference(plan)
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "sha256": digest(output / "plan.json")}


def read_plan(path, sha):
    path = Path(path).resolve()
    require(digest(path) == sha, "Prepared plan SHA-256 changed")
    plan = json.loads(path.read_text())
    cfg = config()
    version = plan_version(plan)
    if version == 1:
        require(plan["configuration"] == cfg
                and plan["schedule"] == schedule(cfg) and plan["execution_output"] == str(path.parent / "execution"),
                "Prepared plan scope or output changed")
    else:
        from experiments.learner.evaluation_revalidation import equal
        require(equal(plan["configuration"], cfg) and equal(plan["schedule"], schedule(cfg))
                and plan["execution_output"] == str(path.parent / "execution"), "Prepared plan scope or output changed")
    if version == 2:
        from experiments.learner.evaluation_revalidation import validate_provenance, match_reference
        validate_provenance(plan, reader=read_plan, legacy_schema=SCHEMA)
        require(reference(plan["reference"]["plan"]["path"], plan["reference"]["report"]["path"]) == plan["reference"],
                "Reference cohort artifacts changed")
        match_reference(plan)
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.inference import source_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    version = plan_version(plan)
    require(implementation(version) == plan["implementation"] and source_identity(INTERFACE) == plan["source"],
            "Authored source changed after preparation")
    require(reference(plan["reference"]["plan"]["path"], plan["reference"]["report"]["path"]) == plan["reference"],
            "Reference cohort artifacts changed")
    if version == 2:
        from experiments.learner.evaluation_revalidation import validate_provenance, match_reference
        validate_provenance(plan, reader=read_plan, legacy_schema=SCHEMA)
        match_reference(plan)
    require(binding(plan["checkpoint"]["path"]) == plan["checkpoint"]
            and binding(plan["replay_binary"]["path"]) == plan["replay_binary"], "Bound artifact changed")
    launcher = VcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]))
    require(launcher.identity() == plan["native"] and runtime_identity(Path(plan["binary"]).parent) == plan["runtime"]
            and hero_map_identity(plan["game_map"]) == plan["map"], "Native runtime or map changed")


def remaining(deadline, wall_deadline):
    return min(deadline - time.monotonic(), wall_deadline - time.time())


class ExecutionBudgetExpired(TimeoutError):
    pass


def require_choice_budget(used, day, cfg):
    require(type(used) is int and 0 <= used < cfg["max_total_choices"],
            "Total policy-choice budget exhausted; unresolved query remains unscored")
    require(type(day) is int and day <= cfg["max_days"],
            "Day budget exhausted; unresolved query remains unscored")


def empty_attempt(row):
    return {**row, "status": "not_started", "reason": "pending_schedule", "outcomes": dict.fromkeys(COLORS),
            "rewards": dict.fromkeys(COLORS), "training_samples": 0, "optimizer_steps": 0}


def can_continue(attempt):
    return (attempt.get("cleanup_verified") is True and attempt.get("identity_verified") is True
            and (attempt["status"] == "passed" or attempt["status"] == "native_failure"
                 and attempt.get("termination_reason") in LOCAL_FAILURES))


def worker(args):
    import torch
    from experiments.headless.env import Limits
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.headless.builtin_replay import verify_episode
    from experiments.learner.builtin_rollout import (BuiltinCollector, audit_persisted_builtin_policy,
        retain_failed_builtin_evidence)
    from experiments.learner.builtin_behavior import analyze_episode
    from validation.probes.validate_hero_policy import frozen_fingerprints
    torch.set_num_threads(1)
    plan = read_plan(args.plan, args.sha256)
    cfg, output = plan["configuration"], Path(plan["execution_output"])
    row = plan["schedule"][args.index]
    result_path = output / f"attempt-{row['episode']:02d}.json"
    attempt = {**empty_attempt(row), "status": "running", "phase": "initialization"}
    attempt.pop("reason")
    started, env, collector, launcher = time.monotonic(), None, None, None
    from experiments.learner.evaluation_revalidation import PhaseTimings, record_progress_safely
    timings = PhaseTimings(started)
    def save(phase):
        durations, now = timings.advance(phase)
        attempt.update(phase=phase, wall_seconds=now - started)
        if plan_version(plan) == 2:
            attempt["phase_wall_seconds"] = durations
        write(result_path, attempt)
    def budget_signal(signum, frame):
        raise ExecutionBudgetExpired("Shared evaluation execution hour is ending")
    signal.signal(signal.SIGUSR1, budget_signal)
    try:
        save("identity_check")
        check_identity(plan)
        loaded, fixed = load_fixed(plan["checkpoint"]["path"], plan["checkpoint_source"])
        require(fixed == plan["frozen"], "Frozen weights changed")
        seat = row["learner_player"]
        opponent = native_opponent(plan["runtime"], seat)
        require(opponent == plan["native_opponents"][seat], "Native opponent identity differs")
        launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]),
            deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
            evidence_path=output / f"native-deadline-{row['episode']:02d}.jsonl")

        class BoundedCollector(BuiltinCollector):
            sampled = 0

            def act(self, request):
                require_choice_budget(self.sampled, request.record["day"], cfg)
                action = super().act(request)
                self.sampled += 1
                if self.sampled % 64 == 0:
                    attempt.update(sampled_choices=self.sampled,
                                   latest_sampled_day=request.record["day"])
                    save("collection")
                return action

        collector = BoundedCollector(loaded.model, loaded.checkpoint, learner_player=seat,
            opponent_identity=opponent, policy_seed=row["policy_seed"])
        limits = Limits(max_days=cfg["max_days"], max_actions=cfg["max_total_choices"],
                        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30)
        env = BuiltinHeadlessEnv(launcher, output, learner_player=seat, limits=limits, ledger=True,
            replay=True, interface=INTERFACE, agents={COLORS[seat]: {"checkpoint": loaded.checkpoint,
                "kind": "fixed-untrained-recurrent", "evaluation_only": True}})
        save("collection")
        collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                          max_requests=cfg["ordinary_choice_limit"])
        folder = env.episode_dir
        write(folder / "builtin-rollout.json", collected.to_dict())
        manifest = json.loads((folder / "manifest.json").read_text())
        count = len(collected.decisions)
        require(count <= cfg["max_total_choices"], "Learner policy-choice cap exceeded")
        require(collected.optimization_eligible is False, "Evaluation became optimizer eligible")
        require(len(collector.sessions) == 1 and collector.session.history.player == seat,
                "Only the actual learner can have an external session")
        attempt.update(episode_directory=str(folder), episode_status=manifest["status"],
            termination_reason=manifest["termination_reason"], sampled_choices=count,
            latest_sampled_day=max((collected.history[d.decision_seq].day for d in collected.decisions), default=None),
            environment_limits=asdict(limits), opponent_identity=opponent, external_player_histories=1)
        if manifest["status"] == "failed":
            attempt.update(status="native_failure", error=manifest["failure_detail"])
        else:
            require(manifest["status"] in ("completed", "truncated"), "Unknown episode boundary")
            save("policy_reconstruction")
            reloaded, _ = load_fixed(plan["checkpoint"]["path"], plan["checkpoint_source"])
            reconstruction = audit_persisted_builtin_policy(folder, reloaded.model, reloaded.checkpoint,
                row["policy_seed"], opponent)
            write(folder / "policy-reconstruction.json", reconstruction)
            save("native_replay")
            seconds = remaining(args.deadline, args.wall_deadline) - cfg["cleanup_reserve_seconds"]
            if seconds <= 0:
                raise ExecutionBudgetExpired("No shared time remains for native replay")
            replay = verify_episode(folder, binary=Path(plan["replay_binary"]["path"]),
                                    data=Path(plan["data"]), timeout=seconds)
            require(replay.get("verified") is True, "Independent native replay failed")
            attempt.update(status="passed", outcomes=manifest["outcomes_by_player"],
                rewards=manifest["rewards_by_player"], reconstruction=reconstruction, replay_verified=True,
                native_opponent_requests=replay["native_opponent_requests"], replay_scope=plan["replay_scope"])
        save("offline_behavior")
        behavior = analyze_episode(folder)
        write(folder / "behavior.json", behavior)
        attempt["behavior"] = binding(folder / "behavior.json")
        require(frozen_fingerprints(loaded, plan["checkpoint"]["path"]) == plan["frozen"], "Weights changed")
        attempt["weights_unchanged"] = True
    except Exception as error:
        attempt.update(status="budget_exhausted" if isinstance(error, ExecutionBudgetExpired) else "failed",
                       error=f"{type(error).__name__}: {error}", outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
        if plan_version(plan) == 2:
            record_progress_safely(save, attempt, "failure_retention")
        # A signal is sent only once. Cleanup consumes remaining shared time,
        # never the old short probe's fresh post-case thirty-second allowance.
        seconds = remaining(args.deadline, args.wall_deadline) - 2
        if seconds > 0:
            retain_failed_builtin_evidence(env, collector, attempt, error, cleanup_seconds=seconds)
        if env is not None and env.episode_dir is not None:
            attempt["episode_directory"] = str(env.episode_dir)
            manifest_path = env.episode_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                attempt.update(episode_status=manifest["status"], termination_reason=manifest["termination_reason"])
    finally:
        if plan_version(plan) == 2:
            record_progress_safely(save, attempt, "cleanup")
        try:
            if env is not None:
                env.close()
        except Exception as error:
            attempt.update(status="failed", environment_cleanup_error=f"{type(error).__name__}: {error}")
        try:
            if launcher is not None:
                launcher.close_owned_groups()
            attempt["cleanup_verified"] = True
        except Exception as error:
            attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}", cleanup_verified=False)
        try:
            check_identity(plan)
            attempt["identity_verified"] = True
        except Exception as error:
            attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}", identity_verified=False)
        if attempt["status"] != "passed":
            attempt.update(outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
        save("finished")
    return attempt


def execute(args):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    plan = read_plan(args.plan, args.sha256)
    check_identity(plan)
    output, cfg = Path(plan["execution_output"]), plan["configuration"]
    output.mkdir(exist_ok=False)  # No execution resume or replacement.
    started, wall_started = time.monotonic(), time.time()
    deadline = started + cfg["shared_wall_seconds"]
    wall_deadline = wall_started + cfg["shared_wall_seconds"]
    hard_deadline = deadline - cfg["hard_stop_reserve_seconds"]
    hard_wall = wall_deadline - cfg["hard_stop_reserve_seconds"]
    report = {"schema": result_schema(plan), "plan_sha256": args.sha256, "status": "running",
        "attempts": [empty_attempt(row) for row in plan["schedule"]], "training_performed": False,
        "optimizer_updates": 0, "optimizer_steps": 0, "training_samples": 0,
        "started_unix": wall_started, "deadline_unix": wall_deadline, "shared_wall_seconds": cfg["shared_wall_seconds"],
        "frozen_before": plan["frozen"], "wall_scope": plan["wall_scope"]}
    report_path = output / "report.json"
    write(report_path, report)
    stop_reason = None
    for index, row in enumerate(plan["schedule"]):
        if remaining(deadline, wall_deadline) <= cfg["cleanup_reserve_seconds"]:
            stop_reason = "shared_deadline"
            break
        entry = report["attempts"][index]
        entry.update(status="running", reason=None)
        write(report_path, report)
        command = [sys.executable, "-m", "experiments.learner.evaluate_builtin_opponent", "worker",
            "--plan", str(args.plan.resolve()), "--sha256", args.sha256, "--index", str(index),
            "--deadline", str(hard_deadline), "--wall-deadline", str(hard_wall)]
        process, signalled = None, False
        try:
            with (output / f"worker-{row['episode']:02d}.log").open("xb") as log:
                process = guarded_popen(command, deadline_monotonic=hard_deadline, deadline_wall=hard_wall,
                    evidence_path=output / f"worker-deadline-{row['episode']:02d}.jsonl",
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    left = remaining(deadline, wall_deadline)
                    if not signalled and left <= cfg["cleanup_reserve_seconds"]:
                        try:
                            os.kill(process.pid, signal.SIGUSR1)
                        except ProcessLookupError:
                            pass
                        signalled = True
                    if left <= cfg["hard_stop_reserve_seconds"]:
                        break
                    time.sleep(min(0.1, max(0.001, left - cfg["hard_stop_reserve_seconds"])))
            result_path = output / f"attempt-{row['episode']:02d}.json"
            require(result_path.exists(), "Worker exited without a durable result")
            result = json.loads(result_path.read_text())
            entry["last_worker_result"] = result
            require(all(result[key] == row[key] for key in row), "Worker result differs from fixed schedule")
            require(result.get("phase") == "finished" and process.returncode == 0, "Worker did not finish cleanup")
            report["attempts"][index] = result
            if not can_continue(result):
                stop_reason = "shared_deadline" if signalled or result["status"] == "budget_exhausted" else "attempt_requires_investigation"
        except Exception as error:
            entry.update(status="incomplete", reason="shared_deadline" if signalled else "supervisor_failure",
                         error=f"{type(error).__name__}: {error}")
            stop_reason = entry["reason"]
        finally:
            cleanup_errors = []
            if process is not None:
                try:
                    close_owned_process(process, timeout=max(0.1, min(1.0, remaining(deadline, wall_deadline))))
                except Exception as error:
                    cleanup_errors.append(f"Worker cleanup: {type(error).__name__}: {error}")
            try:
                groups = cleanup_from_evidence(output / f"native-deadline-{row['episode']:02d}.jsonl",
                    timeout=max(0.1, min(1.0, remaining(deadline, wall_deadline))))
                report["attempts"][index]["supervisor_native_groups_closed"] = list(groups)
            except Exception as error:
                cleanup_errors.append(f"Native cleanup: {type(error).__name__}: {error}")
            if cleanup_errors:
                report["attempts"][index].update(status="incomplete", supervisor_cleanup_errors=cleanup_errors,
                    outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
                stop_reason = "cleanup_failure"
            report["execution_wall_seconds"] = time.monotonic() - started
            write(report_path, report)
        if stop_reason:
            break
    for row in report["attempts"]:
        if row["status"] == "not_started":
            row["reason"] = stop_reason or "schedule_not_reached"
    report.update(status="finished" if stop_reason is None else "stopped", stop_reason=stop_reason,
                  execution_wall_seconds=time.monotonic() - started,
                  execution_wall_clock_seconds=time.time() - wall_started)
    write(report_path, report)
    # No native work below this line. Partial histories remain explicitly partial.
    offline_started = time.monotonic()
    from experiments.learner.builtin_behavior import analyze_episode, aggregate_episodes
    behaviors = []
    for row in report["attempts"]:
        folder = output / f"episode-{row['episode']:06d}"
        if folder.exists():
            row.setdefault("episode_directory", str(folder))
            try:
                behavior = analyze_episode(folder)
                write(folder / "behavior.json", behavior)
                row["behavior"] = binding(folder / "behavior.json")
                behaviors.append(behavior)
            except Exception as error:
                row["behavior_error"] = f"{type(error).__name__}: {error}"
    report.update(behavior_summary=aggregate_episodes(behaviors),
                  offline_report_seconds=time.monotonic() - offline_started,
                  frozen_after_file_sha256=digest(plan["checkpoint"]["path"]))
    write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("output", "checkpoint", "binary", "data", "replay-binary", "reference-plan", "reference-report"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    prep.add_argument("--revalidation-v2", action="store_true")
    prep.add_argument("--prior-plan", type=Path)
    prep.add_argument("--prior-report", type=Path)
    for mode in ("execute", "worker"):
        command = sub.add_parser(mode)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--sha256", required=True)
        if mode == "worker":
            command.add_argument("--index", type=int, required=True)
            command.add_argument("--deadline", type=float, required=True)
            command.add_argument("--wall-deadline", type=float, required=True)
    args = parser.parse_args()
    result = {"prepare": prepare, "execute": execute, "worker": worker}[args.mode](args)
    print(json.dumps(result if args.mode == "prepare" else {key: result.get(key) for key in
        ("status", "stop_reason", "execution_wall_seconds", "optimizer_steps")}))


if __name__ == "__main__":
    main()
