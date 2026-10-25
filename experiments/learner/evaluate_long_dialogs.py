"""Eight fixed v7 evaluations under one independently enforced execution hour.

Preparation is read-only with respect to the engine. Execution never constructs
an optimizer. Every scheduled row survives failures; no attempt is replaced.
The shared deadline includes checkpoint loading, collection, reconstruction,
native replay and cleanup. Final offline report assembly is timed separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/long-dialog-evaluation-v1.json"
INTERFACE = "adventure-dialogs-v1"
COLORS = ("red", "blue")
SCHEMA = "heroes-long-dialog-evaluation-v1"
V2_SCHEMA = "heroes-long-dialog-evaluation-v2"
CHECKPOINT_SHA = "4043b58bb3dc4eb6169514ca1ed7c1be0b373a5139efe04b95dfeda1832232c1"
# Only these native, episode-local failures may advance to the next fixed row.
LOCAL_FAILURES = frozenset(("unsupported_interface", "engine_error", "engine_crash",
    "startup_timeout", "watchdog_timeout", "bridge_disconnected", "action_delivery_failed"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    value = json.loads(CONFIG.read_text())
    expected = {"schema": "heroes-long-dialog-evaluation-config-v1", "interface": INTERFACE,
        "purpose": "evaluation_only", "engine_seeds": [120501, 120502, 120503, 120504],
        "model_policy_seeds": [[120601, 120602], [120603, 120604], [120605, 120606], [120607, 120608]],
        "max_total_choices": 4096, "ordinary_choice_limit": 4088, "reserved_dialog_choices": 8,
        "max_days": 200, "shared_wall_seconds": 3600, "cleanup_reserve_seconds": 30,
        "hard_stop_reserve_seconds": 2, "torch_threads": 1, "optimizer_updates": 0, "optimizer_steps": 0,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "learner_weights_sha256": "1c290bcb1ca37cb876841645fab5cd08a94f999bbc57a56fb285243a8e46bbae",
        "opponent": ["hero-opponent-7176f1a2e81393cb", "7176f1a2e81393cbfe709238fcd3c0060871a9ddaec1cf4181812f75fb08fc82"],
        "map_sha256": "852ea0211d20e1cbfb64790fe8076254e98983ba0ae29e854db192c816c76ad1"}
    require(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
            "Versioned eight-game scope changed; do not expand it through invocation flags")
    return value


def schedule(cfg):
    return [{"episode": 2 * index + seat + 1, "engine_seed": seed, "learner_player": seat,
             "policy_seeds": seeds if seat == 0 else seeds[::-1]}
            for index, (seed, seeds) in enumerate(zip(cfg["engine_seeds"], cfg["model_policy_seeds"]))
            for seat in (0, 1)]


def implementation(version=1):
    from experiments.learner.first_run import implementation_identity
    result = implementation_identity()
    for path in (CONFIG, ROOT / "experiments/explorer/create_hero_scenario.py"):
        result[str(path.relative_to(ROOT))] = digest(path)
    if version == 2:
        path = ROOT / "configs/builtin-opponent-evaluation-v1.json"
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
    from experiments.learner.checkpoints import load_policy_checkpoint
    from validation.probes.validate_hero_policy import frozen_fingerprints
    cfg = config()
    require(digest(path) == cfg["checkpoint_sha256"], "Only the previously validated v7 checkpoint is permitted")
    loaded = load_policy_checkpoint(path, expected_source_metadata=metadata, interface=INTERFACE,
                                   allowed_provenance=("random_initialization",))
    loaded.model.eval().requires_grad_(False)
    fixed = frozen_fingerprints(loaded, path)
    require(fixed["learner_weights_sha256"] == cfg["learner_weights_sha256"]
            and fixed["opponents"] == [cfg["opponent"]], "Frozen policy identities differ")
    return loaded, fixed


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
    from experiments.learner.evaluation_revalidation import AUTHORIZATION, prepare_provenance
    cfg = config()
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Preparation output exists; never replace or retry an earlier cohort")
    revalidation = prepare_provenance(args, reader=read_plan, legacy_schema=SCHEMA,
        configuration=cfg, wanted_schedule=schedule(cfg))
    version = 2 if revalidation is not None else 1
    torch.set_num_threads(1)
    checkpoint = binding(args.checkpoint)
    loaded, fixed = load_fixed(checkpoint["path"])
    if version == 1:
        freshness_path = ROOT / "runs/long-dialog-evaluation-preflight-2026-10-25/seed-freshness.json"
        freshness = json.loads(freshness_path.read_text())
        require(freshness["status"] == "passed" and freshness["candidate_seeds_are_absent"] is True
                and freshness["candidate_engine_seeds"] == cfg["engine_seeds"]
                and freshness["candidate_policy_seeds"] == [v for pair in cfg["model_policy_seeds"] for v in pair]
                and not freshness["parse_errors"], "Fresh seed preflight is missing or inconsistent")
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
        "checkpoint_source": loaded.source_metadata, "checkpoint_provenance": loaded.provenance,
        "source": current, "implementation": implementation(version), "frozen": fixed,
        "binary": str(args.binary.resolve()), "data": str(args.data.resolve()),
        "game_map": str(args.map.resolve()), "replay_binary": binding(args.replay_binary),
        "native": launcher.identity(), "runtime": runtime_identity(args.binary.resolve().parent), "map": scenario,
        "prepared_check": checked.stdout.strip(), "training_performed": False,
        "authorization": {"source_thread": "01a16d9e-5f8d-7cb0-a939-3ff005a741e4",
            "human_turn": "01a170b7-8d98-7fb3-b295-61048c116161",
            "human_message": "01a170b7-8db2-7150-bd30-fc1c70df691c", "confirmed_before_preparation": True},
        "wall_scope": "One shared 3600-second execution clock includes all launches, collection, reconstruction, native replay and cleanup. Final offline report assembly has a separate measured duration; it cannot launch games or replay.",
        "choice_scope": "4088 ordinary policy choices total across both players, plus at most 8 reserved real dialog choices, never more than 4096. Technical cutoff continuation is excluded from the learner prefix.",
        "failure_policy": "No retries/replacements. Preserved isolated native failures may advance; global deadline, unknown invariants, identity drift, cleanup or evidence errors abort remaining rows."}
    if version == 1:
        plan["seed_freshness"] = binding(freshness_path)
    else:
        require(current["tracked_worktree_dirty"] is False, "Revalidation source must be committed and clean")
        plan.update(revalidation=revalidation, authorization=dict(AUTHORIZATION))
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
        from experiments.learner.evaluation_revalidation import validate_provenance
        validate_provenance(plan, reader=read_plan, legacy_schema=SCHEMA)
        require(plan["source"].get("tracked_worktree_dirty") is False,
                "Revalidation source must be committed and clean")
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from experiments.learner.inference import source_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    version = plan_version(plan)
    require(implementation(version) == plan["implementation"] and source_identity(INTERFACE) == plan["source"],
            "Authored source changed after preparation")
    if version == 1:
        require(binding(plan["seed_freshness"]["path"]) == plan["seed_freshness"], "Seed preflight changed")
    else:
        from experiments.learner.evaluation_revalidation import validate_provenance
        validate_provenance(plan, reader=read_plan, legacy_schema=SCHEMA)
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
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.headless.contextual_replay import verify_episode
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.evaluation_behavior import analyze_episode
    from validation.probes.validate_hero_policy import audit_persisted_policy, frozen_fingerprints, retain_failed_evidence
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
        opponent_id = cfg["opponent"][0]
        opponent = loaded.opponent_pool.get(opponent_id)
        seat = row["learner_player"]
        models = (loaded.model, opponent.model) if seat == 0 else (opponent.model, loaded.model)
        checkpoints = (loaded.checkpoint, opponent_id) if seat == 0 else (opponent_id, loaded.checkpoint)
        launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]),
            deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
            evidence_path=output / f"native-deadline-{row['episode']:02d}.jsonl")
        sampled_by_player = [0, 0]

        class BoundedCollector(ContextualCollector):
            sampled = 0

            def act(self, request):
                require_choice_budget(self.sampled, request.record["day"], cfg)
                action = super().act(request)
                self.sampled += 1
                sampled_by_player[request.player] += 1
                if self.sampled % 64 == 0:
                    attempt.update(sampled_choices=self.sampled, sampled_by_player=list(sampled_by_player),
                                   latest_sampled_day=request.record["day"])
                    save("collection")
                return action

        collector = BoundedCollector(models, checkpoints, learner_player=seat,
            opponent_pool="long-dialog-frozen-v1", interface=INTERFACE, purpose="diagnostic", origin="native",
            policy_seeds=tuple(row["policy_seeds"]))
        limits = Limits(max_days=cfg["max_days"], max_actions=cfg["max_total_choices"] + 1,
                        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30)
        env = HeadlessSelfPlayEnv(launcher, output, limits=limits, ledger=True, replay=True, interface=INTERFACE,
            agents={color: {"checkpoint": key, "kind": "fixed-untrained-recurrent", "evaluation_only": True}
                    for color, key in zip(COLORS, checkpoints)})
        save("collection")
        collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                          max_requests=cfg["ordinary_choice_limit"])
        folder = env.episode_dir
        write(folder / "neural-rollout.json", collected.to_dict())
        manifest = json.loads((folder / "manifest.json").read_text())
        count = sum(len(r.decisions) for r in collected.rollouts)
        require(count <= cfg["max_total_choices"], "Shared total policy-choice cap exceeded")
        require(all(not r.optimization_eligible and r.purpose == "diagnostic" for r in collected.rollouts),
                "Evaluation became optimizer eligible")
        require(collector.sessions[0]._state.data_ptr() != collector.sessions[1]._state.data_ptr(),
                "Player memories share storage")
        attempt.update(episode_directory=str(folder), episode_status=manifest["status"],
            termination_reason=manifest["termination_reason"], sampled_choices=count,
            sampled_by_player=[len(r.decisions) for r in collected.rollouts],
            latest_sampled_day=max((r.history[d.decision_seq].day for r in collected.rollouts for d in r.decisions), default=None),
            private_player_memories=True, environment_limits=asdict(limits))
        if manifest["status"] == "failed":
            attempt.update(status="native_failure", error=manifest["failure_detail"])
        else:
            require(manifest["status"] in ("completed", "truncated"), "Unknown episode boundary")
            save("policy_reconstruction")
            reloaded, _ = load_fixed(plan["checkpoint"]["path"], plan["checkpoint_source"])
            reconstruction = audit_persisted_policy(folder, reloaded, row["policy_seeds"], interface=INTERFACE)
            write(folder / "policy-reconstruction.json", reconstruction)
            save("native_replay")
            seconds = remaining(args.deadline, args.wall_deadline) - cfg["cleanup_reserve_seconds"]
            if seconds <= 0:
                raise ExecutionBudgetExpired("No shared time remains for native replay")
            replay = verify_episode(folder, binary=Path(plan["replay_binary"]["path"]),
                                    data=Path(plan["data"]), timeout=seconds)
            require(replay.get("verified") is True, "Independent native replay failed")
            attempt.update(status="passed", outcomes=manifest["outcomes_by_player"],
                rewards=manifest["rewards_by_player"], reconstruction=reconstruction["players"], replay_verified=True)
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
            retain_failed_evidence(env, collector, attempt, error, cleanup_seconds=seconds)
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
        command = [sys.executable, "-m", "experiments.learner.evaluate_long_dialogs", "worker",
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
    from experiments.learner.evaluation_behavior import analyze_episode, aggregate_episodes
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
    for name in ("output", "checkpoint", "binary", "data", "replay-binary"):
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
