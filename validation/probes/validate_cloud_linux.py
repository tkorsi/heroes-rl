"""Execute a fresh Linux frozen-policy diagnostic plan, never a training run.

The four predeclared rows share one clock. A durable worker receipt is required
after cleanup and final identity checks; an interrupted worker cannot publish
an admitted outcome. Raw captures and these reports belong in private storage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from experiments.headless.contextual_evidence import strict_json
from experiments.headless import process_deadline as guard
from tools import linux_validation_plan as planning

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("red", "blue")
SCHEMA = "heroes-cloud-linux-diagnostic-v1-result"
WORKER_RECEIPT = "heroes-cloud-linux-worker-completion-v1"
CAMPAIGN_RECEIPT = "heroes-cloud-linux-campaign-completion-v1"
LAUNCH_SCHEMA = "heroes-cloud-linux-worker-launch-v1"
VERIFIED = frozenset(("verified_natural", "verified_cutoff"))
FINAL_STATUSES = VERIFIED | {"native_failure", "failed", "budget_exhausted"}


class BudgetExpired(TimeoutError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def remaining(deadline, wall_deadline):
    return min(deadline - time.monotonic(), wall_deadline - time.time())


def binding(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def write(path, value, *, exclusive=False):
    """Fsync file and directory; completion receipts are never overwritten."""
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def until(deadline, wall_deadline):
    """Bound blocking Python work to an existing deadline, without fresh time."""
    seconds = remaining(deadline, wall_deadline)
    if seconds <= 0:
        raise BudgetExpired("The existing diagnostic deadline has expired")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    require(previous_timer == (0.0, 0.0), "Nested diagnostic deadline alarms are unsupported")
    def expired(signum, frame):
        raise BudgetExpired("The existing diagnostic deadline has expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        if remaining(deadline, wall_deadline) <= 0:
            raise BudgetExpired("The diagnostic operation exceeded its deadline")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def empty_attempt(row):
    return {**row, "status": "not_started", "reason": "schedule_not_reached",
        "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS),
        "training_performed": False, "training_samples": 0, "optimizer_steps": 0,
        "optimization_eligible": False, "fixture": row["map_kind"] == "fixture"}


def paths(output, episode):
    output = Path(output)
    return (output / f"attempt-{episode:02d}.json", output / f"completion-{episode:02d}.json")


def launch_declaration(plan, sha256, index, started, wall_started):
    """Freeze one row's clocks before any child can launch native work."""
    cfg, row = plan["configuration"], plan["schedule"][index]
    row_started, row_wall = time.monotonic(), time.time()
    deadline, wall_deadline = started + cfg["shared_wall_seconds"], wall_started + cfg["shared_wall_seconds"]
    row_end, row_wall_end = min(deadline, row_started + row["case_wall_seconds"]), min(wall_deadline, row_wall + row["case_wall_seconds"])
    return {"schema": LAUNCH_SCHEMA, "plan_sha256": sha256, "index": index, "row": dict(row),
        "campaign_started_monotonic": started, "campaign_started_unix": wall_started,
        "campaign_deadline_monotonic": deadline, "campaign_deadline_unix": wall_deadline,
        "row_started_monotonic": row_started, "row_started_unix": row_wall,
        "row_deadline_monotonic": row_end, "row_deadline_unix": row_wall_end,
        "deadline": row_end - cfg["hard_stop_reserve_seconds"],
        "wall_deadline": row_wall_end - cfg["hard_stop_reserve_seconds"],
        "work_deadline": row_end - cfg["cleanup_reserve_seconds"],
        "work_wall_deadline": row_wall_end - cfg["cleanup_reserve_seconds"]}


def validate_worker_launch(plan, args):
    """Internal workers cannot replace the supervisor's clock with CLI limits."""
    output, cfg = Path(plan["execution_output"]), plan["configuration"]
    row = plan["schedule"][args.index]
    launch_path = output / f"launch-{row['episode']:02d}.json"
    launch = strict_json(launch_path.read_bytes())
    report = strict_json((output / "report.json").read_bytes())
    fields = {"schema", "plan_sha256", "index", "row", "campaign_started_monotonic", "campaign_started_unix",
        "campaign_deadline_monotonic", "campaign_deadline_unix", "row_started_monotonic", "row_started_unix",
        "row_deadline_monotonic", "row_deadline_unix", "deadline", "wall_deadline", "work_deadline", "work_wall_deadline"}
    require(set(launch) == fields and launch["schema"] == LAUNCH_SCHEMA and launch["plan_sha256"] == args.sha256
            and equal(launch["index"], args.index) and equal(launch["row"], row), "Worker has no matching launch declaration")
    clocks = fields - {"schema", "plan_sha256", "index", "row"}
    require(all(type(launch[key]) in (int, float) and math.isfinite(launch[key]) and launch[key] > 0 for key in clocks),
            "Worker launch clocks must be finite positive timestamps")
    require(report.get("schema") == SCHEMA and report.get("plan_sha256") == args.sha256
            and report.get("status") == "running" and len(report.get("attempts", [])) == len(plan["schedule"]),
            "Worker requires its active supervisor campaign")
    current = report["attempts"][args.index]
    require(current.get("status") == "pending_verification" and equal(current.get("launch"), binding(launch_path))
            and all(equal(current.get(key), value) for key, value in row.items()),
            "Worker launch is not the supervisor's pending row")
    for suffix, report_start, report_end in (("monotonic", "started_monotonic", "deadline_monotonic"),
                                            ("unix", "started_unix", "deadline_unix")):
        begin, end = launch["campaign_started_" + suffix], launch["campaign_deadline_" + suffix]
        row_begin, row_end = launch["row_started_" + suffix], launch["row_deadline_" + suffix]
        require(equal(begin, report.get(report_start)) and equal(end, report.get(report_end))
                and end == begin + cfg["shared_wall_seconds"] and begin <= row_begin < end
                and row_end == min(end, row_begin + row["case_wall_seconds"]),
                "Worker row exceeds the declared shared or case clock")
    require(launch["deadline"] == launch["row_deadline_monotonic"] - cfg["hard_stop_reserve_seconds"]
            and launch["wall_deadline"] == launch["row_deadline_unix"] - cfg["hard_stop_reserve_seconds"]
            and launch["work_deadline"] == launch["row_deadline_monotonic"] - cfg["cleanup_reserve_seconds"]
            and launch["work_wall_deadline"] == launch["row_deadline_unix"] - cfg["cleanup_reserve_seconds"],
            "Worker cleanup reserves differ from the plan")
    require(all(equal(getattr(args, key), launch[key]) for key in
                ("deadline", "wall_deadline", "work_deadline", "work_wall_deadline")),
            "Worker CLI clocks differ from the supervisor declaration")
    require(remaining(args.work_deadline, args.work_wall_deadline) > 0, "Worker work clock already expired")


def unscore(result, status, **fields):
    result.update(status=status, outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS), **fields)


def own_history_coverage(frames, *, player, episode):
    """Count only delivered own frames; disappearance never supplies its cause."""
    counts = Counter()
    previous_decision = None
    intermediate = set()
    interval_views = interval_events = intervals = transient = 0
    for frame in frames:
        require(frame.player == player and frame.episode == str(episode), "Coverage mixed player or episode histories")
        counts[frame.kind] += 1
        if frame.kind == "event" and previous_decision is not None:
            interval_events += 1
        if frame.kind not in ("view", "decision"):
            continue
        opponents = {item.handle for item in frame.observation.objects if item.owner == "opponent"}
        if frame.kind == "view" and previous_decision is not None:
            interval_views += 1
            intermediate.update(opponents)
        elif frame.kind == "decision":
            if previous_decision is not None:
                intervals += 1
                transient += len(intermediate - previous_decision - opponents)
            previous_decision, intermediate = opponents, set()
    return {"player": player, "episode": str(episode), "frame_counts": dict(counts),
        "completed_decision_intervals": intervals, "views_after_first_decision": interval_views,
        "events_after_first_decision": interval_events,
        "opponent_handle_sightings_absent_at_both_surrounding_decisions": transient,
        "transient_sighting_observed": transient > 0,
        "scope": "Delivered own frames after the first decision; transient counts use complete decision intervals. No hidden state, fate, or opponent-turn attribution."}


def capture_bindings(folder, player):
    """Bind retained evidence, excluding the profile's private asset links."""
    folder = Path(folder).resolve()
    color = COLORS[player]
    required = {"manifest.json", "builtin-finalization.json", "builtin-rollout.json", "policy-reconstruction.json",
        "behavior.json", "logs/process.log", "engine/experiment-metadata.json", "engine/replay.jsonl",
        "engine/contextual-replay.jsonl", f"agents/{color}/bridge.jsonl", f"agents/{color}/history-v7.jsonl",
        f"agents/{color}/live-history-v7.jsonl", "builtin-verification/native-report.json",
        "builtin-verification/builtin-replay-verification.json"}
    result = {}
    for path in sorted(folder.rglob("*")):
        name = path.relative_to(folder).as_posix()
        if name == "profile" or name.startswith("profile/"):
            continue
        require(not path.is_symlink(), "Capture evidence cannot be a symbolic link: " + name)
        if path.is_file():
            result[name] = binding(path)
    require(required <= set(result), "Required diagnostic capture evidence is missing")
    return result


def verify_saved_evidence(result, output):
    """Recheck bound bytes and native report semantics without executing replay."""
    from experiments.headless import builtin_replay
    folder = Path(output).resolve() / f"episode-{result['episode']:06d}"
    require(result.get("episode_directory") == str(folder), "Worker evidence is outside its episode")
    current = capture_bindings(folder, result["learner_player"])
    require(equal(current, result.get("evidence_bindings")), "Retained diagnostic evidence changed after completion")
    for key, name in (("manifest", "manifest.json"), ("replay", "builtin-verification/builtin-replay-verification.json"),
                      ("behavior", "behavior.json")):
        require(equal(result.get(key), current[name]), "Result artifact binding differs: " + key)
    manifest = strict_json((folder / "manifest.json").read_bytes())
    require(equal(manifest.get("learner_player"), result["learner_player"])
            and manifest.get("episode") == str(result["episode"])
            and manifest.get("status") == result["episode_status"]
            and manifest.get("termination_reason") == result["termination_reason"]
            and equal(manifest.get("outcomes_by_player"), result["outcomes"])
            and equal(manifest.get("rewards_by_player"), result["rewards"]), "Saved manifest and result differ")
    report = strict_json((folder / "builtin-verification/builtin-replay-verification.json").read_bytes())
    require(report.get("schema") == builtin_replay.REPORT_SCHEMA
            and report.get("kind") == "builtin-captured-command-replay" and report.get("verified") is True
            and report.get("training_performed") is False and report.get("optimization_eligible") is False
            and report.get("opponent_policy_reexecuted") is False
            and type(report.get("training_samples")) is int and report["training_samples"] == 0,
            "Saved report is not a verified frozen-policy native replay")
    require(equal(report.get("inputs", {}).get("manifest"),
                  {k: current["manifest.json"][k] for k in ("sha256", "bytes")}),
            "Replay report does not bind this manifest")
    require(equal(result.get("request_accounting"), report.get("wire", {}).get("request_accounting")),
            "Saved native request accounting differs from the result")
    final = strict_json((folder / "builtin-finalization.json").read_bytes())
    builtin_replay.validate_native_report(report["native"], report["wire"], final,
                                         cutoff=result["status"] == "verified_cutoff")
    require(equal(strict_json((folder / "policy-reconstruction.json").read_bytes()), result["reconstruction"]),
            "Saved policy reconstruction differs from the result")


def check_result(result, row):
    require(all(equal(result.get(key), value) for key, value in row.items()), "Worker schedule changed")
    require(result.get("fixture") is (row["map_kind"] == "fixture"), "Fixture label differs from the declared map")
    require(result.get("phase") == "finished" and result.get("status") in FINAL_STATUSES,
            "Worker has no final diagnostic result")
    require(result.get("training_performed") is False and result.get("optimization_eligible") is False
            and all(type(result.get(k)) is int and result[k] == 0 for k in ("optimizer_steps", "training_samples")),
            "Diagnostic result is not frozen evaluation")
    require(all(result.get(k) is True for k in ("cleanup_verified", "identity_verified", "weights_unchanged")),
            "Worker final cleanup or identities were not verified")
    if result["status"] in VERIFIED:
        reconstruction = result.get("reconstruction", {})
        require(reconstruction.get("passed") is True and reconstruction.get("private_memory_reset") is True
                and reconstruction.get("weights_unchanged") is True
                and equal(reconstruction.get("maximum_absolute_difference"), 0),
                "Verified row lacks exact reconstruction with fresh own memory")
    if result["status"] == "verified_natural":
        outcomes = result.get("outcomes", {})
        require(set(outcomes) == set(COLORS) and set(outcomes.values()) == {"win", "loss"}
                and equal(result.get("rewards"), {k: 1 if v == "win" else -1 for k, v in outcomes.items()})
                and result.get("episode_status") == "completed"
                and result.get("termination_reason") == "natural_terminal"
                and result.get("replay_verified") is True, "Natural outcome lacks its verification")
    else:
        require(equal(result.get("outcomes"), dict.fromkeys(COLORS))
                and equal(result.get("rewards"), dict.fromkeys(COLORS)), "Unscored row acquired an outcome")
        if result["status"] == "verified_cutoff":
            require(result.get("episode_status") == "truncated" and result.get("replay_verified") is True,
                    "Cutoff lacks independent verification")


def publish_completion(output, result, *, plan_sha256, index, deadline=None, wall_deadline=None):
    """Called only after final cleanup and identity verification, for any result."""
    path, receipt = paths(output, result["episode"])
    if deadline is not None and remaining(deadline, wall_deadline) <= 0:
        raise BudgetExpired("Completion cannot publish after the existing worker deadline")
    write(path, result)
    if all(result.get(k) is True for k in ("cleanup_verified", "identity_verified", "weights_unchanged")):
        if deadline is not None and remaining(deadline, wall_deadline) <= 0:
            raise BudgetExpired("Result persistence exhausted the existing worker deadline")
        write(receipt, {"schema": WORKER_RECEIPT, "plan_sha256": plan_sha256,
            "index": index, "episode": result["episode"], "result": binding(path),
            "finished_unix": time.time(), "cleanup_verified": True, "identity_verified": True,
            "weights_unchanged": True}, exclusive=True)


def read_completion(output, row, *, plan_sha256, index, returncode):
    require(returncode == 0, "Worker did not exit successfully after completing its receipt")
    path, receipt_path = paths(output, row["episode"])
    receipt = strict_json(receipt_path.read_bytes())
    require(set(receipt) == {"schema", "plan_sha256", "index", "episode", "result", "finished_unix",
                            "cleanup_verified", "identity_verified", "weights_unchanged"}
            and receipt["schema"] == WORKER_RECEIPT and receipt["plan_sha256"] == plan_sha256
            and equal(receipt["index"], index) and equal(receipt["episode"], row["episode"])
            and all(receipt[k] is True for k in ("cleanup_verified", "identity_verified", "weights_unchanged"))
            and equal(receipt["result"], binding(path)), "Worker completion receipt differs from its bound result")
    result = strict_json(path.read_bytes())
    require(result.get("plan_sha256") == plan_sha256, "Worker result plan binding differs")
    check_result(result, row)
    if result["status"] in VERIFIED:
        verify_saved_evidence(result, output)
    return result


def worker(args):
    # A supervisor may ask for cleanup while a dependency is importing.
    def stop_work(signum, frame):
        raise BudgetExpired("Diagnostic work stopped to preserve shared cleanup time")
    plan = planning.read_plan(args.plan, args.sha256)
    require(type(args.index) is int and 0 <= args.index < len(plan["schedule"]), "Invalid worker index")
    validate_worker_launch(plan, args)
    row, output = plan["schedule"][args.index], Path(plan["execution_output"])
    result = {**empty_attempt(row), "status": "pending_verification", "phase": "initialization",
              "plan_sha256": args.sha256}
    result.pop("reason")
    result_path, receipt_path = paths(output, row["episode"])
    require(not result_path.exists() and not receipt_path.exists(), "Never resume or replace a worker attempt")
    # Claim the row atomically before loading a model or creating any launcher.
    write(result_path, result, exclusive=True)
    old_signal = signal.signal(signal.SIGUSR1, stop_work)
    started, env, collector, launcher, loaded = time.monotonic(), None, None, None, None
    candidate = {"status": "failed", "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
    phase_started = started
    phase_seconds = {}

    def save(phase):
        nonlocal phase_started
        now = time.monotonic()
        previous = result["phase"]
        phase_seconds[previous] = phase_seconds.get(previous, 0.0) + now - phase_started
        phase_started = now
        result.update(phase=phase, wall_seconds=now - started, phase_wall_seconds=dict(phase_seconds))
        # Candidate scores stay in memory until post-run cleanup and identity.
        write(result_path, result)

    def save_without_blocking_cleanup(phase):
        try:
            save(phase)
        except Exception as error:
            result.setdefault("progress_write_errors", []).append(f"{phase}: {type(error).__name__}: {error}")

    try:
        with until(args.work_deadline, args.work_wall_deadline):
            import torch
            from experiments.headless.builtin_env import BuiltinHeadlessEnv
            from experiments.headless.builtin_replay import verify_episode
            from experiments.headless.env import Limits
            from experiments.learner.builtin_rollout import BuiltinCollector, audit_persisted_builtin_policy
            from experiments.learner.builtin_behavior import analyze_episode

            torch.set_num_threads(1)
            save("identity_check")
            planning.check_identity(plan, full_data=True)
            loaded, fixed = planning.load_fixed(plan)
            require(equal(fixed, plan["frozen"]), "Frozen model identity differs from the plan")
            seat = row["learner_player"]
            opponent = plan["native_opponents"][seat]
            launcher = guard.DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]),
                Path(plan["maps"][row["map_kind"]]["path"]), deadline_monotonic=args.deadline,
                deadline_wall=args.wall_deadline,
                evidence_path=output / f"native-deadline-{row['episode']:02d}.jsonl")

            class BoundedCollector(BuiltinCollector):
                sampled = 0

                def act(self, request):
                    require(self.sampled < row["max_total_choices"] and request.record["day"] <= row["max_days"],
                            "Predeclared policy choice/day limit exceeded")
                    if remaining(args.work_deadline, args.work_wall_deadline) <= 0:
                        raise BudgetExpired("No shared work time remains for another policy choice")
                    action = super().act(request)
                    self.sampled += 1
                    if self.sampled % 64 == 0:
                        result.update(sampled_choices=self.sampled, latest_sampled_day=request.record["day"])
                        save("collection")
                    return action

            collector = BoundedCollector(loaded.model, loaded.checkpoint, learner_player=seat,
                opponent_identity=opponent, policy_seed=row["policy_seed"])
            limits = Limits(max_days=row["max_days"], max_actions=row["max_total_choices"],
                watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30)
            env = BuiltinHeadlessEnv(launcher, output, learner_player=seat, limits=limits,
                ledger=True, replay=True, interface="adventure-dialogs-v1",
                agents={COLORS[seat]: {"checkpoint": loaded.checkpoint,
                    "kind": "frozen-linux-diagnostic", "evaluation_only": True}})
            save("collection")
            collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                              max_requests=row["ordinary_choice_limit"])
            folder = env.episode_dir
            write(folder / "builtin-rollout.json", collected.to_dict())
            manifest = strict_json((folder / "manifest.json").read_bytes())
            require(equal(manifest, collected.manifest), "Persisted manifest differs from collected evidence")
            require(collected.optimization_eligible is False and len(collected.decisions) <= row["max_total_choices"],
                    "Diagnostic collection exceeded its scope")
            require(len(collector.sessions) == 1 and collector.session.history.player == seat
                    and collector.session.history.episode == str(row["episode"]),
                    "Only this episode's own learner history is permitted")
            result.update(episode_directory=str(folder), episode_status=manifest["status"],
                termination_reason=manifest["termination_reason"], sampled_choices=len(collected.decisions),
                latest_sampled_day=max((collected.history[d.decision_seq].day for d in collected.decisions), default=None),
                environment_limits=asdict(limits), opponent_identity=opponent, external_player_histories=1,
                opponent_history=None, manifest=binding(folder / "manifest.json"))
            if manifest["status"] == "failed":
                candidate.update(status="native_failure", error=manifest.get("failure_detail"))
            else:
                require(manifest["status"] in ("completed", "truncated"), "Unknown native episode boundary")
                save("policy_reconstruction")
                rebuilt, _ = planning.load_fixed(plan)
                reconstruction = audit_persisted_builtin_policy(folder, rebuilt.model, rebuilt.checkpoint,
                                                                row["policy_seed"], opponent)
                require(reconstruction.get("passed") is True and reconstruction.get("private_memory_reset") is True,
                        "Independent reconstruction did not verify fresh own memory")
                write(folder / "policy-reconstruction.json", reconstruction)
                save("native_replay")
                seconds = remaining(args.work_deadline, args.work_wall_deadline)
                if seconds <= 0:
                    raise BudgetExpired("No existing work budget remains for independent replay")
                replay = verify_episode(folder, binary=Path(plan["replay_binary"]["path"]),
                                        data=Path(plan["data"]), timeout=seconds)
                require(replay.get("verified") is True, "Independent native replay did not verify every request")
                result.update(replay_verified=True, reconstruction=reconstruction,
                    own_history_coverage=own_history_coverage(collected.history, player=seat, episode=row["episode"]),
                    request_accounting=replay["wire"]["request_accounting"],
                    replay=binding(folder / "builtin-verification/builtin-replay-verification.json"))
                if manifest["status"] == "completed":
                    candidate.update(status="verified_natural", outcomes=manifest["outcomes_by_player"],
                                     rewards=manifest["rewards_by_player"])
                else:
                    candidate.update(status="verified_cutoff")
            save("own_history_analysis")
            behavior = analyze_episode(folder)
            write(folder / "behavior.json", behavior)
            result["behavior"] = binding(folder / "behavior.json")
    except Exception as error:
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        candidate = {"status": "budget_exhausted" if isinstance(error, BudgetExpired) else "failed",
            "error": f"{type(error).__name__}: {error}",
            "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
        # No new model choices or replay after work expiry. Retain the permitted
        # partial history only while the same cleanup clock still has time.
        save_without_blocking_cleanup("failure_retention")
        try:
            if env is not None and collector is not None:
                from experiments.learner.builtin_rollout import retain_failed_builtin_evidence
                # This helper has its own alarm around environment finalization;
                # nesting another SIGALRM would prevent it from retaining it.
                if remaining(args.deadline, args.wall_deadline) > 0:
                    retain_failed_builtin_evidence(env, collector, result, error,
                        cleanup_seconds=remaining(args.deadline, args.wall_deadline))
        except Exception as retained:
            result["failure_retention_error"] = f"{type(retained).__name__}: {retained}"
    finally:
        # Ignore repeated stop-work requests while using the original cleanup
        # reserve. The independent worker guardian still enforces its hard end.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        save_without_blocking_cleanup("cleanup")
        try:
            with until(args.deadline, args.wall_deadline):
                if env is not None:
                    env.close()
            result["environment_cleanup_verified"] = True
        except Exception as error:
            result["environment_cleanup_error"] = f"{type(error).__name__}: {error}"
        try:
            if launcher is not None:
                launcher.close_owned_groups(timeout=max(0, remaining(args.deadline, args.wall_deadline)),
                    deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline)
            result["cleanup_verified"] = result.get("environment_cleanup_verified") is True
        except Exception as error:
            result.update(cleanup_verified=False, cleanup_error=f"{type(error).__name__}: {error}")
        try:
            with until(args.deadline, args.wall_deadline):
                planning.check_identity(plan, full_data=True)
                if loaded is not None:
                    from validation.probes.validate_hero_policy import frozen_fingerprints
                    require(equal(frozen_fingerprints(loaded, plan["checkpoint"]["path"]), plan["frozen"]),
                            "In-memory learner or archived opponent weights changed")
                if candidate["status"] in VERIFIED and env is not None:
                    result["evidence_bindings"] = capture_bindings(env.episode_dir, row["learner_player"])
                result.update(identity_verified=True, weights_unchanged=True)
        except Exception as error:
            result.update(identity_verified=False, weights_unchanged=False,
                          identity_error=f"{type(error).__name__}: {error}")
        if env is not None and env.episode_dir is not None:
            result["episode_directory"] = str(env.episode_dir)
        if not result.get("progress_write_errors") and all(
                result.get(k) is True for k in ("cleanup_verified", "identity_verified", "weights_unchanged")):
            result.update(candidate)
        else:
            unscore(result, "failed", candidate_failure=candidate.get("error"))
        result.update(phase="finished", wall_seconds=time.monotonic() - started)
        try:
            publish_completion(output, result, plan_sha256=args.sha256, index=args.index,
                               deadline=args.deadline, wall_deadline=args.wall_deadline)
        finally:
            signal.signal(signal.SIGUSR1, old_signal)
    return result


def _last_worker_result(output, episode):
    path, _ = paths(output, episode)
    try:
        return strict_json(path.read_bytes())
    except (OSError, ValueError):
        return None


def execute(args):
    plan = planning.read_plan(args.plan, args.sha256)
    output, cfg = Path(plan["execution_output"]), plan["configuration"]
    output.mkdir(exist_ok=False)  # An interrupted run cannot resume or replace rows.
    started, wall_started = time.monotonic(), time.time()
    deadline, wall_deadline = started + cfg["shared_wall_seconds"], wall_started + cfg["shared_wall_seconds"]
    report = {"schema": SCHEMA, "plan_sha256": args.sha256, "status": "running",
        "attempts": [empty_attempt(row) for row in plan["schedule"]],
        "started_monotonic": started, "deadline_monotonic": deadline,
        "started_unix": wall_started, "deadline_unix": wall_deadline,
        "shared_wall_seconds": cfg["shared_wall_seconds"], "training_performed": False,
        "optimizer_steps": 0, "training_samples": 0, "optimization_eligible": False,
        "completion_receipt_required": True}
    report_path = output / "report.json"
    write(report_path, report)
    stop_reason = None
    try:
        with until(deadline - cfg["cleanup_reserve_seconds"], wall_deadline - cfg["cleanup_reserve_seconds"]):
            planning.check_identity(plan, full_data=True)
    except Exception as error:
        stop_reason = "preflight_failed"
        report["preflight_error"] = f"{type(error).__name__}: {error}"
    for index, row in enumerate(plan["schedule"]):
        if stop_reason:
            break
        if remaining(deadline, wall_deadline) <= cfg["cleanup_reserve_seconds"]:
            stop_reason = "shared_deadline"
            break
        launch = launch_declaration(plan, args.sha256, index, started, wall_started)
        launch_path = output / f"launch-{row['episode']:02d}.json"
        write(launch_path, launch, exclusive=True)
        case_deadline, case_wall = launch["row_deadline_monotonic"], launch["row_deadline_unix"]
        hard_deadline, hard_wall = launch["deadline"], launch["wall_deadline"]
        work_deadline, work_wall = launch["work_deadline"], launch["work_wall_deadline"]
        entry = report["attempts"][index]
        entry.update(status="pending_verification", reason=None, launch=binding(launch_path))
        write(report_path, report)
        command = [sys.executable, "-m", "validation.probes.validate_cloud_linux", "worker",
            "--plan", str(Path(args.plan).resolve()), "--sha256", args.sha256, "--index", str(index),
            "--deadline", repr(hard_deadline), "--wall-deadline", repr(hard_wall),
            "--work-deadline", repr(work_deadline), "--work-wall-deadline", repr(work_wall)]
        process, signalled, cleanup_errors = None, False, []
        candidate = None
        try:
            with (output / f"worker-{row['episode']:02d}.log").open("xb") as log:
                process = guard.guarded_popen(command, deadline_monotonic=hard_deadline, deadline_wall=hard_wall,
                    evidence_path=output / f"worker-deadline-{row['episode']:02d}.jsonl", cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if not signalled and remaining(work_deadline, work_wall) <= 0:
                        try:
                            os.kill(process.pid, signal.SIGUSR1)
                        except ProcessLookupError:
                            pass
                        signalled = True
                    seconds = remaining(hard_deadline, hard_wall)
                    if seconds <= 0:
                        break
                    try:
                        process.wait(timeout=min(0.1, seconds))
                    except subprocess.TimeoutExpired:
                        pass
            with until(case_deadline, case_wall):
                candidate = read_completion(output, row, plan_sha256=args.sha256, index=index, returncode=process.poll())
        except Exception as error:
            unscore(entry, "incomplete", error=f"{type(error).__name__}: {error}")
            stop_reason = "shared_deadline" if remaining(deadline, wall_deadline) <= cfg["cleanup_reserve_seconds"] else (
                "case_deadline" if signalled else "worker_incomplete")
        finally:
            if process is not None:
                try:
                    guard.close_owned_process(process, timeout=max(0, remaining(case_deadline, case_wall)),
                        deadline_monotonic=case_deadline, deadline_wall=case_wall)
                except Exception as error:
                    cleanup_errors.append(f"Worker cleanup: {type(error).__name__}: {error}")
            try:
                groups = guard.cleanup_from_evidence(output / f"native-deadline-{row['episode']:02d}.jsonl",
                    timeout=max(0, remaining(case_deadline, case_wall)),
                    deadline_monotonic=case_deadline, deadline_wall=case_wall)
                entry["supervisor_native_groups_closed"] = list(groups)
            except Exception as error:
                cleanup_errors.append(f"Native cleanup: {type(error).__name__}: {error}")
            if cleanup_errors:
                unscore(entry, "incomplete", supervisor_cleanup_errors=cleanup_errors)
                stop_reason = "cleanup_failure"
            elif remaining(case_deadline, case_wall) <= 0:
                unscore(entry, "incomplete", reason="case_deadline_before_admission")
                stop_reason = "case_deadline"
            elif candidate is not None:
                entry.update(candidate, supervisor_native_groups_closed=entry["supervisor_native_groups_closed"])
                if candidate["status"] not in VERIFIED:
                    stop_reason = "attempt_requires_investigation"
            if entry["status"] not in VERIFIED:
                entry["last_worker_result"] = _last_worker_result(output, row["episode"])
            report["execution_wall_seconds"] = time.monotonic() - started
            write(report_path, report)
    for row in report["attempts"]:
        if row["status"] == "not_started":
            row["reason"] = stop_reason or "schedule_not_reached"
    try:
        with until(deadline, wall_deadline):
            planning.check_identity(plan, full_data=True)
        report["final_identity_verified"] = True
    except Exception as error:
        report.update(final_identity_verified=False, final_identity_error=f"{type(error).__name__}: {error}")
        stop_reason = stop_reason or "final_identity_failed"
        for row in report["attempts"]:
            if row["status"] in VERIFIED:
                row["unadmitted_worker_result"] = _last_worker_result(output, row["episode"])
                unscore(row, "incomplete", reason="campaign_final_identity_failed")
    report.update(status="finished" if stop_reason is None else "stopped", stop_reason=stop_reason,
        execution_wall_seconds=time.monotonic() - started,
        execution_wall_clock_seconds=time.time() - wall_started,
        deadline_exceeded=remaining(deadline, wall_deadline) < 0)
    write(report_path, report)
    if report["final_identity_verified"] and remaining(deadline, wall_deadline) > 0 and not any(
            r.get("supervisor_cleanup_errors") for r in report["attempts"]):
        write(output / "completion.json", {"schema": CAMPAIGN_RECEIPT, "plan_sha256": args.sha256,
            "report": binding(report_path), "finished_unix": time.time(), "finished_monotonic": time.monotonic(),
            "final_identity_verified": True, "cleanup_verified": True}, exclusive=True)
    if remaining(deadline, wall_deadline) <= 0:
        # Keep any written receipt as evidence, but invalidate its report hash.
        # A late fsync must not leave an apparently admitted campaign behind.
        report.update(status="stopped", stop_reason="shared_deadline", deadline_exceeded=True,
            execution_wall_seconds=time.monotonic() - started,
            execution_wall_clock_seconds=time.time() - wall_started)
        for row in report["attempts"]:
            if row["status"] in VERIFIED:
                unscore(row, "incomplete", reason="completion_after_shared_deadline")
        write(report_path, report)
    return report


def read_report(plan_path, sha256):
    """Read only a durably completed campaign; never admit interrupted reports."""
    plan = planning.read_plan(plan_path, sha256)
    output = Path(plan["execution_output"])
    receipt = strict_json((output / "completion.json").read_bytes())
    require(receipt.get("schema") == CAMPAIGN_RECEIPT and receipt.get("plan_sha256") == sha256
            and receipt.get("final_identity_verified") is True and receipt.get("cleanup_verified") is True
            and equal(receipt.get("report"), binding(output / "report.json")), "Campaign completion receipt changed")
    report = strict_json((output / "report.json").read_bytes())
    require(report.get("schema") == SCHEMA and report.get("plan_sha256") == sha256
            and report.get("status") in ("finished", "stopped")
            and report.get("deadline_exceeded") is False
            and len(report.get("attempts", [])) == len(plan["schedule"]), "Campaign report changed")
    require(all(type(receipt.get(key)) in (int, float) and math.isfinite(receipt[key])
                for key in ("finished_unix", "finished_monotonic"))
            and report["started_unix"] <= receipt["finished_unix"] <= report["deadline_unix"]
            and report["started_monotonic"] <= receipt["finished_monotonic"] <= report["deadline_monotonic"],
            "Campaign completion receipt is outside its declared shared clock")
    for index, (result, row) in enumerate(zip(report["attempts"], plan["schedule"])):
        require(all(equal(result.get(key), value) for key, value in row.items()), "Campaign schedule changed")
        if result["status"] in VERIFIED:
            worker_result = read_completion(output, row, plan_sha256=sha256, index=index, returncode=0)
            require(all(equal(result.get(k), v) for k, v in worker_result.items()), "Campaign row differs from completed worker")
        else:
            require(equal(result.get("outcomes"), dict.fromkeys(COLORS))
                    and equal(result.get("rewards"), dict.fromkeys(COLORS)), "Unverified campaign row was scored")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("execute", "worker", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--sha256", required=True)
        if name == "worker":
            sub.add_argument("--index", type=int, required=True)
            for flag in ("deadline", "wall-deadline", "work-deadline", "work-wall-deadline"):
                sub.add_argument("--" + flag, type=float, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
        return 0
    result = execute(args) if args.command == "execute" else read_report(args.plan, args.sha256)
    print(json.dumps({"status": result["status"], "stop_reason": result["stop_reason"],
        "attempts": [{k: row[k] for k in ("episode", "map_kind", "learner_player", "status", "outcomes", "rewards")}
                     for row in result["attempts"]], "training_performed": False}, indent=2))
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
