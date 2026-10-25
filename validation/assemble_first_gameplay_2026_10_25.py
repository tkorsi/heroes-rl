#!/usr/bin/env python3
"""Aggregate retained first-run artifacts; never launch games or optimization.

The executor is the owner of admission, replay and probability checks. This
stdlib-only reader binds its records, checks their consistency and reports their
scope. It does not turn a partial run into ten completed attempts or rerun it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6"
COLORS = ("red", "blue")
PHASES = ("before", "training", "after")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}


class Evidence:
    def __init__(self):
        self.artifacts = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        value = identity(path)
        require(expected is None or value["sha256"] == expected, "Recorded bytes changed: " + name)
        require(name not in self.artifacts or value == self.artifacts[name], "Artifact changed during assembly: " + name)
        self.artifacts[name] = value
        return name

    def tree(self, directory):
        # Native profiles contain linked copyrighted data; retain only authored
        # capture/replay outputs outside them, and do not follow directory links.
        for path in sorted(Path(directory).rglob("*")):
            if "profile" not in path.relative_to(directory).parts and path.is_file() and not path.is_symlink():
                self.bind(path)


def replay_summary(evidence, folder, name, manifest_sha256):
    path = folder / name / "contextual-replay-verification.json"
    if not path.exists():
        return {"present": False, "verified": False}
    replay = read(path)
    evidence.bind(folder / "manifest.json", replay["inputs"]["manifest"]["sha256"])
    require(replay["inputs"]["manifest"]["sha256"] == manifest_sha256, "Replay binds a different manifest")
    native = replay.get("native", {})
    summary = native.get("summary", {})
    verified = (replay.get("verified") is True and native.get("verified") is True
                and summary.get("server_verified") is True and replay.get("histories", {}).get("verified") is True
                and bool(summary.get("unchecked")) and all(v == 0 for v in summary["unchecked"].values()))
    return {"present": True, "report": evidence.bind(path), "verified": verified,
            "checked": summary.get("checked"), "unchecked": summary.get("unchecked"),
            "boundary": summary.get("boundary"), "captured_prefix_only": summary.get("captured_prefix_only")}


def player_summary(player):
    frames = {frame["seq"]: frame for frame in player["history"]}
    days = [frames[item["decision_seq"]]["day"] for item in player["decisions"]]
    rewards = [sample["reward"] for sample in player["samples"]]
    return {"player": player["player"], "role": player["role"], "purpose": player["purpose"],
            "origin": player["origin"], "boundary": player["boundary"],
            "behavior_checkpoint": player["behavior_checkpoint"], "weights_sha256": player["weights_sha256"],
            "opponent_checkpoint": player["opponent_checkpoint"], "opponent_weights_sha256": player["opponent_weights_sha256"],
            "sampled_choices": len(player["decisions"]), "samples": len(player["samples"]),
            "eligible_samples": sum(sample["eligible"] for sample in player["samples"]),
            "value_estimates": len(player["estimates"]),
            "conditional_probabilities": sum(len(item["conditionals"]) for item in player["decisions"]),
            "last_sampled_day": max(days, default=None),
            "last_observed_day": max((frame["day"] for frame in player["history"]), default=None),
            "reward_counts": dict(sorted(Counter(str(value) for value in rewards).items())),
            "nonzero_reward_samples": sum(value in (-1, 1) for value in rewards),
            "bootstrap_samples": sum(sample["bootstrap_value"] is not None for sample in player["samples"]),
            "action_kinds": dict(sorted(Counter(item["action"]["kind"] for item in player["decisions"]).items()))}


def attempt_summary(evidence, output, report_row, expected, plan):
    row = {key: report_row.get(key) for key in ("phase", "episode", "engine_seed", "learner_player", "policy_seeds", "status",
                                             "termination_reason", "error", "outcomes", "rewards", "weights_sha256")}
    for key in ("phase", "engine_seed", "learner_player", "policy_seeds"):
        require(row[key] == expected[key], "Attempt differs from prepared schedule: " + key)
    folder = output / f"episode-{row['episode']:06d}"
    row["directory"] = str(folder.relative_to(ROOT))
    row["declared_optimizer_eligible"] = expected["optimizer_eligible"]
    row["executor_all_collection_checks_passed"] = report_row.get("replay_verified") is True and row["status"] in ("completed", "truncated")
    row["check_scope"] = "Executor marks a completed collection only after history, native replay, fixed-weight and both-player behavior-probability checks. This assembler does not reexecute these checks."
    manifest_path, rollout_path = folder / "manifest.json", folder / "neural-rollout.json"
    if not manifest_path.exists():
        require(not row["executor_all_collection_checks_passed"], "Passing attempt has no manifest")
        row.update(manifest=None, players=None, collection_replay={"present": False, "verified": False})
        return row
    manifest = read(manifest_path)
    manifest_sha256 = identity(manifest_path)["sha256"]
    require(manifest["episode"] == str(row["episode"]) and manifest["seed"] == row["engine_seed"], "Manifest episode/seed differs")
    require(manifest["identity"]["map_sha256"] == plan["map_identity"]["sha256"]
            and manifest["environment"] == plan["configuration"]["interface"], "Manifest map/profile differs")
    row.update(manifest=evidence.bind(manifest_path), manifest_status=manifest["status"],
               manifest_termination_reason=manifest["termination_reason"], manifest_failure_detail=manifest.get("failure_detail"),
               native_returncode=manifest.get("returncode"), game_wall_seconds=manifest.get("wall_seconds"),
               env_steps=manifest.get("env_steps"), native_outcomes=manifest["outcomes_by_player"],
               native_rewards=manifest["rewards_by_player"], collection_replay=replay_summary(evidence, folder, "collection-replay", manifest_sha256),
               admission_replay=replay_summary(evidence, folder, "admission-replay", manifest_sha256))
    if row["executor_all_collection_checks_passed"]:
        require(row["status"] == manifest["status"] and row["outcomes"] == manifest["outcomes_by_player"]
                and row["rewards"] == manifest["rewards_by_player"], "Collection report disagrees with manifest")
        require(row["collection_replay"]["verified"], "Successful collection lacks a bound verified replay")
    if manifest["status"] != "completed":
        require(all(value is None for key in ("outcomes_by_player", "rewards_by_player") for value in manifest[key].values()),
                "Unscored attempt has an outcome or reward")
    if rollout_path.exists():
        rollout = read(rollout_path)
        require(rollout["manifest"] == manifest, "Neural rollout embeds a different manifest")
        players = [player_summary(player) for player in rollout["rollouts"]]
        require([player["player"] for player in players] == [0, 1], "Unexpected player order")
        seat = row["learner_player"]
        require(players[seat]["role"] == "learner" and players[1 - seat]["role"] == "frozen_opponent", "Roles differ from the schedule")
        require(all(p["purpose"] == ("training" if row["phase"] == "training" else "diagnostic") for p in players),
                "Training/evaluation purpose differs")
        require(players[1 - seat]["weights_sha256"] == plan["initial_checkpoint"]["opponents"][0][1], "Frozen opponent changed")
        if row["phase"] != "after":
            require(players[seat]["weights_sha256"] == plan["initial_checkpoint"]["weights_sha256"], "Pre-update learner changed")
        row.update(neural_rollout=evidence.bind(rollout_path), players=players,
                   sampled_choices=sum(p["sampled_choices"] for p in players),
                   sampled_day_budget_passed=all(p["last_sampled_day"] is None or p["last_sampled_day"] <= plan["configuration"]["max_days"] for p in players))
        row["sampled_choice_budget_passed"] = row["sampled_choices"] <= plan["configuration"]["max_requests"]
    else:
        require(not row["executor_all_collection_checks_passed"], "Passing attempt has no neural rollout")
        row["players"] = None
    receipt_path = folder / "collection-receipt.json"
    row["collection_receipt"] = None
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(row["phase"] == "training", "Evaluation has a training receipt")
        require(receipt["manifest_sha256"] == manifest_sha256 and receipt["artifact_sha256"] == identity(rollout_path)["sha256"],
                "Receipt artifact hashes differ")
        require(receipt["episode"] == str(row["episode"]) and receipt["learner_player"] == row["learner_player"]
                and receipt["engine_seed"] == row["engine_seed"] and receipt["policy_seeds"] == row["policy_seeds"], "Receipt schedule differs")
        row["collection_receipt"] = evidence.bind(receipt_path)
    return row


def phase_summary(rows):
    natural = [row for row in rows if row["status"] == "completed" and row["executor_all_collection_checks_passed"]]
    return {"attempts": len(rows), "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "termination_reasons": dict(sorted(Counter(row.get("termination_reason") or row.get("error") or "unrecorded" for row in rows).items())),
            "verified_natural_outcomes": len(natural),
            "learner_natural_outcomes": dict(sorted(Counter(row["outcomes"][COLORS[row["learner_player"]]] for row in natural).items())),
            "sampled_choices": sum(row.get("sampled_choices", 0) for row in rows),
            "learner_samples": sum(row["players"][row["learner_player"]]["samples"] for row in rows if row.get("players")),
            "learner_nonzero_reward_samples": sum(row["players"][row["learner_player"]]["nonzero_reward_samples"] for row in rows if row.get("players")),
            "passed_collection_replays": sum(row["collection_replay"]["verified"] for row in rows)}


def paired_comparison(rows, schedule):
    by_key = {(row["phase"], row["engine_seed"], row["learner_player"]): row for row in rows}
    result = []
    for planned in schedule:
        if planned["phase"] != "before":
            continue
        pair = {"engine_seed": planned["engine_seed"], "learner_player": planned["learner_player"],
                "paired_policy_seeds": planned["policy_seeds"]}
        for phase in ("before", "after"):
            row = by_key.get((phase, planned["engine_seed"], planned["learner_player"]))
            if row is None:
                pair[phase] = {"attempted": False}
                continue
            p = row["players"][row["learner_player"]] if row.get("players") else None
            pair[phase] = {"attempted": True, "episode": row["episode"], "status": row["status"],
                           "termination_reason": row.get("termination_reason"), "error": row.get("error"),
                           "sampled_choices_total": row.get("sampled_choices"),
                           "learner_outcome": (row.get("outcomes") or {}).get(COLORS[row["learner_player"]]),
                           "learner_reward": (row.get("rewards") or {}).get(COLORS[row["learner_player"]]),
                           "learner_last_sampled_day": p["last_sampled_day"] if p else None,
                           "learner_action_kinds": p["action_kinds"] if p else None,
                           "learner_weights_sha256": p["weights_sha256"] if p else None}
        pair["both_attempted"] = all(pair[phase]["attempted"] for phase in ("before", "after"))
        result.append(pair)
    return result


def complete_budget_check(rows, key):
    values = [row.get(key) for row in rows]
    if False in values:
        return False
    return True if values and all(value is True for value in values) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--launch-git-commit", required=True)
    parser.add_argument("--supporting-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Do not overwrite finalized evidence")
    require(len(args.launch_git_commit) == 40 and all(c in "0123456789abcdef" for c in args.launch_git_commit),
            "Record the full actual launch commit, separate from the prepared implementation commit")
    evidence = Evidence()
    plan_path, output = args.plan.resolve(), args.execution.resolve()
    require(identity(plan_path)["sha256"] == PLAN_SHA256, "This assembler covers only the explicitly approved first plan")
    plan, report = read(plan_path), read(output / "report.json")
    authorization = read(args.authorization)
    require(authorization.get("authorization") == "granted_once"
            and authorization.get("approved_plan_sha256") == PLAN_SHA256
            and authorization.get("configuration") == plan["configuration"]
            and authorization.get("attempt_schedule") == plan["attempt_schedule"], "Authorization does not bind this exact one-run plan")
    require(report["schema"] == "heroes-first-training-result-v1" and report["status"] in ("completed", "stopped"), "Execution is not finalized")
    require(report["plan_sha256"] == PLAN_SHA256 and report["source"] == plan["source"], "Executor plan/source differs")
    require(report["updates"] <= 1 and report["optimizer_steps"] <= 16, "Authorized optimizer budget exceeded")
    schedule = plan["attempt_schedule"]
    require(len(report["attempts"]) <= len(schedule) == 10, "Authorized attempt budget exceeded")
    for index, row in enumerate(report["attempts"], 1):
        require(row["episode"] == index, "Attempts are missing, duplicated or reordered")
    actual_dirs = {int(path.name.split("-")[1]) for path in output.glob("episode-*") if path.is_dir()}
    require(actual_dirs <= {row["episode"] for row in report["attempts"]}, "An on-disk attempt is absent from the report")
    rows = [attempt_summary(evidence, output, row, expected, plan) for row, expected in zip(report["attempts"], schedule)]
    require(report["status"] != "completed" or len(rows) == 10, "Completed run is missing declared attempts")
    for path, expected in plan["implementation_sha256"].items():
        evidence.bind(ROOT / path, expected)
    evidence.bind(plan["initial_checkpoint"]["path"], plan["initial_checkpoint"]["sha256"])
    evidence.bind(plan["binary"], plan["native_identity"]["binary_sha256"])
    evidence.bind(plan["game_map"], plan["map_identity"]["sha256"])
    for path, expected in plan["native_runtime"].items():
        evidence.bind(Path(plan["binary"]).parent / path, expected)
    evidence.bind(plan["readiness"]["evidence"]["path"], plan["readiness"]["evidence"]["sha256"])
    checkpoints = {name: evidence.bind(output / name) if (output / name).exists() else None for name in ("resume.pt", "trained-policy.pt")}
    if report.get("resume_verified"):
        require(all(checkpoints.values()), "Reported checkpoint reload lacks saved checkpoint files")
    phases = {phase: phase_summary([row for row in rows if row["phase"] == phase]) for phase in PHASES}
    training_rows = [row for row in rows if row["phase"] == "training"]
    if report["updates"]:
        require(len(training_rows) == 2 and all(row["collection_receipt"] and row["admission_replay"]["verified"] for row in training_rows),
                "Update lacks both fresh training receipts/admission replays")
        require(report["ppo"]["samples"] == phases["training"]["learner_samples"], "PPO sample count differs from training learners")
    evidence.tree(output)
    supporting = [evidence.bind(path) for path in args.supporting_artifact]
    natural_count = sum(phase["verified_natural_outcomes"] for phase in phases.values())
    result = {
        "schema": "heroes-first-gameplay-evidence-v1", "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_date": "2026-10-25", "experiment_timezone": "Europe/Kiev",
        "scope": "Read-only aggregation of the single authorized prepared local first gameplay experiment. No additional game, replay, optimizer or checkpoint load is run by this assembler.",
        "plan": evidence.bind(plan_path, PLAN_SHA256), "plan_sha256": PLAN_SHA256,
        "authorization_record": evidence.bind(args.authorization), "executor_report": evidence.bind(output / "report.json"),
        "assembler": evidence.bind(__file__), "supporting_artifacts": supporting,
        "launch_git_commit": args.launch_git_commit,
        "prepared_implementation_git_commit": plan["source"]["git_commit"],
        "source_commit_scope": "The plan retains the frozen implementation commit. The actual launch HEAD additionally contains documentation, compact evidence and the one-run authorization; the 173 bound implementation file hashes are checked separately.",
        "source": plan["source"], "implementation_sha256": plan["implementation_sha256"],
        "native_identity": plan["native_identity"], "native_runtime": plan["native_runtime"], "map_identity": plan["map_identity"],
        "initial_checkpoint": plan["initial_checkpoint"], "configuration": plan["configuration"],
        "status": report["status"], "error": report.get("error"), "cleanup_error": report.get("cleanup_error"),
        "wall_seconds": report["wall_seconds"], "wall_time_scope": "Executor monotonic timer includes collection, native replay, PPO, checkpoint/reload and cleanup; prelaunch identity checking and this aggregation are outside it.",
        "attempts_started": len(rows), "attempts_planned": len(schedule), "unattempted_schedule": schedule[len(rows):],
        "training_performed": report["training_performed"], "optimizer_update_attempted": report.get("optimizer_update_attempted", False),
        "completed_updates": report["updates"], "retained_optimizer_steps": report["optimizer_steps"],
        "update_rolled_back": report.get("update_rolled_back", False), "rolled_back_optimizer_steps": report.get("rolled_back_optimizer_steps", 0),
        "ppo": report.get("ppo"), "checkpoint_artifacts": checkpoints,
        "checkpoint_reload_verified_by_executor": report.get("resume_verified", False),
        "checkpoint_scope": "Executor compares reloaded model, Adam, progress, frozen pool and global/dedicated RNG states, then uses restored objects for after-evaluation. No process-restart/resume CLI is implemented.",
        "phases": phases, "attempts": rows, "paired_evaluation": paired_comparison(rows, schedule),
        "budget_checks": {"attempts": len(rows) <= 10, "updates": report["updates"] <= 1,
                          "optimizer_steps": report["optimizer_steps"] <= 16,
                          "wall_seconds": report["wall_seconds"] <= plan["configuration"]["total_wall_seconds"],
                          "sampled_choices": complete_budget_check(rows, "sampled_choice_budget_passed"),
                          "sampled_days": complete_budget_check(rows, "sampled_day_budget_passed"),
                          "sampled_budget_attempts_checked": sum("sampled_choice_budget_passed" in row for row in rows),
                          "sampled_budget_attempts_unavailable": [row["episode"] for row in rows if "sampled_choice_budget_passed" not in row],
                          "null_means": "Bounds are not proven for every attempted episode because its persisted rollout is unavailable; this is not a passing budget check."},
        "verified_natural_outcomes": natural_count,
        "observed_training_terminal_feedback": phases["training"]["learner_nonzero_reward_samples"] > 0,
        "interpretation": ("No verified natural outcome or nonzero terminal feedback was observed. Zero-reward cutoff updates use bootstrap targets; changed weights or behavior do not demonstrate improved strategy."
                           if natural_count == 0 else "This small first pipeline experiment retains observed natural outcomes separately by phase and seat. It is not by itself evidence of strategic improvement."),
        "limitations": ["Four held-out paired seed/seat combinations are a pipeline comparison, not a statistical strength evaluation.",
                        "Technical stops and ordinary cutoffs are unscored, never losses or draws; absent phases remain absent.",
                        "Only fresh designated learner training trajectories may enter PPO; evaluation, frozen opponents and archived diagnostics are excluded.",
                        "Native replay verifies recorded prefixes; cutoff replay does not turn a cutoff into a natural ending.",
                        "Prior live-validation scope exclusions still apply: garrisoned-hero edge cases, initial/timer-origin endings, broader dialogs, other maps and platforms."],
        "artifacts": evidence.artifacts,
    }
    # Rebind after all reads so finalized evidence cannot silently mix a moving
    # report with earlier episode bytes.
    for path in list(evidence.artifacts):
        evidence.bind(ROOT / path if not Path(path).is_absolute() else Path(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "wall_seconds", "attempts_started", "completed_updates", "retained_optimizer_steps", "verified_natural_outcomes")}, indent=2))


if __name__ == "__main__":
    main()
