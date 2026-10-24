#!/usr/bin/env python3
"""Audit and compare the four fixed-checkpoint evaluation pairs, without gameplay.

Only reads existing files and reconstructs inference. It cannot start the engine,
construct an optimizer, change a checkpoint, or make evaluation data trainable.
The single newly created file is the requested compact comparison evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.learner import contextual as contract
from experiments.learner.checkpoints import (FrozenOpponentPool, _model_from_payload,
                                             _read, load_policy_checkpoint)
from experiments.learner.contextual_model import encode_history, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode
from validation.probes.audit_contextual_histories import audit as history_audit

ORIGINAL = ROOT / "runs/first-training-prepared-2026-10-25"
ORIGINAL_EXECUTION = ORIGINAL / "approved-execution"
ORIGINAL_PLAN_SHA = "ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6"
ORIGINAL_REPORT_SHA = "bf6ceac81894310b6a364d6dcc755087b16a21409133398d83db42586f3007b5"
ORIGINAL_EVIDENCE_SHA = "2ebc629c03d6fc864ad878c63747d0235cd73689712bd319e37dca08ed28ab7b"
RESUME_SHA = "df8834bac83c7444ceff5396f06015ab9f14b7f90aea4dd62c1b7a6b3bb2f089"
COLORS = ("red", "blue")
NAVIGATION = {"back", "close", "open_building", "open_hero", "open_panel", "open_town", "select_recruit"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


class Evidence:
    def __init__(self):
        self.artifacts = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        value = identity(path)
        require(expected is None or value["sha256"] == expected, "Artifact hash differs: " + name)
        require(name not in self.artifacts or self.artifacts[name] == value, "Artifact changed while reading: " + name)
        self.artifacts[name] = value
        return name

    def tree(self, directory):
        for path in sorted(Path(directory).rglob("*")):
            # Profile links contain local copyrighted game data, not capture evidence.
            if "profile" not in path.relative_to(directory).parts and path.is_file() and not path.is_symlink():
                self.bind(path)

    def recheck(self):
        for name, value in list(self.artifacts.items()):
            self.bind(ROOT / name, value["sha256"])


def preserved_original(evidence):
    original_path = ROOT / "validation/first-gameplay-2026-10-25.json"
    evidence.bind(original_path, ORIGINAL_EVIDENCE_SHA)
    original = read(original_path)
    changed = {}
    require(len(original["artifacts"]) == 340, "Original evidence artifact inventory differs")
    for name, expected in original["artifacts"].items():
        path = ROOT / name
        if identity(path) == expected:
            evidence.bind(path, expected["sha256"])
            continue
        # These two historical implementation files were deliberately corrected
        # after the stopped run. All captured data must still match on disk.
        require(name in {"experiments/learner/gameplay.py", "tests/test_gameplay_checkpoint.py"},
                "Original retained artifact changed: " + name)
        historical = subprocess.run(["git", "show", original["prepared_implementation_git_commit"] + ":" + name],
                                    cwd=ROOT, check=True, capture_output=True).stdout
        require({"sha256": hashlib.sha256(historical).hexdigest(), "bytes": len(historical)} == expected,
                "Historical implementation bytes no longer match original evidence")
        changed[name] = {"historical": expected, "current": identity(path),
                         "historical_git_commit": original["prepared_implementation_git_commit"]}
        evidence.bind(path)
    return {"all_original_bindings_checked": 340, "unchanged_files_on_disk": 340 - len(changed),
            "historical_source_bindings_verified_in_git": changed,
            "captured_artifacts_unchanged": True,
            "original_report_preserved": evidence.bind(ORIGINAL_EXECUTION / "report.json", ORIGINAL_REPORT_SHA)}


def behavior_audit(model, trajectory, policy_seed):
    decisions = {item.decision_seq: item for item in trajectory.decisions}
    values = {item.decision_seq: item.value for item in trajectory.estimates}
    generator = torch.Generator(device="cpu").manual_seed(policy_seed)
    state = model.initial_state()
    counts = dict(decisions=0, values=0, conditionals=0, seeded_actions_reproduced=0,
                  max_log_prob_difference=0.0, max_value_difference=0.0, max_conditional_difference=0.0)

    def compare(actual, saved, metric):
        difference = abs(actual - saved)
        counts[metric] = max(counts[metric], difference)
        require(math.isclose(actual, saved, rel_tol=2e-5, abs_tol=2e-5), "Inference reconstruction differs: " + metric)

    with torch.no_grad():
        for seq, frame in enumerate(encode_history(trajectory.history)):
            result = model.forward_step(frame, state)
            state = result.state
            if seq in values:
                require(result.decision is not None, "Stored estimate lacks an own decision")
                counts["values"] += 1
                compare(result.decision.value.item(), values[seq], "max_value_difference")
            if seq not in decisions:
                continue
            saved = decisions[seq]
            require(result.decision is not None and result.decision.distribution.support == saved.support,
                    "Saved action support differs from reconstructed own decision")
            distribution = result.decision.distribution
            require(distribution.sample(generator=generator) == saved.action, "Seeded sampled action differs")
            counts["seeded_actions_reproduced"] += 1
            actual = distribution.evaluate(saved.action)
            counts["decisions"] += 1
            compare(actual.log_prob.item(), saved.log_prob, "max_log_prob_difference")
            compare(result.decision.value.item(), saved.value, "max_value_difference")
            require(len(actual.conditionals) == len(saved.conditionals), "Conditional count differs")
            for a, b in zip(actual.conditionals, saved.conditionals):
                require((a.name, a.selected_index, a.support_size) == (b.name, b.selected_index, b.support_size),
                        "Conditional support differs")
                counts["conditionals"] += 1
                compare(a.log_prob.item(), b.log_prob, "max_conditional_difference")
    require(counts["decisions"] == len(decisions) and counts["values"] == len(values), "Some stored inference was not reconstructed")
    return counts


def behavior_metrics(player):
    actions, results, events, movement, economy, captures = (Counter() for _ in range(6))
    latest, pending, previous_owners, previous_towns = None, {}, {}, None
    sampled_days = [player["history"][item["decision_seq"]]["day"] for item in player["decisions"]]
    for frame in player["history"]:
        observation = frame["observation"]
        if observation is not None:
            latest = observation
            for obj in observation["objects"]:
                previous = previous_owners.get(obj["handle"])
                if previous is not None and previous != "self" and obj["owner"] == "self":
                    captures["observed_owner_transitions_to_self"] += 1
                    captures["object_kind_" + str(obj["kind"])] += 1
                previous_owners[obj["handle"]] = obj["owner"]
            towns = {town["handle"] for town in observation["towns"]}
            if previous_towns is not None:
                captures["new_owned_town_observations"] += len(towns - previous_towns)
                captures["lost_owned_town_observations"] += len(previous_towns - towns)
            previous_towns = towns
        if frame["kind"] == "submitted":
            action = frame["action"]
            actions[action["kind"]] += 1
            require(latest is not None, "Action lacks a previous own observation")
            metadata = {"action": action}
            if action["kind"] == "move_to":
                hero = next(hero for hero in latest["heroes"] if hero["handle"] == action["subject"])
                metadata["position"] = hero["position"]
                movement["attempts"] += 1
                movement["attempts_at_zero_movement"] += hero["movement"] == 0
            elif action["kind"] == "recruit":
                metadata["destination_role"] = latest["window"]["town_panel"]["recruitment"]["destination"]["role"]
            elif action["kind"] == "transfer_stack":
                panel = latest["window"]["town_panel"]["armies"]
                metadata["transfer_direction"] = panel["source"]["role"] + "_to_" + panel["destination"]["role"]
            pending[frame["decision_seq"]] = metadata
        elif frame["kind"] == "result":
            result = frame["result"]
            require(frame["decision_seq"] in pending, "Result lacks its own submitted action")
            metadata = pending[frame["decision_seq"]]
            action = metadata["action"]
            results[result["status"]] += 1
            if action["kind"] == "move_to":
                movement["result_" + result["status"]] += 1
                movement["destination_reached"] += result["destination_reached"] is True
                if result["position"] is not None:
                    movement["results_with_position"] += 1
                    movement["endpoint_changed"] += result["position"] != metadata["position"]
            if result["accepted"] and result["status"] == "built":
                economy["successful_constructions"] += 1
            elif result["accepted"] and result["status"] == "recruited":
                economy["successful_recruit_actions"] += 1
                economy["recruited_creatures"] += action["count"]
                economy["recruited_to_" + metadata["destination_role"]] += action["count"]
            elif result["accepted"] and result["status"] == "transferred":
                economy["successful_transfer_actions"] += 1
                economy["transferred_creatures"] += action["count"]
                economy["transferred_" + metadata["transfer_direction"]] += action["count"]
        event = frame["event"]
        if event:
            events[event["kind"]] += 1
            if event["kind"] == "battle_complete":
                events["battle_won" if event["won"] else "battle_lost"] += 1
    rewards = [sample["reward"] for sample in player["samples"]]
    return {"sampled_choices": len(player["decisions"]), "samples": len(player["samples"]),
            "last_sampled_day": max(sampled_days, default=None),
            "last_observed_day": max((frame["day"] for frame in player["history"]), default=None),
            "action_kinds": dict(sorted(actions.items())), "result_statuses": dict(sorted(results.items())),
            "window_navigation_choices": sum(actions[kind] for kind in NAVIGATION),
            "movement": dict(sorted(movement.items())), "economy": dict(sorted(economy.items())),
            "observed_events": dict(sorted(events.items())), "observed_captures": dict(sorted(captures.items())),
            "sample_reward_counts": dict(sorted(Counter(str(value) for value in rewards).items())),
            "nonzero_reward_samples": sum(value in (-1, 1) for value in rewards)}


def replay_summary(evidence, folder, manifest):
    path = folder / "collection-replay/contextual-replay-verification.json"
    if not path.exists():
        return {"verified": False, "present": False}
    value = read(path)
    require(value["inputs"]["manifest"]["sha256"] == identity(folder / "manifest.json")["sha256"], "Replay manifest binding differs")
    native = value.get("native", {})
    summary = native.get("summary", {})
    verified = (value.get("verified") is True and native.get("verified") is True
                and summary.get("server_verified") is True and value.get("histories", {}).get("verified") is True
                and bool(summary.get("unchecked")) and all(count == 0 for count in summary["unchecked"].values()))
    require(not verified or summary.get("captured_prefix_only") == (manifest["status"] != "completed"), "Replay boundary scope differs")
    return {"present": True, "report": evidence.bind(path), "verified": verified,
            "checked": summary.get("checked"), "unchecked": summary.get("unchecked"),
            "boundary": summary.get("boundary"), "captured_prefix_only": summary.get("captured_prefix_only")}


def _checked_attempt(evidence, output, row, expected, model, checkpoint, opponent, original_plan):
    for key in ("episode", "phase", "engine_seed", "learner_player", "policy_seeds"):
        require(row.get(key) == expected[key], "Evaluation schedule differs: " + key)
    result = {key: row.get(key) for key in ("episode", "phase", "engine_seed", "learner_player", "policy_seeds", "status", "termination_reason", "error", "outcomes", "rewards")}
    folder = output / f"episode-{row['episode']:06d}"
    result["players"] = None
    result["collection_replay"] = {"present": False, "verified": False}
    result["checks_passed"] = False
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        require(row.get("replay_verified") is not True, "Passing attempt lacks manifest")
        return result
    manifest = read(manifest_path)
    require(manifest["episode"] == str(row["episode"]) and manifest["seed"] == row["engine_seed"], "Manifest schedule differs")
    require(manifest["environment"] == contract.ARMIES_ENVIRONMENT and manifest["identity"]["map_sha256"] == original_plan["map_identity"]["sha256"], "Environment/map changed")
    require(manifest["identity"]["binary_sha256"] == original_plan["native_identity"]["binary_sha256"], "Native binary changed")
    result.update(manifest=evidence.bind(manifest_path), manifest_status=manifest["status"],
                  manifest_termination_reason=manifest["termination_reason"], native_outcomes=manifest["outcomes_by_player"],
                  native_rewards=manifest["rewards_by_player"], game_wall_seconds=manifest.get("wall_seconds"))
    if manifest["status"] != "completed":
        require(all(v is None for key in ("outcomes_by_player", "rewards_by_player") for v in manifest[key].values()), "Cutoff/failure was scored")
    result["collection_replay"] = replay_summary(evidence, folder, manifest)
    rollout_path = folder / "neural-rollout.json"
    if not rollout_path.exists():
        require(row.get("replay_verified") is not True, "Passing attempt lacks rollout")
        return result
    raw = read(rollout_path)
    require(raw["manifest"] == manifest, "Rollout manifest differs")
    collected = CollectedEpisode.from_dict(raw)
    result["neural_rollout"] = evidence.bind(rollout_path)
    histories = []
    for color in COLORS:
        declaration = manifest["players"][color]["history_v4"]
        path = (folder / declaration["path"]).resolve()
        require(path.is_relative_to(folder.resolve()), "History path escapes episode")
        evidence.bind(path, declaration["sha256"])
        histories.append(path)
    audit = history_audit(*histories, manifest=manifest_path)
    require(audit["record_contract_eligible"] is True, "Independent history audit failed")
    players = []
    for player, (trajectory, saved) in enumerate(zip(collected.rollouts, raw["rollouts"])):
        learner = player == row["learner_player"]
        selected_model = model if learner else opponent.model
        expected_id = checkpoint if learner else opponent.checkpoint
        expected_digest = weights_sha256(selected_model)
        require(trajectory.player == player and trajectory.role == ("learner" if learner else "frozen_opponent"), "Player role differs")
        require(trajectory.behavior_checkpoint == expected_id and trajectory.weights_sha256 == expected_digest, "Frozen behavior identity differs")
        require(trajectory.purpose == "diagnostic" and trajectory.origin == "native" and not trajectory.optimization_eligible, "Evaluation can enter optimization")
        metrics = behavior_metrics(saved)
        require(metrics["last_sampled_day"] is None or metrics["last_sampled_day"] <= 30, "Sampled day cap exceeded")
        require(trajectory.boundary != "cutoff" or all(sample.reward == 0 for sample in trajectory.samples), "Cutoff sample reward is nonzero")
        require([contract.encode_frame(frame) for frame in trajectory.history] ==
                [json.loads(line) for line in histories[player].read_text().splitlines()], "Rollout history differs from bound canonical history")
        players.append({"player": player, "role": trajectory.role, "purpose": trajectory.purpose,
                        "optimizer_eligible": trajectory.optimization_eligible, "weights_sha256": expected_digest,
                        "metrics": metrics, "inference_reconstruction": behavior_audit(selected_model, trajectory, row["policy_seeds"][player])})
    require(len(players) == 2 and sum(p["metrics"]["sampled_choices"] for p in players) <= 256, "Evaluation choice cap exceeded")
    result["players"] = players
    result["checks_passed"] = row.get("replay_verified") is True and result["collection_replay"]["verified"] and row["status"] in ("completed", "truncated")
    if row.get("replay_verified"):
        require(result["checks_passed"] and row["status"] == manifest["status"] and row["outcomes"] == manifest["outcomes_by_player"]
                and row["rewards"] == manifest["rewards_by_player"], "Executor and independent audit disagree")
    require(not (folder / "collection-receipt.json").exists(), "Evaluation has a training receipt")
    return result


def attempt(evidence, output, row, expected, model, checkpoint, opponent, original_plan):
    # A broken recorded attempt remains an attempted slot. A claimed successful
    # attempt still fails this audit if any independent check fails.
    for key in ("episode", "phase", "engine_seed", "learner_player", "policy_seeds"):
        require(row.get(key) == expected[key], "Evaluation schedule differs: " + key)
    try:
        return _checked_attempt(evidence, output, row, expected, model, checkpoint, opponent, original_plan)
    except Exception as error:
        reported_failure = row.get("replay_verified") is not True or row["status"] not in ("completed", "truncated")
        if not reported_failure:
            raise
        result = {key: row.get(key) for key in ("episode", "phase", "engine_seed", "learner_player", "policy_seeds",
                                              "status", "termination_reason", "error", "outcomes", "rewards")}
        result.update(players=None, checks_passed=False, collection_replay={"present": False, "verified": False},
                      independent_validation_error=f"{type(error).__name__}: {error}",
                      scope="Failed attempted slot retained. Incomplete or invalid records are not scored, inferred, or optimizer eligible.")
        folder = output / f"episode-{row['episode']:06d}"
        for name, key in (("manifest.json", "manifest"), ("neural-rollout.json", "neural_rollout"),
                          ("collection-replay/contextual-replay-verification.json", "failed_replay_report")):
            path = folder / name
            if path.exists():
                result[key] = evidence.bind(path)
        result["collection_replay"]["present"] = (folder / "collection-replay/contextual-replay-verification.json").exists()
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = read(manifest_path)
                result.update(manifest_status=manifest.get("status"), manifest_termination_reason=manifest.get("termination_reason"),
                              manifest_failure_detail=manifest.get("failure_detail"),
                              recorded_unverified_outcomes=manifest.get("outcomes_by_player"),
                              recorded_unverified_rewards=manifest.get("rewards_by_player"))
            except (ValueError, TypeError, AttributeError) as decode_error:
                result["manifest_decode_error"] = str(decode_error)
        # Explicitly null the comparison outcome for this failed slot, retaining
        # any raw manifest claims above without treating them as verified facts.
        result["outcomes"] = result["rewards"] = {"red": None, "blue": None}
        return result


def aggregate(rows, learner_only):
    players = [player for row in rows for player in row["players"] or () if not learner_only or player["role"] == "learner"]
    metrics = [player["metrics"] for player in players]
    result = {"attempts": len(rows), "checked_attempts": sum(row["checks_passed"] for row in rows),
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "termination_reasons": dict(Counter(row.get("termination_reason") or row.get("error") or "unrecorded" for row in rows)),
              "verified_natural_outcomes": sum(row["checks_passed"] and row["status"] == "completed" for row in rows),
              "players_summarized": len(players),
              "metrics_unavailable_episodes": [row["episode"] for row in rows if row["players"] is None],
              "scope": "Learner player only" if learner_only else "Both players; event counts may duplicate one battle",
              "last_sampled_days": [value["last_sampled_day"] for value in metrics]}
    for key in ("sampled_choices", "samples", "window_navigation_choices", "nonzero_reward_samples"):
        result[key] = sum(value[key] for value in metrics)
    for key in ("action_kinds", "result_statuses", "movement", "economy", "observed_events", "observed_captures", "sample_reward_counts"):
        counter = Counter()
        for value in metrics:
            counter.update(value[key])
        result[key] = dict(sorted(counter.items()))
    result["natural_learner_outcomes"] = dict(Counter(row["outcomes"][COLORS[row["learner_player"]]] for row in rows
                                                     if row["checks_passed"] and row["status"] == "completed"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--launch-git-commit", required=True)
    parser.add_argument("--supporting-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Finalized evidence must never be overwritten")
    require(not args.output.resolve().is_relative_to(ORIGINAL), "Comparison output cannot enter original run")
    require(len(args.launch_git_commit) == 40 and all(char in "0123456789abcdef" for char in args.launch_git_commit),
            "Actual launch commit must be a full Git object identity")
    torch.set_num_threads(1)
    evidence = Evidence()
    preservation = preserved_original(evidence)
    old_plan, old_report = read(ORIGINAL / "plan.json"), read(ORIGINAL_EXECUTION / "report.json")
    evidence.bind(ORIGINAL / "plan.json", ORIGINAL_PLAN_SHA)
    evidence.bind(ORIGINAL_EXECUTION / "resume.pt", RESUME_SHA)
    plan, report = read(args.plan), read(args.execution / "report.json")
    require(plan["schema"] == "heroes-after-evaluation-plan-v1" and plan["purpose"] == "evaluation_only"
            and report["schema"] == "heroes-after-evaluation-result-v1", "Wrong continuation plan/report schema")
    require(args.execution.resolve() == Path(plan["execution_output"]), "Execution differs from bound output")
    for key, path, expected in (("original_plan", ORIGINAL / "plan.json", ORIGINAL_PLAN_SHA),
                                ("original_report", ORIGINAL_EXECUTION / "report.json", ORIGINAL_REPORT_SHA),
                                ("resume_checkpoint", ORIGINAL_EXECUTION / "resume.pt", RESUME_SHA)):
        require(Path(plan[key]["path"]) == path and plan[key]["sha256"] == expected
                and plan[key]["bytes"] == path.stat().st_size, "Original artifact binding differs: " + key)
    require(report["original_report"] == plan["original_report"] and report["resume_checkpoint"] == plan["resume_checkpoint"],
            "Executor original artifact bindings differ")
    authorization = read(args.authorization)
    require(Path(plan["authorization"]["path"]) == args.authorization.resolve()
            and identity(args.authorization)["sha256"] == plan["authorization"]["sha256"]
            and authorization["approved"] is True and authorization["additional_optimizer_steps"] == 0
            and authorization["additional_ppo_updates"] == 0 and authorization["checkpoint_sha256"] == RESUME_SHA
            and authorization["original_stopped_report_sha256"] == ORIGINAL_REPORT_SHA, "Authorization binding differs")
    for key in ("native_identity", "native_runtime", "map_identity", "binary", "replay_binary", "data", "game_map"):
        require(plan[key] == old_plan[key], "Continuation native environment differs: " + key)
    require(report["status"] in ("completed", "stopped"), "Continuation is not finalized")
    require(args.launch_git_commit == plan["source"]["git_commit"], "Launch commit differs from prepared source commit")
    require(report["plan_sha256"] == identity(args.plan)["sha256"] and report["source"] == plan["source"], "Continuation plan/source mismatch")
    require(report["training_performed"] is False and report["optimizer_steps"] == 0 and report["updates"] == 0 and report["additional_optimizer_steps"] == 0, "Continuation performed optimization")
    for name, digest in plan["implementation_sha256"].items():
        evidence.bind(ROOT / name, digest)
    for name, digest in plan["native_runtime"].items():
        evidence.bind(Path(plan["binary"]).parent / name, digest)
    evidence.bind(plan["binary"], plan["native_identity"]["binary_sha256"])
    evidence.bind(plan["game_map"], plan["map_identity"]["sha256"])
    initial = load_policy_checkpoint(ORIGINAL / "initial-policy.pt", interface=contract.ARMIES_ENVIRONMENT)
    payload = _read(ORIGINAL_EXECUTION / "resume.pt", "optimizer",
                    {"plan_sha256": ORIGINAL_PLAN_SHA, "source": old_plan["source"], "configuration": old_plan["configuration"]},
                    ("approved_gameplay",), contract.ARMIES_ENVIRONMENT)
    after_model = _model_from_payload(payload).eval().requires_grad_(False)
    pool = FrozenOpponentPool.from_state_dict(payload["opponent_pool"])
    require(pool.identities == initial.opponent_pool.identities and len(pool.identities) == 1, "Saved frozen opponent differs")
    require(payload["resume"]["progress"] == {"updates": 1, "rollouts": 2, "decisions": 294}
            and payload["resume"]["sampler_state"] == {"next_episode": 7, "next_phase": "after"}, "Saved training progress differs")
    require(report["saved_progress"] == payload["resume"]["progress"] and report["original_optimizer_steps"] == old_report["optimizer_steps"]
            and payload["weights_sha256"] == plan["learner_weights_sha256"]
            and list(pool.identities[0]) == plan["frozen_opponent"], "Continuation saved policy identity differs")
    opponent = pool.get(pool.identities[0][0])
    schedule = [{"episode": index, **row} for index, row in enumerate(old_plan["attempt_schedule"], 1)]
    planned_after = [row for row in schedule if row["phase"] == "after"]
    require([row["episode"] for row in planned_after] == [7, 8, 9, 10], "Original after schedule differs")
    require(plan["attempt_schedule"] == authorization["attempt_schedule"] == planned_after,
            "New plan or authorization changed the original four after slots")
    require(plan["limits"] == {"max_requests": 256, "max_days": 30, "torch_threads": 1}, "Continuation limits differ")
    require(len(report["attempts"]) <= 4, "Extra after-evaluation attempts")
    actual_dirs = {int(path.name.split("-")[1]) for path in args.execution.glob("episode-*") if path.is_dir()}
    require(actual_dirs <= {row["episode"] for row in report["attempts"]}, "Unreported episode directory")
    before = [attempt(evidence, ORIGINAL_EXECUTION, row, expected, initial.model, initial.checkpoint, opponent, old_plan)
              for row, expected in zip(old_report["attempts"][:4], schedule[:4])]
    after = [attempt(evidence, args.execution, row, expected, after_model, payload["checkpoint"], opponent, old_plan)
             for row, expected in zip(report["attempts"], planned_after)]
    require(report["status"] != "completed" or len(after) == 4 and all(row["checks_passed"] for row in after), "Incomplete after phase marked completed")
    pairs = []
    for index, before_row in enumerate(before):
        after_row = after[index] if index < len(after) else None
        require(after_row is None or all(before_row[key] == after_row[key] for key in ("engine_seed", "learner_player", "policy_seeds")), "Before/after pairing differs")
        pairs.append({"engine_seed": before_row["engine_seed"], "learner_player": before_row["learner_player"],
                      "policy_seeds": before_row["policy_seeds"], "before_episode": before_row["episode"],
                      "after_episode": after_row["episode"] if after_row else None,
                      "both_checked": after_row is not None and before_row["checks_passed"] and after_row["checks_passed"],
                      "before": aggregate([before_row], True), "after": aggregate([after_row], True) if after_row else None})
    total_wall = old_report["wall_seconds"] + report["wall_seconds"]
    require(report["combined_wall_seconds"] == total_wall
            and plan["remaining_wall_seconds"] == old_plan["configuration"]["total_wall_seconds"] - old_report["wall_seconds"],
            "Shared elapsed/remaining wall budget differs")
    within_budget = total_wall <= old_plan["configuration"]["total_wall_seconds"]
    require(report["status"] != "completed" or within_budget and report["fixed_weights_unchanged"] is True
            and report["original_artifacts_unchanged"] is True and report["resume_verified"] is True,
            "Completed continuation lacks budget or final-integrity checks")
    pool.assert_unchanged()
    require(weights_sha256(after_model) == payload["weights_sha256"], "Inference changed saved learner weights")
    evidence.tree(args.execution)
    result = {"schema": "heroes-after-evaluation-comparison-v1", "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": report["status"], "error": report.get("error"), "launch_git_commit": args.launch_git_commit,
              "scope": "Read-only evidence and seeded inference reconstruction; no native process, optimizer construction/step, or checkpoint write.",
              "plan": evidence.bind(args.plan), "executor_report": evidence.bind(args.execution / "report.json"),
              "authorization": evidence.bind(args.authorization), "comparison_source": evidence.bind(__file__),
              "supporting_artifacts": [evidence.bind(path) for path in args.supporting_artifact],
              "original_preservation": preservation, "source": plan["source"], "native_identity": plan["native_identity"],
              "map_identity": plan["map_identity"], "training_performed": False, "additional_optimizer_steps": 0,
              "saved_progress_unchanged": payload["resume"]["progress"], "original_optimizer_steps": old_report["optimizer_steps"],
              "checkpoint": evidence.bind(ORIGINAL_EXECUTION / "resume.pt", RESUME_SHA),
              "before_weights_sha256": initial.weights_sha256, "after_weights_sha256": payload["weights_sha256"],
              "frozen_opponent": {"checkpoint": opponent.checkpoint, "weights_sha256": opponent.weights_sha256},
              "wall_seconds": {"original": old_report["wall_seconds"], "continuation": report["wall_seconds"],
                               "combined": total_wall, "original_total_budget": old_plan["configuration"]["total_wall_seconds"], "within_budget": within_budget},
              "before": before, "after": after, "unattempted_after_schedule": planned_after[len(after):], "paired_evaluation": pairs,
              "aggregate": {phase: {"learner_only": aggregate(rows, True), "all_players": aggregate(rows, False)}
                            for phase, rows in (("before", before), ("after", after))},
              "metric_semantics": {"movement": "An attempt is not success. Endpoint changes compare the returned own position to that action's pre-action own position; route length is not inferred.",
                                   "economy": "Counts accepted typed built/recruited/transferred results; creature counts are accepted requested quantities, not current army estimates.",
                                   "battles": "Player-visible battle events, not deduplicated engine battles. The all-player view can count one PvP battle twice.",
                                   "captures": "Observed object owner transitions to self and new/lost entries in the player's owned-town list. First sightings and hidden ownership changes are not inferred.",
                                   "sample_rewards": "Ordinary transitions are zero; cutoffs/failures keep null episode outcomes and rewards. None is not a draw."},
              "limitations": ["Four paired seeds and seats cannot establish strategic improvement; changed sampled actions or weights are descriptive only.",
                              "Shared seeds do not hold later environment RNG draws or observations fixed once policies choose different actions.",
                              "Before and after are held-out diagnostic trajectories and remain excluded from optimization.",
                              "Native replay validates recorded prefixes; it cannot turn a cutoff into a natural result.",
                              "Inference reconstruction uses the saved implementation in a new process, not a separately implemented policy.",
                              "Original before/training evidence stays unchanged; this separate continuation does not rewrite its stopped status."],
              "artifacts": evidence.artifacts}
    evidence.recheck()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
