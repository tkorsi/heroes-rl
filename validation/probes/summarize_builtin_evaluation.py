"""Offline comparison of matched learner perspectives, without new execution."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.headless.contextual_evidence import strict_json
from experiments.learner import evaluate_builtin_opponent as evaluation
from experiments.learner import evaluate_long_dialogs as neural_evaluation
from experiments.learner.builtin_behavior import analyze_episode as builtin_behavior
from experiments.learner.evaluation_behavior import analyze_episode as neural_behavior, _aggregate
from validation.probes.summarize_long_dialog_evaluation import (
    bind_revalidation, compact_behavior, equal, evidence_schema, STATUSES,
)

SCHEMA = "heroes-builtin-opponent-comparison-evidence-v1"
COLORS = ("red", "blue")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def category(row):
    """Only independently passed native outcomes can become wins or losses."""
    require(row.get("status") in STATUSES, "Unknown evaluation status")
    if row["status"] != "passed":
        require(row.get("outcomes") == dict.fromkeys(COLORS)
                and row.get("rewards") == dict.fromkeys(COLORS), "Unverified attempt received a score")
        return row["status"]
    require(row.get("replay_verified") is True and row.get("weights_unchanged") is True,
            "Passed attempt lacks replay or frozen-weight evidence")
    outcomes, rewards = row["outcomes"], row["rewards"]
    if row.get("termination_reason") == "natural_terminal":
        require(list(outcomes[c] for c in COLORS) in (["win", "loss"], ["loss", "win"])
                and rewards == {c: 1 if outcomes[c] == "win" else -1 for c in COLORS}
                and all(type(v) is int for v in rewards.values()), "Invalid natural result")
        return "natural_" + outcomes[COLORS[row["learner_player"]]]
    require(outcomes == dict.fromkeys(COLORS) and rewards == dict.fromkeys(COLORS), "Cutoff received a score")
    require(row.get("termination_reason") in ("collection_window", "game_day_limit", "agent_action_limit"),
            "Unknown verified cutoff")
    return "verified_cutoff"


def summarize(plan_path, report_path, *, output_path=None):
    plan_path, report_path = Path(plan_path).resolve(), Path(report_path).resolve()
    report = strict_json(report_path.read_bytes())
    plan = evaluation.read_plan(plan_path, report["plan_sha256"])
    version = evaluation.plan_version(plan)
    require(report.get("schema") == evaluation.result_schema(plan), "Unexpected comparison report")
    require(report.get("status") in ("finished", "stopped"), "Comparison is still executing")
    reference = plan["reference"]
    require(evaluation.reference(reference["plan"]["path"], reference["report"]["path"]) == reference,
            "Neural reference identity changed")
    neural_report = strict_json(Path(reference["report"]["path"]).read_bytes())
    neural_plan = neural_evaluation.read_plan(reference["plan"]["path"], neural_report["plan_sha256"])
    require(neural_report.get("schema") == neural_evaluation.result_schema(neural_plan),
            "Unexpected neural reference report")
    require(equal(report.get("frozen_before"), plan["frozen"])
            and report.get("frozen_after_file_sha256") == plan["checkpoint"]["sha256"], "Frozen checkpoint changed")
    bindings, issues = {}, []

    def bind(path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            issues.append({"path": str(path), "issue": "bound_file_missing"})
            return None
        actual = evaluation.binding(path)
        wanted = {"sha256": expected} if isinstance(expected, str) else expected or {}
        require(all(actual[k] == wanted[k] for k in ("sha256", "bytes") if k in wanted),
                "Bound artifact changed: " + str(path))
        bindings[str(path)] = {k: actual[k] for k in ("sha256", "bytes")}
        return actual

    bind(plan_path, report["plan_sha256"])
    bind(report_path)
    for item in (plan["checkpoint"], plan["replay_binary"], reference["plan"], reference["report"]):
        bind(item["path"], item)
    if version == 2:
        bind_revalidation(plan, bind)
        bind_revalidation(neural_plan, bind)
    for name, sha in plan["implementation"].items():
        bind(evaluation.ROOT / name, sha)
    for name, sha in plan["source"]["source_sha256"].items():
        bind(evaluation.ROOT / name, sha)
    binary = Path(plan["binary"])
    bind(binary, plan["native"]["binary_sha256"])
    for name, sha in plan["runtime"].items():
        bind(binary.parent / name, sha)
    bind(plan["game_map"], plan["map"]["sha256"])
    cohorts = {}
    selected = {}
    excluded = Path(output_path).resolve() if output_path else None
    for label, cohort_plan, cohort_report, analyzer in (
        ("frozen_neural_opponent", neural_plan, neural_report, neural_behavior),
        ("builtin_nullkiller_easy", plan, report, builtin_behavior),
    ):
        execution = Path(cohort_plan["execution_output"]).resolve()
        expected_report = execution / "report.json"
        require(expected_report == (report_path if label == "builtin_nullkiller_easy" else Path(reference["report"]["path"])),
                "Report is outside its fixed execution directory")
        require(cohort_report.get("training_performed") is False and all(
            type(cohort_report.get(k)) is int and cohort_report[k] == 0
            for k in ("optimizer_updates", "optimizer_steps", "training_samples")), "Evaluation reports training")
        rows = cohort_report["attempts"]
        require(len(rows) == len(cohort_plan["schedule"]) == 8, "All eight planned rows must survive")
        # Hash every retained artifact; never start gameplay, a model or replay.
        for path in sorted(execution.rglob("*")):
            if not path.is_file() or path.resolve() == excluded or any(".tmp-" in p or p.endswith(".tmp") for p in path.relative_to(execution).parts):
                continue
            require(not path.is_symlink() and path.resolve().is_relative_to(execution), "Artifact escapes execution directory")
            bind(path)
        compact_rows, own_players, categories = [], [], Counter()
        for row, expected in zip(rows, cohort_plan["schedule"]):
            require(all(k in row and equal(row[k], v) for k, v in expected.items()), "Fixed schedule changed")
            require(row.get("training_samples") == 0 and row.get("optimizer_steps") == 0, "Attempt reports training")
            result_category = category(row)
            categories[result_category] += 1
            folder = execution / f"episode-{row['episode']:06d}"
            if row.get("episode_directory"):
                require(Path(row["episode_directory"]).resolve() == folder, "Attempt path changed")
            compact = {k: v for k, v in row.items() if k not in ("last_worker_result", "behavior")}
            compact.update(category=result_category, learner_behavior=None)
            if folder.exists():
                require(row["status"] != "not_started", "Unstarted game has episode artifacts")
                behavior = analyzer(folder)
                for name, identity in behavior["bindings"].items():
                    bind(folder / name, identity)
                if row.get("behavior"):
                    item = row["behavior"]
                    if bind(item["path"], item):
                        require(equal(strict_json(Path(item["path"]).read_bytes()), behavior), "Persisted behavior changed")
                if behavior.get("episode"):
                    episode = behavior["episode"]
                    require(str(episode["episode"]) == str(row["episode"]) and episode["seed"] == row["engine_seed"],
                            "Manifest game or seed differs")
                if row["status"] == "passed":
                    require(behavior["status"] == "complete"
                            and equal(row["outcomes"], behavior["episode"]["outcomes_by_player"])
                            and equal(row["rewards"], behavior["episode"]["rewards_by_player"]), "Passed evidence is incomplete")
                color = COLORS[row["learner_player"]]
                own = behavior["players"].get(color)
                if own is not None:
                    own_players.append(own)
                    compact["learner_behavior"] = compact_behavior(behavior)["players"][color]
                compact["episode_throughput"] = behavior.get("throughput")
            elif row["status"] != "not_started":
                issues.append({"cohort": label, "episode": row["episode"], "issue": "episode_directory_missing"})
            compact_rows.append(compact)
        selected[label] = compact_rows
        cohorts[label] = {"attempts": compact_rows, "categories": dict(categories),
            "planned": 8, "started": 8 - categories["not_started"],
            "available_learner_histories": len(own_players), "learner_behavior": _aggregate(own_players),
            "execution_wall_seconds": cohort_report.get("execution_wall_seconds"),
            "execution_status": cohort_report["status"], "stop_reason": cohort_report.get("stop_reason"),
            "native_opponent_requests": sum(r.get("native_opponent_requests", 0) for r in rows)
                if label == "builtin_nullkiller_easy" else None,
            "native_request_count_scope": "Audited attempts with replay counts only; missing counts are unavailable.",
            "attempts_with_native_request_counts": sum("native_opponent_requests" in r for r in rows)
                if label == "builtin_nullkiller_easy" else 0}
        if version == 2:
            cohorts[label].update(configuration=cohort_plan["configuration"],
                revalidation=cohort_plan["revalidation"],
                offline_report_seconds=cohort_report.get("offline_report_seconds"))
    pairs = []
    for neural, builtin in zip(selected["frozen_neural_opponent"], selected["builtin_nullkiller_easy"]):
        require(all(neural[k] == builtin[k] for k in ("episode", "engine_seed", "learner_player"))
                and neural["policy_seeds"][neural["learner_player"]] == builtin["policy_seed"],
                "Matched learner seed or seat changed")
        pairs.append({"episode": builtin["episode"], "engine_seed": builtin["engine_seed"],
            "learner_player": builtin["learner_player"], "policy_seed": builtin["policy_seed"],
            "neural_category": neural["category"], "builtin_category": builtin["category"]})
    return {"schema": evidence_schema(SCHEMA, version), "plan": plan, "bindings": bindings, "coverage": {"issues": issues,
        "all_planned_games_verified": not issues and all(r["status"] == "passed" for rows in selected.values() for r in rows)},
        "cohorts": cohorts, "matched_pairs": pairs, "training_performed": False, "optimizer_steps": 0,
        "training_samples": 0, "policy_input": False, "reward_input": False,
        "limitations": [plan["choice_scope"], plan["replay_scope"],
            "Only the designated learner perspective contributes to behavior comparisons; no native-opponent trajectory is invented.",
            "Available failed prefixes are descriptive and remain unscored. Unstarted games are retained.",
            "Same engine and learner sampling seeds do not imply identical game trajectories or reproduce native AI randomness.",
            "This small diagnostic sample cannot establish learning, superiority or a causal opponent effect.",
            "This offline assembly checks bindings and structure; it does not rerun native replay or policy reconstruction."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "report", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve the earlier comparison evidence")
    result = summarize(args.plan, args.report, output_path=args.output)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": evaluation.digest(args.output),
        "categories": {key: value["categories"] for key, value in result["cohorts"].items()}}))


if __name__ == "__main__":
    main()
