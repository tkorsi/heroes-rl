"""Bind an eight-game evaluation report and describe its player-visible behavior.

This offline command never loads weights, launches games, replays, or trains.
Missing artifacts remain explicit; changed bound bytes or schedule drift fail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.headless.contextual_evidence import strict_json
from experiments.learner import evaluate_long_dialogs as evaluation
from experiments.learner.evaluation_behavior import analyze_episode, aggregate_episodes

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-long-dialog-evaluation-evidence-v1"
REFERENCES = tuple(ROOT / "runs/adventure-dialogs-2026-10-25/policy-1" / f"episode-{n:06d}" for n in (1, 2))
STATUSES = {"not_started", "running", "passed", "native_failure", "failed", "budget_exhausted", "incomplete"}
METRICS = ("frames", "choices", "decision_frames", "first_public_day", "last_public_day",
    "action_counts", "event_counts", "result_counts", "missing_results", "native_acknowledgements",
    "navigation_and_selection", "army_rearrangement", "decision_context_counts", "movement",
    "known_tiles", "owned_heroes_first_last", "owned_towns_first_last", "recruitment", "hiring",
    "upgrading", "army_operations")


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def compact_behavior(value):
    """Keep denominators and counts, without coordinate or per-action arrays."""
    result = {key: value[key] for key in ("schema", "episode_directory", "status", "bindings", "coverage", "aggregate")}
    result.update(episode=value.get("episode"), throughput=value.get("throughput"), players={})
    for color, player in value["players"].items():
        metrics = player["metrics"]
        compact = None
        if metrics is not None:
            compact = {key: metrics.get(key) for key in METRICS}
            compact["counts"] = {key: len(metrics.get(key, [])) for key in ("own_position_changes",
                "own_hero_roster_changes", "own_town_roster_changes", "town_capture_evidence",
                "visible_town_visits", "visible_town_owner_changes", "dialog_answers", "terminal_messages", "cutoffs")}
            compact["battles"] = {key: metrics.get("battles", {}).get(key) for key in ("starts", "completions")}
        result["players"][color] = {"coverage": player["coverage"], "metrics": compact}
    return result


def summarize(plan_path, report_path, *, reference_episodes=REFERENCES, output_path=None):
    """Return evidence without writing. Declared hash corruption always raises."""
    plan_path, report_path = Path(plan_path).resolve(), Path(report_path).resolve()
    report = strict_json(report_path.read_bytes())
    require(report.get("schema") == evaluation.SCHEMA + "-result", "Unexpected execution report schema")
    plan = evaluation.read_plan(plan_path, report["plan_sha256"])
    execution = Path(plan["execution_output"]).resolve()
    require(report_path == execution / "report.json", "Report is outside the prepared execution output")
    require(report.get("status") in ("running", "finished", "stopped"), "Unknown report status")
    require(plan.get("training_performed") is False and report.get("training_performed") is False
            and all(type(report.get(k)) is int and report[k] == 0 for k in ("optimizer_updates", "optimizer_steps", "training_samples")),
            "Evaluation reports training activity or lacks its zero-update declaration")
    require(equal(report.get("frozen_before"), plan["frozen"]), "Initial frozen identity differs from plan")
    if "frozen_after_file_sha256" in report:
        require(report["frozen_after_file_sha256"] == plan["checkpoint"]["sha256"], "Final checkpoint identity changed")
    rows = report.get("attempts")
    require(isinstance(rows, list) and len(rows) == len(plan["schedule"]) == 8, "Exactly eight scheduled rows must survive")
    bindings, issues = {}, []

    def bind(path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            issues.append({"path": str(path), "issue": "bound_file_missing"})
            return None
        actual = evaluation.binding(path)
        if expected is not None:
            wanted = {"sha256": expected} if isinstance(expected, str) else expected
            require(all(actual[key] == wanted[key] for key in ("sha256", "bytes") if key in wanted), "Bound artifact changed: " + str(path))
        bindings[str(path)] = {key: actual[key] for key in ("sha256", "bytes")}
        return actual

    bind(plan_path, report["plan_sha256"])
    bind(report_path)
    bind(evaluation.CONFIG)
    for key in ("checkpoint", "replay_binary", "seed_freshness"):
        item = plan.get(key)
        if item:
            bind(item["path"], item)
        else:
            issues.append({"issue": "plan_binding_missing", "field": key})
    for name, sha in plan.get("implementation", {}).items():
        bind(ROOT / name, sha)
    for name, sha in plan.get("source", {}).get("source_sha256", {}).items():
        bind(ROOT / name, sha)
    binary = Path(plan["binary"])
    bind(binary, plan["native"]["binary_sha256"])
    for name, sha in plan["runtime"].items():
        bind(binary.parent / name, sha)
    bind(plan["game_map"], plan["map"]["sha256"])
    excluded = Path(output_path).resolve() if output_path else None
    for path in sorted(execution.rglob("*")):
        if not path.is_file() or path.resolve() == excluded or any(".tmp-" in p or p.endswith(".tmp") for p in path.relative_to(execution).parts):
            continue
        require(not path.is_symlink() and path.resolve().is_relative_to(execution), "Execution artifact escapes output")
        bind(path)

    attempts, behaviors = [], []
    for row, expected in zip(rows, plan["schedule"]):
        require(all(key in row and equal(row[key], value) for key, value in expected.items()), "Attempt differs from fixed schedule")
        require(row.get("status") in STATUSES, "Unknown attempt status")
        require(row.get("training_samples") == 0 and row.get("optimizer_steps") == 0, "Attempt is not evaluation-only")
        if row["status"] != "passed":
            require(row.get("outcomes") == {"red": None, "blue": None} and row.get("rewards") == {"red": None, "blue": None}, "Unverified attempt received a score")
        folder = execution / f"episode-{expected['episode']:06d}"
        if row.get("episode_directory"):
            require(Path(row["episode_directory"]).resolve() == folder, "Attempt episode path differs from schedule")
        compact = {key: value for key, value in row.items() if key not in ("last_worker_result", "behavior")}
        compact["behavior"] = None
        if folder.exists():
            require(row["status"] != "not_started", "Unstarted row has episode artifacts")
            value = analyze_episode(folder)
            for name, identity in value["bindings"].items():
                bind(folder / name, identity)
            if row.get("behavior"):
                saved = bind(row["behavior"]["path"], row["behavior"])
                if saved:
                    require(equal(strict_json(Path(saved["path"]).read_bytes()), value), "Saved behavior differs from manifest-bound analysis")
            if value.get("episode"):
                require(str(value["episode"]["episode"]) == str(expected["episode"]) and value["episode"]["seed"] == expected["engine_seed"], "Manifest episode or seed differs from schedule")
            if row["status"] == "passed":
                require(value["status"] == "complete" and row.get("replay_verified") is True
                        and equal(row["outcomes"], value["episode"]["outcomes_by_player"])
                        and equal(row["rewards"], value["episode"]["rewards_by_player"]), "Passed attempt lacks complete matching evidence")
            behaviors.append(value)
            compact["behavior"] = compact_behavior(value)
        elif row["status"] != "not_started":
            issues.append({"episode": expected["episode"], "issue": "episode_directory_missing"})
        attempts.append(compact)

    references = []
    for folder in reference_episodes:
        value = analyze_episode(folder)
        for name, identity in value["bindings"].items():
            bind(Path(folder) / name, identity)
        references.append(value)
    return {"schema": SCHEMA, "plan": plan, "plan_binding": evaluation.binding(plan_path),
        "report_binding": evaluation.binding(report_path), "bindings": bindings,
        "coverage": {"complete": not issues and report["status"] == "finished"
                     and len(behaviors) == 8 and all(row["status"] == "passed" for row in rows)
                     and all(x["status"] == "complete" for x in behaviors + references), "issues": issues},
        "execution": {key: value for key, value in report.items() if key not in ("attempts", "behavior_summary")},
        "attempt_counts": dict(Counter(row["status"] for row in rows)), "attempts": attempts,
        "behavior_summary": aggregate_episodes(behaviors),
        "earlier_ordinary_comparison": {"episodes": [compact_behavior(v) for v in references],
            "summary": aggregate_episodes(references), "new_replays": 0,
            "interpretation": "Descriptive player-visible counts only. Different budgets and seeds do not establish improved strategy or causal effects."},
        "training_performed": False, "optimizer_updates": 0, "optimizer_steps": 0,
        "training_samples": 0, "policy_input": False, "reward_input": False,
        "limitations": ["Missing or failed episode evidence describes only the available observed prefix; null means unavailable, not zero.",
            "Nonpassed attempts remain unscored, including native failures, shared-budget stops and unstarted games.",
            "This command verifies bindings and descriptive history structure; it does not rerun native replay or neural reconstruction.",
            "Battle counts are player perspectives. Army redistribution counts net changes, not gross like-stack swap traffic.",
            "Own roster changes do not reveal hidden enemy fate; captures require player-visible corroboration.",
            "Source bytes are checked against the prepared plan; a later documentation-only Git commit does not invalidate evidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists; preserve earlier evidence")
    result = summarize(args.plan, args.report, output_path=args.output)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": evaluation.digest(args.output), "attempt_counts": result["attempt_counts"]}))


if __name__ == "__main__":
    main()
