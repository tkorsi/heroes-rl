"""Provenance and timing for a new, explicitly matched evaluation cohort.

Historical plans describe their captured identities. They are never required to
match today's files, rewritten, resumed, or admitted by a later verifier here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time


AUTHORIZATION = {
    "source_thread": "01a16d9e-5f8d-7cb0-a939-3ff005a741e4",
    "human_turn": "01a17292-93d4-79b0-b6e7-b9de3cbe4e4a",
    "human_message": "01a17292-93ef-7b70-a4ee-bdb967d25a49",
    "confirmed_before_preparation": True,
}
PROVENANCE = {
    "schema": "heroes-evaluation-revalidation-v2",
    "purpose": "repaired_runtime_revalidation",
    "seed_use": "reused_matched_seeds",
    "held_out": False,
    "replaces_prior_attempts": False,
    "repaired_baseline_commit": "9193db35857e07ddbc45e98a1dd7c3f4a8c07da0",
}
COMMON_FIELDS = ("source", "implementation", "native", "runtime", "map", "checkpoint",
    "frozen", "checkpoint_source", "checkpoint_provenance", "binary", "data", "game_map", "replay_binary")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def binding(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def version(plan, legacy_schema):
    schema = plan["schema"]
    if schema == legacy_schema + "-plan":
        return 1
    require(schema == legacy_schema.removesuffix("v1") + "v2-plan", "Unknown evaluation plan version")
    return 2


def prior_cohort(plan_path, report_path, *, reader, legacy_schema):
    """Bind the stopped v1 schedule without checking its old source at HEAD."""
    plan_path, report_path = Path(plan_path).resolve(), Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    require(report.get("schema") == legacy_schema + "-result" and report.get("status") == "stopped",
            "Revalidation requires the preserved stopped v1 cohort")
    plan = reader(plan_path, report["plan_sha256"])
    require(plan["schema"] == legacy_schema + "-plan", "Prior cohort must be v1")
    require(report_path == Path(plan["execution_output"]) / "report.json", "Prior report is outside its cohort")
    require(plan.get("training_performed") is False and report.get("training_performed") is False
            and all(type(report.get(k)) is int and report[k] == 0
                    for k in ("optimizer_updates", "optimizer_steps", "training_samples"))
            and report.get("frozen_after_file_sha256") == plan["checkpoint"]["sha256"],
            "Prior cohort lacks its frozen zero-update declaration")
    rows = report.get("attempts")
    require(isinstance(rows, list) and len(rows) == len(plan["schedule"]) == 8,
            "Prior cohort must preserve all eight scheduled rows")
    for row, scheduled in zip(rows, plan["schedule"]):
        require(all(equal(row.get(k), v) for k, v in scheduled.items()), "Prior schedule changed")
        require(row.get("status") in ("passed", "native_failure", "failed", "budget_exhausted", "incomplete", "not_started"),
                "Prior cohort has an unfinished or unknown attempt")
        require(type(row.get("training_samples")) is int and row["training_samples"] == 0
                and type(row.get("optimizer_steps")) is int and row["optimizer_steps"] == 0,
                "Prior attempt is not evaluation-only")
        if row["status"] != "passed":
            require(row.get("outcomes") == {"red": None, "blue": None}
                    and row.get("rewards") == {"red": None, "blue": None}, "Prior failed attempt received a score")
    return {"plan": binding(plan_path), "report": binding(report_path), "status": "stopped",
            "schedule": plan["schedule"], "checkpoint_sha256": plan["checkpoint"]["sha256"],
            "map_sha256": plan["map"]["sha256"], "execution_output": plan["execution_output"]}


def prepare_provenance(args, *, reader, legacy_schema, configuration, wanted_schedule):
    enabled = getattr(args, "revalidation_v2", False)
    prior_plan, prior_report = getattr(args, "prior_plan", None), getattr(args, "prior_report", None)
    if not enabled:
        require(prior_plan is None and prior_report is None, "Prior artifacts require explicit --revalidation-v2")
        return None
    require(prior_plan is not None and prior_report is not None,
            "Revalidation requires --prior-plan and --prior-report")
    prior = prior_cohort(prior_plan, prior_report, reader=reader, legacy_schema=legacy_schema)
    require(equal(prior["schedule"], wanted_schedule)
            and prior["checkpoint_sha256"] == configuration["checkpoint_sha256"]
            and prior["map_sha256"] == configuration["map_sha256"], "Reused schedule or fixed artifacts changed")
    require(not args.output.expanduser().resolve().is_relative_to(Path(prior["execution_output"]).parent),
            "New cohort must be outside the preserved prior cohort")
    return {**PROVENANCE, "prior": prior}


def validate_provenance(plan, *, reader, legacy_schema):
    require(equal(plan.get("authorization"), AUTHORIZATION), "Revalidation authorization changed")
    require("seed_freshness" not in plan, "Reused seeds cannot claim new seed freshness")
    value = plan.get("revalidation")
    require(isinstance(value, dict) and set(value) == set(PROVENANCE) | {"prior"}
            and all(equal(value.get(k), v) for k, v in PROVENANCE.items()), "Revalidation seed or preservation claim changed")
    prior = value["prior"]
    actual = prior_cohort(prior["plan"]["path"], prior["report"]["path"], reader=reader, legacy_schema=legacy_schema)
    require(equal(prior, actual), "Preserved prior cohort artifacts changed")
    require(equal(prior["schedule"], plan["schedule"])
            and prior["checkpoint_sha256"] == plan["checkpoint"]["sha256"]
            and prior["checkpoint_sha256"] == plan["configuration"]["checkpoint_sha256"]
            and prior["map_sha256"] == plan["map"]["sha256"]
            and prior["map_sha256"] == plan["configuration"]["map_sha256"],
            "Revalidation is not matched to its prior cohort")
    require(not Path(plan["execution_output"]).parent.is_relative_to(Path(prior["execution_output"]).parent),
            "New cohort overlaps preserved prior output")


def comparison_identity(plan):
    """Identical source and runtime are required across the new A/B cohorts."""
    return copy.deepcopy({key: plan[key] for key in COMMON_FIELDS})


def record_progress_safely(save, attempt, phase):
    """Progress persistence must never prevent failure retention or cleanup."""
    try:
        save(phase)
    except Exception as error:
        attempt.update(status="failed", outcomes={"red": None, "blue": None}, rewards={"red": None, "blue": None})
        attempt.setdefault("progress_write_errors", []).append(f"{phase}: {type(error).__name__}: {error}")


def match_reference(plan):
    reference = plan["reference"]
    require(reference.get("plan_schema") == "heroes-long-dialog-evaluation-v2-plan",
            "Built-in revalidation requires the new v2 neural reference")
    require(equal(reference.get("comparison_identity"), comparison_identity(plan)),
            "Neural and built-in cohort source/runtime/artifact identities differ")
    require(plan["source"].get("tracked_worktree_dirty") is False, "Revalidation source must be committed and clean")
    for own, neural in zip(plan["schedule"], reference["schedule"]):
        require(own["episode"] == neural["episode"] and own["engine_seed"] == neural["engine_seed"]
                and own["learner_player"] == neural["learner_player"]
                and own["policy_seed"] == neural["policy_seeds"][own["learner_player"]],
                "Comparison learner seeds or seats differ")


class PhaseTimings:
    """Cumulative wall time, including repeated progress saves in one phase."""
    def __init__(self, started, clock=time.monotonic):
        self.clock = clock
        self.last = started
        self.phase = "initialization"
        self.elapsed = {}

    def advance(self, phase):
        now = self.clock()
        require(now >= self.last, "Monotonic phase clock moved backwards")
        if self.phase != "finished":
            self.elapsed[self.phase] = self.elapsed.get(self.phase, 0.0) + now - self.last
        self.last, self.phase = now, phase
        return dict(self.elapsed), now
