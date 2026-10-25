"""Compact retained capability diagnostics without converting them to training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.probes.validate_capabilities import CASES, INTERFACE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact_report(path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    plan_path = path.parent / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(report.get("schema") == "heroes-capability-native-validation-v1"
            and report.get("plan_sha256") == digest(plan_path), "Report does not bind its frozen diagnostic plan")
    require(plan.get("interface") == INTERFACE and report.get("training_input") is False
            and report.get("training_performed") is False and report.get("optimizer_steps") == 0,
            "Capability diagnostic cannot be training evidence")
    rows = report.get("cases", [])
    require(report.get("not_started") == len(plan["schedule"]) - len(rows), "Diagnostic schedule accounting changed")
    compact = []
    for actual, scheduled in zip(rows, plan["schedule"]):
        require(all(actual.get(key) == value for key, value in scheduled.items()), "Started attempt differs from its frozen schedule")
        checks = actual.get("mechanics", {}).get("checks", [])
        if actual.get("status") == "passed":
            require(actual.get("replay_verified") is True
                    and actual["mechanics"].get("history_audit_passed") is True
                    and actual["mechanics"].get("probe_status") == "passed"
                    and actual["mechanics"].get("episode_status") in ("completed", "truncated")
                    and checks and all(check.get("status") == "passed" for check in checks),
                    "Passed mechanics lack clean audit, replay or exact feature checks")
        compact.append({**scheduled, "status": actual["status"], "error": actual.get("error"),
                        "identity_error": actual.get("identity_error"), "replay_verified": actual.get("replay_verified", False),
                        "history_audit_passed": actual.get("mechanics", {}).get("history_audit_passed", False),
                        "episode_status": actual.get("mechanics", {}).get("episode_status"),
                        "episode_directory": actual.get("episode_directory"),
                        "elapsed_seconds": actual.get("mechanics", {}).get("elapsed_seconds"),
                        "checks": checks})
    return {"report_path": str(path), "report_sha256": digest(path), "plan_sha256": digest(plan_path),
            "source_commit": plan["source_commit"], "implementation_sha256": canonical_digest(plan["implementation"]),
            "runtime_sha256": canonical_digest({key: plan[key] for key in ("native_runtime", "binary_sha256", "replay_binary_sha256")}),
            "status": report["status"], "not_started": report["not_started"], "rows": compact}


def summarize(reports):
    require(reports, "Supply explicit retained report files")
    require(len({report["report_path"] for report in reports}) == len(reports), "Do not count the same report twice")
    identities = {(report["implementation_sha256"], report["runtime_sha256"]) for report in reports}
    rows = [row for report in reports for row in report["rows"]]
    coverage = [{"case": case, "player": player,
                 "passed": any(row["case"] == case and row["player"] == player and row["status"] == "passed" for row in rows),
                 "attempts": sum(row["case"] == case and row["player"] == player for row in rows)}
                for case in CASES for player in ("red", "blue")]
    clean = all(report["status"] in ("passed", "unverified") and report["not_started"] == 0 for report in reports)
    fog_pairs = []
    for player in ("red", "blue"):
        witnesses = {}
        for row in rows:
            if row["player"] != player or row["status"] != "passed":
                continue
            for check in row.get("checks", []):
                if check["check"] == "native_dimension_door_fog_attempt" and check["status"] == "passed":
                    witnesses.setdefault((row["seed"], row["repeat"]), {})[check["blocked_fixture"]] = check
        matched = []
        for (seed, repeat), pair in witnesses.items():
            if set(pair) != {False, True}:
                continue
            clear, blocked = pair[False], pair[True]
            same = all(clear.get(key) is not None and clear[key] == blocked.get(key) for key in
                       ("target", "player_view_before_sha256", "offered_spell_targets_sha256", "offered_spell_target_count"))
            matched.append({"seed": seed, "repeat": repeat, "same_player_view_and_candidates": same,
                            "offered_spell_target_count": clear.get("offered_spell_target_count"),
                            "clear_result": clear.get("result_status"), "blocked_result": blocked.get("result_status")})
        fog_pairs.append({"player": player, "passed": any(pair["same_player_view_and_candidates"]
                          and pair["clear_result"] == "capability_applied" and pair["blocked_result"] == "rejected"
                          for pair in matched), "matched_pairs": matched})
    return {"schema": "heroes-capability-native-summary-v1", "interface": INTERFACE,
            "native_coverage_complete": len(identities) == 1 and clean and all(row["passed"] for row in coverage)
                and all(pair["passed"] for pair in fog_pairs),
            "one_frozen_implementation_and_runtime": len(identities) == 1,
            "all_declared_attempts_completed_without_failure": clean,
            "coverage": coverage, "dimension_door_hidden_state_comparisons": fog_pairs, "reports": reports,
            "started_attempts": len(rows), "unverified_attempts": sum(row["status"] == "unverified" for row in rows),
            "failed_attempts": sum(row["status"] == "failed" for row in rows),
            "training_performed": False, "training_input": False, "optimizer_steps": 0,
            "qualification": "Native mechanics coverage only, not policy competence or campaign authorization. "
                "Random branches may be covered by any passed row in their predeclared retained cohort; "
                "every missed branch and failure remains listed. Older draft captures cannot be mixed with final sources."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--prior-report", type=Path, action="append", default=[],
                        help="Preserve earlier draft attempts separately; never count them toward final acceptance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Never overwrite previous compact evidence")
    result = summarize([compact_report(path) for path in args.report])
    result["historical_attempt_inventory"] = []
    for path in args.prior_report:
        historical = compact_report(path)
        historical["excluded_from_final_acceptance"] = True
        for row in historical["rows"]:
            row["checks"] = [{"check": check["check"], "status": check["status"]} for check in row["checks"]]
        result["historical_attempt_inventory"].append(historical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("native_coverage_complete", "started_attempts", "unverified_attempts", "failed_attempts")}))
    return 0 if result["native_coverage_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
