#!/usr/bin/env python3
"""Assemble retained evidence only; never launch games, recompute audits or train."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.probes.audit_ordinary_behavior import aggregate as behavior_aggregate

ARTIFACTS = {}
COMMON = None


def require(condition, text):
    if not condition:
        raise ValueError(text)


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def bind(path):
    path = Path(path).resolve()
    name = rel(path)
    if name not in ARTIFACTS:
        ARTIFACTS[name] = identity(path)
    return name


def maybe_bind(path):
    return bind(path) if Path(path).exists() else None


def check_identity(path, digest):
    name = bind(path)
    require(ARTIFACTS[name]["sha256"] == digest, "Changed source/evidence: " + name)
    return name


def common_identity(manifest):
    global COMMON
    source = manifest["identity"]
    shared = {key: source[key] for key in ("binary_sha256", "bridge_source_sha256", "explorer_rl_sha256",
               "engine_revision", "engine_recipe", "native_runtime_sha256", "contextual_python_sha256")}
    if COMMON is None:
        COMMON = shared
        require(identity(Path(source["binary"]))["sha256"] == shared["binary_sha256"], "Active client binary changed")
        for name, digest in shared["native_runtime_sha256"].items():
            require(identity(Path(source["binary"]).parent / name)["sha256"] == digest,
                    "Active native runtime differs from retained attempts: " + name)
        for group in ("engine_recipe", "contextual_python_sha256"):
            for name, digest in shared[group].items():
                require(identity(ROOT / name)["sha256"] == digest, "Active source differs from recorded evidence: " + name)
    else:
        require(shared == COMMON, "Fresh attempts do not share one unchanged native/headless recipe")


def episode_evidence(directory, history_audit, replay_report=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = load(manifest_path)
    common_identity(manifest)
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "profile" not in path.relative_to(directory).parts:
            files.append(bind(path))
    audit = load(history_audit)
    require(audit["consistent"] and audit["manifest_bound"], "Invalid history audit " + str(history_audit))
    check_identity(manifest_path, audit["manifest"]["file_sha256"])
    for color, player in audit["players"].items():
        check_identity(Path(player["path"]), player["file_sha256"])
    summary = {
        "episode_directory": rel(directory), "manifest": bind(manifest_path),
        "environment": manifest["environment"], "status": manifest["status"],
        "termination_reason": manifest["termination_reason"], "outcomes": manifest["outcomes_by_player"],
        "rewards": manifest["rewards_by_player"], "history_audit": bind(history_audit),
        "record_contract_eligible": audit["record_contract_eligible"],
        "has_eligible_history_transitions": audit["learner_eligible"],
        "history_transition_counts": {c: {"completed": p["transitions"]["completed"],
            "effective_eligible": p["transitions"]["effective_eligible"],
            "effective_bootstrap_allowed": p["transitions"]["effective_bootstrap_allowed"],
            "contract_terminal_rewards": p["transitions"]["contract_terminal_rewards"]}
            for c, p in audit["players"].items()},
        "optimization_admission": "excluded_diagnostic", "training_samples": 0,
    }
    if manifest["status"] != "completed":
        require(all(value is None for value in manifest["outcomes_by_player"].values()), "Unscored episode has outcomes")
        require(all(value is None for value in manifest["rewards_by_player"].values()), "Unscored episode has rewards")
    if manifest["status"] == "failed":
        require(not audit["learner_eligible"], "Failed episode eligible")
        require(all(p["transitions"]["effective_eligible"] == p["transitions"]["effective_bootstrap_allowed"] == 0
                    for p in audit["players"].values()), "Failed episode samples/bootstrap")
    with (directory / "engine/replay.jsonl").open() as stream:
        init = json.loads(stream.readline())
    summary["recorded_profile"] = {"human_players": init["human_players"], "turn_timers": init["turn_timers"]}
    require(summary["recorded_profile"] == {"human_players": 0, "turn_timers": False}, "Unsupported actual profile")
    server_receipt = directory / "engine/terminal-barrier-server.jsonl"
    if server_receipt.exists():
        receipt = load(server_receipt)
        summary["terminal_receipt"] = {key: receipt[key] for key in ("origin", "request", "client_packets", "outcomes")}
        summary["terminal_receipt"]["scoring_accepted"] = manifest["status"] == "completed"
    if replay_report is not None:
        replay_report = Path(replay_report)
        report = load(replay_report)
        native = report["native"]["summary"]
        require(report["verified"] and report["native"]["verified"] and native["server_verified"], "Replay did not verify")
        require(all(value == 0 for value in native["unchecked"].values()), "Replay left unchecked evidence")
        require(report["histories"]["verified"], "Replay lacks raw-wire history reconstruction")
        check_identity(manifest_path, report["inputs"]["manifest"]["sha256"])
        for path in replay_report.parent.iterdir():
            if path.is_file(): bind(path)
        summary["native_replay"] = {"report": bind(replay_report), "verified": True,
             "checked": native["checked"], "unchecked": native["unchecked"], "boundary": native["boundary"],
             "captured_prefix_only": native["captured_prefix_only"],
             "open_final_transaction": native["open_final_transaction"],
             "final_client_position": native["final_client_position"]}
    else:
        require(manifest["status"] == "failed", "Valid attempt has no independent replay")
        summary["native_replay"] = {"verified": False, "reason": "Failed unscored recording retained; not accepted as a natural outcome"}
    return summary


def mechanics():
    specs = []
    for name in ("contextual-neutral-terminal-red-98501", "contextual-neutral-terminal-red-98501-r2",
                 "contextual-neutral-terminal-red-98501-r3", "contextual-neutral-terminal-blue-98502"):
        directory = ROOT / "runs" / name
        report = load(directory / "report.json")
        episode = Path(report["manifest"]).parent
        specs.append((directory, directory / "report.json", episode.parent / "history-audit.json",
                      directory / "native-replay/contextual-replay-verification.json" if name != "contextual-neutral-terminal-red-98501" else None))
    town = ROOT / "runs/contextual-town-capture-98201"
    bind(town / "report.json")
    for color in ("red", "blue"):
        specs.append((town, town / f"capture-{color}/report.json", town / f"capture-{color}/history-audit.json",
                      town / f"replay-{color}/contextual-replay-verification.json"))
    for name, case in (("contextual-gap-affordability-97609", "recruitment-budget"),
                       ("contextual-gap-hero-red-84001", "hero-casualties"),
                       ("contextual-gap-hero-blue-84001", "hero-casualties"),
                       ("contextual-gap-encounters-red-97701", "encounters"),
                       ("contextual-gap-encounters-blue-97702", "encounters"),
                       ("contextual-gap-encounters-red-97701-r2", "encounters"),
                       ("contextual-gap-encounters-blue-97702-r2", "encounters")):
        directory = ROOT / "runs" / name
        specs.append((directory, directory / "report.json", directory / case / "history-audit.json",
                      directory / "replay/contextual-replay-verification.json"))
    source_candidates = list((ROOT / "validation/probes").glob("*.py"))
    source_candidates += list((ROOT / "runs").glob("contextual-*/probe-source.py"))
    source_candidates += list((ROOT / "runs").glob("contextual-*/probe-used.py"))
    sources_by_hash = {}
    for candidate in source_candidates:
        sources_by_hash.setdefault(identity(candidate)["sha256"], candidate)
    rows = []
    for directory, report_path, audit_path, replay_path in specs:
        report, plan = load(report_path), load(directory / "plan.json")
        row = episode_evidence(Path(report["manifest"]).parent, audit_path, replay_path)
        row.update(report=bind(report_path), plan=bind(directory / "plan.json"), case=report["case"],
                   probe_status=report["probe_status"], probe_error=report.get("error"), choices=report["attempts"],
                   engine_seed=report["engine_seed"], check_results=[{key: value for key, value in check.items()
                       if key in ("check", "status", "reason", "error", "day", "count", "cost", "stock", "stock_before",
                                  "stock_after", "slider_max", "invalid_count", "resources", "unit_cost", "casualties",
                                  "before_units", "after_units", "terminal_day", "terminal_request", "own_town_count")}
                       for check in report["checks"]])
        require(plan["probe_sha256"] in sources_by_hash, "No preserved exact probe source: " + str(directory))
        row["probe_source"] = bind(sources_by_hash[plan["probe_sha256"]])
        if report.get("neutral_boundary"):
            row["neutral_boundary"] = {key:value for key,value in report["neutral_boundary"].items()
                                       if key != "independent_native_replay_verified"}
        for name in ("probe-source.py", "probe-used.py"):
            if (directory / name).exists(): check_identity(directory / name, plan["probe_sha256"])
        fixture_path = plan.get("fixture_manifest") or report.get("fixture_manifest")
        if not fixture_path and "hero-" in directory.name:
            color = "red" if "hero-red" in directory.name else "blue"
            fixture_path = ROOT / f"runs/contextual-gap-hero-map-{color}-84001/fixture.json"
        if fixture_path:
            fixture_path = Path(fixture_path)
            fixture = load(fixture_path)
            row["fixture_manifest"] = bind(fixture_path)
            for field in ("source", "base", "fixture"):
                if field in fixture: check_identity(fixture[field]["path"], fixture[field]["sha256"])
        attempt_log = Path(report["manifest"]).parent.parent / "attempts.jsonl"
        if attempt_log.exists(): bind(attempt_log)
        rows.append(row)
    require(len(rows) == 13, "Mechanical attempt count changed")
    return rows


def ordinary(*, final):
    rows, batches, causes, missing = [], [], [], []
    declared = load(ROOT / "validation/contextual-gap-plan-2026-10-25.json")["ordinary_cohort"]
    for batch_name in ("batch-a", "batch-b"):
        directory = ROOT / "runs/contextual-gaps-ordinary" / batch_name
        if not (directory / "report.json").exists():
            missing.append(batch_name)
            continue
        report, plan = load(directory / "report.json"), load(directory / "cohort-plan.json")
        require(report["optimizer_updates"] == 0 and report["training_performed"] is False, "Cohort performed training")
        check_identity(directory / "cohort-plan.json", report["cohort_plan"]["sha256"])
        check_identity(directory / "untrained-policy.pt", plan["policy_checkpoint_file_sha256"])
        batches.append({"name": batch_name, "plan": bind(directory / "cohort-plan.json"),
                        "report": bind(directory / "report.json"), "source": report["source"],
                        "checkpoint": bind(directory / "untrained-policy.pt")})
        for attempt in report["attempts"]:
            index = attempt["episode"]
            episode = directory / f"episode-{index:06d}"
            replay = directory / f"replay-{index}/contextual-replay-verification.json"
            neural_path = directory / f"neural-recompute-{index}.json"
            cause_path = directory / f"cause-audit-{index}.json"
            required = (episode / "manifest.json", episode / "neural-audit.json", replay, neural_path, cause_path)
            if not all(path.exists() for path in required) or attempt.get("status") not in ("completed", "truncated", "failed"):
                missing.append(f"{batch_name}/{index}")
                continue
            require(attempt["weights_unchanged"] and attempt["training_samples"] == 0, "Cohort weights/admission changed")
            require(attempt["learner_weights_sha256"] == declared["learner_weights_sha256"]
                    and attempt["opponent_weights_sha256"] == declared["opponent_weights_sha256"], "Checkpoint differs from predeclared actor/value weights")
            row = episode_evidence(episode, episode / "neural-audit.json", replay)
            for name, digest in attempt["files"].items(): check_identity(directory / name, digest)
            neural = load(neural_path)
            require(neural["optimizer_updates"] == 0 and len(neural["episodes"]) == 1, "Invalid neural recompute")
            neural_episode = neural["episodes"][0]
            require(neural_episode["passed"] and Path(neural_episode["episode"]).resolve() == episode.resolve(), "Neural episode mismatch")
            check_identity(directory / "untrained-policy.pt", neural["checkpoint_sha256"])
            for name, digest in neural_episode["files"].items(): check_identity(episode / name, digest)
            for player in neural_episode["players"]:
                require(player["weights_unchanged"] and player["optimizer_admission_rejected"] and not player["optimization_eligible"],
                        "Diagnostic optimization admission did not reject")
            cause = load(cause_path)
            require(Path(cause["episode"]).resolve() == episode.resolve(), "Cause audit episode mismatch")
            causes.append(cause)
            for color, player in cause["players"].items():
                path = episode / f"agents/{color}/history-v3.jsonl"
                require(identity(path) == player["history_identity"], "Cause audit history mismatch")
            row.update(batch=batch_name, episode_index=index, engine_seed=attempt["engine_seed"],
                       learner_player=attempt["learner_player"], policy_seeds=attempt["policy_seeds"],
                       category=attempt["category"], neural_recompute_report=bind(neural_path),
                       cause_audit=bind(cause_path), cause_findings=cause["findings"],
                       choices=sum(player["submitted_total"] for player in cause["players"].values()),
                       neural_recompute={"passed": True, "players": neural_episode["players"]},
                       submitted_actions={color: value["submitted_actions"] for color, value in cause["players"].items()})
            rows.append(row)
    if final:
        require(not missing and len(rows) == 16, "Ordinary cohort incomplete: " + repr(missing))
        expected = []
        for batch_index, execution in enumerate(declared["execution_batches"]):
            local_episode = 0
            for seed in execution["engine_seeds"]:
                for seat in (0, 1):
                    local_episode += 1
                    expected.append((f"batch-{'ab'[batch_index]}", local_episode, seed, seat,
                                     [execution["policy_seed"] + 2 * local_episode,
                                      execution["policy_seed"] + 2 * local_episode + 1]))
        actual = [(row["batch"], row["episode_index"], row["engine_seed"], row["learner_player"], row["policy_seeds"])
                  for row in rows]
        require(actual == expected, "Actual ordinary attempt ordering/seeds/seats differ from predeclaration")
    return rows, batches, causes, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--tests", type=Path, default=ROOT / "runs/contextual-gaps-tests-final.log")
    parser.add_argument("--prepared", type=Path, default=ROOT / "runs/contextual-gaps-prepared.log")
    args = parser.parse_args()
    final = args.output is not None
    require(not final or not args.output.exists(), "Never overwrite a finalized evidence artifact")
    mechanism_rows = mechanics()
    ordinary_rows, batches, causes, missing = ordinary(final=final)
    if not final:
        print(json.dumps({"mechanics": len(mechanism_rows), "ordinary_complete_with_all_audits": len(ordinary_rows),
                          "missing": missing, "artifact_count": len(ARTIFACTS)}, indent=2))
        return
    require(args.aggregate is not None, "Final evidence requires the independent cohort behavior aggregate")
    aggregate = load(args.aggregate)
    require(aggregate["aggregate"] == behavior_aggregate(causes)
            and [Path(path).resolve() for path in aggregate["episode_audits"]]
                == [(ROOT / row["cause_audit"]).resolve() for row in ordinary_rows], "Aggregate differs from individual cause audits")
    tests_text = args.tests.read_text()
    matches = re.findall(r"Ran (\d+) tests in ([\d.]+)s", tests_text)
    require(matches and re.search(r"^OK\s*$", tests_text, re.M) and not re.search(r"^FAILED|skipped=", tests_text, re.M), "Final suite is incomplete or not all-pass")
    require("match exactly" in args.prepared.read_text(), "Missing final exact preparation check")
    plan_path = ROOT / "validation/contextual-gap-plan-2026-10-25.json"
    plan = load(plan_path)
    source_checkpoint = ROOT / plan["ordinary_cohort"]["checkpoint_path"]
    check_identity(source_checkpoint, plan["ordinary_cohort"]["checkpoint_file_sha256"])
    checkpoint_check_path = ROOT / "runs/contextual-gaps-checkpoint-identity.json"
    checkpoint_check = load(checkpoint_check_path)
    require(checkpoint_check["expected_unchanged"] is True
            and checkpoint_check["path"] == rel(source_checkpoint)
            and {key: checkpoint_check[key] for key in ("sha256", "bytes")} == identity(source_checkpoint),
            "Independent original checkpoint identity check disagrees")
    source = batches[0]["source"]
    for batch in batches:
        batch_source = batch.pop("source")
        require(batch_source["source_sha256"] == source["source_sha256"], "Cohort source drift")
        batch["source_commit_at_launch"] = batch_source["git_commit"]
    statuses = Counter(row["status"] for row in ordinary_rows)
    diagnostic_statuses = Counter(row["status"] for row in mechanism_rows)
    all_rows = ordinary_rows + mechanism_rows
    native_checked = Counter()
    for row in all_rows:
        if row["native_replay"]["verified"]:
            native_checked.update(row["native_replay"]["checked"])
    neural_players = [player for row in ordinary_rows for player in row["neural_recompute"]["players"]]
    neural_totals = {name: sum(player[name] for player in neural_players)
                     for name in ("frames", "actions_recomputed", "values_recomputed", "conditionals_recomputed")}
    neural_totals.update(player_rollouts=len(neural_players),
        optimizer_admission_rejections=sum(player["optimizer_admission_rejected"] for player in neural_players),
        maximum_absolute_difference={name: max(player["maximum_absolute_difference"][name] for player in neural_players)
                                     for name in ("joint_log_probability", "conditional_log_probability", "value")})
    result = {
        "schema": "heroes-contextual-validation-gaps-v1", "date": "2026-10-25",
        "assembled_at": datetime.now(timezone.utc).isoformat(), "predeclared_plan": bind(plan_path),
        "scope": "One new bounded fixed-weight cohort plus separately labeled native mechanics diagnostics; no prior-stage attempt counts are included",
        "training_performed": False, "optimizer_updates": 0, "paid_compute": False,
        "common_identity": {**COMMON, "native_gameplay_reference_commit": subprocess.check_output(["git", "rev-parse", "609083a"], cwd=ROOT, text=True).strip(),
                            "evidence_assembly_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                            "ordinary_inference_source": source,
                            "inference_difference_from_native_reference": "Report labels were corrected to require separate replay verification; no action, reward, model or game behavior changed"},
        "checkpoint": {"original": bind(source_checkpoint), "original_file_unchanged": True,
                       "independent_identity_check": bind(checkpoint_check_path),
                       "learner_weights_sha256": plan["ordinary_cohort"]["learner_weights_sha256"],
                       "opponent_weights_sha256": plan["ordinary_cohort"]["opponent_weights_sha256"],
                       "note": "Batch checkpoint containers are reserialized and bound separately; both actor/value parameter digests remain unchanged"},
        "totals": {"native_attempts": len(all_rows), "ordinary_attempts": len(ordinary_rows),
                   "ordinary_statuses": dict(statuses), "diagnostic_attempts": len(mechanism_rows),
                   "diagnostic_statuses": dict(diagnostic_statuses),
                   "independent_native_replays_verified": sum(row["native_replay"]["verified"] for row in all_rows),
                   "native_replay_checked_records": dict(native_checked), "native_replay_unchecked_records": 0,
                   "ordinary_natural_outcomes": statuses["completed"], "diagnostic_natural_outcomes": diagnostic_statuses["completed"],
                   "ordinary_policy_choices": sum(row["choices"] for row in ordinary_rows),
                   "diagnostic_evaluator_choices": sum(row["choices"] for row in mechanism_rows),
                   "training_samples": 0},
        "ordinary": {"planned_attempts": 16, "finished_attempts": 16, "batches": batches, "attempts": ordinary_rows,
                     "behavior_audit": bind(args.aggregate), "observed_behavior_aggregate": aggregate["aggregate"],
                     "neural_recompute_totals": neural_totals,
                     "interpretation": "Paired engine seeds and seats are not independent samples. Cutoffs/failures have no outcome/reward, never draws. Diagnostic records are rejected by optimizer admission even when their trajectories satisfy the history contract."},
        "mechanics": {"attempts": mechanism_rows, "interpretation": "Explicit evaluation actions and isolated initial-state fixtures are not ordinary policy outcomes, demonstrations or training data"},
        "new_coverage": {
            "neutral_battle_terminal": "Both losing colors: real day-2 final-hero defeat, request player=-1/type198, full callback barrier and independent native replay",
            "town_capture": "Both attacking colors on unchanged authored baseline: displayed enemy town capture followed by unchanged seven-day townless loss with surviving defender hero",
            "recruitment_affordability": "Displayed stock14/gold475/price60/max7; count8 rejected unchanged, exact7 purchased for420; remaining stock7/gold55/max0 and count1 rejected unchanged",
            "closed_hero_window": "Both colors: isolated corresponding neutral guard6-to20 fixture; same hero survives, closed fields absent, original snapshot immutable, explicit reopen reveals army25-to20",
            "encounters": "Both colors: selected visible resource/battle/mine with exact visit/consumption/issued/ack/result association, resource500 credit, surviving won battle, mine observed self-owned and no further route commands after encounter",
            "timer_profile": "All fresh actual replay init records confirm human_players0 and turn_timers=false; executable source/config audit verifies default-zero timers, headless AI release, human-only timer effects and fail-closed unsupported origins",
        },
        "retained_failures": {
            "neutral_first_attempt": "Authentic neutral receipt was not scored because Blue never acted; failure history has zero eligible transitions/bootstrap and no outcome/reward. Preserved source snapshot matches original pre-run hash.",
            "first_encounter_probes": "Red/Blue diagnostic setup selected a visible mine without a known route; probe assertions failed. Both actual episodes were valid unscored collection-window cutoffs with passing history audit and native replay. Bounded known-frontier diagnostic repair was rerun separately; engine behavior unchanged.",
            "non_game_cli_refusal": maybe_bind(ROOT / "runs/contextual-gaps-ordinary-98111-98118.log"),
        },
        "remaining_scope": {
            "initial_timer_terminal_origins": "Excluded and unsupported; actual initial/timer natural completion is not claimed and remains fail closed",
            "ordinary_learning_gate": "Zero natural outcomes, if observed, describe sparse terminal signal in this bounded evaluation rather than an independent correctness blocker. Training remains disabled by user authorization; explicit troop transfer is the next requested capability before training preparation. No campaign is launched or authorized here.",
            "exceptional_native_ack_rejection": "Fail-closed handling is covered by contract/source tests; an actual exceptional native negative acknowledgement remains outside newly observed live coverage",
            "unchanged_broader_interface_limits": "Ordinary land movement and implemented hero/town/construction/recruitment panels only; deferred query/special windows, spells and artifact operations, boats/teleports/flight, and broader army controls are not certified by these checks",
            "garrison_hero_recruitment": "Source-audited but not reached through current implemented actions; no new native claim",
        },
        "checks": {"final_full_suite": {"log": bind(args.tests), "tests_passed": int(matches[-1][0]), "seconds": float(matches[-1][1]), "skipped": 0},
                   "exact_root_generated_agreement": bind(args.prepared), "assembler": bind(Path(__file__))},
        "artifact_directory_semantics": "Join each repository-relative directory with its inner filename. Every sha256 and byte count describes the complete retained file bytes, not a parsed/canonical history digest.",
    }
    directories = {}
    for name, value in ARTIFACTS.items():
        path = Path(name)
        directories.setdefault(str(path.parent), {})[path.name] = value
    result["artifact_directories"] = directories
    rendered = json.dumps(result, indent=1, sort_keys=True)
    # Keep each byte-identity pair on one line; structure and all evidence remain
    # unchanged while the large file catalog stays compact and reviewable.
    rendered = re.sub(r'\{\n\s*"bytes": (\d+),\n\s*"sha256": "([a-f0-9]{64})"\n\s*\}',
                      r'{"bytes": \1, "sha256": "\2"}', rendered)
    raw = (rendered + "\n").encode()
    require(len(raw) < 250_000, "Compact evidence grew beyond 250KB: " + str(len(raw)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream: stream.write(raw)
    print(json.dumps({"output": rel(args.output), "bytes": len(raw), "artifacts": len(ARTIFACTS), "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
