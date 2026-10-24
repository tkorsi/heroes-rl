#!/usr/bin/env python3
"""Assemble retained transfer evidence only. No games, replay or optimizer calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.learner import first_run
from experiments.learner.inference import source_identity, ordinary_map_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import VcmiLauncher

ARTIFACTS = {}
COMMON = None


def require(condition, message):
    if not condition: raise ValueError(message)


def load(path): return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    value = first_run.file_identity(path)
    if name in ARTIFACTS: require(value == ARTIFACTS[name], "Artifact changed during assembly: " + name)
    ARTIFACTS[name] = value
    return name


def artifact(path):
    name = bind(path)
    require(not Path(name).is_absolute(), "Readiness artifacts must be in this workspace")
    return {"path": name, **ARTIFACTS[name]}


def check_file(path, expected):
    name = bind(path)
    require(ARTIFACTS[name]["sha256"] == expected, "Recorded byte identity changed: " + name)
    return name


def preserve_tree(directory):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and "profile" not in path.relative_to(directory).parts:
            bind(path)


def episode(directory, audit_path, replay_path):
    global COMMON
    manifest = load(directory / "manifest.json")
    keys = ("binary_sha256", "explorer_rl_sha256", "engine_revision", "engine_recipe", "native_runtime_sha256", "contextual_python_sha256")
    identity = {key: manifest["identity"][key] for key in keys}
    if COMMON is None: COMMON = identity
    else: require(identity == COMMON, "Native/adapter recipe changed between retained transfer attempts")
    preserve_tree(directory)
    audit = load(audit_path)
    require(audit["consistent"] and audit["manifest_bound"], "History audit did not pass")
    check_file(directory / "manifest.json", audit["manifest"]["file_sha256"])
    for player in audit["players"].values(): check_file(player["path"], player["file_sha256"])
    rows = {color: {"effective_eligible": p["transitions"]["effective_eligible"],
                    "effective_bootstrap_allowed": p["transitions"]["effective_bootstrap_allowed"]}
            for color, p in audit["players"].items()}
    row = {"directory": str(directory.relative_to(ROOT)), "manifest": bind(directory / "manifest.json"),
           "interface": manifest["environment"], "status": manifest["status"],
           "termination_reason": manifest["termination_reason"], "outcomes": manifest["outcomes_by_player"],
           "rewards": manifest["rewards_by_player"], "history_audit": bind(audit_path),
           "history_contract_eligible": audit["record_contract_eligible"], "history_transition_counts": rows,
           "training_admission": "excluded_diagnostic", "training_samples": 0}
    if manifest["status"] != "completed":
        require(all(value is None for key in ("outcomes_by_player", "rewards_by_player") for value in manifest[key].values()),
                "Unscored transfer recording has a reward/outcome")
    if manifest["status"] == "failed":
        require(not audit["learner_eligible"] and all(p["effective_eligible"] == p["effective_bootstrap_allowed"] == 0 for p in rows.values()),
                "Failed transfer recording permits learning/bootstrap")
        require(replay_path is None, "Failed capture cannot be certified by replay")
        row["native_replay"] = {"verified": False, "reason": "Intentional crash; all samples excluded"}
    else:
        report = load(replay_path)
        native = report["native"]["summary"]
        require(report["verified"] and report["native"]["verified"] and native["server_verified"]
                and report["histories"]["verified"] and all(value == 0 for value in native["unchecked"].values()),
                "Native replay or actual client/history reconstruction did not verify")
        check_file(directory / "manifest.json", report["inputs"]["manifest"]["sha256"])
        preserve_tree(replay_path.parent)
        row["native_replay"] = {"report": bind(replay_path), "verified": True, "checked": native["checked"],
                                "unchecked": native["unchecked"], "boundary": native["boundary"],
                                "captured_prefix_only": native["captured_prefix_only"]}
    return row


def mechanics():
    specs = [("town-transfer-red-98101", "transfer", "replay/contextual-replay-verification.json"),
             ("town-transfer-blue-98102", "transfer", "replay/contextual-replay-verification.json"),
             ("town-transfer-crash-98103", "crash-after-ack", None),
             ("town-transfer-v3-control-98104", "recruitment-upgrade", "replay/contextual-replay-verification.json")]
    rows = []
    for name, case, replay in specs:
        folder = ROOT / "runs" / name
        report, plan = load(folder / "report.json"), load(folder / "plan.json")
        require(report["probe_status"] == "passed" and report["history_audit_passed"], "Transfer probe failed")
        row = episode(Path(report["manifest"]).parent, folder / case / "history-audit.json", folder / replay if replay else None)
        row.update(report=bind(folder / "report.json"), plan=bind(folder / "plan.json"),
                   checks=report["checks"], evaluator_attempts=report["attempts"], engine_seed=report["engine_seed"],
                   case=case, player=plan.get("player"), probe_source_sha256=plan["probe_sha256"])
        rows.append(row)
    folder = ROOT / "runs/town-transfer-natural-99301"
    for color in ("red", "blue"):
        report = load(folder / f"capture-{color}/report.json")
        require(report["probe_status"] == "passed", "Natural capture diagnostic failed")
        row = episode(Path(report["manifest"]).parent, folder / f"capture-{color}/history-audit.json",
                      folder / f"replay-{color}/contextual-replay-verification.json")
        row.update(report=bind(folder / f"capture-{color}/report.json"), plan=bind(folder / "plan.json"),
                   checks=report["checks"], evaluator_attempts=report["attempts"], engine_seed=report["engine_seed"],
                   case="natural-capture", player=color)
        rows.append(row)
    return rows


def neural(name, *, final):
    folder = ROOT / "runs" / name
    if not (folder / "report.json").exists():
        return [], {"root": str(folder.relative_to(ROOT)), "pending": True}
    report = load(folder / "report.json")
    require(report["training_performed"] is False and report["optimizer_updates"] == 0, "Diagnostic weights were optimized")
    rows = []
    for attempt in report["attempts"]:
        index = attempt["episode"]
        directory = folder / f"episode-{index:06d}"
        replay = folder / f"replay-{index:06d}/contextual-replay-verification.json"
        recompute = folder / f"independent-neural-{index:06d}.json"
        if not replay.exists() or not recompute.exists():
            require(not final, "Frozen neural audit/replay incomplete: " + str(directory))
            continue
        row = episode(directory, directory / "neural-audit.json", replay)
        audit = load(recompute)
        require(audit["optimizer_updates"] == 0 and len(audit["episodes"]) == 1 and audit["episodes"][0]["passed"],
                "Neural recomputation failed")
        check_file(folder / "untrained-policy.pt", audit["checkpoint_sha256"])
        for name, digest in audit["episodes"][0]["files"].items(): check_file(directory / name, digest)
        for player in audit["episodes"][0]["players"]:
            require(player["weights_unchanged"] and player["optimizer_admission_rejected"] and not player["optimization_eligible"],
                    "Diagnostic trajectory admitted or model changed")
            require(all(value == 0 for value in player["maximum_absolute_difference"].values()), "Neural recompute differs")
        row.update(engine_seed=attempt["engine_seed"], learner_player=attempt["learner_player"],
                   policy_seeds=attempt["policy_seeds"], coverage=attempt["coverage"],
                   neural_recompute=bind(recompute), neural_recompute_players=audit["episodes"][0]["players"])
        rows.append(row)
    if final:
        require(len(rows) == 2 and report["source"]["source_sha256"] == source_identity(first_run.INTERFACE)["source_sha256"],
                "Final neural run was not the complete frozen current source snapshot")
    preserve_tree(folder / "source-snapshot")
    snapshot_status = folder / "source-snapshot/snapshot-status.json"
    if snapshot_status.exists():
        snapshot = load(snapshot_status)
        require(snapshot["source_sha256"] == report["source"]["source_sha256"], "Snapshot index differs from collection source manifest")
        for name, digest in snapshot["source_sha256"].items():
            if name not in snapshot["unavailable_exact_files"]:
                check_file(folder / "source-snapshot" / name, digest)
    metadata = {"report": bind(folder / "report.json"), "checkpoint": bind(folder / "untrained-policy.pt"),
                "policy": report["policy"], "opponents": report["opponents"], "source": report["source"],
                "frozen_current_sources": final}
    for name in ("source-snapshot/snapshot-status.json", "sampled-transfer-evidence.json"):
        if (folder / name).exists(): metadata[name] = bind(folder / name)
    if not final:
        metadata["source_scope"] = "Initial diagnostic retained. Dormant gameplay.py changed before exact snapshot capture; source-snapshot/status preserves this limitation. No claim that all unused preparation/execution modules were frozen. Separate final run binds current source hashes."
    return rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--executor-review", type=Path)
    parser.add_argument("--tamper", type=Path, action="append", default=[])
    args = parser.parse_args()
    final = args.output is not None
    require(not final or not args.output.exists(), "Never overwrite finalized evidence")
    mechanical = mechanics()
    initial, initial_meta = neural("town-transfer-neural-99401", final=False)
    frozen, frozen_meta = neural("town-transfer-neural-99401-r2", final=final)
    if not final:
        print(json.dumps({"mechanics": len(mechanical), "initial_neural_verified": len(initial),
                          "frozen_neural_verified": len(frozen), "artifacts": len(ARTIFACTS)}, indent=2))
        return
    require(all((args.tests, args.prepared, args.executor_review)), "Final tests/preparation/executor review are required")
    tests = args.tests.read_text(); match = re.findall(r"Ran (\d+) tests in ([\d.]+)s", tests)
    require(match and re.search(r"^OK\s*$", tests, re.M) and not re.search(r"^FAILED|skipped=", tests, re.M), "Final full suite did not pass without skips")
    require("match exactly" in args.prepared.read_text(), "Missing exact native preparation match")
    source = source_identity(first_run.INTERFACE); implementation = first_run.implementation_identity()
    review = load(args.executor_review)
    require(review.get("status") == "passed" and review.get("execution_performed") is False
            and review.get("source_sha256") == source["source_sha256"]
            and review.get("implementation_sha256") == implementation, "Dormant executor review is not bound to final source")
    check_file(ROOT / "runs/troop-transfer-executor-review-tests.log", review["test_log_sha256"])
    compatibility = load(ROOT / "runs/transfer-old-model-compatibility.json")
    require(compatibility["optimizer_updates"] == 0 and all(e["passed"] for e in compatibility["episodes"]), "Old checkpoint compatibility failed")
    old_checkpoint = ROOT.parents[1] / "headless-review/Heroes/runs/contextual-neural-fixed-95011-95012/untrained-policy.pt"
    check_file(old_checkpoint, compatibility["checkpoint_sha256"])
    for item in compatibility["episodes"]:
        for path, digest in item["files"].items(): check_file(Path(item["episode"]) / path, digest)
        require(all(all(value == 0 for value in p["maximum_absolute_difference"].values()) for p in item["players"]), "Old model behavior differs")
    tamper = []
    for path in args.tamper:
        report = load(path)
        require(report["passed"] is True and report["positive_control"]["verified"] is True
                and report["cases"] and all(case["detected"] is True and case["native"]["verified"] is False for case in report["cases"]),
                "Tamper negative control or positive control failed")
        positive = path.parent / report["positive_control"]["path"]
        check_file(positive, report["positive_control"]["identity"]["sha256"])
        require(load(positive)["verified"] is True, "Tamper positive control does not verify")
        tamper.append({"report": bind(path), "positive_control": bind(positive), "passed": True,
                       "cases": [{"name": case["name"], "detected": case["detected"],
                                  "input": case["input"], "native_returncode": case["native"]["returncode"],
                                  "native_verified": case["native"]["verified"]} for case in report["cases"]]})
        preserve_tree(path.parent)
    launcher = VcmiLauncher(ROOT / "vcmi/build/bin/vcmiclient", Path.home() / "Library/Application Support/vcmi",
                            ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    launcher.preflight()
    runtime = runtime_identity(launcher.binary.parent)
    require(runtime == COMMON["native_runtime_sha256"], "Native runtime changed from validated captures")
    for name, digest in COMMON["engine_recipe"].items(): check_file(ROOT / name, digest)
    for name, digest in COMMON["contextual_python_sha256"].items(): check_file(ROOT / name, digest)
    def passed(*paths): return {"status": "passed", "artifacts": [artifact(p) for p in paths]}
    checks = {
        "full_tests": passed(args.tests), "prepared": passed(args.prepared),
        "transfer_red": passed(ROOT / mechanical[0]["report"], ROOT / mechanical[0]["native_replay"]["report"]),
        "transfer_blue": passed(ROOT / mechanical[1]["report"], ROOT / mechanical[1]["native_replay"]["report"]),
        "natural_red": passed(ROOT / mechanical[4]["report"], ROOT / mechanical[4]["native_replay"]["report"]),
        "natural_blue": passed(ROOT / mechanical[5]["report"], ROOT / mechanical[5]["native_replay"]["report"]),
        "failure_after_ack": passed(ROOT / mechanical[2]["report"], ROOT / mechanical[2]["history_audit"]),
        "v3_checkpoint_compatibility": passed(ROOT / "runs/transfer-old-model-compatibility.json", ROOT / mechanical[3]["native_replay"]["report"]),
        "dormant_executor_review": passed(args.executor_review),
    }
    remaining = ["Invalid transfer choices were rejected before send; no actual negative native acknowledgement is claimed.",
                 "Garrison-hero source and minimum last-creature/int32 overflow branches have source/native-unit coverage, not live current-baseline cases.",
                 "Reverse transfer, swapping unlike creatures, other army editing, hero hiring and special windows remain unsupported.",
                 "Initial/timer-origin natural completion remains excluded; only request-origin completion is supported.",
                 "Random cutoffs provide no observed victory signal; these diagnostics establish neither strategic improvement nor an outcome frequency.",
                 "Execution requires future human authorization of the exact plan; no optimizer has been run on gameplay."]
    readiness = {"interface": first_run.INTERFACE, "source_sha256": source["source_sha256"],
                 "implementation_sha256": implementation, "native_identity": launcher.identity(), "native_runtime": runtime,
                 "map_identity": ordinary_map_identity(launcher.game_map), "checks": checks, "remaining_scope": remaining}
    rows = mechanical + initial + frozen
    checked = Counter()
    for row in rows:
        if row["native_replay"]["verified"]: checked.update(row["native_replay"]["checked"])
    result = {"schema": "heroes-town-transfer-validation-v1", "date": "2026-10-25", "assembled_at": datetime.now(timezone.utc).isoformat(),
              "interface": first_run.INTERFACE, "training_performed": False, "optimizer_updates": 0, "paid_compute": False,
              "authorization": "preparation_only", "common_native_adapter_identity": COMMON,
              "totals": {"native_attempts": len(rows), "mechanics_attempts": len(mechanical), "neural_attempts": len(initial) + len(frozen),
                         "statuses": dict(Counter(r["status"] for r in rows)),
                         "native_replays_verified": sum(r["native_replay"]["verified"] for r in rows),
                         "native_replay_checked_records": dict(checked), "native_replay_unchecked_records": 0,
                         "tests_passed": int(match[-1][0]), "tests_skipped": 0, "training_samples": 0},
              "mechanics": mechanical, "neural": {"initial": {"attempts": initial, **initial_meta}, "frozen_rerun": {"attempts": frozen, **frozen_meta}},
              "v3_compatibility": {"report": bind(ROOT / "runs/transfer-old-model-compatibility.json"), "bit_exact": True,
                                   "native_control": mechanical[3]["directory"]},
              "tamper_checks": tamper, "first_training_readiness": readiness,
              "source": source, "remaining_scope": remaining,
              "preparation_only": {"configuration": bind(first_run.CONFIG), "optimizer_executed": False,
                                   "plan_status": "Generate only after final evidence/source freeze; execution remains unauthorized"},
              "assembler": bind(Path(__file__))}
    directories = {}
    for name, value in ARTIFACTS.items(): directories.setdefault(str(Path(name).parent), {})[Path(name).name] = value
    result["artifact_directories"] = directories
    result["artifact_directory_semantics"] = "Join directory and filename. Relative paths are rooted here; absolute directories identify preserved external prior-version evidence. SHA-256/bytes describe complete file bytes."
    raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    require(len(raw) < 250000, "Evidence exceeds compact size budget")
    with args.output.open("xb") as stream: stream.write(raw)
    # Reuse the exact preparation checker; this creates no checkpoint or launch.
    verified = first_run.check_readiness(args.output, source=source, implementation=implementation,
                                         native_identity=launcher.identity(), native_runtime=runtime,
                                         map_identity=ordinary_map_identity(launcher.game_map))
    require(verified["ready_to_request_launch"], "Assembled readiness did not validate")
    print(json.dumps({"output": str(args.output), "bytes": len(raw), "artifacts": len(ARTIFACTS), "totals": result["totals"]}, indent=2))


if __name__ == "__main__": main()
