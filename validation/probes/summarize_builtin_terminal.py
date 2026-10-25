"""Bind stopped ordinary cohorts and separate terminal diagnostics offline.

No game, model, replay, build or optimizer is executed. Historical executable
and source identities are retained; they are not compared with today's build.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.headless import builtin_evidence as evidence
from experiments.headless import builtin_replay as replay
from experiments.learner import contextual as c
from validation.probes.summarize_builtin_preflight import binding, compact_identity, read, require

SCHEMA = "heroes-builtin-terminal-evidence-v1"
NATURAL_SCHEMA = "heroes-builtin-natural-diagnostic-v1"
COLORS = ("red", "blue")


def mapping_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bind(path, expected=None):
    actual = binding(path)
    expected = {"sha256": expected} if isinstance(expected, str) else expected or {}
    require(all(evidence.same(actual[key], expected[key]) for key in ("path", "sha256", "bytes") if key in expected),
            "Bound artifact changed: " + str(path))
    return actual


def no_training(document, *, counters=("training_samples", "optimizer_steps")):
    require(document.get("training_performed", False) is False and all(
        type(document.get(key)) is int and document[key] == 0 for key in counters), "Diagnostic reports training")


def compact_accounting(accounting):
    """Keep identities and positions, but never commit serialized native packets."""
    return {**accounting, "terminal_unapplied": [
        {**{key:value for key,value in attempt.items() if key != "pack"},
         "pack_sha256":hashlib.sha256(bytes.fromhex(attempt["pack"])).hexdigest()}
        for attempt in accounting["terminal_unapplied"]]}


def check_supplemental_flags(report):
    require(report.get("schema") == replay.DIAGNOSTIC_SCHEMA and report.get("verified") is False
            and type(report.get("diagnostic_verified")) is bool and report.get("original_cohort_admission") is False
            and report.get("original_cohort_result_unchanged") is True and report.get("optimization_eligible") is False
            and all(report.get(key) == dict.fromkeys(COLORS) for key in ("outcomes_by_player", "rewards_by_player")),
            "Supplemental replay must not admit or rescore the original cohort")
    no_training(report, counters=("training_samples",))


def check_compatibility(record, identity):
    require(record.get("mode") == "diagnostic_verifier_only" and all(record.get(key) is True for key in
            ("nonverifier_runtime_unchanged", "client_unchanged", "adapter_sources_unchanged", "nonverifier_recipe_unchanged")),
            "Supplemental verifier lacks its unchanged nonverifier identity checks")
    for changes, captured, allowed in ((record["runtime_changes"], identity["native_runtime_sha256"], lambda n:n == "vcmireplay"),
            (record["recipe_changes"], identity["engine_recipe"], lambda n:n.startswith("engine/ExperimentReplay/") and ".." not in Path(n).parts)):
        require(type(changes) is dict, "Invalid verifier compatibility mapping")
        for name, value in changes.items():
            require(allowed(name) and type(value) is dict and set(value) == {"captured", "current"}
                    and value["captured"] == captured.get(name), "Supplemental compatibility changed gameplay identity")
            require(value["current"] is None or type(value["current"]) is str and len(value["current"]) == 64
                    and all(c in "0123456789abcdef" for c in value["current"]), "Invalid current verifier digest")


def stopped_schedule(plan, report, compact):
    require(report.get("status") == "stopped" and len(plan["schedule"]) == len(report["attempts"]) == len(compact["attempts"]) == 8,
            "Both original stopped eight-row schedules must survive")
    no_training(report, counters=("training_samples", "optimizer_steps", "optimizer_updates"))
    require(evidence.same(report["frozen_before"], plan["frozen"])
            and report["frozen_after_file_sha256"] == plan["checkpoint"]["sha256"], "Frozen cohort checkpoint changed")
    rows = []
    for expected, row, saved in zip(plan["schedule"], report["attempts"], compact["attempts"]):
        require(all(evidence.same(row.get(key), value) for key,value in expected.items()), "Original schedule changed")
        no_training(row)
        require(row.get("status") in ("passed", "failed", "not_started"), "Unsupported stopped attempt status")
        if row["status"] != "passed":
            require(all(row.get(key) == dict.fromkeys(COLORS) for key in ("outcomes", "rewards")), "Failed/unstarted attempt was scored")
        else:
            require(row.get("replay_verified") is True and row.get("weights_unchanged") is True, "Passed row lacks verification")
        keys = set(expected) | {"status", "outcomes", "rewards"}
        require(all(evidence.same(row.get(key), saved.get(key)) for key in keys), "Original compact cohort differs from its execution")
        folder = Path(plan["execution_output"]) / f"episode-{row['episode']:06d}"
        require(row["status"] != "not_started" or not folder.exists(), "Unstarted row acquired execution artifacts")
        rows.append({**{key:row[key] for key in keys}, **{key:row[key] for key in
                    ("error", "sampled_choices", "latest_sampled_day", "termination_reason") if key in row}})
    return {"status":"stopped", "counts":dict(Counter(row["status"] for row in rows)), "attempts":rows,
            "original_results_unchanged":True, "checkpoint":plan["checkpoint"], "frozen":plan["frozen"],
            "historical_source": {"git_commit":plan.get("source", {}).get("git_commit"),
                "mapping_sha256":mapping_digest(plan.get("source", {}))}, "historical_runtime":plan.get("runtime", {})}


def checked_replay(path, folder, manifest, *, supplemental):
    report = read(path)
    bind(folder / "manifest.json", report["inputs"]["manifest"])
    final = evidence.read_finalization(folder, manifest)
    for name, identity in final["native_capture"]["files"].items(): bind(evidence.native.inside(folder, name), identity)
    color = COLORS[manifest["learner_player"]]
    bind(folder / f"agents/{color}/bridge.jsonl", final["raw_bridge_sha256"][color])
    bind(folder / "logs/process.log", final["process_log_sha256"])
    for label in ("history_v7", "live_evidence"):
        item = manifest["players"][color][label]; bind(evidence.native.inside(folder,item["path"]),item["sha256"])
    passed = report.get("diagnostic_verified") is True if supplemental else report.get("verified") is True
    result = {"report":bind(path), "passed":passed, "error":report.get("error"),
              "historical_identity":compact_identity(manifest["identity"]),
              "recorded_verifier_sources":report["inputs"].get("verifier_sources", {})}
    if not passed: return result
    require(report.get("opponent_policy_reexecuted") is False and report["history"]["verified"] is True
            and report["history"]["opponent_history"] is None, "Replay invented a native-opponent history")
    replay.validate_native_report(report["native"], report["wire"], final, cutoff=False)
    native_path = Path(path).parent / "native-report.json"
    require(evidence.same(read(native_path), {k:v for k,v in report["native"].items() if k not in ("returncode", "seconds")}),
            "Native report differs from the verified snapshot")
    bind(folder / "engine/experiment-metadata.json", report["effective_engine"]["metadata_file"])
    copies = {}
    for name in ("replay.jsonl", "contextual-replay.jsonl"):
        source = folder / "engine" / name
        copy = Path(path).parent / name
        copies[name] = {"original":bind(source),"replay_copy":bind(copy)}
        raw, copied = source.read_bytes(), copy.read_bytes()
        require(raw.endswith(b"\n") and copied.endswith(b"\n") and evidence.same(
            [evidence.native.strict_json(line) for line in raw.splitlines()],
            [evidence.native.strict_json(line) for line in copied.splitlines()]), "Replay copy changed complete recorded rows")
    require(report["prefix"] == {"server_partial_bytes":0,"server_unpaired_inputs":0,"contextual_partial_bytes":0},
            "Natural replay lacks a complete capture")
    result.update(native_report=bind(native_path), replay_copies=copies, counts=report["native"]["summary"]["checked"],
        request_accounting=compact_accounting(report["wire"]["request_accounting"]),
        history=report["history"], terminal_outcomes=manifest["outcomes_by_player"],
        terminal_outcome_scope="Separate replay evidence; does not change an original ordinary-cohort row.")
    return result


def natural_history(folder, manifest, row):
    require(manifest.get("status") == "completed" and manifest.get("termination_reason") == "natural_terminal"
            and type(manifest.get("returncode")) is int and manifest["returncode"] == 0
            and manifest.get("episode") == str(row["episode"])
            and type(manifest.get("learner_player")) is int and manifest["learner_player"] == row["learner_player"]
            and type(manifest.get("seed")) is int and manifest["seed"] == row["engine_seed"],
            "Passed fixture does not bind its scheduled natural episode/seat/seed")
    player=manifest["learner_player"]; declaration=manifest["players"][COLORS[player]]["history_v7"]
    path=evidence.native.inside(folder,declaration["path"]);bind(path,declaration["sha256"])
    history=c.PlayerHistory(manifest["episode"],player,schema=c.DIALOGS_SCHEMA)
    for line in path.read_bytes().splitlines():history.append(c.decode_frame(evidence.native.strict_json(line)))
    audit=evidence.validate_manifest_history(history,manifest)
    final=evidence.read_finalization(folder,manifest)
    barrier=evidence.terminal_barrier(folder,episode=manifest["episode"],interface=manifest["environment"],
        outcomes=[manifest["outcomes_by_player"][color] for color in COLORS],learner_player=player)
    require(evidence.same(barrier,final.get("terminal_barrier")),"Natural fixture finalization changed terminal receipts")
    evidence.native.terminal_capture_close(folder,barrier)
    return history,audit


def reconstruction_binding(folder, manifest, row, frozen, history, audit):
    """Validate saved reconstruction claims against actual own records, without resampling."""
    path=folder/"policy-reconstruction.json";record=read(path);rollout_path=folder/"builtin-rollout.json";rollout=read(rollout_path)
    require(evidence.same(record,row["reconstruction"]) and record.get("passed") is True
            and record.get("weights_unchanged") is True and record.get("private_memory_reset") is True
            and type(record.get("maximum_absolute_difference")) in (int,float) and record["maximum_absolute_difference"] == 0
            and type(record.get("player")) is int and record["player"] == history.player
            and evidence.same(record.get("history_audit"),audit),"Fixture reconstruction is not exact for its actual own history")
    require(rollout.get("schema") == "heroes-builtin-evaluation-rollout-v1" and rollout.get("purpose") == "diagnostic"
            and rollout.get("origin") == "native" and rollout.get("optimization_eligible") is False
            and type(rollout.get("training_samples")) is int and rollout["training_samples"] == 0
            and evidence.same(rollout.get("manifest"),manifest) and type(rollout.get("player")) is int
            and rollout["player"] == history.player and rollout.get("policy_seed") == row["policy_seed"]
            and rollout.get("weights_sha256") == frozen["learner_weights_sha256"]
            and evidence.same(rollout.get("opponent_identity"),manifest["opponent_identity"])
            and evidence.same(rollout.get("history"),[c.encode_frame(f) for f in history.frames]),
            "Saved rollout changed frozen policy, episode, player or delivered history")
    decisions,estimates=rollout.get("decisions"),rollout.get("estimates")
    require(type(decisions) is list and type(estimates) is list
            and [x["decision_seq"] for x in decisions] == [f.decision_seq for f in history.frames if f.kind == "submitted"]
            and [x["decision_seq"] for x in estimates] == [f.seq for f in history.frames if f.kind == "decision"],
            "Reconstruction omitted or invented own decisions/values")
    digest=hashlib.sha256();prefixes={}
    for frame in history.frames:
        digest.update((json.dumps(c.encode_frame(frame),sort_keys=True,separators=(",", ":"))+"\n").encode())
        if frame.kind == "decision":prefixes[frame.seq]=digest.hexdigest()
    values={x["decision_seq"]:x["value"] for x in estimates}
    submitted={f.decision_seq:c.encode_action(f.action) for f in history.frames if f.kind == "submitted"}
    for item in [*decisions,*estimates]:
        seq=item["decision_seq"]
        require(type(seq) is int and evidence.same(item.get("prefix"),{
            "episode":history.episode,"player":history.player,"checkpoint":rollout["checkpoint"],
            "weights_sha256":frozen["learner_weights_sha256"],"through_seq":seq,"history_sha256":prefixes[seq]}),
            "Reconstruction record is bound to another recurrent prefix")
    for item in decisions:
        require(evidence.same(item["action"],submitted[item["decision_seq"]]) and item["value"] == values[item["decision_seq"]]
                and type(item.get("conditionals")) is list and bool(item["conditionals"]),"Saved action/value evidence differs from delivered experience")
    wanted={"actions_resampled":len(decisions),"values_recomputed":len(estimates),
            "conditionals_recomputed":sum(len(item["conditionals"]) for item in decisions)}
    require(all(type(record.get(key)) is int and record[key] == value for key,value in wanted.items())
            and type(row.get("sampled_choices")) is int and row["sampled_choices"] == len(decisions)
            and row.get("latest_sampled_day") == max((history.frames[x["decision_seq"]].day for x in decisions),default=None),
            "Reconstruction counts differ from saved decisions/values/conditionals")
    return {"report":bind(path),"rollout":bind(rollout_path),**wanted,"maximum_absolute_difference":0,"history_audit":audit}


def natural_diagnostic(directory, checkpoint, frozen):
    directory = Path(directory).resolve(); plan, report = read(directory/"plan.json"), read(directory/"report.json")
    require(plan.get("schema") == NATURAL_SCHEMA and report.get("schema") == NATURAL_SCHEMA+"-result"
            and report.get("status") in ("passed", "failed") and plan.get("ordinary_cohort_member") is False
            and report.get("ordinary_cohort_member") is False, "Natural diagnostic is not a separate completed fixture run")
    bind(directory/"plan.json", report["plan"]); no_training(plan); no_training(report)
    require(evidence.same(plan["checkpoint"],checkpoint) and evidence.same(plan["frozen"],frozen)
            and evidence.same(report["checkpoint_after"],checkpoint), "Natural diagnostic changed the frozen checkpoint")
    bind(checkpoint["path"],checkpoint)
    require(plan["fixture"].get("ordinary_cohort_member") is False and plan["fixture"].get("natural_outcome_validation") is True,
            "Natural diagnostic fixture scope changed")
    for key in ("map","manifest"):bind(plan["fixture"][key]["path"],plan["fixture"][key])
    require(len(plan["schedule"]) == len(report["attempts"]) == 2, "Both fixture attempts must survive")
    rows=[]
    for expected,row in zip(plan["schedule"],report["attempts"]):
        require(all(evidence.same(row.get(k),v) for k,v in expected.items())
                and row.get("status") in ("passed","failed","not_started"), "Fixture schedule changed")
        no_training(row, counters=tuple(k for k in ("training_samples","optimizer_steps") if k in row))
        result={**expected, **{k:row[k] for k in ("status","error","outcomes","rewards","sampled_choices","latest_sampled_day") if k in row}}
        folder=directory/f"episode-{row['episode']:06d}"
        if row["status"] != "passed":
            require(all(row.get(key) in (None,dict.fromkeys(COLORS)) for key in ("outcomes","rewards")),"Failed fixture was scored")
            require(row["status"] != "not_started" or not folder.exists(),"Unstarted fixture has artifacts")
            result.update(outcomes=dict.fromkeys(COLORS),rewards=dict.fromkeys(COLORS))
            if row.get("manifest"):bind(folder/"manifest.json",row["manifest"])
        else:
            manifest=read(folder/"manifest.json");bind(folder/"manifest.json",row["manifest"])
            history,audit=natural_history(folder,manifest,row)
            result["terminal_day"]=next(frame.day for frame in reversed(history.frames) if frame.kind == "terminal")
            require(row.get("replay_verified") is True and row.get("weights_unchanged") is True
                    and row["outcomes"] == manifest["outcomes_by_player"] and row["rewards"] == manifest["rewards_by_player"],
                    "Passed fixture lacks frozen verified outcomes")
            path=folder/"builtin-verification/builtin-replay-verification.json";bind(path,row["replay"])
            result["verification"]=checked_replay(path,folder,manifest,supplemental=False)
            require(result["verification"]["passed"],"Passed fixture lacks native replay")
            result["reconstruction"]=reconstruction_binding(folder,manifest,row,frozen,history,audit)
        rows.append(result)
    require((report["status"] == "passed") == all(row["status"] == "passed" for row in rows),"Fixture overall status changed")
    return {"plan":bind(directory/"plan.json"),"report":bind(directory/"report.json"),"fixture":plan["fixture"],
            "status":report["status"],"attempts":rows,"ordinary_cohort_member":False}


def summarize(cohort, original_summary, supplemental, *, natural=None, validation_logs=()):
    cohort, original_summary, supplemental = (Path(p).resolve() for p in (cohort,original_summary,supplemental))
    saved=read(original_summary);plan_path=cohort.parent/"plan.json";plan=read(plan_path);report=read(cohort/"report.json")
    require(saved.get("schema") == "heroes-builtin-opponent-comparison-evidence-v1" and evidence.same(saved["plan"],plan)
            and Path(plan["execution_output"]).resolve() == cohort,"Original comparison identity changed")
    bind(plan_path,report["plan_sha256"])
    reference=plan["reference"];other_plan=read(reference["plan"]["path"]);other_report=read(reference["report"]["path"])
    bind(reference["plan"]["path"],reference["plan"]);bind(reference["report"]["path"],reference["report"])
    bind(reference["plan"]["path"],other_report["plan_sha256"])
    checkpoint=plan["checkpoint"];bind(checkpoint["path"],checkpoint)
    require(evidence.same(checkpoint,other_plan["checkpoint"]),"Stopped cohorts used different checkpoints")
    executions=[cohort,Path(other_plan["execution_output"]).resolve()]
    retained={name:identity for name,identity in saved["bindings"].items()
              if any(Path(name).resolve().is_relative_to(directory) for directory in executions)}
    for name,identity in retained.items():bind(name,identity)
    require(all(str(directory/"report.json") in retained for directory in executions),"Original report bindings missing")
    cohorts={label:stopped_schedule(p,r,saved["cohorts"][label]) for label,p,r in (
        ("builtin_nullkiller_easy",plan,report),("frozen_neural_opponent",other_plan,other_report))}
    diagnostic=read(supplemental);check_supplemental_flags(diagnostic)
    episode=cohort/f"episode-{int(diagnostic['episode']):06d}"
    row=next(row for row in report["attempts"] if str(row["episode"]) == str(diagnostic["episode"]))
    require(row["status"] == "failed","Supplemental result is not the preserved failed original attempt")
    manifest=read(episode/"manifest.json")
    preserved=replay.diagnostic_inputs(episode,manifest)
    require(evidence.same(preserved,diagnostic["inputs"]["preserved_original_files"]),"Supplemental original-file bindings changed")
    if diagnostic["diagnostic_verified"]:check_compatibility(diagnostic["identity_check"],manifest["identity"])
    supplemental_result=checked_replay(supplemental,episode,manifest,supplemental=True)
    supplemental_result.update(original_cohort_admission=False,compatibility=diagnostic.get("identity_check"),ordinary_cohort_member=False)
    return {"schema":SCHEMA,"offline_validator":bind(Path(__file__)),"original_summary":bind(original_summary),"original_plan":bind(plan_path),
        "original_reports":[bind(directory/"report.json") for directory in executions],"cohorts":cohorts,
        "retained_original_artifacts":{"count":len(retained),"mapping_sha256":mapping_digest(retained)},
        "supplemental_replay":supplemental_result,
        "natural_fixture":natural_diagnostic(natural,checkpoint,plan["frozen"]) if natural else None,
        "validation_logs":[bind(path) for path in validation_logs],"frozen_checkpoint":bind(checkpoint["path"],checkpoint),
        "training_performed":False,"training_samples":0,"optimizer_steps":0,"optimization_eligible":False,
        "limitations":["Offline file/structure checks only; no game, model, build, test or replay is executed.",
            "Both stopped ordinary schedules remain unchanged; supplemental or fixture success cannot replace their failed or unstarted rows.",
            "The terminal tail is a captured unapplied attempt, not proof of server receipt or rejection.",
            "Historical identities remain attached to each capture. Validation logs are bound, not interpreted as successful checks."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("cohort","original-summary","supplemental","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--natural",type=Path);parser.add_argument("--validation-log",type=Path,action="append",default=[])
    args=parser.parse_args();require(not args.output.exists(),"Preserve earlier terminal summaries")
    result=summarize(args.cohort,args.original_summary,args.supplemental,natural=args.natural,validation_logs=args.validation_log)
    with args.output.open("x") as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({"output":binding(args.output),"supplemental_verified":result["supplemental_replay"]["passed"]}))


if __name__ == "__main__":main()
