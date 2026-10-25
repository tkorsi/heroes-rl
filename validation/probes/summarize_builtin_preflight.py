"""Bind retained zero-action preflights offline, without running native validation.

Historical runtime/source identities stay attached to their own cohort. A later
repair does not replace an earlier failure or establish a natural game outcome.
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

SCHEMA = "heroes-builtin-preflight-summary-v1"
PROBE_SCHEMA = "heroes-builtin-initial-cutoff-diagnostic-v1"
COLORS = ("red", "blue")


def require(value, message):
    if not value:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def read(path):
    return evidence.native.strict_json(Path(path).read_bytes())


def compact_identity(identity):
    recipe = identity.get("engine_recipe", {})
    return {**{key:identity[key] for key in ("project", "engine_revision", "captured_before_launch",
        "binary_sha256", "explorer_rl_sha256", "map_sha256", "bridge_source_sha256",
        "native_runtime_sha256", "contextual_python_sha256") if key in identity},
        "recorded_recipe_entries":len(recipe), "recorded_recipe_mapping_sha256":hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def summarize(cohorts, *, validation_logs=()):
    bindings, summaries, totals = {}, [], Counter()

    def bind(path, expected=None):
        actual = binding(path)
        expected = {"sha256": expected} if isinstance(expected, str) else expected or {}
        require(all(evidence.same(actual[key], expected[key]) for key in ("path", "sha256", "bytes") if key in expected),
                "Bound file differs: " + str(path))
        bindings[actual["path"]] = {key: actual[key] for key in ("sha256", "bytes")}
        return actual

    directories = [Path(path).resolve() for path in cohorts]
    require(directories and len(set(directories)) == len(directories), "Declare distinct preflight cohorts")
    for directory in directories:
        plan_path, report_path = directory / "plan.json", directory / "report.json"
        plan, report = read(plan_path), read(report_path)
        bind(plan_path, report.get("plan")); bind(report_path)
        require(report.get("plan") == binding(plan_path) and plan.get("output") == str(directory)
                and plan.get("schema") == PROBE_SCHEMA and report.get("schema") == PROBE_SCHEMA + "-result",
                "Preflight report/plan identity differs")
        require(report.get("status") in ("passed", "failed") and report.get("ordinary_cohort_member") is False,
                "Preflight is still running or labelled an ordinary cohort")
        for document in (plan, report):
            require(document.get("training_performed") is False and type(document.get("training_samples")) is int
                    and document["training_samples"] == 0, "Preflight reports training")
        require(type(plan.get("policy_actions")) is int and plan["policy_actions"] == 0
                and type(report.get("optimizer_steps")) is int and report["optimizer_steps"] == 0,
                "Preflight is not a zero-action diagnostic")
        fixture = plan["fixture"]
        require(fixture.get("ordinary_cohort_member") is False and fixture.get("natural_outcome_validation") is False,
                "Fixture has unsupported outcome/evaluation claims")
        for item in (fixture["map"], fixture["manifest"]): bind(item["path"], item)
        expected_rows = [{"episode":seat+1, "learner_player":seat, "engine_seed":121901} for seat in (0,1)]
        require(evidence.same(plan.get("schedule"), expected_rows) and len(report.get("attempts", [])) == 2,
                "Both original fixed preflight rows must survive")
        rows = []
        for row, expected in zip(report["attempts"], expected_rows):
            require(all(evidence.same(row.get(key), value) for key, value in expected.items())
                    and row.get("status") in ("passed", "failed", "not_started"), "Preflight schedule/status changed")
            for key in ("policy_actions", "training_samples", "optimizer_steps"):
                require(key not in row or type(row[key]) is int and row[key] == 0, "Preflight contains learner actions/training")
            require(row.get("training_performed", False) is False and all(row.get(key) in (None, dict.fromkeys(COLORS))
                    for key in ("outcomes", "rewards")), "Preflight result must remain unscored")
            item = {**row, "outcomes":dict.fromkeys(COLORS), "rewards":dict.fromkeys(COLORS),
                    "observed_policy_actions":None, "evidence_checked":False, "historical_identity":None}
            folder = directory / f"episode-{row['episode']:06d}"
            if row.get("episode_directory") is not None:
                require(Path(row["episode_directory"]).resolve() == folder, "Episode directory differs from schedule")
            worker_path = directory / f"case-{row['episode']}.json"
            if worker_path.exists(): bind(worker_path)
            if row["status"] == "not_started":
                require(not folder.exists() and not worker_path.exists(), "Not-started row has execution artifacts")
            elif row.get("manifest") is not None:
                manifest_path = folder / "manifest.json"
                bind(manifest_path, row["manifest"])
                manifest = read(manifest_path)
                require(manifest["episode"] == str(row["episode"]) and manifest["learner_player"] == row["learner_player"]
                        and manifest["seed"] == row["engine_seed"], "Manifest changed preflight identity")
                final = evidence.read_finalization(folder, manifest)
                bind(folder / "builtin-finalization.json", manifest["evidence_ref"].split("#sha256=")[1])
                color = COLORS[row["learner_player"]]
                declaration = manifest["players"][color]
                history = c.PlayerHistory(manifest["episode"], row["learner_player"], schema=c.DIALOGS_SCHEMA)
                history_path = evidence.native.inside(folder, declaration["history_v7"]["path"])
                bind(history_path, declaration["history_v7"]["sha256"])
                for line in history_path.read_bytes().splitlines(): history.append(c.decode_frame(evidence.native.strict_json(line)))
                evidence.validate_manifest_history(history, manifest)
                require(not any(frame.kind in ("submitted", "consumed", "issued", "acknowledged", "result") for frame in history.frames)
                        and all(type(declaration.get(key)) is int and declaration[key] == 0 for key in
                                ("actions_sent", "actions_accepted", "actions_rejected", "responses_undelivered")),
                        "Zero-action preflight contains actual learner actions")
                bridge = folder / "agents" / color / "bridge.jsonl"
                bind(bridge, final["raw_bridge_sha256"][color])
                require(all(evidence.native.strict_json(line)["direction"] == "engine" for line in bridge.read_bytes().splitlines()),
                        "A policy reply entered the diagnostic bridge")
                bind(folder / "logs/process.log", final["process_log_sha256"])
                live = declaration["live_evidence"]
                bind(evidence.native.inside(folder, live["path"]), live["sha256"])
                for name, identity in final["native_capture"]["files"].items(): bind(evidence.native.inside(folder, name), identity)
                identity = manifest["identity"]
                require(identity["binary_sha256"] == plan["executables"]["binary"]["sha256"]
                        and identity["native_runtime_sha256"]["vcmireplay"] == plan["executables"]["replay_binary"]["sha256"]
                        and identity["map_sha256"] == fixture["map"]["sha256"], "Captured native identity differs from its own plan")
                item.update(observed_policy_actions=0, evidence_checked=True, historical_identity=compact_identity(identity),
                            observed_episode_status=manifest["status"], termination_reason=manifest["termination_reason"])
                if row["status"] == "passed":
                    require(manifest["status"] == "truncated" and manifest["termination_reason"] == "collection_window"
                            and row.get("replay_verified") is True and history.frames[-1].kind == "cutoff"
                            and len([frame for frame in history.frames if frame.kind == "decision"]) == 1,
                            "Passed preflight is not its initial safe decision cutoff")
                    decision = next(frame for frame in history.frames if frame.kind == "decision")
                    require(all(evidence.same(row.get(key), value) for key,value in {
                        "held_decision_seq":decision.seq, "held_decision_day":decision.day,
                        "cutoff_seq":history.frames[-1].seq, "own_history_frames":len(history.frames),
                        "visible_events_retained":sum(frame.kind == "event" for frame in history.frames)}.items()),
                        "Report changed the actual held-cutoff position")
                    replay_path = folder / "builtin-verification/builtin-replay-verification.json"
                    bind(replay_path, row["replay"])
                    checked = read(replay_path)
                    bind(manifest_path, checked["inputs"]["manifest"])
                    require(checked.get("schema") == replay.REPORT_SCHEMA and checked.get("episode") == manifest["episode"]
                            and checked.get("verified") is True and checked.get("opponent_policy_reexecuted") is False
                            and checked["wire"]["responses"] == 0 and checked["history"]["verified"] is True,
                            "Passed preflight lacks its zero-action replay result")
                    replay.validate_native_report(checked["native"], checked["wire"], final, cutoff=True)
                    native_path = replay_path.parent / "native-report.json"
                    bind(native_path)
                    require(evidence.same(read(native_path), {key:value for key,value in checked["native"].items()
                            if key not in ("returncode", "seconds")}), "Saved native report differs from replay summary")
                    bind(folder / "engine/experiment-metadata.json", checked["effective_engine"]["metadata_file"])
                    require(evidence.same(row.get("effective_engine"), checked["effective_engine"])
                            and evidence.same(row.get("native_opponent_requests"), checked["native_opponent_requests"]),
                            "Report changed verified native settings/counts")
                    item["recorded_verifier_sources"] = checked["inputs"]["verifier_sources"]
            else:
                require(row["status"] != "passed", "Passed preflight has no manifest binding")
            totals[row["status"]] += 1
            rows.append(item)
        require((report["status"] == "passed") == all(row["status"] == "passed" for row in rows), "Overall preflight status changed")
        summaries.append({"directory":str(directory), "plan":binding(plan_path), "report":binding(report_path),
            "status":report["status"], "historical_probe_source":plan["source"],
            "historical_executables":plan["executables"], "fixture":fixture, "attempts":rows})
    logs = [bind(path) for path in validation_logs]
    return {"schema":SCHEMA, "cohorts":summaries, "counts":dict(totals), "bindings":bindings,
        "validation_logs":logs, "policy_actions":0, "training_performed":False, "training_samples":0,
        "optimizer_steps":0, "ordinary_cohort_member":False, "natural_outcome_validation":False,
        "limitations":["Offline binding check only; no game, policy, build, test or replay is executed.",
            "Historical source/runtime hashes are preserved per cohort, not compared with today's files.",
            "A passed row verifies only the recorded initial cutoff; it is not a natural outcome or ordinary-game result.",
            "Absent failed-attempt action evidence remains null. Supplied build/test logs are bound, not interpreted as successful checks."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=Path, required=True)
    parser.add_argument("--validation-log", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve an earlier preflight summary")
    result = summarize(args.cohort, validation_logs=args.validation_log)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":binding(args.output), "counts":result["counts"]}))


if __name__ == "__main__": main()
