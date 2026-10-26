"""Assemble the path-free acceleration evidence record from the outputs of the scripts in this directory.

Usage: record.py NOTES.json RUNS_ROOT OUT
NOTES (record-notes-2026-10-26.json is the one behind the committed record) holds the commits, the platform and
host descriptions, the summary of the independent review, and where each input lies under RUNS_ROOT:
  embedded_ab, cold_ab, embedded_ab_host_a   {"host", "dir"[, "note"]}  ab.sh output directories
  repeated_reset_soak                        {"host", "dir"}            soak.sh outputs (NAME/, NAME.perfstat)
  scaling, scaling_host_a                    {"host", "file"[, "code", "note"]}
  controller_replay, controller_replay_v8,
  controller_replay_two_passes,
  audit_two_passes, audit_two_passes_v8      {"host", "file"}
  review_fix_costs                           {"host", "file"}           review_fix_costs.py output
  capture_verification                       {"host", "file"}           capture_counts.py output
measure.sh writes the line files: "CONFIG k=K " + a scale.sh line for scaling, and "CONFIG " + one JSON line of
controller_replay.py or audit_passes.py for the others, CONFIG being base or accel.
"""
import json
import re
import statistics
import sys
from pathlib import Path

WARMUP = "1"  # round 1 of every A/B is listed but excluded from medians


def load(path):
    return json.loads(Path(path).read_text())


def run_brief(folder):
    report = load(folder / "report.json")
    s = report["summary"]
    rows = report["rows"]
    dec = s["submitted_decisions"] or 1
    lat = [r["step"]["latency"] for r in rows if r.get("step")]
    ctl = [r["controller_time"]["latency"] for r in rows if r.get("controller_time")]
    ad = s.get("adapter_seconds") or {}
    return {
        "verdict": report.get("verdict"), "passed": s["passed"], "rows": s["rows"],
        "natural_completed": s["natural_completed"], "episode_statuses": s["episode_statuses"],
        "replay_verified": s.get("replay_verified"), "submitted_decisions": s["submitted_decisions"],
        "decisions_per_second": s["decisions_per_second"], "completed_games_per_second": s["completed_games_per_second"],
        "collection_seconds": s["collection_seconds"], "controller_seconds": s["controller_seconds"],
        "step_seconds": s["step_seconds"], "engine_wait_seconds": s["engine_wait_seconds"],
        "adapter_seconds": ad, "finalize_seconds": s.get("finalize_seconds"),
        "replay_seconds": sum(r.get("replay_seconds") or 0 for r in rows),
        "audit_seconds": sum(r.get("audit_seconds") or 0 for r in rows),
        "row_seconds": sum(r["row_seconds"] for r in rows),
        "step_latency_per_game": [{"steps": r["step"]["calls"], "p50": r["step"]["latency"]["p50"],
                                   "p99": r["step"]["latency"]["p99"], "max": r["step"]["latency"]["max"]}
                                  for r in rows if r.get("step")],
        "step_latency_p50_max": max(l["p50"] for l in lat) if lat else None,
        "step_latency_p99_max": max(l["p99"] for l in lat) if lat else None,
        "controller_latency_p50_max": max(l["p50"] for l in ctl) if ctl else None,
        "controller_latency_p99_max": max(l["p99"] for l in ctl) if ctl else None,
        "reset": s["reset"], "native_timings": s.get("native_timings"),
        "rss_trajectory": s.get("rss_trajectory"), "thread_trajectory": s.get("thread_trajectory"),
        "descriptor_trajectory": s.get("descriptor_trajectory"),
        "ms_per_decision": {
            "collection": 1e3 * s["collection_seconds"] / dec, "controller": 1e3 * s["controller_seconds"] / dec,
            "step": 1e3 * s["step_seconds"] / dec, "engine_wait": 1e3 * s["engine_wait_seconds"] / dec,
            "ingest": 1e3 * (ad.get("ingest") or 0) / dec, "ledger": 1e3 * (ad.get("ledger") or 0) / dec},
    }


def ab(directory):
    runs, groups = {}, {}
    for report in sorted(directory.glob("*/report.json")):
        m = re.match(r"(v\d-\w+)-(\w+)-(\d+)$", report.parent.name)
        if not m:
            continue
        workload, config, rnd = m.groups()
        brief = run_brief(report.parent)
        runs[report.parent.name] = brief
        if rnd != WARMUP:
            groups.setdefault(workload, {}).setdefault(config, []).append(brief)
    keys = {"decisions_per_second": "rate", "completed_games_per_second": "rate"}
    medians = {}
    for workload, configs in sorted(groups.items()):
        medians[workload] = {}
        for config, items in configs.items():
            row = {k: statistics.median(i[k] for i in items) for k in ("decisions_per_second", "completed_games_per_second",
                                                                        "collection_seconds", "row_seconds", "replay_seconds",
                                                                        "audit_seconds")}
            row["ms_per_decision"] = {k: statistics.median(i["ms_per_decision"][k] for i in items)
                                      for k in items[0]["ms_per_decision"]}
            row["reset_p50"] = statistics.median(i["reset"]["p50"] for i in items)
            row["samples"] = len(items)
            row["all_passed"] = all(i["verdict"] == "passed" for i in items)
            medians[workload][config] = row
        b, a = medians[workload].get("base"), medians[workload].get("accel")
        if b and a:
            medians[workload]["speedup"] = {
                "decisions_per_second": a["decisions_per_second"] / b["decisions_per_second"],
                "completed_games_per_second": a["completed_games_per_second"] / b["completed_games_per_second"],
                "row_seconds": b["row_seconds"] / a["row_seconds"],
                "controller_ms": b["ms_per_decision"]["controller"] / a["ms_per_decision"]["controller"],
                "engine_wait_ms": b["ms_per_decision"]["engine_wait"] / a["ms_per_decision"]["engine_wait"],
                "ingest_ms": b["ms_per_decision"]["ingest"] / a["ms_per_decision"]["ingest"],
                "replay_seconds": b["replay_seconds"] / a["replay_seconds"],
                "audit_seconds": (b["audit_seconds"] / a["audit_seconds"]) if a["audit_seconds"] else None}
    return {"runs": runs, "medians_excluding_round_1": medians,
            "all_runs_passed": all(r["verdict"] == "passed" for r in runs.values()), "run_count": len(runs)}


def perfstat(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split(",")
        if len(parts) > 3 and parts[0] and not line.startswith("#"):
            try:
                result[parts[2]] = float(parts[0])
            except ValueError:
                result[parts[2]] = parts[0]
    return result


def soak(directory):
    result = {}
    for report in sorted(directory.glob("*/report.json")):
        name = report.parent.name
        if "summary" not in load(report):
            continue  # still running
        brief = run_brief(report.parent)
        stat = perfstat(directory / f"{name}.perfstat")
        dec = brief["submitted_decisions"] or 1
        result[name] = {**{k: brief[k] for k in ("verdict", "passed", "rows", "natural_completed", "submitted_decisions",
                                                   "decisions_per_second", "reset", "rss_trajectory", "thread_trajectory",
                                                   "descriptor_trajectory", "native_timings")},
                        "rss_after_load": (load(report).get("resources_start") or {}).get("resident_bytes"),
                        "engine_load_seconds": load(report).get("engine_load_seconds"),
                        "native_load_seconds": (load(report).get("engine_timings_after_load") or {}).get("load_seconds"),
                        "perf_stat": stat,
                        "syscalls_per_decision": stat.get("raw_syscalls:sys_enter", 0) / dec,
                        "context_switches_per_decision": stat.get("context-switches", 0) / dec,
                        "cpu_ms_per_decision": stat.get("task-clock", 0) / dec}
    return result


def lines(path):
    """(config, JSON) pairs of a line file written by measure.sh."""
    for line in Path(path).read_text().splitlines():
        config, payload = line.split(" ", 1)
        if config not in ("base", "accel"):
            raise SystemExit(f"unknown configuration {config!r} in {path}")
        if payload.startswith("k="):
            payload = payload.split(" ", 1)[1]
        yield ("base" if config == "base" else "accelerated"), json.loads(payload)


def scaling(path):
    runs = [{"config": "base" if config == "base" else "accel", **data} for config, data in lines(path)]
    return {"method": ("K concurrent probe processes, each with its own embedded engine: v7 policy (seed 21), 4 games per "
                       "worker with identical engine and policy seeds (121902/121903; seats red, blue, red, blue), full rows "
                       "with vcmireplay and policy audit. Wall time includes process start, imports and engine load. "
                       "One run per worker count and configuration."),
            "runs": runs}


def timing(path, key, method):
    """Milliseconds per decision of interleaved repeats; every repeat must have been identical."""
    values = {"base": [], "accelerated": []}
    for config, data in lines(path):
        if data["identical"] is not True:
            raise SystemExit(f"a repeat in {path} was not identical")
        values[config].append(data[key])
    result = {"method": method, key: values}
    if key == "ms_per_decision":
        result["median_speedup"] = statistics.median(values["base"]) / statistics.median(values["accelerated"])
    else:
        result["second_over_first_median"] = {config: statistics.median(second / first for first, second in items)
                                              for config, items in values.items()}
    return result


REPLAY = ("Engine-free replay of the 4 recorded base-code {version} policy games ({detail}) through BuiltinCollector.act "
          "and finish, requiring every choice (with its log-probability and conditionals), value, estimate, recurrent "
          "prefix and the history to equal the persisted base rollout. Base receives wire records; accelerated also "
          "receives the typed frames, as its environments do. 3 interleaved repeats, torch.set_num_threads(1).")
PASSES = ("The same model replays the 4 recorded base-code {version} policy games twice{how}, 3 interleaved repeats. "
          "The second pass finds what the first left in the model's inference memos, as training's admission audit "
          "finds what collection left when it audits with the collecting model.")

notes, runs, out = load(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def section(name):
    item = notes[name]
    return {"host": item["host"], **({"code": item["code"]} if "code" in item else {}),
            **({"note": item["note"]} if "note" in item else {})}


record = {
    "schema": "heroes-embedded-acceleration-record-v1",
    "date": notes["date"],
    "scope": ("Evaluation-only real VCMI games on Linux. No training, PPO, rewards, demonstrations or map changes. "
              "Policy rows use the labelled freshly initialized v7/v8 model (seed 21); recorded rows resubmit the "
              "delivered actions of an earlier policy run of the base code."),
    "training_performed": False,
    "code": {"base": notes["base"], "accelerated": notes["accelerated"], "engine": notes["engine"], "note": notes["code_note"]},
    "platform": notes["platform"],
    "method": ("Interleaved A/B: each round runs every workload once per configuration, base then accelerated, "
               "sequentially on an otherwise idle machine. 4 episodes per run (seats 0,1,0,1), 1024 choices, 365 days, "
               "natural endings; every row replayed by vcmireplay and every policy row audited. Round 1 is warm-up "
               "and excluded from medians."),
    "embedded_ab": {**section("embedded_ab"), **ab(runs / notes["embedded_ab"]["dir"])},
    "cold_ab": {**section("cold_ab"), **ab(runs / notes["cold_ab"]["dir"])},
    "embedded_ab_host_a": {**section("embedded_ab_host_a"), **ab(runs / notes["embedded_ab_host_a"]["dir"])},
    "repeated_reset_soak": {**section("repeated_reset_soak"), "runs": soak(runs / notes["repeated_reset_soak"]["dir"]),
                            "method": ("Embedded runs use one engine per run with one reused environment per seat (seats 0 and 1 "
                                       "alternating: 12 games each in the 24-game end-turn runs, 6 each in the 12-game "
                                       "policy runs); cold runs start a new game process per game. perf stat counts the "
                                       "whole probe process: 24 end-turn games per run in 3 interleaved rounds per "
                                       "configuration, then 8 cold end-turn games and 12 embedded v7 policy games per "
                                       "configuration; no replay or audit. rss_trajectory is read after each game; "
                                       "rss_after_load before the first.")},
    "scaling": {**section("scaling"), **scaling(runs / notes["scaling"]["file"])},
    "scaling_host_a": {**section("scaling_host_a"), **scaling(runs / notes["scaling_host_a"]["file"])},
    "controller_replay": {**section("controller_replay"), **timing(runs / notes["controller_replay"]["file"], "ms_per_decision",
                          REPLAY.format(version="v7", detail="603 decisions, 4023 frames"))},
    "controller_replay_v8": {**section("controller_replay_v8"), **timing(runs / notes["controller_replay_v8"]["file"],
                             "ms_per_decision", REPLAY.format(version="v8", detail="342 decisions, 2317 frames"))},
    "controller_replay_two_passes": {**section("controller_replay_two_passes"), **timing(
        runs / notes["controller_replay_two_passes"]["file"], "ms_per_decision_per_pass",
        PASSES.format(version="v7", how=" through the collector (controller_replay.py --repeat 2, accelerated with --typed)"))},
    "audit_two_passes": {**section("audit_two_passes"), **timing(runs / notes["audit_two_passes"]["file"],
                         "ms_per_decision_per_pass", PASSES.format(version="v7", how=" through reconstruct_behavior (audit_passes.py)"))},
    "audit_two_passes_v8": {**section("audit_two_passes_v8"), **timing(runs / notes["audit_two_passes_v8"]["file"],
                            "ms_per_decision_per_pass", PASSES.format(version="v8", how=" through reconstruct_behavior (audit_passes.py)"))},
    "review_fix_costs": {**section("review_fix_costs"), **load(runs / notes["review_fix_costs"]["file"]),
                         "method": ("review_fix_costs.py over the 4 recorded base-code v7 policy games: the observation "
                                    "memo's charged bytes against an independent deep measurement of what its entries keep "
                                    "(tensors as data plus 450 bytes), the memo charge timed over all stored entries, and the "
                                    "exact-JSON-type check timed over all delivered records.")},
    "capture_verification": {**section("capture_verification"), **load(runs / notes["capture_verification"]["file"])},
    "independent_review": notes["independent_review"],
}
text = json.dumps(record, indent=1, sort_keys=True)
if "/home/" in text or "/tmp/" in text or "/Users/" in text:
    raise SystemExit("record names a local path")
out.write_text(text + "\n")
print("wrote", out, len(text), "bytes")
