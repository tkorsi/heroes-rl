#!/usr/bin/env python3
"""Condense real-game headless validation into one compact JSON for Git.

Raw runs, engine logs, ledgers, replays and videos stay out of Git. This keeps
what a reviewer needs to check the claims: identity of the build and host, the
attempt series with every attempt's status, replay verification and tamper
results, video hashes, benchmark rates with their spread, and the cost
breakdown. Each input is optional; a missing input is recorded as missing,
and a benchmark that stopped before its last configuration as incomplete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def load(path: Path | None):
    if path is None:
        return None
    return json.loads(path.read_text())


def compact_series(series: dict) -> dict:
    return {key: series.get(key) for key in (
        "attempts", "workers", "attempts_played", "attempts_missing", "attempts_unplanned", "natural_completions",
        "truncations", "failures", "status_counts", "natural_wins_by_color", "episode_days", "failure_details",
        "wall_seconds", "arguments", "host", "build_type", "toolchain", "identity")} | {
        "aggregate": {key: value for key, value in (series.get("aggregate") or {}).items() if key != "coverage_totals"},
        "coverage_totals": (series.get("aggregate") or {}).get("coverage_totals"),
        # One row per attempt: episode, seed, status, reason, outcome, days and decision cycles.
        "attempt_rows": [[row["episode"], row["engine_seed"], row["status"], row["termination_reason"],
                          next((color for color, outcome in (row.get("outcomes") or {}).items() if outcome == "win"), None),
                          max(row["red"]["last_day"], row["blue"]["last_day"]),
                          row["red"]["resolved"] + row["blue"]["resolved"],
                          row["red"]["battles"] + row["blue"]["battles"]] for row in series.get("episodes", [])],
        "attempt_row_fields": ["episode", "engine_seed", "status", "termination_reason", "winner", "days",
                               "decision_cycles", "battles_started_by_either_side"]}


def compact_tamper(tamper: dict) -> dict:
    # A failed or skipped tamper run stays distinguishable from one that never ran.
    return {"all_detected": tamper.get("all_detected"), "error": tamper.get("error"), "skipped": tamper.get("skipped"),
            "crashed_cases": tamper.get("crashed_cases") or [],
            "cases": {case["case"]: case["detected"] for case in tamper.get("cases", [])}}


def compact_replays(replays: dict) -> dict:
    rows = []
    for row in replays.get("episodes", []):
        rows.append({"episode": row["episode"], "status": row["status"], "termination_reason": row["termination_reason"],
                     "verified": row["verified"], "error": row.get("error"), "divergence": row.get("divergence"),
                     "inputs": (row.get("summary") or {}).get("inputs"),
                     "boundaries": (row.get("summary") or {}).get("boundaries"),
                     "observations_checked": (row.get("summary") or {}).get("observations_checked"),
                     "received_updates_checked": {color: check.get("captures_checked")
                                                  for color, check in (row.get("players") or {}).items()},
                     "observation_coverage_ok": (row.get("observation_coverage") or {}).get("ok"),
                     "terminal_evidence": row.get("terminal_evidence"), "seconds": row.get("seconds"),
                     "tamper": compact_tamper(row["tamper"]) if "tamper" in row else None})
    return {"aggregate": replays.get("aggregate"), "episodes": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--series", type=Path, help="run_series.py series.json")
    parser.add_argument("--replays", type=Path, help="replay.py series replay-series.json")
    parser.add_argument("--videos", type=Path, nargs="*", default=[], help="render_replay.py sidecar .json files")
    parser.add_argument("--benchmark", type=Path, help="benchmark.py benchmark.json")
    parser.add_argument("--costs", type=Path, help="cost_breakdown.py output")
    parser.add_argument("--note", action="append", default=[], help="A limitation or observation to record")
    args = parser.parse_args(argv)
    series, replays, benchmark, costs = load(args.series), load(args.replays), load(args.benchmark), load(args.costs)
    notes = list(args.note)
    if benchmark and benchmark.get("complete") is not True:
        notes.append("The benchmark is incomplete: it stopped before its last configuration, so its configurations "
                     "are partial and are not a benchmark result.")
    evidence = {
        "schema_version": 1, "kind": "headless-real-game-validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "series": compact_series(series) if series else "missing",
        "replays": compact_replays(replays) if replays else "missing",
        "videos": [{key: value for key, value in load(path).items() if key != "output"} | {"file": path.with_suffix(".mp4").name}
                   for path in args.videos] or "missing",
        "benchmark": {"complete": benchmark.get("complete") is True, "finished_at": benchmark.get("finished_at"),
                      "host": benchmark.get("host"), "build_type": benchmark.get("build_type"),
                      "toolchain": benchmark.get("toolchain"),
                      "identity": benchmark.get("identity"), "workload": benchmark.get("workload"),
                      "method": benchmark.get("method"),
                      "configurations": [{
                          "workers": c["workers"], "end_to_end_decision_cycles_per_second": c["end_to_end_decision_cycles_per_second"],
                          "scaling_efficiency_mean": c["scaling_efficiency_mean"],
                          "stretch_target_1000_per_second_met": c["stretch_target_1000_per_second_met"],
                          "repetitions": [{k: r.get(k) for k in (
                              "measured_seconds", "episodes_started", "episodes_finished_inside_window", "status_counts",
                              "natural_completions", "truncations_excluding_window_end", "failures", "decision_cycles",
                              "accepted_actions", "rejected_attempts", "end_to_end_decision_cycles_per_second",
                              "end_to_end_accepted_actions_per_second", "end_to_end_rejected_attempts_per_second",
                              "steady_state_decision_cycles_per_second_per_worker", "breakdown", "step_latency_seconds",
                              "startup_seconds", "shutdown_seconds", "cpu", "peak_engine_rss_bytes",
                              "log_bytes_per_decision_cycle", "per_finished_episode", "scaling_efficiency")}
                                          for r in c["repetitions"]]} for c in benchmark.get("configurations", [])]}
        if benchmark else "missing",
        "costs": costs or "missing",
        "notes": notes,
    }
    args.output.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
