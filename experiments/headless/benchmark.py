#!/usr/bin/env python3
"""Measure headless self-play throughput with 1..N isolated worker processes.

Every configuration runs the same frozen workload: the authored map, two
seeded uniform-random agents and one global seed schedule. Worker ``w`` of
``W`` plays schedule entries ``region + w``, ``region + w + W``, ... where
each repetition owns a disjoint region, so all worker counts draw from the
same episodes. Each worker is a separate ``run_headless.py`` process that
launches one isolated engine process per episode.

Each worker first plays unmeasured warm-up episodes, then all workers start
their measured window at the same wall-clock time. The episode in progress
when the window closes is truncated as ``collection_window``. Throughput is
end to end: launches, startup, every step, shutdown and relaunch inside the
window all count. Steady-state rates (time spent stepping only) are reported
separately and never replace it. The report is rewritten after every
configuration and marked ``complete`` only after the last one.

A decision cycle is one submitted adventure action that the engine resolved
with an ``action_result``. Accepted actions and rejected attempts are also
reported on their own, as are moves, end turns, town-only turns, dialogs and
battles, so that a rejection loop or forced end-turn loop cannot hide inside
the headline number.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_headless  # noqa: E402

REGION = 1_000_000  # schedule indices per repetition


def percentile(ordered: list[float], q: float):
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 6) if ordered else None


def summarize_configuration(workers: int, summaries: list[dict], window: float, single: float | None) -> dict:
    rows = [row for summary in summaries for row in summary["episodes"]]
    totals = run_headless.aggregate(rows, window) if rows else {}
    walls = [s["aggregate"]["wall_seconds"] for s in summaries if s.get("aggregate")]
    measured = max(walls) if walls else window
    resolved = sum(row[c]["resolved"] for row in rows for c in ("red", "blue"))
    accepted = sum(row[c]["accepted"] for row in rows for c in ("red", "blue"))
    stepping = [row["engine_wait_seconds"] + row["agent_seconds"] for row in rows]
    latencies = sorted(value for row in rows for value in row.get("step_latencies", []))
    finished = [row for row in rows if row["status"] != "truncated" or row["termination_reason"] != "collection_window"]
    startup = sorted(row["startup_seconds"] for row in rows if row["startup_seconds"] is not None)
    shutdown = sorted(row["shutdown_seconds"] for row in rows if row["shutdown_seconds"] is not None)
    child_cpu = sum(s["aggregate"].get("child_cpu_seconds", 0.0) for s in summaries if s.get("aggregate"))
    harness_cpu = sum(s["aggregate"].get("harness_cpu_seconds", 0.0) for s in summaries if s.get("aggregate"))
    logical = summaries[0]["host"].get("logical_cpus") if summaries else None
    rate = resolved / measured if measured else None
    log_bytes = sum(row["log_bytes"] for row in rows)
    result = {
        "workers": workers, "measured_seconds": round(measured, 3), "episodes_started": len(rows),
        "episodes_finished_inside_window": len(finished),
        "status_counts": totals.get("status_counts", {}),
        "natural_completions": totals.get("natural_completions", 0),
        "truncations_excluding_window_end": sum(1 for row in finished if row["status"] == "truncated"),
        "failures": totals.get("failures", 0),
        "decision_cycles": resolved, "accepted_actions": accepted, "rejected_attempts": resolved - accepted,
        "end_to_end_decision_cycles_per_second": round(rate, 3) if rate is not None else None,
        "end_to_end_accepted_actions_per_second": round(accepted / measured, 3) if measured else None,
        "end_to_end_rejected_attempts_per_second": round((resolved - accepted) / measured, 3) if measured else None,
        "natural_games_per_second": round(totals.get("natural_completions", 0) / measured, 6) if measured else None,
        "steady_state_decision_cycles_per_second_per_worker":
            round(resolved / sum(stepping), 3) if sum(stepping) else None,
        "breakdown": {key: totals.get(key) for key in ("moves", "end_turns", "town_only_turns", "dialogs",
                                                       "battle_participations_red_plus_blue",
                                                       "decisions", "turns")},
        "coverage_totals": totals.get("coverage_totals", {}),
        "per_finished_episode": {
            "decision_cycles_mean": round(sum(r["red"]["resolved"] + r["blue"]["resolved"] for r in finished) / len(finished), 3)
            if finished else None,
            "days_mean": round(sum(max(r["red"]["last_day"], r["blue"]["last_day"]) for r in finished) / len(finished), 3)
            if finished else None,
            "wall_seconds_mean": round(sum(r["wall_seconds"] for r in finished) / len(finished), 3) if finished else None},
        "step_latency_seconds": {"count": len(latencies), "p50": percentile(latencies, 0.5),
                                 "p90": percentile(latencies, 0.9), "p99": percentile(latencies, 0.99),
                                 "max": latencies[-1] if latencies else None},
        "startup_seconds": {"p50": percentile(startup, 0.5), "p90": percentile(startup, 0.9),
                            "max": startup[-1] if startup else None},
        "shutdown_seconds": {"p50": percentile(shutdown, 0.5), "p90": percentile(shutdown, 0.9),
                             "max": shutdown[-1] if shutdown else None},
        "cpu": {"engine_seconds": round(child_cpu, 3), "harness_seconds": round(harness_cpu, 3),
                "utilization_of_all_logical_cpus": round((child_cpu + harness_cpu) / (measured * logical), 4)
                if measured and logical else None},
        "peak_engine_rss_bytes": max((s["aggregate"].get("child_peak_rss_bytes", 0) for s in summaries
                                      if s.get("aggregate")), default=None),
        "log_bytes": log_bytes, "log_bytes_per_decision_cycle": round(log_bytes / resolved, 1) if resolved else None,
        "history_v1_invalid_players": sum(1 for row in rows for c in ("red", "blue") if not row[c]["history_v1"]["valid"]),
        "warmup": [entry for s in summaries for entry in s.get("warmup", [])],
        "measured_start_spread_seconds": round(max(s["measured_start"] for s in summaries)
                                               - min(s["measured_start"] for s in summaries), 3) if summaries else None,
    }
    if single and rate is not None:
        result["scaling_efficiency"] = round(rate / (workers * single), 4)
    return result


def run_configuration(args, workers: int, repetition: int, directory: Path) -> list[dict]:
    directory.mkdir(parents=True)
    start_at = time.time() + args.warmup_allowance
    processes = []
    for worker in range(workers):
        output = directory / f"worker-{worker}"
        command = [sys.executable, str(HERE / "run_headless.py"), "--output", str(output),
                   "--map", str(args.map), "--binary", str(args.binary), "--data", str(args.data),
                   "--episodes", str(args.max_episodes_per_worker), "--seed", str(args.seed),
                   "--schedule-start", str(repetition * REGION + worker), "--schedule-step", str(workers),
                   "--warmup-episodes", str(args.warmup_episodes), "--start-at", f"{start_at:.3f}",
                   "--duration", str(args.duration), "--max-days", str(args.max_days),
                   "--max-actions", str(args.max_actions), "--log-level", args.log_level,
                   "--ledger", args.ledger, "--replay", args.replay, "--snapshot-every", str(args.snapshot_every),
                   "--raw-latencies", "--skip-check"] + (["--profile"] if args.profile else [])
        log = (directory / f"worker-{worker}.log").open("w")
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, output))
    summaries = []
    for process, log, output in processes:
        process.wait()
        log.close()
        summaries.append(json.loads((output / "summary.json").read_text()))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    parser.add_argument("--workers", default="1,2,4,8", help="Comma-separated worker counts")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration", type=float, default=180.0, help="Measured window per repetition, seconds")
    parser.add_argument("--warmup-episodes", type=int, default=1, help="Unmeasured episodes per worker first")
    parser.add_argument("--warmup-allowance", type=float, default=120.0,
                        help="Seconds from launch to the synchronized start of the measured window")
    parser.add_argument("--max-episodes-per-worker", type=int, default=100000)
    parser.add_argument("--map", type=Path, default=run_headless.DEFAULT_MAP)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=run_headless.DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=40001)
    parser.add_argument("--max-days", type=int, default=40)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--log-level", default="info", choices=["trace", "debug", "info"])
    parser.add_argument("--ledger", default="none", choices=["full", "none"])
    parser.add_argument("--replay", default="none", choices=["record", "none"],
                        help="record: also write replay evidence (requires --ledger full)")
    parser.add_argument("--snapshot-every", type=int, default=1,
                        help="Full visible-tile verification period in decisions (0: never)")
    parser.add_argument("--profile", action="store_true", help="Collect ExplorerRL per-message timing fields")
    parser.add_argument("--skip-check", action="store_true", help="Skip tools/prepare_vcmi.py --check")
    args = parser.parse_args()
    worker_counts = [int(value) for value in args.workers.split(",")]
    if args.repetitions < 1 or args.duration <= 0 or min(worker_counts) < 1:
        parser.error("Use positive repetitions, duration and worker counts")
    if args.replay == "record" and args.ledger != "full":
        parser.error("--replay record needs --ledger full")

    launcher = run_headless.env_module.VcmiLauncher(args.binary, args.data, args.map, args.log_level)
    try:
        launcher.preflight()
        if not args.skip_check:
            subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], check=True)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Preflight failed: {error}\n")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": "headless-self-play-benchmark",
              "created_at": datetime.now(timezone.utc).isoformat(), "host": run_headless.host_description(),
              "build_type": run_headless.build_type(launcher.binary),
              "toolchain": run_headless.toolchain(launcher.binary),
              "identity": {**launcher.identity(), "engine": run_headless.env_module._git_state(ROOT / "vcmi"),
                           "project": run_headless.env_module._git_state(ROOT),
                           "engine_recipe": run_headless.scoring.engine_recipe_digests(ROOT)},
              "workload": {"map": str(args.map), "agents": "independent seeded uniform-random agents, both colors",
                           "seed": args.seed, "repetition_region": REGION, "max_days": args.max_days,
                           "max_actions_per_player": args.max_actions, "log_level": args.log_level,
                           "ledger": args.ledger, "replay": args.replay, "snapshot_every": args.snapshot_every},
              "method": {"duration_seconds": args.duration, "repetitions": args.repetitions,
                         "warmup_episodes_per_worker": args.warmup_episodes,
                         "warmup_allowance_seconds": args.warmup_allowance,
                         "decision_cycle": "submitted adventure action resolved by an engine action_result"},
              "configurations": [], "complete": False, "finished_at": None}
    path = output / "benchmark.json"
    run_headless.scoring.write_json(path, report)
    single = None
    for workers in worker_counts:
        repetitions = []
        for repetition in range(args.repetitions):
            summaries = run_configuration(args, workers, repetition, output / f"workers-{workers}" / f"rep-{repetition}")
            repetitions.append(summarize_configuration(workers, summaries, args.duration, single))
            print(json.dumps({"workers": workers, "repetition": repetition,
                              "decision_cycles_per_second": repetitions[-1]["end_to_end_decision_cycles_per_second"],
                              "failures": repetitions[-1]["failures"]}), flush=True)
        rates = [r["end_to_end_decision_cycles_per_second"] or 0.0 for r in repetitions]
        mean = sum(rates) / len(rates)
        if workers == 1:
            single = mean
            for entry in repetitions:
                entry["scaling_efficiency"] = round((entry["end_to_end_decision_cycles_per_second"] or 0.0) / single, 4) if single else None
        report["configurations"].append({
            "workers": workers, "repetitions": repetitions,
            "end_to_end_decision_cycles_per_second": {"mean": round(mean, 3), "min": min(rates), "max": max(rates),
                                                      "relative_spread": round((max(rates) - min(rates)) / mean, 4) if mean else None},
            "scaling_efficiency_mean": round(mean / (workers * single), 4) if single else None,
            "stretch_target_1000_per_second_met": mean >= 1000.0})
        run_headless.scoring.write_json(path, report)
    # Only now is the report a result; an interrupted or failed benchmark stays incomplete.
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    run_headless.scoring.write_json(path, report)
    print(json.dumps([{k: c[k] for k in ("workers", "end_to_end_decision_cycles_per_second", "scaling_efficiency_mean")}
                      for c in report["configurations"]], indent=2))


if __name__ == "__main__":
    main()
