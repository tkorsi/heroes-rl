#!/usr/bin/env python3
"""Run a predetermined series of headless self-play attempts across workers.

The series is entries 0..N-1 of run_headless.py's frozen seed schedule. Worker
``w`` of ``W`` plays entries ``w, w + W, ...`` in its own process and engine
processes, so the attempts and their seeds do not depend on the worker count.
Every attempt is preserved and counted: natural endings, truncations and
failures are reported separately, and a missing attempt is reported as such.
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

FORWARDED = ("map", "binary", "data", "seed", "seed_stride", "red_policy_seed", "blue_policy_seed", "max_days",
             "max_actions", "watchdog", "log_level", "snapshot_every", "ledger", "replay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New series directory")
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--map", type=Path, default=run_headless.DEFAULT_MAP)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=run_headless.DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=40001)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--red-policy-seed", type=int, default=1)
    parser.add_argument("--blue-policy-seed", type=int, default=100001)
    parser.add_argument("--max-days", type=int, default=40)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--watchdog", type=float, default=60.0)
    parser.add_argument("--log-level", default="info", choices=["trace", "debug", "info"])
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--ledger", default="full", choices=["full", "none"])
    parser.add_argument("--replay", default="record", choices=["record", "none"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args(argv)
    if args.attempts < 1 or not 1 <= args.workers <= args.attempts:
        parser.error("Use at least one attempt and 1..attempts workers")
    if not args.skip_check:
        subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], check=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "kind": "headless-self-play-attempt-series",
            "created_at": datetime.now(timezone.utc).isoformat(), "attempts": args.attempts, "workers": args.workers,
            "schedule": "run_headless.py schedule entries 0..attempts-1; worker w plays w, w+W, ...",
            "arguments": {key: str(getattr(args, key)) for key in FORWARDED}}
    run_headless.scoring.write_json(output / "series-plan.json", plan)
    started = time.monotonic()
    processes = []
    for worker in range(args.workers):
        count = len(range(worker, args.attempts, args.workers))
        command = [sys.executable, str(HERE / "run_headless.py"), "--output", str(output / f"worker-{worker}"),
                   "--episodes", str(count), "--schedule-start", str(worker), "--schedule-step", str(args.workers),
                   "--skip-check"]
        for key in FORWARDED:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        if args.profile:
            command.append("--profile")
        log = (output / f"worker-{worker}.log").open("w")
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, worker))
    returncodes = {}
    for process, log, worker in processes:
        returncodes[worker] = process.wait()
        log.close()
    wall = time.monotonic() - started
    rows, summaries = [], []
    for worker in range(args.workers):
        path = output / f"worker-{worker}" / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text())
            summaries.append(summary)
            rows += summary["episodes"]
    rows.sort(key=lambda row: row["episode"])
    expected = {run_headless.schedule_entry(argparse.Namespace(first_episode=1, seed=args.seed, seed_stride=args.seed_stride,
                                                               red_policy_seed=args.red_policy_seed,
                                                               blue_policy_seed=args.blue_policy_seed), index)["episode"]
                for index in range(args.attempts)}
    played = {row["episode"] for row in rows}
    aggregate = run_headless.aggregate(rows, wall) if rows else {}
    lengths = sorted(max(row["red"]["last_day"], row["blue"]["last_day"]) for row in rows)
    report = {**plan, "finished_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": round(wall, 3),
              "worker_returncodes": returncodes, "host": summaries[0]["host"] if summaries else None,
              "build_type": summaries[0]["build_type"] if summaries else None,
              "toolchain": summaries[0].get("toolchain") if summaries else None,
              "identity": summaries[0]["identity"] if summaries else None,
              "attempts_played": len(rows), "attempts_missing": sorted(expected - played),
              "attempts_unplanned": sorted(played - expected),
              "natural_completions": aggregate.get("natural_completions", 0),
              "truncations": aggregate.get("truncations", 0), "failures": aggregate.get("failures", 0),
              "status_counts": aggregate.get("status_counts", {}),
              "natural_wins_by_color": {color: sum(1 for row in rows if row["status"] == "completed"
                                                   and row["outcomes"].get(color) == "win") for color in ("red", "blue")},
              "episode_days": {"min": lengths[0] if lengths else None, "median": lengths[len(lengths) // 2] if lengths else None,
                               "max": lengths[-1] if lengths else None},
              "failure_details": [{"episode": row["episode"], "termination_reason": row["termination_reason"],
                                   "detail": row.get("failure_detail")} for row in rows if row["status"] == "failed"],
              "aggregate": aggregate, "episodes": rows}
    run_headless.scoring.write_json(output / "series.json", report)
    print(json.dumps({key: report[key] for key in ("attempts", "attempts_played", "natural_completions", "truncations",
                                                   "failures", "status_counts", "wall_seconds")}, indent=2))
    return 0 if len(rows) == args.attempts else 1


if __name__ == "__main__":
    sys.exit(main())
