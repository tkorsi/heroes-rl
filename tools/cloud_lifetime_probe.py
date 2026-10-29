#!/usr/bin/env python3
"""Record a short process-lifetime observation without training or API calls.

The probe never detaches or restarts itself. A suspended host cannot execute an
exit: on resumption, either expired clock ends the probe before another sleep.
Saved files alone do not prove that this process ran continuously.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid


SCHEMA = "heroes-cloud-lifetime-probe-v1"
MAX_SECONDS = 20 * 60


def bounded_number(value, minimum, maximum, name):
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must be between {minimum} and {maximum}")
    return number


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def probe(output, seconds, interval):
    output.mkdir(parents=True, exist_ok=False)
    started_wall, started_mono = time.time(), time.monotonic()
    identity = dict(schema=SCHEMA, run_id=uuid.uuid4().hex, pid=os.getpid(),
                    parent_pid=os.getppid(), started_unix=started_wall,
                    started_monotonic=started_mono,
                    deadline_unix=started_wall + seconds,
                    deadline_monotonic=started_mono + seconds,
                    requested_seconds=seconds, interval_seconds=interval,
                    authorizes_training=False, authorizes_continuation=False)
    write_json(output / "session.json", identity)
    previous = None
    sequence = 0
    gaps = 0
    largest_gap = 0.0

    def record(status):
        nonlocal previous, sequence, gaps, largest_gap
        wall, mono = time.time(), time.monotonic()
        wall_delta = wall - previous[0] if previous else 0.0
        mono_delta = mono - previous[1] if previous else 0.0
        gap = max(0.0, max(wall_delta, mono_delta) - interval) if previous else 0.0
        divergence = wall_delta - mono_delta
        gap_observed = previous is not None and (gap > 1.0 or abs(divergence) > 1.0)
        gaps += int(gap_observed)
        largest_gap = max(largest_gap, gap)
        row = dict(schema=SCHEMA, run_id=identity["run_id"], pid=identity["pid"],
                   sequence=sequence, status=status,
                   observed_utc=datetime.fromtimestamp(wall, timezone.utc).isoformat(),
                   observed_unix=wall, observed_monotonic=mono,
                   elapsed_wall_seconds=wall - started_wall,
                   elapsed_monotonic_seconds=mono - started_mono,
                   wall_seconds_remaining=identity["deadline_unix"] - wall,
                   monotonic_seconds_remaining=identity["deadline_monotonic"] - mono,
                   wall_since_previous_seconds=wall_delta,
                   monotonic_since_previous_seconds=mono_delta,
                   wall_minus_monotonic_delta_seconds=divergence,
                   excess_interval_seconds=gap, gap_observed=gap_observed)
        with (output / "heartbeats.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(output / "progress.json", row)
        previous = (wall, mono)
        sequence += 1
        return row

    status = "deadline_reached"
    try:
        while True:
            remaining = min(identity["deadline_unix"] - time.time(),
                            identity["deadline_monotonic"] - time.monotonic())
            if remaining <= 0:
                break
            record("running")
            remaining = min(identity["deadline_unix"] - time.time(),
                            identity["deadline_monotonic"] - time.monotonic())
            if remaining > 0:
                time.sleep(min(interval, remaining))
    except KeyboardInterrupt:
        status = "interrupted"
    final = record(status)
    completion = dict(**identity, final_observation=final, status=status,
                      observation_count=sequence, gaps_observed=gaps,
                      largest_excess_interval_seconds=largest_gap,
                      continuous_execution_proven=False,
                      interpretation="A finite observation, not a service guarantee. Gaps may reflect "
                      "suspension, scheduling delays or clock changes. Missing completion is incomplete evidence.")
    write_json(output / "completion.json", completion)
    print(json.dumps(dict(output=str(output), status=status, observations=sequence,
                          gaps_observed=gaps, continuous_execution_proven=False)))
    return 0 if status == "deadline_reached" else 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh directory; an existing path is rejected")
    parser.add_argument("--seconds", default=120.0,
                        type=lambda value: bounded_number(value, 0.1, MAX_SECONDS, "seconds"))
    parser.add_argument("--interval", default=5.0,
                        type=lambda value: bounded_number(value, 0.1, 60, "interval"))
    args = parser.parse_args()
    try:
        return probe(args.output.expanduser().absolute(), args.seconds, args.interval)
    except (OSError, ValueError) as error:
        print(f"Probe failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
