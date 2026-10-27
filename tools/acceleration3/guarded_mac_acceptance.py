#!/usr/bin/env python3
"""Run the explicitly authorized Mac699 study under an independent absolute guard.

The guardian starts before the harness, covers preparation as well as gameplay,
and stops its private group at the same original deadline even if this launcher
or harness disappears. Nested runner/native groups own their own parent pipes.
No official training is launched and no existing PID is signalled by this tool.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.headless.process_deadline import guarded_popen, close_owned_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, required=True)
    parser.add_argument('--total-hours', type=float, default=8)
    args, rest = parser.parse_known_args()
    if not 0.5 <= args.total_hours <= 24:
        parser.error('--total-hours must be 0.5 to 24')
    if any(x.startswith('--study-deadline') for x in rest):
        parser.error('the independent guard owns the study deadline')
    root = args.bench_root.expanduser().resolve()
    if root.exists():
        parser.error('benchmark directory already exists; attempts are never reused')
    root.parent.mkdir(parents=True, exist_ok=True)
    wall = time.time() + args.total_hours * 3600
    mono = time.monotonic() + args.total_hours * 3600
    command = [sys.executable, str(Path(__file__).with_name('mac_acceptance.py')),
               '--bench-root', str(root), '--total-hours', str(args.total_hours),
               '--study-deadline-unix', str(wall), '--study-deadline-monotonic', str(mono), *rest]
    evidence = root.with_name(root.name + '.guard.jsonl')
    receipt = root.with_name(root.name + '.guard-receipt.json')
    with receipt.open('x') as stream:
        json.dump(dict(schema='heroes-mac699-study-guard-v1', command=command,
                       hard_deadline_unix=wall, hard_deadline_monotonic=mono,
                       hard_deadline_utc=datetime.fromtimestamp(wall, timezone.utc).isoformat(),
                       evidence=str(evidence)), stream, indent=2)
        stream.write('\n')
    process = guarded_popen(command, deadline_monotonic=mono, deadline_wall=wall,
                            evidence_path=evidence, cwd=ROOT)
    status = None
    try:
        status = process.wait(timeout=max(0, min(mono-time.monotonic(), wall-time.time())))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # Ownership release wakes the already armed guardian; never signal an old PID.
        pass
    finally:
        try:
            close_owned_process(process, timeout=3, deadline_monotonic=mono, deadline_wall=wall)
        except (TimeoutError, subprocess.TimeoutExpired):
            # The guardian enforces the original bound independently. Do not wait beyond it.
            pass
    return status if status is not None else 124


if __name__ == '__main__':
    raise SystemExit(main())
