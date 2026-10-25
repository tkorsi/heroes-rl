#!/usr/bin/env python3
"""Run one owned cloud engineering job inside the authorized absolute window.

Training uses its own nested worker/native guardians as well. This wrapper bounds
builds, diagnostics and checks; it never signals a historical process identifier.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.headless.process_deadline import guarded_popen, close_owned_process
from experiments.learner.builtin_training import COMPUTE_DEADLINE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.seconds <= 0 or time.time()+2 >= COMPUTE_DEADLINE:
        parser.error('A command, positive bound and remaining absolute session time are required')
    wall = min(COMPUTE_DEADLINE, time.time()+args.seconds)
    evidence = Path('/workspace/Heroes-artifacts/session/jobs')
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / (str(time.time_ns())+'.jsonl')
    process = guarded_popen(command, deadline_monotonic=time.monotonic()+wall-time.time(),
        deadline_wall=wall, evidence_path=path, stdin=subprocess.DEVNULL)
    try:
        code = process.wait()
    finally:
        close_owned_process(process, timeout=3)
    return code if code >= 0 else 128-code


if __name__ == '__main__':
    raise SystemExit(main())
