#!/usr/bin/env python3
"""Run our untrained uniform agent against fixed Nullkiller Easy, with both videos."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from random_policy import RandomPolicy

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("heroes_record", ROOT / "experiments/recording/record.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--map", type=Path, required=True, help="Authored small scenario with towns, mines and neutrals")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345, help="Engine seed")
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--max-days", type=int, default=40)
    parser.add_argument("--step-delay", type=float, default=0.15, help="Seconds per external choice; timing only")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--view", choices=("both", "player", "omniscient"), default="both")
    parser.add_argument("--player", choices=recorder.PLAYER_COLORS, default="red")
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()
    try:
        policy = RandomPolicy(args.policy_seed, args.max_actions, args.max_days, args.step_delay)
        result = recorder.record(args, policy)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "outcome", "reward", "termination_reason", "policy_execution", "recordings")}, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
