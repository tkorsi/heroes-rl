#!/usr/bin/env python3
"""Run two independent uniform-random agents with each side's fog and paired videos."""
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
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=22345, help="Engine seed")
    parser.add_argument("--red-policy-seed", type=int, default=1)
    parser.add_argument("--blue-policy-seed", type=int, default=101)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--max-actions", type=int, default=2000, help="Action limit per player")
    parser.add_argument("--max-days", type=int, default=40)
    parser.add_argument("--step-delay", type=float, default=0.03, help="Seconds per external choice; timing only")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1800, help="Frame limit per stream")
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--no-encode", action="store_true")
    parser.set_defaults(view="both", player="red")
    args = parser.parse_args()
    try:
        policies = {index: RandomPolicy(seed, args.max_actions, args.max_days, args.step_delay)
                    for index, seed in enumerate((args.red_policy_seed, args.blue_policy_seed))}
        result = recorder.record(args, policies=policies)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    print(json.dumps({key: result[key] for key in (
        "status", "outcomes_by_player", "rewards_by_player", "termination_reason",
        "policy_executions", "recordings")}, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
