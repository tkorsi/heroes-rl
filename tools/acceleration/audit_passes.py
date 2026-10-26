"""The policy audit (reconstruct_behavior) of recorded games, twice with one model.

The first pass starts from empty inference memos; the second finds what the
first left, as training's admission audit finds what collection left when it
audits with the collecting model. Every pass must reconstruct every choice and
value exactly. Prints one JSON line with milliseconds per decision per pass.

Usage: python tools/acceleration/audit_passes.py [--interface NAME] EPISODE_DIR...
It imports the checkout it sits in, or the one HEROES_ROOT names.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, os.environ.get("HEROES_ROOT") or str(Path(__file__).resolve().parents[2]))
import torch
torch.set_num_threads(1)
from experiments.headless import builtin_evidence as evidence
from experiments.learner import contextual as c
from experiments.learner.builtin_rollout import load_builtin_episode
from validation.probes.validate_embedded_engine import random_policy
from validation.probes.validate_hero_policy import reconstruct_behavior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--interface", default=c.DIALOGS_ENVIRONMENT)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    loaded, _ = random_policy(args.seed, args.interface)
    games = [load_builtin_episode(folder / "builtin-rollout.json") for folder in args.episodes]
    passes = []
    for _ in range(args.passes):
        seconds = decisions = 0
        for collected in games:
            schema = evidence.manifest_profile(collected.manifest).history_schema
            begun = time.perf_counter()
            report = reconstruct_behavior(collected.history, collected.decisions, collected.estimates,
                                          loaded.model, collected.policy_seed, schema=schema)
            seconds += time.perf_counter() - begun
            if not (report.get("maximum_absolute_difference") == 0 and report.get("weights_unchanged") is True
                    and report.get("actions_resampled") == len(collected.decisions)
                    and report.get("values_recomputed") == len(collected.estimates)):
                raise SystemExit("the audit did not reconstruct every choice and value exactly")
            decisions += len(collected.decisions)
        passes.append(1e3 * seconds / decisions)
    print(json.dumps({"games": len(games), "decisions_per_pass": decisions, "ms_per_decision_per_pass": passes,
                      "identical": True}))


if __name__ == "__main__":
    main()
