"""Bounded native collection with fixed random weights; no optimizer entry point.

Every invocation is a diagnostic cohort. It retains every attempted episode,
alternates seats against an episode-frozen opponent, and never treats a cutoff
or failure as a draw. Raw histories/checkpoints stay in the ignored run folder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import torch

from experiments.headless.env import HeadlessSelfPlayEnv, Limits, VcmiLauncher
from experiments.learner import contextual as contract
from experiments.learner.checkpoints import (
    FrozenOpponentPool, load_policy_checkpoint, save_policy_checkpoint,
)
from experiments.learner.contextual_model import ContextualActorCritic, weights_sha256
from experiments.learner.contextual_rollout import ContextualCollector
from validation.probes.audit_contextual_histories import audit

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("red", "blue")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = sorted((ROOT / "experiments/learner").glob("*.py"))
    paths += [ROOT / "dependencies/learner.lock"]
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "torch": str(torch.__version__), "python": sys.version,
        "interface": contract.TOWN_ENVIRONMENT,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def history_coverage(histories):
    """Describe actually submitted choices, never labels or shaped rewards."""
    result = {}
    for color, frames in zip(COLORS, histories):
        result[color] = {
            "frames": len(frames),
            "kinds": dict(Counter(f["kind"] for f in frames)),
            "submitted_actions": dict(Counter(f["action"]["kind"] for f in frames if f["kind"] == "submitted")),
            "result_statuses": dict(Counter(f["result"]["status"] for f in frames if f["kind"] == "result")),
            "last_public_day": max((f["day"] for f in frames), default=None),
        }
    return result


def diagnostic_summary(attempts):
    """Keep distinct denominators and make no diagnostic outcome-rate claim."""
    counts = Counter(attempt["status"] for attempt in attempts)
    return {
        "attempts": len(attempts),
        "verified_natural": counts["completed"],
        "verified_cutoff": counts["truncated"],
        "technical_failure": counts["failed"],
        "collection_or_audit_error": counts["collection_error"],
        "outcome_rate": None,
        "rate_reason": "Fixed-weight diagnostic checks are not an outcome-frequency evaluation cohort",
        "training_samples": 0, "optimizer_updates": 0,
    }


def run(args):
    torch.set_num_threads(1)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    if args.checkpoint:
        loaded = load_policy_checkpoint(args.checkpoint, allowed_provenance=("random_initialization",))
        learner, learner_id = loaded.model, loaded.checkpoint
    else:
        torch.manual_seed(args.init_seed)
        learner = ContextualActorCritic()
        learner_id = "untrained-" + weights_sha256(learner)[:16]
    learner.eval()
    learner_digest = weights_sha256(learner)
    pool = FrozenOpponentPool()
    for seed in args.opponent_seeds:
        torch.manual_seed(seed)
        opponent = ContextualActorCritic()
        pool.add("untrained-" + weights_sha256(opponent)[:16], opponent)
    opponent_rng = random.Random(args.opponent_sampling_seed)
    # The same fixed opponent is used for both colors of each engine seed.
    plan = [(seed, pool.sample(opponent_rng).checkpoint) for seed in args.engine_seeds]
    save_policy_checkpoint(
        output / "untrained-policy.pt", learner, checkpoint=learner_id,
        provenance="random_initialization", source_metadata=source,
        python_generators={"opponent": opponent_rng}, opponent_pool=pool)
    report = {
        "schema": "heroes-contextual-neural-diagnostic-v1", "purpose": "diagnostic",
        "training_performed": False, "optimizer_updates": 0,
        "policy": {"checkpoint": learner_id, "weights_sha256": learner_digest,
                   "initialization_seed": None if args.checkpoint else args.init_seed},
        "opponents": list(pool.identities), "source": source,
        "configuration": {"engine_seeds": args.engine_seeds, "opponent_seeds": args.opponent_seeds,
                          "policy_seed": args.policy_seed, "opponent_sampling_seed": args.opponent_sampling_seed,
                          "max_requests": args.max_requests, "max_days": args.max_days,
                          "map": str(args.game_map.resolve()), "map_sha256": digest(args.game_map)},
        "plan": [{"engine_seed": seed, "opponent": opponent, "learner_seats": [0, 1]}
                 for seed, opponent in plan],
        "attempts": [], "contextual_replay": "unsupported",
    }
    write_json(output / "report.json", report)
    episode = 0
    for seed, opponent_id in plan:
        for learner_player in (0, 1):
            episode += 1
            opponent = pool.get(opponent_id)
            models = (learner, opponent.model) if learner_player == 0 else (opponent.model, learner)
            checkpoints = (learner_id, opponent_id) if learner_player == 0 else (opponent_id, learner_id)
            policy_seeds = (args.policy_seed + 2 * episode, args.policy_seed + 2 * episode + 1)
            collector = ContextualCollector(
                models=models, checkpoints=checkpoints, learner_player=learner_player,
                opponent_pool="fixed-diagnostic-v1", interface=contract.TOWN_ENVIRONMENT,
                purpose="diagnostic", origin="native", policy_seeds=policy_seeds)
            env = HeadlessSelfPlayEnv(
                VcmiLauncher(args.binary, args.data, args.game_map), output,
                limits=Limits(max_days=args.max_days, max_actions=args.max_requests + 1),
                ledger=True, replay=True, interface=contract.TOWN_ENVIRONMENT,
                agents={color: {"checkpoint": checkpoint, "kind": "fixed-untrained-recurrent"}
                        for color, checkpoint in zip(COLORS, checkpoints)})
            attempt = {"episode": episode, "engine_seed": seed, "learner_player": learner_player,
                       "opponent_checkpoint": opponent_id, "policy_seeds": list(policy_seeds)}
            try:
                collected = collector.run_episode(env, episode=episode, seed=seed, max_requests=args.max_requests)
                folder = env.episode_dir
                histories = tuple(env.history(color) for color in COLORS)
                manifest_path = folder / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                independent = audit(folder / "agents/red/history-v3.jsonl",
                                    folder / "agents/blue/history-v3.jsonl", manifest_path)
                write_json(folder / "neural-rollout.json", collected.to_dict())
                write_json(folder / "neural-audit.json", independent)
                if not independent["consistent"] or not independent["manifest_bound"]:
                    raise ValueError("Independent player-history audit failed")
                if weights_sha256(learner) != learner_digest:
                    raise ValueError("Learner weights changed during fixed-weight collection")
                pool.assert_unchanged()
                if source_identity()["source_sha256"] != source["source_sha256"]:
                    raise ValueError("Learner sources changed during fixed-weight collection")
                attempt.update(status=manifest["status"], termination_reason=manifest["termination_reason"],
                               outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                               coverage=history_coverage(histories), audit_consistent=True,
                               weights_unchanged=True, training_samples=0,
                               files={str(p.relative_to(output)): digest(p) for p in (
                                   manifest_path, folder / "neural-rollout.json", folder / "neural-audit.json",
                                   folder / "agents/red/history-v3.jsonl", folder / "agents/blue/history-v3.jsonl")})
            except Exception as error:
                attempt.update(status="collection_error", error=f"{type(error).__name__}: {error}",
                               outcomes=None, rewards=None, training_samples=0)
            finally:
                env.close()
            report["attempts"].append(attempt)
            report["summary"] = diagnostic_summary(report["attempts"])
            write_json(output / "report.json", report)
            print(json.dumps(attempt, sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--map", dest="game_map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--init-seed", type=int, default=95001)
    parser.add_argument("--opponent-seed", dest="opponent_seeds", type=int, action="append")
    parser.add_argument("--engine-seed", dest="engine_seeds", type=int, action="append", required=True)
    parser.add_argument("--policy-seed", type=int, default=95003)
    parser.add_argument("--opponent-sampling-seed", type=int, default=95004)
    parser.add_argument("--max-requests", type=int, default=128)
    parser.add_argument("--max-days", type=int, default=40)
    args = parser.parse_args(argv)
    args.opponent_seeds = args.opponent_seeds or [95002]
    if not 1 <= args.max_requests <= 2048 or not 1 <= args.max_days <= 100:
        parser.error("Diagnostic bounds require 1..2048 requests and 1..100 days")
    if not 1 <= len(args.engine_seeds) <= 4 or any(not 1 <= n <= 2147483647 for n in args.engine_seeds):
        parser.error("Supply 1..4 positive signed-32-bit engine seeds")
    sampling_seeds = [args.init_seed, args.policy_seed, args.opponent_sampling_seed, *args.opponent_seeds]
    if any(not 0 <= seed < 2**63 - 32 for seed in sampling_seeds):
        parser.error("Initialization and sampling seeds must be nonnegative signed-63-bit integers")
    if len(set(args.opponent_seeds)) != len(args.opponent_seeds):
        parser.error("Frozen opponent initialization seeds must be distinct")
    report = run(args)
    return int(any(item["status"] in ("collection_error", "failed") for item in report["attempts"]))


if __name__ == "__main__":
    raise SystemExit(main())
