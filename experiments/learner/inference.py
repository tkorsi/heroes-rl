"""Bounded native collection with fixed random weights; no optimizer entry point.

Collection is always diagnostic and excluded from optimization. An explicitly
declared ordinary-evaluation cohort measures bounded outcome frequency on the
unchanged baseline. It retains every attempt and never scores cutoffs or errors
as draws. Raw histories/checkpoints stay in the ignored run folder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, model_schema_for, weights_sha256
from experiments.learner.contextual_rollout import ContextualCollector
from validation.probes.audit_contextual_histories import audit
from validation.probes.create_boundary_fixture import BASELINE, UPSTREAM, source_contents

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("red", "blue")
BASELINE_SHA256 = "6fc3f50ac769ed6030a21e6ed7c4ba3b0a1341b3d16e9ec72e8a330ba91db58d"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(interface=contract.TOWN_ENVIRONMENT):
    paths = sorted((ROOT / "experiments/learner").glob("*.py"))
    paths += [ROOT / "dependencies/learner.lock"]
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "torch": str(torch.__version__), "python": sys.version,
        "interface": interface,
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


def ordinary_map_identity(path):
    """Refuse fixtures, changed authored maps and engine pins before launch."""
    path = Path(path).expanduser().resolve()
    if digest(BASELINE) != BASELINE_SHA256 or digest(path) != BASELINE_SHA256:
        raise ValueError("Ordinary evaluation requires the exact unchanged canonical baseline map")
    source_contents(path)  # independently check all decoded entries and the native format pin
    return {"path": str(path), "sha256": BASELINE_SHA256, "authored_seed": 0,
            "canonical_path": str(BASELINE), "semantics_verified": True, "fixture": False,
            "upstream_commit": UPSTREAM}


def attempt_roster(plan, *, learner_id, learner_digest, pool, policy_seed, max_requests, limits):
    """The complete schedule, including all RNGs and budgets, before any run."""
    rows = []
    for seed, opponent_id in plan:
        opponent = pool.get(opponent_id)
        for learner_player in (0, 1):
            episode = len(rows) + 1
            rows.append({"episode": episode, "engine_seed": seed, "learner_player": learner_player,
                         "learner_checkpoint": learner_id, "learner_weights_sha256": learner_digest,
                         "opponent_checkpoint": opponent_id, "opponent_weights_sha256": opponent.weights_sha256,
                         "policy_seeds": [policy_seed + 2 * episode, policy_seed + 2 * episode + 1],
                         "budgets": {"max_requests": max_requests, "environment_limits": asdict(limits)}})
    return rows


def attempt_category(attempt):
    """Classify only audited native outcomes; a collection error never scores."""
    status = attempt.get("status")
    if status == "running":
        return "in_progress"
    if status == "collection_error":
        return "collection_or_audit_error"
    if attempt.get("audit_consistent") is not True or attempt.get("weights_unchanged") is not True:
        return "collection_or_audit_error"
    outcomes, rewards = attempt.get("outcomes"), attempt.get("rewards")
    if status == "completed":
        if (outcomes not in ({"red": "win", "blue": "loss"}, {"red": "loss", "blue": "win"})
                or type(rewards) is not dict or set(rewards) != set(COLORS)
                or any(type(rewards[color]) is not int
                       or rewards[color] != (1 if outcomes[color] == "win" else -1) for color in COLORS)):
            return "collection_or_audit_error"
        return "verified_natural"
    if outcomes != dict.fromkeys(COLORS) or rewards != dict.fromkeys(COLORS):
        return "collection_or_audit_error"
    if status == "truncated":
        return "verified_cutoff"
    if status == "failed":
        return "unsupported_interface" if attempt.get("termination_reason") == "unsupported_interface" else "technical_failure"
    return "collection_or_audit_error"


def ordinary_summary(attempts, *, planned_attempts):
    """A bounded, paired-seat frequency report with every attempt accounted for."""
    categories = ("verified_natural", "verified_cutoff", "unsupported_interface",
                  "technical_failure", "collection_or_audit_error", "in_progress")
    counts = Counter(attempt_category(attempt) for attempt in attempts)
    completed = [attempt for attempt in attempts if attempt_category(attempt) == "verified_natural"]
    wins = sum(attempt["outcomes"][COLORS[attempt["learner_player"]]] == "win" for attempt in completed)
    seats = {}
    for seat, color in enumerate(COLORS):
        rows = [attempt for attempt in attempts if attempt["learner_player"] == seat]
        natural = [attempt for attempt in completed if attempt["learner_player"] == seat]
        seats[color] = {"attempts": len(rows), **{key: sum(attempt_category(row) == key for row in rows) for key in categories},
                        "learner_wins": sum(row["outcomes"][color] == "win" for row in natural),
                        "learner_losses": sum(row["outcomes"][color] == "loss" for row in natural)}
    return {
        "planned_attempts": planned_attempts, "attempts": len(attempts),
        "not_started": planned_attempts - len(attempts), **{key: counts[key] for key in categories},
        "finished_attempts": len(attempts) - counts["in_progress"],
        "natural_outcome_frequency": {"numerator": len(completed), "denominator": len(attempts),
                                      "value": len(completed) / len(attempts) if attempts else None},
        "learner_wins_among_completed": {"numerator": wins, "denominator": len(completed),
                                         "value": wins / len(completed) if completed else None},
        "learner_losses": len(completed) - wins, "by_learner_seat": seats,
        "outcomes_by_player": {color: {outcome: sum(row["outcomes"][color] == outcome for row in completed)
                                        for outcome in ("win", "loss")} for color in COLORS},
        "termination_reasons": dict(Counter(row.get("termination_reason", row["status"]) for row in attempts)),
        "training_samples": 0, "optimizer_updates": 0,
        "limitations": ["Small bounded fixed-weight cohort; no population win-rate or learning claim.",
                        "Seats share engine seeds and opponents; attempts are paired, not independent seed samples.",
                        "Cutoffs and failures are unscored, never draws; they remain in the all-attempt denominator.",
                        "Completed-only outcome fractions are conditional on reaching a natural end within the budgets.",
                        "Replay verification is separate from collection; diagnostics remain excluded from PPO."],
    }


def run(args):
    torch.set_num_threads(1)
    interface = getattr(args, "interface", contract.TOWN_ENVIRONMENT)
    profile = contract.profile_for(interface)
    if profile.environment != interface:
        raise ValueError("Inference requires an explicit environment profile")
    config = ModelConfig(schema=model_schema_for(interface))
    purpose = getattr(args, "purpose", "diagnostic")
    if purpose not in ("diagnostic", "ordinary-evaluation"):
        raise ValueError("Unsupported cohort purpose")
    ordinary = purpose == "ordinary-evaluation"
    map_identity = ordinary_map_identity(args.game_map) if ordinary else None
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity(interface)
    if args.checkpoint:
        loaded = load_policy_checkpoint(args.checkpoint, allowed_provenance=("random_initialization",), interface=interface)
        learner, learner_id = loaded.model, loaded.checkpoint
    else:
        torch.manual_seed(args.init_seed)
        learner = ContextualActorCritic(config)
        learner_id = "untrained-" + weights_sha256(learner)[:16]
    learner.eval()
    learner_digest = weights_sha256(learner)
    pool = FrozenOpponentPool()
    for seed in args.opponent_seeds:
        torch.manual_seed(seed)
        opponent = ContextualActorCritic(config)
        pool.add("untrained-" + weights_sha256(opponent)[:16], opponent)
    opponent_rng = random.Random(args.opponent_sampling_seed)
    # The same fixed opponent is used for both colors of each engine seed.
    plan = [(seed, pool.sample(opponent_rng).checkpoint) for seed in args.engine_seeds]
    save_policy_checkpoint(
        output / "untrained-policy.pt", learner, checkpoint=learner_id,
        provenance="random_initialization", source_metadata=source,
        python_generators={"opponent": opponent_rng}, opponent_pool=pool, interface=interface)
    limits = Limits(max_days=args.max_days, max_actions=args.max_requests + 1)
    roster = attempt_roster(plan, learner_id=learner_id, learner_digest=learner_digest, pool=pool,
                           policy_seed=args.policy_seed, max_requests=args.max_requests, limits=limits)
    report = {
        "schema": "heroes-contextual-neural-ordinary-evaluation-v1" if ordinary else "heroes-contextual-neural-diagnostic-v1",
        "purpose": purpose,
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
        "attempts": [], "contextual_replay": "not_verified_by_collector",
    }
    if ordinary:
        planned = {"schema": "heroes-contextual-ordinary-evaluation-plan-v1", "purpose": purpose,
                   "collector_purpose": "diagnostic", "training_performed": False, "source": source,
                   "baseline": map_identity, "attempts": roster,
                   "policy_checkpoint_file_sha256": digest(output / "untrained-policy.pt"),
                   "native_identity": VcmiLauncher(args.binary, args.data, args.game_map).identity()}
        write_json(output / "cohort-plan.json", planned)
        report["cohort_plan"] = {"path": "cohort-plan.json", "sha256": digest(output / "cohort-plan.json")}
        report["collector_purpose"] = "diagnostic"
        report["summary"] = ordinary_summary([], planned_attempts=len(roster))
    write_json(output / "report.json", report)
    episode = 0
    for seed, opponent_id in plan:
        for learner_player in (0, 1):
            episode += 1
            checkpoints = (learner_id, opponent_id) if learner_player == 0 else (opponent_id, learner_id)
            policy_seeds = (args.policy_seed + 2 * episode, args.policy_seed + 2 * episode + 1)
            attempt = {"episode": episode, "engine_seed": seed, "learner_player": learner_player,
                       "opponent_checkpoint": opponent_id, "policy_seeds": list(policy_seeds)}
            if ordinary:
                attempt.update(roster[episode - 1], status="running")
                report["attempts"].append(attempt)
                report["summary"] = ordinary_summary(report["attempts"], planned_attempts=len(roster))
                write_json(output / "report.json", report)
            env = None
            try:
                opponent = pool.get(opponent_id)
                models = (learner, opponent.model) if learner_player == 0 else (opponent.model, learner)
                if ordinary:
                    ordinary_map_identity(args.game_map)
                    if digest(output / "cohort-plan.json") != report["cohort_plan"]["sha256"]:
                        raise ValueError("Predeclared cohort plan changed")
                    if source_identity(interface)["source_sha256"] != source["source_sha256"]:
                        raise ValueError("Learner sources changed before fixed-weight collection")
                collector = ContextualCollector(
                    models=models, checkpoints=checkpoints, learner_player=learner_player,
                    opponent_pool="fixed-diagnostic-v1", interface=interface,
                    purpose="diagnostic", origin="native", policy_seeds=policy_seeds)
                env = HeadlessSelfPlayEnv(
                    VcmiLauncher(args.binary, args.data, args.game_map), output, limits=limits,
                    ledger=True, replay=True, interface=interface,
                    agents={color: {"checkpoint": checkpoint, "kind": "fixed-untrained-recurrent"}
                            for color, checkpoint in zip(COLORS, checkpoints)})
                collected = collector.run_episode(env, episode=episode, seed=seed, max_requests=args.max_requests)
                folder = env.episode_dir
                histories = tuple(env.history(color) for color in COLORS)
                manifest_path = folder / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                independent = audit(folder / "agents/red" / profile.history_filename,
                                    folder / "agents/blue" / profile.history_filename, manifest_path)
                write_json(folder / "neural-rollout.json", collected.to_dict())
                write_json(folder / "neural-audit.json", independent)
                if not independent["consistent"] or not independent["manifest_bound"]:
                    raise ValueError("Independent player-history audit failed")
                if weights_sha256(learner) != learner_digest:
                    raise ValueError("Learner weights changed during fixed-weight collection")
                pool.assert_unchanged()
                if source_identity()["source_sha256"] != source["source_sha256"]:
                    raise ValueError("Learner sources changed during fixed-weight collection")
                if ordinary and (manifest["identity"]["map_sha256"] != BASELINE_SHA256
                                 or ordinary_map_identity(args.game_map) != map_identity):
                    raise ValueError("Ordinary evaluation map identity changed")
                if ordinary and any(manifest["identity"].get(key) != planned["native_identity"][key]
                                    for key in ("binary_sha256", "explorer_rl_sha256")):
                    raise ValueError("Native binaries differ from the predeclared cohort plan")
                attempt.update(status=manifest["status"], termination_reason=manifest["termination_reason"],
                               outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                               coverage=history_coverage(histories), audit_consistent=True,
                               weights_unchanged=True, training_samples=0,
                               files={str(p.relative_to(output)): digest(p) for p in (
                                   manifest_path, folder / "neural-rollout.json", folder / "neural-audit.json",
                                   folder / "agents/red" / profile.history_filename,
                                   folder / "agents/blue" / profile.history_filename)})
            except Exception as error:
                attempt.update(status="collection_error", error=f"{type(error).__name__}: {error}",
                               outcomes=None, rewards=None, training_samples=0)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        attempt.update(status="collection_error", cleanup_error=f"{type(error).__name__}: {error}",
                                       outcomes=None, rewards=None, training_samples=0)
            if ordinary:
                attempt["category"] = attempt_category(attempt)
                report["summary"] = ordinary_summary(report["attempts"], planned_attempts=len(roster))
            else:
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
    parser.add_argument("--interface", choices=(contract.ENVIRONMENT, contract.TOWN_ENVIRONMENT, contract.ARMIES_ENVIRONMENT),
                        default=contract.TOWN_ENVIRONMENT)
    parser.add_argument("--purpose", choices=("diagnostic", "ordinary-evaluation"), default="diagnostic")
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
    if args.purpose == "ordinary-evaluation" and len(set(args.engine_seeds)) != len(args.engine_seeds):
        parser.error("Ordinary evaluation requires distinct engine seeds, each evaluated in both seats")
    sampling_seeds = [args.init_seed, args.policy_seed, args.opponent_sampling_seed, *args.opponent_seeds]
    if any(not 0 <= seed < 2**63 - 32 for seed in sampling_seeds):
        parser.error("Initialization and sampling seeds must be nonnegative signed-63-bit integers")
    if len(set(args.opponent_seeds)) != len(args.opponent_seeds):
        parser.error("Frozen opponent initialization seeds must be distinct")
    report = run(args)
    return int(any(item["status"] in ("collection_error", "failed")
                   or item.get("category") == "collection_or_audit_error" for item in report["attempts"]))


if __name__ == "__main__":
    raise SystemExit(main())
