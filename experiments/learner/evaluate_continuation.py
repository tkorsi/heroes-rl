"""Finish only the four explicitly authorized held-out evaluations, without PPO.

The old first-run execution plan remains stale and is never passed to its
executor. A new immutable plan binds current code to the saved stopped run.
Optimizer state is retained in the original checkpoint but is never instantiated
or applied here. Every collected trajectory is diagnostic and ineligible for PPO.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import HeadlessSelfPlayEnv, Limits, VcmiLauncher
from experiments.learner import contextual as c
from experiments.learner.checkpoints import FrozenOpponentPool, _model_from_payload, _read
from experiments.learner.contextual_model import encode_history, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector
from experiments.learner.first_run import ROOT, canonical, file_identity, implementation_identity, schedule, validate_config
from experiments.learner.gameplay import BudgetExpired, assert_same_state, check_identity, require, verify_collected
from experiments.learner.inference import digest, history_coverage, ordinary_map_identity, source_identity, write_json
from experiments.learner.ppo import _verify_behavior

PLAN_SCHEMA = "heroes-after-evaluation-plan-v1"
RESULT_SCHEMA = "heroes-after-evaluation-result-v1"
ORIGINAL_PLAN_SHA256 = "ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6"
ORIGINAL_REPORT_SHA256 = "bf6ceac81894310b6a364d6dcc755087b16a21409133398d83db42586f3007b5"
RESUME_SHA256 = "df8834bac83c7444ceff5396f06015ab9f14b7f90aea4dd62c1b7a6b3bb2f089"
LEARNER_SHA256 = "f4af57d1e47702200b576d0110b2ede13be1db25fc60976af8f82fc114d7b48c"
OPPONENT = ("random-v4-b1f6bc72889164ab", "b1f6bc72889164abb59f5d312bbf450091d587f2eb70d053f01e457deabd6d3f")
PROGRESS = {"updates": 1, "rollouts": 2, "decisions": 294}
ORIGINAL_WALL_SECONDS = 164.8517872090015
REMAINING_SECONDS = 3600 - ORIGINAL_WALL_SECONDS
COLORS = ("red", "blue")
POOL = "first-training-frozen-v1"


def fixed_schedule():
    return [{"episode": 7 + index, "phase": "after", "engine_seed": seed, "learner_player": seat,
             "policy_seeds": [101003 + 2 * index, 101004 + 2 * index], "optimizer_eligible": False}
            for index, (seed, seat) in enumerate(((99201, 0), (99201, 1), (99202, 0), (99202, 1)))]


def binding(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), **file_identity(path)}


def check_binding(item):
    require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}
            and Path(item["path"]).is_absolute(), "Artifact needs an exact absolute identity")
    require(binding(item["path"]) == item, "Bound artifact changed: " + item["path"])


def check_authorization(item):
    check_binding(item)
    authorization = json.loads(Path(item["path"]).read_text())
    require(authorization.get("schema") == "heroes-after-evaluation-authorization-v1"
            and authorization.get("approved") is True
            and authorization.get("allowed_episodes") == [7, 8, 9, 10]
            and type(authorization.get("additional_optimizer_steps")) is int
            and authorization["additional_optimizer_steps"] == 0,
            "Explicit four-game evaluation-only authorization is required")


def check_original(plan):
    """Validate immutable evidence, not authorization to rerun its old executor."""
    for name, expected in (("original_plan", ORIGINAL_PLAN_SHA256),
                           ("original_report", ORIGINAL_REPORT_SHA256), ("resume_checkpoint", RESUME_SHA256)):
        check_binding(plan[name])
        require(plan[name]["sha256"] == expected, "Continuation refers to a different stopped run: " + name)
    original = json.loads(Path(plan["original_plan"]["path"]).read_text())
    report = json.loads(Path(plan["original_report"]["path"]).read_text())
    cfg = validate_config(original["configuration"])
    require(original["schema"] == "heroes-first-training-plan-v1"
            and original["attempt_schedule"] == schedule(cfg), "Original schedule changed")
    require(report["schema"] == "heroes-first-training-result-v1"
            and report["plan_sha256"] == ORIGINAL_PLAN_SHA256 and report["status"] == "stopped"
            and report["updates"] == 1 and report["optimizer_steps"] == 10
            and report["wall_seconds"] == ORIGINAL_WALL_SECONDS
            and len(report["attempts"]) == 6, "Original execution does not have the expected stopped boundary")
    for episode, (attempt, expected) in enumerate(zip(report["attempts"], schedule(cfg)), start=1):
        require(attempt["episode"] == episode and attempt["status"] in ("completed", "truncated")
                and all(attempt[key] == expected[key] for key in ("phase", "engine_seed", "learner_player", "policy_seeds")),
                "Original completed attempts differ from their declared schedule")
    initial = original["initial_checkpoint"]
    require(file_identity(initial["path"]) == {key: initial[key] for key in ("sha256", "bytes")},
            "Original initial checkpoint changed")
    require(initial["opponents"] == [list(OPPONENT)], "Original frozen opponent differs")
    require(plan["remaining_wall_seconds"] == cfg["total_wall_seconds"] - report["wall_seconds"]
            == REMAINING_SECONDS, "Continuation cannot reset or expand the original wall-time budget")
    return original


@dataclass(frozen=True)
class FixedCheckpoint:
    model: object
    checkpoint: str
    opponent_pool: FrozenOpponentPool


def decode_fixed_checkpoint(path, original):
    """Decode saved tensors without creating an optimizer or restoring any RNG.

    Explicit per-player evaluation seeds govern sampling. Model construction in
    the checkpoint loader preserves the caller's RNG and loads every parameter.
    """
    metadata = {"plan_sha256": ORIGINAL_PLAN_SHA256, "source": original["source"],
                "configuration": original["configuration"]}
    payload = _read(path, "optimizer", metadata, ("approved_gameplay",), c.ARMIES_ENVIRONMENT)
    resume = payload["resume"]
    require(type(resume) is dict and resume.get("at_rollout_boundary") is True
            and resume.get("progress") == PROGRESS
            and resume.get("sampler_state") == {"next_phase": "after", "next_episode": 7}
            and resume.get("ppo_config") == original["configuration"]["ppo"],
            "Saved checkpoint does not begin the missing after-evaluation phase")
    require(payload["weights_sha256"] == LEARNER_SHA256
            and payload["checkpoint"] == "first-update-" + LEARNER_SHA256[:16],
            "Saved learner is not the authorized post-update checkpoint")
    model = _model_from_payload(payload).eval().requires_grad_(False)
    # Exercise the repaired recursive comparator on the real OrderedDict state,
    # including a second independent load, before any native episode is started.
    reloaded = _model_from_payload(payload).eval().requires_grad_(False)
    assert_same_state(model.state_dict(), reloaded.state_dict())
    for name, value in model.state_dict().items():
        require(torch.equal(value, payload["model_state"][name]), "Saved model tensor differs after loading")
    pool = FrozenOpponentPool.from_state_dict(payload["opponent_pool"])
    require(pool.identities == (OPPONENT,), "Saved frozen opponent differs from the original evaluation")
    return FixedCheckpoint(model, payload["checkpoint"], pool)


def check_fixed(loaded):
    require(loaded.checkpoint == "first-update-" + LEARNER_SHA256[:16]
            and weights_sha256(loaded.model) == LEARNER_SHA256
            and not loaded.model.training and all(not p.requires_grad for p in loaded.model.parameters()),
            "Evaluation learner weights or inference mode changed")
    loaded.opponent_pool.assert_unchanged()
    require(loaded.opponent_pool.identities == (OPPONENT,), "Evaluation opponent identity changed")


def validate_plan(plan):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("purpose") == "evaluation_only"
            and canonical(plan.get("attempt_schedule")) == canonical(fixed_schedule())
            and type(plan.get("additional_optimizer_steps")) is int and plan["additional_optimizer_steps"] == 0
            and plan.get("remaining_wall_seconds") == REMAINING_SECONDS
            and canonical(plan.get("limits")) == canonical({"max_requests": 256, "max_days": 30, "torch_threads": 1}),
            "Continuation scope must be exactly the four missing fixed-weight evaluations")
    require(Path(plan["execution_output"]).is_absolute(), "Continuation output must be explicit")
    original = check_original(plan)
    check_authorization(plan["authorization"])
    for key in ("native_identity", "native_runtime", "map_identity", "binary", "replay_binary", "data", "game_map"):
        require(plan[key] == original[key], "Continuation changes the original native environment: " + key)
    old_folder = Path(plan["original_report"]["path"]).parent
    require(not Path(plan["execution_output"]).is_relative_to(old_folder.parent),
            "Continuation cannot write inside the original prepared or executed run")
    return original


def prepare(*, original_plan, original_report, resume_checkpoint, authorization, output):
    output = Path(output).expanduser().resolve()
    require(not output.exists(), "Continuation preparation output already exists")
    original = json.loads(Path(original_plan).read_text())
    plan = {"schema": PLAN_SCHEMA, "purpose": "evaluation_only", "additional_optimizer_steps": 0,
            "authorization": binding(authorization), "original_plan": binding(original_plan),
            "original_report": binding(original_report), "resume_checkpoint": binding(resume_checkpoint),
            "remaining_wall_seconds": REMAINING_SECONDS, "original_wall_seconds": ORIGINAL_WALL_SECONDS,
            "limits": {"max_requests": 256, "max_days": 30, "torch_threads": 1},
            "attempt_schedule": fixed_schedule(), "execution_output": str(output / "execution"),
            "learner_weights_sha256": LEARNER_SHA256, "frozen_opponent": list(OPPONENT),
            **{key: original[key] for key in ("binary", "data", "game_map", "replay_binary", "native_identity",
                                             "native_runtime", "map_identity")}}
    validate_plan(plan)
    torch.set_num_threads(1)
    loaded = decode_fixed_checkpoint(plan["resume_checkpoint"]["path"], original)
    check_fixed(loaded)
    launcher = VcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]))
    launcher.preflight()
    require(launcher.identity() == plan["native_identity"]
            and runtime_identity(Path(plan["binary"]).parent) == plan["native_runtime"]
            and ordinary_map_identity(Path(plan["game_map"])) == plan["map_identity"],
            "Original native environment or map changed")
    checked = subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
                             cwd=ROOT, text=True, capture_output=True)
    require(checked.returncode == 0, "Prepared native sources differ from their authored recipe: " + checked.stderr)
    plan.update(source=source_identity(c.ARMIES_ENVIRONMENT), implementation_sha256=implementation_identity(),
                prepared_check={"returncode": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr})
    check_identity(plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    return plan


def read_plan(path, approved_sha256):
    path = Path(path).expanduser().resolve()
    require(type(approved_sha256) is str and len(approved_sha256) == 64
            and digest(path) == approved_sha256, "Execution requires the exact new continuation plan SHA-256")
    plan = json.loads(path.read_text())
    validate_plan(plan)
    require(Path(plan["execution_output"]) == path.parent / "execution", "Execution must use the bound new output folder")
    check_identity(plan)
    return plan


def check_integrity(plan, loaded):
    check_original(plan)
    check_authorization(plan["authorization"])
    launcher = check_identity(plan)
    check_fixed(loaded)
    return launcher


def validate_collected(collected, spec, loaded):
    require(len(collected.rollouts) == 2, "Evaluation needs both player trajectories")
    count = sum(len(item.decisions) for item in collected.rollouts)
    require(count <= 256, "Evaluation exceeds its original sampled-choice budget")
    for player, item in enumerate(collected.rollouts):
        learner = player == spec["learner_player"]
        expected = (loaded.checkpoint, LEARNER_SHA256) if learner else OPPONENT
        other = OPPONENT if learner else (loaded.checkpoint, LEARNER_SHA256)
        require(item.player == player and item.episode == str(spec["episode"])
                and item.role == ("learner" if learner else "frozen_opponent")
                and (item.behavior_checkpoint, item.weights_sha256) == expected
                and (item.opponent_checkpoint, item.opponent_weights_sha256) == other
                and item.opponent_pool == POOL and item.purpose == "diagnostic" and item.origin == "native"
                and item.optimization_eligible is False, "Evaluation provenance, weights or exclusion from PPO differs")
        require(all(item.history[decision.decision_seq].day <= 30 for decision in item.decisions),
                "Evaluation sampled an action beyond the original day budget")
    return count


def collect_attempt(plan, output, spec, loaded, remaining):
    launcher = check_integrity(plan, loaded)
    seat = spec["learner_player"]
    opponent = loaded.opponent_pool.get(OPPONENT[0])
    models = (loaded.model, opponent.model) if seat == 0 else (opponent.model, loaded.model)
    identities = (loaded.checkpoint, OPPONENT[0]) if seat == 0 else (OPPONENT[0], loaded.checkpoint)
    collector = ContextualCollector(models, identities, learner_player=seat, opponent_pool=POOL,
        interface=c.ARMIES_ENVIRONMENT, purpose="diagnostic", origin="native", policy_seeds=tuple(spec["policy_seeds"]))
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=30, max_actions=257), ledger=True, replay=True,
        interface=c.ARMIES_ENVIRONMENT,
        agents={color: {"checkpoint": identity, "kind": "recurrent-after-evaluation", "phase": "after"}
                for color, identity in zip(COLORS, identities)})
    try:
        remaining()
        collected = collector.run_episode(env, episode=spec["episode"], seed=spec["engine_seed"], max_requests=256)
        folder = env.episode_dir
        write_json(folder / "neural-rollout.json", collected.to_dict())
        # Reconstruct the immutable trajectories from their serialized records.
        persisted = CollectedEpisode.from_dict(json.loads((folder / "neural-rollout.json").read_text()))
        require(persisted.to_dict() == collected.to_dict(), "Persisted evaluation rollout differs")
        count = validate_collected(persisted, spec, loaded)
        manifest, _ = verify_collected(folder, persisted, plan, replay_name="collection-replay", remaining=remaining)
        require(manifest["episode"] == str(spec["episode"]) and manifest["seed"] == spec["engine_seed"],
                "Native episode differs from the original paired evaluation seed")
        remaining()
        for policy, trajectory in zip(models, persisted.rollouts):
            _verify_behavior(policy, (trajectory,), (encode_history(trajectory.history),))
            remaining()
        check_integrity(plan, loaded)
        return {"status": manifest["status"], "termination_reason": manifest["termination_reason"],
                "outcomes": manifest["outcomes_by_player"], "rewards": manifest["rewards_by_player"],
                "coverage": history_coverage(tuple(env.history(color) for color in COLORS)),
                "replay_verified": True, "behavior_recomputed": True, "sampled_choices": count,
                "weights_sha256": LEARNER_SHA256, "opponent_weights_sha256": OPPONENT[1],
                "folder": str(folder.resolve()), "purpose": "diagnostic", "origin": "native",
                "optimization_eligible": False}
    finally:
        # A cleanup exception propagates into this attempt's persisted error;
        # no replacement or subsequent game is launched after any failure.
        env.close()


def execute(plan_path, approved_sha256):
    plan = read_plan(plan_path, approved_sha256)
    output = Path(plan["execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + plan["remaining_wall_seconds"]

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise BudgetExpired("Original shared wall-time budget expired")
        return seconds

    def alarm(signum, frame):
        raise BudgetExpired("Original shared wall-time budget expired")

    report = {"schema": RESULT_SCHEMA, "plan_sha256": approved_sha256, "status": "running", "attempts": [],
              "training_performed": False, "updates": 0, "optimizer_steps": 0, "additional_optimizer_steps": 0,
              "source": plan["source"], "original_report": plan["original_report"],
              "resume_checkpoint": plan["resume_checkpoint"], "saved_progress": dict(PROGRESS),
              "original_optimizer_steps": 10, "remaining_wall_seconds": plan["remaining_wall_seconds"]}
    write_json(output / "report.json", report)
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    loaded = None
    try:
        original = check_original(plan)
        loaded = decode_fixed_checkpoint(plan["resume_checkpoint"]["path"], original)
        check_integrity(plan, loaded)
        report["resume_verified"] = True
        for spec in plan["attempt_schedule"]:
            remaining()
            row = {**deepcopy(spec), "status": "running"}
            report["attempts"].append(row)
            write_json(output / "report.json", report)
            try:
                row.update(collect_attempt(plan, output, spec, loaded, remaining))
                remaining()
            except BaseException as error:
                row.update(status="collection_error", error=f"{type(error).__name__}: {error}",
                           outcomes=None, rewards=None)
                raise
            finally:
                write_json(output / "report.json", report)
        check_integrity(plan, loaded)
        remaining()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        try:
            check_original(plan)
            if loaded is not None:
                check_fixed(loaded)
            remaining()
            report["original_artifacts_unchanged"] = True
            report["fixed_weights_unchanged"] = loaded is not None
        except BaseException as error:
            report.update(status="stopped", final_integrity_error=f"{type(error).__name__}: {error}")
        report["wall_seconds"] = time.monotonic() - started
        report["combined_wall_seconds"] = ORIGINAL_WALL_SECONDS + report["wall_seconds"]
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Bind current source to the saved checkpoint; no games")
    for name in ("original-plan", "original-report", "resume-checkpoint", "authorization", "output"):
        preparation.add_argument("--" + name, type=Path, required=True)
    execution = commands.add_parser("execute", help="Only the four authorized evaluation games; no optimizer")
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(**{key: value for key, value in vars(args).items() if key != "command"})
        print(json.dumps({"plan": str(args.output.resolve() / "plan.json"),
                          "sha256": digest(args.output / "plan.json"), "schedule": plan["attempt_schedule"]}, indent=2))
        return 0
    result = execute(args.plan, args.approved_plan_sha256)
    print(json.dumps(result, indent=2))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
