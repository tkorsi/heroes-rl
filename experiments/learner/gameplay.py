"""Prepared, explicitly invoked first gameplay update; never a default command.

Execution requires the exact SHA-256 of a reviewed preparation plan. The human
must authorize that launch separately; creating a plan does not grant approval.
The default PPO entry point and every diagnostic collector remain ineligible.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import signal
import time

import torch

from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.contextual_replay import verify_episode
from experiments.headless.env import HeadlessSelfPlayEnv, Limits, VcmiLauncher
from experiments.learner import contextual as c
from experiments.learner.checkpoints import (TrainingProgress, load_checkpoint, load_policy_checkpoint,
    save_checkpoint, save_policy_checkpoint)
from experiments.learner.contextual_model import ContextualActorCritic, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector
from experiments.learner.inference import digest, history_coverage, ordinary_map_identity, source_identity, write_json
from experiments.learner.ppo import PPOConfig, _admit, _update, _verify_behavior
from validation.probes.audit_contextual_histories import audit

COLORS = ("red", "blue")
PROVENANCE = "approved_gameplay"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class BudgetExpired(RuntimeError):
    """A technical stop, never a game outcome or bootstrap boundary."""


def paired_schedule(seeds):
    return tuple((seed, seat) for seed in seeds for seat in (0, 1))


@dataclass(frozen=True)
class CollectionReceipt:
    """Held in memory from this run's fresh capture through optimizer admission.

    The persisted copy is evidence only: admission never reloads a receipt supplied
    alongside an arbitrary artifact, so relabelling an old diagnostic cannot make
    it one of the captures selected by the running collector.
    """
    folder: str
    episode: str
    engine_seed: int
    learner_player: int
    policy_seeds: tuple[int, int]
    artifact_sha256: str
    manifest_sha256: str
    learner_checkpoint: str
    learner_weights_sha256: str
    opponent_checkpoint: str
    opponent_weights_sha256: str
    opponent_pool: str

    def __post_init__(self):
        require(type(self.folder) is str and Path(self.folder).is_absolute(), "Receipt needs its exact absolute capture folder")
        require(type(self.episode) is str and self.episode.isdigit() and int(self.episode) > 0, "Receipt episode must be explicit")
        require(type(self.engine_seed) is int and 1 <= self.engine_seed <= 2147483647, "Invalid receipt engine seed")
        require(type(self.learner_player) is int and self.learner_player in (0, 1), "Invalid receipt learner seat")
        require(type(self.policy_seeds) is tuple and len(self.policy_seeds) == 2
                and all(type(seed) is int and 0 <= seed < 2**63 for seed in self.policy_seeds), "Invalid receipt policy seeds")
        for value in (self.artifact_sha256, self.manifest_sha256, self.learner_weights_sha256, self.opponent_weights_sha256):
            require(type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
                    "Receipt needs exact SHA-256 identities")
        for value in (self.learner_checkpoint, self.opponent_checkpoint, self.opponent_pool):
            require(type(value) is str and bool(value.strip()), "Receipt needs fixed policy identities")


def validate_collection_receipt(receipt, folder, collected, *, episode, seed, seat, policy_seeds,
                                learner_checkpoint, learner_digest, opponent_checkpoint, opponent_digest,
                                opponent_pool, max_requests, max_days):
    """Pure artifact/provenance checks; this does not establish native validity."""
    require(type(receipt) is CollectionReceipt, "Admission requires this run's immutable collection receipt")
    folder = Path(folder).resolve()
    require((receipt.folder, receipt.episode, receipt.engine_seed, receipt.learner_player, receipt.policy_seeds) ==
            (str(folder), str(episode), seed, seat, policy_seeds), "Collection receipt differs from the exact run schedule")
    require((receipt.learner_checkpoint, receipt.learner_weights_sha256, receipt.opponent_checkpoint,
             receipt.opponent_weights_sha256, receipt.opponent_pool) ==
            (learner_checkpoint, learner_digest, opponent_checkpoint, opponent_digest, opponent_pool),
            "Collection receipt differs from the fixed initial learner/opponent")
    require(digest(folder / "neural-rollout.json") == receipt.artifact_sha256
            and digest(folder / "manifest.json") == receipt.manifest_sha256,
            "Fresh collection artifact changed after capture")
    require(json.loads((folder / "neural-rollout.json").read_text()) == collected.to_dict(),
            "In-memory collection differs from its persisted artifact")
    manifest = json.loads(collected.manifest_json)
    require(manifest == json.loads((folder / "manifest.json").read_text())
            and manifest["episode"] == str(episode) and manifest["seed"] == seed
            and manifest["environment"] == c.ARMIES_ENVIRONMENT, "Collection manifest differs from the run schedule/profile")
    learner, opponent = collected.rollouts[seat], collected.rollouts[1 - seat]
    require((learner.player, learner.role, learner.behavior_checkpoint, learner.weights_sha256,
             opponent.player, opponent.role, opponent.behavior_checkpoint, opponent.weights_sha256) ==
            (seat, "learner", learner_checkpoint, learner_digest,
             1 - seat, "frozen_opponent", opponent_checkpoint, opponent_digest),
            "Collection policy roles or identities differ from the frozen pair")
    require(all(item.episode == str(episode) and item.purpose == "training" and item.origin == "native"
                and item.opponent_pool == opponent_pool for item in collected.rollouts),
            "Only this run's original training-purpose collection can be admitted")
    require(sum(len(item.decisions) for item in collected.rollouts) <= max_requests,
            "Collection exceeds the total sampled-choice budget")
    # A legitimate day cutoff holds an unacted decision on day max_days+1.
    # Only sampled choices are restricted to the allowed gameplay days.
    require(all(item.history[decision.decision_seq].day <= max_days
                for item in collected.rollouts for decision in item.decisions),
            "Collection contains a sampled action beyond the day budget")
    return learner


def read_plan(path, approved_sha256):
    from .first_run import check_readiness, validate_config
    path = Path(path).expanduser().resolve()
    require(type(approved_sha256) is str and len(approved_sha256) == 64
            and digest(path) == approved_sha256, "Execution needs the exact separately approved plan SHA-256")
    plan = json.loads(path.read_text())
    require(plan.get("schema") == "heroes-first-training-plan-v1"
            and plan.get("authorization") == "not_granted", "Expected an immutable preparation plan")
    validate_config(plan["configuration"])
    require(plan.get("readiness", {}).get("ready_to_request_launch") is True,
            "Preparation has unresolved correctness or implementation conditions")
    checkpoint = plan["initial_checkpoint"]
    require(digest(checkpoint["path"]) == checkpoint["sha256"], "Initial checkpoint bytes changed")
    check_identity(plan)
    readiness = check_readiness(Path(plan["readiness"]["evidence"]["path"]), source=plan["source"],
        implementation=plan["implementation_sha256"], native_identity=plan["native_identity"],
        native_runtime=plan["native_runtime"], map_identity=plan["map_identity"])
    require(readiness == plan["readiness"], "Prepared readiness evidence or retained checks changed")
    return plan


def check_identity(plan):
    actual = source_identity(c.ARMIES_ENVIRONMENT)
    for key in ("source_sha256", "torch", "python", "interface"):
        require(actual[key] == plan["source"][key], "Prepared learner source/runtime changed: " + key)
    launcher = VcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]))
    launcher.preflight()
    require(launcher.identity() == plan["native_identity"], "Prepared native client/map identity changed")
    require(runtime_identity(Path(plan["binary"]).parent) == plan["native_runtime"], "Native runtime changed")
    require(ordinary_map_identity(Path(plan["game_map"])) == plan["map_identity"], "Ordinary map changed")
    # Preparation binds the authored recipe and Python adapter, beyond the
    # learner-only source manifest. Refuse edits made after the reviewed plan.
    from .first_run import implementation_identity
    require(implementation_identity() == plan["implementation_sha256"],
            "Prepared implementation inventory or file bytes changed")
    return launcher


def verify_collected(folder, collected, plan, *, replay_name, remaining):
    """Verify every persisted attempt; no failed episode can be silently dropped."""
    profile = c.ARMIES_PROFILE
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest == json.loads(collected.manifest_json), "Persisted episode manifest changed")
    paths = [folder / manifest["players"][color][profile.history_key]["path"] for color in COLORS]
    independent = audit(*paths, folder / "manifest.json")
    write_json(folder / (replay_name + "-history.json"), independent)
    require(independent["consistent"] and independent["manifest_bound"], "Independent history audit failed")
    require(manifest["status"] in ("completed", "truncated"), "Technical/unsupported/collection failure stops the run")
    require(manifest["environment"] == c.ARMIES_ENVIRONMENT
            and manifest["identity"]["map_sha256"] == plan["map_identity"]["sha256"], "Episode profile/map differs")
    replay = verify_episode(folder, binary=plan["replay_binary"], data=plan["data"],
                            output=folder / replay_name, timeout=min(300, remaining()))
    require(replay["verified"] is True, "Native replay failed")
    return manifest, replay


def native_update(model, optimizer, folders, plan, *, receipts, generator, remaining):
    """Only original training-purpose native captures enter this separate gate.

    Independent native replay runs again immediately before admission. Archived
    diagnostics cannot be relabelled by the runner or used as demonstrations.
    """
    from .contextual_model import encode_history
    config = PPOConfig(**plan["configuration"]["ppo"])
    expected = paired_schedule(plan["configuration"]["training_engine_seeds"])
    require(type(receipts) is tuple and len(folders) == len(receipts) == len(expected),
            "Training batch must include every declared fresh capture receipt")
    require(digest(plan["initial_checkpoint"]["path"]) == plan["initial_checkpoint"]["sha256"],
            "Initial checkpoint changed before admission")
    initial = load_policy_checkpoint(plan["initial_checkpoint"]["path"], interface=c.ARMIES_ENVIRONMENT)
    require(initial.opponent_pool is not None and len(initial.opponent_pool.identities) == 1,
            "Admission requires the single prepared frozen opponent")
    opponent_checkpoint, opponent_digest = initial.opponent_pool.identities[0]
    cfg = plan["configuration"]
    first_episode = len(paired_schedule(cfg["evaluation_engine_seeds"])) + 1
    rollouts = []
    for index, (folder, receipt, (seed, seat)) in enumerate(zip(folders, receipts, expected)):
        raw = json.loads((folder / "neural-rollout.json").read_text())
        collected = CollectedEpisode.from_dict(raw)
        learner = validate_collection_receipt(receipt, folder, collected, episode=first_episode + index,
            seed=seed, seat=seat, policy_seeds=(cfg["policy_seed"] + 1000 + 2 * index, cfg["policy_seed"] + 1001 + 2 * index),
            learner_checkpoint=initial.checkpoint, learner_digest=initial.weights_sha256,
            opponent_checkpoint=opponent_checkpoint, opponent_digest=opponent_digest,
            opponent_pool="first-training-frozen-v1", max_requests=cfg["max_requests"], max_days=cfg["max_days"])
        manifest, _ = verify_collected(folder, collected, plan, replay_name="admission-replay", remaining=remaining)
        rollouts.append(learner)
    rollouts = tuple(rollouts)
    check_identity(plan)
    _admit(rollouts, model, native=True)
    # This explicit precheck also runs inside the numerical kernel; neither
    # trusts saved hidden tensors or a serialized behavior-probability claim.
    _verify_behavior(model, rollouts, tuple(encode_history(item.history) for item in rollouts))
    result = _transactional_update(model, optimizer, rollouts, config=config, generator=generator,
                                   remaining=remaining, max_optimizer_steps=cfg["max_optimizer_steps"])
    return replace(result, purpose="approved_gameplay"), rollouts


class UpdateRolledBack(RuntimeError):
    def __init__(self, error, completed_steps):
        super().__init__(f"{type(error).__name__}: {error}; complete behavior/optimizer/RNG state restored")
        self.completed_steps = completed_steps


def _transactional_update(model, optimizer, rollouts, *, config, generator, remaining, max_optimizer_steps):
    """An interrupted update never leaves a partially updated policy/resume state."""
    state, optimizer_state = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    rng, minibatch_rng = torch.get_rng_state(), generator.get_state()
    completed_steps = 0

    def completed():
        nonlocal completed_steps
        completed_steps += 1

    try:
        result = _update(model, optimizer, rollouts, config=config, generator=generator,
                         before_step=remaining, after_step=completed, max_optimizer_steps=max_optimizer_steps)
        remaining()
    except BaseException as error:
        model.load_state_dict(state)
        optimizer.load_state_dict(optimizer_state)
        torch.set_rng_state(rng)
        generator.set_state(minibatch_rng)
        raise UpdateRolledBack(error, completed_steps) from error
    return result


def assert_same_state(left, right):
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and torch.equal(left, right), "Resume tensor differs")
    elif isinstance(left, Mapping):
        require(type(right) is type(left) and left.keys() == right.keys(), "Resume mapping differs")
        for key in left: assert_same_state(left[key], right[key])
    elif type(left) in (tuple, list):
        require(type(right) is type(left) and len(left) == len(right), "Resume sequence differs")
        for a, b in zip(left, right): assert_same_state(a, b)
    else:
        require(type(left) is type(right) and left == right, "Resume scalar differs")


def execute(plan_path, approved_sha256, output):
    plan = read_plan(plan_path, approved_sha256)
    cfg = plan["configuration"]
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg["torch_threads"])
    started = time.monotonic()
    deadline = started + cfg["total_wall_seconds"]

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0: raise BudgetExpired("Global wall-time budget expired")
        return seconds

    def alarm(signum, frame):
        raise BudgetExpired("Global wall-time budget expired")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, cfg["total_wall_seconds"])
    report = {"schema": "heroes-first-training-result-v1", "plan_sha256": approved_sha256,
              "status": "running", "attempts": [], "updates": 0, "optimizer_steps": 0,
              "training_performed": False, "source": plan["source"]}
    write_json(output / "report.json", report)
    env = None
    try:
        torch.manual_seed(cfg["initialization_seed"])
        random.seed(cfg["policy_seed"])
        loaded = load_policy_checkpoint(plan["initial_checkpoint"]["path"], interface=c.ARMIES_ENVIRONMENT)
        model, checkpoint, pool = loaded.model, loaded.checkpoint, loaded.opponent_pool
        require(pool is not None and len(pool.identities) == 1, "Exactly one frozen initial opponent is required")
        opponent_id = pool.identities[0][0]
        optimizer = torch.optim.Adam(model.parameters(), **cfg["adam"])
        generator = torch.Generator(device="cpu").manual_seed(cfg["policy_seed"])
        python_rng = random.Random(cfg["policy_seed"])
        generators, python_generators = {"minibatch": generator}, {"schedule": python_rng}
        metadata = {"plan_sha256": approved_sha256, "source": plan["source"], "configuration": cfg}
        folders, receipts = [], []
        episode = 0
        evaluation = paired_schedule(cfg["evaluation_engine_seeds"])
        training = paired_schedule(cfg["training_engine_seeds"])
        for phase, schedule in (("before", evaluation), ("training", training), ("after", evaluation)):
            if phase == "after":
                remaining()
                report["optimizer_update_attempted"] = True
                write_json(output / "report.json", report)
                result, rollouts = native_update(model, optimizer, folders, plan, receipts=tuple(receipts),
                                                generator=generator, remaining=remaining)
                report.update(updates=1, optimizer_steps=result.optimizer_steps, training_performed=True, ppo=asdict(result))
                checkpoint = "first-update-" + weights_sha256(model)[:16]
                progress = TrainingProgress(1, len(rollouts), sum(len(item.samples) for item in rollouts))
                save_checkpoint(output / "resume.pt", model, optimizer, checkpoint=checkpoint,
                    ppo_config=cfg["ppo"], progress=progress, source_metadata=metadata,
                    opponent_pool=pool, sampler_state={"next_phase": "after", "next_episode": episode + 1},
                    generators=generators, python_generators=python_generators, at_rollout_boundary=True,
                    interface=c.ARMIES_ENVIRONMENT, provenance=PROVENANCE)
                saved_rng = {"torch": torch.get_rng_state().clone(), "minibatch": generator.get_state().clone(),
                             "python": random.getstate(), "schedule": python_rng.getstate()}
                resumed = ContextualActorCritic(model.config)
                resumed_optimizer = torch.optim.Adam(resumed.parameters(), **cfg["adam"])
                state = load_checkpoint(output / "resume.pt", resumed, resumed_optimizer,
                    expected_ppo_config=cfg["ppo"], expected_source_metadata=metadata, generators=generators,
                    python_generators=python_generators, at_rollout_boundary=True,
                    interface=c.ARMIES_ENVIRONMENT, allowed_provenance=(PROVENANCE,))
                assert_same_state(model.state_dict(), resumed.state_dict())
                assert_same_state(optimizer.state_dict(), resumed_optimizer.state_dict())
                require(state.progress == progress and state.opponent_pool.identities == pool.identities,
                        "Resumed progress or frozen opponent identity differs")
                # Construction consumed global RNG; load must restore the
                # checkpoint's original global and dedicated sampler states.
                assert_same_state(saved_rng["torch"], torch.get_rng_state())
                assert_same_state(saved_rng["python"], random.getstate())
                assert_same_state(saved_rng["minibatch"], generator.get_state())
                assert_same_state(saved_rng["schedule"], python_rng.getstate())
                model, optimizer, pool = resumed, resumed_optimizer, state.opponent_pool
                report["resume_verified"] = True
                save_policy_checkpoint(output / "trained-policy.pt", model, checkpoint=checkpoint,
                    provenance=PROVENANCE, source_metadata=metadata, opponent_pool=pool,
                    interface=c.ARMIES_ENVIRONMENT)
                write_json(output / "report.json", report)
            for index, (seed, seat) in enumerate(schedule):
                remaining()
                launcher = check_identity(plan)
                episode += 1
                # Held-out before/after use the same sampling seeds. No held-out
                # observations or losses enter the update.
                offset = 1000 if phase == "training" else 2000
                policy_seeds = (cfg["policy_seed"] + offset + 2 * index,
                                cfg["policy_seed"] + offset + 2 * index + 1)
                opponent = pool.get(opponent_id)
                models = (model, opponent.model) if seat == 0 else (opponent.model, model)
                identities = (checkpoint, opponent_id) if seat == 0 else (opponent_id, checkpoint)
                row = {"phase": phase, "episode": episode, "engine_seed": seed, "learner_player": seat,
                       "policy_seeds": policy_seeds, "status": "running"}
                report["attempts"].append(row)
                write_json(output / "report.json", report)
                collector = ContextualCollector(models, identities, learner_player=seat,
                    opponent_pool="first-training-frozen-v1", interface=c.ARMIES_ENVIRONMENT,
                    purpose="training" if phase == "training" else "diagnostic", origin="native", policy_seeds=policy_seeds)
                env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=cfg["max_days"], max_actions=cfg["max_requests"] + 1),
                    ledger=True, replay=True, interface=c.ARMIES_ENVIRONMENT,
                    agents={color: {"checkpoint": identity, "kind": "recurrent-first-run", "phase": phase}
                            for color, identity in zip(COLORS, identities)})
                before = weights_sha256(model)
                try:
                    collected = collector.run_episode(env, episode=episode, seed=seed, max_requests=cfg["max_requests"])
                    folder = env.episode_dir
                    write_json(folder / "neural-rollout.json", collected.to_dict())
                    receipt = None
                    if phase == "training":
                        receipt = CollectionReceipt(str(folder.resolve()), str(episode), seed, seat, policy_seeds,
                            digest(folder / "neural-rollout.json"), digest(folder / "manifest.json"),
                            checkpoint, before, opponent_id, opponent.weights_sha256, "first-training-frozen-v1")
                        validate_collection_receipt(receipt, folder, collected, episode=episode, seed=seed, seat=seat,
                            policy_seeds=policy_seeds, learner_checkpoint=checkpoint, learner_digest=before,
                            opponent_checkpoint=opponent_id, opponent_digest=opponent.weights_sha256,
                            opponent_pool="first-training-frozen-v1", max_requests=cfg["max_requests"], max_days=cfg["max_days"])
                        write_json(folder / "collection-receipt.json", asdict(receipt))
                    manifest, _ = verify_collected(folder, collected, plan, replay_name="collection-replay", remaining=remaining)
                    require(weights_sha256(model) == before, "Collection changed policy weights")
                    pool.assert_unchanged()
                    from .contextual_model import encode_history
                    for policy, trajectory in zip(models, collected.rollouts):
                        _verify_behavior(policy, (trajectory,), (encode_history(trajectory.history),))
                    row.update(status=manifest["status"], termination_reason=manifest["termination_reason"],
                               outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                               coverage=history_coverage(tuple(env.history(color) for color in COLORS)),
                               replay_verified=True, weights_sha256=before)
                    if phase == "training":
                        folders.append(folder)
                        receipts.append(receipt)
                except BaseException as error:
                    row.update(status="collection_error", error=f"{type(error).__name__}: {error}",
                               outcomes=None, rewards=None)
                    raise
                finally:
                    env.close()
                    env = None
                    write_json(output / "report.json", report)
        remaining()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
        if isinstance(error, UpdateRolledBack):
            report.update(update_rolled_back=True, rolled_back_optimizer_steps=error.completed_steps,
                          training_performed=error.completed_steps > 0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                report.update(status="stopped", cleanup_error=f"{type(error).__name__}: {error}")
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True,
                        help="Only supply after the human separately authorizes this exact prepared run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.plan, args.approved_plan_sha256, args.output)
    print(json.dumps(result, indent=2))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
