"""Recompute persisted native policy/value evidence without an engine or optimizer.

This verifies fixed-checkpoint inference and recorded trajectory consistency.
It cannot establish native event completeness, fog correctness or replay fidelity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint
from experiments.learner.contextual_model import ContextualEncoder, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, behavior_guard_version
from experiments.learner.inference import digest, write_json
from experiments.learner.ppo import _admit
from validation.probes.audit_contextual_histories import audit

COLORS = ("red", "blue")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_episode(folder, loaded, *, townless_color=None):
    folder = Path(folder).resolve()
    artifact = folder / "neural-rollout.json"
    raw = json.loads(artifact.read_text())
    episode = CollectedEpisode.from_dict(raw)
    require(episode.to_dict() == raw, "Artifact does not round-trip exactly")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest == json.loads(episode.manifest_json), "Rollout embeds a different manifest")
    profile = c.profile_for(manifest["environment"])
    history_paths = tuple((folder / manifest["players"][color][profile.history_key]["path"]).resolve() for color in COLORS)
    require(all(folder in path.parents for path in history_paths), "History path escapes its episode folder")
    independent = audit(*history_paths, manifest_path)
    require(independent["consistent"] and independent["manifest_bound"], "Persisted history audit failed")
    models = {loaded.checkpoint: loaded.model}
    opponent_ids = set()
    if loaded.opponent_pool is not None:
        for checkpoint, _ in loaded.opponent_pool.identities:
            require(checkpoint not in models, "Checkpoint identifiers collide across learner and opponent roles")
            opponent_ids.add(checkpoint)
            models[checkpoint] = loaded.opponent_pool.get(checkpoint).model
    players = []
    for rollout, history_path, config in zip(episode.rollouts, history_paths, episode.model_configs_json):
        color = COLORS[rollout.player]
        require(rollout.origin == "native" and rollout.purpose == "diagnostic", "Expected native diagnostic evidence")
        require((rollout.role == "learner" and rollout.behavior_checkpoint == loaded.checkpoint)
                or (rollout.role == "frozen_opponent" and rollout.behavior_checkpoint in opponent_ids),
                "Behavior role does not match the supplied learner checkpoint and frozen pool")
        model = models[rollout.behavior_checkpoint]
        require(model.config.to_dict() == json.loads(config), "Checkpoint configuration mismatch")
        before = weights_sha256(model)
        require(before == rollout.weights_sha256, "Checkpoint weights differ from behavior")
        persisted = tuple(c.decode_frame(json.loads(line)) for line in history_path.read_text().splitlines())
        require(persisted == rollout.history, "Rollout omits or changes persisted history")
        encoder = ContextualEncoder(rollout.episode, rollout.player, rollout.history[0].schema,
                                    guard_version=behavior_guard_version(rollout.decisions))
        state = model.initial_state()
        estimates = {item.decision_seq: item for item in rollout.estimates}
        decisions = {item.decision_seq: item for item in rollout.decisions}
        values, probabilities, conditionals = [], [], []
        with torch.no_grad():
            for frame in rollout.history:
                output = model.forward_step(encoder.encode(frame), state)
                state = output.state
                if frame.seq in estimates:
                    require(output.decision is not None, "Actual decision has no value")
                    values.append(abs(output.decision.value.item() - estimates[frame.seq].value))
                if frame.seq in decisions:
                    saved = decisions[frame.seq]
                    require(output.decision.distribution.support == saved.support, "Offered action support changed")
                    actual = output.decision.distribution.evaluate(saved.action)
                    probabilities.append(abs(actual.log_prob.item() - saved.log_prob))
                    require(len(actual.conditionals) == len(saved.conditionals), "Conditional count differs")
                    for left, right in zip(actual.conditionals, saved.conditionals):
                        require((left.name, left.selected_index, left.support_size) ==
                                (right.name, right.selected_index, right.support_size), "Conditional support differs")
                        conditionals.append(abs(left.log_prob.item() - right.log_prob))
        maxima = {"value": max(values, default=0), "joint_log_probability": max(probabilities, default=0),
                  "conditional_log_probability": max(conditionals, default=0)}
        require(all(value == 0 for value in maxima.values()), "Inference is not bit-exact against the fixed checkpoint")
        require(weights_sha256(model) == before, "Inference changed checkpoint weights")
        try:
            _admit((rollout,), model)
        except ValueError as error:
            require("synthetic" in str(error), "PPO rejected evidence for an unexpected reason")
        else:
            raise ValueError("Native diagnostic trajectory passed optimizer admission")
        closures = []
        for sample in rollout.samples:
            closure = getattr(sample, "terminal_closure", None)
            if closure is not None:
                require(sample.result_seq is None and sample.successor_seq is None
                        and sample.bootstrap_value is None, "Terminal closure invented a result or bootstrap")
                closures.append({"decision_seq": sample.decision.decision_seq, "reward": sample.reward,
                                 **asdict(closure)})
        terminals = [frame for frame in rollout.history if frame.kind == "terminal"]
        tail = [frame for frame in rollout.history if terminals and frame.seq > terminals[0].seq]
        tail_views = [frame for frame in tail if frame.kind == "view"]
        terminal_rewards = [sample.reward for sample in rollout.samples if sample.reward in (-1, 1)]
        if rollout.boundary == "natural":
            require(rollout.trajectory_eligible, "Natural trajectory is structurally incomplete")
            expected = 1 if manifest["outcomes_by_player"][color] == "win" else -1
            require(terminal_rewards == [expected] and rollout.samples[-1].reward == expected,
                    "Natural reward is missing, duplicated or attached to an earlier sample")
        elif rollout.boundary == "cutoff":
            require(rollout.trajectory_eligible, "Cutoff trajectory is structurally incomplete")
            require(not terminal_rewards and not closures, "Cutoff supplies a terminal reward or closure")
            final = rollout.samples[-1]
            require(final.successor_seq not in decisions and final.successor_seq in estimates
                    and final.bootstrap_value == estimates[final.successor_seq].value,
                    "Cutoff bootstrap is not the actual unsampled own decision")
        else:
            require(not rollout.trajectory_eligible and all(not sample.eligible and sample.reward is None
                    and sample.bootstrap_value is None for sample in rollout.samples), "Failure supplies learning targets")
        if townless_color == color:
            require(rollout.boundary == "natural" and manifest["outcomes_by_player"][color] == "loss",
                    "Townless fixture did not produce the declared native loss")
            require(len(closures) == 1, "Townless loser lacks exactly one consumed-action terminal closure")
            require(tail_views and not tail_views[-1].observation.heroes,
                    "Townless loser lacks its expected final observed hero removal")
            require(any(frame.kind == "event" and frame.event.kind == "hero_lost" for frame in tail),
                    "Townless loser lacks its post-terminal hero_lost event")
        players.append({"color": color, "frames": len(rollout.history), "values_recomputed": len(values),
                        "actions_recomputed": len(probabilities), "conditionals_recomputed": len(conditionals),
                        "maximum_absolute_difference": maxima, "weights_unchanged": True,
                        "trajectory_eligible": rollout.trajectory_eligible,
                        "optimization_eligible": rollout.optimization_eligible, "optimizer_admission_rejected": True,
                        "samples": len(rollout.samples), "terminal_rewards": terminal_rewards,
                        "terminal_closures": closures,
                        "postterminal_view_sequences": [frame.seq for frame in tail_views],
                        "last_observed_owned_hero_count": len(tail_views[-1].observation.heroes) if tail_views else None})
    return {"episode": str(folder), "passed": True, "status": manifest["status"],
            "termination_reason": manifest["termination_reason"], "players": players,
            "files": {str(path.relative_to(folder)): digest(path) for path in (artifact, manifest_path, *history_paths)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interface", choices=(c.ENVIRONMENT, c.TOWN_ENVIRONMENT, c.ARMIES_ENVIRONMENT),
                        default=c.TOWN_ENVIRONMENT)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--expect-townless-color", choices=COLORS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Audit output exists; retain previous evidence")
    torch.set_num_threads(1)
    loaded = load_policy_checkpoint(args.checkpoint, interface=args.interface)
    report = {"schema": "heroes-neural-rollout-audit-v1", "checkpoint_sha256": digest(args.checkpoint),
              "probe_sha256": digest(__file__), "optimizer_updates": 0, "episodes": [],
              "scope": "Fixed-checkpoint inference and recorded trajectory consistency; native event completeness and contextual replay remain unverified"}
    for folder in args.episode:
        try:
            result = audit_episode(folder, loaded, townless_color=args.expect_townless_color)
        except Exception as error:
            result = {"episode": str(folder.resolve()), "passed": False, "error": f"{type(error).__name__}: {error}"}
        report["episodes"].append(result)
        write_json(args.output, report)
        print(json.dumps(result, sort_keys=True), flush=True)
    return int(not all(item["passed"] for item in report["episodes"]))


if __name__ == "__main__":
    raise SystemExit(main())
