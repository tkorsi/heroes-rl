"""Kill this probe's real VCMI process after fixed-weight neural actions.

This is a technical-failure diagnostic, never training experience or outcome-rate
evidence. Actions come from the random-initialized policy, not a scripted route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.headless.env import HeadlessSelfPlayEnv, Limits, VcmiLauncher
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint
from experiments.learner.contextual_model import weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector
from experiments.learner.inference import digest, source_identity, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine-seed", type=int, default=95201)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    loaded = load_policy_checkpoint(args.checkpoint)
    opponent = loaded.opponent_pool.get(loaded.opponent_pool.identities[0][0])
    models = (loaded.model, opponent.model)
    before = tuple(weights_sha256(model) for model in models)
    collector = ContextualCollector(models, (loaded.checkpoint, opponent.checkpoint), learner_player=0,
        opponent_pool="fixed-failure-diagnostic", purpose="diagnostic", origin="native", policy_seeds=(95202, 95203))
    env = HeadlessSelfPlayEnv(VcmiLauncher(ROOT / "vcmi/build/bin/vcmiclient",
        Path.home() / "Library/Application Support/vcmi", ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"),
        args.output, limits=Limits(max_days=40, max_actions=100), interface=c.TOWN_ENVIRONMENT, ledger=True, replay=True)
    report = {"schema": "heroes-neural-failure-diagnostic-v1", "purpose": "diagnostic",
              "engine_seed": args.engine_seed, "source": source_identity(),
              "probe_sha256": digest(__file__), "checkpoint_sha256": digest(args.checkpoint),
              "training_performed": False, "optimizer_updates": 0}
    write_json(args.output / "report.json", report)
    try:
        collector.start("1")
        request = env.reset(episode=1, seed=args.engine_seed)
        for _ in range(4):
            if request.done:
                raise ValueError("Game ended before the declared crash boundary")
            request = env.step(collector.act(request))
        if request.done:
            raise ValueError("No pending decision at crash boundary")
        # The handle belongs only to the engine started by this diagnostic.
        env.process.kill()
        env.process.wait(timeout=30)
        end = env.step(collector.act(request))
        if not end.done or end.status != "failed":
            raise ValueError("Actual process crash did not finalize as technical failure")
        collected = collector.finish(end, histories=(env.history("red"), env.history("blue")),
                                     manifest_path=env.episode_dir / "manifest.json")
        if any(rollout.optimization_eligible or any(sample.eligible or sample.reward is not None
                or sample.bootstrap_value is not None for sample in rollout.samples) for rollout in collected.rollouts):
            raise ValueError("Technical failure retained learning targets")
        if not any(rollout.samples for rollout in collected.rollouts):
            raise ValueError("Crash check did not exercise prior resolved actions")
        if tuple(weights_sha256(model) for model in models) != before:
            raise ValueError("Policy weights changed")
        serialized = collected.to_dict()
        CollectedEpisode.from_dict(serialized)
        write_json(env.episode_dir / "neural-rollout.json", serialized)
        report.update(passed=True, status=end.status, termination_reason=end.termination_reason,
                      rewards=end.rewards, outcomes=end.outcomes, bootstrap=end.bootstrap,
                      prior_resolved_actions=sum(len(rollout.samples) for rollout in collected.rollouts),
                      weights_unchanged=True, training_samples=0,
                      files={str(p.relative_to(args.output)): digest(p) for p in (
                          env.episode_dir / "manifest.json", env.episode_dir / "neural-rollout.json",
                          env.episode_dir / "agents/red/history-v3.jsonl", env.episode_dir / "agents/blue/history-v3.jsonl")})
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        env.close()
        write_json(args.output / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
