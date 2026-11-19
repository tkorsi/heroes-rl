"""Compare what whole PPO updates learn from the same games, for several update settings.

    python tools/perf/update_quality.py --batch STATE/batch --holdout OTHER/batch \
        --checkpoint STATE/checkpoints/update-NNNNNN.pt --work STATE/quality \
        --windows-per-replay 1,8 --seeds 0,1

Every run starts from the checkpoint as training would, with its Adam state,
and makes one whole update on ``--batch``. Its weights are kept in ``--work``
as ``weights-<name>.pt``, so an interrupted comparison continues where it
stopped, and a weights file put there by other means is judged like the rest.

Each set of weights, and the checkpoint's own, is then judged without taking a
step: the PPO losses over all of ``--batch`` and over ``--holdout``, games
played by the same checkpoint that no update saw. The held-out value loss says
how much better the value head predicts new games, the held-out policy loss
how much of the policy's gain carries over to them. Last come the length of
each update's change of the weights and the cosines between those changes.
Two runs of the same setting that differ only in the shuffle seed show how far
apart chance alone puts two updates.

No checkpoint is written. One JSON line per result goes to stdout and ``--out``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from heroes import checkpoint  # noqa: E402
from heroes.learner.ppo import PPOConfig  # noqa: E402

# Judging takes no optimizer step, so every state is exact however many windows share a replay.
JUDGE = PPOConfig(windows_per_replay=64)


def games(folder):
    trajectories = []
    for item in sorted(Path(folder).glob("game-*.pkl")):
        with item.open("rb") as stream:
            trajectories.append(pickle.load(stream).trajectory)
    if not trajectories:
        raise SystemExit(f"No games in {folder}")
    return trajectories


def train(path, trajectories, config, seed):
    """One whole update from the checkpoint; returns the new weights and a record of the run."""
    from heroes.learner.mlx.update import update
    state = checkpoint.load(path, learning_rate=config.learning_rate)
    started = time.perf_counter()
    report = update(state.model, state.optimizer, trajectories, config=config,
                    generator=torch.Generator().manual_seed(seed))
    return state.model.state_dict(), {
        "seconds": round(time.perf_counter() - started, 1), "optimizer_steps": report.optimizer_steps,
        "samples": report.samples, "policy_loss": report.policy_loss, "value_loss": report.value_loss,
        "approximate_kl": report.approximate_kl, "clip_fraction": report.clip_fraction,
        "timings": {key: round(value, 1) for key, value in report.timings_seconds.items()}}


def judge(path, weights, trajectories):
    """The PPO losses of ``weights`` over the games, against the policy that played them. No step is taken."""
    from heroes.learner.mlx import model as mlx_model
    from heroes.learner.mlx.update import update
    state = checkpoint.load(path, learning_rate=JUDGE.learning_rate)
    state.model.load_state_dict(weights)
    optimizer = torch.optim.Adam(state.model.parameters(), lr=JUDGE.learning_rate)
    optimizer.param_groups[0]["lr"] = 0.0
    # The update refuses to start when its first predictions differ from the ones recorded
    # in play. That guards training; updated weights are meant to differ.
    limits = mlx_model.INITIAL_LOG_PROBABILITY_LIMIT, mlx_model.INITIAL_VALUE_LIMIT
    mlx_model.INITIAL_LOG_PROBABILITY_LIMIT = mlx_model.INITIAL_VALUE_LIMIT = math.inf
    try:
        report = update(state.model, optimizer, trajectories, config=JUDGE, generator=torch.Generator().manual_seed(0))
    finally:
        mlx_model.INITIAL_LOG_PROBABILITY_LIMIT, mlx_model.INITIAL_VALUE_LIMIT = limits
    return {"policy_loss": report.policy_loss, "value_loss": report.value_loss,
            "approximate_kl": report.approximate_kl, "clip_fraction": report.clip_fraction}


def change(weights, base):
    """The update's change of all weights as one vector."""
    return torch.cat([(weights[name] - base[name]).reshape(-1).double() for name in sorted(base)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True, help="folder of game-*.pkl files the updates train on")
    parser.add_argument("--holdout", type=Path, help="folder of game-*.pkl files no update sees")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the checkpoint that played both")
    parser.add_argument("--work", type=Path, required=True, help="folder that keeps each update's weights")
    parser.add_argument("--windows-per-replay", default="1,8", help="comma-separated values, one update each")
    parser.add_argument("--windows-per-minibatch", type=int, default=2)
    parser.add_argument("--seeds", default="0", help="comma-separated shuffle seeds, one update each")
    parser.add_argument("--out", type=Path, help="append the JSON lines to this file")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    arguments.work.mkdir(parents=True, exist_ok=True)

    def write(record):
        line = json.dumps(record)
        print(line, flush=True)
        if arguments.out is not None:
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            with arguments.out.open("a") as stream:
                stream.write(line + "\n")

    trajectories = games(arguments.batch)
    for shared in (int(item) for item in arguments.windows_per_replay.split(",")):
        for seed in (int(item) for item in arguments.seeds.split(",")):
            name = f"minibatch{arguments.windows_per_minibatch}-replay{shared}-seed{seed}"
            target = arguments.work / f"weights-{name}.pt"
            if target.exists():
                continue
            config = replace(PPOConfig(), windows_per_minibatch=arguments.windows_per_minibatch, windows_per_replay=shared)
            weights, record = train(arguments.checkpoint, trajectories, config, seed)
            torch.save(weights, target)
            write({"part": "update", "name": name, "games": len(trajectories), **record})

    base = checkpoint.load_model(arguments.checkpoint).state_dict()
    sets = {"checkpoint": base}
    for item in sorted(arguments.work.glob("weights-*.pt")):
        sets[item.stem.removeprefix("weights-")] = torch.load(item, map_location="cpu", weights_only=True)
    held = games(arguments.holdout) if arguments.holdout is not None else None
    for name, weights in sets.items():
        record = {"part": "judged", "name": name, "trained_on": judge(arguments.checkpoint, weights, trajectories)}
        if held is not None:
            record["held_out"] = judge(arguments.checkpoint, weights, held)
        write(record)
    changes = {name: change(weights, base) for name, weights in sets.items() if name != "checkpoint"}
    names = list(changes)
    write({"part": "changes", "length": {name: float(changes[name].norm()) for name in names},
           "cosine": {first: {second: float(torch.dot(changes[first], changes[second])
                                            / (changes[first].norm() * changes[second].norm() + 1e-30))
                              for second in names} for first in names}})


if __name__ == "__main__":
    main()
