"""Compare what whole PPO updates learn from the same games, for several update settings.

    python tools/perf/update_quality.py --batch STATE/batch --holdout OTHER/batch \
        --checkpoint STATE/checkpoints/update-NNNNNN.pt --work STATE/quality \
        --settings 2:1,16:8,16:8:8.5e-4 --seeds 0,1

A setting is ``windows per minibatch : windows per game [: learning rate]``.
Every run starts from the checkpoint as training would, with its Adam state,
and makes one whole update on ``--batch``. Its weights are kept in ``--work``
as ``weights-<name>.pt`` and its verdict as ``verdict-<name>.json``, so an
interrupted comparison continues where it stopped, a later call adds settings
without repeating the earlier ones, and a weights file put there by other means
is judged like the rest.

Each set of weights, and the checkpoint's own, is judged without taking a
step, on ``--batch`` and on ``--holdout``: games played by the same checkpoint
that no update saw. Per game and over all games it reports

- ``gain``: the mean of (ratio - 1) x advantage, the first-order estimate of
  how much better the new policy does than the one that played. The advantage
  is the game's result minus the old value, not normalized. Higher is better.
- ``kl``: the mean of (ratio - 1) - log ratio, how far the policy moved.
- ``clip_fraction``: decisions whose ratio left the clip range.
- ``value_loss``: half the mean squared error of the new value against the result.

Then, against ``--reference``, the difference of each of these over all games
with a 90% interval from resampling the games, the length of each update's
change of the weights and the cosines between those changes. Two runs of the
same setting that differ only in the shuffle seed show how far apart chance
alone puts two updates.

No checkpoint is written. One JSON line per result goes to stdout and ``--out``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from heroes import checkpoint  # noqa: E402
from heroes.learner.ppo import PPOConfig  # noqa: E402

# Judging takes no optimizer step, so the minibatches only decide how many windows share a replay.
JUDGE = PPOConfig(windows_per_minibatch=64, windows_per_game=32, normalize_advantages=False)
MEASURES = ("gain", "kl", "clip_fraction", "value_loss")


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
    """For each game, its samples and the sums of the four measures over them. No step is taken."""
    import numpy as np
    from heroes.learner.mlx.model import MLXBackend
    from heroes.learner.mlx.update import update
    state = checkpoint.load(path, learning_rate=JUDGE.learning_rate)
    state.model.load_state_dict(weights)
    optimizer = torch.optim.Adam(state.model.parameters(), lr=JUDGE.learning_rate)
    sums = None
    original = MLXBackend.update

    def measured(self, step, prefixes):
        """In place of an optimizer step: the forward pass of the minibatch with the weights as they are."""
        plan = self._prepare_plan(step["plan"])
        targets = tuple(self._array(step[key]) for key in ("old_log_probs", "old_values", "advantages", "returns"))
        values = self.losses(self.p, step, plan, prefixes, targets)
        self.mx.eval(values)
        logs, new_values = (np.array(item, dtype=np.float64) for item in values[-2:])
        log_ratio = logs - step["old_log_probs"].astype(np.float64)
        ratio = np.exp(log_ratio)
        sums[:] += (logs.size, ((ratio - 1) * step["advantages"]).sum(), ((ratio - 1) - log_ratio).sum(),
                    (np.abs(ratio - 1) > JUDGE.clip_ratio).sum(), 0.5 * ((new_values - step["returns"]) ** 2).sum())
        return [float(item.item()) for item in values[1:5]], int(logs.size)

    MLXBackend.update = measured
    result = []
    try:
        for trajectory in trajectories:
            sums = np.zeros(5)
            update(state.model, optimizer, [trajectory], config=JUDGE, generator=torch.Generator().manual_seed(0))
            result.append(sums.tolist())
    finally:
        MLXBackend.update = original
    return result


def overall(rows):
    """The measures over all games: every decision counts the same."""
    samples = sum(row[0] for row in rows)
    return {name: sum(row[position] for row in rows) / samples for position, name in enumerate(MEASURES, 1)}


def against(rows, reference, draws=2000):
    """Each measure minus the reference's, with the 5th and 95th percentile over resampled games."""
    generator = torch.Generator().manual_seed(0)
    picks = torch.randint(len(rows), (draws, len(rows)), generator=generator).tolist()
    result = {}
    for position, name in enumerate(MEASURES, 1):
        def difference(chosen):
            samples = sum(rows[index][0] for index in chosen)
            return sum(rows[index][position] - reference[index][position] for index in chosen) / samples

        spread = sorted(difference(chosen) for chosen in picks)
        result[name] = [difference(range(len(rows))), spread[int(0.05 * draws)], spread[int(0.95 * draws)]]
    return result


def change(weights, base):
    """The update's change of all weights as one vector."""
    return torch.cat([(weights[name] - base[name]).reshape(-1).double() for name in sorted(base)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True, help="folder of game-*.pkl files the updates train on")
    parser.add_argument("--holdout", type=Path, help="folder of game-*.pkl files no update sees")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the checkpoint that played both")
    parser.add_argument("--work", type=Path, required=True, help="folder that keeps each update's weights and verdict")
    parser.add_argument("--settings", default="", help="comma-separated minibatch:per game[:learning rate], one update "
                        "per setting and seed; empty to judge only what --work already holds")
    parser.add_argument("--seeds", default="0", help="comma-separated shuffle seeds")
    parser.add_argument("--reference", default="minibatch2-game1-seed0", help="the update the others are compared with")
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
    for setting in (item.split(":") for item in arguments.settings.split(",") if item):
        config = replace(PPOConfig(), windows_per_minibatch=int(setting[0]), windows_per_game=int(setting[1]))
        label = f"minibatch{setting[0]}-game{setting[1]}"
        if len(setting) > 2:
            config, label = replace(config, learning_rate=float(setting[2])), f"{label}-rate{setting[2]}"
        for seed in (int(item) for item in arguments.seeds.split(",")):
            target = arguments.work / f"weights-{label}-seed{seed}.pt"
            if not target.exists():
                weights, record = train(arguments.checkpoint, trajectories, config, seed)
                torch.save(weights, target)
                write({"part": "update", "name": f"{label}-seed{seed}", "games": len(trajectories), **record})

    base = checkpoint.load_model(arguments.checkpoint).state_dict()
    sets = {"checkpoint": base}
    for item in sorted(arguments.work.glob("weights-*.pt")):
        sets[item.stem.removeprefix("weights-")] = torch.load(item, map_location="cpu", weights_only=True)
    held = games(arguments.holdout) if arguments.holdout is not None else None
    verdicts = {}
    for name, weights in sets.items():
        kept = arguments.work / f"verdict-{name}.json"  # judged once, like the update itself
        if not kept.exists():
            verdict = {"trained_on": judge(arguments.checkpoint, weights, trajectories)}
            if held is not None:
                verdict["held_out"] = judge(arguments.checkpoint, weights, held)
            kept.write_text(json.dumps(verdict))
        verdicts[name] = json.loads(kept.read_text())
        write({"part": "judged", "name": name, **{where: overall(rows) for where, rows in verdicts[name].items()}})
    if arguments.reference in verdicts:
        for name, verdict in verdicts.items():
            if name not in ("checkpoint", arguments.reference):
                write({"part": "against " + arguments.reference, "name": name,
                       **{where: against(rows, verdicts[arguments.reference][where]) for where, rows in verdict.items()}})
    changes = {name: change(weights, base) for name, weights in sets.items() if name != "checkpoint"}
    names = list(changes)
    write({"part": "changes", "length": {name: round(float(changes[name].norm()), 3) for name in names},
           "cosine": {first: {second: round(float(torch.dot(changes[first], changes[second])
                                                  / (changes[first].norm() * changes[second].norm() + 1e-30)), 3)
                              for second in names} for first in names}})


if __name__ == "__main__":
    main()
