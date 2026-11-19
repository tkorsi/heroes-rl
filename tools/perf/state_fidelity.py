"""Measure what reusing window states costs in accuracy, on a fixed batch of games.

    python tools/perf/state_fidelity.py --batch STATE/batch --checkpoint STATE/checkpoints/update-NNNNNN.pt \
        --refresh 64,256,1024,1000000 --steps 2000 --every 20

The update itself runs exactly (``state_refresh_steps`` 1): every step replays
its games with the current weights. Beside it, for every value of ``--refresh``,
the script keeps the window states that setting would have used, and at every
``--every``-th step it compares, for the same minibatch and the same weights:

- the states the windows start from (relative L2 error),
- the log-probabilities and values the loss sees (largest absolute difference),
- the gradient (cosine with the exact gradient, and relative norm error).

Nothing is saved. One JSON line per refresh value and per fifth of the run goes
to stdout and to ``--out``. ``--synthetic`` runs on scripted games instead.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from bench_update import _Enough, collected, synthetic  # noqa: E402
from heroes.learner.ppo import PPOConfig, prepare  # noqa: E402


def probe(weights, trajectories, config, refreshes, steps, every):
    from heroes.learner.mlx import update as update_module
    from heroes.learner.mlx.model import MLXBackend
    from heroes.model.policy import ContextualActorCritic, ModelConfig

    batch = prepare(trajectories, config)
    jobs = [[index for index, frame in enumerate(item.recipe.frames) if frame["root"] >= 0] for item in trajectories]
    kept = {refresh: batch.kept(replace(config, state_refresh_steps=refresh)) for refresh in refreshes}
    saved = {refresh: {} for refresh in refreshes}  # game -> (step, table)
    records = {refresh: [] for refresh in refreshes}
    counter = {"step": 0}
    original = MLXBackend.update

    def whole(index):
        starts = batch.starts[index]
        return {"history": index, "frame_limit": bisect_left(jobs[index], starts[-1]),
                "states": [{"stop": stop, "state_index": bisect_left(jobs[index], stop) - 1} for stop in starts]}

    def probed(self, step, *, prefixes=None):
        mx, number = self.mx, counter["step"]
        games = sorted({history for history, _ in prefixes})
        for refresh in refreshes:
            for index in games:
                if index in kept[refresh] and (index not in saved[refresh]
                                               or number - saved[refresh][index][0] >= refresh):
                    saved[refresh][index] = (number, self.prefix_table(whole(index)))
        if number % every == 0:
            plan = self._prepare_plan(step["plan"])
            targets = tuple(self._array(step[key]) for key in ("old_log_probs", "old_values", "advantages", "returns"))
            active = step["active_parameters"]

            def evaluate(states):
                values, gradients = mx.value_and_grad(
                    lambda parameters: self.losses(parameters, step, plan, states, targets))(self.p)
                flat = mx.concatenate([gradients[name].reshape(-1) for name in active])
                return values, flat

            exact_values, exact_gradient = evaluate(prefixes)
            exact_states = mx.concatenate([prefixes[key] for key in sorted(prefixes)], axis=0)
            for refresh in refreshes:
                reused = {}
                for history, stop in prefixes:
                    if history in kept[refresh]:
                        row = bisect_left(batch.starts[history], stop)
                        reused[(history, stop)] = saved[refresh][history][1][row:row + 1]
                    else:
                        reused[(history, stop)] = prefixes[(history, stop)]
                values, gradient = evaluate(reused)
                states = mx.concatenate([reused[key] for key in sorted(reused)], axis=0)
                norm = mx.sqrt(mx.sum(exact_gradient ** 2))
                other = mx.sqrt(mx.sum(gradient ** 2))
                numbers = [
                    mx.sqrt(mx.sum((states - exact_states) ** 2)) / (mx.sqrt(mx.sum(exact_states ** 2)) + 1e-12),
                    mx.max(mx.abs(values[-2] - exact_values[-2])), mx.max(mx.abs(values[-1] - exact_values[-1])),
                    mx.sum(gradient * exact_gradient) / (norm * other + 1e-30), mx.abs(other - norm) / (norm + 1e-30),
                    mx.abs(values[0] - exact_values[0]),
                ]
                mx.eval(numbers)
                age = max((number - saved[refresh][history][0] for history, _ in prefixes
                           if history in kept[refresh]), default=0)
                records[refresh].append((number, age, *(float(item.item()) for item in numbers)))
        counter["step"] += 1
        return original(self, step, prefixes=prefixes)

    model = ContextualActorCritic(ModelConfig())
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    planned = batch.planned_steps(config)
    limit = planned if steps is None else min(steps, planned)

    def progress(step, total):
        if step >= limit and limit < planned:
            raise _Enough

    MLXBackend.update = probed
    try:
        update_module.update(model, optimizer, trajectories, config=config,
                             generator=torch.Generator().manual_seed(0), progress=progress)
    except _Enough:
        pass
    finally:
        MLXBackend.update = original
        for tables in saved.values():
            tables.clear()
    return batch, kept, records, counter["step"]


def summarize(rows):
    def column(position):
        return sorted(row[position] for row in rows)

    def at(values, fraction):
        return values[min(len(values) - 1, int(fraction * len(values)))]

    state, logs, values, cosine, norm, loss = (column(position) for position in range(2, 8))
    return {
        "probes": len(rows), "largest_state_age_steps": max(row[1] for row in rows),
        "state_error_median": at(state, 0.5), "state_error_p95": at(state, 0.95), "state_error_max": state[-1],
        "log_prob_difference_median": at(logs, 0.5), "log_prob_difference_p95": at(logs, 0.95), "log_prob_difference_max": logs[-1],
        "value_difference_median": at(values, 0.5), "value_difference_p95": at(values, 0.95), "value_difference_max": values[-1],
        "gradient_cosine_median": at(cosine, 0.5), "gradient_cosine_p05": at(cosine, 0.05), "gradient_cosine_min": cosine[0],
        "gradient_norm_error_median": at(norm, 0.5), "gradient_norm_error_p95": at(norm, 0.95),
        "loss_difference_median": at(loss, 0.5), "loss_difference_max": loss[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", help="comma-separated lengths, in days, of scripted games to play")
    parser.add_argument("--map", type=int, default=20)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--batch", type=Path, help="folder of game-*.pkl files from collect_games.py")
    parser.add_argument("--checkpoint", type=Path, help="the checkpoint that played --batch")
    parser.add_argument("--refresh", default="64,256,1024,1000000000")
    parser.add_argument("--steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--every", type=int, default=20, help="compare at every N-th step")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if (arguments.synthetic is None) == (arguments.batch is None):
        parser.error("give either --synthetic or --batch")
    torch.set_num_threads(1)
    if arguments.synthetic is not None:
        weights, trajectories = synthetic([int(item) for item in arguments.synthetic.split(",")],
                                          arguments.map, arguments.cache)
        source = f"scripted games of {arguments.synthetic} days, map {arguments.map}"
    else:
        weights, trajectories = collected(arguments.batch, arguments.checkpoint)
        source = f"{len(trajectories)} games from {arguments.batch}"
    refreshes = [int(item) for item in arguments.refresh.split(",")]
    batch, kept, records, ran = probe(weights, trajectories, PPOConfig(), refreshes, arguments.steps, arguments.every)
    for refresh in refreshes:
        rows = records[refresh]
        parts = [("all", rows)] + [(f"fifth {part + 1}", [row for row in rows if row[0] * 5 // max(1, ran) == part])
                                   for part in range(5)]
        for name, chosen in parts:
            if not chosen:
                continue
            line = json.dumps({"source": source, "samples": batch.samples, "steps_run": ran,
                               "planned_steps": batch.planned_steps(PPOConfig()), "state_refresh_steps": refresh,
                               "kept_games": len(kept[refresh]), "games": len(trajectories), "part": name,
                               **summarize(chosen)})
            print(line, flush=True)
            if arguments.out is not None:
                arguments.out.parent.mkdir(parents=True, exist_ok=True)
                with arguments.out.open("a") as stream:
                    stream.write(line + "\n")


if __name__ == "__main__":
    main()
