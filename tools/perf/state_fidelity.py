"""Measure what sharing a replay among the steps of a round costs in accuracy.

    python tools/perf/state_fidelity.py --batch STATE/batch --checkpoint STATE/checkpoints/update-NNNNNN.pt \
        --windows-per-replay 8 --every 10

The update runs as training would with that ``--windows-per-replay``: a round
replays its games once and its steps start from those states. In every
``--every``-th round the script also replays the games with the weights of each
step, which is what ``windows_per_replay`` 1 would do, and compares for the
same minibatch and the same weights:

- the states the windows start from (relative L2 error),
- the log-probabilities and values the loss sees (largest absolute difference),
- the gradient (cosine with the exact gradient, and relative norm error).

The first step of a round uses fresh states and differs by nothing; it is
left out of the summaries, which are given for all later steps, by the age of
the states in steps, and by fifths of the update.

Nothing is saved. One JSON line per summary goes to stdout and to ``--out``.
``--synthetic`` runs on scripted games instead.
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


def probe(weights, trajectories, config, steps, every):
    """Run the update and return ``[(step, age, state error, log-probability difference, value
    difference, gradient cosine, gradient norm error, loss difference)]`` and the update's report."""
    from heroes.learner.mlx import update as update_module
    from heroes.learner.mlx.model import MLXBackend
    from heroes.model.policy import ContextualActorCritic, ModelConfig

    jobs = [[index for index, frame in enumerate(item.recipe.frames) if frame["root"] >= 0] for item in trajectories]
    # The schedule the update will follow: for every step, its round and its place in it.
    epochs, planned = prepare(trajectories, config).schedule(config, torch.Generator().manual_seed(0))
    places = [(number, age) for number, round_ in enumerate(round_ for rounds in epochs for round_ in rounds)
              for age in range(len(round_))]
    records, counter = [], {"step": 0}
    original = MLXBackend.update

    def probed(self, step, prefixes):
        mx, number = self.mx, counter["step"]
        round_, age = places[number]
        if age and round_ % every == 0:
            stops = {}
            for segment in step["sequences"]:
                stops.setdefault(segment["history"], set()).add(segment["start"])
            exact = self.prefix_states([
                {"history": index, "frame_limit": bisect_left(jobs[index], max(starts)),
                 "states": [{"stop": stop, "state_index": bisect_left(jobs[index], stop) - 1} for stop in sorted(starts)]}
                for index, starts in stops.items()])
            plan = self._prepare_plan(step["plan"])
            targets = tuple(self._array(step[key]) for key in ("old_log_probs", "old_values", "advantages", "returns"))
            active = step["active_parameters"]

            def evaluate(states):
                values, gradients = mx.value_and_grad(
                    lambda parameters: self.losses(parameters, step, plan, states, targets))(self.p)
                return values, mx.concatenate([gradients[name].reshape(-1) for name in active])

            exact_values, exact_gradient = evaluate(exact)
            values, gradient = evaluate(prefixes)
            exact_states = mx.concatenate([exact[key] for key in sorted(exact)], axis=0)
            states = mx.concatenate([prefixes[key] for key in sorted(exact)], axis=0)
            norm, other = mx.sqrt(mx.sum(exact_gradient ** 2)), mx.sqrt(mx.sum(gradient ** 2))
            numbers = [
                mx.sqrt(mx.sum((states - exact_states) ** 2)) / (mx.sqrt(mx.sum(exact_states ** 2)) + 1e-12),
                mx.max(mx.abs(values[-2] - exact_values[-2])), mx.max(mx.abs(values[-1] - exact_values[-1])),
                mx.sum(gradient * exact_gradient) / (norm * other + 1e-30), mx.abs(other - norm) / (norm + 1e-30),
                mx.abs(values[0] - exact_values[0]),
            ]
            mx.eval(numbers)
            records.append((number, age, *(float(item.item()) for item in numbers)))
        counter["step"] += 1
        return original(self, step, prefixes)

    model = ContextualActorCritic(ModelConfig())
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def progress(step, total):
        if steps is not None and steps <= step < total:
            raise _Enough

    MLXBackend.update = probed
    report = None
    try:
        report = update_module.update(model, optimizer, trajectories, config=config,
                                      generator=torch.Generator().manual_seed(0), progress=progress)
    except _Enough:
        pass
    finally:
        MLXBackend.update = original
    return records, counter["step"], planned, report


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
    parser.add_argument("--windows-per-replay", default="8", help="comma-separated values to measure, one update each")
    parser.add_argument("--windows-per-minibatch", type=int, default=2)
    parser.add_argument("--steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--every", type=int, default=10, help="compare in every N-th round")
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

    def write(record):
        line = json.dumps(record)
        print(line, flush=True)
        if arguments.out is not None:
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            with arguments.out.open("a") as stream:
                stream.write(line + "\n")

    for shared in (int(item) for item in arguments.windows_per_replay.split(",")):
        config = replace(PPOConfig(), windows_per_minibatch=arguments.windows_per_minibatch, windows_per_replay=shared)
        records, ran, planned, report = probe(weights, trajectories, config, arguments.steps, arguments.every)
        common = {"source": source, "games": len(trajectories), "windows_per_minibatch": config.windows_per_minibatch,
                  "windows_per_replay": shared, "steps_run": ran, "planned_steps": planned}
        if report is not None:
            write({**common, "part": "the update itself", "samples": report.samples,
                   "policy_loss": report.policy_loss, "value_loss": report.value_loss,
                   "approximate_kl": report.approximate_kl, "clip_fraction": report.clip_fraction})
        parts = [("all", records)]
        parts += [(f"age {age}", [row for row in records if row[1] == age]) for age in sorted({row[1] for row in records})]
        parts += [(f"fifth {part + 1}", [row for row in records if row[0] * 5 // max(1, ran) == part]) for part in range(5)]
        for name, chosen in parts:
            if chosen:
                write({**common, "part": name, **summarize(chosen)})


if __name__ == "__main__":
    main()
