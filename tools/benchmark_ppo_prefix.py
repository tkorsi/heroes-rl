#!/usr/bin/env python3
"""Offline recurrent-prefix benchmark; never collects or admits native training.

An existing recording supplies input shapes and legal chosen actions only. Two
fresh random models perform disposable numerical steps; no learned checkpoint,
receipt, engine, campaign or input artifact is changed. Run as a module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import torch

from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history, weights_sha256
from experiments.learner.contextual_rollout import BehaviorDecision
from experiments.learner.ppo import PPOConfig, _evaluate_from_state, clipped_losses


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def benchmark(path, *, steps=2, decisions_per_window=16, seed=8713):
    if steps < 2 or decisions_per_window < 1:
        raise ValueError("At least two disposable steps and positive windows required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    experience = raw.get("experience", raw)
    frames = tuple(c.decode_frame(frame) for frame in experience["history"])
    records = tuple(BehaviorDecision.from_dict({**record, "opponent_checkpoint": "offline-shape-fixture"},
                    frames[record["decision_seq"]]) for record in experience["decisions"])
    if len(records) < 4 * decisions_per_window:
        raise ValueError("Use a long recording with at least four windows")
    inputs = encode_history(frames)
    selected = []
    for first in (len(records) - 2 * decisions_per_window, len(records) - decisions_per_window):
        selected.append((records[first - 1].decision_seq + 1, records[first:first + decisions_per_window]))
    model = ContextualActorCritic(ModelConfig.from_dict(experience["model_config"]))
    models = {name: deepcopy(model) for name in ("reference", "cached")}
    optimizers = {name: torch.optim.Adam(current.parameters(), lr=0.0003) for name, current in models.items()}
    config = PPOConfig(epochs=1)
    frozen = None
    results = {name: [] for name in models}
    snapshots = {}
    for step in range(steps):
        for name in (("reference", "cached") if step % 2 == 0 else ("cached", "reference")):
            current, optimizer = models[name], optimizers[name]
            timing = dict(prefix_reconstruction=0.0, loss_forward=0.0)
            began = perf_counter()
            evaluated = []
            if name == "cached":
                phase = perf_counter()
                states = current.prefix_states(inputs, tuple(start for start, _ in selected))
                timing["prefix_reconstruction"] = perf_counter() - phase
            for start, decisions in selected:
                if name == "reference":
                    phase = perf_counter()
                    state = current.initial_state()
                    with torch.no_grad():
                        for frame in inputs[:start]:
                            state = current.forward_step(frame, state).state
                    timing["prefix_reconstruction"] += perf_counter() - phase
                else:
                    state = states[start]
                phase = perf_counter()
                evaluated.extend(_evaluate_from_state(current, inputs, decisions, start, state))
                timing["loss_forward"] += perf_counter() - phase
            phase = perf_counter()
            log_probs = torch.stack([value.log_prob for value in evaluated])
            values = torch.stack([value.value for value in evaluated])
            if frozen is None:
                frozen = (log_probs.detach().clone(), values.detach().clone())
            advantages = torch.linspace(-1, 1, len(evaluated))
            returns = -torch.ones(len(evaluated))
            losses = clipped_losses(log_probs, values, *frozen, advantages, returns, config)
            timing["loss_forward"] += perf_counter() - phase
            phase = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            gradients = [None if p.grad is None else p.grad.clone() for p in current.parameters()]
            torch.nn.utils.clip_grad_norm_(current.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            timing["backward_and_clip"] = perf_counter() - phase
            phase = perf_counter()
            optimizer.step()
            timing["optimizer_step"] = perf_counter() - phase
            timing["total"] = perf_counter() - began
            snapshots[name] = dict(log_probs=log_probs.detach(), values=values.detach(),
                                   losses=[v.detach() for v in losses], gradients=gradients,
                                   model=deepcopy(current.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
            results[name].append(timing)
        assert_equal(snapshots["reference"], snapshots["cached"])
    totals = {name: {phase: sum(row[phase] for row in rows) for phase in rows[0]}
              for name, rows in results.items()}
    return dict(schema="heroes-ppo-prefix-benchmark-v1", purpose="offline_disposable_numerics_only",
                input_sha256=hashlib.sha256(raw_bytes).hexdigest(), history_frames=len(frames),
                recorded_decisions=len(records), model_config=model.config.to_dict(), random_seed=seed,
                torch_version=torch.__version__, torch_threads=torch.get_num_threads(), platform=platform.platform(),
                selected_windows=[dict(start_seq=start, end_seq=decisions[-1].decision_seq,
                                       decisions=len(decisions)) for start, decisions in selected],
                disposable_optimizer_steps_per_model=steps,
                exact_checks=["decision_log_probs", "decision_values", "five_loss_metrics", "unclipped_gradients",
                              "all_model_tensors", "all_Adam_state_tensors"],
                exact_after_every_step=True, final_disposable_weights_sha256=weights_sha256(models["cached"]),
                timing_seconds=results, total_timing_seconds=totals,
                prefix_speedup=totals["reference"]["prefix_reconstruction"] / totals["cached"]["prefix_reconstruction"],
                numerical_minibatch_speedup=totals["reference"]["total"] / totals["cached"]["total"],
                limitations=["Late windows in one archived long episode, not a whole campaign throughput result",
                             "Fresh random disposable weights; no trained policy is updated or published",
                             "Admission, history decoding, gameplay, replay and checkpoint I/O excluded from timing",
                             "Every GRU transition remains sequential; prefixes still rebuild after every Adam step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new compact report")
    report = benchmark(args.recorded_rollout, steps=args.steps)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("exact_after_every_step", "prefix_speedup", "numerical_minibatch_speedup")}))


if __name__ == "__main__":
    main()
