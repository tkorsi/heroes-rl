"""Time the MLX PPO update on a fixed batch of games.

Scripted games (no VCMI, random weights, temporary folders):

    python tools/perf/bench_update.py --synthetic 2000,500,500 --refresh 1,1024

Real games collected by ``collect_games.py``, with the checkpoint that played them:

    python tools/perf/bench_update.py --batch STATE/batch --checkpoint STATE/checkpoints/update-NNNNNN.pt \
        --refresh 1,1024 --steps 300

Every value of ``--refresh`` runs the update from the same weights, shuffle
seed and games; nothing is saved. ``--steps`` stops each run after that many
optimizer steps. One JSON line per run goes to stdout and to ``--out``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import stat
import sys
import tempfile
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

import torch  # noqa: E402

from heroes import actor, checkpoint  # noqa: E402
from heroes.engine.env import Env, Limits  # noqa: E402
from heroes.engine.process import Engine  # noqa: E402
from heroes.game import guard  # noqa: E402
from heroes.learner.ppo import PPOConfig, prepare  # noqa: E402


class _Enough(Exception):
    pass


def synthetic(days, side, cache):
    """Scripted games of ``days`` days each, played by a fresh random policy."""
    key = f"synthetic-{'-'.join(map(str, days))}-map{side}.pkl"
    if cache is not None and (cache / key).is_file():
        with (cache / key).open("rb") as stream:
            return pickle.load(stream)
    model = checkpoint.fresh(learning_rate=3e-4).model.eval()
    trajectories = []
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        binary = folder / "runtime" / "vcmiclient"
        binary.parent.mkdir()
        binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{REPOSITORY / "tests" / "fake_vcmi.py"}" "$@"\n')
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        (folder / "data" / "Data").mkdir(parents=True)
        engine = Engine(binary, folder / "data", REPOSITORY / "maps" / "hero-management-v4.vmap")
        for number, length in enumerate(days, 1):
            os.environ["FAKE_VCMI"] = json.dumps({"days": length, "winner": number % 2, **({"map": side} if side else {})})
            env = Env(engine, Limits(max_days=10**9, max_actions=10**9))
            try:
                game = actor.play(env, model, actor.Job(number, 7, number % 2, (number, number + 1)),
                                  folder / f"game-{number}", guard=guard.VERSION)
            finally:
                env.close()
            started = time.perf_counter()
            actor.compile_recipe(model, game.trajectory)
            print(f"# scripted game {number}: {len(game.trajectory.decisions)} decisions, "
                  f"{game.trajectory.recipe.frame_count} frames, played in {game.seconds:.1f}s, "
                  f"compiled in {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
            trajectories.append(game.trajectory)
    result = (model.state_dict(), trajectories)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / key).open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def collected(batch, path):
    """Real games written by ``collect_games.py`` and the weights that played them."""
    trajectories = []
    for item in sorted(Path(batch).glob("game-*.pkl")):
        with item.open("rb") as stream:
            trajectories.append(pickle.load(stream).trajectory)
    if not trajectories:
        raise SystemExit(f"No games in {batch}")
    return checkpoint.load_model(path).state_dict(), trajectories


def run(weights, trajectories, config, steps, profile=None):
    """One update from ``weights``; returns its timings and metrics. The weights are not kept.

    With ``profile`` the update runs under cProfile, which slows its Python
    side, and the statistics are written to that file.
    """
    from heroes.learner.mlx.update import update
    from heroes.model.policy import ContextualActorCritic, ModelConfig
    model = ContextualActorCritic(ModelConfig())
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch = prepare(trajectories, config)
    planned = batch.planned_steps(config)
    limit = planned if steps is None else min(steps, planned)
    stamps = []

    def progress(step, total):
        stamps.append(time.perf_counter())
        if step >= limit and limit < planned:
            raise _Enough

    profiler = None
    if profile is not None:
        import cProfile
        profiler = cProfile.Profile()
    started, report = time.perf_counter(), None
    try:
        if profiler is not None:
            profiler.enable()
        report = update(model, optimizer, trajectories, config=config,
                        generator=torch.Generator().manual_seed(0), progress=progress)
    except _Enough:
        pass
    finally:
        if profiler is not None:
            profiler.disable()
    finished = time.perf_counter()
    if profiler is not None:
        import pstats
        profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile)
        pstats.Stats(profiler, stream=sys.stderr).sort_stats("cumulative").print_stats(35)
    warm = max(1, len(stamps) // 5)  # the first fifth carries plan building and kernel compilation
    result = {
        "state_refresh_steps": config.state_refresh_steps,
        "windows_per_minibatch": config.windows_per_minibatch, "windows_per_game": config.windows_per_game,
        "decisions_per_window": config.decisions_per_window,
        "games": len(trajectories), "samples": batch.samples, "planned_steps": planned,
        "steps_run": len(stamps), "kept_games": len(batch.kept(config)),
        "seconds_to_last_step": round(stamps[-1] - started, 3),
        "seconds_total": round(finished - started, 3),
        "first_step_seconds": round(stamps[0] - started, 3),
        "steady_seconds_per_step": round((stamps[-1] - stamps[warm - 1]) / max(1, len(stamps) - warm), 5),
        "projected_full_update_seconds": round((stamps[-1] - started) / len(stamps) * planned, 1),
    }
    if report is not None:
        result.update(policy_loss=report.policy_loss, value_loss=report.value_loss,
                      approximate_kl=report.approximate_kl, clip_fraction=report.clip_fraction,
                      game_replays=report.game_replays,
                      timings={key: round(value, 3) for key, value in report.timings_seconds.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", help="comma-separated lengths, in days, of scripted games to play")
    parser.add_argument("--map", type=int, default=20, help="side of the scripted games' changing map; 0 for the fixed view")
    parser.add_argument("--cache", type=Path, help="folder that keeps scripted games between runs")
    parser.add_argument("--batch", type=Path, help="folder of game-*.pkl files from collect_games.py")
    parser.add_argument("--checkpoint", type=Path, help="the checkpoint that played --batch")
    parser.add_argument("--refresh", default="1", help="comma-separated values of state_refresh_steps to time")
    parser.add_argument("--windows-per-minibatch", default="2", help="comma-separated values to time")
    parser.add_argument("--windows-per-game", default="1",
                        help="comma-separated values, paired in order with --windows-per-minibatch")
    parser.add_argument("--decisions-per-window", type=int, default=16)
    parser.add_argument("--steps", type=int, help="stop each run after this many optimizer steps")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--copies", type=int, default=1,
                        help="use every game this many times, to time a batch of production size from a few games; "
                             "the copies share their CPU plans but each has its own plan on the GPU")
    parser.add_argument("--out", type=Path, help="append the JSON lines to this file")
    parser.add_argument("--profile", type=Path, help="run under cProfile and write the statistics here; "
                        "timings of such a run are not comparable")
    parser.add_argument("--plan-cache-gib", type=float,
                        help="for this run only, let the GPU keep this many GiB of whole-game plans (4 in training)")
    arguments = parser.parse_args()
    if (arguments.synthetic is None) == (arguments.batch is None):
        parser.error("give either --synthetic or --batch")
    if arguments.batch is not None and arguments.checkpoint is None:
        parser.error("--batch needs --checkpoint")
    torch.set_num_threads(1)
    if arguments.plan_cache_gib is not None:
        from heroes.learner.mlx import model as mlx_model
        mlx_model.PREFIX_CONSTANT_CACHE_BYTES = int(arguments.plan_cache_gib * 1024**3)
    if arguments.synthetic is not None:
        weights, trajectories = synthetic([int(item) for item in arguments.synthetic.split(",")],
                                          arguments.map, arguments.cache)
        source = f"scripted games of {arguments.synthetic} days, map {arguments.map}"
    else:
        weights, trajectories = collected(arguments.batch, arguments.checkpoint)
        source = f"{len(trajectories)} games from {arguments.batch}"
    if arguments.copies > 1:
        trajectories = trajectories * arguments.copies
        source += f", each used {arguments.copies} times"
    sizes = sorted(len(item.decisions) for item in trajectories)
    frames = sum(item.recipe.frame_count for item in trajectories)
    print(f"# {source}: {sum(sizes)} samples, {frames} frames, game sizes {sizes}", file=sys.stderr, flush=True)
    for _ in range(arguments.repeat):
        sizes_per_step = [int(item) for item in arguments.windows_per_minibatch.split(",")]
        sizes_per_game = [int(item) for item in arguments.windows_per_game.split(",")]
        if len(sizes_per_game) == 1:
            sizes_per_game *= len(sizes_per_step)
        for windows, per_game in zip(sizes_per_step, sizes_per_game, strict=True):
            for refresh in (int(item) for item in arguments.refresh.split(",")):
                config = replace(PPOConfig(), state_refresh_steps=refresh, windows_per_minibatch=windows,
                                 windows_per_game=per_game, decisions_per_window=arguments.decisions_per_window)
                result = {"source": source, "frames": frames, "profiled": arguments.profile is not None,
                          "plan_cache_gib": arguments.plan_cache_gib,
                          **run(weights, trajectories, config, arguments.steps, arguments.profile)}
                line = json.dumps(result)
                print(line, flush=True)
                if arguments.out is not None:
                    arguments.out.parent.mkdir(parents=True, exist_ok=True)
                    with arguments.out.open("a") as stream:
                        stream.write(line + "\n")


if __name__ == "__main__":
    main()
