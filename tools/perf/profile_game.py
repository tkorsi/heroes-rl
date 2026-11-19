"""Play one real game in this process and show where the actor's time goes.

    python tools/perf/profile_game.py STATE [--max-days 60] [--seed 1] [--profile out.prof]

STATE is a state folder of its own with a checkpoint and ``latest.json``. The
game is played like a training game (both seats, the newest weights, recipe
compiled at the end) but nothing is trained or saved besides the report. The
wall time is split into waiting for the engine, the policy's forward pass,
sampling, everything else in Python, and the recipe compilation. With
``--profile`` the game also runs under cProfile, which slows the Python side
and so overstates its share.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from heroes import actor, checkpoint  # noqa: E402
from heroes.config import Config  # noqa: E402
from heroes.engine.env import Env  # noqa: E402


class Clock:
    """Wall time spent inside chosen functions, measured by wrapping them."""

    def __init__(self):
        self.seconds, self.calls, self.depth = {}, {}, {}

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def timed(*arguments, **options):
            # Only the outermost call of a label counts, so nested calls are not counted twice.
            self.depth[label] = self.depth.get(label, 0) + 1
            started = time.perf_counter()
            try:
                return original(*arguments, **options)
            finally:
                self.depth[label] -= 1
                if not self.depth[label]:
                    self.seconds[label] = self.seconds.get(label, 0.0) + time.perf_counter() - started
                    self.calls[label] = self.calls.get(label, 0) + 1

        setattr(owner, name, timed)


def measure(config, seed=1, profile=None):
    """Play one game with ``config`` and return the report; ``profile`` adds a cProfile file."""
    config.engine().check()
    path = checkpoint.latest(config.state)
    if path is None:
        raise SystemExit(f"{config.state} has no latest.json")
    model = checkpoint.load_model(path)
    rng = random.Random(seed)
    job = actor.Job(900000 + seed, rng.randrange(1, 2**31 - 1), seed % 2, (rng.getrandbits(62), rng.getrandbits(62)))
    directory = config.state / "profile-games" / f"game-{job.number}"
    shutil.rmtree(directory, ignore_errors=True)

    from heroes.engine import protocol
    from heroes.model import policy
    clock = Clock()
    wrapped = [(Env, "_wait", "waiting for the engine"), (protocol.Producer, "ingest", "decoding engine messages"),
               (policy.ContextualEncoder, "encode", "encoding frames"), (type(model), "forward_step", "policy forward"),
               (policy.ContextualDistribution, "sample_evaluated", "sampling the action"),
               (protocol.Producer, "response", "checking and encoding the answer"),
               (actor, "build_trajectory", "building the trajectory")]
    originals = [(owner, name, owner.__dict__[name]) for owner, name, _ in wrapped]
    for owner, name, label in wrapped:
        clock.wrap(owner, name, label)

    profiler = None
    if profile is not None:
        import cProfile
        profiler = cProfile.Profile()
    env = Env(config.engine(), config.limits)
    started = time.perf_counter()
    try:
        if profiler is not None:
            profiler.enable()
        game = actor.play(env, model, job, directory, guard=config.guard)
        played, during_play, calls = time.perf_counter(), dict(clock.seconds), dict(clock.calls)
        if game.trajectory is not None:
            actor.compile_recipe(model, game.trajectory)
        compiled = time.perf_counter()
    finally:
        if profiler is not None:
            profiler.disable()
        env.close()
        for owner, name, original in originals:
            setattr(owner, name, original)
        size = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file()) if directory.exists() else 0
        shutil.rmtree(directory, ignore_errors=True)
    decisions = sum(game.choices)
    play_seconds = played - started
    report = {
        "status": game.status, "reason": game.reason, "detail": game.detail, "day": game.day,
        "choices": list(game.choices),
        "samples": len(game.trajectory.decisions) if game.trajectory is not None else 0,
        "frames": game.trajectory.recipe.frame_count if game.trajectory is not None else 0,
        "play_seconds": round(play_seconds, 2), "decisions_per_second": round(decisions / play_seconds, 1),
        "milliseconds_per_decision": round(1000 * play_seconds / max(1, decisions), 3),
        "compile_seconds": round(compiled - played, 2),
        "engine_folder_bytes": size, "profiled": profiler is not None,
        # The parts of the play time. Compiling the recipe encodes the learner's frames once more.
        "parts_seconds": {label: round(value, 2)
                          for label, value in sorted(during_play.items(), key=lambda item: -item[1])},
        "parts_calls": calls,
        "other_python_seconds": round(play_seconds - sum(during_play.values()), 2),
        "compile_encoding_seconds": round(clock.seconds.get("encoding frames", 0.0)
                                          - during_play.get("encoding frames", 0.0), 2),
    }
    if profiler is not None:
        import pstats
        profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile)
        pstats.Stats(profiler, stream=sys.stderr).sort_stats("tottime").print_stats(45)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--max-days", type=int, default=60, help="stop the game after this game day")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile", type=Path, help="also run under cProfile and write the statistics here")
    parser.add_argument("--out", type=Path, help="append the JSON report to this file")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    config = Config()
    config = replace(config, state=arguments.state, limits=replace(config.limits, max_days=arguments.max_days))
    line = json.dumps(measure(config, arguments.seed, arguments.profile))
    print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.out.open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
