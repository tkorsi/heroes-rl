"""One game played as a training actor plays it, with the actor's and the engine's processor time.

    python profile_one.py STATE SEED [--max-days 60] [--profile OUT.prof] [--tree TREE]

The game is a training game: the configuration the command line builds (heroes.__main__._config), the seat
that learns recorded as arrays, the other seat played by the same policy. Prints one JSON line: wall time,
decisions, the actor's own processor time (this process), the engine's (its children), and the digest of the
learner's trajectory as the learner reads it.
"""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("state", type=Path)
parser.add_argument("seed", type=int)
parser.add_argument("--max-days", type=int)
parser.add_argument("--profile", type=Path)
parser.add_argument("--tree", type=Path, default=Path.cwd())
arguments = parser.parse_args()
sys.path.insert(0, str(arguments.tree))
sys.path.insert(0, str(arguments.tree / "tools" / "perf"))
from heroes.__main__ import _config  # noqa: E402
from heroes import actor, checkpoint  # noqa: E402
from heroes.engine.env import Env  # noqa: E402

namespace = argparse.Namespace(state=str(arguments.state), max_days=arguments.max_days)
config = _config(namespace)
from dataclasses import replace as _replace
config = _replace(config, state=Path(config.state))
model = checkpoint.load_model(checkpoint.latest(config.state)).freeze()
rng = random.Random(arguments.seed)
job = actor.Job(700000 + arguments.seed, rng.randrange(1, 2**31 - 1), arguments.seed % 2,
                (rng.getrandbits(62), rng.getrandbits(62)))
directory = config.state / "profile-one" / f"game-{job.number}"
shutil.rmtree(directory, ignore_errors=True)
record = "arrays" if config.ppo.games_per_minibatch is not None else True
import gc  # noqa: E402
gc.disable()  # as an actor plays
profiler = None
if arguments.profile is not None:
    import cProfile
    profiler = cProfile.Profile()
env = Env(config.engine(), config.limits)
own0, kids0 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.perf_counter()
try:
    if profiler is not None:
        profiler.enable()
    game = actor.play(env, model, job, directory, guard=config.guard, both_seats=config.both_seats, record=record)
    played = time.perf_counter()
    for trajectory in game.trajectories:
        if trajectory.recipe is None:
            actor.compile_recipe(model, trajectory)
        if config.ppo.games_per_minibatch is not None:
            actor.compact_recipe(trajectory)
    finished = time.perf_counter()
    if profiler is not None:
        profiler.disable()
finally:
    env.close()
    shutil.rmtree(directory, ignore_errors=True)
own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
decisions = sum(game.choices)
try:
    from profile_game import digest
    learned = digest(game.trajectory)[:16] if game.trajectory is not None else None
except Exception as error:  # noqa: BLE001
    learned = f"no digest: {error}"
report = {"seed": arguments.seed, "status": game.status, "reason": game.reason, "day": game.day,
          "choices": list(game.choices), "samples": sum(len(t.decisions) for t in game.trajectories),
          "wall_seconds": round(played - started, 2), "prepare_seconds": round(finished - played, 2),
          "decisions_per_second": round(decisions / (played - started), 1),
          "actor_cpu_seconds": round(own.ru_utime + own.ru_stime - own0.ru_utime - own0.ru_stime, 2),
          "engine_cpu_seconds": round(kids.ru_utime + kids.ru_stime - kids0.ru_utime - kids0.ru_stime, 2),
          "actor_us_per_decision": round(1e6 * (own.ru_utime + own.ru_stime - own0.ru_utime - own0.ru_stime) / decisions, 1),
          "engine_us_per_decision": round(1e6 * (kids.ru_utime + kids.ru_stime - kids0.ru_utime - kids0.ru_stime) / decisions, 1),
          "max_rss_mb": round(own.ru_maxrss / 1048576, 1), "digest": learned, "tree": str(arguments.tree)}
print(json.dumps(report), flush=True)
if profiler is not None:
    import pstats
    profiler.dump_stats(arguments.profile)
    pstats.Stats(profiler, stream=sys.stderr).sort_stats("tottime").print_stats(40)
