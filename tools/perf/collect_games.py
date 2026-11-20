"""Play a few real games and keep their trajectories for offline update benchmarks.

    python tools/perf/collect_games.py STATE GAMES [--workers 3] [--seed 1]

STATE is a state folder of its own that already holds a checkpoint and its
``latest.json``. Every usable game is written to ``STATE/batch/game-NNNNNNN.pkl``
as the pickled ``heroes.actor.Game`` the trainer would have received, compiled
recipe included. Nothing is trained and no checkpoint is written.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import pickle
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heroes import checkpoint  # noqa: E402
from heroes.actor import Job  # noqa: E402
from heroes.config import Config  # noqa: E402
from heroes.train import Pool  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("games", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    config = replace(Config(), state=arguments.state, workers=arguments.workers)
    config.engine().check()
    path = checkpoint.latest(config.state)
    if path is None:
        raise SystemExit(f"{config.state} has no latest.json")
    out = config.state / "batch"
    out.mkdir(exist_ok=True)
    number = max((int(item.stem.split("-")[1]) for item in out.glob("game-*.pkl")), default=0)
    rng = random.Random(arguments.seed * 1_000_003 + number)
    pool, done, started = Pool(config), 0, time.monotonic()
    try:
        while done < arguments.games:
            while pool.idle and done + len(pool.busy) < arguments.games:
                number += 1
                pool.submit(path, Job(number, rng.randrange(1, 2**31 - 1), number % 2,
                                      (rng.getrandbits(62), rng.getrandbits(62))))
            message = pool.next()
            if message is None or message[0] == "progress":
                continue
            _, _, game = message
            samples = len(game.trajectory.decisions) if game.trajectory is not None else 0
            print(f"{time.monotonic() - started:8.1f}s game {game.job.number} {game.status} {game.reason} day {game.day} "
                  f"choices {game.choices} samples {samples} seconds {game.seconds:.1f} "
                  f"compile {game.compile_seconds:.1f}", flush=True)
            if game.trajectory is None:
                print(game.detail, flush=True)
                continue
            with (out / f"game-{game.job.number:07d}.pkl").open("wb") as stream:
                pickle.dump(game, stream, protocol=pickle.HIGHEST_PROTOCOL)
            done += 1
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
