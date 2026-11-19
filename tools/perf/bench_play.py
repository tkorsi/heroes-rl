"""Play real games on several actors and report how fast they are played.

    python tools/perf/bench_play.py STATE --workers 10 --games 10 [--max-days 60] [--seed 1]
    python tools/perf/bench_play.py STATE --workers 14 --seconds 180 [--warmup 30]

STATE is a state folder of its own with a checkpoint and ``latest.json``.
Nothing is trained or saved besides the report, one JSON line.

With ``--games`` the same seeded games are played to their end, whatever the
number of actors, and the line carries one digest per game of everything the
learner reads of it. A change to the play path that is meant to be exact must
leave every digest as it was; the times then compare the two versions on
identical work.

With ``--seconds`` every actor is kept busy for that long after the warm-up and
the line carries the decisions answered per second in the window, both seats
of all games together. That is what a training run gets from this number of
actors while games are being collected, without the idle actors at the end of
a batch. Games are stopped at the end of the window.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heroes import checkpoint  # noqa: E402
from heroes.actor import Job  # noqa: E402
from heroes.config import Config  # noqa: E402
from heroes.train import Pool  # noqa: E402


def jobs(seed):
    """The endless list of seeded games: game ``n`` is the same game in every run."""
    rng, number = random.Random(seed), 0
    while True:
        number += 1
        yield Job(800000 + 1000 * seed + number, rng.randrange(1, 2**31 - 1), number % 2,
                  (rng.getrandbits(62), rng.getrandbits(62)))


def measure(config, *, games=None, seconds=None, warmup=30.0, seed=1):
    from profile_game import digest
    config.engine().check()
    path = checkpoint.latest(config.state)
    if path is None:
        raise SystemExit(f"{config.state} has no latest.json")
    source, pool = jobs(seed), Pool(config)
    finished, playing, handed = [], {}, 0  # playing: actor -> decisions answered in its current game
    started = time.monotonic()
    window = None  # (time, decisions) at the start of the measured window

    def answered():
        return sum(sum(game.choices) for game in finished) + sum(playing.values())

    try:
        while True:
            now = time.monotonic()
            if seconds is not None:
                if window is None and now - started >= warmup:
                    window = (now, answered())
                if window is not None and now - window[0] >= seconds:
                    break
            elif len(finished) == games:
                break
            while pool.idle and (games is None or handed < games):
                pool.submit(path, next(source))
                handed += 1
            message = pool.next()
            if message is None:
                continue
            kind, index, payload = message
            if kind == "progress":
                playing[index] = sum(payload[2])
            else:
                playing.pop(index, None)
                finished.append(payload)
        ended = time.monotonic()
        total = answered()
    finally:
        pool.shutdown()
    report = {"workers": config.workers, "vcmi": str(config.vcmi), "max_days": config.limits.max_days, "seed": seed}
    if seconds is not None:
        elapsed = ended - window[0]
        report.update(mode="window", window_seconds=round(elapsed, 1), warmup_seconds=warmup,
                      decisions=total - window[1],
                      decisions_per_second=round((total - window[1]) / elapsed, 1),
                      games_finished=len(finished), games_failed=sum(game.status == "failed" for game in finished))
        return report
    finished.sort(key=lambda game: game.job.number)
    decisions = sum(sum(game.choices) for game in finished)
    in_games = sum(game.seconds + game.compile_seconds for game in finished)
    report.update(mode="games", games=len(finished), wall_seconds=round(ended - started, 1), decisions=decisions,
                  decisions_per_second=round(decisions / (ended - started), 1),
                  samples=sum(len(game.trajectory.decisions) for game in finished if game.trajectory is not None),
                  # What one actor needs per decision of a game, with its engine and the recipe compilation.
                  actor_milliseconds_per_decision=round(1000 * in_games / max(1, decisions), 3),
                  play_seconds=round(sum(game.seconds for game in finished), 1),
                  compile_seconds=round(sum(game.compile_seconds for game in finished), 1),
                  statuses=[f"{game.status}:{game.reason}" for game in finished],
                  days=[game.day for game in finished], choices=[list(game.choices) for game in finished],
                  game_seconds=[round(game.seconds, 1) for game in finished],
                  digests=[digest(game.trajectory)[:16] if game.trajectory is not None else None for game in finished])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("state", type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--games", type=int, help="play this many seeded games to their end")
    parser.add_argument("--seconds", type=float, help="instead: keep every actor busy for this long and count decisions")
    parser.add_argument("--warmup", type=float, default=30.0, help="seconds before the window starts (default 30)")
    parser.add_argument("--max-days", type=int, help="stop each game after this game day (default: the training limit)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vcmi", type=lambda text: Path(text).expanduser().resolve(), help="another vcmiclient")
    parser.add_argument("--label", help="copied into the report")
    parser.add_argument("--out", type=Path, help="append the JSON report to this file")
    arguments = parser.parse_args()
    if (arguments.games is None) == (arguments.seconds is None):
        parser.error("give either --games or --seconds")
    config = Config()
    config = replace(config, state=arguments.state, workers=arguments.workers)
    if arguments.vcmi is not None:
        config = replace(config, vcmi=arguments.vcmi)
    if arguments.max_days is not None:
        config = replace(config, limits=replace(config.limits, max_days=arguments.max_days))
    report = measure(config, games=arguments.games, seconds=arguments.seconds, warmup=arguments.warmup,
                     seed=arguments.seed)
    if arguments.label:
        report = {"label": arguments.label, **report}
    line = json.dumps(report)
    print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.out.open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
