#!/usr/bin/env python3
"""Let two checkpoints play each other and count who wins.

    python tools/match.py WORK --first A.pt --second B.pt [--games 100] [--workers 14]
                               [--guard NAME] [--max-days 365] [--seed 1] [--out FILE]

WORK is a folder for the games in play; every call uses a folder of its own
inside it and removes it at the end. This starts the real engine, so by rule 1
of AGENTS.md the user launches it. Nothing is trained and no checkpoint is
written: the games are played for their result alone.

Games come in pairs that share the engine seed and swap the colours, so each
checkpoint plays red once and blue once on the same start, with the same
sampling seed of its own. One seat wins most self-play games on this map,
which is why a single game, or games of one colour, say little. A game that
reaches the day limit has no winner and counts for neither.

One JSON line reports every game and the totals: the wins of each checkpoint,
by the colour the first one played, and the first checkpoint's share of the
decided games with a 95% interval. Both seats play under the same interaction
guard; give the one the checkpoints were trained with.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import sqrt
from pathlib import Path
import random
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heroes.actor import Job  # noqa: E402
from heroes.config import Config  # noqa: E402
from heroes.engine.env import COLORS  # noqa: E402
from heroes.game import guard  # noqa: E402
from heroes.train import Pool  # noqa: E402


def pairs(seed):
    """The endless list of seeded games as ``(job, seat of the first checkpoint)``, two games per start."""
    rng, number = random.Random(seed), 0
    while True:
        engine, first, second = rng.randrange(1, 2**31 - 1), rng.getrandbits(62), rng.getrandbits(62)
        for seat in (0, 1):
            number += 1
            # No learner: nothing is trained on. Each checkpoint keeps its sampling seed in both games.
            yield Job(600000 + 1000 * seed + number, engine, None, (first, second) if seat == 0 else (second, first)), seat


def interval(wins, decided):
    """Wilson 95% interval of a share of ``wins`` among ``decided`` games."""
    if not decided:
        return None
    z, share = 1.96, wins / decided
    centre = (share + z * z / (2 * decided)) / (1 + z * z / decided)
    half = z * sqrt(share * (1 - share) / decided + z * z / (4 * decided * decided)) / (1 + z * z / decided)
    return [round(centre - half, 3), round(centre + half, 3)]


def measure(config, first, second, *, games, seed=1):
    """Play ``games`` games between the checkpoints at ``first`` and ``second`` and return the report."""
    config.engine().check()
    for path in (first, second):
        if not Path(path).is_file():
            raise SystemExit(f"No checkpoint at {path}")
    guard.Tracker(config.guard)  # an unknown guard name stops here, not in every actor
    config.state.mkdir(parents=True, exist_ok=True)
    # The actors name a game's folder by its number alone. Give this call its own place.
    folder = Path(tempfile.mkdtemp(prefix="match-", dir=config.state))
    source, pool = pairs(seed), Pool(replace(config, state=folder, both_seats=False))
    finished, seats, handed = [], {}, 0  # seats: game number -> the seat the first checkpoint plays
    try:
        while len(finished) < games:
            while pool.idle and handed < games:
                job, seat = next(source)
                seats[job.number] = seat
                pool.submit((first, second) if seat == 0 else (second, first), job)
                handed += 1
            message = pool.next()
            if message is not None and message[0] == "game":
                finished.append(message[2])
    finally:
        pool.shutdown()
        shutil.rmtree(folder, ignore_errors=True)
    finished.sort(key=lambda game: game.job.number)
    rows = [{"number": game.job.number, "engine_seed": game.job.engine_seed, "first_plays": COLORS[seats[game.job.number]],
             "status": game.status, "reason": game.reason,
             "first": game.outcomes[seats[game.job.number]] if game.outcomes else None,
             "day": game.day, "choices": list(game.choices), "seconds": round(game.seconds, 1)} for game in finished]
    won = {colour: sum(row["first"] == "win" and row["first_plays"] == colour for row in rows) for colour in COLORS}
    lost = {colour: sum(row["first"] == "loss" and row["first_plays"] == colour for row in rows) for colour in COLORS}
    first_wins, second_wins = sum(won.values()), sum(lost.values())
    decided = first_wins + second_wins
    return {
        "first": str(first), "second": str(second), "guard": config.guard, "max_days": config.limits.max_days,
        "workers": config.workers, "vcmi": str(config.vcmi), "seed": seed, "games": len(rows),
        "first_wins": first_wins, "second_wins": second_wins,
        "stopped": sum(row["status"] == "truncated" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        # By the colour the first checkpoint played: its wins, then its losses.
        "first_as": {colour: [won[colour], lost[colour]] for colour in COLORS},
        "first_share_of_decided": round(first_wins / decided, 3) if decided else None,
        "interval_95": interval(first_wins, decided),
        "mean_day_decided": (round(sum(row["day"] for row in rows if row["first"]) / decided, 1) if decided else None),
        "decisions": sum(sum(row["choices"]) for row in rows),
        "failures": [f"game {game.job.number} {game.reason}: {(game.detail or '')[-600:]}"
                     for game in finished if game.status == "failed"],
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work", type=lambda text: Path(text).expanduser().resolve(), help="folder for the games in play")
    parser.add_argument("--first", required=True, type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--second", required=True, type=lambda text: Path(text).expanduser().resolve())
    parser.add_argument("--games", type=int, default=100, help="an even number: games come in pairs (default 100)")
    parser.add_argument("--workers", type=int, help="games in play at once (default: as in training)")
    parser.add_argument("--guard", help="interaction guard both seats play under (default: the standard one)")
    parser.add_argument("--max-days", type=int, help="stop a game after this game day (default: the training limit)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vcmi", type=lambda text: Path(text).expanduser().resolve(), help="another vcmiclient")
    parser.add_argument("--out", type=Path, help="append the JSON report to this file")
    arguments = parser.parse_args()
    if arguments.games < 2 or arguments.games % 2:
        parser.error("--games must be a positive even number")
    config = replace(Config(), state=arguments.work)
    for name in ("workers", "guard", "vcmi"):
        if getattr(arguments, name) is not None:
            config = replace(config, **{name: getattr(arguments, name)})
    if arguments.max_days is not None:
        config = replace(config, limits=replace(config.limits, max_days=arguments.max_days))
    report = measure(config, arguments.first, arguments.second, games=arguments.games, seed=arguments.seed)
    line = json.dumps(report)
    print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.out.open("a") as stream:
            stream.write(line + "\n")
    summary = {name: report[name] for name in ("games", "first_wins", "second_wins", "stopped", "failed", "first_as",
                                                "first_share_of_decided", "interval_95", "mean_day_decided")}
    print(json.dumps(summary), file=sys.stderr)
    if report["failed"]:
        sys.exit(f"{report['failed']} of {report['games']} games failed; the report lists each one.")


if __name__ == "__main__":
    main()
