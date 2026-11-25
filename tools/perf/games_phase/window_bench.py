"""Keep N training actors busy (the training configuration) for S seconds after a warm-up; decisions per second.

    window_bench.py STATE --workers N --seconds S --warmup W --seed K   (run from the tree to measure)
"""
import argparse, json, random, shutil, sys, tempfile, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from heroes.__main__ import _config
from heroes import checkpoint
from heroes.actor import Job
from heroes.train import Pool
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seconds", type=float, default=150)
    parser.add_argument("--warmup", type=float, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    config = _config(argparse.Namespace(state=str(arguments.state), workers=arguments.workers))
    config = replace(config, state=Path(config.state), workers=arguments.workers)
    path = checkpoint.latest(config.state)
    games = Path(tempfile.mkdtemp(prefix="window-", dir=config.state))
    pool = Pool(replace(config, state=games))
    rng = random.Random(arguments.seed)
    number = 0
    finished, playing = [], {}
    started = time.monotonic(); window = None
    def answered():
        return sum(sum(g.choices) for g in finished) + sum(playing.values())
    try:
        while True:
            now = time.monotonic()
            if window is None and now - started >= arguments.warmup:
                window = (now, answered())
            if window is not None and now - window[0] >= arguments.seconds:
                break
            while pool.idle:
                number += 1
                pool.submit(path, Job(600000 + number, rng.randrange(1, 2**31 - 1), number % 2, (rng.getrandbits(62), rng.getrandbits(62))))
            message = pool.next(1.0)
            if message is None:
                continue
            kind, index, payload = message
            if kind == "progress":
                playing[index] = sum(payload[2])
            else:
                playing.pop(index, None)
                finished.append(payload)
        ended = time.monotonic(); total = answered()
    finally:
        pool.shutdown()
        shutil.rmtree(games, ignore_errors=True)
    report = {"label": arguments.label, "tree": str(Path.cwd()), "vcmi": str(config.vcmi), "workers": arguments.workers,
              "seconds": round(ended - window[0], 1), "decisions": total - window[1],
              "decisions_per_second": round((total - window[1]) / (ended - window[0]), 1), "games_finished": len(finished),
              "failed": sum(g.status == "failed" for g in finished)}
    print(json.dumps(report))
    if arguments.out:
        with arguments.out.open("a") as handle:
            handle.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
