"""Record the engine's messages of one real training game, and replay them through the actor without the engine.

    python actor_replay.py record STATE SEED OUT.jsonl.gz [--max-days 40] --tree TREE
    python actor_replay.py replay STATE OUT.jsonl.gz [--repeat 3] [--tree TREE]

``record`` plays the game as profile_one.py does and writes every message the environment folds, in the
order it folds them, with the answers it sent; the header has the job. ``replay`` plays the same seats (same
model, same seeds, the learner recorded as arrays) on those messages alone: it folds them in order, and
whenever a seat has a decision it observes, samples and answers as the actor does. Every answer must equal
the recorded one, so the stream stays the stream of this game. Prints the actor's processor time per
decision, and digests of the answers, log-probabilities, values and the learner's compacted trajectory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("record", "replay"))
parser.add_argument("state", type=Path)
parser.add_argument("rest", nargs="+")
parser.add_argument("--max-days", type=int, default=40)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--tree", type=Path, default=Path.cwd())
parser.add_argument("--profile", type=Path)
arguments = parser.parse_args()
sys.path.insert(0, str(arguments.tree))
sys.path.insert(0, str(arguments.tree / "tools" / "perf"))
from dataclasses import replace  # noqa: E402
from heroes.__main__ import _config  # noqa: E402
from heroes import actor, checkpoint  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)  # as a training actor (heroes.train.worker)
import gc  # noqa: E402
gc.disable()  # as a training actor: its games run without the cyclic collector
from heroes.engine import env as envmod, protocol  # noqa: E402
from heroes.engine.env import Env  # noqa: E402

config = _config(argparse.Namespace(state=str(arguments.state), max_days=arguments.max_days))
config = replace(config, state=Path(config.state))
model = checkpoint.load_model(checkpoint.latest(config.state)).freeze()
record_mode = "arrays" if config.ppo.games_per_minibatch is not None else True


def job_for(seed):
    rng = random.Random(seed)
    return actor.Job(700000 + seed, rng.randrange(1, 2**31 - 1), seed % 2, (rng.getrandbits(62), rng.getrandbits(62)))


def record(seed, out):
    import shutil
    job = job_for(seed)
    lines = []
    folded = Env._fold

    def fold(self, event):
        if event[0] == "message":
            lines.append(json.dumps({"p": event[1], "m": event[2]}, separators=(",", ":")))
        return folded(self, event)

    sent = Env.step

    def step(self, action):
        lines.append(json.dumps({"a": self.request.player, "action": repr(action)}))
        return sent(self, action)

    Env._fold, Env.step = fold, step
    directory = config.state / "replay-record" / f"game-{job.number}"
    shutil.rmtree(directory, ignore_errors=True)
    env = Env(config.engine(), config.limits)
    try:
        game = actor.play(env, model, job, directory, guard=config.guard, both_seats=config.both_seats, record=record_mode)
    finally:
        env.close()
        shutil.rmtree(directory, ignore_errors=True)
    with gzip.open(out, "wt") as handle:
        handle.write(json.dumps({"seed": seed, "job": [job.number, job.engine_seed, job.learner, list(job.policy_seeds)],
                                 "status": game.status, "reason": game.reason, "day": game.day,
                                 "choices": list(game.choices), "max_days": arguments.max_days}) + "\n")
        for line in lines:
            handle.write(line + "\n")
    print(json.dumps({"recorded": str(out), "lines": len(lines), "status": game.status, "day": game.day,
                      "choices": list(game.choices)}))


def replay(path):
    with gzip.open(path, "rt") as handle:
        header = json.loads(handle.readline())
        items = [json.loads(line) for line in handle]
    number, engine_seed, learner, seeds = header["job"]
    from heroes.engine.process import Engine  # noqa: F401
    import resource
    best = None
    for attempt in range(arguments.repeat):
        producers = [_producer(number, player) for player in (0, 1)]
        seats = [actor.Seat(model, seeds[player], guard=config.guard, record=record_mode if player == learner or config.both_seats else False)
                 for player in (0, 1)]
        unread = [0, 0]
        issued = [None, None]
        answers, mismatch = [], None
        cpu0, wall0 = time.process_time(), time.perf_counter()
        profiler = None
        if arguments.profile is not None and attempt == 0:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        expected = iter(item for item in items if "a" in item)
        for item in items:
            if "a" in item:
                continue
            producers[item["p"]].ingest(item["m"])
            for player in (0, 1):
                history = producers[player].history
                decision = history.decision
                if decision is None or history.action is not None or issued[player] == decision.seq:
                    continue
                if any(producer.terminal for producer in producers):
                    continue
                issued[player] = decision.seq
                seat = seats[player]
                frames = history._frames
                seat.observe(frames, decision.seq + 1)
                action = seat.act()
                want = next(expected)
                if want["a"] != player or want["action"] != repr(action):
                    mismatch = {"at": len(answers), "player": player, "got": repr(action), "want": want}
                    break
                producers[player].response(action)
                producers[player].submitted(action)
                answers.append((player, repr(action), seat.choices[decision.seq][1], seat.values.get(decision.seq)))
            if mismatch:
                break
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(arguments.profile)
        cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
        sha = hashlib.sha256()
        for answer in answers:
            sha.update(repr(answer).encode())
        report = {"replay": str(path), "attempt": attempt, "answers": len(answers), "mismatch": mismatch,
                  "cpu_seconds": round(cpu, 2), "wall_seconds": round(wall, 2),
                  "us_per_decision": round(1e6 * cpu / max(1, len(answers)), 1), "answers_digest": sha.hexdigest()[:16]}
        # The learner's trajectory as training builds it from these frames.
        try:
            seat = seats[learner]
            frames = producers[learner].history._frames
            natural = header["status"] == "completed"
            outcome = producers[learner].terminal
            trajectory = actor.build_trajectory(frames, seat, natural=natural, outcome=outcome, guard=config.guard)
            if seat.recorded(frames, trajectory) or trajectory.recipe is None:
                if trajectory.recipe is None:
                    actor.compile_recipe(model, trajectory)
            if config.ppo.games_per_minibatch is not None:
                actor.compact_recipe(trajectory)
            from profile_game import digest
            report["trajectory_digest"] = digest(trajectory)[:16]
            report["samples"] = len(trajectory.decisions)
        except Exception as error:  # noqa: BLE001
            report["trajectory_digest"] = f"{type(error).__name__}: {error}"
        print(json.dumps(report), flush=True)
        if best is None or cpu < best:
            best = cpu


def _producer(number, player):
    engine = config.engine()
    return protocol.Producer(str(number), player, interface=engine.interface, army_information=engine.army_information)


if arguments.command == "record":
    record(int(arguments.rest[0]), Path(arguments.rest[1]))
else:
    replay(Path(arguments.rest[0]))
