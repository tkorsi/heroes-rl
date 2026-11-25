"""The normal training command, with what PPO receives made comparable between two trees:

    python verify_train.py TREE OUT.jsonl -- train 20 --state S --seed N --workers W

Runs ``python -m heroes`` from TREE with two additions, both outside the timed path of a normal run:
the batch is put in game-number order before the update (the trainer appends games in the order they
finish, and PPO's shuffle works on positions, so two runs whose games finish in another order train on
other minibatches even when every game is the same), and every trajectory's digest (tools/perf/profile_game:
actions, probabilities, values, rewards, bootstrap, guard, compiled recipe) is written to OUT.jsonl with the
game number and seat, then the weights before and after the update. Two trees that play the same games
and train the same way give the same lines.
"""
import hashlib
import json
from pathlib import Path
import sys

tree, out = Path(sys.argv[1]), Path(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, str(tree))
sys.path.insert(0, str(tree / "tools" / "perf"))
from heroes import train  # noqa: E402
from heroes.__main__ import main  # noqa: E402
from profile_game import digest  # noqa: E402

collected = train.collect
updated = train.update_function


def collect(*args, **kwargs):
    batch, tally, average_days = collected(*args, **kwargs)
    batch.sort(key=lambda game: game.job.number)
    with out.open("a") as handle:
        for game in batch:
            for seat, trajectory in ((game.job.learner, game.trajectory), (1 - game.job.learner, game.other)):
                if trajectory is not None:
                    handle.write(json.dumps({"game": game.job.number, "seat": seat, "choices": list(game.choices),
                                             "samples": len(trajectory.decisions), "digest": digest(trajectory)}) + "\n")
    return batch, tally, average_days


def update_function(config):
    update = updated(config)

    def checked(model, optimizer, trajectories, **kwargs):
        before = train.weights_sha256(model)
        sha = hashlib.sha256()
        for trajectory in trajectories:
            sha.update(digest(trajectory).encode())
        report = update(model, optimizer, trajectories, **kwargs)
        with out.open("a") as handle:
            handle.write(json.dumps({"update": True, "trajectories": len(trajectories), "inputs": sha.hexdigest(),
                                     "weights_before": before, "weights_after": train.weights_sha256(model),
                                     "samples": report.samples, "optimizer_steps": report.optimizer_steps,
                                     "policy_loss": report.policy_loss, "value_loss": report.value_loss}) + "\n")
        return report
    return checked


if __name__ == "__main__":  # actors are spawned and import this file again
    train.collect, train.update_function = collect, update_function
    sys.exit(main(argv))
