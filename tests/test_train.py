"""The whole loop against the scripted engine: play, update, save, resume."""
from dataclasses import replace
import fcntl
import io
import json
import os
import signal
import time

import pytest

from heroes import checkpoint
from heroes.actor import Job
from heroes.config import Config
from heroes.model.policy import weights_sha256
from heroes.train import Pool, train
from test_checkpoint import write_legacy


@pytest.fixture
def config(fake_engine, tmp_path):
    engine = fake_engine(days=3)
    return Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "state",
                  legacy=tmp_path / "legacy", workers=2, games_per_update=2, backend="torch")


def _log(config):
    return [json.loads(line) for line in (config.state / "log.jsonl").read_text().splitlines()]


def test_training_plays_updates_saves_and_resumes(config):
    with pytest.raises(SystemExit, match="--fresh"):
        train(config, 2, out=io.StringIO())  # nothing to continue from
    out = io.StringIO()
    state = train(config, 4, fresh=True, out=out)
    assert state.progress["updates"] == 2 and state.progress["games"] == 4
    assert state.progress["decisions"] == 12 and state.progress["optimizer_steps"] == 2
    assert sorted(path.name for path in (config.state / "checkpoints").iterdir()) == [
        "update-000000.pt", "update-000001.pt", "update-000002.pt"]
    records = _log(config)
    games = [record for record in records if record["event"] == "game"]
    assert [record["event"] for record in records].count("update") == 2
    assert sorted(record["number"] for record in games) == [1, 2, 3, 4]
    assert {record["status"] for record in games} == {"completed"}
    assert {record["learner"] for record in games} == {"red", "blue"}  # the learner's seat alternates
    assert [record["update"] for record in sorted(games, key=lambda record: record["number"])] == [1, 1, 2, 2]
    assert not any((config.state / "games").iterdir())  # game folders are deleted once read
    assert "update 2 saved" in out.getvalue()

    # A second run continues from the saved state: same weights, next game numbers, next update.
    assert weights_sha256(checkpoint.load_model(checkpoint.latest(config.state))) == weights_sha256(state.model)
    resumed = train(config, 2, out=io.StringIO())
    assert resumed.progress["updates"] == 3 and resumed.progress["games"] == 6
    assert weights_sha256(resumed.model) != weights_sha256(state.model)
    assert [record["number"] for record in _log(config) if record["event"] == "game"][-2:] in ([5, 6], [6, 5])


def test_training_continues_from_the_previous_pipelines_checkpoint(config):
    original = write_legacy(config.legacy, updates=1784)
    state = train(config, 2, out=io.StringIO())
    assert state.origin == "legacy checkpoint update-001784"
    assert state.progress == {"updates": 1785, "games": 9824, "decisions": 12926536, "optimizer_steps": 406885}
    # The imported state was saved untouched as this folder's first checkpoint.
    imported = checkpoint.load_model(config.state / "checkpoints" / "update-001784.pt")
    assert weights_sha256(imported) == weights_sha256(original.model)
    assert [record["number"] for record in _log(config) if record["event"] == "game"] in ([9823, 9824], [9824, 9823])


def test_training_stops_when_every_game_fails(config, monkeypatch):
    monkeypatch.setenv("FAKE_VCMI", json.dumps({"end": "crash"}))
    with pytest.raises(RuntimeError, match="unclean_exit"):
        train(config, 2, fresh=True, out=io.StringIO())
    # Nothing was learned, so the state is still the initial checkpoint.
    assert json.loads((config.state / "latest.json").read_text())["updates"] == 0
    failed = [record["number"] for record in _log(config) if record["event"] == "game"]

    # The next run does not reuse the numbers of the failed games.
    monkeypatch.setenv("FAKE_VCMI", "{}")
    assert train(config, 2, out=io.StringIO()).progress["games"] == 2
    numbers = [record["number"] for record in _log(config) if record["event"] == "game"]
    assert len(set(numbers)) == len(numbers) and min(numbers[len(failed):]) > max(failed)


def test_one_state_folder_serves_one_run_at_a_time(config):
    config.state.mkdir()
    with (config.state / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="Another training run"):
            train(config, 2, fresh=True, out=io.StringIO())
    assert not (config.state / "latest.json").exists()


def test_an_actor_that_dies_is_replaced(config, monkeypatch):
    monkeypatch.setenv("FAKE_VCMI", json.dumps({"end": "hang"}))
    path = checkpoint.save(config.state, checkpoint.fresh(learning_rate=3e-4))
    pool = Pool(config)

    def wait_until(reached, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = reached()
            if value:
                return value
        raise AssertionError("the actors did not get there in time")

    def started(number):
        return wait_until(lambda: (config.state / "games" / f"game-{number:07d}" / "logs" / "process.log").exists()
                          or time.sleep(0.05))

    try:
        for number in (1, 2):
            pool.submit(path, Job(number, 7, number % 2, (1, 2)))
        index = next(index for index, job in pool.busy.items() if job.number == 1)
        started(1), started(2)
        # Killed in the middle of a game: the game comes back as failed.
        os.kill(pool.processes[index].pid, signal.SIGKILL)
        _, died, game = wait_until(pool.next)
        assert (died, game.job.number, game.reason) == (index, 1, "actor_died")
        assert pool.idle == [index] and list(pool.busy) == [1 - index]
        # Killed while waiting for a game: its replacement still receives the next one.
        os.kill(pool.processes[index].pid, signal.SIGKILL)
        pool.processes[index].join()
        assert pool.next() is None and pool.idle == [index]
        pool.submit(path, Job(3, 7, 1, (1, 2)))
        started(3)
        assert pool.busy[index].number == 3
    finally:
        pool.shutdown()
    # Actors that were told to stop in the middle of a game removed its folder.
    assert sorted(folder.name for folder in (config.state / "games").iterdir()) == ["game-0000001"]


def test_a_stopped_game_is_trained_on_with_a_bootstrap(config, monkeypatch):
    monkeypatch.setenv("FAKE_VCMI", json.dumps({"end": "endless"}))
    from heroes.engine.env import Limits
    state = train(replace(config, limits=Limits(max_days=3)), 2, fresh=True, out=io.StringIO())
    games = [record for record in _log(config) if record["event"] == "game"]
    assert {(record["status"], record["reason"]) for record in games} == {("truncated", "game_day_limit")}
    assert state.progress["updates"] == 1
