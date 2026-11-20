"""The measurement scripts in tools/perf run end to end on tiny scripted games.

The two that time the update need an Apple GPU; the two that play need none.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "perf"))


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("the MLX update needs an Apple GPU")
    import bench_update
    monkeypatch.setenv("FAKE_VCMI", "{}")  # restored after the test; the script sets it per game
    weights, trajectories = bench_update.synthetic([20, 3], 20, tmp_path)
    assert (tmp_path / "synthetic-20-3-map20.pkl").is_file()
    assert [len(item.decisions) for item in trajectories] == [20, 3]
    return weights, trajectories


def test_the_update_benchmark_reports_each_run(scripted):
    import bench_update
    from heroes.learner.ppo import PPOConfig
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=4, windows_per_game=2)
    whole = bench_update.run(*scripted, config, None)
    # Ten and two windows: five groups and one, two per step.
    assert whole["steps_run"] == whole["planned_steps"] == 3 and whole["samples"] == 23 and "value_loss" in whole
    cut = bench_update.run(*scripted, config, 2)
    assert cut["steps_run"] == 2 and "value_loss" not in cut  # stopped early: no report, nothing copied back


def test_update_quality_judges_updates_on_games_they_did_not_see(scripted, tmp_path, monkeypatch, capsys):
    import json
    import pickle
    import torch
    import update_quality
    from heroes import checkpoint
    from heroes.actor import Game, Job
    weights, trajectories = scripted
    state = checkpoint.fresh(learning_rate=3e-4)
    state.model.load_state_dict(weights)
    path = checkpoint.save(tmp_path / "state", state)
    for folder, chosen in (("batch", trajectories), ("holdout", trajectories[1:])):
        (tmp_path / folder).mkdir()
        for number, trajectory in enumerate(chosen, 1):
            game = Game(Job(number, 7, 0, (1, 2)), "completed", "", None, None, 1, (1, 1), 1.0, trajectory)
            with (tmp_path / folder / f"game-{number:07d}.pkl").open("wb") as stream:
                pickle.dump(game, stream)
    small = update_quality.PPOConfig(decisions_per_window=2)
    monkeypatch.setattr(update_quality, "PPOConfig", lambda: small)
    monkeypatch.setattr(update_quality, "JUDGE", replace(small, windows_per_minibatch=4, windows_per_game=2, normalize_advantages=False))
    monkeypatch.setattr(sys, "argv", ["update_quality.py", "--batch", str(tmp_path / "batch"), "--holdout",
                                      str(tmp_path / "holdout"), "--checkpoint", str(path), "--work", str(tmp_path / "work"),
                                      "--settings", "2:1,4:2,4:2:1e-3", "--seeds", "0,1"])
    update_quality.main()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    updates = [line for line in lines if line["part"] == "update"]
    names = ["minibatch2-game1-seed0", "minibatch2-game1-seed1", "minibatch4-game2-seed0",
             "minibatch4-game2-seed1", "minibatch4-game2-rate1e-3-seed0", "minibatch4-game2-rate1e-3-seed1"]
    assert [line["name"] for line in updates] == names
    assert all(line["samples"] == 23 for line in updates) and updates[0]["optimizer_steps"] == 6
    judged = {line["name"]: line for line in lines if line["part"] == "judged"}
    assert set(judged) == {"checkpoint", *names}
    # Judged without a step, the checkpoint agrees with the play that recorded it.
    assert judged["checkpoint"]["trained_on"]["kl"] == pytest.approx(0, abs=1e-6)
    assert judged["checkpoint"]["trained_on"]["gain"] == pytest.approx(0, abs=1e-4)
    assert judged["checkpoint"]["held_out"]["clip_fraction"] == 0
    assert all(line["held_out"]["value_loss"] > 0 for line in judged.values())
    # The scripted games offer one legal action, so only the value can change.
    assert judged[names[0]]["trained_on"]["value_loss"] != judged["checkpoint"]["trained_on"]["value_loss"]
    compared = {line["name"]: line for line in lines if line["part"].startswith("against ")}
    assert set(compared) == set(names[1:])
    low, high = compared[names[1]]["trained_on"]["value_loss"][1:]
    assert low <= compared[names[1]]["trained_on"]["value_loss"][0] <= high
    changes = lines[-1]
    assert changes["part"] == "changes" and all(length > 0 for length in changes["length"].values())
    assert changes["cosine"][names[0]][names[0]] == pytest.approx(1)
    # A second call finds every update's weights and verdict and repeats nothing.
    assert torch.load(tmp_path / "work" / f"weights-{names[3]}.pt", weights_only=True).keys() == weights.keys()
    monkeypatch.setattr(update_quality, "train", None)
    monkeypatch.setattr(update_quality, "judge", None)
    update_quality.main()
    assert len(capsys.readouterr().out.splitlines()) == len(lines) - len(updates)
def test_the_game_profiler_splits_a_game_into_its_parts(fake_engine, tmp_path):
    import profile_game
    from heroes import checkpoint
    from heroes.config import Config
    engine = fake_engine(days=5, map=20)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "state")
    checkpoint.save(config.state, checkpoint.fresh(learning_rate=3e-4))
    report = profile_game.measure(config, seed=1, profile=tmp_path / "game.prof")
    assert (report["status"], report["choices"], report["samples"]) == ("completed", [5, 5], 5)
    assert report["parts_calls"]["policy forward"] == report["parts_calls"]["encoding frames"] >= 30
    assert report["parts_calls"]["sampling the action"] == 10
    assert 0 <= sum(report["parts_seconds"].values()) <= report["play_seconds"] and (tmp_path / "game.prof").is_file()
    assert 0 <= report["compile_encoding_seconds"] <= report["compile_seconds"]
    assert not (config.state / "profile-games" / "game-900001").exists()  # the engine folder is removed
    # The wrappers are taken off again.
    from heroes.engine.env import Env
    assert Env._wait.__name__ == "_wait"


def test_the_play_benchmark_reports_games_and_a_window(fake_engine, tmp_path):
    import bench_play
    from heroes import checkpoint
    from heroes.config import Config
    engine = fake_engine(days=4, map=8)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "state",
                    workers=2)
    checkpoint.save(config.state, checkpoint.fresh(learning_rate=3e-4))
    report = bench_play.measure(config, games=3)
    assert (report["games"], report["decisions"], report["samples"]) == (3, 24, 12)
    assert report["statuses"] == ["completed:natural_terminal"] * 3 and all(report["digests"])
    # The same seeded games on another number of actors are the same games.
    again = bench_play.measure(replace(config, workers=1), games=3)
    assert again["digests"] == report["digests"] and again["choices"] == report["choices"]
    assert "seat_digests" not in report  # one seat: the report is what it was
    # Both seats: the learner seat's digests are those of the one-seat run, and every colour has its own.
    both = bench_play.measure(replace(config, both_seats=True), games=3)
    assert both["digests"] == report["digests"] and both["samples"] == 24
    assert [sorted(seats) for seats in both["seat_digests"]] == [["blue", "red"]] * 3
    assert [seats[("red", "blue")[number % 2]] for number, seats in enumerate(both["seat_digests"], 1)] == both["digests"]
    assert all(len(set(seats.values())) == 2 and all(seats.values()) for seats in both["seat_digests"])
    assert bench_play.measure(replace(config, both_seats=True), games=3)["seat_digests"] == both["seat_digests"]
    window = bench_play.measure(config, seconds=1.5, warmup=0.5)
    assert window["mode"] == "window" and window["decisions"] > 0 and window["games_failed"] == 0
    # Every call played in a folder of its own and took it away again.
    assert sorted(item.name for item in config.state.iterdir()) == ["checkpoints", "latest.json"]


def test_a_seat_without_a_trajectory_shows_in_the_two_seat_digests(fake_engine, tmp_path, monkeypatch):
    import bench_play
    from heroes import checkpoint
    from heroes.config import Config
    engine = fake_engine(days=4, map=8)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "state",
                    workers=1, both_seats=True)
    checkpoint.save(config.state, checkpoint.fresh(learning_rate=3e-4))
    whole = bench_play.measure(config, games=2)
    original = bench_play.Pool.next

    def next_without_the_opponent(self, *arguments):  # digests are taken here, after an actor handed its game back
        message = original(self, *arguments)
        if message is not None and message[0] == "game":
            message[2].other = None
        return message

    monkeypatch.setattr(bench_play.Pool, "next", next_without_the_opponent)
    partial = bench_play.measure(config, games=2)
    assert partial["digests"] == whole["digests"]
    for learner, seats, complete in zip(("blue", "red"), partial["seat_digests"], whole["seat_digests"]):
        opponent = "red" if learner == "blue" else "blue"
        assert seats == {learner: complete[learner], opponent: None} and complete[opponent]


def test_two_play_benchmarks_on_one_state_folder_leave_each_other_alone(fake_engine, tmp_path):
    import bench_play
    from heroes import checkpoint
    from heroes.config import Config
    engine = fake_engine(days=4, map=8)
    config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "state",
                    workers=2)
    checkpoint.save(config.state, checkpoint.fresh(learning_rate=3e-4))
    # The same folder and seed give both calls the same game numbers, as a before and after comparison does.
    with ThreadPoolExecutor(2) as threads:
        first, second = threads.map(lambda _: bench_play.measure(config, games=4), range(2))
    assert first["failures"] == second["failures"] == []
    assert first["digests"] == second["digests"] and all(first["digests"])
    assert sorted(item.name for item in config.state.iterdir()) == ["checkpoints", "latest.json"]


def test_the_play_benchmark_fails_when_a_game_gives_no_trajectory(fake_engine, tmp_path, monkeypatch, capsys):
    import bench_play
    from heroes import checkpoint
    from heroes.config import Config

    def run(*arguments, **script):
        engine = fake_engine(**script)
        config = Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map)
        monkeypatch.setattr(bench_play, "Config", lambda: config)
        monkeypatch.setattr(sys, "argv", ["bench_play.py", str(tmp_path / "state"), "--workers", "2", "--out",
                                          str(tmp_path / "play.jsonl"), *arguments])
        capsys.readouterr()
        try:
            bench_play.main()
            status = 0
        except SystemExit as stop:
            status = stop.code
        return status, json.loads(capsys.readouterr().out)

    checkpoint.save(tmp_path / "state", checkpoint.fresh(learning_rate=3e-4))
    # A game stopped at the day limit can be trained on and compared: it counts.
    status, report = run("--games", "2", "--max-days", "2", days=4)
    assert status == 0 and report["statuses"] == ["truncated:game_day_limit"] * 2 and all(report["digests"])
    # With both seats a game counts when either seat gives a trajectory. Stopped at red's decision of day 2,
    # blue's one action has no decision after it: red's digest and null for blue, whoever the learner was.
    status, report = run("--games", "2", "--max-days", "1", "--both-seats", days=4)
    assert status == 0 and report["statuses"] == ["truncated:game_day_limit"] * 2 and report["samples"] == 2
    assert [list(seats) for seats in report["seat_digests"]] == [["red", "blue"]] * 2
    assert all(seats["red"] and seats["blue"] is None for seats in report["seat_digests"])
    assert report["digests"] == [None, report["seat_digests"][1]["red"]]  # the learner seats: blue, then red
    # The same games with one seat: the game whose learner is blue gives nothing, as before.
    status, report = run("--games", "2", "--max-days", "1", days=4)
    assert isinstance(status, str) and status.startswith("1 of 2 games gave no trajectory")
    assert report["statuses"] == ["failed:unusable_trajectory", "truncated:game_day_limit"] and "seat_digests" not in report
    # Two engines that die early are a fast batch and no comparison.
    status, report = run("--games", "2", days=4, end="crash")
    assert isinstance(status, str) and status.startswith("2 of 2 games gave no trajectory")
    assert report["digests"] == [None, None] and len(report["failures"]) == 2
    assert all(entry.startswith("failed:") for entry in report["statuses"])
    status, report = run("--games", "2", "--both-seats", days=4, end="crash")  # nor with both seats
    assert isinstance(status, str) and status.startswith("2 of 2 games gave no trajectory")
    assert report["seat_digests"] == [{"red": None, "blue": None}] * 2
    # The report was written all the same, for the diagnosis.
    assert json.loads((tmp_path / "play.jsonl").read_text().splitlines()[-1]) == report
    # A window counts decisions, not games; its report carries the failures and the rate stands.
    status, report = run("--seconds", "1", "--warmup", "0.2", days=4)
    assert status == 0 and report["mode"] == "window"
