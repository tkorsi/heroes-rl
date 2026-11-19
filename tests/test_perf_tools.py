"""The measurement scripts in tools/perf run end to end on tiny scripted games. Needs an Apple GPU."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="the MLX update needs an Apple GPU")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "perf"))


@pytest.fixture
def scripted(monkeypatch, tmp_path):
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
