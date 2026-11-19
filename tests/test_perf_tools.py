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
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2, windows_per_replay=4)
    whole = bench_update.run(*scripted, config, None)
    # Ten and two windows: three groups and one, so two rounds, of four and two steps or of three each.
    assert whole["steps_run"] == whole["planned_steps"] == 6
    assert whole["samples"] == 23 and whole["windows_per_replay"] == 4 and "value_loss" in whole
    cut = bench_update.run(*scripted, config, 2)
    assert cut["steps_run"] == 2 and "value_loss" not in cut  # stopped early: no report, nothing copied back


def test_the_fidelity_probe_compares_shared_states_with_replayed_ones(scripted):
    import state_fidelity
    from heroes.learner.ppo import PPOConfig
    # With a vanishing learning rate the weights stand still, so a state from the start of a round is exact.
    still = PPOConfig(decisions_per_window=2, windows_per_minibatch=2, windows_per_replay=4, learning_rate=1e-12)
    records, ran, planned, report = state_fidelity.probe(*scripted, still, None, 1)
    assert ran == planned == report.optimizer_steps == 6
    # Two rounds, of four and two steps or of three each: all but each round's first step are compared.
    assert len(records) == ran - 2 and {1, 2} <= {row[1] for row in records} <= {1, 2, 3}
    summary = state_fidelity.summarize(records)
    assert summary["state_error_max"] < 1e-5 and summary["gradient_cosine_min"] > 0.9999
    assert summary["largest_state_age_steps"] in (2, 3)
    # At a large learning rate the weights move within a round, and its later states drift measurably.
    records, _, _, _ = state_fidelity.probe(*scripted, replace(still, learning_rate=3e-2), None, 1)
    assert state_fidelity.summarize(records)["state_error_max"] > 1e-5
    # Without sharing every step is the first of its round, and there is nothing to compare.
    records, ran, _, _ = state_fidelity.probe(*scripted, replace(still, windows_per_replay=1), None, 1)
    assert records == [] and ran == 6


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
    monkeypatch.setattr(update_quality, "PPOConfig", lambda: update_quality.JUDGE.__class__(decisions_per_window=2))
    monkeypatch.setattr(update_quality, "JUDGE", update_quality.JUDGE.__class__(decisions_per_window=2, windows_per_replay=4))
    monkeypatch.setattr(sys, "argv", ["update_quality.py", "--batch", str(tmp_path / "batch"), "--holdout",
                                      str(tmp_path / "holdout"), "--checkpoint", str(path), "--work", str(tmp_path / "work"),
                                      "--windows-per-replay", "1,4", "--seeds", "0,1"])
    update_quality.main()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    updates = [line for line in lines if line["part"] == "update"]
    assert [line["name"] for line in updates] == ["minibatch2-replay1-seed0", "minibatch2-replay1-seed1",
                                                  "minibatch2-replay4-seed0", "minibatch2-replay4-seed1"]
    assert all(line["samples"] == 23 for line in updates) and updates[0]["optimizer_steps"] == 6
    judged = {line["name"]: line for line in lines if line["part"] == "judged"}
    assert set(judged) == {"checkpoint"} | {line["name"] for line in updates}
    # Judged without a step, the checkpoint agrees with the play that recorded it.
    assert judged["checkpoint"]["trained_on"]["approximate_kl"] == pytest.approx(0, abs=1e-6)
    assert judged["checkpoint"]["held_out"]["clip_fraction"] == 0
    assert all(line["held_out"]["value_loss"] > 0 for line in judged.values())
    changes = lines[-1]
    assert changes["part"] == "changes" and all(length > 0 for length in changes["length"].values())
    assert changes["cosine"]["minibatch2-replay1-seed0"]["minibatch2-replay1-seed0"] == pytest.approx(1)
    # A second call finds every update's weights and trains nothing again.
    assert torch.load(tmp_path / "work" / "weights-minibatch2-replay4-seed1.pt", weights_only=True).keys() == weights.keys()
    update_quality.main()
    assert not [line for line in capsys.readouterr().out.splitlines() if json.loads(line)["part"] == "update"]


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
