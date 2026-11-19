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
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2, state_refresh_steps=4)
    whole = bench_update.run(*scripted, config, None)
    # Ten and two windows in six steps: both games are kept, and the long one expires once.
    assert whole["steps_run"] == whole["planned_steps"] == 6 and whole["game_replays"] == 3
    assert whole["kept_games"] == 2 and whole["samples"] == 23
    cut = bench_update.run(*scripted, config, 2)
    assert cut["steps_run"] == 2 and "game_replays" not in cut  # stopped early: no report, nothing copied back


def test_the_fidelity_probe_compares_reused_states_with_replayed_ones(scripted):
    import state_fidelity
    from heroes.learner.ppo import PPOConfig
    # With a vanishing learning rate the weights stand still, so a reused state is the replayed one.
    still = PPOConfig(decisions_per_window=2, windows_per_minibatch=2, learning_rate=1e-12)
    batch, kept, records, ran = state_fidelity.probe(*scripted, still, [2, 10**9], None, 1)
    assert ran == 6 and kept[2] == {0} and len(records[2]) == len(records[10**9]) == 6
    summary = state_fidelity.summarize(records[10**9])
    assert summary["state_error_max"] < 1e-5 and summary["gradient_cosine_min"] > 0.9999
    assert summary["largest_state_age_steps"] >= 3
    # At the usual learning rate the weights move, and states reused for the whole update drift measurably.
    moving = replace(still, learning_rate=3e-2)
    _, _, records, _ = state_fidelity.probe(*scripted, moving, [10**9], None, 1)
    assert state_fidelity.summarize(records[10**9])["state_error_max"] > 1e-5
