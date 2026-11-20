"""The MLX update on scripted games, against the torch reference. Needs an Apple GPU."""
from dataclasses import replace

import pytest
import torch

from heroes import actor
from heroes.learner.ppo import PPOConfig
from heroes.learner.ppo import update as reference_update
from heroes.model.policy import ContextualActorCritic

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="the MLX update needs an Apple GPU")


def _pair(play_fake, **script):
    """One scripted game as two equal trajectories: typed frames for torch, a recipe for MLX."""
    trajectory = play_fake(**script).trajectory
    return replace(trajectory), actor.compile_recipe(play_fake.model, trajectory)


def _copy(model):
    twin = ContextualActorCritic(model.config)
    twin.load_state_dict(model.state_dict())
    for parameter in twin.parameters():
        parameter.requires_grad_(True)
    return twin, torch.optim.Adam(twin.parameters(), lr=3e-4)


def _run(update, model, trajectories, config, seed=0):
    twin, optimizer = _copy(model)
    report = update(twin, optimizer, trajectories, config=config, generator=torch.Generator().manual_seed(seed))
    return twin, report


def _metrics(report):
    return (report.policy_loss, report.value_loss, report.approximate_kl, report.clip_fraction)


def test_the_mlx_update_follows_the_reference(play_fake):
    from heroes.learner.mlx.update import update as mlx_update
    pairs = [_pair(play_fake, learner=0, days=9, winner=0), _pair(play_fake, learner=1, days=7, winner=0)]
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2)
    _, expected = _run(reference_update, play_fake.model, [typed for typed, _ in pairs], config)
    model, report = _run(mlx_update, play_fake.model, [compiled for _, compiled in pairs], config)
    assert (report.samples, report.optimizer_steps) == (expected.samples, expected.optimizer_steps) == (16, 5)
    assert _metrics(report) == pytest.approx(_metrics(expected), rel=2e-2, abs=2e-4)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert any(not torch.equal(value, play_fake.model.state_dict()[name]) for name, value in model.state_dict().items())


def test_a_stopped_game_that_both_seats_lost_follows_the_reference(play_fake):
    from heroes.engine.env import Limits
    from heroes.learner.mlx.update import update as mlx_update
    pairs = [_pair(play_fake, learner=0, limits=Limits(max_days=8), end="endless", stopped_reward=-1),
             _pair(play_fake, learner=1, days=7, winner=0)]
    assert [(typed.reward, typed.bootstrap) for typed, _ in pairs] == [(-1, 0.0), (-1, 0.0)]
    assert len(pairs[0][0].decisions) == 8
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2)
    _, expected = _run(reference_update, play_fake.model, [typed for typed, _ in pairs], config)
    _, report = _run(mlx_update, play_fake.model, [compiled for _, compiled in pairs], config)
    assert (report.samples, report.optimizer_steps) == (expected.samples, expected.optimizer_steps)
    assert _metrics(report) == pytest.approx(_metrics(expected), rel=2e-2, abs=2e-4)


def test_a_changing_map_takes_the_sparse_prefix_path(play_fake):
    from heroes.learner.mlx.update import update as mlx_update
    typed, compiled = _pair(play_fake, days=12, map=20)
    assert compiled.recipe.plan(0, compiled.recipe.frame_count)["maps"]["delta"] is not None
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=2)
    _, expected = _run(reference_update, play_fake.model, [typed], config)
    _, report = _run(mlx_update, play_fake.model, [compiled], config)
    assert _metrics(report) == pytest.approx(_metrics(expected), rel=2e-2, abs=2e-4)


def test_grouped_minibatches_follow_the_reference(play_fake):
    from heroes.learner.mlx.update import update as mlx_update
    pairs = [_pair(play_fake, learner=0, days=17, winner=0, map=20), _pair(play_fake, learner=1, days=6, winner=1)]
    # Four windows of one game per step share a single replay of that game.
    config = PPOConfig(decisions_per_window=2, windows_per_minibatch=4, windows_per_game=4)
    _, expected = _run(reference_update, play_fake.model, [typed for typed, _ in pairs], config)
    _, report = _run(mlx_update, play_fake.model, [compiled for _, compiled in pairs], config)
    assert (report.samples, report.optimizer_steps) == (expected.samples, expected.optimizer_steps) == (23, 4)
    assert _metrics(report) == pytest.approx(_metrics(expected), rel=2e-2, abs=2e-4)


def test_a_plans_gpu_bytes_count_every_array_and_no_cpu_sequence(play_fake):
    import numpy as np
    from heroes.learner.mlx.model import plan_bytes
    from heroes.learner.mlx.plans import _prefix_plan
    from heroes.learner.mlx.update import _host
    _, compiled = _pair(play_fake, days=12, map=20)
    host = _host(_prefix_plan(compiled.recipe))
    assert isinstance(host["jobs"], tuple) and host["maps"]["delta"] is not None  # CPU sequences are present

    def every_array(value):  # the plain walk, integer sequences included
        if isinstance(value, np.ndarray):
            return value.size * 4
        if isinstance(value, dict):
            return sum(every_array(item) for item in value.values())
        return sum(every_array(item) for item in value) if isinstance(value, (list, tuple)) else 0

    assert plan_bytes(host) == every_array(host) > 0
