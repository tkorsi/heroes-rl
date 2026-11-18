"""Targets, loss windows and the reference update."""
import math

import pytest
import torch

from heroes.learner.ppo import PPOConfig, Trajectory, gae, prepare, update
from heroes.model.policy import ContextualActorCritic, encode_history


def test_an_advantage_is_the_final_return_minus_the_value():
    won = gae((0.0, 0.0, 1.0), (0.2, 0.5, 0.9), (0.5, 0.9, 0.0))
    assert won.returns == pytest.approx((1.0, 1.0, 1.0))
    assert won.advantages == pytest.approx((0.8, 0.5, 0.1))
    # A stopped game has no reward; the value after its last action stands in for the rest.
    stopped = gae((0.0, 0.0), (0.2, 0.5), (0.5, 0.7))
    assert stopped.returns == pytest.approx((0.7, 0.7))


def test_loss_windows_cover_every_decision_once():
    game = Trajectory(decisions=(2, 5, 9, 14, 20), log_probs=(0.0,) * 5, values=(0.1,) * 5, reward=1, bootstrap=0.0)
    batch = prepare([game, game], PPOConfig(decisions_per_window=2, normalize_advantages=False))
    # A window starts right after the previous trained decision, the first one at the start of the game.
    assert [(w.rollout_index, w.sample_indices, w.start_seq, w.end_seq) for w in batch.windows] == [
        (0, (0, 1), 0, 5), (0, (2, 3), 6, 14), (0, (4,), 15, 20),
        (1, (0, 1), 0, 5), (1, (2, 3), 6, 14), (1, (4,), 15, 20)]
    assert (batch.offsets, batch.samples) == ([0, 5], 10)
    assert batch.advantages == pytest.approx([0.9] * 10)
    assert batch.planned_steps(PPOConfig(windows_per_minibatch=4)) == 2


def test_advantages_are_normalized_over_the_whole_batch():
    win = Trajectory(decisions=(0, 1), log_probs=(0.0, 0.0), values=(0.0, 0.5), reward=1, bootstrap=0.0)
    loss = Trajectory(decisions=(0, 1), log_probs=(0.0, 0.0), values=(0.0, 0.5), reward=-1, bootstrap=0.0)
    advantages = prepare([win, loss], PPOConfig()).advantages
    assert sum(advantages) == pytest.approx(0, abs=1e-6)
    assert math.sqrt(sum(value ** 2 for value in advantages) / 4) == pytest.approx(1, abs=1e-6)
    assert advantages[0] > 0 > advantages[2]


def test_a_trajectory_ends_with_a_reward_or_a_bootstrap_never_both():
    with pytest.raises(ValueError):
        Trajectory(decisions=(0,), log_probs=(0.0,), values=(0.0,), reward=1, bootstrap=0.3)
    with pytest.raises(ValueError):
        Trajectory(decisions=(3, 2), log_probs=(0.0, 0.0), values=(0.0, 0.0), reward=1, bootstrap=0.0)


def _optimizer(model):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def test_training_replays_exactly_what_the_actor_computed(play_fake):
    games = [play_fake(learner=0, days=3, winner=0), play_fake(learner=1, days=4, winner=0)]
    trajectories = [game.trajectory for game in games]
    model = play_fake.model
    # One minibatch holding every window: all values are recomputed before the first step.
    config = PPOConfig(windows_per_minibatch=100, normalize_advantages=False)
    report = update(model, _optimizer(model), trajectories, config=config, generator=torch.Generator().manual_seed(0))
    assert (report.samples, report.optimizer_steps) == (7, 1)
    # The recomputed policy is the behaviour policy: no divergence and nothing clipped.
    assert report.approximate_kl == pytest.approx(0, abs=1e-6) and report.clip_fraction == 0
    expected = [(1 - value) ** 2 for value in trajectories[0].values] + [(-1 - value) ** 2 for value in trajectories[1].values]
    assert report.value_loss == pytest.approx(0.5 * sum(expected) / 7, rel=1e-4)


def test_the_reference_update_changes_the_weights(play_fake):
    trajectory = play_fake(days=3).trajectory
    model = play_fake.model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(0)
    report = update(model, _optimizer(model), [trajectory], config=PPOConfig(decisions_per_window=2),
                    generator=generator)
    assert report.optimizer_steps == 1  # two windows, two per minibatch
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_encodings_reused_while_acting_never_outlive_the_weights(play_fake):
    trajectory = play_fake(days=3).trajectory
    model = play_fake.model
    inputs = encode_history(trajectory.frames, guard=trajectory.guard)

    def values(policy):
        state, seen = policy.initial_state(), []
        with torch.no_grad():
            for frame in inputs:
                output = policy.forward_step(frame, state)
                state = output.state
                if output.decision is not None:
                    seen.append(output.decision.value)
        return torch.stack(seen)

    before = values(model)  # acting fills the model's memo of encodings
    update(model, _optimizer(model), [trajectory], config=PPOConfig(), generator=torch.Generator().manual_seed(0))
    untouched = ContextualActorCritic(model.config)
    untouched.load_state_dict(model.state_dict())
    after = values(model)
    assert not torch.equal(before, after)
    assert torch.equal(after, values(untouched.eval()))
