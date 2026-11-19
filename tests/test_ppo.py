"""Targets, loss windows and the reference update."""
from dataclasses import replace
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
    assert batch.schedule(PPOConfig(windows_per_minibatch=4), torch.Generator().manual_seed(0))[1] == 2


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

    # Weights replaced as whole tensors instead of written in place.
    torch.manual_seed(7)
    other = ContextualActorCritic(model.config).eval()
    expected = values(other)
    assert not torch.equal(after, expected)
    model.load_state_dict({name: tensor.clone() for name, tensor in other.state_dict().items()}, assign=True)
    assert torch.equal(values(model), expected)
    model.load_state_dict(untouched.state_dict())
    assert torch.equal(values(model), after)
    for parameter, replacement in zip(model.parameters(), other.parameters()):
        parameter.data = replacement.data.clone()
    assert torch.equal(values(model), expected)


def _game(decisions):
    return Trajectory(decisions=tuple(range(0, 2 * decisions, 2)), log_probs=(0.0,) * decisions,
                      values=(0.0,) * decisions, reward=1, bootstrap=0.0)


def test_a_round_is_one_step_unless_windows_share_a_replay():
    plain = PPOConfig(decisions_per_window=2, windows_per_minibatch=4)
    batch = prepare([_game(40), _game(14), _game(6)], plain)  # 20, 7 and 3 windows
    # Without sharing, the schedule is the plain shuffle it always was, one step per round.
    expected = torch.randperm(30, generator=torch.Generator().manual_seed(5)).tolist()
    rounds = batch.rounds(plain, torch.Generator().manual_seed(5))
    assert rounds == [[expected[left:left + 4]] for left in range(0, 30, 4)]
    epochs, steps = batch.schedule(replace(plain, epochs=2), torch.Generator().manual_seed(5))
    assert epochs[0] == rounds and epochs[1] != rounds and steps == 16

    shared = PPOConfig(decisions_per_window=2, windows_per_minibatch=2, windows_per_replay=4)
    rounds = batch.rounds(shared, torch.Generator().manual_seed(5))
    assert sorted(index for round_ in rounds for step in round_ for index in step) == list(range(30))
    assert len(rounds) == 4  # 5 + 2 + 1 groups, two per round
    for round_ in rounds:
        games = {batch.windows[index].rollout_index for step in round_ for index in step}
        assert len(games) <= 2 and sum(len(step) for step in round_) <= 8  # two groups: at most two games to replay
        assert all(len(step) == 2 for step in round_[:-1]) and 1 <= len(round_[-1]) <= 2 and len(round_) <= 4
    assert sum(len(round_) == 4 for round_ in rounds) >= 2  # a full round makes windows_per_replay steps
    # The steps keep their size and nearly their number: 15 without sharing, one more for each odd round.
    assert 15 <= batch.schedule(shared, torch.Generator().manual_seed(5))[1] <= 17
    assert rounds != batch.rounds(shared, torch.Generator().manual_seed(6))
    for bad in (dict(windows_per_replay=0), dict(windows_per_replay=2.0), dict(windows_per_minibatch=0)):
        with pytest.raises(ValueError):
            replace(plain, **bad)


def test_a_round_replays_each_of_its_games_once(play_fake, monkeypatch):
    from heroes.learner import ppo
    trajectories = [play_fake(learner=0, days=12).trajectory, play_fake(learner=1, days=5).trajectory]
    model = play_fake.model
    replays = []
    original = ppo._states_before
    monkeypatch.setattr(ppo, "_states_before", lambda model, frames, starts: replays.append(len(starts)) or original(
        model, frames, starts))
    reports = {}
    for size in (1, 3):
        config = PPOConfig(decisions_per_window=2, windows_per_replay=size)
        rounds = prepare(trajectories, config).rounds(config, torch.Generator().manual_seed(0))
        optimizer = _optimizer(model)
        optimizer.param_groups[0]["lr"] = 0.0
        replays.clear()
        reports[size] = update(model, optimizer, trajectories, generator=torch.Generator().manual_seed(0), config=config)
        assert reports[size].optimizer_steps == sum(len(round_) for round_ in rounds)
        if size == 1:
            assert len(replays) <= 2 * len(rounds) and max(replays) <= 2
        else:
            assert len(replays) <= 2 * len(rounds) < reports[size].optimizer_steps and max(replays) == 3
    assert (reports[1].samples, reports[1].optimizer_steps) == (17, 5)  # 6 and 3 windows
    # While the weights stand still, a state from the start of a round is the state a replay at the step gives.
    assert reports[3].value_loss == pytest.approx(reports[1].value_loss, rel=1e-6)
    assert reports[3].policy_loss == pytest.approx(reports[1].policy_loss, rel=1e-6, abs=1e-9)
