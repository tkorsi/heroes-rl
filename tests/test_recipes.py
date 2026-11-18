"""The compiled recipe the MLX learner reads, up to the point where the GPU takes over."""
from heroes import actor
from heroes.learner.mlx.plans import _joined_plan, _prefix_plan
from heroes.learner.ppo import PPOConfig, prepare


def test_a_game_compiles_into_plans_for_the_learner(play_fake):
    trajectories = [play_fake(learner=seat, days=4).trajectory for seat in (0, 1)]
    frames = [len(item.frames) for item in trajectories]
    for item in trajectories:
        actor.compile_recipe(play_fake.model, item)
    assert all(item.frames is None for item in trajectories)  # the recipe replaces the typed frames
    assert [item.recipe.frame_count for item in trajectories] == frames
    histories = [item.recipe for item in trajectories]
    # The no-gradient prefix runs over the whole game and scores nothing.
    whole = _prefix_plan(histories[0])
    assert whole["decision_count"] == 0 and len(whole["jobs"]) >= 4
    # A minibatch joins loss windows of different games into one plan that scores exactly their decisions.
    windows = prepare(trajectories, PPOConfig(decisions_per_window=3)).windows
    assert len(windows) == 4
    joined, sequences = _joined_plan(histories, [windows[1], windows[2]], trajectories)
    assert joined["decision_count"] == 1 + 3
    assert [(part["history"], part["start"]) for part in sequences] == [(0, windows[1].start_seq), (1, 0)]
