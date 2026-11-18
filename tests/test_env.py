"""The environment and the actor against the scripted engine."""
from pathlib import Path

import pytest

from heroes.engine.env import Env, Limits, Request
from heroes.game import contract


def test_a_natural_game_scores_only_the_last_action(play_fake):
    for learner, reward in ((0, 1), (1, -1)):
        game = play_fake(learner=learner, days=3, winner=0)
        assert (game.status, game.outcomes) == ("completed", ("win", "loss"))
        assert game.choices == (3, 3) and game.day == 3
        trajectory = game.trajectory
        assert len(trajectory.decisions) == 3 and trajectory.reward == reward and trajectory.bootstrap == 0
        assert [trajectory.frames[seq].kind for seq in trajectory.decisions] == ["decision"] * 3


def test_an_action_closed_by_the_terminal_still_owns_the_reward(play_fake):
    # Blue's last action is consumed by the engine but ends in the terminal instead of a result.
    game = play_fake(learner=1, days=3, winner=1, end="closure")
    assert game.status == "completed" and game.trajectory.reward == 1
    assert len(game.trajectory.decisions) == 3
    assert not any(frame.kind == "result" and frame.decision_seq == game.trajectory.decisions[-1]
                   for frame in game.trajectory.frames)


def test_a_game_past_the_day_limit_is_stopped_and_bootstrapped(play_fake):
    # Red meets the limit at its own decision: every action it finished counts,
    # and the value of that unanswered decision stands in for the rest.
    red = play_fake(learner=0, limits=Limits(max_days=4), end="endless")
    assert (red.status, red.reason, red.outcomes) == ("truncated", "game_day_limit", None)
    assert len(red.trajectory.decisions) == 4 and red.trajectory.reward == 0 and red.trajectory.bootstrap != 0
    # Blue was waiting for its next turn: its last action has no following decision and is left out.
    blue = play_fake(learner=1, limits=Limits(max_days=4), end="endless")
    assert blue.status == "truncated" and len(blue.trajectory.decisions) == 3


def test_the_action_limit_stops_a_game(play_fake):
    game = play_fake(limits=Limits(max_actions=2), end="endless")
    assert (game.status, game.reason) == ("truncated", "agent_action_limit")


def test_a_crashed_engine_is_a_failed_game(play_fake):
    game = play_fake(end="crash")
    assert (game.status, game.reason, game.trajectory) == ("failed", "unclean_exit", None)


def test_an_engine_that_dies_before_connecting_reports_its_own_error(play_fake):
    game = play_fake(end="abort")
    assert (game.status, game.reason) == ("failed", "engine_error")
    assert "Headless game start failed" in game.detail and game.seconds < 3


def test_a_relative_game_folder_still_reaches_the_engine(fake_engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the engine runs somewhere else
    env = Env(fake_engine())
    try:
        assert isinstance(env.reset(Path("state") / "game", 1, 7), Request)
        assert (tmp_path / "state" / "game" / "logs" / "process.log").is_file()
    finally:
        env.close()


def test_a_silent_engine_fails_when_an_idle_limit_is_set(play_fake):
    game = play_fake(limits=Limits(idle_seconds=0.5), end="hang")
    assert (game.status, game.reason, game.trajectory) == ("failed", "engine_silent", None)


def test_a_game_can_be_stopped_while_a_decision_is_open(fake_engine, tmp_path):
    env = Env(fake_engine())
    try:
        assert isinstance(env.reset(tmp_path / "game", 1, 7), Request)
        assert env.stop("game_day_limit").status == "truncated"
        # The same environment plays the next game.
        request = env.reset(tmp_path / "again", 2, 7)
        assert isinstance(request, Request) and request.frames[-1].kind == "decision"
    finally:
        env.close()


def test_an_action_the_decision_does_not_offer_is_refused(fake_engine, tmp_path):
    env = Env(fake_engine())
    try:
        request = env.reset(tmp_path / "game", 1, 7)
        decision = request.frames[-1]
        with pytest.raises(ValueError):
            env.step(contract.Action("move_to", "hero-1", 0, 0, schema=decision.offers.schema))
        # The refused action changed nothing: the same decision can still be answered.
        following = env.step(contract.Action("end_turn", schema=decision.offers.schema))
        assert isinstance(following, Request) and following.player == 1
    finally:
        env.close()
