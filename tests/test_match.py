"""Two checkpoints against each other on the scripted engine, where the script decides who wins."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from heroes import actor, checkpoint  # noqa: E402
from heroes.config import Config  # noqa: E402
from heroes.engine.env import Env, Limits  # noqa: E402
from heroes.game import guard  # noqa: E402


@pytest.fixture
def checkpoints(tmp_path):
    """Two different freshly initialized policies, saved as checkpoints."""
    return [checkpoint.save(tmp_path / f"policy-{seed}", checkpoint.fresh(learning_rate=3e-4, seed=seed))
            for seed in (1, 2)]


def _config(fake_engine, tmp_path, **script):
    engine = fake_engine(**script)
    return Config(vcmi=engine.binary, game_data=engine.data, game_map=engine.game_map, state=tmp_path / "work", workers=2)


def test_a_game_can_be_played_by_two_policies_and_without_a_learner(fake_engine, checkpoints, tmp_path):
    red, blue = (checkpoint.load_model(path) for path in checkpoints)
    env = Env(fake_engine(days=3, winner=1))
    try:
        game = actor.play(env, (red, blue), actor.Job(1, 7, None, (1, 2)), tmp_path / "game", guard=guard.VERSION)
    finally:
        env.close()
    # The game has its result and nothing to train on; neither seat was the learner.
    assert (game.status, game.outcomes, game.choices) == ("completed", ("loss", "win"), (3, 3))
    assert game.trajectory is None and game.other is None and game.trajectories == ()
    assert any(not (a == b).all() for a, b in zip(red.state_dict().values(), blue.state_dict().values()))
    # Each seat really played with its own policy: with blue's policy on both seats red answers from other estimates.
    seen = []
    original = actor.Seat.observe

    def watched(self, frames):
        original(self, frames)
        seen.append((self.model, self.pending[0]))

    actor.Seat.observe = watched
    try:
        env = Env(fake_engine(days=3, winner=1))
        try:
            actor.play(env, (red, blue), actor.Job(2, 7, None, (1, 2)), tmp_path / "watched", guard=guard.VERSION)
        finally:
            env.close()
    finally:
        actor.Seat.observe = original
    assert [model is red for model, _ in seen] == [True, False] * 3


def test_a_match_swaps_the_colours_and_counts_each_checkpoints_wins(fake_engine, checkpoints, tmp_path):
    import match
    first, second = checkpoints
    report = match.measure(_config(fake_engine, tmp_path, days=3, winner=0), first, second, games=4)  # red always wins
    assert (report["games"], report["first_wins"], report["second_wins"], report["stopped"], report["failed"]) == (4, 2, 2, 0, 0)
    assert report["first_as"] == {"red": [2, 0], "blue": [0, 2]} and report["first_share_of_decided"] == 0.5
    assert report["interval_95"][0] < 0.5 < report["interval_95"][1] and report["mean_day_decided"] == 3
    rows = report["results"]
    assert [row["first_plays"] for row in rows] == ["red", "blue", "red", "blue"]
    assert [row["first"] for row in rows] == ["win", "loss", "win", "loss"]
    # The two games of a pair start from the same engine seed; the next pair from another.
    assert rows[0]["engine_seed"] == rows[1]["engine_seed"] != rows[2]["engine_seed"] == rows[3]["engine_seed"]
    assert report["decisions"] == 24 and report["failures"] == []
    # The work folder is left as it was found: every call plays in a folder of its own.
    assert list((tmp_path / "work").iterdir()) == []
    # The same match again is the same match.
    again = match.measure(_config(fake_engine, tmp_path, days=3, winner=0), first, second, games=4)
    assert [(row["number"], row["engine_seed"], row["first"]) for row in again["results"]] == [
        (row["number"], row["engine_seed"], row["first"]) for row in rows]


def test_a_match_of_games_that_reach_the_day_limit_has_no_winner(fake_engine, checkpoints, tmp_path):
    import match
    first, second = checkpoints
    config = replace(_config(fake_engine, tmp_path, end="endless"), limits=Limits(max_days=2))
    report = match.measure(config, first, second, games=2)
    assert (report["first_wins"], report["second_wins"], report["stopped"], report["failed"]) == (0, 0, 2, 0)
    assert report["first_share_of_decided"] is None and report["interval_95"] is None and report["mean_day_decided"] is None
    assert [row["first"] for row in report["results"]] == [None, None]


def test_the_match_command_writes_its_report_and_refuses_what_it_cannot_play(fake_engine, checkpoints, tmp_path,
                                                                             monkeypatch, capsys):
    import match
    first, second = checkpoints
    config = _config(fake_engine, tmp_path, days=3, winner=1)
    monkeypatch.setattr(match, "Config", lambda: config)

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["match.py", str(tmp_path / "work"), *arguments])
        capsys.readouterr()
        match.main()
        return json.loads(capsys.readouterr().out)

    report = run("--first", str(first), "--second", str(second), "--games", "2", "--workers", "2",
                 "--guard", guard.version(stack_moves=4, windows=6), "--out", str(tmp_path / "match.jsonl"))
    assert (report["first_wins"], report["second_wins"]) == (1, 1) and report["first_as"] == {"red": [0, 1], "blue": [1, 0]}
    assert report["guard"].endswith("+stack-moves-4+windows-6")
    assert json.loads((tmp_path / "match.jsonl").read_text()) == report
    with pytest.raises(SystemExit) as refused:  # games come in pairs
        run("--first", str(first), "--second", str(second), "--games", "3")
    assert refused.value.code == 2
    with pytest.raises(SystemExit, match="No checkpoint"):
        run("--first", str(first), "--second", str(tmp_path / "missing.pt"), "--games", "2")
    with pytest.raises(ValueError, match="Unsupported interaction guard"):
        run("--first", str(first), "--second", str(second), "--games", "2", "--guard", "no-such-guard")
