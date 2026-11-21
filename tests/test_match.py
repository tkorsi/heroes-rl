"""Two checkpoints against each other on the scripted engine, where the script decides who wins."""
from dataclasses import replace
from itertools import islice
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


def test_a_match_reports_what_each_seat_expected_day_by_day(fake_engine, checkpoints, tmp_path):
    import match
    first, second = checkpoints
    report = match.measure(_config(fake_engine, tmp_path, days=3, winner=0), first, second, games=2)
    assert report["memory_days"] == {"first": None, "second": None}
    one, two = report["results"]
    assert all(len(seat) == 3 and all(isinstance(value, float) for value in seat) for row in (one, two) for seat in row["values"])
    # A policy's estimate at its first decision is the same from either colour here: the scripted seats see the same.
    assert one["values"][0][0] == two["values"][1][0] != one["values"][1][0] == two["values"][0][0]
    # A game that is trained on carries no such list.
    env = Env(fake_engine(days=3, winner=1))
    try:
        game = actor.play(env, checkpoint.load_model(first), actor.Job(9, 7, 0, (1, 2)), tmp_path / "trained",
                          guard=guard.VERSION)
    finally:
        env.close()
    assert game.trajectory is not None and game.daily_values is None


def test_a_seat_with_a_short_memory_starts_anew_when_it_ends_its_turn(fake_engine, checkpoints, tmp_path, monkeypatch):
    model = checkpoint.load_model(checkpoints[0])
    sizes = {}  # a seat's memory -> how much state it held before each of its observations
    original = actor.Seat.observe

    def watched(self, frames):
        sizes.setdefault(self.memory_days, []).append(float(self.state.abs().sum()))
        original(self, frames)

    monkeypatch.setattr(actor.Seat, "observe", watched)

    def play(number, memory, learner=None):
        sizes.clear()
        env = Env(fake_engine(days=4, winner=1))
        try:
            return actor.play(env, model, actor.Job(number, 7, learner, (1, 2), memory), tmp_path / f"game-{number}",
                              guard=guard.VERSION)
        finally:
            env.close()

    whole = play(1, (None, None))
    assert [size > 0 for size in sizes[None][::2]] == [False, True, True, True]  # red's four days
    daily = play(2, (1, None))
    # Red ends its turn once a day and forgets each time; blue remembers the whole game.
    assert sizes[1] == [0.0] * 4 and [size > 0 for size in sizes[None]] == [False, True, True, True]
    assert daily.daily_values[1] == whole.daily_values[1]  # blue sees and answers as before
    assert daily.daily_values[0][0] == whole.daily_values[0][0] and daily.daily_values[0][1:] != whole.daily_values[0][1:]
    play(3, (2, 1))
    assert [size > 0 for size in sizes[2]] == [False, True, False, True] and sizes[1] == [0.0] * 4
    with pytest.raises(ValueError, match="shortened memory"):  # the update replays whole games
        play(4, (1, None), learner=0)


def test_a_match_gives_each_checkpoint_its_memory_in_both_colours(fake_engine, checkpoints, tmp_path, monkeypatch, capsys):
    import match
    first, second = checkpoints
    assert [job.memory_days for job, _ in islice(match.pairs(1, (1, None)), 2)] == [(1, None), (None, 1)]
    assert [job.memory_days for job, _ in islice(match.pairs(1), 2)] == [(None, None)] * 2
    config = _config(fake_engine, tmp_path, days=3, winner=1)
    plain = match.measure(config, first, first, games=2)
    short = match.measure(config, first, first, games=2, memory=(1, None))
    assert short["memory_days"] == {"first": 1, "second": None}
    for before, after in zip(plain["results"], short["results"]):
        seat = ("red", "blue").index(after["first_plays"])
        # The side that remembers answers as in the plain match; the other one's later estimates differ.
        assert after["values"][1 - seat] == before["values"][1 - seat]
        assert after["values"][seat][0] == before["values"][seat][0] and after["values"][seat][1:] != before["values"][seat][1:]
    monkeypatch.setattr(match, "Config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["match.py", str(tmp_path / "work"), "--first", str(first), "--second", str(first),
                                      "--games", "2", "--first-memory-days", "7", "--second-memory-days", "1"])
    capsys.readouterr()
    match.main()
    assert json.loads(capsys.readouterr().out)["memory_days"] == {"first": 7, "second": 1}
    monkeypatch.setattr(sys, "argv", ["match.py", str(tmp_path / "work"), "--first", str(first), "--second", str(first),
                                      "--games", "2", "--first-memory-days", "0"])
    with pytest.raises(SystemExit) as refused:
        match.main()
    assert refused.value.code == 2


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
