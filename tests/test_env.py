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
    assert "Headless game start failed" in game.detail


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


def _latest_view(frames, decision):
    return next(frame for frame in reversed(frames[:decision.seq]) if frame.kind == "view")


def test_a_decision_shares_the_observation_of_its_view(play_fake):
    # The engine sends what a seat sees once, as a view. The decision after it carries no copy.
    frames = play_fake(days=4, map=8).trajectory.frames
    decisions = [frame for frame in frames if frame.kind == "decision"]
    assert len(decisions) == 4
    assert all(decision.observation is _latest_view(frames, decision).observation for decision in decisions)


def test_an_engine_that_repeats_the_observation_plays_the_same_game(play_fake):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "perf"))
    from profile_game import digest
    from heroes import actor
    games = [play_fake(days=6, map=8, repeat=repeat) for repeat in (False, True)]
    for game in games:
        assert game.status == "completed" and len(game.trajectory.decisions) == 6
    repeated = games[1].trajectory.frames
    decision = next(frame for frame in repeated if frame.kind == "decision")
    view = _latest_view(repeated, decision)
    assert decision.observation == view.observation and decision.observation is not view.observation
    # Everything the learner reads is the same: actions, probabilities, values and the compiled recipe.
    assert len({digest(actor.compile_recipe(play_fake.model, game.trajectory)) for game in games}) == 1


def test_a_view_that_left_the_map_as_it_was_keeps_the_map_of_the_view_before(play_fake):
    # Only the resources change from day to day, so every view after the first arrives without tiles and objects.
    frames = play_fake(days=5, map=8, still=True).trajectory.frames
    views = [frame.observation for frame in frames if frame.kind == "view"]
    assert len(views) == 5 and len({view.resources for view in views}) == 5
    assert all(view.tiles is views[0].tiles and view.objects is views[0].objects for view in views)
    assert len(views[0].tiles) == 64 and len(views[0].objects) == 14


@pytest.mark.parametrize("still", (True, False))
def test_an_engine_that_repeats_the_map_plays_the_same_game(play_fake, still):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "perf"))
    from profile_game import digest
    from heroes import actor
    games = [play_fake(days=6, map=8, still=still, repeat_map=repeat) for repeat in (False, True)]
    for game in games:
        assert game.status == "completed" and len(game.trajectory.decisions) == 6
    told = [[(frame.kind, frame.day, frame.observation, frame.action) for frame in game.trajectory.frames] for game in games]
    assert told[0] == told[1]
    repeated = [frame.observation for frame in games[1].trajectory.frames if frame.kind == "view"]
    assert repeated[1].objects == repeated[0].objects or not still
    assert repeated[1].objects is not repeated[0].objects  # each view of this engine brought its own copy
    # Everything the learner reads is the same: actions, probabilities, values and the compiled recipe.
    assert len({digest(actor.compile_recipe(play_fake.model, game.trajectory)) for game in games}) == 1


@pytest.mark.parametrize("still", (True, False))
def test_a_view_that_changed_some_tiles_brings_only_those(play_fake, still):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "perf"))
    from profile_game import digest
    from heroes import actor
    games = [play_fake(days=6, map=8, still=still, flicker=True, repeat_map=repeat) for repeat in (False, True)]
    told = [[(frame.kind, frame.day, frame.observation, frame.action) for frame in game.trajectory.frames] for game in games]
    assert told[0] == told[1]
    changed, whole = ([frame.observation for frame in game.trajectory.frames if frame.kind == "view"] for game in games)
    assert len({view.tiles for view in changed}) == 6  # the map differs every day
    # Sixty-two or more of the sixty-four tiles of a view are the objects of the view before; a whole map brings its own.
    assert all(sum(a is b for a, b in zip(view.tiles, before.tiles)) >= 60 for before, view in zip(changed, changed[1:]))
    assert len({digest(actor.compile_recipe(play_fake.model, game.trajectory)) for game in games}) == 1


def test_tile_changes_must_fit_the_previous_view(fake_engine):
    producer, envelope = _ready(fake_engine)
    tiles = tuple(contract.Tile((x, 0, 0), 1, 0, False, False) for x in range(2))
    observation = contract.Observation((2, 2, 1), tiles, (), (0,) * 7, (), (),
                                       army_information=producer.army_information, own_armies=())
    whole = contract.encode_observation(observation, schema=producer.profile.history_schema)
    bare = {name: value for name, value in whole.items() if name != "tiles"}
    blocked = {**whole["tiles"][1], "blocked": True}

    def view(day, **parts):
        return producer.ingest({**envelope, "records": [{"kind": "view", "day": day, "observation": {**bare, **parts}}]})[0]

    with pytest.raises(ValueError, match="need the previous view"):  # nothing to change yet
        view(1, tile_changes=[blocked])
    first = view(1, tiles=whole["tiles"])
    second = view(2, tile_changes=[blocked])
    assert second.observation.tiles == (tiles[0], contract.Tile((1, 0, 0), 1, 0, True, False))
    assert second.observation.tiles[0] is first.observation.tiles[0]
    assert view(3, tile_changes=[]).observation.tiles is second.observation.tiles  # only the objects could differ
    for bad in ([{**blocked, "position": [1, 1, 0]}], [blocked, whole["tiles"][1]], [{**blocked, "terrain": True}], {}):
        with pytest.raises(ValueError):
            view(4, tile_changes=bad)
    with pytest.raises(ValueError, match="need the previous view"):
        view(4, tile_changes=[blocked], tiles=whole["tiles"])
    with pytest.raises(ValueError, match="need the previous view"):
        producer.ingest({**envelope, "records": [{"kind": "view", "day": 4, "observation": {
            name: value for name, value in {**bare, "tile_changes": [blocked]}.items() if name != "objects"}}]})


def _ready(fake_engine):
    from heroes.engine.protocol import Producer
    engine = fake_engine()
    producer = Producer("1", 0, interface=engine.interface, army_information=engine.army_information)
    envelope = {"protocol_version": producer.profile.protocol, "episode": "1", "player": 0}
    producer.ingest({**envelope, "type": "ready", "interface": producer.profile.environment,
                     "history_schema": producer.profile.history_schema, "action_schema": producer.profile.action_schema,
                     "army_information": engine.army_information})
    return producer, {**envelope, "type": "contextual"}


def test_a_view_without_a_map_needs_an_earlier_view_and_leaves_out_both_parts(fake_engine):
    producer, envelope = _ready(fake_engine)
    observation = contract.Observation((2, 2, 1), (contract.Tile((0, 0, 0), 1, 0, False, False),), (), (0,) * 7, (), (),
                                       army_information=producer.army_information, own_armies=())
    whole = contract.encode_observation(observation, schema=producer.profile.history_schema)
    bare = {name: value for name, value in whole.items() if name not in ("tiles", "objects")}
    with pytest.raises(ValueError, match="Incomplete observation"):  # the first view has nothing to keep
        producer.ingest({**envelope, "records": [{"kind": "view", "day": 1, "observation": bare}]})
    first, = producer.ingest({**envelope, "records": [{"kind": "view", "day": 1, "observation": whole}]})
    for part in ("tiles", "objects"):  # one part alone is no unchanged map
        with pytest.raises(ValueError, match="Incomplete observation"):
            producer.ingest({**envelope, "records": [{"kind": "view", "day": 1, "observation": {**bare, part: whole[part]}}]})
    second, = producer.ingest({**envelope, "records": [{"kind": "view", "day": 2, "observation": bare}]})
    assert second.observation == first.observation and second.observation.tiles is first.observation.tiles


def test_a_checked_tile_tuple_is_checked_again_for_another_map_size():
    tiles = (contract.Tile((0, 0, 0), 1, 0, False, False), contract.Tile((3, 3, 0), 1, 0, False, False))
    contract.Observation((4, 4, 1), tiles, (), (0,) * 7, (), ())
    contract.Observation((4, 4, 1), tiles, (), (0,) * 7, (), ())  # the same tuple again: checked once
    with pytest.raises(ValueError, match="outside public map bounds"):
        contract.Observation((3, 3, 1), tiles, (), (0,) * 7, (), ())
    with pytest.raises(ValueError, match="visible tile"):  # an object beside the checked tiles is still found
        contract.Observation((4, 4, 1), tiles, (contract.ObjectView("object-0", (1, 1, 0), 5, 0, "neutral"),),
                             (0,) * 7, (), ())


def test_a_decision_without_an_observation_needs_a_view(fake_engine):
    from heroes.engine.protocol import Producer
    engine = fake_engine()
    producer = Producer("1", 0, interface=engine.interface, army_information=engine.army_information)
    envelope = {"protocol_version": producer.profile.protocol, "episode": "1", "player": 0}
    producer.ingest({**envelope, "type": "ready", "interface": producer.profile.environment,
                     "history_schema": producer.profile.history_schema, "action_schema": producer.profile.action_schema,
                     "army_information": engine.army_information})
    with pytest.raises(ValueError, match="needs an earlier view"):
        producer.ingest({**envelope, "type": "contextual",
                         "records": [{"kind": "decision", "day": 1, "decision_id": 0, "revision": 0}]})
