"""Synthetic Tavern cutoff reconstruction; no native-game validation claim."""
from pathlib import Path
import tempfile
import unittest

from experiments.headless.contextual_protocol import ContextualProducer
from experiments.learner import contextual as c
from tests.test_contextual_heroes import action, observation, tavern
from tests.test_contextual_replay_history import Fixture


class TavernCutoffFixture(Fixture):
    """Exercise real producer history transitions and persisted bridge replay."""
    def __init__(self, directory, continuing_player):
        self.directory, self.profile = directory, c.HEROES_PROFILE
        self.producers = [ContextualProducer("1", player, interface=self.profile.environment) for player in (0, 1)]
        self.rows, self.natural, self.outcomes = [[], []], False, ["win", "loss"]
        self.identity = {"engine_revision": "synthetic-only", "map_sha256": "a" * 64}
        for player in (0, 1):
            (directory / "agents" / ("red", "blue")[player]).mkdir(parents=True)
            self.engine(player, dict(protocol_version=7, type="ready", episode="1", player=player,
                interface=self.profile.environment, history_schema=c.HEROES_SCHEMA, action_schema=c.HEROES_ACTION_SCHEMA))
            self.batch(player, self.held(0, 0, observation()))
            self.navigate(player, 0, action("open_town", subject="town"), observation("overview"), 1, "opened")
            self.navigate(player, 1, action("open_panel", subject="town", panel="tavern"), observation(tavern()), 3, "opened")
            self.producers[player].freeze_cutoff("collection_window")
            if player == continuing_player:
                self.navigate(player, 2, action("back", subject="town"), observation("overview"), 5, "closed", outside=True)
                self.navigate(player, 3, action("close"), observation(), 7, "closed", outside=True)
                self.submit(player, action("end_turn"), outside=True)
                self.batch(player, dict(kind="consumed", day=1, decision_id=4, consumer="engine"),
                    dict(kind="issued", day=1, decision_id=4, command_index=0),
                    dict(kind="acknowledged", day=1, decision_id=4, command_index=0, accepted=True),
                    dict(kind="result", day=1, decision_id=4, result=dict(status="turn_ended", accepted=True,
                        position=None, movement=None, destination_reached=None)))
        self.write()

    def held(self, native, revision, obs):
        return dict(kind="decision", day=1, decision_id=native, revision=revision,
                    observation=c.encode_observation(obs, schema=c.HEROES_SCHEMA))

    def navigate(self, player, native, selected, obs, revision, status, outside=False):
        self.submit(player, selected, outside=outside)
        self.batch(player, dict(kind="consumed", day=1, decision_id=native, consumer="adapter"),
            dict(kind="context", day=1, decision_id=native, context=c.encode_observation(obs, schema=c.HEROES_SCHEMA)["window"]["context"]
                 if obs.window is not None else {"kind": "map", "subject": None}, revision=revision),
            dict(kind="view", day=1, observation=c.encode_observation(obs, schema=c.HEROES_SCHEMA)),
            dict(kind="result", day=1, decision_id=native, result=dict(status=status, accepted=True,
                position=None, movement=None, destination_reached=None)), self.held(native + 1, revision + 1, obs))


class TavernCutoffReplayTests(unittest.TestCase):
    def test_both_colors_reconstruct_back_close_end_turn_outside_frozen_tavern_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            for player in (0, 1):
                fixture = TavernCutoffFixture(Path(temp) / str(player), player)
                original = {str(path.relative_to(fixture.directory)): path.read_bytes()
                            for path in fixture.directory.rglob("*") if path.is_file()}
                report = fixture.verify()
                self.assertTrue(report["verified"])
                self.assertFalse(report["native_replay_verified"])
                self.assertEqual(report["boundary"], "cutoff")
                self.assertEqual([report["players"][color]["continuation_steps"] for color in ("red", "blue")],
                                 [3 if index == player else 0 for index in (0, 1)])
                frozen = fixture.producers[player].history
                self.assertEqual(frozen.decision.observation.context.panel, "tavern")
                self.assertEqual([frame.action.kind for frame in frozen.frames if frame.kind == "submitted"],
                                 ["open_town", "open_panel"])
                self.assertEqual([frame.action.kind for frame in fixture.producers[player].live.frames if frame.kind == "submitted"],
                                 ["open_town", "open_panel", "back", "close", "end_turn"])
                self.assertEqual(original, {str(path.relative_to(fixture.directory)): path.read_bytes()
                                           for path in fixture.directory.rglob("*") if path.is_file()})

    def test_tavern_continuation_cannot_select_or_recruit_heroes(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("select_hero", "hire_hero"):
                fixture = TavernCutoffFixture(Path(temp) / kind, 0)
                def mutate(rows):
                    response = next(row["message"] for row in rows if row["direction"] == "policy"
                                    and row["message"]["outside_learner_prefix"])
                    response["action"] = c.encode_action(action(kind, subject="town", selection="offer-a"))
                fixture.change_bridge(mutate)
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "technical continuation"):
                    fixture.verify()


if __name__ == "__main__": unittest.main()
