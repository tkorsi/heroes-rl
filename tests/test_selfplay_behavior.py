from copy import deepcopy
from dataclasses import replace
import unittest

from experiments.learner import contextual as c
from experiments.learner.selfplay_behavior import SCHEMA, aggregate_episodes, travel_metrics
from tests.test_evaluation_behavior import Frames
from tests.test_contextual_management import observation


class SelfPlayBehaviorTests(unittest.TestCase):
    def test_rejected_route_is_not_travel_and_initial_tiles_are_not_exploration(self):
        frames = Frames()
        obs = observation()
        seq = frames.decision(obs)
        frames.submit(seq, "move_to", "hero", x=2, y=1)
        frames.add("result", decision_seq=seq, result=c.Result("no_route_known", True, (1, 1, 0), 100, False))
        frames.cutoff(obs)
        row = travel_metrics(frames.frames)
        self.assertEqual(row["own_position_changes"], 0)
        self.assertEqual(row["newly_observed_tiles_after_initial"], 0)
        self.assertEqual(row["days_without_observed_travel_reveal_or_capture"], [1])
        self.assertEqual(row["movement"]["results"], {"no_route_known": 1})

    def test_outward_move_and_return_are_travel_but_only_return_is_revisit(self):
        frames = Frames()
        obs = observation()
        frames.add("view", observation=obs)
        for position in ((2, 1, 0), (1, 1, 0)):
            seq = frames.decision(obs)
            frames.submit(seq, "move_to", "hero", x=position[0], y=position[1])
            obs = replace(obs, heroes=(replace(obs.heroes[0], position=position), obs.heroes[1]))
            frames.add("view", observation=obs)
            frames.add("result", decision_seq=seq, result=c.Result("reached", True, position, 80, True))
        frames.cutoff(obs)
        row = travel_metrics(frames.frames)
        self.assertEqual(row["own_position_changes"], 2)
        self.assertEqual(row["position_changes_revisiting_observed_hero_position"], 1)
        self.assertEqual(row["days_without_observed_travel_reveal_or_capture"], [])

    def test_cutoffs_are_unscored_and_both_players_do_not_double_game_count(self):
        frames = Frames()
        frames.cutoff(observation())
        metrics = travel_metrics(frames.frames)
        rows = []
        for i, (status, outcome) in enumerate((("completed", "win"), ("truncated", None))):
            rows.append({"schema": SCHEMA, "episode_directory": str(i), "status": status,
                         "termination_reason": "natural_terminal" if outcome else "collection_window",
                         "learner_outcome": outcome, "players": {
                             "red": {"role": "learner", "metrics": deepcopy(metrics)},
                             "blue": {"role": "frozen_opponent", "metrics": deepcopy(metrics)}}})
        result = aggregate_episodes(rows)
        self.assertEqual(result["physical_games"], 2)
        self.assertEqual(result["natural_games"], 1)
        self.assertEqual(result["cutoff_games"], 1)
        self.assertEqual(result["learner_outcomes"], {"win": 1, "unscored_cutoff": 1})
        self.assertEqual(result["roles"]["learner"]["choices"]["available_games"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_episodes([rows[0], rows[0]])


if __name__ == "__main__":
    unittest.main()
