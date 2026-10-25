"""Pure selector and streamed evidence tests; no engine or simulated episodes."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.learner.contextual import Hero, ObjectView, Observation, Tile, Town
from validation.probes.audit_ordinary_behavior import summarize_player
from validation.probes.validate_town_capture import attacker_destination, defender_destination


def observed():
    return Observation((20, 20, 1), tuple(Tile(pos, 2, 0, False, False) for pos in
                       ((4, 7, 0), (2, 8, 0), (5, 8, 0), (9, 9, 0), (8, 9, 0), (15, 7, 0))),
                       (), (0,) * 7, (Hero("hero", (4, 7, 0), 1000, 1000),), (Town("town", (4, 7, 0)),))


class CaptureSelectionTests(unittest.TestCase):
    def test_only_current_empty_displayed_tiles_selected(self):
        obs = observed()
        self.assertEqual(defender_destination(obs), (2, 8, 0))
        self.assertEqual(attacker_destination(obs, set()), ((9, 9, 0), "displayed_empty_centre_tile"))
        self.assertEqual(attacker_destination(obs, {(9, 9, 0)})[0], (8, 9, 0))
        blocked = replace(obs, tiles=tuple(replace(tile, blocked=True) for tile in obs.tiles))
        self.assertIsNone(defender_destination(blocked))
        self.assertIsNone(attacker_destination(blocked, set())[0])

    def test_visible_enemy_town_takes_precedence_but_hidden_memory_does_not(self):
        obs = replace(observed(), objects=(ObjectView("enemy", (15, 7, 0), 98, 0, "opponent"),))
        self.assertEqual(attacker_destination(obs, set()), ((15, 7, 0), "currently_observed_enemy_town"))
        self.assertEqual(attacker_destination(replace(obs, objects=()), set())[0], (9, 9, 0))

    def test_defender_geometry_mirrors_without_fixed_enemy_coordinate(self):
        obs = observed()
        mirror = lambda pos: (19 - pos[0], pos[1], pos[2])
        other = replace(obs, tiles=tuple(replace(tile, position=mirror(tile.position)) for tile in obs.tiles),
                        heroes=tuple(replace(hero, position=mirror(hero.position)) for hero in obs.heroes),
                        towns=tuple(replace(town, position=mirror(town.position)) for town in obs.towns))
        self.assertEqual(defender_destination(other), mirror(defender_destination(obs)))


class VisibleAuditTests(unittest.TestCase):
    def write(self, directory, frames):
        path = Path(directory) / "history.jsonl"
        data = b"".join((json.dumps(frame) + "\n").encode() for frame in frames)
        path.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    def frames(self):
        obs = asdict(observed())
        # This deliberately is a descriptive record fragment, not a synthetic
        # proof of a complete game or valid terminal/finalization contract.
        payloads = [("decision", {"observation": obs}),
                    ("submitted", {"decision_seq": 0, "action": {"kind": "move_to", "subject": "hero", "x": 12, "y": 12}}),
                    ("result", {"decision_seq": 0, "result": {"status": "no_route_known", "accepted": True}})]
        return [dict(episode="visible-test", player=0, schema="heroes-player-history-v3", seq=seq, day=1, kind=kind, **extra)
                for seq, (kind, extra) in enumerate(payloads)]

    def test_unknown_destination_is_not_mislabeled_as_known_or_enemy(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.write(directory, self.frames())
            report = summarize_player(path, 0, sha)
        self.assertEqual(report["movement"]["unknown_targets"], 1)
        self.assertEqual(report["movement"].get("known_targets", 0), 0)
        self.assertEqual(report["movement"]["targeting_visible_opponent_town"], 0)
        self.assertEqual(report["movement"]["result_no_route_known"], 1)

    def test_mutated_hash_foreign_player_and_gap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.write(directory, self.frames())
            with self.assertRaises(ValueError):
                summarize_player(path, 0, "0" * 64)
            with self.assertRaises(ValueError):
                summarize_player(path, 1, sha)
            frames = self.frames()
            frames[-1]["seq"] = 10
            path, sha = self.write(directory, frames)
            with self.assertRaises(ValueError):
                summarize_player(path, 0, sha)

    def test_closed_hero_army_remains_historical_after_map_refresh(self):
        frames = self.frames()
        window = {"context": {"kind": "hero", "subject": "hero"}, "army": [{"slot": 0, "creature": 0, "count": 25}]}
        frames[0]["observation"]["window"] = window
        frames.append(dict(episode="visible-test", player=0, schema="heroes-player-history-v3", seq=3, day=8,
                           kind="view", observation=asdict(observed())))
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.write(directory, frames)
            report = summarize_player(path, 0, sha)
        historical = report["last_explicit_hero_window_armies_historical_only"]["hero"]
        self.assertEqual((historical["day"], historical["sequence"]), (1, 0))
        self.assertEqual(report["last_observed_day"], 8)

    def test_successful_recruitment_uses_selected_displayed_destination_and_exact_count(self):
        frames = self.frames()
        frames[0]["observation"]["window"] = {
            "context": {"kind": "town", "subject": "town"}, "town_panel": {
                "recruitment": {"destination": {"ref": "upper", "role": "town_garrison"}}}}
        frames[1]["action"] = {"kind": "recruit", "destination": "upper", "count": 7}
        frames[2]["result"] = {"status": "recruited", "accepted": True}
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.write(directory, frames)
            report = summarize_player(path, 0, sha)
            self.assertEqual(report["successful_recruitment"], {"actions": 1, "creatures": 7, "town_garrison_creatures": 7})
            frames[1]["action"]["destination"] = "undisplayed-hero"
            path, sha = self.write(directory, frames)
            with self.assertRaises(ValueError):
                summarize_player(path, 0, sha)


if __name__ == "__main__":
    unittest.main()
