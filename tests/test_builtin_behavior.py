"""One-seat descriptive metrics from synthetic manifest-bound histories."""
from pathlib import Path
import tempfile
import unittest

from experiments.learner import builtin_behavior as behavior
from tests.test_builtin_replay import HistoryFixture, write


class BuiltinBehaviorTests(unittest.TestCase):
    def test_red_and_blue_count_only_the_actual_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for player in (0, 1):
                root = Path(tmp) / str(player)
                fixture = HistoryFixture(root, player, natural=True)
                result = behavior.analyze_episode(root)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(set(result["players"]), {fixture.color})
                self.assertEqual(result["aggregate"]["expected_player_histories"], 1)
                self.assertEqual(result["aggregate"]["choices"], 1)
                self.assertIsNone(result["opponent_history"])
                results.append(result)
            total = behavior.aggregate_episodes(results)
            self.assertEqual(total["aggregate"]["choices"], 2)
            self.assertEqual(total["aggregate"]["expected_player_histories"], 2)
            self.assertEqual(total["training_samples"], 0)

    def test_missing_history_is_unavailable_not_an_empty_native_opponent(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = HistoryFixture(Path(tmp), 1)
            (fixture.root / fixture.manifest["players"]["blue"]["history_v7"]["path"]).unlink()
            result = behavior.analyze_episode(fixture.root)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["aggregate"])
            self.assertIsNone(result["players"]["blue"]["metrics"])
            self.assertNotIn("red", result["players"])

    def test_wrong_seat_extra_opponent_and_hash_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = HistoryFixture(Path(tmp), 0)
            fixture.manifest["players"]["blue"] = {"history_v7": None}
            write(fixture.root / "manifest.json", fixture.manifest)
            with self.assertRaisesRegex(ValueError, "fabricated history"):
                behavior.analyze_episode(fixture.root)
            del fixture.manifest["players"]["blue"]
            write(fixture.root / "manifest.json", fixture.manifest)
            path = fixture.root / fixture.manifest["players"]["red"]["history_v7"]["path"]
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                behavior.analyze_episode(fixture.root)

    def test_missing_manifest_duplicate_episode_and_paired_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = behavior.analyze_episode(Path(tmp))
            self.assertIsNone(result["learner_player"])
            self.assertEqual(result["players"], {})
            self.assertIsNone(behavior.aggregate_episodes([result])["aggregate"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                behavior.aggregate_episodes([result, result])
            with self.assertRaisesRegex(ValueError, "denominators"):
                behavior.aggregate_episodes([{**result, "schema": "heroes-v7-evaluation-behavior-v1"}])


if __name__ == "__main__":
    unittest.main()
