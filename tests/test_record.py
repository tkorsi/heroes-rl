"""A truncated or failed recording must never become an RL loss/reward."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("record", Path(__file__).resolve().parents[1] / "experiments/recording/record.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class TerminationTests(unittest.TestCase):
    def test_verified_natural_outcomes(self):
        for word, outcome in (("won", "win"), ("lost", "loss")):
            self.assertEqual(record.classify_run(0, None, f"Red player {word}. Ending game."),
                             ("completed", outcome, "natural_terminal"))

    def test_limits_and_errors_are_unscored_even_if_terminal_log_arrived(self):
        for reason in ("wall_time_limit", "frame_limit", "recorder_error"):
            result = record.classify_run(-15, reason, "Red player lost. Ending game.")
            self.assertIsNone(result[1])
            self.assertNotEqual(result[0], "completed")

    def test_exit_without_game_result_or_crash_is_failure(self):
        for code, text in ((0, ""), (-11, "Red player won. Ending game."),
                           (0, "Red player won. Ending game.\nRed player lost. Ending game.")):
            self.assertEqual(record.classify_run(code, None, text)[0:2], ("failed", None))


if __name__ == "__main__":
    unittest.main()
