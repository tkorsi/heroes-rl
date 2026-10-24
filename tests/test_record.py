"""A truncated or failed recording must never become an RL loss/reward."""
import importlib.util
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

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


def synthetic_png(path: Path) -> None:
    """A tiny explicit test fixture, never VCMI or learning evidence."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + b"\x40\x80\xc0" * 2) * 2)) + chunk(b"IEND", b""))


class VisibilityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.metadata = {"schema_version": 1, "phase": "before_first_turn", "game_day": 0,
                         "cheats_allowed": False, "requested_no_cheats": True, "difficulty_index": 0,
                         "map_width": 4, "map_height": 4, "map_levels": 1,
                         "players": [{"player_index": i, "team_index": i, "human": False,
                                      "visible_tiles": 3, "total_tiles": 16} for i in (0, 1)]}
        self.log = "\n".join(f"Experiment AI visibility: player={i} cheats_allowed=0 open_map=0 use_object_graph=0"
                              for i in (0, 1))
        self.manifest = {"schema_version": 2, "status": "timeout", "outcome": None,
                         "termination_reason": "frame_limit", "outcome_player": "red", "outcome_player_index": 0,
                         "recording_valid": False, "recordings": {}}
        self.streams = {}
        for view in ("omniscient", "player"):
            directory = self.output / "frames" / view
            directory.mkdir(parents=True)
            self.manifest["recordings"][view] = {"capture": {"fps": 6, "frames": 0}, "video": None,
                                                  "recording_valid": False}
            entries = []
            for i in (1, 2):
                filename = f"frame{i:06d}.png"
                synthetic_png(directory / filename)
                entries.append({"index": i, "file": filename, "width": 2, "height": 2,
                                "sample_id": i, "elapsed_seconds": (i - 1) / 6, "game_day": i - 1,
                                "view": view, "player_index": 1 if view == "player" else None,
                                "observer_player_index": 1, "visible_tiles": i + 2})
            self.streams[view] = entries
        self.write_evidence()

    def tearDown(self):
        self.temp.cleanup()

    def write_evidence(self):
        (self.output / "frames" / "experiment-metadata.json").write_text(json.dumps(self.metadata))
        for view, entries in self.streams.items():
            (self.output / "frames" / view / "frames.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    def test_day_zero_effective_settings_and_nonred_observer(self):
        evidence = record.validate_experiment_metadata(self.metadata, 1, self.log)
        self.assertEqual(evidence["observer_player_index"], 1)
        self.assertEqual(len(evidence["ai_players"]), 2)
        record.finalize_recordings(self.manifest, self.output, 1, self.log, no_encode=True)
        result = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(result["recording_valid"])
        self.assertEqual(result["outcome_player"], "red")
        self.assertEqual(result["paired_capture_validation"]["frames"], 2)
        for recording in result["recordings"].values():
            self.assertTrue(recording["recording_valid"])
            self.assertIsNone(recording["video"])

    def test_rejects_cheats_wrong_difficulty_and_missing_effective_evidence(self):
        for key, value in (("cheats_allowed", True), ("cheats_allowed", 0),
                           ("requested_no_cheats", False), ("requested_no_cheats", None),
                           ("difficulty_index", 1), ("players", [])):
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.metadata)
                changed[key] = value
                with self.assertRaises(ValueError):
                    record.validate_experiment_metadata(changed, 1, self.log)
        with self.assertRaisesRegex(ValueError, "absent"):
            record.validate_experiment_metadata(self.metadata, 2, self.log)

    def test_rejects_actual_ai_map_access_and_missing_ai_diagnostics(self):
        for log in (self.log.splitlines()[0], self.log.replace("open_map=0", "open_map=1", 1),
                    self.log.replace("use_object_graph=0", "use_object_graph=1", 1),
                    self.log.replace("cheats_allowed=0", "cheats_allowed=1", 1)):
            with self.subTest(log=log), self.assertRaises(ValueError):
                record.validate_experiment_metadata(self.metadata, 1, log)

    def test_rejects_missing_new_frame_metadata_and_wrong_observer(self):
        for entry in self.streams["omniscient"]:
            del entry["observer_player_index"]
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "missing native"):
            record.validate_capture_view(self.output / "frames" / "omniscient", "omniscient", 1, 16)
        for entry in self.streams["omniscient"]:
            entry["observer_player_index"] = 0
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            record.validate_capture_view(self.output / "frames" / "omniscient", "omniscient", 1, 16)

    def test_frame_or_pair_failure_is_saved_as_failed_and_unscored(self):
        self.manifest.update(status="completed", outcome="win", termination_reason="natural_terminal")
        self.streams["player"][1]["visible_tiles"] = 5
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "visible_tiles"):
            record.finalize_recordings(self.manifest, self.output, 1, self.log, no_encode=True)
        result = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["outcome"])
        self.assertFalse(result["recording_valid"])
        self.assertTrue(all(not view["recording_valid"] for view in result["recordings"].values()))

    def test_truncated_ledger_does_not_leave_running_manifest(self):
        self.manifest["status"] = "running"
        (self.output / "frames" / "player" / "frames.jsonl").write_text('{"index":')
        with self.assertRaises(ValueError):
            record.finalize_recordings(self.manifest, self.output, 1, self.log, no_encode=True)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_native_error_log_is_fatal_even_without_writable_error_marker(self):
        self.assertFalse((self.output / "frames" / "recording-error.txt").exists())
        with self.assertRaisesRegex(ValueError, "Native recorder reported"):
            record.finalize_recordings(self.manifest, self.output, 1,
                                       self.log + "\nExperiment recording failed: cannot write next frame",
                                       no_encode=True)
        self.assertFalse(json.loads((self.output / "manifest.json").read_text())["recording_valid"])

    def test_single_player_view_is_valid_without_inventing_a_pair(self):
        del self.manifest["recordings"]["omniscient"]
        record.finalize_recordings(self.manifest, self.output, 1, self.log, no_encode=True)
        self.assertTrue(self.manifest["recording_valid"])
        self.assertNotIn("paired_capture_validation", self.manifest)
        self.assertEqual(set(self.manifest["recordings"]), {"player"})

    def test_encoding_refuses_existing_video_and_fails_closed(self):
        existing = self.output / "omniscient.mp4"
        existing.write_bytes(b"pre-existing output must survive")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            record.finalize_recordings(self.manifest, self.output, 1, self.log, no_encode=False)
        self.assertEqual(existing.read_bytes(), b"pre-existing output must survive")
        result = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["recording_valid"])


if __name__ == "__main__":
    unittest.main()
