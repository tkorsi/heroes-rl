"""Real ffmpeg integration against explicitly synthetic frames, not game/RL evidence."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "recording" / "video.py"
SPEC = importlib.util.spec_from_file_location("recording_video", MODULE_PATH)
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


def synthetic_png(path: Path, color: tuple[int, int, int], width=32, height=24):
    """A solid RGB fixture; deliberately not represented as a VCMI frame."""
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    pixels = b"".join(b"\0" + bytes(color) * width for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def synthetic_capture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)], 1):
        filename = f"frame{index:06d}.png"
        synthetic_png(directory / filename, color)
        records.append({"index": index, "file": filename, "elapsed_seconds": (index - 1) / 6,
                        "width": 32, "height": 24})
    (directory / "frames.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.records = synthetic_capture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite(self):
        (self.directory / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_rejects_frame_gap_and_extra_frames(self):
        self.records[1]["index"] = 3
        self.rewrite()
        with self.assertRaisesRegex(video.VideoError, "gap"):
            video.validate_frames(self.directory)
        self.records[1]["index"] = 2
        self.rewrite()
        synthetic_png(self.directory / "frame999999.png", (0, 0, 0))
        with self.assertRaisesRegex(video.VideoError, "do not match"):
            video.validate_frames(self.directory)

    def test_rejects_mismatched_dimensions_and_reversed_time(self):
        synthetic_png(self.directory / "frame000002.png", (0, 0, 0), width=34)
        with self.assertRaisesRegex(video.VideoError, "dimensions"):
            video.validate_frames(self.directory)
        synthetic_png(self.directory / "frame000002.png", (0, 0, 0))
        self.records[2]["elapsed_seconds"] = 0
        self.rewrite()
        with self.assertRaisesRegex(video.VideoError, "timestamps"):
            video.validate_frames(self.directory)

    def test_error_marker_is_fatal_and_limit_marker_is_valid(self):
        (self.directory / "recording-limit-reached.txt").write_text("bounded capture completed")
        self.assertEqual(len(video.validate_frames(self.directory)), 3)
        (self.directory / "recording-error.txt").write_text("capture write failed")
        with self.assertRaisesRegex(video.VideoError, "Recorder reported an error"):
            video.validate_frames(self.directory)

    def test_sampling_covers_endpoints_and_uses_exact_frame_budget(self):
        self.assertEqual(video.select_indices(1000, 3), [0, 500, 999])
        self.assertEqual(video.select_indices(4, 0), [0, 1, 2, 3])
        budgets = video.frame_budgets(1800, 1000)
        self.assertEqual(sum(budgets), 1800)
        self.assertEqual(min(budgets), 1)
        with self.assertRaisesRegex(video.VideoError, "frame budget"):
            video.frame_budgets(30, 31)

    def test_smoke_overlay_never_claims_learning(self):
        source = {"episode": 1, "kind": "recorder-smoke", "policy": {"label": "Synthetic fixture"},
                  "status": "timeout", "outcome": None,
                  "perspective": {"mode": "player", "player": "red", "player_index": 0}}
        lines = video.overlay_lines(source)
        self.assertIn("RECORDER SMOKE (NO RL TRAINING)", lines[0])
        self.assertIn("PLAYER RED | FOG OF WAR", lines[1])
        self.assertIn("OUTCOME: UNKNOWN", lines[3])
        source["kind"] = "rl-evaluation"
        self.assertIn("RL POLICY EVALUATION", video.overlay_lines(source)[0])
        source["perspective"] = {"mode": "omniscient", "player": None, "player_index": None}
        self.assertIn("OMNISCIENT | FULL MAP", video.overlay_lines(source)[1])

    def test_paired_capture_identity_and_nested_error_markers(self):
        omni = copy.deepcopy(self.records)
        player = copy.deepcopy(self.records)
        for stream, view, index in ((omni, "omniscient", None), (player, "player", 0)):
            for i, record in enumerate(stream, 1):
                record.update(sample_id=i, game_day=0, view=view, player_index=index, visible_tiles=12)
        result = video.validate_paired_frames(omni, player)
        self.assertTrue(result["shared_sample_ids_verified"])
        player[1]["visible_tiles"] = 13
        with self.assertRaisesRegex(video.VideoError, "visible_tiles"):
            video.validate_paired_frames(omni, player)
        player[1]["visible_tiles"] = 12
        for record in player:
            record["sample_id"] += 10
        with self.assertRaisesRegex(video.VideoError, "Paired view sample_id"):
            video.validate_paired_frames(omni, player)
        player = copy.deepcopy(omni)
        for record in player:
            record.update(view="player", player_index=0)
        player[1]["player_index"] = 1
        with self.assertRaisesRegex(video.VideoError, "identity changes"):
            video.validate_frame_identity(player)
        nested = self.directory / "frames" / "player"
        synthetic_capture(nested)
        (nested.parent / "recording-error.txt").write_text("common recorder failure")
        with self.assertRaisesRegex(video.VideoError, "Recorder reported an error"):
            video.validate_frames(nested)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires real ffmpeg and ffprobe")
class EncodingIntegrationTests(unittest.TestCase):
    def test_encoding_montage_provenance_duration_and_no_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "synthetic-frames"
            synthetic_capture(capture)
            originals = {p.name: p.read_bytes() for p in capture.iterdir()}
            source = root / "fixture.mp4"
            info = video.encode(capture, source, fps=6)
            self.assertEqual(info["frames"], 3)
            self.assertAlmostEqual(info["duration_seconds"], 0.5, places=3)
            self.assertEqual(originals, {p.name: p.read_bytes() for p in capture.iterdir()})
            with self.assertRaisesRegex(video.VideoError, "overwrite"):
                video.encode(capture, source)
            runs = root / "runs"
            for episode, outcome in [(1, "loss"), (2, "win"), (3, "loss")]:
                run = runs / str(episode)
                run.mkdir(parents=True)
                shutil.copy2(source, run / "source.mp4")
                (run / "manifest.json").write_text(json.dumps({
                    "schema_version": 1, "kind": "recorder-smoke", "episode": episode,
                    "policy": {"label": "SYNTHETIC TEST FIXTURE", "checkpoint": None},
                    "status": "completed", "outcome": outcome,
                    "capture": {"fps": 6, "frames": 3, "elapsed_seconds": 1 / 3}, "video": "source.mp4"}))
            output = root / "synthetic-reel.mp4"
            report = video.montage([runs], output, seconds=1, samples=2, view="omniscient")
            self.assertEqual(report["output_video"]["frames"], 30)
            self.assertAlmostEqual(report["output_video"]["duration_seconds"], 1, places=3)
            self.assertEqual([r["episode"] for r in report["selected"]], [1, 3])
            self.assertEqual([r["episode"] for r in report["skipped"]], [2])
            self.assertFalse(report["full_source_episode_coverage"])
            self.assertEqual(report["discovered_episode_count"], 3)
            self.assertTrue(all(r["video_sha256"] == video.sha256(source) for r in report["selected"]))
            self.assertTrue(all("NO RL TRAINING" in r["overlay_lines"][0] for r in report["selected"]))
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text()), report)
            with self.assertRaisesRegex(video.VideoError, "overwrite"):
                video.montage([runs], output, seconds=1, samples=2, view="omniscient")
            all_output = root / "synthetic-all.mp4"
            all_report = video.montage([runs], all_output, seconds=0.1, samples=0, view="omniscient")
            self.assertTrue(all_report["full_source_episode_coverage"])
            self.assertEqual(all_report["output_video"]["frames"], 3)
            self.assertEqual([r["output_frames"] for r in all_report["selected"]], [1, 1, 1])
            (runs / "2" / "frames").mkdir()
            (runs / "2" / "frames" / "recording-error.txt").write_text("synthetic recorder failure")
            available, skipped = video.discover([runs], view="omniscient")
            self.assertEqual([r["episode"] for r in available], [1, 3])
            self.assertEqual(skipped[0]["reason"], "Native recorder reported an error")

    def test_dual_view_selection_legacy_refusal_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "synthetic-frames"
            synthetic_capture(capture)
            source = root / "fixture.mp4"
            video.encode(capture, source)
            runs = root / "runs"
            dual = runs / "dual"
            dual.mkdir(parents=True)
            recordings = {}
            for view, player, index in (("omniscient", None, None), ("player", "red", 0)):
                filename = f"{view}.mp4"
                shutil.copy2(source, dual / filename)
                recordings[view] = {"video": filename, "capture": {"fps": 6, "frames": 3, "elapsed_seconds": 1 / 3},
                                    "perspective": {"mode": view, "player": player, "player_index": index}}
            base = {"schema_version": 2, "kind": "recorder-smoke", "episode": 2,
                    "policy": {"label": "SYNTHETIC TEST FIXTURE", "checkpoint": None},
                    "status": "timeout", "outcome": None, "outcome_player": "red", "recordings": recordings,
                    "visibility_validation": {"status": "synthetic-metadata-only", "fog_proven": False}}
            manifest = dual / "manifest.json"
            manifest.write_text(json.dumps(base))
            player_sources, skipped = video.discover([dual])
            self.assertEqual(skipped, [])
            self.assertEqual(len(player_sources), 1)
            self.assertEqual(player_sources[0]["perspective"]["player"], "red")
            self.assertEqual(Path(player_sources[0]["video"]).name, "player.mp4")
            omni_sources, _ = video.discover([dual], view="omniscient")
            self.assertEqual(Path(omni_sources[0]["video"]).name, "omniscient.mp4")
            self.assertEqual(omni_sources[0]["episode"], player_sources[0]["episode"])
            legacy = runs / "legacy"
            legacy.mkdir()
            shutil.copy2(source, legacy / "source.mp4")
            legacy_data = {**base, "schema_version": 1, "episode": 1, "video": "source.mp4", "capture": recordings["omniscient"]["capture"]}
            legacy_data.pop("recordings")
            (legacy / "manifest.json").write_text(json.dumps(legacy_data))
            with self.assertRaisesRegex(video.VideoError, "Legacy schema v1 has only an omniscient"):
                video.montage([legacy], root / "must-not-exist.mp4", seconds=1)
            self.assertFalse((root / "must-not-exist.mp4").exists())
            unavailable = runs / "unavailable"
            unavailable.mkdir()
            (unavailable / "manifest.json").write_text(json.dumps({**base, "episode": 3, "recordings": {}}))
            blue = runs / "blue"
            blue.mkdir()
            blue_data = copy.deepcopy(base)
            blue_data["episode"] = 4
            blue_data["recordings"]["player"]["perspective"] = {"mode": "player", "player": "blue", "player_index": 1}
            blue_data["recordings"].pop("omniscient")
            shutil.copy2(source, blue / "player.mp4")
            (blue / "manifest.json").write_text(json.dumps(blue_data))
            report = video.montage([runs], root / "player-reel.mp4", seconds=0.1)
            self.assertEqual(report["view"], "player")
            self.assertEqual(report["selected_episode_count"], 2)
            self.assertEqual(len(report["skipped"]), 2)
            self.assertEqual(report["selected"][0]["visibility_validation"], base["visibility_validation"])
            self.assertIn("PLAYER RED | FOG OF WAR", report["selected"][0]["overlay_lines"][1])
            self.assertIn("PLAYER BLUE | FOG OF WAR", report["selected"][1]["overlay_lines"][1])
            self.assertIn("OUTCOME RED: UNKNOWN", report["selected"][1]["overlay_lines"][3])
            marker = dual / "frames" / "player" / "recording-error.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("synthetic player capture error")
            invalid_sources, errors = video.discover([dual])
            self.assertEqual(invalid_sources, [])
            self.assertEqual(errors[0]["reason"], "Native recorder reported an error")
            self.assertEqual(len(video.discover([dual], view="omniscient")[0]), 1)
            marker.unlink()
            base["recordings"]["player"]["perspective"]["mode"] = "omniscient"
            manifest.write_text(json.dumps(base))
            with self.assertRaisesRegex(video.VideoError, "does not match"):
                video.discover([dual])
            base["recordings"]["player"]["perspective"] = {"mode": "player", "player": "red", "player_index": 1}
            manifest.write_text(json.dumps(base))
            with self.assertRaisesRegex(video.VideoError, "matching player color"):
                video.discover([dual])
            base["recordings"]["player"].pop("perspective")
            manifest.write_text(json.dumps(base))
            with self.assertRaisesRegex(video.VideoError, "perspective"):
                video.discover([dual])


if __name__ == "__main__":
    unittest.main()
