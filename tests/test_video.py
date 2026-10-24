"""Real ffmpeg integration against explicitly synthetic frames, not game/RL evidence."""
from __future__ import annotations

import importlib.util
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
                  "status": "timeout", "outcome": None}
        lines = video.overlay_lines(source)
        self.assertIn("RECORDER SMOKE (NO RL TRAINING)", lines[0])
        self.assertIn("OUTCOME: UNKNOWN", lines[2])
        source["kind"] = "rl-evaluation"
        self.assertIn("RL POLICY EVALUATION", video.overlay_lines(source)[0])


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
            report = video.montage([runs], output, seconds=1, samples=2)
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
                video.montage([runs], output, seconds=1, samples=2)
            all_output = root / "synthetic-all.mp4"
            all_report = video.montage([runs], all_output, seconds=0.1, samples=0)
            self.assertTrue(all_report["full_source_episode_coverage"])
            self.assertEqual(all_report["output_video"]["frames"], 3)
            self.assertEqual([r["output_frames"] for r in all_report["selected"]], [1, 1, 1])
            (runs / "2" / "frames").mkdir()
            (runs / "2" / "frames" / "recording-error.txt").write_text("synthetic recorder failure")
            available, skipped = video.discover([runs])
            self.assertEqual([r["episode"] for r in available], [1, 3])
            self.assertEqual(skipped[0]["reason"], "Native recorder reported an error")


if __name__ == "__main__":
    unittest.main()
