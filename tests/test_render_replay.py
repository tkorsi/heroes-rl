"""Offline replay renderer against an explicitly synthetic trace, not game evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("render_replay", ROOT / "experiments/headless/render_replay.py")
render_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_replay)

WIDTH = HEIGHT = 20


def visible(columns):
    return "".join("1" if x in columns else "0" for y in range(HEIGHT) for x in range(WIDTH))


def synthetic_trace(frames=12):
    header = {"kind": "map", "width": WIDTH, "height": HEIGHT, "levels": 1,
              "terrain": [(x + y) % 10 for y in range(HEIGHT) for x in range(WIDTH)]}
    rows = []
    for seq in range(-1, frames - 1):
        blue_x = 14 - max(0, seq)  # the blue hero walks west into red's sight
        objects = [[1, 3, 7, 0, 34, 0, 0], [2, blue_x, 7, 0, 34, 1, 1], [3, 4, 7, 0, 98, 0, 0], [4, 6, 12, 0, 54, 0, 255]]
        red_columns, blue_columns = set(range(0, 8)), set(range(blue_x - 2, blue_x + 3))
        rows.append({"kind": "frame", "seq": seq, "day": 1 + max(0, seq) // 4,
                     "boundary": "game_end" if seq == frames - 2 else ("end_turn" if seq == 3 else None),
                     "battles": [[blue_x, 7, 0]] if seq >= frames - 4 else [], "objects": objects,
                     "players": {"0": {"visible": visible(red_columns),
                                       "objects": [o[0] for o in objects if o[1] in red_columns]},
                                 "1": {"visible": visible(blue_columns),
                                       "objects": [o[0] for o in objects if o[1] in blue_columns]}}})
    return header, rows


class RenderReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-render-test-")
        self.base = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def episode(self, verified=True):
        episode = self.base / "episode-000001"
        (episode / "verification").mkdir(parents=True)
        header, frames = synthetic_trace()
        (episode / "verification" / "trace.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in [header, *frames]))
        (episode / "verification" / "replay-verification.json").write_text(json.dumps({"verified": verified}))
        (episode / "manifest.json").write_text(json.dumps({"episode": 1, "seed": 7, "status": "completed",
                                                           "termination_reason": "natural_terminal",
                                                           "outcomes_by_player": {"red": "win", "blue": "loss"}}))
        return episode

    def test_memory_changes_only_through_the_players_own_sightings(self):
        header, frames = synthetic_trace()
        renderer = render_replay.TraceRenderer(header, {})
        objects = lambda frame: {entry[0]: tuple(entry[1:]) for entry in frame["objects"]}  # noqa: E731
        renderer.update_memory(1, frames[0], objects(frames[0]))
        self.assertIn(2, renderer.memory[1])
        self.assertNotIn(1, renderer.memory[1])  # the red hero was never in blue's sight
        renderer.update_memory(0, frames[0], objects(frames[0]))
        self.assertNotIn(2, renderer.memory[0])
        for frame in frames:
            renderer.update_memory(0, frame, objects(frame))
        self.assertEqual(renderer.memory[0][2][:2], (14 - (len(frames) - 2), 7))
        # A remembered object stays where it was last seen while its tile is hidden...
        moved = json.loads(json.dumps(frames[-1]))
        moved["objects"][1][1] = 19
        moved["players"]["0"] = {"visible": visible(set(range(0, 4))), "objects": [1, 3]}
        renderer.update_memory(0, moved, objects(moved))
        self.assertEqual(renderer.memory[0][2][:2], (14 - (len(frames) - 2), 7))
        # ...and is forgotten only once the player sees that tile without it.
        moved["players"]["0"] = {"visible": visible(set(range(0, 8))), "objects": [1, 3]}
        renderer.update_memory(0, moved, objects(moved))
        self.assertNotIn(2, renderer.memory[0])

    def test_frame_selection_keeps_boundaries_and_battles(self):
        _, frames = synthetic_trace(frames=40)
        selected = render_replay.select_frames(frames, 10)
        self.assertLessEqual(len(selected), 10 + 1)
        self.assertIn(next(i for i, f in enumerate(frames) if f["boundary"] == "game_end"), selected)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires ffmpeg and ffprobe")
    def test_renders_a_labelled_synchronized_h264_video(self):
        episode = self.episode()
        output = self.base / "videos" / "episode-1.mp4"
        output.parent.mkdir()
        result = render_replay.render(episode, output, fps=10, max_seconds=5, hold_seconds=1)
        self.assertEqual((result["codec"], result["pixel_format"]), ("h264", "yuv420p"))
        self.assertEqual(result["frames_written"], result["frames_drawn"] + 10)
        self.assertGreater(result["battle_frames_drawn"], 0)
        self.assertEqual(result["width"] % 2, 0)
        self.assertTrue(output.with_suffix(".json").is_file())

    def test_refuses_unverified_replays(self):
        episode = self.episode(verified=False)
        with self.assertRaises(render_replay.video.VideoError):
            render_replay.render(episode, self.base / "x.mp4")


if __name__ == "__main__":
    unittest.main()
