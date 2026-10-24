"""Offline replay renderer against an explicitly synthetic trace, not game evidence."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("render_replay", ROOT / "experiments/headless/render_replay.py")
render_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_replay)

WIDTH = HEIGHT = 20
TILE = render_replay.TILE
FAILING_FFMPEG = "#!{python}\nimport sys\nsys.stderr.write(\"Unknown encoder 'libx264'\\n\")\nsys.exit(1)\n"


def visible(columns):
    return "".join("1" if x in columns else "0" for y in range(HEIGHT) for x in range(WIDTH))


def synthetic_trace(frames=12):
    header = {"kind": "map", "width": WIDTH, "height": HEIGHT, "levels": 1,
              "terrain": [(x + y) % 10 for y in range(HEIGHT) for x in range(WIDTH)]}
    rows = []
    for seq in range(-1, frames - 1):
        blue_x = 14 - max(0, seq)  # the blue hero walks west into red's sight
        objects = [[1, 3, 7, 0, 34, 0, 0], [2, blue_x, 7, 0, 34, 1, 1], [3, 4, 7, 0, 98, 0, 0], [4, 6, 12, 0, 54, 0, -1]]
        red_columns, blue_columns = set(range(0, 8)), set(range(blue_x - 2, blue_x + 3))
        rows.append({"kind": "frame", "seq": seq, "day": 1 + max(0, seq) // 4,
                     "boundary": "game_end" if seq == frames - 2 else ("end_turn" if seq == 3 else None),
                     "battles": [[blue_x, 7, 0, 1, 0]] if seq >= frames - 4 else [], "objects": objects,
                     "players": {"0": {"visible": visible(red_columns),
                                       "objects": [o[0] for o in objects if o[1] in red_columns]},
                                 "1": {"visible": visible(blue_columns),
                                       "objects": [o[0] for o in objects if o[1] in blue_columns]}}})
    return header, rows


def glimpse_trace(frames=41):
    """Red's hero moves every frame; blue's hero stands on a red-visible tile only at frame 7."""
    red_columns, blue_columns = set(range(0, 5)), set(range(4, WIDTH))
    rows = []
    for index in range(frames):
        objects = [[1, 0, index % HEIGHT, 0, 34, 0, 0], [2, 4 if index == 7 else 5, 5, 0, 34, 1, 1]]
        rows.append({"kind": "frame", "seq": index - 1, "day": 1, "boundary": None, "battles": [], "objects": objects,
                     "players": {"0": {"visible": visible(red_columns),
                                       "objects": [o[0] for o in objects if o[1] in red_columns]},
                                 "1": {"visible": visible(blue_columns),
                                       "objects": [o[0] for o in objects if o[1] in blue_columns]}}})
    return rows


def changed_frames(kinds):
    """One changed frame per character: b battle, e game end, anything else plain."""
    return [{"kind": "frame", "seq": index - 1, "day": 1, "boundary": "game_end" if kind == "e" else None,
             "battles": [[1, 1, 0, 0, -1]] if kind == "b" else [], "objects": [[1, index, 0, 0, 34, 0, 0]],
             "players": {"0": {"visible": visible(set()), "objects": []}}} for index, kind in enumerate(kinds)]


def pixel(canvas, x, y):
    start = (y * canvas.width + x) * 3
    return tuple(canvas.pixels[start:start + 3])


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
        self.assertLessEqual(len(selected), 10)
        self.assertIn(next(i for i, f in enumerate(frames) if f["boundary"] == "game_end"), selected)

    def test_frame_selection_keeps_the_ending_within_the_budget(self):
        for kinds, budget in (("b" * 27 + "." * 5 + "e", 13), ("b" * 21 + "e", 10), ("b" * 15, 10), ("." * 50, 10),
                              ("b" * 21 + "e..", 10), ("b" * 5 + "e.", 2)):
            with self.subTest(kinds=kinds, budget=budget):
                frames = changed_frames(kinds)
                selected = render_replay.select_frames(frames, budget)
                self.assertLessEqual(len(selected), budget)
                self.assertEqual(selected, sorted(set(selected)))
                self.assertIn(len(frames) - 1, selected)  # the final hold shows the true last state
                if "e" in kinds:
                    self.assertIn(kinds.index("e"), selected)  # and the game end banner

    def test_thinning_draws_a_sighting_that_ends_between_drawn_frames(self):
        header, _ = synthetic_trace()
        frames = glimpse_trace()
        selected = render_replay.select_frames(frames, 10)
        self.assertLessEqual(len(selected), 10)
        renderer = render_replay.TraceRenderer(header, {})
        shown = False
        for index, frame in enumerate(frames):  # as render() walks the trace
            objects = {entry[0]: tuple(entry[1:]) for entry in frame["objects"]}
            if index in selected:
                panel = renderer.player_panel(0, frame, objects)
                shown |= pixel(panel, 4 * TILE + 8, 5 * TILE + 8) == render_replay.OWNER_COLORS[1]
            else:
                renderer.update_memory(0, frame, objects)
        self.assertTrue(shown)  # red's fog view shows blue's hero at least once

    def test_fog_views_mark_only_battles_the_player_fights(self):
        header, frames = synthetic_trace()
        renderer = render_replay.TraceRenderer(header, {})
        frame = json.loads(json.dumps(frames[0]))
        frame["players"]["0"]["visible"] = frame["players"]["1"]["visible"] = visible(set(range(WIDTH)))
        objects = {entry[0]: tuple(entry[1:]) for entry in frame["objects"]}
        marked = lambda panel: pixel(panel, 6 * TILE - 1, 12 * TILE + 8) == render_replay.BATTLE  # noqa: E731
        # Blue against the neutral stack on tile (6, 12), red against blue, and a legacy entry without sides.
        for battle, red, blue in (([6, 12, 0, 1, -1], False, True), ([6, 12, 0, 0, 1], True, True),
                                  ([6, 12, 0], False, False)):
            frame["battles"] = [battle]
            with self.subTest(battle=battle):
                self.assertTrue(marked(renderer.omniscient_panel(frame, objects)))
                self.assertEqual(marked(renderer.player_panel(0, frame, objects)), red)
                self.assertEqual(marked(renderer.player_panel(1, frame, objects)), blue)

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

    def test_reports_ffmpegs_error_when_it_exits_early(self):
        episode = self.episode()
        tools = self.base / "bin"
        tools.mkdir()
        ffmpeg = tools / "ffmpeg"
        ffmpeg.write_text(FAILING_FFMPEG.format(python=sys.executable))
        ffmpeg.chmod(ffmpeg.stat().st_mode | stat.S_IXUSR)
        output = self.base / "episode-1.mp4"
        with mock.patch.dict(os.environ, {"PATH": f"{tools}{os.pathsep}{os.environ.get('PATH', '')}"}):
            with self.assertRaises(render_replay.video.VideoError) as caught:
                render_replay.render(episode, output, fps=10, max_seconds=5, hold_seconds=1)
        self.assertIn("Unknown encoder 'libx264'", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, BrokenPipeError)
        self.assertFalse(output.with_suffix(".partial.mp4").exists())
        self.assertFalse(output.exists())

    def test_refuses_unverified_replays(self):
        episode = self.episode(verified=False)
        with self.assertRaises(render_replay.video.VideoError):
            render_replay.render(episode, self.base / "x.mp4")


if __name__ == "__main__":
    unittest.main()
