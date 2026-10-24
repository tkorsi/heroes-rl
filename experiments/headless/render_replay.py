#!/usr/bin/env python3
"""Render a verified replay trace as a labelled, synchronized video.

``replay.py verify --trace`` writes ``verification/trace.jsonl`` while the
engine replays an episode: one frame after every recorded input, with every
object's position and each player's currently visible tiles and objects. This
tool draws three synchronized views from it, entirely after the episode:

* omniscient: the whole map and every object, for verification only;
* each player's fog view: tiles the player sees now in full color, tiles it
  saw before dimmed with its last-seen objects, never-seen tiles black.

Last-seen memory changes only through the player's own sightings: an object's
remembered position updates when the player sees it, and the memory is cleared
only when the player sees that tile without it. Hidden moves never update it.
Simulation never waits on rendering; this runs on recorded traces only.

Python standard library + ffmpeg/ffprobe only, like experiments/recording/video.py.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_VIDEO_SPEC = importlib.util.spec_from_file_location("recording_video", ROOT / "experiments/recording/video.py")
video = importlib.util.module_from_spec(_VIDEO_SPEC)
_VIDEO_SPEC.loader.exec_module(video)
_REPLAY_SPEC = importlib.util.spec_from_file_location("headless_replay_scope", HERE / "replay.py")
replay = importlib.util.module_from_spec(_REPLAY_SPEC)
_REPLAY_SPEC.loader.exec_module(replay)

# The replay verification report version whose manifest label checks the drawn labels rely on.
VERIFICATION_VERSION = 2
TILE = 16
GUTTER = 12
HEADER = 44
LABEL = 18
FOOTER = 26
COLORS = ("red", "blue")
OWNER_COLORS = {0: (230, 57, 70), 1: (58, 134, 255)}
NEUTRAL = (205, 205, 205)
BACKGROUND = (18, 18, 22)
TEXT = (235, 235, 235)
MUTED = (150, 150, 158)
BATTLE = (255, 214, 10)
# VCMI TerrainId order: dirt, sand, grass, snow, swamp, rough, subterranean, lava, water, rock.
TERRAIN = [(122, 94, 64), (206, 182, 120), (78, 132, 62), (226, 232, 240), (72, 96, 74), (132, 112, 84),
           (96, 86, 78), (120, 52, 40), (44, 92, 160), (60, 60, 60)]
HERO, TOWN, MINE, MONSTER, RESOURCE = 34, 98, 53, 54, 79

# 5x7 glyphs, one string of 7 rows of 5 bits per character.
FONT = {
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110",
    "C": "01110100011000010000100001000101110", "D": "11110100011000110001100011000111110",
    "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01110100011000010111100011000101111", "H": "10001100011000111111100011000110001",
    "I": "01110001000010000100001000010001110", "J": "00111000100001000010000101001001100",
    "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001100011100110101100111000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110", "V": "10001100011000110001100010101000100",
    "W": "10001100011000110101101011010101010", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111",
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    " ": "0" * 35, ":": "00000011000110000000011000110000000", "-": "00000000000000011111000000000000000",
    ".": "00000000000000000000000000110001100", ",": "00000000000000000000001100010001000",
    "/": "00001000010001000100010001000010000", "(": "00010001000100001000010000010000010",
    ")": "01000001000001000010000100010001000", "=": "00000000001111100000111110000000000",
    "#": "01010010101111101010111110101001010", "+": "00000001000010011111001000010000000",
}


class Canvas:
    def __init__(self, width: int, height: int, fill=BACKGROUND):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(fill) * (width * height))

    def rect(self, x: int, y: int, w: int, h: int, color) -> None:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            start = (yy * self.width + x0) * 3
            self.pixels[start:start + len(row)] = row

    def outline(self, x: int, y: int, w: int, h: int, color, thickness: int = 1) -> None:
        self.rect(x, y, w, thickness, color)
        self.rect(x, y + h - thickness, w, thickness, color)
        self.rect(x, y, thickness, h, color)
        self.rect(x + w - thickness, y, thickness, h, color)

    def text(self, x: int, y: int, value: str, color=TEXT, scale: int = 1) -> int:
        for character in value.upper():
            glyph = FONT.get(character, FONT[" "])
            for row in range(7):
                for column in range(5):
                    if glyph[row * 5 + column] == "1":
                        self.rect(x + column * scale, y + row * scale, scale, scale, color)
            x += 6 * scale
        return x

    def blit(self, other: "Canvas", x: int, y: int) -> None:
        for row in range(other.height):
            source = row * other.width * 3
            target = ((y + row) * self.width + x) * 3
            self.pixels[target:target + other.width * 3] = other.pixels[source:source + other.width * 3]


def dim(color, factor: float = 0.38):
    return tuple(int(channel * factor) for channel in color)


def glyph(canvas: Canvas, x: int, y: int, kind: int, owner: int, faded: bool = False) -> None:
    """One object marker inside the tile at pixel (x, y)."""
    color = OWNER_COLORS.get(owner, NEUTRAL)
    if faded:
        color = dim(color, 0.6)
    if kind == HERO:
        rim = (255, 255, 255) if not faded else dim((255, 255, 255), 0.5)
        for row, width in enumerate((6, 10, 12, 14, 14, 14, 14, 12, 10, 6)):
            canvas.rect(x + 8 - width // 2, y + 3 + row, width, 1, rim)
        for row, width in enumerate((4, 8, 10, 10, 10, 10, 8, 4)):
            canvas.rect(x + 8 - width // 2, y + 4 + row, width, 1, color)
    elif kind == TOWN:
        canvas.outline(x + 1, y + 1, 14, 14, color, 3)
    elif kind == MINE:
        for row in range(9):
            canvas.rect(x + 8 - row // 2 - 1, y + 4 + row, row + 2, 1, color)
    elif kind == MONSTER:
        shade = (176, 40, 40) if not faded else (96, 32, 32)
        for step in range(10):
            canvas.rect(x + 3 + step, y + 3 + step, 2, 2, shade)
            canvas.rect(x + 12 - step, y + 3 + step, 2, 2, shade)
    elif kind == RESOURCE:
        canvas.rect(x + 5, y + 5, 6, 6, (250, 204, 21) if not faded else (125, 102, 10))
    else:
        canvas.rect(x + 6, y + 6, 4, 4, color)


class TraceRenderer:
    def __init__(self, header: dict, manifest: dict):
        self.width, self.height, self.levels = header["width"], header["height"], header["levels"]
        if self.levels != 1:
            raise video.VideoError("The renderer draws single-level maps")
        self.manifest = manifest
        self.terrain = header["terrain"]
        size = TILE * max(self.width, self.height)
        self.panel = size
        self.frame_width = 3 * size + 4 * GUTTER
        self.frame_height = HEADER + LABEL + size + FOOTER
        self.frame_width += self.frame_width % 2
        self.frame_height += self.frame_height % 2
        self.full = self._terrain_panel(1.0)
        self.memory_terrain = self._terrain_panel(0.42)
        self.memory = {player: {} for player in range(len(COLORS))}  # player -> id -> (x, y, type, owner)
        self.seen = {player: set() for player in range(len(COLORS))}
        self.battle_frames = 0

    def _terrain_panel(self, factor: float) -> Canvas:
        panel = Canvas(self.panel, self.panel)
        for y in range(self.height):
            for x in range(self.width):
                terrain = self.terrain[x + self.width * y]
                color = TERRAIN[terrain] if 0 <= terrain < len(TERRAIN) else (90, 90, 90)
                panel.rect(x * TILE, y * TILE, TILE, TILE, dim(color, factor) if factor != 1.0 else color)
        return panel

    def update_memory(self, player: int, frame: dict, objects: dict) -> tuple[str, set]:
        view = frame["players"][str(player)]
        visible = view["visible"]
        visible_ids = set(view.get("objects", []))
        tiles = {index for index, flag in enumerate(visible) if flag == "1"}
        self.seen[player] |= tiles
        memory = self.memory[player]
        for object_id in visible_ids:
            x, y, _, kind, _, owner = objects[object_id]
            memory[object_id] = (x, y, kind, owner)
        for object_id in [oid for oid, (x, y, _, _) in memory.items()
                          if oid not in visible_ids and x + self.width * y in tiles]:
            del memory[object_id]  # the player sees the tile and the object is not there
        return visible, visible_ids

    def player_panel(self, player: int, frame: dict, objects: dict) -> Canvas:
        visible, visible_ids = self.update_memory(player, frame, objects)
        panel = Canvas(self.panel, self.panel, (0, 0, 0))
        for index in self.seen[player]:
            x, y = index % self.width, index // self.width
            source = self.full if visible[index] == "1" else self.memory_terrain
            for row in range(TILE):
                start = ((y * TILE + row) * self.panel + x * TILE) * 3
                panel.pixels[start:start + TILE * 3] = source.pixels[start:start + TILE * 3]
        for object_id, (x, y, kind, owner) in sorted(self.memory[player].items()):
            glyph(panel, x * TILE, y * TILE, kind, owner, faded=object_id not in visible_ids)
        for x, y, _, *sides in frame.get("battles", []):
            if player in sides and visible[x + self.width * y] == "1":  # only the two sides learn of a battle
                panel.outline(x * TILE - 2, y * TILE - 2, TILE + 4, TILE + 4, BATTLE, 2)
        return panel

    def omniscient_panel(self, frame: dict, objects: dict) -> Canvas:
        panel = Canvas(self.panel, self.panel)
        panel.pixels[:] = self.full.pixels
        for object_id, (x, y, _, kind, _, owner) in sorted(objects.items()):
            if 0 <= x < self.width and 0 <= y < self.height:
                glyph(panel, x * TILE, y * TILE, kind, owner)
        for x, y, *_ in frame.get("battles", []):
            panel.outline(x * TILE - 2, y * TILE - 2, TILE + 4, TILE + 4, BATTLE, 2)
        return panel

    def render(self, frame: dict, total_inputs: int) -> bytes:
        objects = {entry[0]: tuple(entry[1:]) for entry in frame["objects"]}
        canvas = Canvas(self.frame_width, self.frame_height)
        episode = self.manifest.get("episode")
        canvas.text(GUTTER, 8, f"EPISODE {episode}  SEED {self.manifest.get('seed')}  DAY {frame['day']}", TEXT, 2)
        status = f"INPUT {max(frame['seq'] + 1, 0)}/{total_inputs}"
        boundary = frame.get("boundary")
        if boundary == "game_end":
            outcomes = self.manifest.get("outcomes_by_player") or {}
            winner = next((color for color, outcome in outcomes.items() if outcome == "win"), None)
            status += f"  GAME END: {winner.upper()} WINS" if winner else "  GAME END"
        elif boundary:
            status += "  " + boundary.replace("_", " ").upper()
        if frame.get("battles"):
            status += "  BATTLE"
            self.battle_frames += 1
        canvas.text(GUTTER, 30, status, BATTLE if frame.get("battles") or boundary == "game_end" else MUTED)
        labels = ("OMNISCIENT - VERIFICATION ONLY", "RED PLAYER FOG VIEW", "BLUE PLAYER FOG VIEW")
        panels = (self.omniscient_panel(frame, objects), self.player_panel(0, frame, objects),
                  self.player_panel(1, frame, objects))
        for index, (label, panel) in enumerate(zip(labels, panels)):
            x = GUTTER + index * (self.panel + GUTTER)
            color = TEXT if index == 0 else OWNER_COLORS[index - 1]
            canvas.text(x, HEADER + 4, label, color)
            canvas.blit(panel, x, HEADER + LABEL)
        canvas.text(GUTTER, HEADER + LABEL + self.panel + 9,
                    "OFFLINE RECONSTRUCTION FROM A VERIFIED REPLAY. DIMMED: LAST SEEN BY THAT PLAYER. BLACK: NEVER SEEN.",
                    MUTED)
        return bytes(canvas.pixels)


def read_trace(path: Path, data: bytes | None = None) -> tuple[dict, list[dict]]:
    text = (path.read_bytes() if data is None else data).decode()
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not lines or lines[0].get("kind") != "map" or any(line.get("kind") != "frame" for line in lines[1:]):
        raise video.VideoError(f"Not a replay trace: {path}")
    return lines[0], lines[1:]


def select_frames(frames: list[dict], max_frames: int) -> list[int]:
    """Frames whose drawn content changed, thinned evenly to the budget.

    The game end and the final trace frame are always kept. Battles, battle ends and new sightings (a player sees
    an object it did not see on the previous frame) come next, strided evenly only if they alone overflow.
    """
    def content(frame):
        return (json.dumps(frame["objects"]), json.dumps(frame["players"]), json.dumps(frame.get("battles")), frame["day"])
    if not frames:
        return []
    changed, forced, previous, sighted = [], set(), None, {}
    for index, frame in enumerate(frames):
        key = content(frame)
        if key != previous or frame.get("boundary"):
            changed.append(index)
        previous = key
        for player, view in frame["players"].items():
            ids = set(view.get("objects", []))
            if ids - sighted.get(player, set()):
                forced.add(index)  # a sighting can end before the next drawn frame
            sighted[player] = ids
    ends = {index for index in changed if frames[index].get("boundary") == "game_end"} | {len(frames) - 1}
    ends = set(sorted(ends)[-max_frames:])
    keep = ends | set(changed)
    if len(keep) <= max_frames:
        return sorted(keep)
    forced |= {index for index in changed if frames[index].get("boundary") in ("game_end", "battle_end")}
    forced |= {index for index in changed if frames[index].get("battles")}
    forced = sorted(forced - ends)
    room = max_frames - len(ends)
    if len(forced) > room:
        forced = forced[:: math.ceil(len(forced) / room)] if room else []
    keep = ends | set(forced)
    rest = [index for index in changed if index not in keep]
    budget = max(0, max_frames - len(keep))
    if budget and rest:
        step = len(rest) / budget
        keep |= {rest[int(i * step)] for i in range(budget)}
    return sorted(keep)


def render(episode_dir: Path, output: Path, *, fps: float = 15.0, max_seconds: float = 90.0, hold_seconds: float = 3.0) -> dict:
    episode_dir = episode_dir.expanduser().resolve()
    verification = json.loads((episode_dir / "verification" / "replay-verification.json").read_text())
    if not verification.get("verified"):
        raise video.VideoError("Render only episodes whose replay verified")
    if verification.get("schema_version") != VERIFICATION_VERSION:
        raise video.VideoError("The verification predates the manifest label checks; verify again")
    # Render exactly the bytes the verification hashed; a changed manifest or trace fails closed.
    trace_path = episode_dir / "verification" / "trace.jsonl"
    manifest_bytes, trace_bytes = (episode_dir / "manifest.json").read_bytes(), trace_path.read_bytes()
    inputs = verification.get("inputs") or {}
    hashes = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "trace_sha256": hashlib.sha256(trace_bytes).hexdigest()}
    changed = [key for key, value in hashes.items() if inputs.get(key) != value]
    if changed:
        raise video.VideoError(f"Inputs differ from what the replay verified ({', '.join(changed)}); verify again")
    manifest = json.loads(manifest_bytes)
    try:
        replay.require_legacy_interface(manifest)
    except replay.ReplayError as error:
        raise video.VideoError(str(error)) from None
    header, frames = read_trace(trace_path, trace_bytes)
    renderer = TraceRenderer(header, manifest)
    budget = max(1, int(fps * max_seconds))
    selected = select_frames(frames, budget)
    hold = int(fps * hold_seconds)
    output = output.expanduser().resolve()
    video.prepare_output(output)
    staged = output.with_suffix(".partial.mp4")
    staged.unlink(missing_ok=True)
    command = [video.binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-xerror", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{renderer.frame_width}x{renderer.frame_height}", "-framerate", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(staged)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total_inputs = max((frame["seq"] for frame in frames), default=-1) + 1
    written = 0
    try:
        # Memory follows every recorded frame, drawn or not. select_frames keeps every frame with a new
        # sighting, so thinning skips one only when forced frames alone overflow the budget.
        wanted = set(selected)
        last = None
        for index, frame in enumerate(frames):
            if index in wanted:
                last = renderer.render(frame, total_inputs)
                process.stdin.write(last)
                written += 1
            else:
                objects = {entry[0]: tuple(entry[1:]) for entry in frame["objects"]}
                for player in range(len(COLORS)):
                    renderer.update_memory(player, frame, objects)
        for _ in range(hold if last is not None else 0):
            process.stdin.write(last)
            written += 1
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        process.stderr.close()
        if process.wait() != 0:
            raise video.VideoError(f"ffmpeg failed: {error[-2000:]}")
    except BrokenPipeError as broken:  # ffmpeg exited before taking every frame; its stderr says why
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        process.stderr.close()
        process.wait()
        staged.unlink(missing_ok=True)
        raise video.VideoError(f"ffmpeg failed: {error[-2000:]}") from broken
    except BaseException:
        process.kill()
        process.wait()
        staged.unlink(missing_ok=True)
        raise
    info = video.check_video(staged, written, fps)
    os.replace(staged, output)
    result = {"schema_version": 1, "kind": "headless-replay-video", "episode": manifest["episode"],
              "status": manifest["status"], "termination_reason": manifest["termination_reason"],
              "outcomes": manifest.get("outcomes_by_player"), "output": str(output), "sha256": video.sha256(output),
              "fps": fps, "frames_written": written, "trace_frames": len(frames), "frames_drawn": len(selected),
              "hold_frames": hold, "battle_frames_drawn": renderer.battle_frames,
              "views": ["omniscient (verification only)", "red fog", "blue fog"],
              "replay_verified": True, **hashes,
              **{key: info[key] for key in ("width", "height", "codec", "pixel_format", "duration_seconds")}}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path, help="An episode directory verified with replay.py verify --trace")
    parser.add_argument("--output", type=Path, required=True, help="New .mp4 file")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=float, default=90.0, help="Budget before the final hold")
    args = parser.parse_args(argv)
    try:
        result = render(args.episode, args.output, fps=args.fps, max_seconds=args.max_seconds)
    except (video.VideoError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Render failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
