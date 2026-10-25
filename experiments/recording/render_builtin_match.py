#!/usr/bin/env python3
"""Present an already verified builtin-opponent match as a schematic MP4.

This is an offline map reconstruction, not footage of the game window. It
preserves every native trace frame, including automatic-battle frames, without
interpolating positions or inventing gameplay. Requires ffmpeg and the optional
Pillow dependency: pip install -r dependencies/video-viewer.lock.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .match_army import build_hero_army_timeline
else:
    from match_army import build_hero_army_timeline

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("recording_video", HERE / "video.py")
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)

WIDTH, HEIGHT, FPS = 1600, 1248, 15
TILE, MAP_TOP = 31, 278
BACKGROUND, CARD = "#101722", "#182332"
TEXT, MUTED = "#edf3fc", "#acbbd0"
BLUE, RED, GOLD = "#63a7ff", "#ff777d", "#ffda77"
OWNER = {0: RED, 1: BLUE}
TERRAIN = [(111, 91, 67), (195, 175, 118), (75, 119, 67), (216, 224, 234),
           (65, 88, 68), (120, 104, 82), (85, 77, 70), (110, 49, 39),
           (42, 82, 136), (55, 59, 67)]
HERO, TOWN, MINE, MONSTER, RESOURCE = 34, 98, 53, 54, 79


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise video.VideoError(message)


def inspect_inputs(episode: Path, trace: Path, native_report: Path) -> dict:
    manifest_path = episode / "manifest.json"
    original_path = episode / "builtin-verification/builtin-replay-verification.json"
    manifest, original, native = load(manifest_path), load(original_path), load(native_report)
    require(original.get("verified") is True, "Original builtin replay did not verify")
    require(native.get("verified") is True and native.get("divergence") is None,
            "Fresh native replay did not verify")
    require(native.get("schema") == "heroes-contextual-builtin-replay-verification-v1",
            "Expected the builtin contextual native replay report")
    require(original["inputs"]["manifest"]["sha256"] == digest(manifest_path),
            "Original verified manifest hash changed")
    require(Path(native["log"]).resolve() == (episode / "engine/replay.jsonl").resolve(),
            "Fresh native report belongs to another capture")
    require(not any(native["summary"]["unchecked"].values()), "Native replay has unchecked entries")
    require(native["summary"]["server_verified"] is True, "Native server did not verify")
    require(native["summary"]["checked"] == original["native"]["summary"]["checked"],
            "Fresh and original replay counts disagree")
    require(manifest.get("status") == "completed" and manifest.get("termination_reason") == "natural_terminal",
            "This presentation requires a completed natural match")
    require(manifest.get("training_performed") is False, "Expected an evaluation-only match")
    require(manifest.get("learner_player") == 1, "This presentation labels the blue learner")
    require(manifest.get("outcomes_by_player") == {"red": "win", "blue": "loss"},
            "This presentation labels a verified blue loss")
    opponent = manifest["opponent_identity"]
    require(opponent.get("module") == "Nullkiller" and opponent.get("difficulty") == 0,
            "Expected Nullkiller on Easy")
    history_path = episode / "agents/blue/history-v7.jsonl"
    require(digest(history_path) == original["history"]["learner"]["sha256"],
            "Original verified learner history hash changed")
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    require(history[-1].get("kind") == "terminal" and history[-1].get("outcome") == "loss",
            "Learner history does not close with a loss")
    actions = {}
    for row in history:
        if row.get("kind") == "submitted" and row.get("action"):
            kind = row["action"]["kind"]
            actions[kind] = actions.get(kind, 0) + 1
    require(sum(actions.values()) == manifest["env_steps"], "Learner action count disagrees with manifest")
    windows = sum(actions.get(kind, 0) for kind in ("open_hero", "open_town", "open_panel", "close", "back"))
    army = sum(actions.get(kind, 0) for kind in ("split_stack", "swap_stacks", "swap_garrison"))
    action_summary = (f"Learner's full match: {windows} window actions | {army} army changes | "
                      f"{actions.get('move_to', 0)} move | {actions.get('end_turn', 0)} end turn")
    plan_path = episode.parent.parent / "plan.json"
    plan = load(plan_path)
    require(plan.get("checkpoint_provenance") == "random_initialization", "Expected random initial weights")
    require(plan["configuration"]["optimizer_updates"] == 0, "Evaluation plan performed optimization")
    map_path = episode / "profile/Maps/headless.vmap"
    require(digest(map_path) == manifest["identity"]["map_sha256"], "Captured map hash changed")
    with zipfile.ZipFile(map_path) as archive:
        map_header = json.loads(archive.read("header.json"))
    require(map_header["players"]["blue"]["heroes"]["hero_1"]["type"] == "core:edric",
            "Expected blue Edric")
    require(map_header["players"]["red"]["heroes"]["hero_0"]["type"] == "core:orrin",
            "Expected red Orrin")
    for color in ("red", "blue"):
        require(map_header["players"][color]["allowedFactions"] == {"anyOf": ["core:castle"]},
                "Expected Castle on both sides")
    lines = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    header, frames = lines[0], lines[1:]
    require(header.get("kind") == "map" and (header["width"], header["height"], header["levels"]) == (20, 20, 1),
            "Expected a 20 by 20 single-level native map trace")
    inputs = native["summary"]["request_accounting"]["authoritative_inputs"]
    require([frame.get("seq") for frame in frames] == list(range(-1, inputs)),
            "Trace must contain the initial state and every authoritative input exactly once")
    require(all(frame.get("kind") == "frame" for frame in frames), "Invalid trace frame")
    require(frames[-1].get("boundary") == "game_end", "Trace does not reach game end")
    for frame in frames:
        require(len(frame["players"]["1"]["visible"]) == 400, "Invalid learner visibility trace")
    generation_path = trace.parent / "generation.json"
    generation = load(generation_path)
    require(generation.get("returncode") == 0 and generation.get("native_verified") is True,
            "Native trace generation did not succeed")
    require(Path(generation["episode"]).resolve() == episode.resolve(), "Trace receipt belongs to another episode")
    command = generation["command"]
    for option, expected in (("--log", episode / "engine/replay.jsonl"), ("--trace", trace),
                             ("--contextual", episode / "engine/contextual-replay.jsonl"), ("--report", native_report)):
        require(option in command and Path(command[command.index(option) + 1]).resolve() == expected.resolve(),
                f"Trace generation receipt has a different {option}")
    for path_text, expected_hash in generation["sha256"].items():
        require(digest(Path(path_text)) == expected_hash, f"Trace generation input changed: {path_text}")
    for path in (trace, native_report, manifest_path, original_path, history_path, map_path,
                 episode / "engine/replay.jsonl", episode / "engine/contextual-replay.jsonl"):
        require(generation["sha256"].get(str(path)) == digest(path), f"Missing trace receipt binding: {path}")
    paths = {"manifest": manifest_path, "original_builtin_verification": original_path,
             "learner_history": history_path, "plan": plan_path, "map": map_path,
             "server_capture": episode / "engine/replay.jsonl",
             "contextual_capture": episode / "engine/contextual-replay.jsonl",
             "native_report": native_report, "native_trace": trace, "generation_receipt": generation_path,
             "army_annotation_source": HERE / "match_army.py",
             "renderer": Path(__file__)}
    return {"manifest": manifest, "header": header, "frames": frames, "history": history,
            "hero_army": build_hero_army_timeline(episode, frames, player=1),
            "native": native, "plan": plan, "map_name": map_header["name"],
            "action_counts": actions, "action_summary": action_summary,
            "provenance": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()}}


def font(size: int, bold: bool = False):
    candidates = [Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


FONTS = {size: font(size) for size in (18, 20, 22, 24, 26, 30)}
FONTS[42] = font(42, True)


def text(draw, position, value, size=22, color=TEXT):
    draw.text(position, value, font=FONTS[size], fill=color)


def marker(draw, cx: int, cy: int, kind: int, owner: int, radius: int = 11):
    color = OWNER.get(owner, "#d4dce7")
    if kind == TOWN:
        draw.rectangle((cx - 14, cy - 14, cx + 14, cy + 14), fill="#27313f", outline=color, width=3)
    elif kind == HERO:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline="white", width=2)
    elif kind == MINE:
        draw.polygon([(cx, cy - 11), (cx - 11, cy + 9), (cx + 11, cy + 9)], fill="#b7c6d8")
    elif kind == MONSTER:
        draw.line((cx - 8, cy - 8, cx + 8, cy + 8), fill="#f4b294", width=4)
        draw.line((cx + 8, cy - 8, cx - 8, cy + 8), fill="#f4b294", width=4)
    elif kind == RESOURCE:
        draw.polygon([(cx, cy - 9), (cx - 9, cy), (cx, cy + 9), (cx + 9, cy)], fill=GOLD)
    else:
        draw.rectangle((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)


def panel(draw, left: int, header: dict, frame: dict, fog: bool):
    visible = frame["players"]["1"]["visible"]
    object_ids = set(frame["players"]["1"]["objects"])
    for index, terrain in enumerate(header["terrain"]):
        x, y = index % 20, index // 20
        color = TERRAIN[terrain] if 0 <= terrain < len(TERRAIN) else (80, 80, 80)
        if fog and visible[index] != "1":
            color = (6, 11, 19)
        draw.rectangle((left + x * TILE, MAP_TOP + y * TILE,
                        left + (x + 1) * TILE - 1, MAP_TOP + (y + 1) * TILE - 1), fill=color)
    for line in range(21):
        draw.line((left + line * TILE, MAP_TOP, left + line * TILE, MAP_TOP + 620), fill="#24303b", width=1)
        draw.line((left, MAP_TOP + line * TILE, left + 620, MAP_TOP + line * TILE), fill="#24303b", width=1)
    # Towns are drawn before heroes so a hero visiting a town keeps both markers.
    for entry in sorted(frame["objects"], key=lambda row: row[4] == HERO):
        object_id, x, y, level, kind, _, owner = entry
        if level != 0 or not (0 <= x < 20 and 0 <= y < 20) or (fog and object_id not in object_ids):
            continue
        marker(draw, left + x * TILE + TILE // 2, MAP_TOP + y * TILE + TILE // 2, kind, owner)
    for x, y, level, *sides in frame.get("battles", []):
        if level == 0 and (not fog or (1 in sides and visible[x + 20 * y] == "1")):
            draw.rectangle((left + x * TILE - 3, MAP_TOP + y * TILE - 3,
                            left + (x + 1) * TILE + 3, MAP_TOP + (y + 1) * TILE + 3), outline=GOLD, width=4)
    for index in (0, 5, 10, 15, 19):
        text(draw, (left + index * TILE + 5, MAP_TOP - 25), str(index), 18, MUTED)
        text(draw, (left - 28, MAP_TOP + index * TILE + 4), str(index), 18, MUTED)


def event_label(frame: dict, previous: dict | None) -> str:
    if frame.get("boundary") == "game_end":
        return "NATURAL RESULT: red Nullkiller wins; blue learner loses"
    if previous is None:
        return "Day 1: both heroes start at their Castle towns"
    if frame.get("battles"):
        return "Automatic battle: BattleAI resolves combat"
    old = {row[0]: row for row in previous["objects"]}
    for row in frame["objects"]:
        before = old.get(row[0])
        if before and row[4] == TOWN and before[-1] != row[-1]:
            return f"Town captured by {'red' if row[-1] == 0 else 'blue'}"
        if before and row[4] == HERO and before[1:4] != row[1:4]:
            return f"{'Blue learner' if row[-1] == 1 else 'Red Nullkiller'} hero moves to ({row[1]}, {row[2]})"
    if frame.get("boundary") == "end_turn":
        return "Turn completed"
    return "Recorded engine state after a native input"


def hero_army_lines(row: dict) -> tuple[str, str]:
    """Describe the viewed hero without presenting pre-battle counts as live losses."""
    if row["status"] == "absent":
        return (f"Viewed hero: {row['name']} / Blue / Castle - no longer on the map",
                "Army: no active hero army remains on the map")
    x, y, _ = row["position"]
    hero = f"Viewed hero: {row['name']} / Blue / Castle    Position: ({x}, {y})"
    totals = {}
    for stack in row["slots"]:
        totals[stack["creature"]] = totals.get(stack["creature"], 0) + stack["count"]
    names = {0: "Pikemen"}
    units = ", ".join(f"{count} {names.get(creature, f'creature #{creature}')}"
                      for creature, count in sorted(totals.items())) or "empty"
    by_slot = {stack["slot"]: stack["count"] for stack in row["slots"]}
    slots = " / ".join(str(by_slot[slot]) if slot in by_slot else "-" for slot in range(7))
    if row["status"] == "battle":
        prefix = "Army entering automatic battle"
    elif row["freshness"] == "pre_battle":
        prefix = "Last known army (before battle)"
    else:
        prefix = "Army"
    return hero, f"{prefix}: {units}    Seven slots: {slots}"


def draw_frame(data: dict, frame: dict, previous: dict | None) -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, (44, 23), "VERIFIED REAL MATCH  /  SCHEMATIC RECONSTRUCTION", 20, GOLD)
    text(draw, (44, 57), "Castle vs Castle", 42)
    text(draw, (44, 113), f"{data['map_name']}  |  20 x 20  |  Episode {data['manifest']['episode']}", 24, MUTED)
    text(draw, (44, 154), "BLUE  Edric / frozen random v7 network", 26, BLUE)
    text(draw, (821, 154), "RED  Orrin / Nullkiller Easy", 26, RED)
    for left in (44, 821):
        draw.rounded_rectangle((left, 202, left + 735, 920), radius=18, fill=CARD)
    text(draw, (70, 220), "FULL MAP", 26)
    text(draw, (244, 224), "Spectator view only", 20, MUTED)
    text(draw, (847, 220), "LEARNER FOG VIEW", 26, BLUE)
    text(draw, (1133, 224), "Recorded visibility", 20, MUTED)
    panel(draw, 104, data["header"], frame, False)
    panel(draw, 881, data["header"], frame, True)
    total = data["native"]["summary"]["request_accounting"]["authoritative_inputs"]
    text(draw, (44, 936), f"DAY {frame['day']}    Native input {max(0, frame['seq'] + 1)} / {total}", 26)
    text(draw, (920, 940), f"Learner choices in this match: {data['manifest']['env_steps']}", 22, BLUE)
    label = event_label(frame, previous)
    text(draw, (44, 977), label, 26, GOLD if frame.get("battles") or frame.get("boundary") == "game_end" else TEXT)
    legend = [(55, HERO, "Hero"), (210, TOWN, "Town"), (370, MINE, "Mine"),
              (530, MONSTER, "Neutral army"), (780, RESOURCE, "Resource")]
    for x, kind, label in legend:
        marker(draw, x, 1030, kind, 1 if kind in (HERO, TOWN) else -1, 9)
        text(draw, (x + 24, 1018), label, 20, MUTED)
    text(draw, (1010, 1018), "Black = outside learner fog view", 20, MUTED)
    hero_line, army_line = hero_army_lines(data["hero_army"][frame["seq"]])
    text(draw, (44, 1056), hero_line, 24, BLUE)
    text(draw, (44, 1093), army_line, 24)
    text(draw, (44, 1142), data["action_summary"], 20, BLUE)
    text(draw, (44, 1176), "Offline map replay, not game-window footage. No PPO update. Window actions are not animated.", 20, MUTED)
    progress = (frame["seq"] + 1) / max(1, total)
    draw.rectangle((44, 1230, 1556, 1235), fill="#2a384b")
    if progress:
        draw.rectangle((44, 1230, 44 + int(1512 * progress), 1235), fill=BLUE)
    return image.tobytes()


def frame_repeats(frames: list[dict], seconds: float = 56.0) -> list[int]:
    """Keep every trace state; give visible changes and the result more screen time."""
    weights = []
    previous = None
    for index, frame in enumerate(frames):
        changed = previous is None or frame["objects"] != previous["objects"] or frame["players"] != previous["players"]
        weight = 2.8 if changed else 0.5
        if frame.get("battles"):
            weight = max(weight, 0.7)
        if frame.get("boundary") == "end_turn":
            weight = max(weight, 3.0)
        if index == 0:
            weight = 8.0
        if index == len(frames) - 1:
            weight = 13.0
        weights.append(weight)
        previous = frame
    budget = int(seconds * FPS)
    require(len(frames) <= budget, "Too many trace states for the presentation budget")
    remaining = budget - len(frames)
    portions = [remaining * weight / sum(weights) for weight in weights]
    repeats = [1 + int(portion) for portion in portions]
    for index in sorted(range(len(frames)), key=lambda i: portions[i] % 1, reverse=True)[:budget - sum(repeats)]:
        repeats[index] += 1
    return repeats


def render(episode: Path, trace: Path, native_report: Path, output: Path) -> dict:
    episode, trace, native_report, output = [path.expanduser().resolve() for path in (episode, trace, native_report, output)]
    data = inspect_inputs(episode, trace, native_report)
    video.prepare_output(output, sidecar=True)
    staged = output.with_suffix(".partial.mp4")
    require(not staged.exists(), f"Staged output already exists: {staged}")
    repeats = frame_repeats(data["frames"])
    command = [video.binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-xerror", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(staged)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    previous = None
    try:
        for frame, count in zip(data["frames"], repeats):
            pixels = draw_frame(data, frame, previous)
            for _ in range(count):
                process.stdin.write(pixels)
            previous = frame
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        process.stderr.close()
        if process.wait() != 0:
            raise video.VideoError(f"ffmpeg failed: {error[-2000:]}")
        probe = video.check_video(staged, sum(repeats), FPS)
        os.replace(staged, output)
    except BaseException:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        process.kill()
        process.wait()
        staged.unlink(missing_ok=True)
        raise
    result = {"schema": "heroes-builtin-schematic-video-v1", "output": str(output), "sha256": digest(output),
              "episode_directory": str(episode), "episode": data["manifest"]["episode"],
              "presentation": "Offline schematic map reconstruction; not game-window footage",
              "map": data["map_name"], "size": [20, 20, 1], "learner": "blue Edric, Castle, frozen random v7",
              "opponent": "red Orrin, Castle, Nullkiller Easy", "outcomes": {"blue": "loss", "red": "win"},
              "training_performed": False, "checkpoint_sha256": data["plan"]["checkpoint"]["sha256"],
              "native_inputs": len(data["frames"]) - 1, "learner_choices": data["manifest"]["env_steps"],
              "history_action_counts": data["action_counts"], "all_trace_frames_preserved": True,
              "viewed_hero_army_by_native_input": data["hero_army"],
              "army_annotation_scope": "Viewer-only. Initial map and packet-bound own-window snapshots; automatic combat shows the army entering battle, not live tactical losses.",
              "trace_frames": len(data["frames"]), "trace_frame_repeats": repeats, "fps": FPS,
              "views": ["omniscient spectator", "blue recorded fog"], "provenance": data["provenance"],
              "original_builtin_replay_verified": True, "fresh_native_replay_verified": True,
              "verification_scope": "Native trace-generation receipt binds the successful command, source captures, verified report and trace by hash; the original builtin replay remains unchanged.",
              "limitations": ["Window actions are not animated. Army slots update from recorded snapshots.", "Automatic battles are map markers; the army panel shows pre-battle counts, not live tactical losses.",
                              "Presentation time is slowed and does not measure simulation speed.", "The fog panel is an engine map view, not the network's complete structured input."],
              "ffprobe": probe}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "trace", "native-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = render(args.episode, args.trace, args.native_report, args.output)
    except (video.VideoError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Render failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
