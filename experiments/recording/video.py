#!/usr/bin/env python3
"""Encode VCMI captures and assemble transparent, chronological experiment reels.

Python standard library + ffmpeg/ffprobe only. Original captures are never edited.
The tiny bitmap overlay avoids relying on optional FFmpeg font libraries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib


class VideoError(ValueError):
    """Invalid recording or an unsuccessful video operation."""


def run(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise VideoError(f"Required executable unavailable: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise VideoError(f"{argv[0]} failed: {exc.stderr[-4000:]}") from exc
    return result.stdout


def binary(name: str) -> str:
    found = shutil.which(name)
    if not found and Path(f"/opt/homebrew/bin/{name}").is_file():
        found = f"/opt/homebrew/bin/{name}"
    if not found:
        raise VideoError(f"Install {name} and make it available on PATH")
    return found


def probe(path: Path) -> dict:
    data = json.loads(run([binary("ffprobe"), "-v", "error", "-count_frames",
                           "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = [s for s in data["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        raise VideoError(f"Expected exactly one video stream: {path}")
    stream = streams[0]
    try:
        return {"width": int(stream["width"]), "height": int(stream["height"]),
                "frames": int(stream["nb_read_frames"]),
                "duration_seconds": float(stream.get("duration", data["format"]["duration"])),
                "codec": stream["codec_name"], "pixel_format": stream["pix_fmt"]}
    except (KeyError, ValueError) as exc:
        raise VideoError(f"Incomplete video metadata: {path}") from exc


def check_video(path: Path, frames: int, fps: float) -> dict:
    info = probe(path)
    if info["frames"] != frames:
        raise VideoError(f"Encoded frame count {info['frames']} differs from expected {frames}")
    if info["codec"] != "h264" or info["pixel_format"] != "yuv420p":
        raise VideoError("Expected H.264 / yuv420p output")
    if abs(info["duration_seconds"] - frames / fps) > 1 / fps + 0.001:
        raise VideoError("Encoded duration differs from the requested frame budget")
    return info


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise VideoError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def validate_frames(directory: Path) -> list[dict]:
    if (directory / "recording-error.txt").exists():
        raise VideoError(f"Recorder reported an error: {directory / 'recording-error.txt'}")
    metadata = directory / "frames.jsonl"
    if not metadata.is_file():
        raise VideoError(f"Missing {metadata}")
    records = []
    dimensions = None
    previous = -math.inf
    for line_number, line in enumerate(metadata.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("error") or record.get("errors") or record.get("type") == "error":
                raise VideoError(f"Recorder error/invalid record on line {line_number}")
            index = len(records) + 1
            expected = f"frame{index:06d}.png"
            if type(record["index"]) is not int or record["index"] != index or record["file"] != expected:
                raise VideoError(f"Frame sequence gap or unexpected filename on line {line_number}")
            actual = png_dimensions(directory / expected)
            if actual != (record["width"], record["height"]) or min(actual) < 1:
                raise VideoError(f"PNG dimensions disagree with metadata: {expected}")
            if dimensions is not None and actual != dimensions:
                raise VideoError("Frame dimensions changed during capture")
            dimensions = actual
            elapsed = float(record["elapsed_seconds"])
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed <= previous:
                raise VideoError("Frame timestamps must be finite, nonnegative and strictly increasing")
            previous = elapsed
            records.append(record)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if isinstance(exc, VideoError):
                raise
            raise VideoError(f"Invalid frame metadata on line {line_number}: {exc}") from exc
    if not records:
        raise VideoError("Recording contains no frames")
    if {p.name for p in directory.glob("frame*.png")} != {r["file"] for r in records}:
        raise VideoError("PNG files and frame metadata do not match")
    return records


def prepare_output(output: Path, sidecar: bool = False) -> None:
    if output.suffix.lower() != ".mp4":
        raise VideoError("Output must have an .mp4 extension")
    if output.exists() or (sidecar and output.with_suffix(".manifest.json").exists()):
        raise VideoError(f"Refusing to overwrite existing output or manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def encode(frames: Path, output: Path, fps: float = 6) -> dict:
    if not math.isfinite(fps) or fps <= 0:
        raise VideoError("FPS must be positive and finite")
    records = validate_frames(frames)
    prepare_output(output)
    with tempfile.TemporaryDirectory(prefix="vcmi-encode-", dir=output.parent) as temp:
        staged = Path(temp) / "encoded.mp4"
        run([binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-xerror", "-n",
             "-framerate", str(fps), "-start_number", "1", "-i", str(frames / "frame%06d.png"),
             "-frames:v", str(len(records)), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(staged)])
        info = check_video(staged, len(records), fps)
        os.link(staged, output)  # Atomic publication, and fails if another writer created output.
    return {"output": str(output), "fps": fps, **info,
            "capture_elapsed_seconds": records[-1]["elapsed_seconds"] - records[0]["elapsed_seconds"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(roots: list[Path]) -> tuple[list[dict], list[dict]]:
    paths = set()
    for root in roots:
        if not root.exists():
            raise VideoError(f"Input run directory does not exist: {root}")
        if not root.is_dir():
            raise VideoError(f"Expected a run directory: {root}")
        paths.update(p.resolve() for p in root.rglob("manifest.json"))
    sources, skipped = [], []
    for path in sorted(paths):
        try:
            data = json.loads(path.read_text())
            if data["schema_version"] != 1 or data["kind"] not in {"recorder-smoke", "rl-evaluation"}:
                raise VideoError(f"Unsupported run manifest: {path}")
            if type(data["episode"]) is not int or data["episode"] <= 0:
                raise VideoError(f"Episode must be a positive integer: {path}")
            if data["status"] not in {"timeout", "completed", "failed"} or data["outcome"] not in {None, "win", "loss"}:
                raise VideoError(f"Invalid status/outcome: {path}")
            if not isinstance(data["policy"]["label"], str) or not data["policy"]["label"].strip():
                raise VideoError(f"Missing policy label: {path}")
            if "checkpoint" not in data["policy"]:
                raise VideoError(f"Missing policy checkpoint field: {path}")
            base = {"manifest": str(path), "manifest_sha256": sha256(path),
                    "episode": data["episode"], "kind": data["kind"], "status": data["status"],
                    "outcome": data["outcome"], "policy": data["policy"],
                    "run_wall_seconds": data.get("wall_seconds")}
            if any(marker.exists() for marker in (path.parent / "recording-error.txt",
                                                  path.parent / "frames" / "recording-error.txt")):
                skipped.append({**base, "reason": "Native recorder reported an error"})
                continue
            if not data.get("video"):
                skipped.append({**base, "reason": "No source video"})
                continue
            relative = Path(data["video"])
            video = (path.parent / relative).resolve()
            if relative.is_absolute() or not video.is_relative_to(path.parent):
                raise VideoError(f"Video must be inside its run directory: {path}")
            if not video.is_file():
                skipped.append({**base, "reason": "Source video missing", "video": str(video)})
                continue
            info = probe(video)
            if info["frames"] <= 0 or info["duration_seconds"] <= 0:
                raise VideoError(f"Empty source video: {video}")
            capture = data["capture"]
            if type(capture["frames"]) is not int or capture["frames"] != info["frames"]:
                raise VideoError(f"Capture frame count disagrees with source video: {path}")
            if not math.isfinite(float(capture["fps"])) or float(capture["fps"]) <= 0:
                raise VideoError(f"Invalid capture FPS: {path}")
            sources.append({**base, "video": str(video), "video_sha256": sha256(video),
                            "capture": capture, "capture_elapsed_seconds": capture.get("elapsed_seconds"),
                            "source_video": info})
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if isinstance(exc, VideoError):
                raise
            raise VideoError(f"Invalid manifest {path}: {exc}") from exc
    sources.sort(key=lambda item: (item["episode"], item["manifest"]))
    if not sources:
        raise VideoError("No valid episode videos found")
    return sources, skipped


def select_indices(count: int, samples: int) -> list[int]:
    if count < 1 or samples < 0:
        raise VideoError("Need at least one episode; samples cannot be negative")
    take = count if samples == 0 else min(samples, count)
    if take == 1:
        return [0]
    return [round(i * (count - 1) / (take - 1)) for i in range(take)]


def frame_budgets(total: int, count: int) -> list[int]:
    if total < count:
        raise VideoError("Output frame budget cannot show every selected episode; increase duration or reduce samples")
    quotient, remainder = divmod(total, count)
    return [quotient + (i < remainder) for i in range(count)]


# Each character is seven rows of five bits. Uppercase improves legibility at small sizes.
_FONT_ROWS = {
    "A":"14,17,17,31,17,17,17", "B":"30,17,17,30,17,17,30", "C":"14,17,16,16,16,17,14",
    "D":"30,17,17,17,17,17,30", "E":"31,16,16,30,16,16,31", "F":"31,16,16,30,16,16,16",
    "G":"14,17,16,23,17,17,15", "H":"17,17,17,31,17,17,17", "I":"14,4,4,4,4,4,14",
    "J":"7,2,2,2,18,18,12", "K":"17,18,20,24,20,18,17", "L":"16,16,16,16,16,16,31",
    "M":"17,27,21,21,17,17,17", "N":"17,25,21,19,17,17,17", "O":"14,17,17,17,17,17,14",
    "P":"30,17,17,30,16,16,16", "Q":"14,17,17,17,21,18,13", "R":"30,17,17,30,20,18,17",
    "S":"15,16,16,14,1,1,30", "T":"31,4,4,4,4,4,4", "U":"17,17,17,17,17,17,14",
    "V":"17,17,17,17,17,10,4", "W":"17,17,17,21,21,21,10", "X":"17,17,10,4,10,17,17",
    "Y":"17,17,10,4,4,4,4", "Z":"31,1,2,4,8,16,31", "0":"14,17,19,21,25,17,14",
    "1":"4,12,4,4,4,4,14", "2":"14,17,1,2,4,8,31", "3":"30,1,1,14,1,1,30",
    "4":"2,6,10,18,31,2,2", "5":"31,16,16,30,1,1,30", "6":"14,16,16,30,17,17,14",
    "7":"31,1,2,4,8,8,8", "8":"14,17,17,14,17,17,14", "9":"14,17,17,15,1,1,14",
    " ":"0,0,0,0,0,0,0", "-":"0,0,0,31,0,0,0", ".":"0,0,0,0,0,12,12",
    ":":"0,12,12,0,12,12,0", "/":"1,2,2,4,8,8,16", "#":"10,31,10,10,31,10,0",
    "(":"2,4,8,8,8,4,2", ")":"8,4,2,2,2,4,8", "|":"4,4,4,4,4,4,4",
    "_":"0,0,0,0,0,0,31", "?":"14,17,1,2,4,0,4", "+":"0,4,4,31,4,4,0",
}
FONT = {char: [int(row) for row in rows.split(",")] for char, rows in _FONT_ROWS.items()}


def overlay_lines(source: dict) -> list[str]:
    kind = "RECORDER SMOKE (NO RL TRAINING)" if source["kind"] == "recorder-smoke" else "RL POLICY EVALUATION"
    lines = [f"EPISODE #{source['episode']} | {kind}", f"POLICY: {source['policy']['label']}",
             f"STATUS: {source['status']} | OUTCOME: {source['outcome'] or 'UNKNOWN'}"]
    return ["".join(char if char in FONT else "?" for char in line.upper())[:68] for line in lines]


def write_banner(path: Path, lines: list[str]) -> None:
    width, height, scale = 1280, 106, 3
    pixels = bytearray(bytes((0, 0, 0, 215)) * width * height)
    for row, line in enumerate(lines):
        for column, char in enumerate(line):
            for y, bits in enumerate(FONT.get(char, FONT["?"])):
                for x in range(5):
                    if bits & (1 << (4 - x)):
                        for dy in range(scale):
                            for dx in range(scale):
                                px = 18 + column * 6 * scale + x * scale + dx
                                py = 12 + row * 30 + y * scale + dy
                                offset = (py * width + px) * 4
                                pixels[offset:offset + 4] = b"\xff\xff\xff\xff"
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\0" + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def montage(roots: list[Path], output: Path, seconds: float = 60, samples: int = 12) -> dict:
    if not math.isfinite(seconds) or seconds <= 0:
        raise VideoError("Duration must be positive and finite")
    prepare_output(output, sidecar=True)
    sources, skipped = discover(roots)
    discovered_count = len(sources) + len(skipped)
    selected_indices = select_indices(len(sources), samples)
    budgets = frame_budgets(round(seconds * 30), len(selected_indices))
    selected_set = set(selected_indices)
    skipped.extend({**source, "reason": "Not selected by evenly spaced chronological sampling"}
                   for index, source in enumerate(sources) if index not in selected_set)
    selected = []
    with tempfile.TemporaryDirectory(prefix="vcmi-montage-", dir=output.parent) as temp:
        work = Path(temp)
        for position, (index, frames) in enumerate(zip(selected_indices, budgets)):
            source = sources[index]
            banner = work / f"banner{position}.png"
            clip = work / f"clip{position}.mp4"
            lines = overlay_lines(source)
            write_banner(banner, lines)
            duration = frames / 30
            ratio = duration / source["source_video"]["duration_seconds"]
            filters = (f"[0:v]setpts=(PTS-STARTPTS)*{ratio:.15g},fps=30,"
                       "scale=1280:720:force_original_aspect_ratio=decrease,"
                       "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                       f"tpad=stop_mode=clone:stop_duration={duration},trim=end_frame={frames}[v];"
                       "[v][1:v]overlay=x=0:y=614:shortest=1:format=auto,format=yuv420p[out]")
            run([binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-xerror", "-n",
                 "-i", source["video"], "-loop", "1", "-framerate", "30", "-i", str(banner),
                 "-filter_complex", filters, "-map", "[out]", "-frames:v", str(frames), "-an",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-movflags", "+faststart", str(clip)])
            check_video(clip, frames, 30)
            selected.append({**source, "output_start_frame": sum(budgets[:position]), "output_frames": frames,
                             "output_seconds": duration, "source_seconds": source["source_video"]["duration_seconds"],
                             "speed_factor": 1 / ratio, "overlay_lines": lines})
        # The concat list contains generated filenames only, never user-controlled paths.
        concat = work / "concat.txt"
        concat.write_text("".join(f"file 'clip{i}.mp4'\n" for i in range(len(selected))))
        staged = work / "montage.mp4"
        run([binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-xerror", "-n",
             "-f", "concat", "-safe", "1", "-i", str(concat), "-map", "0:v:0", "-c", "copy",
             "-movflags", "+faststart", str(staged)])
        info = check_video(staged, sum(budgets), 30)
        report = {"schema_version": 1, "kind": "experiment-montage", "output": output.name,
                  "output_sha256": sha256(staged), "requested_seconds": seconds, "fps": 30,
                  "output_video": info, "selection": "Evenly spaced episode order; outcomes are not used for selection",
                  "source_time_sampling": "Each selected source is uniformly rescaled over its whole video timeline",
                  "samples_requested": samples, "discovered_episode_count": discovered_count,
                  "available_episode_count": len(sources), "selected_episode_count": len(selected),
                  "full_source_episode_coverage": not skipped,
                  "full_source_frame_coverage": False,
                  "selected": selected, "skipped": skipped}
        sidecar = output.with_suffix(".manifest.json")
        # Exclusive creation and hard-link publication protect every pre-existing file.
        with sidecar.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        try:
            os.link(staged, output)
        except OSError:
            sidecar.unlink()
            raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("encode", help="Validate native frame capture and encode an MP4")
    encoder.add_argument("--frames", type=Path, required=True)
    encoder.add_argument("--output", type=Path, required=True)
    encoder.add_argument("--fps", type=float, default=6)
    reel = commands.add_parser("montage", help="Create a chronological, labeled experiment reel")
    reel.add_argument("runs", type=Path, nargs="+")
    reel.add_argument("--output", type=Path, required=True)
    reel.add_argument("--seconds", type=float, default=60)
    reel.add_argument("--samples", type=int, default=12, help="Evenly spaced episodes; 0 includes all")
    args = parser.parse_args(argv)
    try:
        result = encode(args.frames, args.output, args.fps) if args.command == "encode" else montage(args.runs, args.output, args.seconds, args.samples)
        print(json.dumps(result, indent=2))
        return 0
    except (VideoError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
