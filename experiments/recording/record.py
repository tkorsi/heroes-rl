#!/usr/bin/env python3
"""Record a bounded real VCMI built-in-AI smoke game. This does not train an agent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git_state(path: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stop_owned_process(process: subprocess.Popen) -> None:
    """Only stop the private process group that this recorder started."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def classify_run(returncode: int, reason: str | None, log: str) -> tuple[str, str | None, str]:
    outcomes = set(re.findall(r"Red player (won|lost)\. Ending game\.", log))
    outcome = ({"won": "win", "lost": "loss"}[next(iter(outcomes))]
               if len(outcomes) == 1 else None)
    if reason:
        # Even a delayed terminal log cannot turn an interrupted run into a scored game.
        return ("timeout" if reason in {"wall_time_limit", "frame_limit"} else "failed",
                None, reason)
    if returncode == 0 and outcome:
        return "completed", outcome, "natural_terminal"
    return "failed", None, "process_exit_without_verified_terminal"


def record(args: argparse.Namespace) -> dict:
    if sys.platform != "darwin":
        raise ValueError("This runner currently uses the macOS VCMI profile override")
    binary = args.binary.expanduser().resolve()
    data = args.data.expanduser().resolve()
    game_map = args.map.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Build the recorder-enabled client first: {binary}")
    # Refuse a stock/old client before it can touch a normal game profile.
    if b"VCMI_RECORD_DIR" not in binary.read_bytes():
        raise ValueError("Client has no native recorder; rebuild vcmi/build")
    if not (data / "Data").is_dir():
        raise ValueError(f"Missing original game assets: {data / 'Data'}")
    if not game_map.is_file() or game_map.suffix.lower() not in {".vmap", ".h3m"}:
        raise ValueError(f"Expected a real .vmap or .h3m file: {game_map}")
    if not 1 <= args.fps <= 60 or not 1 <= args.max_frames <= 1_000_000:
        raise ValueError("FPS must be 1..60; max-frames must be 1..1000000")
    if not 1 <= args.seconds <= 86400 or not 1 <= args.seed <= 2147483647 or args.episode < 1:
        raise ValueError("Use seconds 1..86400, nonzero signed-32-bit seed, positive episode")
    if not args.no_encode and not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise ValueError("ffmpeg and ffprobe must be on PATH")

    output.mkdir(parents=True, exist_ok=False)
    frames, profile, logs = (output / name for name in ("frames", "profile", "logs"))
    for folder in (frames, profile / "config", profile / "Maps", logs):
        folder.mkdir(parents=True, exist_ok=True)
    (profile / "Data").symlink_to(data / "Data", target_is_directory=True)
    if (data / "Mp3").is_dir():
        (profile / "Mp3").symlink_to(data / "Mp3", target_is_directory=True)
    local_map = profile / "Maps" / ("recording" + game_map.suffix.lower())
    shutil.copyfile(game_map, local_map)
    settings = {
        "general": {"lastDifficulty": 0, "music": 0, "sound": 0, "autosaveCountLimit": 1},
        "video": {"fullscreen": False, "resolution": {"width": 960, "height": 720, "scaling": 100},
                  "targetfps": 30},
        "server": {"seed": args.seed, "useProcess": False, "localPort": 0},
        "ai": {"adventureEnemyAI": "Nullkiller", "adventureAlliedAI": "Nullkiller",
               "combatEnemyAI": "BattleAI", "combatAlliedAI": "BattleAI", "combatNeutralAI": "BattleAI"},
    }
    write_json(profile / "config" / "settings.json", settings)
    command = [str(binary), "--testmap", "Maps/" + local_map.name, "--spectate",
               "--spectate-skip-battle", "--spectate-skip-battle-result", "--spectate-hero-speed", "4",
               "--nointro", "--disable-video", "--logLocation", str(logs)]
    capture_env = {"VCMI_USER_DATA_DIR": str(profile), "VCMI_RECORD_DIR": str(frames),
                   "VCMI_RECORD_FPS": str(args.fps), "VCMI_RECORD_MAX_FRAMES": str(args.max_frames)}
    manifest = {
        "schema_version": 1, "kind": "recorder-smoke", "episode": args.episode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {"label": "Built-in Nullkiller (no learning)", "checkpoint": None},
        "opponent": {"module": "Nullkiller", "difficulty": "Easy", "difficulty_index": 0},
        "combat": "BattleAI; automatic; spectator battle display skipped",
        "perspective": "omniscient spectator, follows active heroes; never a learner input",
        "seed": args.seed, "deterministic_replay_verified": False,
        "map": {"name": game_map.name, "sha256": digest(game_map)},
        "engine": {**git_state(ROOT / "vcmi"), "binary": str(binary), "binary_sha256": digest(binary)},
        "project": git_state(ROOT), "command": command, "environment": capture_env,
        "requested_settings": settings, "settings_sha256_before": digest(profile / "config" / "settings.json"),
        "limits": {"wall_seconds": args.seconds, "max_frames": args.max_frames},
        "capture": {"fps": args.fps, "frames": 0, "source": "native SDL game surface"},
        "status": "running", "outcome": None, "video": None,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    started = time.monotonic()
    process = None
    reason = None
    try:
        with (logs / "process.log").open("w") as log:
            process = subprocess.Popen(command, cwd=binary.parent,
                                       env={**os.environ, **capture_env}, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if (frames / "recording-error.txt").exists():
                    reason = "recorder_error"
                    break
                if (frames / "recording-limit-reached.txt").exists():
                    reason = "frame_limit"
                    break
                if time.monotonic() - started >= args.seconds:
                    reason = "wall_time_limit"
                    break
                time.sleep(0.1)
            stop_owned_process(process)
        returncode = process.returncode
    except BaseException as error:
        if process:
            stop_owned_process(process)
        manifest.update(status="failed", outcome=None, termination_reason=type(error).__name__, error=str(error))
        write_json(manifest_path, manifest)
        raise
    manifest["wall_seconds"] = round(time.monotonic() - started, 3)
    manifest["returncode"] = returncode
    log_text = (logs / "process.log").read_text(errors="replace")
    # Catch an error marker written immediately before process exit as well.
    if (frames / "recording-error.txt").exists():
        reason = "recorder_error"
    status, outcome, termination = classify_run(returncode, reason, log_text)
    manifest.update(status=status, outcome=outcome, termination_reason=termination)
    manifest["capture"]["frames"] = len(list(frames.glob("frame*.png")))
    if manifest["capture"]["frames"] == 0:
        manifest.update(status="failed", outcome=None, termination_reason="no_gameplay_frames")
    manifest["recording_valid"] = False
    write_json(manifest_path, manifest)
    if (frames / "frames.jsonl").exists():
        manifest["capture"]["ledger_sha256"] = digest(frames / "frames.jsonl")
        try:
            entries = [json.loads(line) for line in (frames / "frames.jsonl").read_text().splitlines()]
            if entries:
                manifest["capture"]["elapsed_seconds"] = entries[-1]["elapsed_seconds"] - entries[0]["elapsed_seconds"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            manifest.update(status="failed", outcome=None, recording_error=f"Invalid frame ledger: {error}")
            write_json(manifest_path, manifest)
            raise ValueError(manifest["recording_error"]) from error
    write_json(manifest_path, manifest)
    if not args.no_encode:
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name("video.py")), "encode",
                            "--frames", str(frames), "--output", str(output / "episode.mp4"),
                            "--fps", str(args.fps)], check=True)
            manifest["video"] = "episode.mp4"
            manifest["recording_valid"] = True
        except subprocess.CalledProcessError:
            manifest["recording_error"] = "Frame validation or MP4 encoding failed; see command output"
            write_json(manifest_path, manifest)
            raise
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--map", type=Path, default=Path.home() / "Library/Application Support/vcmi/Maps/Key to Victory.h3m")
    parser.add_argument("--output", type=Path, required=True, help="New run directory (must not exist)")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()
    try:
        result = record(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Recording failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "outcome", "termination_reason", "capture", "video")}, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
