#!/usr/bin/env python3
"""Record a bounded real VCMI built-in-AI smoke game. This does not train an agent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
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
PLAYER_COLORS = ("red", "blue", "tan", "green", "orange", "purple", "teal", "pink")
VIDEO_SPEC = importlib.util.spec_from_file_location("heroes_recording_video", Path(__file__).with_name("video.py"))
video = importlib.util.module_from_spec(VIDEO_SPEC)
VIDEO_SPEC.loader.exec_module(video)


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


def validate_experiment_metadata(metadata: dict, player_index: int, log: str) -> dict:
    """Verify effective engine state and actual AI flags, independently of requested settings."""
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Missing or unsupported effective experiment metadata")
    if metadata.get("cheats_allowed") is not False or metadata.get("requested_no_cheats") is not True:
        raise ValueError("Actual engine cheats must be disabled with an explicit no-cheats request")
    if type(metadata.get("difficulty_index")) is not int or metadata["difficulty_index"] != 0:
        raise ValueError("Actual engine difficulty must be Easy (index 0)")
    if metadata.get("phase") != "before_first_turn" or type(metadata.get("game_day")) is not int or metadata["game_day"] < 0:
        raise ValueError("Missing initial game-state phase/day evidence")
    sizes = [metadata.get(key) for key in ("map_width", "map_height", "map_levels")]
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Invalid effective map dimensions")
    total_tiles = sizes[0] * sizes[1] * sizes[2]
    players = metadata.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("Missing initial player visibility evidence")
    indexed = {}
    for player in players:
        if not isinstance(player, dict):
            raise ValueError("Invalid initial player evidence")
        index = player.get("player_index")
        visible = player.get("visible_tiles")
        if type(index) is not int or not 0 <= index < len(PLAYER_COLORS) or index in indexed:
            raise ValueError("Invalid or duplicate initial player index")
        if type(player.get("human")) is not bool or type(player.get("team_index")) is not int or player["team_index"] < 0:
            raise ValueError("Missing initial player identity")
        if type(visible) is not int or not 0 <= visible <= total_tiles or player.get("total_tiles") != total_tiles:
            raise ValueError("Invalid initial player visibility count")
        indexed[index] = player
    if player_index not in indexed or indexed[player_index]["human"]:
        raise ValueError("Requested observer player is absent or is not controlled by AI")
    ai_players = {index for index, player in indexed.items() if not player["human"]}
    ai_evidence = {}
    pattern = r"Experiment AI visibility: player=(\d+) cheats_allowed=(\d+) open_map=(\d+) use_object_graph=(\d+)"
    for values in re.findall(pattern, log):
        index, cheats, open_map, graph = map(int, values)
        if index not in ai_players or cheats or open_map or graph:
            raise ValueError("Actual AI visibility flags permit cheats/open-map access or name an unexpected player")
        ai_evidence[index] = {"player_index": index, "cheats_allowed": False,
                              "open_map": False, "use_object_graph": False}
    if set(ai_evidence) != ai_players:
        raise ValueError("Missing actual Nullkiller visibility flags for one or more AI players")
    return {"validated": True, "cheats_allowed": False, "requested_no_cheats": True,
            "difficulty_index": 0, "observer_player_index": player_index,
            "initial_visible_tiles": indexed[player_index]["visible_tiles"], "total_tiles": total_tiles,
            "ai_players": [ai_evidence[index] for index in sorted(ai_evidence)],
            "scope": "Effective engine settings, AI flags and fog counts; pixel rendering requires separate visual verification"}


def validate_capture_view(directory: Path, view: str, player_index: int, total_tiles: int) -> list[dict]:
    records = video.validate_frames(directory)
    required = {"sample_id", "game_day", "view", "player_index", "observer_player_index", "visible_tiles"}
    for index, entry in enumerate(records, 1):
        if not required.issubset(entry):
            raise ValueError("Recording is missing native perspective/visibility metadata")
        if entry["sample_id"] != index or entry["view"] != view or entry["observer_player_index"] != player_index:
            raise ValueError("Native capture identity differs from the requested view/player or sample sequence")
        if entry["player_index"] != (player_index if view == "player" else None):
            raise ValueError("Native frame claims the wrong player perspective")
        if type(entry["visible_tiles"]) is not int or not 0 <= entry["visible_tiles"] <= total_tiles:
            raise ValueError("Native frame visibility count exceeds actual map bounds")
    return records


def _finalize_recordings(manifest: dict, output: Path, player_index: int, log: str, no_encode: bool) -> None:
    """Publish valid views only after effective-state, frame and pair checks all pass."""
    if "Experiment recording failed:" in log:
        raise ValueError("Native recorder reported a failure in its log, even if its error marker could not be written")
    metadata_path = output / "frames" / "experiment-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    evidence = validate_experiment_metadata(metadata, player_index, log)
    evidence.update(metadata_file="frames/experiment-metadata.json", metadata_sha256=digest(metadata_path))
    manifest["effective_experiment_state"] = metadata
    manifest["visibility_validation"] = evidence
    streams = {}
    for view, recording in manifest["recordings"].items():
        directory = output / "frames" / view
        entries = validate_capture_view(directory, view, player_index, evidence["total_tiles"])
        streams[view] = entries
        recording["capture"].update(frames=len(entries), ledger_sha256=digest(directory / "frames.jsonl"),
                                    elapsed_seconds=entries[-1]["elapsed_seconds"] - entries[0]["elapsed_seconds"],
                                    initial_visible_tiles=entries[0]["visible_tiles"],
                                    final_visible_tiles=entries[-1]["visible_tiles"],
                                    initial_game_day=entries[0]["game_day"], final_game_day=entries[-1]["game_day"])
    if set(streams) == {"omniscient", "player"}:
        manifest["paired_capture_validation"] = video.validate_paired_frames(streams["omniscient"], streams["player"])
    if not no_encode:
        for view, recording in manifest["recordings"].items():
            name = "omniscient.mp4" if view == "omniscient" else f"player-{PLAYER_COLORS[player_index]}.mp4"
            recording["encoded_video"] = video.encode(output / "frames" / view, output / name,
                                                      recording["capture"]["fps"])
            recording["video"] = name
    for recording in manifest["recordings"].values():
        recording["recording_valid"] = True
    manifest["recording_valid"] = True


def fail_recording(manifest: dict, output: Path, error: BaseException, reason: str) -> None:
    manifest["process_status"] = manifest.get("status")
    manifest["process_termination_reason"] = manifest.get("termination_reason")
    manifest.update(status="failed", outcome=None, recording_valid=False,
                    termination_reason=reason, recording_error=str(error))
    for recording in manifest["recordings"].values():
        recording["recording_valid"] = False
        if recording.get("video"):
            recording["partial_video"] = recording["video"]
            recording["video"] = None
    write_json(output / "manifest.json", manifest)


def finalize_recordings(manifest: dict, output: Path, player_index: int, log: str, no_encode: bool) -> None:
    try:
        _finalize_recordings(manifest, output, player_index, log, no_encode)
    except BaseException as error:
        fail_recording(manifest, output, error, "recording_verification_or_encoding_failed")
        raise
    write_json(output / "manifest.json", manifest)


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
    if b"VCMI_RECORD_VIEWS" not in binary.read_bytes():
        raise ValueError("Client has no native perspective recorder; rebuild vcmi/build")
    if not (data / "Data").is_dir():
        raise ValueError(f"Missing original game assets: {data / 'Data'}")
    if not game_map.is_file() or game_map.suffix.lower() not in {".vmap", ".h3m"}:
        raise ValueError(f"Expected a real .vmap or .h3m file: {game_map}")
    if not 1 <= args.fps <= 60 or not 1 <= args.max_frames <= 1_000_000:
        raise ValueError("FPS must be 1..60; max-frames must be 1..1000000")
    if not 1 <= args.seconds <= 86400 or not 1 <= args.seed <= 2147483647 or args.episode < 1:
        raise ValueError("Use seconds 1..86400, nonzero signed-32-bit seed, positive episode")
    if args.view not in {"both", "omniscient", "player"} or args.player not in PLAYER_COLORS:
        raise ValueError("Select a supported view and player color")
    player_index = PLAYER_COLORS.index(args.player)
    views = ("omniscient", "player") if args.view == "both" else (args.view,)
    if not args.no_encode and not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise ValueError("ffmpeg and ffprobe must be on PATH")
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], check=True)

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
                   "VCMI_RECORD_FPS": str(args.fps), "VCMI_RECORD_MAX_FRAMES": str(args.max_frames),
                   "VCMI_RECORD_VIEWS": args.view, "VCMI_RECORD_PLAYER": str(player_index),
                   "VCMI_EXPERIMENT_NO_CHEATS": "1"}
    manifest = {
        "schema_version": 2, "kind": "recorder-smoke", "episode": args.episode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {"label": "Built-in Nullkiller (no learning)", "checkpoint": None},
        "opponent": {"module": "Nullkiller", "difficulty": "Easy", "difficulty_index": 0},
        "combat": "BattleAI; automatic; spectator battle display skipped",
        "observer": {"player": args.player, "player_index": player_index},
        "seed": args.seed, "deterministic_replay_verified": False,
        "map": {"name": game_map.name, "sha256": digest(game_map)},
        "engine": {**git_state(ROOT / "vcmi"), "binary": str(binary), "binary_sha256": digest(binary)},
        "engine_recipe": {str(path.relative_to(ROOT)): digest(path) for path in sorted([
            ROOT / "dependencies/vcmi.json", *ROOT.glob("engine/patches/*.patch"),
            *ROOT.glob("engine/ExplorerRL/*")]) if path.is_file()},
        "project": git_state(ROOT), "command": command, "environment": capture_env,
        "requested_settings": settings, "settings_sha256_before": digest(profile / "config" / "settings.json"),
        "limits": {"wall_seconds": args.seconds, "max_frames_per_view": args.max_frames},
        "recordings": {view: {
            "perspective": {"mode": view, "player": args.player if view == "player" else None,
                            "player_index": player_index if view == "player" else None},
            "capture": {"fps": args.fps, "frames": 0, "source": "native SDL game surface",
                        "directory": f"frames/{view}"},
            "video": None, "recording_valid": False,
        } for view in views},
        "status": "running", "outcome": None, "outcome_player": "red", "outcome_player_index": 0,
        "recording_valid": False,
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
            try:
                stop_owned_process(process)
            except BaseException as cleanup_error:
                manifest["cleanup_error"] = str(cleanup_error)
        fail_recording(manifest, output, error, type(error).__name__)
        raise
    manifest["wall_seconds"] = round(time.monotonic() - started, 3)
    manifest["returncode"] = returncode
    try:
        log_text = (logs / "process.log").read_text(errors="replace")
        # Catch a final marker written immediately before process exit as well.
        if (frames / "recording-error.txt").exists() or "Experiment recording failed:" in log_text:
            reason = "recorder_error"
        elif (frames / "recording-limit-reached.txt").exists():
            reason = "frame_limit"
        status, outcome, termination = classify_run(returncode, reason, log_text)
        manifest.update(status=status, outcome=outcome, termination_reason=termination)
        write_json(manifest_path, manifest)
    except BaseException as error:
        fail_recording(manifest, output, error, "process_evidence_unreadable")
        raise
    finalize_recordings(manifest, output, player_index, log_text, args.no_encode)
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
    parser.add_argument("--view", choices=("both", "omniscient", "player"), default="both")
    parser.add_argument("--player", choices=PLAYER_COLORS, default="red", help="AI player whose fog is recorded")
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()
    try:
        result = record(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Recording failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "outcome", "outcome_player", "termination_reason", "recordings")}, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
