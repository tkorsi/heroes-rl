#!/usr/bin/env python3
"""Record a bounded real VCMI built-in-AI smoke game. This does not train an agent."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VIDEO_SPEC = importlib.util.spec_from_file_location("heroes_recording_video", Path(__file__).with_name("video.py"))
video = importlib.util.module_from_spec(VIDEO_SPEC)
VIDEO_SPEC.loader.exec_module(video)
# Episode identity and scoring are shared with headless runs, which have no recordings.
SCORING_SPEC = importlib.util.spec_from_file_location("heroes_episode_scoring", ROOT / "experiments/episodes/scoring.py")
scoring = importlib.util.module_from_spec(SCORING_SPEC)
SCORING_SPEC.loader.exec_module(scoring)
PLAYER_COLORS = scoring.PLAYER_COLORS
digest, git_state, write_json = scoring.digest, scoring.git_state, scoring.write_json
classify_run, classify_agent_run = scoring.classify_run, scoring.classify_agent_run
classify_self_play_run = scoring.classify_self_play_run
validate_experiment_metadata = scoring.validate_experiment_metadata


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
    self_play = manifest.get("kind") == "random-self-play"
    evidence = validate_experiment_metadata(metadata, player_index, log, manifest.get("agent_player_index"),
                                            manifest.get("agent_player_indices"))
    if "agent_player_index" in manifest or self_play:
        manifest["effective_engine_rng"] = scoring.engine_seed_evidence(log, manifest["seed"])
    evidence.update(metadata_file="frames/experiment-metadata.json", metadata_sha256=digest(metadata_path))
    manifest["effective_experiment_state"] = metadata
    manifest["visibility_validation"] = evidence
    streams = {}
    for key, recording in manifest["recordings"].items():
        mode = recording.get("perspective", {}).get("mode", key)
        observer = recording.get("perspective", {}).get("observer_player_index", player_index)
        if self_play:
            expected_key = {f"{view}-{color}": (view, index, color)
                            for index, color in enumerate(("red", "blue")) for view in ("omniscient", "player")}
            if key not in expected_key:
                raise ValueError("Unexpected self-play recording stream")
            expected_mode, expected_observer, color = expected_key[key]
            perspective = recording.get("perspective", {})
            if (mode != expected_mode or type(observer) is not int or observer != expected_observer
                    or perspective.get("observer_player") != color
                    or perspective.get("player") != (color if mode == "player" else None)
                    or perspective.get("player_index") != (observer if mode == "player" else None)):
                raise ValueError("Self-play stream identity differs from its required view/player")
        directory = output / "frames" / key
        entries = validate_capture_view(directory, mode, observer, evidence["total_tiles"])
        streams[key] = entries
        recording["capture"].update(frames=len(entries), ledger_sha256=digest(directory / "frames.jsonl"),
                                    elapsed_seconds=entries[-1]["elapsed_seconds"] - entries[0]["elapsed_seconds"],
                                    initial_visible_tiles=entries[0]["visible_tiles"],
                                    final_visible_tiles=entries[-1]["visible_tiles"],
                                    initial_game_day=entries[0]["game_day"], final_game_day=entries[-1]["game_day"])
    if set(streams) == {"omniscient", "player"}:
        manifest["paired_capture_validation"] = video.validate_paired_frames(streams["omniscient"], streams["player"])
    if self_play:
        expected = {f"{view}-{color}" for color in ("red", "blue") for view in ("omniscient", "player")}
        if set(streams) != expected:
            raise ValueError("Self-play requires both perspectives for both players")
        manifest["paired_capture_validation_by_player"] = {
            color: video.validate_paired_frames(streams[f"omniscient-{color}"], streams[f"player-{color}"])
            for color in ("red", "blue")}
        manifest["synchronized_capture_validation"] = video.validate_synchronized_frames(streams)
    if not no_encode:
        for key, recording in manifest["recordings"].items():
            name = key + ".mp4" if self_play else ("omniscient.mp4" if key == "omniscient" else f"player-{PLAYER_COLORS[player_index]}.mp4")
            recording["encoded_video"] = video.encode(output / "frames" / key, output / name,
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
    if "reward" in manifest:
        manifest["reward"] = None
    if "rewards_by_player" in manifest:
        manifest["rewards_by_player"] = {color: None for color in manifest["rewards_by_player"]}
        manifest["outcomes_by_player"] = {color: None for color in manifest["outcomes_by_player"]}
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


def freeze_policy_evidence(controller) -> tuple[dict, str]:
    """Drain delivered terminals, then stop the writer before freezing evidence."""
    if controller.thread is not None:
        controller.thread.join(timeout=1)
    controller.close()
    return controller.snapshot(), digest(controller.path)


def record(args: argparse.Namespace, policy=None, *, policies=None) -> dict:
    all_policies = list(policies.values()) if policies is not None else ([policy] if policy is not None else [])
    try:
        return _record(args, policy, policies=policies)
    finally:
        errors = []
        for controller in all_policies:
            try:
                controller.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Could not stop every policy connection") from errors[0]


def _record(args: argparse.Namespace, policy=None, *, policies=None) -> dict:
    self_play = policies is not None
    if self_play and (policy is not None or set(policies) != {0, 1} or len({id(p) for p in policies.values()}) != 2):
        raise ValueError("Self-play requires distinct Red and Blue policy instances and no single policy")
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
    if self_play and (args.view != "both" or player_index != 0):
        raise ValueError("Self-play records both views for both players, with Red as the default observer")
    controllers = policies if self_play else ({player_index: policy} if policy is not None else {})
    if controllers:
        if b"EXPLORER_RL_PLAYER" not in binary.read_bytes() or not (binary.parent / "AI/libExplorerRL.dylib").is_file():
            raise ValueError("Build the player-routing client and ExplorerRL module first")
    if self_play and (b"EXPLORER_RL_PLAYERS" not in binary.read_bytes() or b"VCMI_RECORD_PLAYERS" not in binary.read_bytes()
                      or b"EXPLORER_RL_PORT_" not in (binary.parent / "AI/libExplorerRL.dylib").read_bytes()):
        raise ValueError("Rebuild the self-play client, multi-player recorder and ExplorerRL module")
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
    if policy is not None:
        capture_env.update(policy.start(output, str(args.episode), player_index))
    elif self_play:
        capture_env.pop("VCMI_RECORD_PLAYER")
        capture_env.update(VCMI_RECORD_PLAYERS="0,1", EXPLORER_RL_PLAYERS="0,1",
                           EXPLORER_RL_EPISODE=str(args.episode), EXPLORER_RL_TIMEOUT="60")
        for index, controller in sorted(controllers.items()):
            folder = output / "agents" / PLAYER_COLORS[index]
            folder.mkdir(parents=True)
            endpoint = controller.start(folder, str(args.episode), index)
            capture_env[f"EXPLORER_RL_PORT_{index}"] = endpoint["EXPLORER_RL_PORT"]
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
        "engine_recipe": scoring.engine_recipe_digests(ROOT),
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
    if policy is not None:
        manifest.update(kind="random-baseline", policy=policy.description,
                        agent_player_index=player_index, outcome_player=args.player,
                        outcome_player_index=player_index, reward=None)
        manifest["limits"].update(max_actions=policy.max_actions, max_days=policy.max_days)
    if self_play:
        manifest.update(schema_version=3, kind="random-self-play", agent_player_indices=[0, 1],
                        policy={"label": "Two uniform random agents (no learning)", "checkpoint": None, "learning": False},
                        policies_by_player={PLAYER_COLORS[i]: p.description for i, p in sorted(controllers.items())},
                        opponent={"module": "ExplorerRL", "kind": "independent_uniform_random", "learning": False},
                        outcome_player=None, outcome_player_index=None,
                        outcomes_by_player={"red": None, "blue": None}, rewards_by_player={"red": None, "blue": None})
        manifest["limits"]["by_player"] = {PLAYER_COLORS[i]: {"max_actions": p.max_actions, "max_days": p.max_days}
                                              for i, p in sorted(controllers.items())}
        manifest["recordings"] = {f"{view}-{color}": {
            "perspective": {"mode": view, "player": color if view == "player" else None,
                            "player_index": index if view == "player" else None,
                            "observer_player": color, "observer_player_index": index},
            "capture": {"fps": args.fps, "frames": 0, "source": "native SDL game surface",
                        "directory": f"frames/{view}-{color}"}, "video": None, "recording_valid": False,
        } for index, color in enumerate(("red", "blue")) for view in views}
    if controllers:
        manifest["policy_source"] = {"file": "experiments/explorer/random_policy.py",
                                     "sha256": digest(ROOT / "experiments/explorer/random_policy.py")}
        manifest["engine"]["ai_libraries"] = {name: digest(binary.parent / "AI" / name)
             for name in ("libExplorerRL.dylib", "libNullkiller.dylib", "libBattleAI.dylib")}
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    started = time.monotonic()
    process = None
    reason = None
    try:
        with (logs / "process.log").open("w") as log:
            process = subprocess.Popen(command, cwd=binary.parent,
                                       env={**{key: value for key, value in os.environ.items()
                                               if not key.startswith(("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_"))},
                                            **capture_env}, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                stopping = [(index, controller.stop_reason) for index, controller in controllers.items() if controller.stop_reason]
                if stopping:
                    index, reason = stopping[0]
                    manifest["termination_player"] = PLAYER_COLORS[index]
                    break
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
            if reason:
                for controller in controllers.values():
                    controller.prepare_stop()
            stop_owned_process(process)
        returncode = process.returncode
    except BaseException as error:
        for controller in controllers.values():
            controller.prepare_stop()
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
        if self_play:
            executions, ledgers = {}, {}
            for index, controller in sorted(controllers.items()):
                color = PLAYER_COLORS[index]
                executions[color], ledger_digest = freeze_policy_evidence(controller)
                relative = f"agents/{color}/policy.jsonl"
                ledgers[color] = {"file": relative, "sha256": ledger_digest}
            manifest.update(policy_executions=executions, policy_ledgers=ledgers)
            status, outcomes, termination, rewards = classify_self_play_run(returncode, reason, log_text, executions)
            manifest.update(outcomes_by_player=outcomes, rewards_by_player=rewards)
            outcome = None
        elif policy is not None:
            execution, ledger_digest = freeze_policy_evidence(policy)
            manifest["policy_execution"] = execution
            manifest["policy_ledger"] = {"file": "policy.jsonl", "sha256": ledger_digest}
            status, outcome, termination, manifest["reward"] = classify_agent_run(
                returncode, reason, log_text, player_index, execution)
        else:
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
