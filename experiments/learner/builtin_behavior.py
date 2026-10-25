"""Descriptive metrics for one learner against a native built-in opponent.

Only manifest-bound learner observations enter these metrics. The opponent has
no invented observation stream, neural choices, values, or training samples.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.headless import builtin_evidence as evidence
from experiments.learner import contextual as c
from experiments.learner import evaluation_behavior as shared

SCHEMA = "heroes-builtin-evaluation-behavior-v1"


def analyze_episode(folder):
    folder = Path(folder).resolve()
    result = {"schema": SCHEMA, "episode_directory": str(folder), "status": "incomplete", "bindings": {},
        "coverage": {"complete": False, "issues": []}, "players": {}, "learner_player": None,
        "aggregate": None, "opponent_history": None, "training_input": False, "policy_input": False,
        "reward_input": False, "training_samples": 0, "limitations": list(shared.LIMITATIONS) + [
            "Exactly one controlled learner is measured; native opponent decisions and hidden state are unavailable.",
            "Compare with the corresponding learner seat of self-play, not a sum over both neural policies."]}
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        result["coverage"]["issues"].append("manifest_missing")
        return result
    data = manifest_path.read_bytes()
    manifest = evidence.native.strict_json(data)
    result["bindings"]["manifest.json"] = shared._binding(data)
    player = manifest.get("learner_player")
    evidence.require(manifest.get("kind") == evidence.MANIFEST_KIND and manifest.get("history_schema") == c.DIALOGS_SCHEMA
        and type(player) is int and player in (0, 1), "Built-in behavior needs an explicit one-learner v7 manifest")
    color = evidence.COLORS[player]
    evidence.require(manifest.get("controllers") == evidence.controllers(player)
        and type(manifest.get("players")) is dict and set(manifest["players"]) == {color},
        "Native opponent must not acquire a fabricated history")
    result.update(learner_player=player, episode={key: manifest.get(key) for key in
        ("episode", "seed", "status", "termination_reason", "failure_detail", "wall_seconds", "outcomes_by_player", "rewards_by_player")})
    valid = manifest.get("status") in ("completed", "truncated")
    if not valid:
        result["coverage"]["issues"].append("episode_not_validly_finalized")
    entry = {"coverage": {"complete": False, "hash_verified": False, "issues": []}, "metrics": None}
    result["players"][color] = entry
    declaration = manifest["players"][color].get("history_v7")
    if not declaration or not declaration.get("path") or not declaration.get("sha256"):
        entry["coverage"]["issues"].append("manifest_history_binding_missing")
        return result
    path = evidence.native.inside(folder, declaration["path"])
    entry["history"] = declaration["path"]
    if not path.is_file():
        entry["coverage"]["issues"].append("bound_history_file_missing")
        return result
    data = path.read_bytes()
    identity = shared._binding(data)
    evidence.require(identity["sha256"] == declaration["sha256"]
        and ("bytes" not in declaration or identity["bytes"] == declaration["bytes"]), "Learner history bytes changed")
    result["bindings"][declaration["path"]] = identity
    entry["coverage"]["hash_verified"] = True
    history, frames = c.PlayerHistory(str(manifest["episode"]), player, schema=c.DIALOGS_SCHEMA), []
    lines = data.splitlines()
    for index, line in enumerate(lines):
        try:
            raw = evidence.native.strict_json(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not valid and index == len(lines)-1 and not data.endswith(b"\n"):
                entry["coverage"]["issues"].append("trailing_partial_json_line")
                break
            raise ValueError("Malformed learner history record") from None
        frame = c.decode_frame(raw)
        evidence.require(frame.schema == c.DIALOGS_SCHEMA and frame.player == player and frame.episode == history.episode
            and frame.seq == index, "Wrong history player/episode/schema/sequence")
        history.append(frame)
        frames.append(raw)
    if not frames:
        entry["coverage"]["issues"].append("no_available_frames")
        return result
    entry["metrics"], issues = shared._player_metrics(frames)
    entry["coverage"]["issues"].extend(issues)
    if valid:
        evidence.validate_manifest_history(history, manifest)
        final = folder / "builtin-finalization.json"
        if not final.is_file():
            entry["coverage"]["issues"].append("finalization_file_missing")
        else:
            evidence.read_finalization(folder, manifest)
            result["bindings"]["builtin-finalization.json"] = evidence.native.file_identity(final)
        entry["coverage"]["complete"] = not entry["coverage"]["issues"]
    if not entry["coverage"]["complete"]:
        entry["coverage"]["issues"].append("observed_prefix_only")
    result["aggregate"] = shared._aggregate([entry])
    result["coverage"]["complete"] = entry["coverage"]["complete"]
    result["status"] = "complete" if entry["coverage"]["complete"] else "incomplete"
    wall = manifest.get("wall_seconds")
    wall = wall if type(wall) in (int, float) and math.isfinite(wall) and wall > 0 else None
    result["throughput"] = {"manifest_wall_seconds": wall,
        "observed_learner_choices_per_second": entry["metrics"]["choices"] / wall if wall else None,
        "scope": "One learner's captured episode choices per wall second; no native-opponent choice equivalent."}
    return result


def aggregate_episodes(reports):
    reports = list(reports)
    evidence.require(all(row.get("schema") == SCHEMA for row in reports), "Cannot mix one-seat and paired metric denominators")
    evidence.require(len({row["episode_directory"] for row in reports}) == len(reports), "Duplicate episode would double count")
    players = [entry for row in reports for entry in row["players"].values()]
    timed = [row for row in reports if row.get("throughput", {}).get("manifest_wall_seconds") and row["aggregate"] is not None]
    wall = sum(row["throughput"]["manifest_wall_seconds"] for row in timed)
    return {"schema": SCHEMA, "episodes": len(reports), "complete_episodes": sum(row["status"] == "complete" for row in reports),
        "incomplete_episodes": sum(row["status"] != "complete" for row in reports),
        "unavailable_learner_identity_episodes": sum(row["learner_player"] is None for row in reports),
        "aggregate": shared._aggregate(players), "timed_episodes_with_metrics": len(timed),
        "timed_episode_wall_seconds": wall if timed else None,
        "observed_learner_choices_per_second": sum(row["aggregate"]["choices"] for row in timed) / wall if wall else None,
        "opponent_history": None, "training_input": False, "policy_input": False, "reward_input": False, "training_samples": 0}
