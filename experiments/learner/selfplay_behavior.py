"""Post-run travel summaries from both players' manifest-bound visible histories.

This is an offline report, never a policy input, reward or admission criterion.
Revisiting a tile and ending a turn do not by themselves establish aimlessness.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from experiments.learner import contextual as c
from experiments.learner.evaluation_behavior import _player_metrics
from validation.probes.audit_contextual_histories import audit, _json

SCHEMA = "heroes-selfplay-travel-v1"
COLORS = ("red", "blue")


def _binding(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def travel_metrics(frames):
    """Summarize observed travel without assigning strategic intent or rewards."""
    metrics, issues = _player_metrics(frames)
    if issues:
        raise ValueError("Incomplete visible travel evidence: " + repr(issues))
    days = {f["day"] for f in frames if f["kind"] == "submitted"}
    progress_days = set()
    seen_tiles, previous_heroes, seen_positions = set(), {}, set()
    revisits = 0
    first_observation = True
    for frame in frames:
        obs = frame.get("observation")
        if obs is None:
            continue
        tiles = {tuple(tile["position"]) for tile in obs["tiles"]}
        if not first_observation and tiles - seen_tiles:
            progress_days.add(frame["day"])
        seen_tiles.update(tiles)
        heroes = {hero["handle"]: tuple(hero["position"]) for hero in obs["heroes"]}
        for hero, position in heroes.items():
            if hero in previous_heroes and previous_heroes[hero] != position:
                progress_days.add(frame["day"])
                revisits += (hero, position) in seen_positions
            seen_positions.add((hero, position))
        previous_heroes = heroes
        first_observation = False
    day_at = {frame["seq"]: frame["day"] for frame in frames}
    progress_days.update(day_at[row["seq"]] for row in metrics["town_capture_evidence"])
    return {
        "choices": metrics["choices"], "last_public_day": metrics["last_public_day"],
        "action_counts": metrics["action_counts"], "movement": metrics["movement"],
        "own_position_changes": len(metrics["own_position_changes"]),
        "unique_hero_position_pairs_including_initial": sum(len(v) for v in metrics["own_hero_positions"].values()),
        "position_changes_revisiting_observed_hero_position": revisits,
        "newly_observed_tiles_after_initial": metrics["known_tiles"]["new_after_first_observation"] if metrics["known_tiles"] else None,
        "town_captures": len(metrics["town_capture_evidence"]),
        "town_capture_evidence": metrics["town_capture_evidence"],
        "owned_towns_first_last": metrics["owned_towns_first_last"],
        "days_with_policy_actions": sorted(days),
        "days_without_observed_travel_reveal_or_capture": sorted(days - progress_days),
        "recruitment": metrics["recruitment"], "hiring": metrics["hiring"],
        "upgrading": metrics["upgrading"], "battle_completions": metrics["battles"]["completions"],
        "cutoffs": metrics["cutoffs"],
    }


def analyze_episode(folder, learner_player):
    folder = Path(folder).resolve()
    if type(learner_player) is not int or learner_player not in (0, 1):
        raise ValueError("An explicit learner seat is required")
    manifest_path = folder / "manifest.json"
    manifest = _json(manifest_path.read_bytes())
    if (manifest.get("kind") != "contextual-headless-self-play"
            or manifest.get("environment") != c.EXPANDED_ENVIRONMENT
            or manifest.get("status") not in ("completed", "truncated")):
        raise ValueError("Travel summary requires a finalized v8 self-play episode")
    profile = c.profile_for(manifest["environment"])
    paths, rows = [], {}
    for player, color in enumerate(COLORS):
        declared = manifest["players"][color][profile.history_key]
        path = (folder / declared["path"]).resolve()
        if not path.is_relative_to(folder) or _binding(path)["sha256"] != declared["sha256"]:
            raise ValueError("Manifest-bound visible history changed or escaped its episode")
        paths.append(path)
        raw = [_json(line) for line in path.read_bytes().splitlines()]
        if not raw or any(c.decode_frame(frame).player != player for frame in raw):
            raise ValueError("Missing or mixed player-local history")
        rows[color] = {"role": "learner" if player == learner_player else "frozen_opponent",
                       "history": _binding(path), "metrics": travel_metrics(raw)}
    checked = audit(*paths, manifest_path)
    if not checked["consistent"] or not checked["manifest_bound"]:
        raise ValueError("Visible-history audit failed")
    return {"schema": SCHEMA, "episode_directory": str(folder), "manifest": _binding(manifest_path),
            "learner_player": learner_player, "players": rows,
            "status": manifest["status"], "termination_reason": manifest["termination_reason"],
            "learner_outcome": manifest["outcomes_by_player"][COLORS[learner_player]],
            "training_input": False, "policy_input": False, "reward_input": False,
            "limitations": ["Observed revisits can be purposeful; these counts do not establish aimlessness.",
                "A day without travel, revelation or capture can still contain construction or army preparation.",
                "This report checks visible history binding; native replay and policy reconstruction are separate gates."]}


def aggregate_episodes(reports):
    reports = list(reports)
    if any(row.get("schema") != SCHEMA for row in reports):
        raise ValueError("Mixed travel report schemas")
    if len({row["episode_directory"] for row in reports}) != len(reports):
        raise ValueError("Duplicate game would double count travel")
    result = {"schema": SCHEMA, "physical_games": len(reports),
              "natural_games": sum(row["status"] == "completed" for row in reports),
              "cutoff_games": sum(row["status"] == "truncated" for row in reports),
              "termination_reasons": dict(Counter(row["termination_reason"] for row in reports)),
              "learner_outcomes": dict(Counter(row["learner_outcome"] or "unscored_cutoff" for row in reports)),
              "training_input": False, "policy_input": False, "reward_input": False, "roles": {}}
    for role in ("learner", "frozen_opponent"):
        rows = [entry["metrics"] for report in reports for entry in report["players"].values() if entry["role"] == role]
        metrics = {}
        for key in ("choices", "last_public_day", "own_position_changes", "town_captures",
                    "unique_hero_position_pairs_including_initial", "newly_observed_tiles_after_initial",
                    "position_changes_revisiting_observed_hero_position"):
            values = [row[key] for row in rows if row[key] is not None]
            metrics[key] = {"available_games": len(values), "total": sum(values),
                            "mean": mean(values) if values else None, "median": median(values) if values else None,
                            "minimum": min(values) if values else None, "maximum": max(values) if values else None}
        metrics["games_with_no_observed_hero_position_change"] = sum(row["own_position_changes"] == 0 for row in rows)
        metrics["games_with_town_capture"] = sum(row["town_captures"] > 0 for row in rows)
        metrics["days_with_policy_actions"] = sum(len(row["days_with_policy_actions"]) for row in rows)
        metrics["days_without_observed_travel_reveal_or_capture"] = sum(len(row["days_without_observed_travel_reveal_or_capture"]) for row in rows)
        metrics["movement_results"] = dict(sum((Counter(row["movement"]["results"]) for row in rows), Counter()))
        metrics["action_counts"] = dict(sum((Counter(row["action_counts"]) for row in rows), Counter()))
        result["roles"][role] = metrics
    return result
