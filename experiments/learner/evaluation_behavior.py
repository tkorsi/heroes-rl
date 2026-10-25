"""Offline descriptive metrics from manifest-bound v7 player histories.

This module neither runs a policy nor reads omniscient engine state. Missing
evidence stays unavailable; a failed episode's observed prefix is not a whole
episode. Metrics are never observations, rewards, or training inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path

from . import contextual as c

SCHEMA = "heroes-v7-evaluation-behavior-v1"
NAVIGATION = frozenset(("back", "close", "open_building", "open_hero", "open_panel",
                       "open_stack", "open_town", "select_recruit", "select_hero", "select_upgrade"))
ARMY = frozenset(("split_stack", "merge_stacks", "swap_stacks", "swap_garrison"))
ARMY_EFFECTS = ARMY | {"recruit", "hire_hero", "upgrade_stack"}
LIMITATIONS = (
    "Counts describe available player-local prefixes, not hidden state or strategic quality.",
    "Current tile support comes from that decision's observation; the historical tile union is separately labelled offline memory.",
    "Battle events are counted per player perspective, not deduplicated into unique battles.",
    "Own roster loss does not identify its cause or reveal an enemy's hidden fate.",
    "Army deltas compare only directly displayed rows around an action; closed panels are not refreshed.",
    "Cross-army count redistribution is net change, not gross creatures moved by a swap of like stacks.",
    "This analyzer validates history bindings and frame structure, not native replay or policy reconstruction.",
)


def _binding(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _episode_path(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("History path escapes episode directory")
    return path


def _context(obs):
    return obs["window"]["context"]["kind"] if obs["window"] else "map"


def _rows(obs):
    return {row["subject"]: row for row in (obs["window"] or {}).get("armies", [])}


def _counts(row):
    total = Counter()
    for slot in row["slots"]:
        if slot["count"]:
            total[slot["creature"]] += slot["count"]
    return total


def _army_effect(action, before, after):
    rows = _rows(before)
    slots = {slot["ref"]: {"army_subject": row["subject"], "role": row["role"], **slot}
             for row in rows.values() for slot in row["slots"]}
    source, target = slots.get(action.get("selection")), slots.get(action.get("destination"))
    effect = {"source_before": source, "destination_before": target,
              "requested_count": action.get("count"), "transfer_scope": None,
              "comparison": "unavailable", "count_deltas": None, "slot_changes": None,
              "observed_cross_army_net_redistribution": None}
    if source and target:
        effect["transfer_scope"] = "within_army" if source["army_subject"] == target["army_subject"] else "across_armies"
    if after is None or not rows or not _rows(after):
        return effect
    following = _rows(after)
    shared = rows.keys() & following.keys()
    effect.update(comparison="displayed_rows_only", count_deltas=[], slot_changes=[],
                  comparable_subjects=sorted(shared),
                  no_longer_displayed=sorted(rows.keys() - following.keys()),
                  newly_displayed=sorted(following.keys() - rows.keys()))
    for subject in sorted(shared):
        old, new = _counts(rows[subject]), _counts(following[subject])
        for creature in sorted(old.keys() | new.keys()):
            if old[creature] != new[creature]:
                effect["count_deltas"].append({"subject": subject, "creature": creature,
                    "before": old[creature], "after": new[creature], "delta": new[creature] - old[creature]})
        for left, right in zip(rows[subject]["slots"], following[subject]["slots"]):
            if (left["creature"], left["count"]) != (right["creature"], right["count"]):
                effect["slot_changes"].append({"subject": subject, "slot": left["slot"],
                    "before": {k: left[k] for k in ("creature", "count")},
                    "after": {k: right[k] for k in ("creature", "count")}})
    if effect["transfer_scope"] == "across_armies" and {source["army_subject"], target["army_subject"]} <= shared:
        gains, losses = Counter(), Counter()
        for delta in effect["count_deltas"]:
            (gains if delta["delta"] > 0 else losses)[delta["creature"]] += abs(delta["delta"])
        if gains == losses:
            effect["observed_cross_army_net_redistribution"] = [{"creature": creature, "count": count} for creature, count in sorted(gains.items())]
    return effect


def _player_metrics(frames):
    observations = [f for f in frames if f.get("observation") is not None]
    observation_seqs = [f["seq"] for f in observations]
    decisions = {f["seq"]: f for f in frames if f["kind"] == "decision"}
    submitted = [f for f in frames if f["kind"] == "submitted"]
    results = {f["decision_seq"]: f for f in frames if f["kind"] == "result"}
    if len({f["decision_seq"] for f in submitted}) != len(submitted) or len(results) != sum(f["kind"] == "result" for f in frames):
        raise ValueError("Duplicate action or result for one decision")
    actions = Counter(f["action"]["kind"] for f in submitted)
    events = Counter(f["event"]["kind"] for f in frames if f.get("event"))
    issues, moves, army_effects, dialogs = [], [], [], []
    for frame in submitted:
        action = frame["action"]
        decision = decisions.get(frame["decision_seq"])
        result_frame = results.get(frame["decision_seq"])
        result = result_frame["result"] if result_frame else None
        if decision is None:
            raise ValueError("Submitted action has no matching decision")
        obs = decision["observation"]
        if action["kind"] == "move_to":
            hero = next((h for h in obs["heroes"] if h["handle"] == action["subject"]), None)
            if hero is None:
                raise ValueError("Movement refers to a hero absent from its own decision")
            target = [action["x"], action["y"], hero["position"][2]]
            tile = next((t for t in obs["tiles"] if t["position"] == target), None)
            towns = [o for o in obs["objects"] if o["kind"] == 98 and o["owner"] == "opponent"]
            endpoint = result.get("position") if result else None
            moves.append({"decision_seq": decision["seq"], "day": decision["day"],
                "subject": action["subject"], "origin": hero["position"], "target": target,
                "target_in_current_known_tiles": tile is not None,
                "observed_target_blocked": tile["blocked"] if tile else None,
                "opponent_town_visible": bool(towns),
                "targets_visible_opponent_town": any(o["position"] == target for o in towns),
                "result_status": result["status"] if result else None,
                "endpoint": endpoint, "endpoint_changed": endpoint != hero["position"] if endpoint is not None else None})
        if action["kind"] in ARMY_EFFECTS:
            # Native refreshes precede the result. Do not borrow a future panel
            # opened by another action or use a stale cached private army.
            index = bisect_left(observation_seqs, result_frame["seq"]) - 1 if result_frame else -1
            after = observations[index]["observation"] if index >= 0 and observation_seqs[index] > decision["seq"] else None
            army_effects.append({"decision_seq": decision["seq"], "kind": action["kind"],
                "result_status": result["status"] if result else None,
                "accepted": result["accepted"] if result else None,
                **_army_effect(action, obs, after)})
        if action["kind"] == "choose_dialog":
            panel = (obs["window"] or {}).get("dialog_panel")
            choice = next((x for x in panel["choices"] if x["ref"] == action["selection"]), None) if panel else None
            dialogs.append({"decision_seq": decision["seq"], "family": panel["family"] if panel else None,
                            "choice_kind": choice["kind"] if choice else None,
                            "component_index": choice["component_index"] if choice else None,
                            "result_status": result["status"] if result else None})
    positions, position_changes = defaultdict(set), []
    known, new_tiles = set(), set()
    hero_changes, town_changes, ownership_changes, captures, town_visits = [], [], [], [], []
    previous, previous_seq, latest_visits = None, None, {}
    for frame in frames:
        event = frame.get("event")
        if event and event["kind"] == "visit" and event["object_kind"] == 98:
            visit = {"seq": frame["seq"], "day": frame["day"], "subject": event["subject"], "owner": event["owner"]}
            town_visits.append(visit); latest_visits[event["subject"]] = visit
        obs = frame.get("observation")
        if obs is None:
            continue
        tiles = {tuple(t["position"]) for t in obs["tiles"]}
        if previous is not None:
            new_tiles.update(tiles - known)
        known.update(tiles)
        for hero in obs["heroes"]:
            positions[hero["handle"]].add(tuple(hero["position"]))
        if previous is not None:
            for field, changes in (("heroes", hero_changes), ("towns", town_changes)):
                old = {x["handle"] for x in previous[field]}; new = {x["handle"] for x in obs[field]}
                if old != new:
                    changes.append({"seq": frame["seq"], "day": frame["day"], "added": sorted(new-old), "removed": sorted(old-new)})
            old_heroes = {x["handle"]: x for x in previous["heroes"]}
            for hero in obs["heroes"]:
                old = old_heroes.get(hero["handle"])
                if old and old["position"] != hero["position"]:
                    position_changes.append({"seq": frame["seq"], "day": frame["day"], "subject": hero["handle"],
                                             "before": old["position"], "after": hero["position"]})
            old_towns = {x["handle"]: x for x in previous["objects"] if x["kind"] == 98}
            owner_changes = {}
            for obj in obs["objects"]:
                old = old_towns.get(obj["handle"])
                if obj["kind"] == 98 and old and old["owner"] != obj["owner"]:
                    change = {"seq": frame["seq"], "previous_observation_seq": previous_seq,
                              "subject": obj["handle"], "before": old["owner"], "after": obj["owner"]}
                    ownership_changes.append(change); owner_changes[obj["handle"]] = change
            added = {t["handle"] for t in obs["towns"]} - {t["handle"] for t in previous["towns"]}
            for subject in sorted(added):
                visit = latest_visits.get(subject)
                recent = visit if visit and previous_seq < visit["seq"] < frame["seq"] and visit["owner"] != "self" else None
                change = owner_changes.get(subject)
                if recent or change and change["after"] == "self":
                    captures.append({"seq": frame["seq"], "subject": subject,
                                     "own_roster_added": True, "visible_visit": recent, "visible_owner_change": change})
        previous, previous_seq = obs, frame["seq"]
    first = observations[0]["observation"] if observations else None
    last = observations[-1]["observation"] if observations else None
    successes = Counter(f["action"]["kind"] for f in submitted if f["decision_seq"] in results
                        and results[f["decision_seq"]]["result"]["accepted"]
                        and results[f["decision_seq"]]["result"]["status"] in ("recruited", "hired", "upgraded"))
    metrics = {"frames": len(frames), "choices": len(submitted), "decision_frames": len(decisions),
        "first_public_day": min((f["day"] for f in frames), default=None),
        "last_public_day": max((f["day"] for f in frames), default=None),
        "action_counts": dict(sorted(actions.items())), "event_counts": dict(sorted(events.items())),
        "result_counts": dict(sorted(Counter(x["result"]["status"] for x in results.values()).items())),
        "missing_results": sum(f["decision_seq"] not in results for f in submitted),
        "native_acknowledgements": dict(Counter("accepted" if f["accepted"] else "rejected" for f in frames if f["kind"] == "acknowledged")),
        "navigation_and_selection": sum(actions[k] for k in NAVIGATION),
        "army_rearrangement": sum(actions[k] for k in ARMY),
        "decision_context_counts": dict(Counter(_context(f["observation"]) for f in decisions.values())),
        "movement": {"attempts": len(moves), "known_targets": sum(x["target_in_current_known_tiles"] for x in moves),
            "unknown_targets": sum(not x["target_in_current_known_tiles"] for x in moves),
            "endpoint_changes": sum(x["endpoint_changed"] is True for x in moves),
            "missing_endpoints": sum(x["endpoint_changed"] is None for x in moves),
            "while_opponent_town_visible": sum(x["opponent_town_visible"] for x in moves),
            "targets_visible_opponent_town": sum(x["targets_visible_opponent_town"] for x in moves),
            "results": dict(Counter(x["result_status"] or "no_result" for x in moves))},
        "movement_attempts": moves, "own_hero_positions": {k: [list(p) for p in sorted(v)] for k, v in sorted(positions.items())},
        "own_position_changes": position_changes,
        "known_tiles": {"first": len(first["tiles"]), "last": len(last["tiles"]),
                        "observed_union": len(known), "new_after_first_observation": len(new_tiles)} if first else None,
        "own_hero_roster_changes": hero_changes, "own_town_roster_changes": town_changes,
        "owned_heroes_first_last": [len(first["heroes"]), len(last["heroes"])] if first else None,
        "owned_towns_first_last": [len(first["towns"]), len(last["towns"])] if first else None,
        "visible_town_visits": town_visits, "visible_town_owner_changes": ownership_changes,
        "town_capture_evidence": captures,
        "recruitment": {"attempts": actions["recruit"], "completed": successes["recruit"]},
        "hiring": {"attempts": actions["hire_hero"], "completed": successes["hire_hero"]},
        "upgrading": {"attempts": actions["upgrade_stack"], "completed": successes["upgrade_stack"]},
        "army_effects": army_effects, "dialog_answers": dialogs,
        "army_operations": {"displayed_comparisons": sum(x["comparison"] == "displayed_rows_only" for x in army_effects),
            "unavailable_comparisons": sum(x["comparison"] == "unavailable" for x in army_effects),
            "across_armies": sum(x["transfer_scope"] == "across_armies" for x in army_effects),
            "within_army": sum(x["transfer_scope"] == "within_army" for x in army_effects),
            "observed_cross_army_net_count_redistribution": sum(v["count"] for x in army_effects for v in x["observed_cross_army_net_redistribution"] or [])},
        "battles": {"starts": events["battle_start"], "completions": events["battle_complete"],
                    "visible_results": [{"seq": f["seq"], "day": f["day"], "side": f["event"]["side"], "won": f["event"]["won"]}
                                        for f in frames if f.get("event") and f["event"]["kind"] == "battle_complete"]},
        "terminal_messages": [{"seq": f["seq"], "outcome": f["outcome"]} for f in frames if f["kind"] == "terminal"],
        "cutoffs": [{"seq": f["seq"], "reason": f["reason"]} for f in frames if f["kind"] == "cutoff"]}
    return metrics, issues


def _aggregate(players):
    available = [p["metrics"] for p in players if p["metrics"] is not None]
    if not available:
        return None
    result = {"available_player_histories": len(available), "expected_player_histories": len(players),
              "complete_player_histories": sum(p["coverage"]["complete"] for p in players),
              "first_public_day_min": min(m["first_public_day"] for m in available),
              "last_public_day_max": max(m["last_public_day"] for m in available)}
    for key in ("frames", "choices", "decision_frames", "missing_results", "navigation_and_selection", "army_rearrangement"):
        result[key] = sum(m[key] for m in available)
    for key in ("action_counts", "event_counts", "result_counts", "native_acknowledgements", "decision_context_counts"):
        result[key] = dict(sorted(sum((Counter(m[key]) for m in available), Counter()).items()))
    result["movement"] = {k: sum(m["movement"][k] for m in available) for k in available[0]["movement"] if k != "results"}
    result["movement"]["results"] = dict(sorted(sum((Counter(m["movement"]["results"]) for m in available), Counter()).items()))
    for key in ("recruitment", "hiring", "upgrading"):
        result[key] = {k: sum(m[key][k] for m in available) for k in ("attempts", "completed")}
    result["army_operations"] = {k: sum(m["army_operations"][k] for m in available) for k in available[0]["army_operations"]}
    result["player_visible_battle_starts"] = sum(m["battles"]["starts"] for m in available)
    result["player_visible_battle_completions"] = sum(m["battles"]["completions"] for m in available)
    for key in ("own_hero_roster_changes", "own_town_roster_changes", "town_capture_evidence", "dialog_answers", "army_effects"):
        result[key] = sum(len(m[key]) for m in available)
    result["new_known_tiles_across_player_perspectives"] = sum(m["known_tiles"]["new_after_first_observation"] for m in available if m["known_tiles"] is not None)
    return result


def analyze_episode(folder):
    """Read a v7 manifest and its declared histories, without modifying files.

    Missing evidence returns explicit incomplete coverage. Binding mismatches,
    invalid frames, cross-player data and escaping paths raise ValueError.
    A bound failed file may end in one partial JSON line; only its valid prefix
    is described. Histories not declared with a hash are never analyzed.
    """
    folder = Path(folder).resolve()
    result = {"schema": SCHEMA, "episode_directory": str(folder), "status": "incomplete",
              "bindings": {}, "coverage": {"complete": False, "issues": []},
              "players": {color: {"coverage": {"complete": False, "hash_verified": False,
                                                 "issues": ["manifest_missing"]}, "metrics": None}
                          for color in ("red", "blue")},
              "aggregate": None, "training_input": False, "policy_input": False,
              "reward_input": False, "limitations": list(LIMITATIONS)}
    path = folder / "manifest.json"
    if not path.is_file():
        result["coverage"]["issues"].append("manifest_missing")
        return result
    data = path.read_bytes(); manifest = json.loads(data)
    result["bindings"]["manifest.json"] = _binding(data)
    if manifest.get("history_schema") != c.DIALOGS_SCHEMA:
        raise ValueError("Behavior analysis requires explicit history v7")
    result["episode"] = {k: manifest.get(k) for k in ("episode", "seed", "status", "termination_reason",
        "failure_detail", "wall_seconds", "outcomes_by_player", "rewards_by_player")}
    valid_status = manifest.get("status") in ("truncated", "completed")
    if not valid_status:
        result["coverage"]["issues"].append("episode_not_validly_finalized")
    for player, color in enumerate(("red", "blue")):
        declared = manifest.get("players", {}).get(color, {}).get("history_v7")
        entry = {"coverage": {"complete": False, "hash_verified": False, "issues": []}, "metrics": None}
        result["players"][color] = entry
        if not declared or not declared.get("path") or not declared.get("sha256"):
            entry["coverage"]["issues"].append("manifest_history_binding_missing")
            continue
        path = _episode_path(folder, declared["path"])
        entry["history"] = str(path.relative_to(folder))
        if not path.is_file():
            entry["coverage"]["issues"].append("bound_history_file_missing")
            continue
        data = path.read_bytes(); identity = _binding(data)
        if identity["sha256"] != declared["sha256"] or ("bytes" in declared and identity["bytes"] != declared["bytes"]):
            raise ValueError(f"Manifest-bound {color} history changed")
        result["bindings"][entry["history"]] = identity
        entry["coverage"]["hash_verified"] = True
        frames, lines = [], data.splitlines()
        for index, line in enumerate(lines):
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not valid_status and index == len(lines) - 1 and not data.endswith(b"\n"):
                    entry["coverage"]["issues"].append("trailing_partial_json_line")
                    break
                raise ValueError(f"Malformed {color} history line {index + 1}") from None
            frame = c.decode_frame(raw)
            if frame.schema != c.DIALOGS_SCHEMA or frame.player != player or frame.seq != index or frame.episode != str(manifest["episode"]):
                raise ValueError("History schema, sequence, episode or player mismatch")
            frames.append(raw)
        if not frames:
            entry["coverage"]["issues"].append("no_available_frames")
            continue
        entry["metrics"], issues = _player_metrics(frames)
        entry["coverage"]["issues"].extend(issues)
        expected = "terminal" if manifest.get("status") == "completed" else "cutoff"
        has_boundary = any(f["kind"] == expected for f in frames)
        entry["coverage"]["complete"] = valid_status and declared.get("complete") is True and has_boundary and not entry["coverage"]["issues"]
        if not entry["coverage"]["complete"]:
            entry["coverage"]["issues"].append("observed_prefix_only")
    result["aggregate"] = _aggregate(list(result["players"].values()))
    result["coverage"]["complete"] = valid_status and all(x["coverage"]["complete"] for x in result["players"].values())
    result["status"] = "complete" if result["coverage"]["complete"] else "incomplete"
    wall = manifest.get("wall_seconds")
    wall = wall if type(wall) in (int, float) and math.isfinite(wall) and wall > 0 else None
    result["throughput"] = {"manifest_wall_seconds": wall,
        "observed_choices_per_second": result["aggregate"]["choices"] / wall if wall and result["aggregate"] else None,
        "scope": "Captured episode wall time; excludes replay, checkpoint setup and offline analysis."}
    return result


def aggregate_episodes(reports):
    """Add observed-prefix counts with explicit evidence denominators."""
    reports = list(reports)
    if any(r.get("schema") != SCHEMA for r in reports):
        raise ValueError("Cannot mix behavior report schemas")
    if len({r["episode_directory"] for r in reports}) != len(reports):
        raise ValueError("Duplicate episode would double-count behavior")
    players = [p for r in reports for p in r["players"].values()]
    aggregate = _aggregate(players)
    wall = [r.get("throughput", {}).get("manifest_wall_seconds") for r in reports]
    timed = [r for r in reports if r.get("throughput", {}).get("manifest_wall_seconds") and r["aggregate"] is not None]
    seconds = sum(r["throughput"]["manifest_wall_seconds"] for r in timed)
    return {"schema": SCHEMA, "episodes": len(reports), "complete_episodes": sum(r["status"] == "complete" for r in reports),
            "incomplete_episodes": sum(r["status"] != "complete" for r in reports), "aggregate": aggregate,
            "wall_seconds_available": sum(x is not None for x in wall),
            "timed_episodes_with_metrics": len(timed), "timed_episode_wall_seconds": seconds if timed else None,
            "observed_choices_per_second": sum(r["aggregate"]["choices"] for r in timed) / seconds if seconds else None,
            "training_input": False, "policy_input": False, "reward_input": False}
