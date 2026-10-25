#!/usr/bin/env python3
"""Stream player-local contextual histories into descriptive behavior evidence.

This reads the manifest and each player's own canonical learner history only.
It never reads engine state, bridges, neural rollouts, opponents' private windows,
or replay tapes. Content hashes and frame identity/order are checked. This is a
descriptive audit, not an independent episode/finalization/replay verifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.probes.create_boundary_fixture import decode

COLORS = ("red", "blue")
NAVIGATION = {"back", "close", "open_building", "open_hero", "open_panel", "open_town", "select_recruit"}


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def summarize_player(path, player, expected_sha256):
    actions, results, kinds, events, acknowledgements = (Counter() for _ in range(5))
    periods, movement, endings, recruits, destinations, buildings, visits = (
        defaultdict(Counter), Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
    battles, losses, hero_changes, town_changes = [], [], [], []
    hero_windows, pending, roads, counts = {}, {}, set(), []
    initial = latest = last_action = None
    episode = schema = None
    digest, byte_count, frame_count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            byte_count += len(raw)
            frame = decode(raw)
            if (not raw.endswith(b"\n") or type(frame) is not dict
                    or frame.get("seq") != frame_count or type(frame.get("seq")) is not int
                    or frame.get("player") != player or type(frame.get("player")) is not int
                    or frame.get("schema") not in ("heroes-player-history-v2", "heroes-player-history-v3")):
                raise ValueError("History identity, order, framing or schema mismatch")
            if episode is None:
                episode, schema = frame.get("episode"), frame["schema"]
            if not isinstance(episode, str) or not episode or frame.get("episode") != episode or frame["schema"] != schema:
                raise ValueError("History episode/schema changed")
            seq, day, kind = frame["seq"], frame["day"], frame["kind"]
            if type(day) is not int or day < 1:
                raise ValueError("Invalid history day")
            frame_count += 1
            kinds[kind] += 1
            obs = frame.get("observation")
            if obs is not None:
                current = (len(obs["heroes"]), len(obs["towns"]))
                if counts:
                    for index, changes in ((0, hero_changes), (1, town_changes)):
                        if current[index] != counts[-1][index]:
                            changes.append({"day": day, "sequence": seq, "before": counts[-1][index], "after": current[index]})
                counts.append(current)
                initial = initial or obs
                latest = obs
                roads.update(tile["road"] for tile in obs["tiles"])
                window = obs.get("window")
                if window and window["context"]["kind"] == "hero":
                    army = window["army"]
                    hero_windows[window["context"]["subject"]] = {
                        "day": day, "sequence": seq, "army": army, "total_creatures": sum(stack["count"] for stack in army)}
            if kind == "submitted":
                if latest is None:
                    raise ValueError("Submitted action precedes its own observation")
                action = frame["action"]
                name = action["kind"]
                actions[name] += 1
                quarter = (day - 1) // 25
                periods[f"days-{quarter * 25 + 1}-{quarter * 25 + 25}"][name] += 1
                last_action = {"action": action, "day": day, "sequence": seq, "decision_seq": frame["decision_seq"]}
                metadata = {"action": action}
                if name == "move_to":
                    target = [action["x"], action["y"], 0]
                    tiles = {tuple(tile["position"]): tile for tile in latest["tiles"]}
                    visible = [item for item in latest["objects"] if item["position"] == target]
                    towns = [item for item in latest["objects"] if item["kind"] == 98 and item["owner"] == "opponent"]
                    movement["attempts"] += 1
                    movement["known_targets" if tuple(target) in tiles else "unknown_targets"] += 1
                    movement["observed_blocked_targets"] += bool(tiles.get(tuple(target), {}).get("blocked"))
                    movement["while_opponent_town_visible"] += bool(towns)
                    movement["targeting_visible_opponent_town"] += any(item["position"] == target for item in towns)
                    own = next((hero for hero in latest["heroes"] if hero["handle"] == action["subject"]), None)
                    if own is None:
                        raise ValueError("Move subject absent from its player's current owned list")
                    movement["at_zero_movement"] += own["movement"] == 0
                    last_action["visible_destination_objects"] = visible
                elif name == "end_turn":
                    endings["total"] += 1
                    heroes = latest["heroes"]
                    endings["with_owned_hero"] += bool(heroes)
                    endings["with_positive_movement"] += any(hero["movement"] > 0 for hero in heroes)
                    endings["with_full_movement"] += any(hero.get("movement_max") is not None
                                                        and hero["movement"] == hero["movement_max"] for hero in heroes)
                elif name == "recruit":
                    panel = latest["window"]["town_panel"]["recruitment"]
                    if panel["destination"]["ref"] != action["destination"]:
                        raise ValueError("Recruitment destination differs from its own displayed destination")
                    metadata["destination_role"] = panel["destination"]["role"]
                    destinations[metadata["destination_role"]] += 1
                elif name == "construct":
                    metadata["building_name"] = latest["window"]["town_panel"]["building"]["name_text_id"]
                pending[frame["decision_seq"]] = metadata
            elif kind == "result":
                result = frame["result"]
                results[result["status"]] += 1
                metadata = pending.get(frame["decision_seq"])
                if metadata is None:
                    raise ValueError("Result has no player-local submitted action")
                action = metadata["action"]
                if action["kind"] == "move_to":
                    movement["result_" + result["status"]] += 1
                if result["status"] == "recruited" and result["accepted"]:
                    recruits["actions"] += 1
                    recruits["creatures"] += action["count"]
                    recruits[metadata["destination_role"] + "_creatures"] += action["count"]
                if result["status"] == "built" and result["accepted"]:
                    buildings[metadata["building_name"]] += 1
            elif kind == "acknowledged":
                if type(frame["accepted"]) is not bool:
                    raise ValueError("Native acknowledgement is not boolean")
                acknowledgements["accepted" if frame["accepted"] else "rejected"] += 1
            event = frame.get("event")
            if event:
                event_kind = event["kind"]
                events[event_kind] += 1
                if event_kind == "hero_lost":
                    losses.append({"day": day, "sequence": seq, "event": event})
                if event_kind in ("battle_start", "battle_complete"):
                    battles.append({"day": day, "sequence": seq, "kind": event_kind,
                                    "side": event.get("side"), "won": event.get("won"),
                                    "most_recent_own_submitted_action": last_action})
                if event_kind == "visit" and event.get("object_kind") is not None:
                    visits[f"{event['object_kind']}:{event['owner']}"] += 1
    if digest.hexdigest() != expected_sha256 or initial is None:
        raise ValueError("History content hash mismatch or no player observation")
    total = sum(actions.values())
    navigation = sum(actions[name] for name in NAVIGATION)
    return {
        "history_identity": {"bytes": byte_count, "sha256": digest.hexdigest()}, "history_schema": schema,
        "episode": episode, "player": player, "frames": frame_count, "frame_kinds": dict(kinds),
        "submitted_total": total, "submitted_actions": dict(actions), "consumed_actions": kinds["consumed"],
        "resolved_actions": kinds["result"], "action_results": dict(results),
        "navigation_actions": navigation, "navigation_fraction": navigation / total if total else None,
        "actions_by_25_day_period": {key: dict(value) for key, value in periods.items()},
        "movement": dict(movement), "end_turn": dict(endings), "native_acknowledgements": dict(acknowledgements),
        "events": dict(events), "hero_loss_events": losses, "battles": battles,
        "owned_hero_changes": hero_changes, "owned_town_changes": town_changes,
        "initial_owned_heroes": len(initial["heroes"]), "initial_owned_towns": len(initial["towns"]),
        "final_observed_owned_heroes": len(latest["heroes"]), "final_observed_owned_towns": len(latest["towns"]),
        "owned_hero_count_range": [min(item[0] for item in counts), max(item[0] for item in counts)],
        "owned_town_count_range": [min(item[1] for item in counts), max(item[1] for item in counts)],
        "last_observed_day": day, "observed_road_ids": sorted(roads),
        "resources_initial": initial["resources"], "resources_final_observed": latest["resources"],
        "last_explicit_hero_window_armies_historical_only": hero_windows,
        "recruit_actions_by_visible_destination_role": dict(destinations), "successful_recruitment": dict(recruits),
        "successful_construction_by_displayed_name": dict(buildings), "visits_by_object_kind_relative_owner": dict(visits),
    }


def audit_episode(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = decode(manifest_bytes)
    players = {}
    for player, color in enumerate(COLORS):
        declarations = manifest["players"][color]
        profiles = [name for name in ("history_v2", "history_v3") if name in declarations]
        if len(profiles) != 1:
            raise ValueError("Expected exactly one contextual learner-history profile")
        declaration = declarations[profiles[0]]
        path = (directory / declaration["path"]).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError("History path escapes the episode")
        players[color] = summarize_player(path, player, declaration["sha256"])
    if players["red"]["episode"] != players["blue"]["episode"] or players["red"]["history_schema"] != players["blue"]["history_schema"]:
        raise ValueError("Paired player episode/schema identities differ")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during audit")
    return {
        "schema": "heroes-ordinary-behavior-audit-v1", "episode": str(directory),
        "episode_status": manifest.get("status"), "termination_reason": manifest.get("termination_reason"),
        "inputs": {"manifest.json": {"bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}},
        "players": players, "audit_source": identity(Path(__file__)),
        "findings": {
            "both_players_retained_one_own_town_in_all_observations": all(row["owned_town_count_range"] == [1, 1] for row in players.values()),
            "both_players_last_observed_without_heroes": all(row["final_observed_owned_heroes"] == 0 for row in players.values()),
            "all_successful_recruitment_to_town_garrisons": all(row["successful_recruitment"].get("creatures", 0)
                == row["successful_recruitment"].get("town_garrison_creatures", 0) for row in players.values()),
            "ordinary_native_rejections": sum(row["native_acknowledgements"].get("rejected", 0) for row in players.values()),
        },
        "player_memory_semantics": "Hero armies are timestamped last explicitly opened window observations, never current after close.",
        "scope": "Descriptive player-local history inventory; requires separate contract, finalization and native replay audits.",
        "engine_launched": False, "training_performed": False,
    }


def aggregate(reports):
    rows = [row for report in reports for row in report["players"].values()]
    def summed(field):
        result = Counter()
        for row in rows:
            result.update(row[field])
        return dict(result)
    return {"episodes": len(reports), "episode_statuses": dict(Counter(report["episode_status"] for report in reports)),
            "submitted_total": sum(row["submitted_total"] for row in rows),
            "navigation_actions": sum(row["navigation_actions"] for row in rows),
            "submitted_actions": summed("submitted_actions"), "movement": summed("movement"),
            "end_turn": summed("end_turn"), "successful_recruitment": summed("successful_recruitment"),
            "native_acknowledgements": summed("native_acknowledgements"),
            "hero_loss_events": sum(len(row["hero_loss_events"]) for row in rows),
            "own_town_count_changes": sum(len(row["owned_town_changes"]) for row in rows),
            "both_heroless_episodes": sum(report["findings"]["both_players_last_observed_without_heroes"] for report in reports)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")
    reports = [audit_episode(path) for path in args.episodes]
    output = reports[0] if len(reports) == 1 else {
        "schema": "heroes-ordinary-behavior-cohort-v1", "episodes": reports, "aggregate": aggregate(reports),
        "engine_launched": False, "training_performed": False}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(reports),
                      "aggregate": aggregate(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
