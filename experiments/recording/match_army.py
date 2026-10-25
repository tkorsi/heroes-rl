"""Time-bound spectator annotations for the recorded two-Castle scenario.

This is presentation data, never learner input. Panel observations refresh an
army only at or after their captured client-packet position. During automatic
combat the panel shows the army entering battle, not invented live casualties.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile


class ArmyTimelineError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ArmyTimelineError(message)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _slots(rows):
    result = []
    seen = set()
    for row in rows:
        slot, count, creature = row["slot"], row["count"], row["creature"]
        _require(type(slot) is int and 0 <= slot < 7 and slot not in seen,
                 "Invalid or duplicate army slot")
        _require(type(count) is int and count >= 0, "Invalid troop count")
        seen.add(slot)
        if count:
            _require(type(creature) is int and creature >= 0, "Missing creature identity")
            result.append({"slot": slot, "creature": creature, "count": count})
    return sorted(result, key=lambda row: row["slot"])


def _assemble_timeline(frames, endpoints, snapshots, *, player, hero_id, name, initial_slots):
    """Bind observations to authoritative input endpoints without looking ahead."""
    _require([frame["seq"] for frame in frames] == sorted(set(frame["seq"] for frame in frames)),
             "Trace frame sequence must be unique and increasing")
    _require(all(frame["seq"] in endpoints for frame in frames), "Missing native input endpoint")
    _require([row["packet"] for row in snapshots] == sorted(row["packet"] for row in snapshots),
             "Army observations must preserve packet order")
    current, source, fresh = deepcopy(initial_slots), 0, "initial_map"
    cursor = 0
    entering = None
    last_entry = None
    last_battle_packet = None
    result = {}
    for frame in frames:
        endpoint = endpoints[frame["seq"]]
        while cursor < len(snapshots) and snapshots[cursor]["packet"] <= endpoint:
            current = deepcopy(snapshots[cursor]["slots"])
            source = snapshots[cursor]["packet"]
            fresh = "observed"
            cursor += 1
        hero = next((row for row in frame["objects"] if row[0] == hero_id), None)
        if hero is not None:
            _require(hero[4] == 34 and hero[6] == player, "Selected hero identity changed")
        in_battle = hero is not None and any(player in battle[3:] for battle in frame.get("battles", []))
        if in_battle and entering is None:
            entering = (deepcopy(current), source)
            last_entry = deepcopy(entering)
        if in_battle:
            last_battle_packet = endpoint
        if hero is None:
            row = {"status": "absent", "slots": [], "position": None,
                   "source_packet": endpoint, "freshness": "absent"}
        elif in_battle:
            row = {"status": "battle", "slots": deepcopy(entering[0]), "position": hero[1:4],
                   "source_packet": entering[1], "freshness": "pre_battle"}
        else:
            # A surviving army remains explicitly stale until a later panel
            # observation; a closed window cannot supply post-battle losses.
            freshness = "pre_battle" if last_battle_packet is not None and source <= last_battle_packet else fresh
            displayed, displayed_packet = last_entry if freshness == "pre_battle" else (current, source)
            row = {"status": "active", "slots": deepcopy(displayed), "position": hero[1:4],
                   "source_packet": displayed_packet, "freshness": freshness}
        row.update(name=name, hero_id=hero_id)
        _require(row["source_packet"] <= endpoint, "Army annotation would reveal a future snapshot")
        result[frame["seq"]] = row
        if not in_battle:
            entering = None
    return result


def build_hero_army_timeline(episode: Path, frames: list[dict], player: int = 1) -> dict[int, dict]:
    """Annotate the selected side's original hero in a verified native trace.

    Supported initial hero/creature identities are deliberately restricted to
    this authored map. A different scenario fails instead of guessing names.
    The caller must verify/bind the trace and source capture before rendering.
    """
    episode = Path(episode)
    _require(player in (0, 1), "Expected red or blue player")
    color = ("red", "blue")[player]
    hero_type, name, subtype = (("core:orrin", "Orrin", 0), ("core:edric", "Edric", 2))[player]
    map_path = episode / "profile/Maps/headless.vmap"
    manifest = json.loads((episode / "manifest.json").read_text())
    _require(hashlib.sha256(map_path.read_bytes()).hexdigest() == manifest["identity"]["map_sha256"],
             "Captured map hash changed")
    with zipfile.ZipFile(map_path) as archive:
        objects = json.loads(archive.read("objects.json"))
    candidates = [obj for obj in objects if obj["type"] == "hero" and obj["options"]["owner"] == color]
    _require(len(candidates) == 1, "Expected one initial hero for this side")
    original = candidates[0]
    _require(original["options"]["type"] == hero_type, "Unknown original hero identity")
    _require(original["template"]["mask"] == ["VVV", "VAV"], "Unknown hero position anchor")
    initial_slots = []
    for slot, stack in enumerate(original["options"]["army"]):
        _require(stack["type"] == "core:pikeman", "Unknown initial creature identity")
        initial_slots.append({"slot": slot, "creature": 0, "count": stack["amount"]})
    initial_slots = _slots(initial_slots)
    position = [original["x"] - 1, original["y"], original["l"]]
    _require(bool(frames) and frames[0]["seq"] == -1, "Trace must include its initial state")
    native = [obj for obj in frames[0]["objects"]
              if obj[4] == 34 and obj[6] == player and obj[5] == subtype and obj[1:4] == position]
    _require(len(native) == 1, "Map hero does not match the native initial trace")
    hero_id = native[0][0]
    server = _records(episode / "engine/replay.jsonl")
    endpoints = {row["seq"]: row["client_out"] for row in server if row["kind"] == "applied"}
    endpoints[-1] = next(row["client_out"] for row in server if row["kind"] == "start")
    tape = _records(episode / "engine/contextual-replay.jsonl")
    packet, handle, snapshots = 0, None, []
    for transaction in tape:
        if transaction["kind"] == "packet_complete":
            _require(transaction["completed"] >= packet, "Client packet positions moved backwards")
            packet = transaction["completed"]
        if transaction["kind"] != "message" or transaction.get("player") != player:
            continue
        for record in transaction["message"].get("records", []):
            observation = record.get("observation") or {}
            if handle is None:
                matches = [obj for obj in observation.get("objects", [])
                           if obj["kind"] == 34 and obj["owner"] == "self"
                           and obj["subtype"] == subtype and obj["position"] == position]
                if matches:
                    _require(len(matches) == 1, "Ambiguous observed hero identity")
                    handle = matches[0]["handle"]
            window = observation.get("window") or {}
            for army in window.get("armies", []):
                if handle is not None and army["subject"] == handle:
                    _require(army["role"] in ("hero", "visiting_hero", "garrison_hero"),
                             "Selected army is not a hero army")
                    snapshots.append({"packet": packet, "slots": _slots(army["slots"])})
    _require(handle is not None and snapshots, "No recorded panel observations for this side's hero")
    _require(all(row["packet"] <= max(endpoints.values()) for row in snapshots),
             "Army observation lies beyond the authoritative replay")
    return _assemble_timeline(frames, endpoints, snapshots, player=player, hero_id=hero_id,
                              name=name, initial_slots=initial_slots)
