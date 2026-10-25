#!/usr/bin/env python3
"""Create isolated hidden-world evaluation fixtures from a real history-v2 view.

Only the authored random-baseline-v1 layout is supported. The base archive is a
byte-for-byte copy; the variant changes unknown terrain/roads and/or appends one
fully hidden neutral guard. Existing objects, starting state and victory rules
are untouched. These are evaluation fixtures, never policy inputs or training
data. The supplied record must come from real VCMI; this tool cannot authenticate
that claim. Real initial-prefix equality and gameplay checks remain mandatory.

Example (no engine is launched):
  python3 validation/probes/create_contextual_fixtures.py \
      --observation runs/context-check/initial-red-frame.json \
      --variant guard --output runs/context-hidden-guard-fixtures
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explorer.create_scenario import scenario_payloads
from experiments.learner import contextual as contract

UPSTREAM = "6901839ae9760dfcd1d5ff2c4aa4ef6968fee752"
BASELINE = ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"
VARIANTS = ("terrain", "roads", "guard", "all")
# Audited against the pinned loader/saver and definitions listed in the manifest.
# Water's ordinary views include 20..32; rock's include 0..7. Horizontal road
# dead ends use view 15, with the right endpoint horizontally flipped.
TERRAIN_TOKENS = (("water", "wt20_"), ("rock", "rc0_"))
ROAD_SUFFIXES = ("pd15_", "pd15-")
DIRT_TOKEN = re.compile(r"dt(?:2[1-9]|3[0-9]|4[0-3])_")


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _positions(obj, *, visible_only=False):
    """Native mask rows are reversed relative to the bottom-right anchor.

    ObjectTemplate::readJson maps V/B/A to VISIBLE; H/T have no visible image.
    Conservatively reserve every nonempty cell when checking collisions.
    """
    mask = obj["template"]["mask"]
    result = set()
    for row_index, row in enumerate(mask):
        for column, cell in enumerate(row):
            if cell not in "0 VBAHT":
                # The authored scenario has rectangular V/B/A masks. Reject
                # unsupported symbols rather than silently treating them empty.
                raise ValueError(f"Unsupported object mask symbol: {cell!r}")
            if cell in " 0" or visible_only and cell not in "VBA":
                continue
            result.add((obj["x"] - len(row) + 1 + column,
                        obj["y"] - len(mask) + 1 + row_index, obj["l"]))
    return result


def _ordered(positions):
    return sorted(positions, key=lambda pos: (pos[2], pos[1], pos[0]))


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) if a[2] == b[2] else 10**9


def _load_inputs(baseline_map, observation_path):
    pin = _json((ROOT / "dependencies/vcmi.json").read_bytes())["commit"]
    if pin != UPSTREAM:
        raise ValueError("VCMI pin changed; re-audit fixture terrain encoding and object visibility first")
    source_bytes = baseline_map.read_bytes()
    with zipfile.ZipFile(baseline_map) as archive:
        infos = archive.infolist()
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError("Duplicate source archive entries")
        contents = {info.filename: archive.read(info) for info in infos}
    expected = scenario_payloads(0)
    if set(contents) != set(expected) or any(_json(contents[name]) != value for name, value in expected.items()):
        raise ValueError("Only the unchanged authored random-baseline-v1 seed-0 scenario is supported")
    frame_bytes = observation_path.read_bytes()
    raw = _json(frame_bytes)
    frame = contract.decode_frame(raw)
    if frame.kind not in ("view", "decision") or frame.observation is None:
        raise ValueError("Supply a complete history-v2 view or decision frame, including player/day identity")
    observation = frame.observation
    if frame.day != 1 or observation.context != contract.Context():
        raise ValueError("The fixture reference must be a day-1 map-context observation")
    if observation.size != (20, 20, 1):
        raise ValueError("Observation dimensions do not match the authored source map")
    start = (4 if frame.player == 0 else 15, 7, 0)
    if (len(observation.heroes) != 1 or len(observation.towns) != 1
            or observation.heroes[0].position != start or observation.towns[0].position != start):
        raise ValueError("Observation must show the player's unchanged initial hero and town positions")
    if not observation.tiles or any(tile.terrain != 0 or tile.road != 0 for tile in observation.tiles):
        raise ValueError("Reference terrain/roads do not match the all-dirt, road-free baseline")
    known = {tile.position for tile in observation.tiles}
    if start not in known:
        raise ValueError("Initial owned hero tile is missing from the supplied current view")
    return source_bytes, infos, contents, deepcopy(expected), frame_bytes, frame, known


def _guard_variant(objects, known, all_tiles, reserved):
    template = next(obj for obj in objects if obj["instanceName"] == "guard_0")
    if template["options"].get("owner") is not None:
        raise ValueError("Guard template must use native neutral ownership by omission")
    candidates = []
    for anchor in _ordered(all_tiles - known - reserved):
        guard = deepcopy(template)
        guard.update(instanceName="validation_guard_0", x=anchor[0], y=anchor[1], l=anchor[2])
        footprint, visible = _positions(guard), _positions(guard, visible_only=True)
        if not footprint <= all_tiles or footprint & (known | reserved):
            continue
        # Known targets must be free of all existing object footprints. The
        # appended guard's native zone can then reach a known ordinary land tile.
        adjacent = {pos for pos in known - reserved if _distance(pos, anchor) == 1}
        if adjacent:
            candidates.append((guard, footprint, visible, adjacent))
    if not candidates:
        raise ValueError("No fully hidden, nonoverlapping guard footprint borders a known clear tile")
    guard, footprint, visible, adjacent = candidates[0]
    objects.append(guard)
    return footprint, {
        "operation": "append_neutral_guard", "object": guard,
        "source_template": "guard_0", "occupied_footprint": [list(pos) for pos in _ordered(footprint)],
        "visible_footprint": [list(pos) for pos in _ordered(visible)],
        "known_adjacent_clear_tiles": [list(pos) for pos in _ordered(adjacent)],
        "static_visibility_check": "Every occupied and visible template cell is outside the supplied known set",
    }


def build_fixtures(observation_path, output, *, baseline_map=BASELINE, variant="all"):
    """Create new fixture files; refuse ambiguous references and occupied output."""
    observation_path, output, baseline_map = (Path(path).resolve() for path in (observation_path, output, baseline_map))
    if variant not in VARIANTS:
        raise ValueError("Unsupported fixture variant")
    if output.exists():
        raise FileExistsError("Fixture output must be a new directory")
    for authored in (ROOT / "engine", ROOT / "experiments", ROOT / "dependencies"):
        if output == authored or authored in output.parents:
            raise ValueError("Write fixture outputs outside authored engine, experiment and dependency sources")
    source_bytes, infos, contents, payloads, frame_bytes, frame, known = _load_inputs(baseline_map, observation_path)
    objects, terrain = payloads["objects.json"], payloads["surface_terrain.json"]
    original_objects = deepcopy(objects)
    all_tiles = {(x, y, 0) for y in range(20) for x in range(20)}
    reserved = set().union(*(_positions(obj) for obj in objects))
    changes = []
    if variant in ("guard", "all"):
        guard_footprint, change = _guard_variant(objects, known, all_tiles, reserved)
        reserved |= guard_footprint
        changes.append(change)
    # Keep water/rock beyond the immediate known border to avoid editing a
    # neighbouring coastal tile's currently visible interpretation. The actual
    # engine can still have initialization effects; later prefix equality is required.
    hidden_clear = {pos for pos in all_tiles - known - reserved
                    if all(_distance(pos, visible) > 1 for visible in known)}
    if variant in ("terrain", "all"):
        if len(hidden_clear) < len(TERRAIN_TOKENS):
            raise ValueError("Insufficient unknown, object-free terrain with a one-tile known-border margin")
        for (name, token), position in zip(TERRAIN_TOKENS, _ordered(hidden_clear)):
            x, y, _ = position
            before = terrain[y][x]
            if not DIRT_TOKEN.fullmatch(before):
                raise ValueError("Unexpected authored terrain token")
            terrain[y][x] = token
            hidden_clear.remove(position)
            changes.append({"operation": "terrain", "position": list(position),
                            "before": before, "after": token, "terrain": name})
    if variant in ("roads", "all"):
        road_pair = next(((pos, (pos[0] + 1, pos[1], 0)) for pos in _ordered(hidden_clear)
                          if (pos[0] + 1, pos[1], 0) in hidden_clear), None)
        if road_pair is None:
            raise ValueError("No adjacent unknown object-free dirt tiles can hold the road pair")
        for position, suffix in zip(road_pair, ROAD_SUFFIXES):
            x, y, _ = position
            before = terrain[y][x]
            if not DIRT_TOKEN.fullmatch(before):
                raise ValueError("Road must preserve original dirt terrain")
            terrain[y][x] = before + suffix
            changes.append({"operation": "road", "position": list(position),
                            "before": before, "after": terrain[y][x], "road": "core:dirtRoad"})
    original = scenario_payloads(0)
    tile_changes = {(x, y, 0) for y in range(20) for x in range(20)
                    if terrain[y][x] != original["surface_terrain.json"][y][x]}
    declared_tiles = {tuple(change["position"]) for change in changes if "position" in change}
    if (tile_changes != declared_tiles or tile_changes & known or tile_changes & reserved
            or objects[:len(original_objects)] != original_objects or payloads["header.json"] != original["header.json"]):
        raise ValueError("Fixture change isolation invariant failed")
    changed_entries = {"surface_terrain.json"} if tile_changes else set()
    if len(objects) != len(original_objects):
        changed_entries.add("objects.json")
    output.mkdir(parents=True, exist_ok=False)
    base_path, variant_path = output / "base.vmap", output / "variant.vmap"
    base_path.write_bytes(source_bytes)
    with zipfile.ZipFile(variant_path, "w") as archive:
        for info in infos:
            data = ((json.dumps(payloads[info.filename], indent=2, sort_keys=True) + "\n").encode()
                    if info.filename in changed_entries else contents[info.filename])
            archive.writestr(copy(info), data)
    if base_path.read_bytes() != source_bytes:
        raise ValueError("Base fixture does not exactly match the inspected source bytes")
    manifest = {
        "schema": "heroes-contextual-hidden-fixtures-v1", "variant": variant,
        "scope": "Isolated evaluation fixtures only; never training data, demonstrations or production policy inputs",
        "engine_launched": False, "training_performed": False, "native_validated": False,
        "source_map": {"path": str(baseline_map), "sha256": _sha(source_bytes)},
        "observation_reference": {"path": str(observation_path), "sha256": _sha(frame_bytes),
                                  "schema": frame.schema, "episode": frame.episode, "player": frame.player,
                                  "day": frame.day, "seq": frame.seq,
                                  "known_tiles": [list(pos) for pos in _ordered(known)],
                                  "authenticity": "Caller-supplied; strict decoding does not prove this came from real VCMI"},
        "base": {"path": str(base_path), "sha256": _sha(base_path.read_bytes()), "byte_identical_to_source": True},
        "variant_map": {"path": str(variant_path), "sha256": _sha(variant_path.read_bytes())},
        "changes": changes, "changed_archive_entries": sorted(changed_entries),
        "preserved": {"header_and_victory_rules": True, "all_existing_objects": True,
                      "initial_heroes_towns_and_resource_settings": True,
                      "all_known_terrain_and_road_tokens": True},
        "native_format_basis": {
            "upstream_commit": UPSTREAM,
            "terrain_and_road_tokens": "lib/mapping/MapFormatJson.cpp: readTerrainTile/writeTerrainTile",
            "terrain_identifiers_and_views": "config/terrains.json and config/terrainViewPatterns.json",
            "road_identifier_and_dead_end_views": "config/roads.json and lib/mapping/CDrawRoadsOperation.cpp",
            "visible_footprint": "lib/mapObjects/ObjectTemplate.cpp and CGameState::isVisibleFor(object, player)",
        },
        "required_native_checks": [
            "Load both fixtures with the same engine build, settings and seed, using separate bounded local episodes",
            "Verify the base reproduces the supplied permitted initial observation; normalize only bookkeeping identity",
            "Require exact player-visible prefix, current observation and offered support equality between fixtures before comparing routes",
            "For guard variants, independently verify the appended guard remains invisible to the selected player at comparison time",
            "Issue the same destination action in both fixtures and compare ordered visible results until a legitimate new observation differs",
            "Do not demand equality after movement reveals changed terrain, roads or guard; report that visibility boundary explicitly",
        ],
        "limitations": [
            "Static unknown-coordinate checks do not prove native initial fog, event completeness or pathfinder isolation",
            "Initialization, coastal interpretation or object registration can differ; a prefix mismatch invalidates this comparison",
            "Terrain transition artwork is not retiled; tokens use valid native views, and these fixtures are for structured headless checks",
            "Known-border guard targets and every exact hidden modification are validation metadata, never learner observations",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observation", type=Path, required=True, help="Real day-1 initial map view/decision as a complete history-v2 JSON frame")
    parser.add_argument("--baseline-map", type=Path, default=BASELINE, help="Unchanged authored random-baseline-v1 seed-0 map")
    parser.add_argument("--variant", choices=VARIANTS, default="all")
    parser.add_argument("--output", type=Path, required=True, help="New isolated fixture directory, normally under runs/")
    args = parser.parse_args(argv)
    result = build_fixtures(args.observation, args.output, baseline_map=args.baseline_map, variant=args.variant)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"), "base": result["base"],
                      "variant_map": result["variant_map"], "native_validated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
