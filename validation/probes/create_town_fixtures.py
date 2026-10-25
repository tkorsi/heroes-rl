#!/usr/bin/env python3
"""Create isolated town-capacity diagnostic maps without running VCMI.

Only one town's initial army changes. These evaluator inputs are excluded from
training, demonstrations and random-baseline outcome evidence. A low-resource
variant is intentionally absent: the pinned JSON map player format has no
initial-resource field; difficulty and scenario handicap supply those resources.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.probes.create_boundary_fixture import (
    BASELINE, UPSTREAM, archive_contents, canonical, decode, sha, source_contents,
)

KINDS = ("capacity-full", "capacity-merge")
PLAYERS = ("red", "blue")
SCHEMA = "heroes-town-capacity-fixture-v1"
# Ordinary Castle definitions at the pinned upstream commit, each explicitly
# listed in config/creatures/castle.json and config/factions/castle.json:/castle/town/creatures.
FULL_CREATURES = ("halberdier", "archer", "marksman", "griffin", "royalGriffin", "swordsman", "crusader")
MERGE_CREATURES = ("pikeman", *FULL_CREATURES[1:])


def variant_payloads(original, kind, player):
    if kind not in KINDS or player not in PLAYERS:
        raise ValueError("Unsupported town fixture kind or player")
    payloads = deepcopy(original)
    name = f"town_{PLAYERS.index(player)}"
    objects = payloads["objects.json"]
    index = next(i for i, obj in enumerate(objects) if obj["instanceName"] == name)
    town = objects[index]
    if town["type"] != "town" or town["subtype"] != "castle" or town["options"]["owner"] != player:
        raise ValueError("Fixture target must be the authored player's Castle town")
    old_army = deepcopy(town["options"]["army"])
    if old_army != []:
        raise ValueError("Authored town must have its unchanged empty army")
    creatures = FULL_CREATURES if kind == "capacity-full" else MERGE_CREATURES
    town["options"]["army"] = [{"type": "core:" + creature, "amount": 1} for creature in creatures]
    changes = [{"entry": "objects.json", "json_pointer": f"/{index}/options/army",
                "instance_name": name, "operation": "replace", "before": old_army,
                "after": deepcopy(town["options"]["army"])}]
    restored = deepcopy(payloads)
    restored["objects.json"][index]["options"]["army"] = old_army
    if canonical(restored) != canonical(original):
        raise ValueError("Fixture modified undeclared fields")
    return payloads, changes


def verify_fixture(path, *, kind, player="red", baseline=BASELINE):
    """Reconstruct the exact allowed delta and check unchanged entry bytes."""
    _, source, original = source_contents(Path(baseline))
    expected, changes = variant_payloads(original, kind, player)
    _, actual = archive_contents(Path(path))
    if set(actual) != set(source):
        raise ValueError("Fixture archive entries differ from the authored baseline")
    changed_entries = sorted({change["entry"] for change in changes})
    for name in source:
        if name not in changed_entries and actual[name] != source[name]:
            raise ValueError("Undeclared archive entry bytes changed: " + name)
        if canonical(decode(actual[name])) != canonical(expected[name]):
            raise ValueError("Fixture fields differ from the exact declared delta: " + name)
    return changed_entries, actual


def create_fixture(output, *, kind, player="red", baseline=BASELINE):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output.exists():
        raise FileExistsError("Fixture output must be a new directory")
    if output == ROOT or ROOT in output.parents and ROOT / "runs" not in output.parents:
        raise ValueError("Keep generated fixtures outside authored sources; use ignored runs/")
    source_bytes = baseline.read_bytes()
    infos, source, original = source_contents(baseline)
    expected, changes = variant_payloads(original, kind, player)
    changed_entries = sorted({change["entry"] for change in changes})
    output.mkdir(parents=True, exist_ok=False)
    base, target = output / "base.vmap", output / f"{kind}.vmap"
    base.write_bytes(source_bytes)
    with zipfile.ZipFile(target, "w") as archive:
        for info in infos:
            content = ((json.dumps(expected[info.filename], indent=2, sort_keys=True) + "\n").encode()
                       if info.filename in changed_entries else source[info.filename])
            archive.writestr(copy(info), content)
    actual_changes, actual = verify_fixture(target, kind=kind, player=player, baseline=baseline)
    if baseline.read_bytes() != source_bytes or base.read_bytes() != source_bytes:
        raise ValueError("Source or preserved base archive bytes changed")
    creatures = FULL_CREATURES if kind == "capacity-full" else MERGE_CREATURES
    manifest = {
        "schema": SCHEMA, "kind": kind, "player": player,
        "scope": "Diagnostic evaluator input only; excluded from training, demonstrations, production policy and random-baseline outcome evidence",
        "engine_launched": False, "native_validated": False, "training_performed": False,
        "training_eligible": False, "random_baseline_outcome_claim": False,
        "upstream_commit": UPSTREAM,
        "source": {"path": str(baseline), "sha256": sha(source_bytes)},
        "base": {"path": str(base), "sha256": sha(base.read_bytes()), "byte_identical_to_source": True},
        "fixture": {"path": str(target), "sha256": sha(target.read_bytes())},
        "changes": changes, "changed_archive_entries": actual_changes,
        "entries": {name: {"source_sha256": sha(source[name]), "fixture_sha256": sha(actual[name]),
                           "byte_identical": source[name] == actual[name]} for name in source},
        "preserved": {"all_undeclared_fields": True, "untouched_archive_entry_bytes": True,
                      "header_bytes": True, "heroes_and_their_armies": True, "hero_count": True,
                      "town_buildings_and_tavern_prohibition": True, "other_town_army": True,
                      "fog_and_victory_rules": True, "terrain_roads": True, "source_archive_bytes": True},
        "native_basis": {
            "creatures": {"core:" + creature: f"config/creatures/castle.json:/{creature}" for creature in creatures},
            "castle_roster": "config/factions/castle.json:/castle/town/creatures",
            "town_army_format": "lib/mapObjects/army/CArmedInstance.cpp:serializeJsonOptions",
            "upper_army": "lib/mapObjects/CGTownInstance.cpp:879-883",
            "slot_selection": "lib/mapObjects/army/CCreatureSet.cpp:73-83,150-159",
            "native_no_room": "client/windows/GUIClasses.cpp:204-258",
            "native_recruitment": "server/CGameHandler.cpp:2465-2559",
        },
        "evaluator_plan": {
            "expected_condition_only": True, "observed_result": None,
            "offered_creature": "core:pikeman", "explicit_count": 1,
            "require_actual_observation": "Own town upper army, selected pikeman offer, positive stock and sufficient resources; no map metadata supplies policy input",
            "expected_capacity": "no_slot" if kind == "capacity-full" else "existing_same_creature_slot",
            "expected_merge_slot": None if kind == "capacity-full" else 0,
            "success_evidence": "Actual native/adapter result, issuance/acknowledgement, exact resources/stock/army deltas and independent history audit; this manifest proves none of them",
        },
        "omitted_low_resource_fixture": {
            "reason": "Pinned JSON map player data has no initial-resource field; do not add an ignored field or change difficulty/handicap",
            "native_basis": ["lib/mapping/MapFormatJson.cpp:423-568", "lib/gameState/CGameState.cpp:407-440"],
        },
    }
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--player", choices=PLAYERS, default="red")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    result = create_fixture(args.output, kind=args.kind, player=args.player, baseline=args.baseline)
    print(json.dumps({"fixture": result["fixture"], "manifest": str(args.output.resolve() / "fixture.json"),
                      "native_validated": False}, indent=2))


if __name__ == "__main__":
    main()
