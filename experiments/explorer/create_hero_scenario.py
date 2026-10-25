"""Versioned ordinary scenario permitting native Tavern construction.

The restricted baseline is immutable. This variant changes only its title,
description and both towns' Tavern prohibition; it grants no building, hero,
resource or army. No game or learning process is launched by this generator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from experiments.explorer.create_scenario import scenario_payloads

SCENARIO_ID = "hero-management-v1"
ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"
BASELINE_SHA256 = "6fc3f50ac769ed6030a21e6ed7c4ba3b0a1341b3d16e9ec72e8a330ba91db58d"


def hero_scenario_payloads(seed=0):
    original = scenario_payloads(seed)
    result = deepcopy(original)
    result["header.json"]["name"] = "Hero management: two castles"
    result["header.json"]["description"] = (
        "Versioned baseline with native Tavern construction permitted for both sides. "
        "Initial buildings, heroes, armies, resources, terrain, fog, victory rules "
        "and level cap 1 are unchanged. Hero limits are native engine settings."
    )
    towns = [obj for obj in result["objects.json"] if obj["type"] == "town"]
    if len(towns) != 2:
        raise ValueError("Expected both unchanged baseline towns")
    for town in towns:
        if town["options"]["buildings"]["noneOf"] != ["core:tavern"]:
            raise ValueError("Baseline Tavern prohibition differs")
        town["options"]["buildings"]["noneOf"] = []
    return result


def verify_hero_scenario(path, seed=0):
    if hashlib.sha256(BASELINE.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("Restricted baseline bytes changed")
    expected = hero_scenario_payloads(seed)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Invalid scenario archive")
        actual = {name: json.loads(archive.read(name)) for name in archive.namelist()}
    # Canonical JSON comparison distinguishes booleans from integers.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(actual) != canonical(expected):
        raise ValueError("Scenario differs from the exact declared variant")
    return actual


def create_hero_scenario(path, seed=0):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Do not overwrite an existing scenario")
    original, payloads = scenario_payloads(seed), hero_scenario_payloads(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, json.dumps(data, indent=2, sort_keys=True) + "\n")
    verify_hero_scenario(path, seed)
    return {
        "schema": "heroes-hero-management-scenario-v1", "scenario_id": SCENARIO_ID,
        "layout_seed": seed, "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_sha256": BASELINE_SHA256,
        "changes": [
            {"entry": "header.json", "field": name, "before": original["header.json"][name],
             "after": payloads["header.json"][name]} for name in ("name", "description")
        ] + [{"entry": "objects.json", "instance": obj["instanceName"],
              "field": "options.buildings.noneOf", "before": ["core:tavern"], "after": []}
             for obj in payloads["objects.json"] if obj["type"] == "town"],
        "preserved": {"level_cap": 1, "initial_buildings_and_armies": True,
                      "initial_heroes": True, "terrain_and_objects": True,
                      "resources": True, "fog": True, "victory_and_defeat_rules": True,
                      "automatic_battles": True, "terminal_only_rewards": True},
        "hero_limits": "Use effective native heroesPerPlayerOnMapCap and heroesPerPlayerTotalCap settings; no scenario override",
        "tavern": "Construction is permitted, not prebuilt. All hiring is an explicit policy action at native cost.",
        "engine_verified": False, "training_performed": False,
        "note": "Generator metadata only; source-pinned native validation is recorded separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    args = parser.parse_args()
    result = create_hero_scenario(args.output, args.seed)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"map": str(args.output.resolve()), "sha256": result["map_sha256"]}))


if __name__ == "__main__":
    main()
