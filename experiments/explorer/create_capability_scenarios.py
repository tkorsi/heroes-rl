"""Versioned level-enabled scenario and isolated v8 capability fixtures.

The ordinary v2 map changes only its title, description and native level limit.
Diagnostic maps additionally declare authored starting equipment, armies and
objects. They are excluded from training; no controller, reward or engine rule
is installed by this generator. Real loading and branch coverage are separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from experiments.explorer.create_hero_scenario import hero_scenario_payloads
from experiments.explorer.create_dialog_scenarios import HERO_MAP, HERO_SHA256, _rewardable
from experiments.explorer.create_scenario import _object

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "experiments/explorer/maps"
ORDINARY_ID = "hero-management-v2"
COLORS = ("red", "blue")
FIXTURES = ("town_controls", "stables", "neutral_overflow", "artifacts",
            "adventure_spells", "level_up", "eagle_eye")
FULL_ARMY = ("halberdier", "archer", "marksman", "griffin", "royalGriffin", "swordsman", "crusader")
COMBINATION = {
    "head": "helmOfHeavenlyEnlightenment", "neck": "celestialNecklaceOfBliss",
    "rightHand": "swordOfJudgement", "leftHand": "lionsShieldOfCourage",
    "torso": "armorOfWonder", "feet": "sandalsOfTheSaint",
}
ADVENTURE_SPELLS = ("summonBoat", "scuttleBoat", "visions", "viewEarth", "viewAir",
                    "disguise", "fly", "waterWalk", "dimensionDoor", "townPortal")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_id(fixture=None, player=None):
    if fixture is None and player is None:
        return ORDINARY_ID
    if fixture not in FIXTURES or player not in COLORS:
        raise ValueError("Unknown capability fixture/player")
    return f"capability-{fixture.replace('_', '-')}-{player}-v1"


def ordinary_payloads(seed=0):
    result = deepcopy(hero_scenario_payloads(seed))
    header = result["header.json"]
    header.update(name="Hero management: levels enabled", heroLevelLimit=0,
                  description="Versioned two-Castle scenario with normal native hero leveling. "
                  "Only the previous level-1 cap is removed; starting heroes, buildings, "
                  "armies, resources, terrain, objects, fog and outcomes are unchanged.")
    return result


def capability_payloads(fixture, player, seed=0):
    ident = scenario_id(fixture, player)
    result = ordinary_payloads(seed)
    header, objects = result["header.json"], result["objects.json"]
    header.update(name=ident, description="Isolated native capability diagnostic; excluded from training. "
                  "All authored initial-state differences are recorded in its manifest. "
                  "Normal fog, VCMI mechanics, automatic battles and native leveling remain enabled.")
    objects[:] = [obj for obj in objects if obj["type"] in ("town", "hero")]
    actor = next(obj for obj in objects if obj["instanceName"] == f"hero_{COLORS.index(player)}")
    town = next(obj for obj in objects if obj["instanceName"] == f"town_{COLORS.index(player)}")
    x = 4 if player == "red" else 15
    if fixture == "town_controls":
        town["options"]["buildings"]["allOf"].extend(
            ["core:marketplace", "core:mageGuild1", "core:blacksmith"])
        # Native initial resources and the visitor's absent book/war machines
        # are deliberate: purchases must deduct their real displayed costs.
    elif fixture == "stables":
        town["options"]["buildings"]["allOf"].extend(
            ["core:blacksmith", "core:dwellingLvl4", "core:special2"])
        # Enter the town through an actual movement action, not startup effects.
        actor.update(x=x + 1, y=9)
    elif fixture == "neutral_overflow":
        actor["options"]["army"] = [{"type": "core:" + name, "amount": 10} for name in FULL_ARMY]
        objects.append(_object("diagnostic_neutral", "monster", "pikeman", (x, 10),
                               "AvWPike.def", ["VV", "VA"],
                               {"amount": 6, "character": "compliant", "neverFlees": True, "noGrowing": True}))
    elif fixture == "artifacts":
        actor["options"]["artifacts"] = {slot: "core:" + name for slot, name in COMBINATION.items()}
        actor["options"]["artifacts"]["backpack"] = ["core:centaurAxe"]
        partner = deepcopy(actor)
        partner.update(instanceName="diagnostic_partner", x=x + 1, y=10)
        partner["options"].update(type="core:valeska", artifacts={})
        objects.append(partner)
        header["players"][player]["heroes"]["diagnostic_partner"] = {"type": "core:valeska"}
    elif fixture == "adventure_spells":
        actor.update(x=x + 1, y=10)
        actor["options"].update(
            artifacts={"spellbook": "core:spellBook"},
            spellBook=["core:" + name for name in ADVENTURE_SPELLS],
            secondarySkills=[{"skill": "core:" + name, "level": "expert"}
                             for name in ("wisdom", "airMagic", "earthMagic", "waterMagic", "fireMagic")])
        actor["options"]["primarySkills"]["knowledge"] = 20
        # A visible shore and boat exercise native water-target affordances.
        # No water terrain is introduced into the ordinary training scenario.
        for row in (11, 12, 13):
            for column in range(max(0, x - 2), min(20, x + 3)):
                result["surface_terrain.json"][row][column] = "wt0_"
        objects.append(_object("diagnostic_boat", "boat", "boatCastle", (x + 1, 12),
                               "AB02_.def", ["VVV", "VAV"], {}))
    elif fixture == "level_up":
        actor["options"]["experience"] = 999
        # Chest contents remain native random rewards; every predeclared seed
        # is retained even when none of its chests offers an experience choice.
        columns = (2, 4, 6) if player == "red" else (13, 15, 17)
        for number, tile in enumerate((column, row) for row in (9, 11, 13) for column in columns):
            objects.append(_rewardable(f"diagnostic_chest_{number}", "treasureChest", tile))
    elif fixture == "eagle_eye":
        actor.update(x=x + 1, y=10)
        actor["options"].update(artifacts={"spellbook": "core:spellBook"}, spellBook=[],
                                secondarySkills=[{"skill": "core:eagleEye", "level": "expert"}],
                                army=[{"type": "core:pikeman", "amount": 200}])
        enemy = next(obj for obj in objects if obj["type"] == "hero" and obj["options"]["owner"] != player)
        enemy.update(x=x + (3 if player == "red" else -1), y=10)
        enemy["options"].update(artifacts={"spellbook": "core:spellBook"}, spellBook=["core:magicArrow"],
                                army=[{"type": "core:pikeman", "amount": 20}])
        enemy["options"]["primarySkills"]["knowledge"] = 10
        # Whether BattleAI casts and Eagle Eye learns is native chance, never
        # guaranteed or overridden. A missing notice is incomplete coverage.
    return result


def verify_scenario(path, *, fixture=None, player=None, seed=0):
    if hashlib.sha256(HERO_MAP.read_bytes()).hexdigest() != HERO_SHA256:
        raise ValueError("Immutable hero-management-v1 bytes changed")
    expected = ordinary_payloads(seed) if fixture is None else capability_payloads(fixture, player, seed)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() or len(names) != len(set(names)):
            raise ValueError("Invalid or duplicate scenario archive entries")
        actual = {name: json.loads(archive.read(name)) for name in names}
    if canonical(actual) != canonical(expected):
        raise ValueError("Scenario differs from the exact declared capability variant")
    return actual


def create_scenario(path, *, fixture=None, player=None, seed=0):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing capability scenarios")
    ident = scenario_id(fixture, player)
    original = hero_scenario_payloads(seed)
    payloads = ordinary_payloads(seed) if fixture is None else capability_payloads(fixture, player, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type, entry.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(entry, json.dumps(value, indent=2, sort_keys=True) + "\n")
    verify_scenario(path, fixture=fixture, player=player, seed=seed)
    before = {obj["instanceName"]: obj for obj in original["objects.json"]}
    after = {obj["instanceName"]: obj for obj in payloads["objects.json"]}
    terrain_changes = [{"x": x, "y": y, "before": old, "after": payloads["surface_terrain.json"][y][x]}
                       for y, row in enumerate(original["surface_terrain.json"]) for x, old in enumerate(row)
                       if old != payloads["surface_terrain.json"][y][x]]
    return {"schema": "heroes-capability-scenario-v1", "scenario_id": ident,
            "fixture": fixture, "player": player, "layout_seed": seed,
            "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parent_sha256": HERO_SHA256,
            "changed_header_fields": {name: {"before": original["header.json"][name], "after": value}
                                      for name, value in payloads["header.json"].items()
                                      if value != original["header.json"][name]},
            "object_changes": [{"instance": name, "before": before.get(name), "after": after.get(name)}
                               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)],
            "terrain_changes": terrain_changes, "native_hero_level_limit": 0,
            "fog": "normal native team fog", "runtime_state_grants": False,
            "rules_and_rewards_unchanged": True, "automatic_battles": True,
            "diagnostic_only": fixture is not None, "training_performed": False,
            "engine_verified": False,
            "note": "Generator identity only. Native loading, triggered branches and independent replay need separate evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=(*FIXTURES, "ordinary", "all"), default="all")
    parser.add_argument("--player", choices=(*COLORS, "both"), default="both")
    parser.add_argument("--output-directory", type=Path, default=MAPS)
    args = parser.parse_args(argv)
    choices = [None, *FIXTURES] if args.fixture == "all" else [None if args.fixture == "ordinary" else args.fixture]
    for fixture in choices:
        for player in [None] if fixture is None else COLORS if args.player == "both" else [args.player]:
            path = args.output_directory / (scenario_id(fixture, player) + ".vmap")
            manifest = create_scenario(path, fixture=fixture, player=player)
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"map": str(path), "sha256": manifest["map_sha256"]}))


if __name__ == "__main__":
    main()
