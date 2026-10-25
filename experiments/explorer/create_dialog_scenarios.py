"""Exact, isolated adventure-dialog diagnostic fixtures; never training maps.

These authored starting positions and objects exercise native controls. Rewards,
join decisions, Scholar rewards and chest contents are resolved by VCMI. This
generator neither reads game assets nor supplies hidden state to a controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from experiments.explorer.create_hero_scenario import BASELINE, BASELINE_SHA256, hero_scenario_payloads
from experiments.explorer.create_scenario import _object

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "experiments/explorer/maps"
HERO_MAP = MAPS / "hero-management-v1.vmap"
HERO_SHA256 = "852ea0211d20e1cbfb64790fe8076254e98983ba0ae29e854db192c816c76ad1"
FIXTURES = ("guarded_mine", "abandoned_mine", "neutral_free", "neutral_flee", "neutral_paid",
            "chest", "arena", "school_magic", "school_war", "scholar", "scholar_exchange", "neutral_free_attack",
            "neutral_paid_unaffordable")
COLORS = ("red", "blue")


def scenario_id(fixture, player):
    if fixture not in FIXTURES or player not in COLORS:
        raise ValueError("Unknown declared diagnostic fixture/player")
    return f"dialog-{fixture.replace('_', '-')}-{player}-v1"


def _mask(blocked, visit):
    """Native 8x6 object-mask bits, bottom/right first, converted without edits."""
    blocked, visit = blocked.ljust(48, "1"), visit.ljust(48, "0")
    return ["".join("A" if visit[y * 8 + x] == "1" else
                    "B" if blocked[y * 8 + x] == "0" else "V"
                    for x in reversed(range(8))) for y in reversed(range(6))]


def _rewardable(name, kind, tile):
    # Core object template identifiers and collision masks, not asset bytes.
    native = {
        "treasureChest": ("AVTchst0.def", "01111111", "10000000", 0),
        "scholar": ("AVXschl0.def", "01111111", "10000000", 0),
        "schoolOfMagic": ("AVSschm0.def", "01111111", "10000000", 0),
        "arena": ("AVSarna0.def", "0001111100011111", "01000000", 1),
        "schoolOfWar": ("AVSwar20.def", "100111110000111110011111", "01000000", 1),
    }
    animation, block, visit, offset = native[kind]
    return _object(name, kind, kind, (tile[0] + offset, tile[1]), animation,
                   _mask(block, visit), {})


def _native_hero(obj, hero_type, hero_class):
    obj["subtype"] = hero_class
    obj["options"]["type"] = "core:" + hero_type
    # Absence means the real hero type's native skills and starting spellbook.
    # No explicit skill/spell or artifact is granted by the fixture.
    for field in ("primarySkills", "secondarySkills", "spellBook", "artifacts"):
        obj["options"].pop(field, None)


def dialog_scenario_payloads(fixture, player, seed=0):
    ident = scenario_id(fixture, player)
    payloads = deepcopy(hero_scenario_payloads(seed))
    header, objects = payloads["header.json"], payloads["objects.json"]
    header["name"] = ident
    header["description"] = (
        "Isolated native adventure-dialog diagnostic; excluded from training. "
        "Authored initial objects/armies only. Normal fog, native random rewards and "
        "creature behavior, automatic battles, standard outcomes and level cap 1."
    )
    # Remove unrelated baseline obstacles/resources, with the exact delta listed
    # in the manifest. Both original towns and the other side's hero are kept.
    objects[:] = [obj for obj in objects if obj["type"] in ("hero", "town")]
    index = COLORS.index(player)
    hero = next(obj for obj in objects if obj["instanceName"] == f"hero_{index}")
    x = 4 if player == "red" else 15
    if fixture in ("neutral_paid", "neutral_paid_unaffordable", "scholar"):
        hero_type, hero_class = ("thane", "alchemist") if fixture == "scholar" else ("adela", "cleric")
        _native_hero(hero, hero_type, hero_class)
        header["players"][player]["heroes"][hero["instanceName"]]["type"] = "core:" + hero_type
    if fixture in ("neutral_flee", "abandoned_mine"):
        hero["options"]["army"] = [{"type": "core:pikeman", "amount": 250}]
    if fixture == "neutral_paid_unaffordable":
        hero["options"]["army"] = [{"type": "core:pikeman", "amount": 1500}]
    if fixture == "scholar_exchange":
        _native_hero(hero, "thane", "alchemist")
        partner = deepcopy(hero)
        partner.update(instanceName="diagnostic_partner", x=x + 1, y=10)
        _native_hero(partner, "josephine", "alchemist")
        objects.append(partner)
        header["players"][player]["heroes"] = {
            hero["instanceName"]: {"type": "core:thane"}, "diagnostic_partner": {"type": "core:josephine"}}
        return payloads
    if fixture in ("guarded_mine", "abandoned_mine"):
        abandoned = fixture == "abandoned_mine"
        objects.append(_object("diagnostic_target_0", "mine", "abandoned" if abandoned else "sawmill",
            (x + 1, 10), "AVXabnd0.def" if abandoned else "AVMsawd0.def",
            _mask("00011111" if abandoned else "0000111100111111", "01000000"),
            {"possibleResources": ["core:wood", "core:mercury", "core:ore", "core:sulfur", "core:crystal", "core:gems", "core:gold"]}
            if abandoned else {"army": [{"type": "core:pikeman", "amount": 6}]}, from_above=False))
        return payloads
    if fixture.startswith("neutral_"):
        character = {"neutral_free": "compliant", "neutral_free_attack": "compliant",
                     "neutral_flee": "hostile", "neutral_paid": "friendly", "neutral_paid_unaffordable": "friendly"}[fixture]
        objects.append(_object("diagnostic_target_0", "monster", "pikeman", (x, 10), "AvWPike.def",
            ["VV", "VA"], {"amount": 180 if fixture == "neutral_paid_unaffordable" else 6, "character": character,
                            "neverFlees": fixture == "neutral_free_attack", "noGrowing": True}))
        return payloads
    kind = {"chest": "treasureChest", "arena": "arena", "school_magic": "schoolOfMagic",
            "school_war": "schoolOfWar", "scholar": "scholar"}[fixture]
    # Schools have nine independently visitable objects so legitimate payments
    # can exhaust native starting income. No resource override/event is used.
    tiles = [(column, row) for row in (11, 15, 19) for column in (2, 7, 12)]
    if fixture in ("chest", "scholar"):
        columns = (2, 4, 6) if player == "red" else (13, 15, 17)
        tiles = [(column, row) for row in (9, 11, 13) for column in columns]
    if fixture == "arena":
        tiles = [(x, 10), (x, 14)]
    for number, tile in enumerate(tiles):
        objects.append(_rewardable(f"diagnostic_target_{number}", kind, tile))
    return payloads


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_dialog_scenario(path, fixture, player, seed=0):
    for original, expected in ((BASELINE, BASELINE_SHA256), (HERO_MAP, HERO_SHA256)):
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise ValueError("An immutable earlier scenario changed")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Invalid diagnostic archive")
        actual = {name: json.loads(archive.read(name)) for name in archive.namelist()}
    if _canonical(actual) != _canonical(dialog_scenario_payloads(fixture, player, seed)):
        raise ValueError("Map differs from the exact declared diagnostic fixture")
    return actual


def create_dialog_scenario(path, fixture, player, seed=0):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing diagnostic maps")
    payloads = dialog_scenario_payloads(fixture, player, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type, entry.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(entry, json.dumps(value, indent=2, sort_keys=True) + "\n")
    verify_dialog_scenario(path, fixture, player, seed)
    original = hero_scenario_payloads(seed)
    before = {obj["instanceName"]: obj for obj in original["objects.json"]}
    after = {obj["instanceName"]: obj for obj in payloads["objects.json"]}
    return {"schema": "heroes-dialog-diagnostic-scenario-v1", "scenario_id": scenario_id(fixture, player),
            "fixture": fixture, "player": player, "layout_seed": seed,
            "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "baseline_sha256": BASELINE_SHA256, "hero_management_sha256": HERO_SHA256,
            "changed_header_fields": {key: {"before": original["header.json"][key], "after": value}
                                      for key, value in payloads["header.json"].items()
                                      if value != original["header.json"][key]},
            "object_changes": [{"instance": key, "before": before.get(key), "after": after.get(key)}
                               for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)],
            "terrain_unchanged": True, "level_cap": 1, "fog": "normal native team fog",
            "rules_and_rewards_unchanged": True, "runtime_resource_or_army_grants": False,
            "native_reward_and_join_randomness": True, "variable_presets": False,
            "training_input": False, "training_performed": False, "engine_verified": False,
            "note": "Exact generator identity only. Native loading, observed branches and replay require separate validation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=(*FIXTURES, "all"), default="all")
    parser.add_argument("--player", choices=(*COLORS, "both"), default="both")
    parser.add_argument("--output-directory", type=Path, default=MAPS)
    args = parser.parse_args(argv)
    for fixture in FIXTURES if args.fixture == "all" else (args.fixture,):
        for player in COLORS if args.player == "both" else (args.player,):
            path = args.output_directory / (scenario_id(fixture, player) + ".vmap")
            manifest = create_dialog_scenario(path, fixture, player)
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"path": str(path), "sha256": manifest["map_sha256"]}))


if __name__ == "__main__":
    main()
