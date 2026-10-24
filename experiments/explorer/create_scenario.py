#!/usr/bin/env python3
"""Create the authored, fixed 20x20 scenario for the first random-policy run.

Only JSON and references to externally installed game assets enter the archive.
The layout is fixed; ``seed`` selects cosmetic dirt tile variants, not a policy.
Engine startup/episode evidence belongs in validation/, separately from this
generator's reproducibility manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile


SIZE = 20
SCENARIO_ID = "random-baseline-v1"
TOWN_MASK = ["VVVVVV", "VVVVVV", "VVVVVV", "VVBBBV", "VBBBBB", "VBBABB"]


def _object(name: str, kind: str, subtype: str, anchor: tuple[int, int],
            animation: str, mask: list[str], options: dict,
            *, from_above: bool = True) -> dict:
    return {
        "instanceName": name,
        "type": kind,
        "subtype": subtype,
        "x": anchor[0], "y": anchor[1], "l": 0,
        "template": {
            "animation": animation,
            "mask": mask,
            "visitableFrom": ["+++" if from_above else "---", "+-+", "+++"],
        },
        "options": options,
    }


def scenario_payloads(seed: int = 0) -> dict:
    """Return VCMI version-3 map entries without opening proprietary assets."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    starts = {"red": (4, 7, "orrin"), "blue": (15, 7, "edric")}
    players = {}
    objects = []
    for index, (color, (x, y, hero_type)) in enumerate(starts.items()):
        hero_name, town_name = f"hero_{index}", f"town_{index}"
        players[color] = {
            "canPlay": "PlayerOrAI",
            "mainHero": hero_name,
            "heroes": {hero_name: {"type": f"core:{hero_type}"}},
            # VCMI stores the town's visitable tile here, not its sprite anchor.
            "mainTown": {"x": x, "y": y, "l": 0, "generateHero": False},
            "allowedFactions": {"anyOf": ["core:castle"]},
        }
        objects.append(_object(
            town_name, "town", "castle", (x + 2, y), "AVCcasx0.def",
            TOWN_MASK,
            {
                "owner": color,
                "buildings": {
                    "allOf": ["core:fort", "core:dwellingLvl1"],
                    # No tavern/recruiting additional heroes in this one-hero
                    # interface. This restriction applies equally to both sides.
                    "noneOf": ["core:tavern"],
                },
                "army": [],
                "possibleSpells": [], "obligatorySpells": [],
            },
            from_above=False,
        ))
        objects.append(_object(
            hero_name, "hero", "knight", (x + 1, y), "AH00_",
            ["VVV", "VAV"],
            {
                "owner": color, "type": f"core:{hero_type}", "experience": 0,
                "primarySkills": {"attack": 1, "defence": 1, "spellpower": 1, "knowledge": 1},
                "secondarySkills": [], "spellBook": [], "artifacts": {},
                "army": [{"type": "core:pikeman", "amount": 25}],
                "formation": "loose",
            },
        ))

    for name, kind, subtype, tile, animation, mask, options, from_above in [
        # Omitted owner is VCMI's neutral default. The literal "neutral" is
        # not a registered playerColor identifier in the v3 map loader.
        ("mine_0", "mine", "sawmill", (5, 16), "AVMsawd0.def",
         ["VVVVV", "VVVBB", "VBBAB"], {"army": []}, False),
        ("mine_1", "mine", "sawmill", (14, 16), "AVMsawd0.def",
         ["VVVVV", "VVVBB", "VBBAB"], {"army": []}, False),
        ("guard_0", "monster", "pikeman", (6, 12), "AvWPike.def",
         ["VV", "VA"], {"amount": 6, "character": "savage", "neverFlees": True, "noGrowing": True}, True),
        ("guard_1", "monster", "pikeman", (13, 12), "AvWPike.def",
         ["VV", "VA"], {"amount": 6, "character": "savage", "neverFlees": True, "noGrowing": True}, True),
        ("gold_0", "resource", "gold", (4, 10), "AVTgold0.def",
         ["VA"], {"amount": 500, "army": []}, True),
        ("gold_1", "resource", "gold", (15, 10), "AVTgold0.def",
         ["VA"], {"amount": 500, "army": []}, True),
    ]:
        # In the sawmill footprint A is one column left of its anchor. The
        # resource/monster footprints put A at the anchor itself.
        anchor = (tile[0] + 1, tile[1]) if kind == "mine" else tile
        objects.append(_object(name, kind, subtype, anchor, animation, mask,
                               options, from_above=from_above))

    header = {
        "versionMajor": 3, "versionMinor": 0,
        "name": "Random baseline: two castles",
        "description": (
            "Authored movement baseline: equal Castle starts, one hero each, "
            "two sawmills, two six-pikeman neutral stacks and two gold piles. "
            "Normal fog and victory rules; level cap 1; no taverns."
        ),
        "difficulty": "NORMAL", "heroLevelLimit": 1,
        "mapLevels": {"surface": {"width": SIZE, "height": SIZE, "index": 0}},
        "mods": [], "players": players,
        "allowedHeroes": {}, "allowedAbilities": {},
        "allowedArtifacts": {}, "allowedSpells": {},
        "victoryIconIndex": 11, "defeatIconIndex": 3,
        "victoryMessage": "Defeat all opponents.",
        "defeatMessage": "Lose all heroes and towns, or spend seven days without a town.",
        "triggeredEvents": {
            "standardVictory": {
                "condition": ["standardWin"],
                "effect": {"type": "victory", "messageToSend": "Your opponent won."},
                "message": "All opponents have been defeated.",
            },
            "standardDefeat": {
                "condition": ["daysWithoutTown", {"value": 7}],
                "effect": {"type": "defeat", "messageToSend": "Your opponent lost their last town."},
                "message": "Seven days without a town have passed.",
            },
        },
    }
    return {
        "header.json": header,
        # VCMI version 3 uses an array; every object needs its own instanceName.
        "objects.json": objects,
        "surface_terrain.json": [[f"dt{rng.randrange(21, 44)}_" for _ in range(SIZE)] for _ in range(SIZE)],
    }


def create_scenario(path: Path, seed: int = 0) -> dict:
    path = Path(path)
    payloads = scenario_payloads(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {
        "schema_version": 1, "scenario_id": SCENARIO_ID,
        "layout_seed": seed, "seed_effect": "cosmetic dirt variants only; engine seed is supplied separately",
        "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": [SIZE, SIZE, 1],
        "objects": {"town": 2, "hero": 2, "mine": 2, "monster": 2, "resource": 2},
        "starts": {color: {"visitable": [x, 7, 0], "hero_anchor": [x + 1, 7, 0],
                            "town_anchor": [x + 2, 7, 0], "hero": hero}
                   for color, x, hero in (("red", 4, "orrin"), ("blue", 15, "edric"))},
        "army_each": [{"type": "core:pikeman", "amount": 25}],
        "targets": {
            "mine_0": [5, 16, 0], "mine_1": [14, 16, 0],
            "guard_0": [6, 12, 0], "guard_1": [13, 12, 0],
            "gold_0": [4, 10, 0], "gold_1": [15, 10, 0],
        },
        "rules": {
            "fog": "normal engine team fog; no reveal events",
            "symmetry": "equal starting armies and primary stats; approximate geometric symmetry does not promise equal initial visibility",
            "hero_level_limit": 1,
            "hero_specialties": "Orrin: archery; Edric: griffins; neither applies to the starting pikemen",
            "additional_heroes": "taverns forbidden for both sides; no other hero sources",
            "towns": "fort and first-tier dwelling; other construction is not scripted or automated for the agent",
            "neutral_stacks": "six pikemen each, savage, never flee, no growth; no reward artifact/resources",
            "victory": "standard engine elimination of opponents",
            "defeat": "standard loss of all heroes and towns, or seven days without a town",
            "reward_shaping": False,
        },
        "engine_verified": False,
        "verification_note": "Generator metadata only; consult committed run validation for engine loading and gameplay.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "maps" / f"{SCENARIO_ID}.vmap")
    args = parser.parse_args()
    metadata = create_scenario(args.output, args.seed)
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"map": str(args.output.resolve()), "manifest": str(manifest.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
