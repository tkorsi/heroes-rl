#!/usr/bin/env python3
"""Create a deterministic, real VCMI exploration duel map using stdlib only.

The experiment controls adventure movement; VCMI resolves the eventual battle.
There are no towns, pickups, economic actions, spells or scripted exploration.
The standard seven-day town-loss event is deliberately absent. Losing the only
hero still causes standard elimination, and the remaining player wins.

Map object coordinates are NOT movement coordinates: a hero's visitable tile is
one column left of its object anchor. Metadata below exposes both explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile


def _hero(owner: str, hero_type: str, tile: tuple[int, int]) -> dict:
    x, y = tile
    return {
        "type": "hero",
        "subtype": "knight",
        "x": x + 1,
        "y": y,
        "l": 0,
        "template": {
            "animation": "AH00_",
            "editorAnimation": "AH00_E",
            "mask": ["VV", "AV"],
            "visitableFrom": ["+++", "+-+", "+++"],
        },
        "options": {
            "owner": owner,
            "type": f"core:{hero_type}",
            "experience": 0,
            "primarySkills": {
                "attack": 1, "defence": 1, "spellpower": 1, "knowledge": 1,
            },
            "secondarySkills": [],
            "spellBook": [],
            "artifacts": {},
            "army": [{"type": "core:pikeman", "amount": 25}],
            "formation": "loose",
        },
    }


def create_map(path: Path, seed: int, size: int = 12) -> dict:
    """Write a .vmap at ``path`` and return reproducibility metadata.

    Initial positions vary with seed and have Chebyshev separation >= 6, to
    avoid immediate adjacency. The exact fog visibility remains engine-owned.
    Identical seed and size produce byte-identical ZIP contents and metadata.
    """
    path = Path(path)
    if not 8 <= size <= 72:
        raise ValueError("size must be between 8 and 72")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    # Keep the hero anchor and its 2x2 visual template within map bounds.
    tiles = [(x, y) for y in range(1, size) for x in range(size - 1)]
    minimum_separation = max(6, size // 2)
    start = rng.choice(tiles)
    alternatives = [
        tile for tile in tiles
        if max(abs(tile[0] - start[0]), abs(tile[1] - start[1])) >= minimum_separation
    ]
    # A central start on a tiny map can have no sufficiently distant partner.
    while not alternatives:
        start = rng.choice(tiles)
        alternatives = [
            tile for tile in tiles
            if max(abs(tile[0] - start[0]), abs(tile[1] - start[1])) >= minimum_separation
        ]
    starts = [start, rng.choice(alternatives)]
    rng.shuffle(starts)
    # Their specialties concern archery/griffins, neither present in this map.
    # Override both heroes' normal starting skills and armies.
    hero_types = ["orrin", "edric"]
    rng.shuffle(hero_types)
    colors = ("red", "blue")
    players = {
        color: {
            "canPlay": "PlayerOrAI",
            "mainHero": f"hero_{color}",
            "heroes": {f"hero_{color}": {"type": f"core:{hero_type}"}},
            "allowedFactions": {"anyOf": ["core:castle"]},
        }
        for color, hero_type in zip(colors, hero_types)
    }
    header = {
        "versionMajor": 3,
        "versionMinor": 0,
        "name": f"Explorer duel {size} seed {seed}",
        "description": (
            "Two equal armies explore under normal fog. No towns or economic "
            "objects. Defeat the opposing hero. No seven-day town-loss rule."
        ),
        "difficulty": "NORMAL",
        "heroLevelLimit": 1,
        "mapLevels": {"surface": {"width": size, "height": size, "index": 0}},
        "mods": [],
        "players": players,
        "allowedHeroes": {},
        "allowedAbilities": {},
        "allowedArtifacts": {},
        "allowedSpells": {},
        "victoryIconIndex": 11,
        "defeatIconIndex": 3,
        "triggeredEvents": {
            "standardVictory": {
                "condition": ["standardWin"],
                "effect": {"type": "victory"},
                "message": "The opposing hero has been defeated.",
            },
        },
    }
    objects = {
        f"hero_{color}": _hero(color, hero_type, tile)
        for color, hero_type, tile in zip(colors, hero_types, starts)
    }
    # All tiles are the same passable dirt terrain: visual variants have no
    # effect on movement cost. No blocked tiles, water or hidden stat changes.
    terrain = [[f"dt{rng.randrange(21, 44)}_" for _ in range(size)] for _ in range(size)]
    payloads = {
        "header.json": header,
        "objects.json": objects,
        "surface_terrain.json": terrain,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in payloads.items():
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {
        "schema": 1,
        "seed": seed,
        "size": size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "starts": {
            color: {
                "hero": hero_type,
                "visitable": [tile[0], tile[1], 0],
                "anchor": [tile[0] + 1, tile[1], 0],
            }
            for color, hero_type, tile in zip(colors, hero_types, starts)
        },
        "army_each": [{"type": "core:pikeman", "amount": 25}],
        "terminal_rule": "last surviving hero wins; no seven-day town-loss event",
        "engine_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.output or Path(__file__).parent / "maps" / f"explorer-{args.size}-seed-{args.seed}.vmap"
    metadata = create_map(path, seed=args.seed, size=args.size)
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"map": str(path.resolve()), "manifest": str(manifest.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
