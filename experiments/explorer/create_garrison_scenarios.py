"""Deterministic native garrison-dialog fixtures, excluded from PPO.

Only authored map state changes. No runtime state injection, reward changes,
automatic dialog answers or learned choices are provided by this generator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from experiments.explorer.create_capability_scenarios import COLORS, FULL_ARMY, ordinary_payloads
from experiments.explorer.create_dialog_scenarios import _mask
from experiments.explorer.create_scenario import _object

CASES = ("mine-repeat", "mine-transfer", "garrison-transfer", "antimagic-transfer",
         "vertical-transfer", "garrison-locked", "antimagic-locked", "garrison-full")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_id(case, player):
    if case not in CASES or player not in COLORS:
        raise ValueError("Unknown garrison case/player")
    return f"garrison-{case}-{player}-v1"


def payloads(case, player):
    ident = scenario_id(case, player)
    result = deepcopy(ordinary_payloads())
    header, objects = result["header.json"], result["objects.json"]
    header.update(name=ident, description="Isolated garrison interface diagnostic; never training input. "
                  "Native outcomes, fog, army constraints and automatic battles remain unchanged.")
    objects[:] = [obj for obj in objects if obj["type"] in ("town", "hero")]
    actor = next(obj for obj in objects if obj["instanceName"] == f"hero_{COLORS.index(player)}")
    x, y = (4 if player == "red" else 15), 12
    actor.update(x=x + 1, y=y - 2)
    actor["options"]["army"] = [{"type": "core:pikeman", "amount": 25}]
    options = {"army": [], "owner": player}
    if case.startswith("mine-"):
        if case == "mine-repeat":
            options.pop("owner")
        target = _object("diagnostic_0", "mine", "sawmill", (x + 1, y), "AVMsawd0.def",
                         ["VVVVV", "VVVBB", "VBBAB"], options, from_above=False)
    else:
        options["removableUnits"] = not case.endswith("-locked")
        options["army"] = [{"type": "core:archer", "amount": 3}]
        if case.endswith("-locked"):
            # Different source/destination slots expose native locked-row
            # checks that accidentally inspect the source slot in both armies.
            options["army"].append({"type": "core:pikeman", "amount": 2})
        vertical, anti = case == "vertical-transfer", case.startswith("antimagic-")
        # Canonical HoMM3 object identifiers/footprints; no asset bytes copied.
        template = ("AVCvgr.def", "011111110111111101111111", "0000000010000000") if vertical else (
            "AVCgar20.def" if anti else "AVCgar10.def", "00011111", "01000000")
        target = _object("diagnostic_0", "garrisonVertical" if vertical else "garrisonHorizontal",
                         "antiMagic" if anti else "normal", (x, y + 1) if vertical else (x + 1, y),
                         template[0], _mask(template[1], template[2]), options)
    if case == "garrison-full":
        actor["options"]["army"] = [{"type": "core:" + name, "amount": 10} for name in FULL_ARMY]
        target["options"]["army"] = [{"type": "core:" + name, "amount": 3} for name in FULL_ARMY]
    objects.append(target)
    return result


def verify_scenario(path, case, player):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() or len(names) != len(set(names)):
            raise ValueError("Invalid/duplicate garrison map archive entries")
        actual = {name: json.loads(archive.read(name)) for name in names}
    if canonical(actual) != canonical(payloads(case, player)):
        raise ValueError("Garrison fixture differs from its exact authored declaration")
    return actual


def create_scenario(path, case, player):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve previous garrison fixtures")
    before, after = ordinary_payloads(), payloads(case, player)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in after.items():
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type, entry.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(entry, json.dumps(value, sort_keys=True, indent=2) + "\n")
    verify_scenario(path, case, player)
    old_objects = {obj["instanceName"]: obj for obj in before["objects.json"]}
    new_objects = {obj["instanceName"]: obj for obj in after["objects.json"]}
    return {"schema": "heroes-garrison-fixture-v1", "case": case, "player": player,
            "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parent_payload_sha256": hashlib.sha256(canonical(before).encode()).hexdigest(),
            "changed_header_fields": {key: {"before": before["header.json"].get(key), "after": value}
                                      for key, value in after["header.json"].items()
                                      if value != before["header.json"].get(key)},
            "object_changes": [{"instance": key, "before": old_objects.get(key), "after": new_objects.get(key)}
                               for key in sorted(old_objects.keys() | new_objects.keys())
                               if old_objects.get(key) != new_objects.get(key)],
            "terrain_changes": [], "diagnostic_only": True, "training_input": False,
            "training_performed": False, "optimizer_steps": 0, "engine_verified": False,
            "rules_and_rewards_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--player", choices=(*COLORS, "both"), default="both")
    args = parser.parse_args(argv)
    for case in CASES if args.case == "all" else (args.case,):
        for player in COLORS if args.player == "both" else (args.player,):
            path = args.output / (scenario_id(case, player) + ".vmap")
            manifest = create_scenario(path, case, player)
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
