#!/usr/bin/env python3
"""Make an isolated real-VCMI battle input; never training or baseline evidence.

Only one authored neutral guard's initial count changes. This does not claim
that combat produces casualties: a bounded native run must establish survival,
losses, closed-window absence, and fresh data after reopening the same hero.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explorer.create_scenario import scenario_payloads

BASELINE = ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate archive JSON key")
            result[name] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def create_fixture(output, *, baseline=BASELINE, guard="guard_0", count=20):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if guard not in ("guard_0", "guard_1") or type(count) is not int or not 7 <= count <= 100:
        raise ValueError("Select one authored guard and a bounded stronger count from 7 through 100")
    if output.exists():
        raise FileExistsError("Fixture output must be a new directory")
    for authored in (ROOT / "engine", ROOT / "experiments", ROOT / "dependencies", ROOT / "validation", ROOT / "tests"):
        if output == authored or authored in output.parents:
            raise ValueError("Fixture archives must be outside authored sources; use ignored runs/")
    source = baseline.read_bytes()
    with zipfile.ZipFile(baseline) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source archive CRC failure")
        infos = archive.infolist()
        if len(infos) != len({info.filename for info in infos}):
            raise ValueError("Duplicate source archive entry")
        contents = {info.filename: archive.read(info) for info in infos}
    original = scenario_payloads(0)
    canonical = lambda data: json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if set(contents) != set(original) or any(canonical(_json(contents[name])) != canonical(data)
                                           for name, data in original.items()):
        raise ValueError("Fixture requires the unchanged authored baseline seed-0 map")
    objects = deepcopy(original["objects.json"])
    index = next(i for i, obj in enumerate(objects) if obj["instanceName"] == guard)
    objects[index]["options"]["amount"] = count
    replacement = (json.dumps(objects, indent=2, sort_keys=True) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    base_path, variant_path = output / "base.vmap", output / "casualty.vmap"
    base_path.write_bytes(source)
    with zipfile.ZipFile(variant_path, "w") as archive:
        for info in infos:
            archive.writestr(copy(info), replacement if info.filename == "objects.json" else contents[info.filename])
    with zipfile.ZipFile(variant_path) as archive:
        if archive.testzip() is not None or archive.namelist() != [info.filename for info in infos]:
            raise ValueError("Written archive validation failed")
        actual = {name: archive.read(name) for name in archive.namelist()}
    if baseline.read_bytes() != source or base_path.read_bytes() != source:
        raise ValueError("Source/base bytes changed")
    changed = [name for name in contents if contents[name] != actual[name]]
    decoded = _json(actual["objects.json"])
    decoded[index]["options"]["amount"] = 6
    if changed != ["objects.json"] or decoded != original["objects.json"]:
        raise ValueError("Fixture changes extend beyond the selected guard count")
    manifest = {
        "schema": "heroes-contextual-casualty-fixture-v1",
        "scope": "Isolated real-VCMI validation input only; never training, demonstrations, production policy or natural-baseline evidence",
        "engine_launched": False, "native_validated": False, "casualties_verified": False,
        "training_performed": False,
        "source": {"path": str(baseline), "sha256": sha(source)},
        "base": {"path": str(base_path), "sha256": sha(base_path.read_bytes()), "byte_identical_to_source": True},
        "fixture": {"path": str(variant_path), "sha256": sha(variant_path.read_bytes())},
        "change": {"entry": "objects.json", "json_pointer": f"/{index}/options/amount",
                   "instance_name": guard, "before": 6, "after": count,
                   "position": [objects[index][key] for key in ("x", "y", "l")]},
        "changed_archive_entries": changed,
        "entries": {name: {"source_sha256": sha(contents[name]), "fixture_sha256": sha(actual[name]),
                           "byte_identical": contents[name] == actual[name]} for name in contents},
        "preserved": {"initial_heroes_towns_resources": True, "all_other_object_fields": True,
                      "header_fog_and_victory_rules": True, "terrain_roads": True,
                      "neutral_behavior_and_no_growth": True, "source_archive_bytes": True},
        "native_format_basis": "Pinned VCMI lib/mapObjects/CGCreature.cpp:732-758 deserializes options.amount into the initial native stack count",
        "required_native_checks": [
            "Run the existing bounded closed-window-battle probe with --map fixture.path",
            "Verify a real automatic battle completes and the same owned hero survives",
            "Require observed reopened army count lower than the immutable earlier open-window count",
            "Require no current window-only fields throughout the closed interval",
            "Report no-casualty or hero-loss outcomes as unverified; do not fabricate freshness coverage",
        ],
    }
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--guard", choices=("guard_0", "guard_1"), default="guard_0")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    result = create_fixture(args.output, baseline=args.baseline, guard=args.guard, count=args.count)
    print(json.dumps({"fixture": result["fixture"], "manifest": str(args.output.resolve() / "fixture.json"),
                      "change": result["change"], "native_validated": False}, indent=2))


if __name__ == "__main__":
    main()
