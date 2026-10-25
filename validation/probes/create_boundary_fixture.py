#!/usr/bin/env python3
"""Create isolated native terminal or unsupported-window validation inputs.

These maps and their explicit evaluator actions are never training examples,
strategies, or random-baseline outcome evidence. Creation runs no engine.
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
UPSTREAM = "6901839ae9760dfcd1d5ff2c4aa4ef6968fee752"
KINDS = ("natural-townless", "natural-townless-blue", "unsupported-den")
SCHEMAS = {"natural-townless": "heroes-contextual-natural-townless-fixture-v1",
           "natural-townless-blue": "heroes-contextual-natural-townless-blue-fixture-v1",
           "unsupported-den": "heroes-contextual-unsupported-interface-fixture-v1"}
TOWNLESS_PLAYERS = {"natural-townless": ("red", "town_0"),
                    "natural-townless-blue": ("blue", "town_1")}
DEN_TEMPLATE = {"allowedTerrains": ["grass", "swamp", "dirt"], "animation": "avxdent.def",
                "mask": ["VV", "BA"], "visitableFrom": ["---", "+++", "+++"]}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode(data):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate JSON field")
            result[name] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def archive_contents(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
        infos = archive.infolist()
        if len(infos) != len({info.filename for info in infos}):
            raise ValueError("Duplicate archive entry")
        return infos, {info.filename: archive.read(info) for info in infos}


def source_contents(baseline):
    if decode((ROOT / "dependencies/vcmi.json").read_bytes())["commit"] != UPSTREAM:
        raise ValueError("Engine pin changed; re-audit the fixture format first")
    infos, contents = archive_contents(baseline)
    expected = scenario_payloads(0)
    if set(contents) != set(expected) or any(canonical(decode(contents[name])) != canonical(payload)
                                           for name, payload in expected.items()):
        raise ValueError("Fixture requires the unchanged authored baseline seed-0 map")
    return infos, contents, expected


def footprint(obj):
    result = set()
    mask = obj["template"]["mask"]
    for row, values in enumerate(mask):
        for column, cell in enumerate(values):
            if cell not in "0 VBAHT":
                raise ValueError("Unsupported object footprint")
            if cell not in " 0":
                result.add((obj["x"] - len(values) + 1 + column,
                            obj["y"] - len(mask) + 1 + row, obj["l"]))
    return result


def variant_payloads(original, kind):
    if kind not in KINDS:
        raise ValueError("Unknown boundary fixture kind")
    payloads = deepcopy(original)
    objects = payloads["objects.json"]
    changes = []
    if kind in TOWNLESS_PLAYERS:
        color, town_name = TOWNLESS_PLAYERS[kind]
        town_index = next(i for i, obj in enumerate(objects) if obj["instanceName"] == town_name)
        old_town = objects.pop(town_index)
        old_main = payloads["header.json"]["players"][color].pop("mainTown")
        changes = [{"entry": "header.json", "json_pointer": f"/players/{color}/mainTown", "operation": "remove",
                    "before": old_main, "after": None},
                   {"entry": "objects.json", "json_pointer": f"/{town_index}", "instance_name": town_name,
                    "operation": "remove", "before": old_town, "after": None}]
        # Verify exact field isolation independently of the construction above.
        restored = deepcopy(payloads)
        restored["header.json"]["players"][color]["mainTown"] = old_main
        restored["objects.json"].insert(town_index, old_town)
    else:
        index = next(i for i, obj in enumerate(objects) if obj["instanceName"] == "gold_0")
        replacement = {"instanceName": "validation_den_0", "options": {}, "subtype": "object",
                       "template": deepcopy(DEN_TEMPLATE), "type": "denOfThieves"}
        old_object = deepcopy(objects[index])
        for name, value in replacement.items():
            changes.append({"entry": "objects.json", "json_pointer": f"/{index}/{name}",
                            "before": deepcopy(objects[index][name]), "after": deepcopy(value)})
            objects[index][name] = value
        occupied = footprint(objects[index])
        others = set().union(*(footprint(obj) for i, obj in enumerate(objects) if i != index))
        if occupied & others or any(not (0 <= x < 20 and 0 <= y < 20 and z == 0) for x, y, z in occupied):
            raise ValueError("Unsupported Den footprint overlaps another object or the map edge")
        restored = deepcopy(payloads)
        restored["objects.json"][index] = old_object
    if canonical(restored) != canonical(original):
        raise ValueError("Fixture modified undeclared fields")
    return payloads, changes


def verify_fixture(path, *, kind, baseline=BASELINE):
    """Verify exact fixture semantics and untouched entry bytes before native use."""
    _, source, original = source_contents(Path(baseline))
    expected, changes = variant_payloads(original, kind)
    _, actual = archive_contents(Path(path))
    changed_entries = sorted({change["entry"] for change in changes})
    if set(actual) != set(source):
        raise ValueError("Fixture archive entries differ from baseline")
    for name in source:
        if name not in changed_entries and actual[name] != source[name]:
            raise ValueError("Undeclared archive entry bytes changed: " + name)
        if canonical(decode(actual[name])) != canonical(expected[name]):
            raise ValueError("Fixture fields differ from exact declared changes: " + name)
    return changed_entries, actual


def create_fixture(output, *, kind, baseline=BASELINE):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output.exists():
        raise FileExistsError("Fixture output must be a new directory")
    for name in ("engine", "experiments", "dependencies", "validation", "tests"):
        authored = ROOT / name
        if output == authored or authored in output.parents:
            raise ValueError("Write validation fixtures outside authored sources; use ignored runs/")
    source_bytes = baseline.read_bytes()
    infos, source, original = source_contents(baseline)
    expected, changes = variant_payloads(original, kind)
    changed_entries = sorted({change["entry"] for change in changes})
    output.mkdir(parents=True, exist_ok=False)
    base, target = output / "base.vmap", output / f"{kind}.vmap"
    base.write_bytes(source_bytes)
    with zipfile.ZipFile(target, "w") as archive:
        for info in infos:
            content = ((json.dumps(expected[info.filename], indent=2, sort_keys=True) + "\n").encode()
                       if info.filename in changed_entries else source[info.filename])
            archive.writestr(copy(info), content)
    actual_changes, actual = verify_fixture(target, kind=kind, baseline=baseline)
    if baseline.read_bytes() != source_bytes or base.read_bytes() != source_bytes:
        raise ValueError("Source or base archive bytes changed")
    manifest = {
        "schema": SCHEMAS[kind], "kind": kind,
        "scope": "Isolated evaluator input only; never training, demonstrations, production policy or random-baseline outcome evidence",
        "engine_launched": False, "native_validated": False, "training_performed": False,
        "source": {"path": str(baseline), "sha256": sha(source_bytes)},
        "base": {"path": str(base), "sha256": sha(base.read_bytes()), "byte_identical_to_source": True},
        "fixture": {"path": str(target), "sha256": sha(target.read_bytes())},
        "changes": changes, "changed_archive_entries": actual_changes,
        "entries": {name: {"source_sha256": sha(source[name]), "fixture_sha256": sha(actual[name]),
                           "byte_identical": source[name] == actual[name]} for name in source},
        "preserved": {"all_undeclared_fields": True, "untouched_archive_entry_bytes": True,
                      "initial_heroes_and_armies": True, "fog_and_victory_rules": True,
                      "terrain_roads": True, "source_archive_bytes": True},
        "upstream_commit": UPSTREAM,
    }
    if kind in TOWNLESS_PLAYERS:
        color, _ = TOWNLESS_PLAYERS[kind]
        manifest.update(natural_outcome_verified=False,
            evaluator_plan={"actions": "Explicit end_turn at each real player decision; no injected outcome or counter",
                            "expected_outcomes": {player: "loss" if player == color else "win"
                                                  for player in ("red", "blue")}, "day_budget": 10,
                            "trigger": "Unchanged native seven-days-without-a-town rule",
                            "postterminal_views": "Coverage only; absence for one player is not a failed natural boundary"},
            native_basis={"missing_main_town": "lib/mapping/MapFormatJson.cpp:490-501",
                          "townless_counter": "server/processors/NewTurnProcessor.cpp:163-187",
                          "unchanged_seven_day_threshold": "lib/gameState/CGameState.cpp:1364-1369",
                          "terminal_and_final_removals": "server/CGameHandler.cpp:3746-3804"})
        # Keep the original Red manifest schema and map bytes unchanged.
        if kind == "natural-townless-blue":
            manifest["townless_player"] = color
    else:
        den = next(obj for obj in expected["objects.json"] if obj["instanceName"] == "validation_den_0")
        manifest.update(evaluator_target={"native_kind": 97, "native_subtype": 0,
                                          "position": [den[key] for key in ("x", "y", "l")],
                                          "expected_callback": "showThievesGuildWindow",
                                          "expected_failure": "unsupported_interface"},
                        footprint=[list(pos) for pos in sorted(footprint(den))], footprint_nonoverlapping=True,
                        native_basis={"template": "config/objects/generic.json:261-278 denOfThieves/object/green",
                                      "visit": "lib/mapObjects/MiscObjects.cpp:1166-1169",
                                      "callback": "client/NetPacksClient.cpp:1025-1033"})
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    result = create_fixture(args.output, kind=args.kind, baseline=args.baseline)
    print(json.dumps({"fixture": result["fixture"], "manifest": str(args.output.resolve() / "fixture.json"),
                      "native_validated": False}, indent=2))


if __name__ == "__main__":
    main()
