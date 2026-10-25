#!/usr/bin/env python3
"""Create a diagnostic last-hero neutral-battle map; never training input.

Only the selected player's town/mainTown and one existing neutral stack count
change. Victory conditions, fog, terrain, heroes, and all other fields remain
the authored baseline. Creation neither launches VCMI nor proves an outcome.
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
    variant_payloads as townless_payloads,
)

SCHEMA = "heroes-contextual-neutral-terminal-fixture-v1"
PLAYERS = ("red", "blue")
GUARD_COUNT = 100


def variant_payloads(original, player):
    if player not in PLAYERS:
        raise ValueError("Fixture player must be red or blue")
    kind = "natural-townless" if player == "red" else "natural-townless-blue"
    expected, changes = townless_payloads(original, kind)
    guard_name = f"guard_{PLAYERS.index(player)}"
    index = next(i for i, obj in enumerate(expected["objects.json"]) if obj["instanceName"] == guard_name)
    guard = expected["objects.json"][index]
    before = guard["options"]["amount"]
    if guard["type"] != "monster" or guard["subtype"] != "pikeman" or type(before) is not int or before != 6:
        raise ValueError("Expected the audited six-pikeman neutral stack")
    guard["options"]["amount"] = GUARD_COUNT
    changes.append({"entry": "objects.json", "json_pointer": f"/{index}/options/amount",
                    "instance_name": guard_name, "operation": "replace", "before": before, "after": GUARD_COUNT})
    restored = deepcopy(expected)
    restored["objects.json"][index]["options"]["amount"] = before
    town_change = changes[1]
    restored["objects.json"].insert(int(town_change["json_pointer"][1:]), town_change["before"])
    restored["header.json"]["players"][player]["mainTown"] = changes[0]["before"]
    if canonical(restored) != canonical(original):
        raise ValueError("Fixture changed an undeclared field")
    return expected, changes


def verify_fixture(path, *, player, baseline=BASELINE):
    _, source, original = source_contents(Path(baseline))
    expected, changes = variant_payloads(original, player)
    _, actual = archive_contents(Path(path))
    if set(actual) != set(source):
        raise ValueError("Fixture archive entries differ from baseline")
    changed = sorted({item["entry"] for item in changes})
    for name in source:
        if name not in changed and actual[name] != source[name]:
            raise ValueError("Undeclared archive entry bytes changed: " + name)
        if canonical(decode(actual[name])) != canonical(expected[name]):
            raise ValueError("Fixture differs from its exact three declared changes: " + name)
    return changed, actual


def fixture_input(path):
    path = Path(path).resolve()
    manifest = decode(path.read_bytes())
    if type(manifest) is not dict or manifest.get("schema") != SCHEMA or manifest.get("player") not in PLAYERS:
        raise ValueError("Expected a neutral-terminal fixture manifest")
    source, game_map = Path(manifest["source"]["path"]), Path(manifest["fixture"]["path"])
    if not source.is_absolute() or not game_map.is_absolute():
        raise ValueError("Fixture source and map paths must be absolute")
    if sha(source.read_bytes()) != manifest["source"]["sha256"] or sha(game_map.read_bytes()) != manifest["fixture"]["sha256"]:
        raise ValueError("Fixture/source hash disagrees with manifest")
    verify_fixture(game_map, player=manifest["player"], baseline=source)
    _, _, original = source_contents(source)
    _, changes = variant_payloads(original, manifest["player"])
    if canonical(manifest.get("changes")) != canonical(changes):
        raise ValueError("Fixture change declaration disagrees with exact source diff")
    return manifest["player"], game_map, manifest


def create_fixture(output, *, player, baseline=BASELINE):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output.exists():
        raise FileExistsError("Fixture output must be a new directory")
    for name in ("engine", "experiments", "dependencies", "validation", "tests"):
        if output == ROOT / name or ROOT / name in output.parents:
            raise ValueError("Write fixtures outside authored sources; use ignored runs/")
    source_bytes = baseline.read_bytes()
    infos, source, original = source_contents(baseline)
    expected, changes = variant_payloads(original, player)
    changed = {item["entry"] for item in changes}
    output.mkdir(parents=True, exist_ok=False)
    base, target = output / "base.vmap", output / f"neutral-terminal-{player}.vmap"
    base.write_bytes(source_bytes)
    with zipfile.ZipFile(target, "w") as archive:
        for info in infos:
            content = ((json.dumps(expected[info.filename], indent=2, sort_keys=True) + "\n").encode()
                       if info.filename in changed else source[info.filename])
            archive.writestr(copy(info), content)
    changed_entries, actual = verify_fixture(target, player=player, baseline=baseline)
    if baseline.read_bytes() != source_bytes or base.read_bytes() != source_bytes:
        raise ValueError("Source/base archive bytes changed")
    manifest = {
        "schema": SCHEMA, "player": player, "upstream_commit": UPSTREAM,
        "scope": "Isolated native diagnostic only; never training, demonstrations, product policy or random-baseline outcome evidence",
        "engine_launched": False, "native_validated": False, "natural_outcome_verified": False,
        "neutral_terminal_request_verified": False, "training_performed": False,
        "source": {"path": str(baseline), "sha256": sha(source_bytes)},
        "base": {"path": str(base), "sha256": sha(base.read_bytes()), "byte_identical_to_source": True},
        "fixture": {"path": str(target), "sha256": sha(target.read_bytes())},
        "changes": changes, "changed_archive_entries": changed_entries,
        "entries": {name: {"source_sha256": sha(source[name]), "fixture_sha256": sha(actual[name]),
                            "byte_identical": source[name] == actual[name]} for name in source},
        "preserved": {"all_undeclared_fields": True, "initial_heroes_and_armies": True,
                      "fog_and_victory_rules": True, "terrain_roads": True, "source_archive_bytes": True},
        "evaluator_plan": {
            "actions": "Both players explicitly end one initial turn, then approach a currently visible gold pile if necessary and attack a currently visible neutral stack; other player explicitly ends turns",
            "day_budget": 4, "trigger": "Unchanged native defeat after losing the final hero with no town",
            "expected_outcomes": {color: "loss" if color == player else "win" for color in PLAYERS},
            "neutral_request_claim": "Require actual authenticated barrier player=-1 and request type=198; a player-caused battle ending does not cover this case",
            "never_substitute": "No seven-day townless ending, timeout, technical failure, injected terminal or hidden action coordinate"},
        "native_basis": {"monster_count": "lib/mapObjects/CGCreature.cpp:serializeJsonOptions amount",
                         "natural_defeat": "lib/gameState/CGameState.cpp:checkForVictoryAndLoss no towns/heroes",
                         "neutral_request_authentication": "server/CGameHandler.cpp:maybeSendExperimentTerminalBarrier",
                         "request_type": "lib/serializer/RegisterTypes.h:MakeAction (198)"},
    }
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    fixture_input(output / "fixture.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=PLAYERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    result = create_fixture(args.output, player=args.player, baseline=args.baseline)
    print(json.dumps({"fixture": result["fixture"], "manifest": str(args.output.resolve() / "fixture.json"),
                      "native_validated": False}, indent=2))


if __name__ == "__main__":
    main()
