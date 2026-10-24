#!/usr/bin/env python3
"""Bounded review probes: synthetic history and native standalone regression only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True, help="Reviewed PR checkout root")
parser.add_argument("--vcmi-source", type=Path, required=True, help="Existing prepared VCMI source (read only)")
parser.add_argument("--learner-parent", type=Path, required=True, help="Folder containing learner/contract.py")
parser.add_argument("--output", type=Path, required=True, help="JSON report in an existing output directory")
args = parser.parse_args()
ROOT, UPSTREAM, LEARNER_PARENT = (p.resolve() for p in (args.source, args.vcmi_source, args.learner_parent))
OUT = args.output.resolve()
OUT.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments/headless"))
sys.path.insert(0, str(LEARNER_PARENT))
from bridge_protocol import BridgeValidator
from player_history import PlayerHistory
from learner.contract import HistoryCursor, decode_frame


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wire(kind: str, **kwargs) -> dict:
    return {"type": kind, "protocol_version": 3, "player": 0, "episode": "probe", **kwargs}


own = {"hero": [0, 0, 0], "movement": 1, "movement_max": 1,
       "resources": [0] * 7, "army": [{"slot": 0, "creature": 1, "count": 10}],
       "towns": [], "mines": 0}
hero = {"id": 1, "x": 0, "y": 0, "z": 0, "type": 34, "subtype": 0, "owner": 0}
enemy = {"id": 2, "x": 2, "y": 1, "z": 0, "type": 34, "subtype": 0, "owner": 1}
tiles = [{"x": x, "y": y, "z": 0, "terrain": 2, "blocked": False, "visitable": False}
         for y in range(3) for x in range(3)]


def state(day: int) -> dict:
    return {"day": day, "hero_id": 1, "x": 0, "y": 0, "z": 0,
            "own_heroes": 1, "own_towns": 0, "own_mines": 0,
            **{k: own[k] for k in ("movement", "movement_max", "resources", "army", "towns")}}


def observation(action: int, day: int, events: list[dict]) -> dict:
    visible = []
    for tile in tiles:
        objects = [{k: obj[k] for k in ("id", "type", "subtype", "owner")}
                   for obj in (hero, enemy) if all(tile[k] == obj[k] for k in ("x", "y", "z"))]
        visible.append({**tile, "objects": objects})
    return wire("observation", action_id=action, day=day, hero_id=1, x=0, y=0, z=0,
                width=20, height=20, levels=1, movement=1, movement_max=1,
                action_mask=[True] * 9, state=state(day), tiles=visible, events=events)


def update(sequence: int, day: int, **kwargs) -> dict:
    return {"kind": "visible_update", "sequence": sequence, "day": day,
            "appeared": [], "changed": [], "disappeared": [], "revealed": [],
            "tiles_changed": [], "hidden": [], "own": None, **kwargs}


validator = BridgeValidator(0, "probe")
history = PlayerHistory(0, "red", "probe")


def ingest(message: dict) -> None:
    kind = validator.check(message)
    if kind == "ready":
        history.start(message["map_size"])
    else:
        history.ingest(message)


ingest(wire("ready", adventure_actions=9, observation_events="visible_update-v2",
            map_size=[20, 20, 1], snapshot_every=1))
ingest(observation(0, 1, [update(0, 1, appeared=[hero, enemy], revealed=tiles, own=own)]))
validator.respond(8)
history.act(8)
ingest(wire("action_result", action_id=0, action=8, accepted=True,
            before=state(1), after=state(1), events=[]))
ingest(wire("turn_end"))
ingest(observation(1, 3, []))
visible_day_three = next(item for item in history.memory() if item["handle"] == 2)
assert history.records[-1]["day"] == 3
assert any(obj["handle"] == "2" for obj in history.v1.frames[-1]["view"]["objects"])
validator.respond(8)
history.act(8)
ingest(wire("action_result", action_id=1, action=8, accepted=True,
            before=state(3), after=state(4), events=[update(1, 4, disappeared=[2])]))
ingest(wire("turn_end"))
hidden_day_four = next(item for item in history.memory() if item["handle"] == 2)
cursor = HistoryCursor("probe", 0)
for frame in history.v1.frames:
    cursor.consume(decode_frame(frame))
assert visible_day_three["day"] == 1, "Known stale-last-seen reproduction no longer holds"
assert hidden_day_four["day"] == 1 and not hidden_day_four["visible"]
assert hidden_day_four["fact"] == visible_day_three["fact"], "Hidden state changed a remembered fact"

command = [sys.executable, "-m", "unittest", "discover", "-s", "tests",
           "-p", "test_native_visible_state.py", "-v"]
native = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=90)
log = OUT.with_name("native-visible-state.log")
log.write_text(native.stdout + native.stderr)
assert native.returncode == 0, native.stderr
pin = json.loads((ROOT / "dependencies/vcmi.json").read_text())["commit"]
paths = ["lib/battle/SideInBattle.cpp", "lib/constants/EntityIdentifiers.cpp",
         "lib/mapObjects/CGObjectInstance.cpp"]
upstream_evidence = []
for relative in paths:
    path = UPSTREAM / relative
    pinned = subprocess.check_output(["git", "-C", str(UPSTREAM), "show", f"{pin}:{relative}"])
    upstream_evidence.append({"path": str(path), "sha256": sha(path),
                              "matches_pinned_upstream": path.read_bytes() == pinned})
assert all(e["matches_pinned_upstream"] for e in upstream_evidence)
report = {
    "reviewed_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "scope": "Synthetic PlayerHistory reproduction, source semantics review, and standalone native visibility regression. No real games, full engine build, benchmark or training.",
    "last_seen_time": {"priority": "P2", "bug_reproduced": True,
                       "interpretation": "This confirms the known defect; it is not a passing correct-behavior test.",
                       "latest_confirmed_sighting_day": 3,
                       "memory_while_visible_on_day_three": visible_day_three,
                       "memory_after_disappearance_on_day_four": hidden_day_four,
                       "hidden_facts_preserved": True, "strict_history_frames_validated": len(history.v1.frames),
                       "history_v1_summary": history.v1.summary(),
                       "source": "experiments/headless/player_history.py:188"},
    "battle_classification": {
        "verdict": "No actionable issue found in bbfc955 for supported normal owners and creature-generator guards.",
        "upstream_pin": pin, "sources": upstream_evidence,
        "reason": "Pinned SideInBattle::init sets creature-generator1..4 guards to NEUTRAL and maps UNFLAGGABLE to NEUTRAL. CGObjectInstance defaults to UNFLAGGABLE (-2), while NEUTRAL is -1. The new predicate includes both unowned owners and generator guards; hero and town-garrison labels retain precedence.",
        "limitation": "The native test covers the engine-free helper; engine integration was inspected against pinned source, not rebuilt or exercised live. isValidPlayer also rejects special sentinels outside normal army ownership; no such army scenario was established."
    },
    "native_visibility_regression": {"command": command, "cwd": str(ROOT), "returncode": native.returncode,
                                     "correct_behavior_regression_passed": True, "tests_passed": 1, "coverage": "10 scenarios, 200 randomized folds, opponent kinds",
                                     "log": str(log), "log_sha256": sha(log)},
    "reviewed_files": {str(path.relative_to(ROOT)): sha(path) for path in
                       [ROOT / "experiments/headless/player_history.py", ROOT / "engine/ExplorerRL/ExplorerRL.cpp",
                        ROOT / "engine/ExplorerRL/VisibleState.h", ROOT / "tests/native/visible_state.cpp"]},
    "probe": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__)),
              "argv": sys.argv, "exit_semantics": "0 means the known last-seen bug was reproduced and the native correctness regression passed; it does not mean the history bug is fixed."}
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output": str(OUT), "reviewed_commit": report["reviewed_commit"],
                  "last_seen_reproduced": True, "native_regression_passed": True,
                  "strict_frames": len(history.v1.frames)}, indent=2))
