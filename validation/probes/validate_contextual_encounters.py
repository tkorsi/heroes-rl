#!/usr/bin/env python3
"""Strict real-game encounter/window diagnostics; never learner examples.

The baseline encounter case visits only currently observed objects. The casualty
case accepts only an exactly declared single-guard-count variant of that map.
Neither case changes rules, initial player resources, rewards or engine state.
An absent required condition fails the diagnostic; its actual unscored episode
and all evidence are retained independently of the probe result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.learner import contextual as contract
from validation.probes import create_boundary_fixture as boundary
from validation.probes import validate_contextual_native as base


def casualty_input(path, player):
    """Verify actual archive isolation, not merely a manifest's own claim."""
    path = Path(path).resolve()
    declaration = boundary.decode(path.read_bytes())
    base.ensure(declaration.get("schema") == "heroes-contextual-casualty-fixture-v1",
                "Expected the existing declared casualty fixture")
    source = Path(declaration["source"]["path"])
    target = Path(declaration["fixture"]["path"])
    base.ensure(source.is_absolute() and target.is_absolute(), "Fixture paths must be absolute")
    base.ensure(base.sha(source) == declaration["source"]["sha256"]
                and base.sha(target) == declaration["fixture"]["sha256"], "Fixture digest mismatch")
    _, original, payloads = boundary.source_contents(source)
    _, contents = boundary.archive_contents(target)
    change = declaration["change"]
    guard = "guard_0" if player == "red" else "guard_1"
    count = change["after"]
    base.ensure(change["instance_name"] == guard and change["before"] == 6
                and type(count) is int and 7 <= count <= 100, "Wrong player guard or casualty count")
    expected = deepcopy(payloads["objects.json"])
    index = next(i for i, obj in enumerate(expected) if obj["instanceName"] == guard)
    expected[index]["options"]["amount"] = count
    base.ensure(change["entry"] == "objects.json" and change["json_pointer"] == f"/{index}/options/amount"
                and set(contents) == set(original), "Unexpected casualty archive declaration")
    base.ensure(boundary.canonical(boundary.decode(contents["objects.json"])) == boundary.canonical(expected)
                and all(contents[name] == original[name] for name in original if name != "objects.json"),
                "Casualty fixture changed fields beyond the declared guard count")
    return target, declaration


def require_checks(checks, names):
    """A missing cause, casualty or fresh reopening cannot satisfy acceptance."""
    for name in names:
        matching = [row for row in checks if row["check"] == name]
        base.ensure(matching and all(row["status"] == "passed" for row in matching),
                    f"Required native condition was not established: {name}")


def own_decision(run, color):
    while run.live():
        frame = run.inspect_request()
        if run.state.color == color:
            return frame
        run.call(contract.Action("end_turn"), "explicit_other_player_turn")
    raise base.ProbeFailure("Native episode/budget ended before the required own decision")


def encounter_evidence(kind, target, before, after, frames, result):
    """Require the selected object's real effect and cessation of native steps."""
    results = [frame for frame in frames if frame.kind == "result" and frame.result == result]
    base.ensure(len(results) == 1, "Encounter requires one matching completed action")
    final = results[0]
    consumed = [frame for frame in frames if frame.kind == "consumed"
                and frame.decision_seq == final.decision_seq and frame.seq < final.seq]
    base.ensure(len(consumed) == 1, "Encounter requires its own native consumption")
    active = [frame for frame in frames if consumed[0].seq <= frame.seq <= final.seq]
    stop = base.route_stop_evidence(frames, result)["encounter"]
    base.ensure(stop["status"] == "passed" and result is not None and result.accepted,
                "Encounter lacks its real cause, interaction result or route cessation")
    events = [frame.event for frame in active if frame.event is not None]
    visits = [frame.seq for frame in active if frame.event is not None and frame.event.kind == "visit"
              and frame.event.subject == target.handle and frame.event.object_kind == target.kind]
    base.ensure(visits,
                "Encounter did not visit the selected currently observed object")
    issued = [frame for frame in active if frame.kind == "issued" and frame.decision_seq == final.decision_seq]
    acknowledged = [frame for frame in active if frame.kind == "acknowledged"
                    and frame.decision_seq == final.decision_seq]
    base.ensure(issued and len(issued) == len(acknowledged)
                and {f.command_index for f in issued} == {f.command_index for f in acknowledged}
                and all(frame.accepted for frame in acknowledged),
                "Encounter lacks an issued and accepted native move")
    report = {"target": target.handle, "target_position": target.position, "stop": stop,
              "decision_seq": final.decision_seq, "consumed_seq": consumed[0].seq,
              "visit_seqs": visits, "issued_seqs": [f.seq for f in issued],
              "acknowledged_seqs": [f.seq for f in acknowledged]}
    if kind == "resource":
        amounts = [component.amount for event in events if event.kind == "notice"
                   for component in event.notice.components
                   if component.kind == "resource" and component.resource == "gold"]
        base.ensure(amounts and sum(amounts) > 0 and after.resources[6] - before.resources[6] == sum(amounts),
                    "Resource encounter lacks its displayed notice and exact native gold credit")
        base.ensure(all(obj.handle != target.handle for obj in after.objects), "Collected resource remains current")
        report["gold_credit"] = sum(amounts)
        report["notice_seqs"] = [f.seq for f in active if f.event is not None and f.event.kind == "notice"]
    elif kind == "battle":
        base.ensure(any(event.kind == "battle_start" for event in events)
                    and any(event.kind == "battle_complete" and event.won is True for event in events),
                    "Neutral encounter lacks a completed surviving battle")
        base.ensure(len(before.heroes) == 1 and any(hero.handle == before.heroes[0].handle for hero in after.heroes),
                    "Neutral battle did not retain the same owned hero")
        report["battle_won"] = True
        report["battle_seqs"] = [f.seq for f in active if f.event is not None
                                 and f.event.kind in ("battle_start", "battle_complete")]
    elif kind == "mine":
        base.ensure(any(obj.handle == target.handle and obj.owner == "self" for obj in after.objects),
                    "Mine visit did not produce a currently observed owned mine")
        report["observed_owner"] = "self"
    else:
        raise ValueError("Unknown encounter evidence kind")
    return report


def observed_encounter_destination(obs, object_kind, tried_frontiers, approach=None):
    """Diagnostic-only explicit approach after an unknown route, using visible tiles.

    A visible mine can have an unseen entrance approach. This does not query a
    pathfinder or reveal a tile: the next external action selects a currently
    observed free frontier, then retries the actual object in a later decision.
    """
    targets = sorted((obj for obj in obs.objects if obj.kind == object_kind and obj.owner == "neutral"
                      and (object_kind != 79 or obj.subtype == 6)), key=lambda obj: obj.position)
    if targets and approach is None:
        return targets[0], targets[0].position
    known = {tile.position for tile in obs.tiles}
    candidates = [tile.position for tile in obs.tiles if not tile.blocked and not tile.visitable
                  and tile.position != obs.heroes[0].position and tile.position not in tried_frontiers and any(
                      0 <= tile.position[0] + dx < obs.size[0] and 0 <= tile.position[1] + dy < obs.size[1]
                      and (tile.position[0] + dx, tile.position[1] + dy, 0) not in known
                      for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
    base.ensure(candidates, "No untried observed frontier for the diagnostic approach")
    # Target proximity only chooses an evaluator setup action. This function is
    # never imported by a learner, collector, policy, planner or optimizer.
    position = min(candidates, key=lambda p: (sum(abs(a - b) for a, b in zip(p, approach)), p)
                   if approach is not None else p)
    return None, position


def encounters(run):
    color = run.args.player
    tried_frontiers = set()
    movements = 0
    for name, object_kind in (("resource", 79), ("battle", 54), ("mine", 53)):
        approach = None
        while True:
            frame = own_decision(run, color)
            obs = frame.observation
            base.ensure(obs.context.kind == "map" and len(obs.heroes) == 1,
                        "Encounter diagnostic requires its original owned hero on the map")
            hero = obs.heroes[0]
            if hero.movement == 0:
                run.call(contract.Action("end_turn"), "explicit_turn_after_exhausted_movement")
                continue
            target, position = observed_encounter_destination(obs, object_kind, tried_frontiers, approach)
            if target is None:
                base.ensure(len(tried_frontiers) < run.args.frontier_attempts,
                            f"No observed {name} within the declared exploration bound")
                tried_frontiers.add(position)
            base.ensure(movements < run.args.movement_attempts, "Encounter movement bound exhausted")
            movements += 1
            _, additions, result = run.call(contract.Action("move_to", hero.handle, *position[:2]),
                                            f"explicit_observed_{name}" if target else "explicit_known_frontier")
            after = own_decision(run, color).observation
            base.ensure(result is not None, "Diagnostic movement has no completed result")
            if result.status == "no_route_known":
                base.ensure(not any(f.kind == "issued" for f in additions)
                            and after.heroes == obs.heroes, "Unknown route issued or moved a hero")
                approach = position if target is not None else approach
                run.check("native_unknown_route_requires_explicit_observed_approach", "passed",
                          target=target.handle if target is not None else None, destination=position,
                          decision_seq=frame.seq, native_issued=0)
                continue
            approach = None
            if target and any(f.event is not None and f.event.kind == "visit" and f.event.subject == target.handle
                              for f in additions):
                evidence = encounter_evidence(name, target, obs, after, additions, result)
                run.check(f"native_{name}_encounter_stops_route", "passed", **evidence)
                break
            base.ensure(result.status in ("movement_limit", "reached", "interaction"),
                        "Diagnostic movement ended without the required encounter")
            if result.status == "movement_limit":
                run.call(contract.Action("end_turn"), "explicit_turn_after_partial_encounter_route")
    require_checks(run.checks, [f"native_{name}_encounter_stops_route" for name in ("resource", "battle", "mine")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("encounters", "hero-casualties"), required=True)
    parser.add_argument("--player", choices=base.COLORS, default="red")
    parser.add_argument("--casualty-fixture", type=Path)
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--max-days", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--case-seconds", type=float, default=120)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    parser.add_argument("--frontier-attempts", type=int, default=12)
    parser.add_argument("--movement-attempts", type=int, default=32)
    args = parser.parse_args(argv)
    if (not 1 <= args.seed <= 2147483647 or not 1 <= args.max_days <= 40
            or not 1 <= args.max_decisions <= 1000 or not 1 <= args.frontier_attempts <= 100
            or not 1 <= args.movement_attempts <= 100 or any(not math.isfinite(v) or not 0 < v <= 180
            for v in (args.case_seconds, args.watchdog, args.startup_seconds))):
        parser.error("Use bounded positive seeds, days, choices and finite time limits")
    for name in ("map", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; native evidence is never overwritten")
    declaration = None
    if args.casualty_fixture:
        if args.case != "hero-casualties":
            parser.error("Only the hero-casualty case accepts a single-guard-count fixture")
        args.casualty_fixture = args.casualty_fixture.expanduser().resolve()
        args.map, declaration = casualty_input(args.casualty_fixture, args.player)
    else:
        boundary.source_contents(args.map)
    env = base.load("heroes_encounter_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_encounter_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    base.write(args.output / "plan.json", {
        "schema": "heroes-contextual-encounter-validation-v1", "case": args.case, "player": args.player,
        "engine_seed": args.seed, "map": str(args.map), "map_sha256": base.sha(args.map),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": base.sha(args.binary), "probe_sha256": base.sha(Path(__file__)),
        "shared_probe_sha256": base.sha(Path(base.__file__)), "fixture": declaration,
        "fixture_manifest_sha256": base.sha(args.casualty_fixture) if args.casualty_fixture else None,
        "bounds": {name: getattr(args, name) for name in ("max_days", "max_decisions", "case_seconds", "watchdog",
                                                         "frontier_attempts", "movement_attempts")},
        "required_conditions": (["resource", "battle", "mine"] if args.case == "encounters" else
                                ["same_hero_survives", "closed_fields_absent", "casualties_visible_after_reopen"]),
        "training_performed": False, "training_input": False, "ordinary_outcome_claim": False})
    run = base.CaseRun(args.case, args, env, contract, auditor, args.seed)
    error = None
    try:
        run.start()
        own_decision(run, args.player)
        if args.case == "encounters":
            encounters(run)
        else:
            base.closed_window_battle(run)
            require_checks(run.checks, ("closed_window_snapshot_immutable_through_battle_attempts",
                                       "closed_window_native_battle_completed", "closed_window_encounter_stop_observed",
                                       "closed_window_changed_army_then_reopen"))
        run.finish()
        run.persist_and_audit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("required_native_condition", "failed", error=error)
        try:
            run.finish()
            if run.end is not None and not run.audit_attempted:
                run.persist_and_audit()
        except Exception as cleanup:
            run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup).__name__}: {cleanup}")
    finally:
        run.env.close()
    report = run.summary(error)
    report.update(training_input=False, ordinary_outcome_claim=False)
    base.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "error": error, "report": str(args.output / "report.json")}))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
