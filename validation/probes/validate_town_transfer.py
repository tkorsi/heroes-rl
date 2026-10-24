#!/usr/bin/env python3
"""Bounded native v4 transfer diagnostics on the unchanged baseline; never training.

Every move, construction, purchase and transfer is an explicit evaluator action.
API rejections are labelled separately from actual native requests and effects.
No quantity, destination, acknowledgement or engine state is fabricated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from validation.probes import validate_town_native as town


def exact_transfer(before, after, action, additions, result):
    """Check native evidence and independent two-row arithmetic, not status alone."""
    left = before.observation.window.town_panel.armies
    right = after.observation.window.town_panel.armies
    base.ensure(left.destination is not None and right.destination is not None, "Missing actual visitor row")
    source = next((slot for slot in left.source.slots if slot.ref == action.selection), None)
    destination = next((slot for slot in left.destination.slots if slot.ref == action.destination), None)
    base.ensure(source is not None and destination is not None, "Wrong explicit transfer row or slot")
    base.ensure(0 < action.count <= source.count and destination.creature in (None, source.creature),
                "Transfer is not an exact supported forward quantity")
    base.ensure(source.count + destination.count <= 2**31 - 1, "Native split combined count overflows")
    base.ensure(not (left.source.requires_nonempty and sum(slot.count > 0 for slot in left.source.slots) == 1
                         and action.count == source.count), "Transfer removes a required last creature")
    consumed = [f for f in additions if f.kind == "consumed" and f.decision_seq == before.seq]
    issued = [f for f in additions if f.kind == "issued" and f.decision_seq == before.seq]
    acknowledged = [f for f in additions if f.kind == "acknowledged" and f.decision_seq == before.seq]
    results = [f for f in additions if f.kind == "result" and f.decision_seq == before.seq and f.result == result]
    base.ensure(result is not None and result.status == "transferred" and result.accepted
                and len(consumed) == len(issued) == len(acknowledged) == len(results) == 1
                and consumed[0].consumer == "engine" and acknowledged[0].accepted
                and issued[0].command_index == acknowledged[0].command_index
                and consumed[0].seq < issued[0].seq < acknowledged[0].seq < results[0].seq,
                "Transfer lacks one consumed, issued, accepted and completed native request")
    expected = asdict(left)
    from_slot = expected["source"]["slots"][source.slot]
    to_slot = expected["destination"]["slots"][destination.slot]
    from_slot["count"] -= action.count
    if from_slot["count"] == 0:
        from_slot["creature"] = None
    to_slot["creature"] = source.creature
    to_slot["count"] += action.count
    base.ensure(asdict(right) == expected, "Transfer changed another row, slot, identity or quantity")
    base.ensure(before.observation.resources == after.observation.resources
                and before.observation.context == after.observation.context
                and not any(f.kind in ("context", "context_invalidated") for f in additions),
                "Transfer changed resources or navigated a window")
    return {"source_slot": source.slot, "destination_slot": destination.slot,
            "creature": source.creature, "count": action.count, "native_destination_total": to_slot["count"],
            "decision_seq": before.seq, "consumed_seq": consumed[0].seq, "issued_seq": issued[0].seq,
            "acknowledged_seq": acknowledged[0].seq, "result_seq": results[0].seq,
            "source_remaining": from_slot["count"], "native_rejection": False}


class TransferRun(town.TownRun):
    def expected_offers(self, frame):
        return contract.Offers.from_observation(frame.observation, schema=contract.ARMIES_ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return contract.Action(kind, schema=contract.ARMIES_ACTION_SCHEMA, **parameters)

    def armies(self):
        rows = self.panel("overview").armies
        base.ensure(rows is not None and rows.source.role == "town_garrison", "Baseline lacks its native town row")
        return rows

    def transfer(self, source_slot, destination_slot, count, label):
        rows = self.armies()
        base.ensure(rows.destination is not None, "Transfer requires a displayed owned visitor")
        action = self.town_action("transfer_stack", selection=rows.source.slots[source_slot].ref,
                                  destination=rows.destination.slots[destination_slot].ref, count=count)
        before, additions, result = self.call(action, label)
        evidence = exact_transfer(before, self.inspect_request(), action, additions, result)
        self.check(label, "passed", **evidence)

    def reject(self, action, label):
        self.call(action, label, expect_rejection=True)
        self.checks[-1].update(rejection_layer="python_pre_send", native_issued=0, native_rejected_ack=False)


def diagnostic(run, crash=False):
    color = run.state.color
    original = run.inspect_request()
    base.ensure(len(original.observation.heroes) == 1 and len(original.observation.towns) == 1,
                "Transfer diagnostic requires the unchanged one-hero/one-town baseline")
    hero = original.observation.heroes[0]
    own_town = original.observation.towns[0]
    run.call(run.action("open_hero", subject=hero.handle), "explicit_initial_hero_window")
    initial_army = run.inspect_request().observation.window.army
    initial_prefix = run.history(color)
    run.call(run.action("close"), "explicit_close_initial_hero_window")
    run.open_town()
    initial_rows = run.armies()
    base.ensure(initial_rows.destination is not None and initial_rows.destination.subject == hero.handle,
                "Native baseline did not register the actual own visiting hero")
    remembered_destination = initial_rows.destination.slots[1].ref
    run.call(run.action("close"), "explicit_close_town_before_departure")
    obs = run.inspect_request().observation
    adjacent = sorted(tile.position for tile in obs.tiles if not tile.blocked and not tile.visitable
                      and tile.position[2] == hero.position[2]
                      and max(abs(tile.position[i] - hero.position[i]) for i in (0, 1)) == 1)
    base.ensure(adjacent, "No observed free departure tile for diagnostic setup")
    _, _, result = run.call(run.action("move_to", subject=hero.handle, x=adjacent[0][0], y=adjacent[0][1]),
                            "explicit_visible_departure")
    base.ensure(result is not None and result.status == "reached", "Native departure did not reach the chosen tile")
    run.open_town()
    base.ensure(run.armies().destination is None, "Absent visitor row remained current after departure")
    run.check("native_visitor_absence_requires_reentry", "passed")

    # Real native construction exposes both level-one creature cards. No map,
    # hero, resource or army fixture supplies the diagnostic stacks.
    hall = run.open_hall()
    upgrade = next((choice for choice in hall.buildings if choice.status == "allowed"
                    and ".dwellingUpLvl1." in choice.name_text_id), None)
    base.ensure(upgrade is not None, "Native baseline does not offer the declared dwelling upgrade")
    details = run.open_building(upgrade)
    _, additions, result = run.call(run.town_action("construct", selection=details.ref), "explicit_upgrade_for_two_creature_cards")
    town.request_evidence(run, additions, result, "built")
    panel = run.open_recruitment()
    panel = run.select_creature(0)
    base.ensure(panel.slider_max >= 7, "Native panel lacks seven affordable pikemen")
    army, _ = town.successful_purchase(run, (), panel, 7, "explicit_recruit_seven_pikemen")
    run.call(run.town_action("back"), "explicit_back_after_first_purchase")
    rows = run.armies()
    source_slot = next(slot for slot in rows.source.slots if slot.creature == 0)
    run.reject(run.town_action("transfer_stack", selection=source_slot.ref,
                              destination=remembered_destination, count=1), "absent_visitor_transfer_rejected_before_send")
    run.call(run.action("close"), "explicit_close_town_before_return")
    run.call(run.action("move_to", subject=hero.handle, x=own_town.position[0], y=own_town.position[1]),
             "explicit_return_to_observed_owned_town")
    run.open_town()
    rows = run.armies()
    base.ensure(rows.destination is not None and rows.destination.subject == hero.handle,
                "Reentry did not bind the actual same owned visitor")
    source_slot = next(slot for slot in rows.source.slots if slot.creature == 0)
    for label, changes in (
        ("foreign_source_slot", {"selection": "not-displayed-source"}),
        ("reverse_source_row", {"selection": rows.destination.slots[0].ref}),
        ("foreign_destination_slot", {"destination": "not-displayed-destination"}),
        ("upper_destination_row", {"destination": rows.source.slots[1].ref}),
        ("over_source_count", {"count": source_slot.count + 1}),
        ("foreign_town", {"subject": "not-owned-town"}),
    ):
        parameters = {"subject": own_town.handle, "selection": source_slot.ref,
                      "destination": rows.destination.slots[1].ref, "count": 1, **changes}
        run.reject(run.action("transfer_stack", **parameters), label + "_rejected_before_send")
    run.transfer(source_slot.slot, 1, 7, "native_full_stack_move_to_selected_empty_slot")
    base.ensure(all(slot.count == 0 for slot in run.armies().source.slots), "Town did not permit an empty source army")
    if crash:
        run.kill_owned_process()
        run.finish()
        base.ensure(run.end.status == "failed" and all(v is None for v in run.end.bootstrap.values()),
                    "Failure after transfer acknowledgement became an eligible cutoff")
        run.check("native_crash_after_transfer_is_unscored", "passed")
        return

    panel = run.open_recruitment()
    panel = run.select_creature(1)
    base.ensure(panel.slider_max >= 7, "Native shared stock lacks the remaining seven halberdiers")
    town.successful_purchase(run, (), panel, 7, "explicit_recruit_seven_halberdiers")
    run.call(run.town_action("back"), "explicit_back_after_second_purchase")
    source = next(slot for slot in run.armies().source.slots if slot.creature == 1)
    for target in range(2, 7):
        run.transfer(source.slot, target, 1, "native_partial_split_to_explicit_slot_" + str(target))
    rows = run.armies()
    base.ensure(all(slot.count > 0 for slot in rows.destination.slots), "Hero capacity check requires all seven slots occupied")
    run.reject(run.town_action("transfer_stack", selection=rows.source.slots[source.slot].ref,
                              destination=rows.destination.slots[0].ref, count=1),
               "unlike_creature_full_army_target_rejected_before_send")
    run.transfer(source.slot, 6, rows.source.slots[source.slot].count, "native_exact_merge_into_selected_full_army_slot")
    base.ensure(all(slot.count == 0 for slot in run.armies().source.slots), "Final town source is not empty")
    final_rows = run.armies()
    run.call(run.action("close"), "explicit_close_after_transfers")
    base.ensure(run.inspect_request().observation.window is None, "Closed town retained private army rows")
    run.open_town()
    base.ensure(run.armies() == final_rows, "Reopened town did not show exact transferred armies")
    run.call(run.action("close"), "explicit_close_reopened_town")
    run.call(run.action("open_hero", subject=hero.handle), "explicit_reopen_same_hero_after_transfer")
    final_army = run.inspect_request().observation.window.army
    expected = {(slot.slot, slot.creature, slot.count) for slot in final_rows.destination.slots if slot.count}
    base.ensure({(slot.slot, slot.creature, slot.count) for slot in final_army} == expected,
                "Hero window does not match the actual transferred visitor army")
    base.ensure(sum(slot.count for slot in final_army) - sum(slot.count for slot in initial_army) == 14
                and run.history(color)[:len(initial_prefix)] == initial_prefix,
                "Transfer refreshed old hero-window history or changed the total quantity")
    run.check("native_closed_window_immutable_and_explicit_reopen_fresh", "passed",
              initial_units=sum(slot.count for slot in initial_army), final_units=sum(slot.count for slot in final_army))
    run.call(run.action("close"), "explicit_final_close")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("transfer", "crash-after-ack"), default="transfer")
    parser.add_argument("--player", choices=base.COLORS, default="red")
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=98101)
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--case-seconds", type=float, default=120)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147483647 or not 1 <= args.max_days <= 20 or not 1 <= args.max_decisions <= 500 or any(
            not math.isfinite(value) or not 0 < value <= 180 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, choices, days and finite time limits")
    for name in ("map", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; never overwrite native evidence")
    boundary.source_contents(args.map)
    args.interface = contract.ARMIES_ENVIRONMENT
    env = base.load("heroes_transfer_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_transfer_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    base.write(args.output / "plan.json", {
        "schema": "heroes-town-transfer-validation-v1", "case": args.case, "player": args.player,
        "interface": args.interface, "seed": args.seed, "map": str(args.map), "map_sha256": base.sha(args.map),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary_sha256": base.sha(args.binary), "probe_sha256": base.sha(Path(__file__)),
        "shared_probe_sha256": base.sha(Path(base.__file__)), "town_probe_sha256": base.sha(Path(town.__file__)),
        "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog")},
        "training_input": False, "training_performed": False, "ordinary_outcome_claim": False,
        "rejection_scope": "Invalid choices rejected by Python before send; no native rejected-ack claim"})
    run = TransferRun(args.case, args, env, contract, auditor, args.seed)
    error = None
    try:
        run.start()
        diagnostic(run, args.case == "crash-after-ack")
        run.finish()
        run.persist_and_audit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("native_transfer_execution", "failed", error=error)
        try:
            run.finish()
            if run.end is not None and not run.audit_attempted:
                run.persist_and_audit()
        except Exception as cleanup:
            run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup).__name__}: {cleanup}")
    finally:
        run.env.close()
    report = run.summary(error)
    report.update(interface=args.interface, training_input=False, ordinary_outcome_claim=False)
    base.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "error": error, "report": str(args.output / "report.json")}))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
