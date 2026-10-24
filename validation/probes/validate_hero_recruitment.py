#!/usr/bin/env python3
"""Bounded native Tavern mechanics validation, never a strategy or learner data.

The operator chooses explicit actions solely to exercise displayed controls and
native effects. Every attempted run is retained, including technical failures.
This probe neither trains a policy nor changes game rules or outcome evidence.
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

from experiments.explorer.create_hero_scenario import verify_hero_scenario
from experiments.learner import contextual as contract
from experiments.learner.first_run import implementation_identity
from validation.probes import validate_contextual_native as base
from validation.probes import validate_town_native as town


def exact_hire(before, after, action, additions, result):
    """Bind one native receipt to its explicit choice, exact gold and new visitor."""
    context = before.observation.context
    panel = before.observation.window.town_panel.tavern
    base.ensure(context.kind == "town" and context.panel == "tavern" and panel is not None,
                "Hire requires its previously opened Tavern")
    base.ensure(action.kind == "hire_hero" and action.subject == context.subject
                and action.selection == panel.selected_offer_ref and panel.recruit_status == "allowed",
                "Hire does not bind its selected displayed offer")
    consumed = [f for f in additions if f.kind == "consumed" and f.decision_seq == before.seq]
    issued = [f for f in additions if f.kind == "issued" and f.decision_seq == before.seq]
    acknowledged = [f for f in additions if f.kind == "acknowledged" and f.decision_seq == before.seq]
    results = [f for f in additions if f.kind == "result" and f.decision_seq == before.seq and f.result == result]
    base.ensure(result is not None and result.status == "hired" and result.accepted
                and len(consumed) == len(issued) == len(acknowledged) == len(results) == 1
                and consumed[0].consumer == "engine" and acknowledged[0].accepted
                and issued[0].command_index == acknowledged[0].command_index
                and consumed[0].seq < issued[0].seq < acknowledged[0].seq < results[0].seq,
                "Hire lacks one consumed, issued, accepted and completed native request")
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources))
                == (0, 0, 0, 0, 0, 0, panel.gold_cost), "Hire changed resources beyond its displayed gold cost")
    previous = {hero.handle: hero for hero in before.observation.heroes}
    current = {hero.handle: hero for hero in after.observation.heroes}
    base.ensure(all(current.get(ref) == hero for ref, hero in previous.items())
                and len(current) == len(previous) + 1, "Hire replaced, changed or failed to add one owned hero")
    added = next(hero for ref, hero in current.items() if ref not in previous)
    base.ensure(before.observation.towns == after.observation.towns,
                "Hire changed an owned town identity or position")
    own_town = next((item for item in before.observation.towns if item.handle == context.subject), None)
    base.ensure(own_town is not None and added.position == own_town.position,
                "Hired hero did not appear at the selected town's visitor position")
    window = after.observation.window
    base.ensure(window is not None and window.context == contract.Context("town", context.subject, "overview")
                and window.town_panel.tavern is None and window.town_panel.armies is not None
                and window.town_panel.armies.destination is not None
                and window.town_panel.armies.destination.subject == added.handle,
                "Hire failed to close its Tavern and expose the actual new visiting hero")
    return {"offer_ref": action.selection, "selected_detail": asdict(panel.selected_detail),
            "gold_cost": panel.gold_cost, "new_hero": added.handle,
            "heroes_before": len(previous), "heroes_after": len(current),
            "decision_seq": before.seq, "consumed_seq": consumed[0].seq,
            "issued_seq": issued[0].seq, "acknowledged_seq": acknowledged[0].seq,
            "result_seq": results[0].seq, "native_rejection": False}


def finalize_diagnostic_failure(run, error):
    """Fail a still-live native attempt, without issuing cutoff unwind actions.

    A diagnostic assertion or Python exception invalidates the attempt. The
    environment's existing bounded shutdown/finalizer records the real streams
    and process result. Already finalized evidence is never rewritten.
    """
    try:
        existing = getattr(run.env, "result", None)
        if existing is not None:
            run.end = run.state = existing
        elif getattr(run.env, "process", None) is not None:
            run.env._note_problem("protocol_error", "Mechanics diagnostic failed: " + error)
            run.end = run.state = run.env._finish(None)
            base.ensure(run.end.status == "failed"
                        and all(value is None for value in run.end.rewards.values())
                        and all(value is None for value in run.end.outcomes.values())
                        and all(value is None for value in run.end.bootstrap.values()),
                        "Diagnostic failure was not finalized as an unscored native failure")
        elif run.state is not None and run.state.done:
            run.end = run.state
        else:
            raise base.ProbeFailure("No native process or finalized episode is available; preserve raw startup evidence")
        if run.end is not None and not run.audit_attempted:
            run.persist_and_audit()
    except Exception as cleanup:
        run.check("failure_evidence_finalization", "failed", original_error=error,
                  error=f"{type(cleanup).__name__}: {cleanup}")


def close_diagnostic(run):
    """Retain secondary close errors instead of hiding the diagnostic report."""
    try:
        run.env.close()
    except Exception as cleanup:
        run.check("failure_process_cleanup", "failed", error=f"{type(cleanup).__name__}: {cleanup}")


class HeroRun(town.TownRun):
    def expected_offers(self, frame):
        return contract.Offers.from_observation(frame.observation, schema=contract.HEROES_ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return contract.Action(kind, schema=contract.HEROES_ACTION_SCHEMA, **parameters)

    def call(self, action, label, *, expect_rejection=False):
        base.ensure(self.live(), "Mechanics probe exhausted its declared bounds")
        return super().call(action, label, expect_rejection=expect_rejection)

    def reject(self, action, label):
        self.call(action, label, expect_rejection=True)
        self.checks[-1].update(rejection_layer="python_pre_send", native_issued=0, native_rejected_ack=False)

    def open_tavern(self):
        overview = self.panel("overview")
        base.ensure(any(control.panel == "tavern" for control in overview.controls),
                    "The real overview did not expose a built Tavern")
        self.call(self.town_action("open_panel", panel="tavern"), "explicit_open_tavern")
        return self.panel("tavern").tavern

    def select_hero(self, ref):
        before = self.panel("tavern").tavern
        frame, additions, result = self.call(self.town_action("select_hero", selection=ref),
                                             "explicit_select_displayed_hero")
        selected = self.panel("tavern").tavern
        base.ensure(result is not None and result.status == "selected" and result.accepted
                    and not any(item.kind in ("issued", "acknowledged") for item in additions)
                    and selected.selected_offer_ref == ref and selected.offers == before.offers
                    and frame.observation.resources == self.inspect_request().observation.resources,
                    "Tavern selection changed offers/resources, issued a packet or selected another card")
        return selected

    def depart(self, ref, label):
        observation = self.inspect_request().observation
        base.ensure(observation.context.kind == "map", "Explicit diagnostic movement requires the map")
        hero = next((hero for hero in observation.heroes if hero.handle == ref), None)
        base.ensure(hero is not None, "Requested own hero is not current")
        if not hero.movement:
            self.next_own_day(self.state.player, self.inspect_request().day)
            observation = self.inspect_request().observation
            hero = next(hero for hero in observation.heroes if hero.handle == ref)
        occupied = {item.position for item in observation.heroes if item.handle != ref}
        adjacent = sorted(tile.position for tile in observation.tiles if not tile.blocked and not tile.visitable
                          and tile.position not in occupied and tile.position[2] == hero.position[2]
                          and max(abs(tile.position[i] - hero.position[i]) for i in (0, 1)) == 1)
        base.ensure(adjacent, "No displayed free adjacent tile for this mechanics check")
        target = adjacent[0]
        before, additions, result = self.call(self.action("move_to", subject=ref, x=target[0], y=target[1]), label)
        after = self.inspect_request().observation
        base.ensure(result is not None and result.status == "reached" and result.accepted
                    and next(item for item in after.heroes if item.handle == ref).position == target,
                    "Explicit selected-hero movement did not reach the displayed adjacent tile")
        base.ensure(all(next(item for item in after.heroes if item.handle == other.handle).position == other.position
                        for other in before.observation.heroes if other.handle != ref),
                    "Moving one owned hero changed another owned hero's position")
        self.check(label, "passed", hero=ref, before=list(hero.position), after=list(target),
                   native_issued=sum(item.kind == "issued" for item in additions))


def diagnostic(run):
    initial = run.inspect_request()
    color = run.state.color
    base.ensure(len(initial.observation.heroes) == len(initial.observation.towns) == 1,
                "The declared scenario must start with one native hero and town per side")
    original = initial.observation.heroes[0]
    original_prefix = run.history(color)
    own_town = run.open_town()
    overview = run.panel("overview")
    base.ensure(overview.armies.destination is not None and overview.armies.destination.subject == original.handle,
                "Initial hero was not the native town visitor")
    base.ensure(not any(control.panel == "tavern" for control in overview.controls),
                "Scenario unexpectedly granted a prebuilt Tavern")
    run.reject(run.town_action("open_panel", panel="tavern"), "unbuilt_tavern_rejected_before_send")
    hall = run.open_hall()
    choice = next((item for item in hall.buildings if ".tavern." in item.name_text_id), None)
    base.ensure(choice is not None and choice.status == "allowed", "Native Tavern construction is not available")
    detail = run.open_building(choice)
    before, additions, result = run.call(run.town_action("construct", selection=detail.ref), "explicit_construct_tavern")
    town.request_evidence(run, additions, result, "built")
    after = run.inspect_request()
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == detail.cost,
                "Native Tavern construction did not debit its exact displayed resource cost")
    run.check("native_tavern_construction_exact_cost", "passed", cost=list(detail.cost))
    tavern = run.open_tavern()
    base.ensure(tavern.recruit_status == "visitor_present" and len(tavern.offers) == 2,
                "Actual visitor did not disable hiring or native two-card offer is missing")
    run.reject(run.town_action("hire_hero", selection=tavern.selected_offer_ref),
               "visitor_occupied_hire_rejected_before_send")
    run.reject(run.town_action("select_hero", selection="not-displayed-offer"),
               "foreign_hero_offer_rejected_before_send")
    run.reject(run.action("select_hero", subject="not-owned-town", selection=tavern.selected_offer_ref),
               "foreign_town_hero_selection_rejected_before_send")
    other = next(item.ref for item in tavern.offers if item.ref != tavern.selected_offer_ref)
    selected = run.select_hero(other)
    run.check("native_displayed_selected_hero_detail", "passed", offer=other, details=asdict(selected.selected_detail),
              price=selected.gold_cost, native_status=selected.recruit_status)
    remembered_ref = selected.selected_offer_ref
    run.call(run.town_action("back"), "explicit_back_from_tavern")
    base.ensure(run.panel("overview").tavern is None, "Closed Tavern refreshed private offers")
    run.reject(run.town_action("hire_hero", selection=remembered_ref), "closed_tavern_hire_rejected_before_send")
    run.call(run.action("close"), "explicit_close_town_before_departure")
    base.ensure(run.inspect_request().observation.window is None, "Closed town retained private Tavern fields")
    run.depart(original.handle, "explicit_original_hero_departure")
    run.open_town()
    overview = run.panel("overview")
    base.ensure(overview.armies.destination is None, "Native visitor slot remained occupied after departure")
    town_army = overview.armies.source
    tavern = run.open_tavern()
    base.ensure(tavern.recruit_status == "allowed", "Native hire remains blocked after the real visitor departed")
    old_selection = tavern.selected_offer_ref
    other = next(item.ref for item in tavern.offers if item.ref != old_selection)
    selected = run.select_hero(other)
    run.reject(run.town_action("hire_hero", selection=old_selection), "old_card_selection_rejected_before_send")
    prior_prefix = run.history(color)
    action = run.town_action("hire_hero", selection=selected.selected_offer_ref)
    before, additions, result = run.call(action, "explicit_hire_selected_hero")
    evidence = exact_hire(before, run.inspect_request(), action, additions, result)
    base.ensure(run.panel("overview").armies.source == town_army, "Hiring changed the existing upper town army")
    base.ensure(run.history(color)[:len(prior_prefix)] == prior_prefix, "Hiring refreshed previously delivered Tavern history")
    run.check("native_exact_hero_hire_and_receipt", "passed", **evidence)
    hired = evidence["new_hero"]
    run.call(run.action("close"), "explicit_close_town_after_hire")
    for ref in (original.handle, hired):
        run.call(run.action("open_hero", subject=ref), "explicit_open_distinct_owned_hero_" + ref)
        window = run.inspect_request().observation.window
        base.ensure(window.context.subject == ref and window.context.kind == "hero" and window.army,
                    "Explicit hero window did not bind its selected owned hero")
        run.check("native_distinct_hero_window", "passed", hero=ref, army=[asdict(stack) for stack in window.army])
        run.call(run.action("close"), "explicit_close_distinct_hero_" + ref)
        run.depart(ref, "explicit_command_distinct_owned_hero_" + ref)
    base.ensure(run.history(color)[:len(original_prefix)] == original_prefix,
                "Multi-hero commands changed earlier observations")
    run.check("native_immutable_history_and_separate_hero_commands", "passed", owned_heroes=[original.handle, hired])
    run.open_town()
    current = run.open_tavern()
    base.ensure(selected.selected_offer_ref not in {item.ref for item in current.offers},
                "The hired hero remained an offered Tavern card")
    run.reject(run.town_action("hire_hero", selection=selected.selected_offer_ref),
               "hired_old_offer_rejected_before_send")
    # Deliberately finish with the Tavern open: paired cutoff continuation must
    # return to overview before closing, without refreshing the frozen prefix.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=base.COLORS, default="red")
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=102001)
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=100)
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
    verify_hero_scenario(args.map)
    args.interface = contract.HEROES_ENVIRONMENT
    env = base.load("heroes_hire_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_hire_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    implementation = implementation_identity()
    from experiments.headless.contextual_evidence import runtime_identity
    base.write(args.output / "plan.json", {
        "schema": "heroes-hero-recruitment-validation-v1", "player": args.player,
        "interface": args.interface, "seed": args.seed, "map": str(args.map), "map_sha256": base.sha(args.map),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation": implementation,
        "native_runtime": runtime_identity(args.binary.parent), "binary_sha256": base.sha(args.binary),
        "source_files": {str(path.relative_to(ROOT)): base.sha(path) for path in
                         (Path(__file__), Path(base.__file__), Path(town.__file__), Path(contract.__file__),
                          ROOT / "experiments/explorer/create_hero_scenario.py")},
        "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog")},
        "training_input": False, "training_performed": False, "ordinary_outcome_claim": False,
        "rejection_scope": "Invalid choices rejected by Python before send; no native rejected-ack claim"})
    run = HeroRun("recruitment", args, env, contract, auditor, args.seed)
    error = None
    try:
        run.start()
        diagnostic(run)
        run.finish()
        run.persist_and_audit()
        base.ensure(implementation_identity() == implementation, "Authored implementation changed during native validation")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("native_hero_recruitment_execution", "failed", error=error)
        finalize_diagnostic_failure(run, error)
    finally:
        close_diagnostic(run)
    report = run.summary(error)
    report.update(interface=args.interface, training_input=False, ordinary_outcome_claim=False,
                  scope_limits=["No replacement-after-battle-loss assertion in this mechanics case",
                                "No actual server rejected acknowledgement is claimed"])
    base.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "error": error, "report": str(args.output / "report.json")}))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
