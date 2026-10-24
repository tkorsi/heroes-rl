#!/usr/bin/env python3
"""Bounded real-VCMI town-interface diagnostics, excluded from training.

Choices deliberately exercise mechanics using only the acting player's displayed
panel data. They are neither a strategy nor a source of training experience.
Every attempted run, including failures, retains its records and independent
history audit. No engine state, outcome, request or acknowledgement is emulated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.learner import contextual as contract
from validation.probes import validate_contextual_native as base


class TownRun(base.CaseRun):
    def expected_offers(self, frame):
        return contract.Offers.from_observation(frame.observation, schema=contract.TOWN_ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return contract.Action(kind, schema=contract.TOWN_ACTION_SCHEMA, **parameters)

    def town_action(self, kind, **parameters):
        frame = self.inspect_request()
        base.ensure(frame.observation.context.kind == "town", "Town action outside town")
        return self.action(kind, subject=frame.observation.context.subject, **parameters)

    def panel(self, expected):
        frame = self.inspect_request()
        window = frame.observation.window
        base.ensure(window is not None and window.context.kind == "town"
                    and window.context.panel == expected and window.town_panel.kind == expected,
                    "Wrong active town panel")
        if expected != "overview":
            base.ensure(window.army is None and window.faction is None,
                        "Overlay refreshed underlying town army/faction")
        return window.town_panel

    def open_town(self):
        frame = self.inspect_request()
        base.ensure(frame.observation.towns, "Diagnostic requires an owned town")
        ref = frame.observation.towns[0].handle
        self.call(self.action("open_town", subject=ref), "explicit_open_owned_town")
        panel = self.panel("overview")
        base.ensure(not panel.buildings and panel.building is None,
                    "Overview exposed unopened hall or building details")
        return ref

    def open_hall(self):
        self.call(self.town_action("open_panel", panel="hall"), "explicit_open_hall")
        panel = self.panel("hall")
        base.ensure(panel.buildings and panel.building is None,
                    "Hall lacks actual displayed choices or exposes selected details prematurely")
        return panel

    def open_building(self, choice):
        self.call(self.town_action("open_building", selection=choice.ref), "explicit_open_building_details")
        details = self.panel("building").building
        base.ensure(details.ref == choice.ref and details.name_text_id == choice.name_text_id,
                    "Selected details do not match displayed hall choice")
        base.ensure((details.requirements is not None) == (details.status in ("prerequires", "missing_base")),
                    "Requirements visibility differs from native status")
        return details

    def unwind(self):
        while self.inspect_request().observation.context.kind == "town":
            context = self.inspect_request().observation.context
            self.call(self.action("close") if context.panel == "overview" else self.town_action("back"),
                      "explicit_close" if context.panel == "overview" else "explicit_back")

    def next_own_day(self, player, day):
        self.unwind()
        while self.live():
            frame = self.inspect_request()
            if self.state.player == player and frame.day > day:
                return frame
            self.call(self.action("end_turn"), "explicit_turn_advance_for_daily_rule")
        raise base.ProbeFailure("Native game ended before next owned decision")


def request_evidence(run, additions, result, status):
    issued = [frame for frame in additions if frame.kind == "issued"]
    accepted = [frame for frame in additions if frame.kind == "acknowledged" and frame.accepted]
    base.ensure(result is not None and result.accepted and result.status == status,
                "Native operation lacks the expected successful result")
    base.ensure(len(issued) == len(accepted) == 1, "Operation did not issue and acknowledge exactly one native request")


def construction(run):
    owner = run.state.player
    run.open_town()
    hall = run.open_hall()
    run.call(run.town_action("open_building", selection="unoffered-building"),
             "foreign_building_ref_rejected_without_consumption", expect_rejection=True)
    disabled = next((choice for choice in hall.buildings if choice.status == "forbidden"), None)
    base.ensure(disabled is not None, "Authored tavern restriction is absent from native hall")
    forbidden = run.open_building(disabled)
    run.call(run.town_action("construct", selection=forbidden.ref),
             "forbidden_build_control_rejected_without_consumption", expect_rejection=True)
    run.call(run.town_action("back"), "back_from_disabled_detail")
    hall = run.panel("hall")
    prerequisite = next((choice for choice in hall.buildings if choice.status in ("prerequires", "missing_base")), None)
    base.ensure(prerequisite is not None, "No native unmet-requirement detail available")
    details = run.open_building(prerequisite)
    run.check("native_conditional_requirement_panel", "passed", content_name=details.name_text_id, native_status=details.status)
    run.call(run.town_action("back"), "back_from_requirement_detail")
    choices = run.panel("hall").buildings
    chosen = next((choice for choice in choices if choice.status == "allowed"
                   and ".dwellingUpLvl1." in choice.name_text_id), None)
    base.ensure(chosen is not None, "The unchanged baseline has no enabled level-one dwelling upgrade")
    details = run.open_building(chosen)
    before = run.inspect_request()
    old_history = run.history(run.state.color)
    _, additions, result = run.call(run.town_action("construct", selection=details.ref), "exact_native_build")
    request_evidence(run, additions, result, "built")
    run.panel("overview")
    after = run.inspect_request()
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == details.cost,
                "Actual construction resource debit differs from displayed price")
    base.ensure(run.history(run.state.color)[:len(old_history)] == old_history, "Build mutated prior panel history")
    run.check("exact_build_cost_and_native_request", "passed", content_name=details.name_text_id,
              cost=list(details.cost), resources_before=list(before.observation.resources),
              resources_after=list(after.observation.resources), returned_to="overview")
    run.call(run.town_action("construct", selection=details.ref),
             "repeated_build_from_closed_details_rejected", expect_rejection=True)
    hall = run.open_hall()
    same = next((choice for choice in hall.buildings if choice.ref == details.ref), None)
    base.ensure(same is not None and same.status == "already_present", "Built upgrade absent from reopened hall")
    run.open_building(same)
    run.call(run.town_action("construct", selection=same.ref), "already_built_rejected", expect_rejection=True)
    run.call(run.town_action("back"), "back_after_repeat_check")
    capped = next((choice for choice in run.panel("hall").buildings if choice.status == "cant_build_today"), None)
    base.ensure(capped is not None, "No daily construction cap observed after native build")
    capped_details = run.open_building(capped)
    run.call(run.town_action("construct", selection=capped.ref), "daily_build_cap_rejected", expect_rejection=True)
    run.check("native_daily_build_cap", "passed", ref=capped.ref, content_name=capped_details.name_text_id)
    day = run.inspect_request().day
    run.next_own_day(owner, day)
    base.ensure(run.inspect_request().observation.window is None, "Closed panel refreshed on subsequent day")
    run.open_town()
    hall = run.open_hall()
    refreshed = next((choice for choice in hall.buildings if choice.ref == capped.ref), None)
    base.ensure(refreshed is not None and refreshed.status != "cant_build_today", "Daily build cap did not reset")
    run.open_building(refreshed)
    run.check("closed_panel_fresh_reopen_and_daily_reset", "passed", old_day=day,
              new_day=run.inspect_request().day, new_status=refreshed.status)
    # Deliberately request cutoff inside details, requiring explicit native unwind
    # outside both immutable player learner prefixes.
    run.finish()
    base.ensure(run.end.status == "truncated", "Panel cutoff was not an unscored truncation")
    run.check("paired_cutoff_from_building_details", "passed")


def construction_budget(run):
    """Spend through real UI requests until a native affordability control appears."""
    owner = run.state.player
    built = []
    while run.live():
        run.open_town()
        hall = run.open_hall()
        poor = next((choice for choice in hall.buildings if choice.status == "no_resources"), None)
        if poor is not None:
            details = run.open_building(poor)
            resources = run.inspect_request().observation.resources
            base.ensure(any(cost > amount for cost, amount in zip(details.cost, resources)),
                        "Native insufficient-resource status is not supported by displayed price/resources")
            run.call(run.town_action("construct", selection=poor.ref),
                     "insufficient_construction_resources_rejected", expect_rejection=True)
            run.check("native_construction_affordability", "passed", price=list(details.cost),
                      resources=list(resources), prior_real_builds=built)
            return
        enabled = [choice for choice in hall.buildings if choice.status == "allowed"]
        base.ensure(enabled, "No native unaffordable or enabled building remained in bounded fixture")
        viewed = []
        for choice in enabled:
            details = run.open_building(choice)
            viewed.append(details)
            run.call(run.town_action("back"), "back_after_displayed_price_inspection")
        chosen = max(viewed, key=lambda details: (sum(details.cost), details.ref))
        # A deterministic evaluator selects a displayed costly operation solely
        # to exercise affordability. This is never a learner policy/example.
        choice = next(choice for choice in run.panel("hall").buildings if choice.ref == chosen.ref)
        details = run.open_building(choice)
        before = run.inspect_request()
        _, additions, result = run.call(run.town_action("construct", selection=details.ref), "native_affordability_setup_build")
        request_evidence(run, additions, result, "built")
        after = run.inspect_request()
        base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == details.cost,
                    "Affordability setup build charged a different price")
        built.append({"name": details.name_text_id, "day": before.day, "cost": list(details.cost)})
        run.next_own_day(owner, before.day)
    raise base.ProbeFailure("Affordability condition was not reached within declared bounds")


CASES = {"construction": construction, "construction-budget": construction_budget}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="construction")
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--max-days", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=200)
    parser.add_argument("--case-seconds", type=float, default=60)
    parser.add_argument("--watchdog", type=float, default=20)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (not 1 <= args.seed <= 2147483647 or not 1 <= args.max_days <= 40
            or not 1 <= args.max_decisions <= 1000 or any(not math.isfinite(value) or not 0 < value <= 120
            for value in (args.case_seconds, args.watchdog, args.startup_seconds))):
        parser.error("Use bounded positive seeds, days, decisions and finite time limits")
    for name in ("map", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; never overwrite native evidence")
    args.interface = contract.TOWN_ENVIRONMENT
    env = base.load("heroes_town_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_town_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    base.write(args.output / "plan.json", {
        "schema": "heroes-town-native-validation-v1", "case": args.case, "interface": args.interface,
        "engine_seed": args.seed, "map": str(args.map), "map_sha256": base.sha(args.map),
        "binary_sha256": base.sha(args.binary), "probe_sha256": base.sha(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "bounds": {name: getattr(args, name) for name in ("max_days", "max_decisions", "case_seconds", "watchdog")},
        "training_performed": False, "training_input": False, "random_policy_outcome_claim": False})
    run = TownRun(args.case, args, env, contract, auditor, args.seed)
    error = None
    try:
        run.start()
        CASES[args.case](run)
        run.finish()
        run.persist_and_audit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("native_town_execution", "failed", error=error)
        try:
            run.finish()
            if run.end is not None and not run.audit_attempted:
                run.persist_and_audit()
        except Exception as cleanup:
            run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup).__name__}: {cleanup}")
    finally:
        run.env.close()
    report = run.summary(error)
    report.update(interface=args.interface, training_input=False, random_policy_outcome_claim=False)
    base.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "report": str(args.output / "report.json")}, indent=2))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
