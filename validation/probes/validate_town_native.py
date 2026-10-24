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
    def start(self):
        super().start()
        while self.state.color != self.args.player:
            self.call(self.action("end_turn"), "explicit_turn_to_selected_diagnostic_player")

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

    def open_recruitment(self, control_ref=None):
        panel = self.panel("overview")
        controls = [control for control in panel.controls if control.panel == "recruitment"]
        control = next((item for item in controls if item.ref == control_ref), None) if control_ref else next(iter(controls), None)
        base.ensure(control is not None, "Actual town overview has no selected dwelling control")
        self.call(self.town_action("open_panel", panel="recruitment", selection=control.ref),
                  "explicit_open_dwelling_recruitment")
        recruitment = self.panel("recruitment").recruitment
        base.ensure(recruitment.control_ref == control.ref and recruitment.destination.role == "town_garrison",
                    "Baseline recruitment did not target its actual town garrison")
        return recruitment

    def select_creature(self, creature):
        recruitment = self.panel("recruitment").recruitment
        card = next((offer for offer in recruitment.offers if offer.creature == creature), None)
        base.ensure(card is not None, "Requested diagnostic creature card is not displayed")
        _, additions, result = self.call(self.town_action("select_recruit", selection=card.ref), "explicit_select_creature_card")
        base.ensure(result is not None and result.status == "selected" and result.accepted
                    and not any(frame.kind in ("issued", "acknowledged") for frame in additions),
                    "Card selection issued a game command or lacks its adapter result")
        selected = self.panel("recruitment").recruitment
        base.ensure(selected.selected_offer_ref == card.ref and selected.stock == recruitment.stock,
                    "Card selection changed shared stock or selected a different card")
        return selected

    def purchase(self, panel, count):
        return self.town_action("recruit", selection=panel.selected_offer_ref, count=count,
                                destination=panel.destination.ref)

    def read_hero_army(self):
        frame = self.inspect_request()
        base.ensure(frame.observation.context.kind == "map" and len(frame.observation.heroes) == 1,
                    "Hero comparison requires the unchanged one-hero baseline")
        self.call(self.action("open_hero", subject=frame.observation.heroes[0].handle), "explicit_read_hero_for_destination_check")
        army = self.inspect_request().observation.window.army
        self.call(self.action("close"), "close_hero_after_destination_check")
        return army


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


def successful_purchase(run, before_army, panel, count, label):
    before = run.inspect_request()
    selected = next(offer for offer in panel.offers if offer.ref == panel.selected_offer_ref)
    _, additions, result = run.call(run.purchase(panel, count), label)
    request_evidence(run, additions, result, "recruited")
    overview = run.panel("overview")
    base.ensure(overview.recruitment is None, "Closed recruitment panel continued to refresh stock/cost")
    after = run.inspect_request()
    cost = tuple(price * count for price in panel.unit_cost)
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == cost,
                "Recruitment debit differs from explicit count times selected unit price")
    expected = {stack.slot: (stack.creature, stack.count) for stack in before_army}
    same = next((slot for slot, (creature, _) in expected.items() if creature == selected.creature), None)
    slot = same if same is not None else next((slot for slot in range(7) if slot not in expected), None)
    base.ensure(slot is not None, "Successful diagnostic purchase had no native destination slot")
    expected[slot] = (selected.creature, count + (expected[slot][1] if same is not None else 0))
    army = after.observation.window.army
    base.ensure({stack.slot: (stack.creature, stack.count) for stack in army} == expected,
                "Recruitment changed another stack, destination or quantity")
    refreshed = run.open_recruitment(panel.control_ref)
    base.ensure(refreshed.stock == panel.stock - count, "Reopened shared stock did not decrease by exact selected count")
    run.check(label, "passed", creature=selected.creature, count=count, destination=panel.destination.role,
              slot=slot, cost=list(cost), stock_before=panel.stock, stock_after=refreshed.stock,
              army_before=[(stack.slot, stack.creature, stack.count) for stack in before_army],
              army_after=[(stack.slot, stack.creature, stack.count) for stack in army])
    return army, refreshed


def recruitment(run, *, upgraded=False):
    hero_army = run.read_hero_army()
    run.open_town()
    if upgraded:
        hall = run.open_hall()
        choice = next((choice for choice in hall.buildings if choice.status == "allowed"
                       and ".dwellingUpLvl1." in choice.name_text_id), None)
        base.ensure(choice is not None, "Upgrade setup is not offered in unchanged baseline")
        details = run.open_building(choice)
        _, additions, result = run.call(run.town_action("construct", selection=details.ref), "native_upgrade_for_shared_stock_test")
        request_evidence(run, additions, result, "built")
    before_army = run.inspect_request().observation.window.army
    panel = run.open_recruitment()
    base.ensure({card.creature for card in panel.offers} == ({0, 1} if upgraded else {0}),
                "Level-one native cards do not match the actual authored/build state")
    panel = run.select_creature(0)
    base.ensure(panel.slider_max >= 2, "Baseline lacks stock/resources for two real diagnostic purchases")
    run.call(run.purchase(panel, panel.slider_max + 1), "out_of_slider_quantity_rejected", expect_rejection=True)
    run.call(run.town_action("recruit", selection=panel.selected_offer_ref, count=1, destination="foreign-army"),
             "foreign_destination_rejected", expect_rejection=True)
    run.call(run.town_action("recruit", selection="foreign-offer", count=1, destination=panel.destination.ref),
             "foreign_creature_offer_rejected", expect_rejection=True)
    before = run.history(run.state.color)
    request = run.env.request
    for invalid in (0, -1, True, 1.5):
        try:
            run.purchase(panel, invalid)
        except ValueError:
            pass
        else:
            raise base.ProbeFailure("Invalid exact count was accepted by typed action")
    base.ensure(run.history(run.state.color) == before and run.env.request is request, "Count type rejection changed request/history")
    run.check("invalid_count_types_rejected_before_submission", "passed", invalid=[0, -1, True, 1.5])
    army, _ = successful_purchase(run, before_army, panel, 1, "exact_native_recruitment")
    panel = run.select_creature(0)
    army, _ = successful_purchase(run, army, panel, 1, "repeat_recruitment_merges_same_creature")
    if upgraded:
        panel = run.select_creature(1)
        base_price = next(check["cost"] for check in run.checks if check["check"] == "exact_native_recruitment")
        base.ensure(list(panel.unit_cost) != base_price, "Changed creature card did not change its selected unit price")
        army, _ = successful_purchase(run, army, panel, 1, "upgraded_card_exact_purchase_shared_stock")
    run.unwind()
    base.ensure(run.read_hero_army() == hero_army, "Town recruitment changed the separate hero army")
    run.open_town()
    base.ensure(run.inspect_request().observation.window.army == army, "Fresh town overview changed after hero comparison")
    run.open_recruitment()
    run.finish()
    base.ensure(run.end.status == "truncated", "Recruitment-panel cutoff was not an unscored truncation")
    run.check("destination_hero_unchanged_and_recruitment_cutoff", "passed")


def capacity(run):
    base.ensure(run.args.fixture_kind in ("capacity-full", "capacity-merge"), "Capacity check requires declared isolated fixture")
    run.open_town()
    army = run.inspect_request().observation.window.army
    base.ensure(len(army) == 7 and len({stack.creature for stack in army}) == 7, "Fixture did not create seven distinct native stacks")
    run.open_recruitment()
    panel = run.select_creature(0)
    base.ensure(panel.slider_max > 0, "Capacity condition must not be stock/resource-limited")
    if run.args.fixture_kind == "capacity-merge":
        base.ensure(any(stack.creature == 0 for stack in army), "Merge fixture lacks the offered creature")
        successful_purchase(run, army, panel, 1, "full_army_same_creature_native_merge")
    else:
        base.ensure(all(stack.creature != 0 for stack in army), "Full fixture unexpectedly has a merge slot")
        resources = run.inspect_request().observation.resources
        _, additions, result = run.call(run.purchase(panel, 1), "native_no_capacity_response")
        base.ensure(result is not None and result.status == "no_capacity" and not result.accepted,
                    "Native no-room branch was not recorded distinctly")
        base.ensure(not any(frame.kind in ("issued", "acknowledged") for frame in additions), "No-room branch sent a purchase")
        base.ensure(any(frame.kind == "consumed" and frame.consumer == "adapter" for frame in additions),
                    "No-room UI operation lacks adapter consumption")
        base.ensure(any(frame.event and frame.event.notice and any(token.text_id == "core.genrltxt.17"
                    for token in frame.event.notice.text) for frame in additions), "Visible native no-room notice is missing")
        base.ensure(run.panel("recruitment").recruitment == panel
                    and run.inspect_request().observation.resources == resources, "No-room branch changed price/stock/resources")
        run.call(run.town_action("back"), "back_after_no_room")
        base.ensure(run.inspect_request().observation.window.army == army, "No-room branch changed a stack")
        run.check("native_no_capacity_is_consumed_without_purchase", "passed", stock=panel.stock,
                  resources=list(resources), destination=panel.destination.role)


def recruitment_budget(run):
    """Reach a real resource-limited slider by explicit native purchases/builds."""
    owner = run.state.player
    purchases = 0
    while run.live():
        day = run.inspect_request().day
        run.open_town()
        hall = run.open_hall()
        viewed = []
        for choice in hall.buildings:
            if choice.status != "allowed":
                continue
            viewed.append(run.open_building(choice))
            run.call(run.town_action("back"), "back_after_budget_setup_price")
        if viewed:
            details = max(viewed, key=lambda value: (sum(value.cost), value.ref))
            choice = next(choice for choice in run.panel("hall").buildings if choice.ref == details.ref)
            run.open_building(choice)
            _, additions, result = run.call(run.town_action("construct", selection=details.ref), "native_recruitment_budget_setup_build")
            request_evidence(run, additions, result, "built")
        else:
            run.call(run.town_action("back"), "close_hall_without_enabled_build")
        controls = [control for control in run.panel("overview").controls if control.panel == "recruitment"]
        for control in controls:
            army = run.inspect_request().observation.window.army
            panel = run.open_recruitment(control.ref)
            panel = run.select_creature(panel.offers[0].creature)
            resources = run.inspect_request().observation.resources
            affordability = min((amount // price for amount, price in zip(resources, panel.unit_cost) if price), default=2**31 - 1)
            base.ensure(panel.slider_max == min(panel.stock, affordability), "Slider differs from displayed stock/affordability")
            if panel.stock > affordability:
                run.call(run.purchase(panel, affordability + 1), "resource_limited_recruit_count_rejected", expect_rejection=True)
                run.check("native_recruitment_affordability", "passed", day=day, stock=panel.stock,
                          unit_cost=list(panel.unit_cost), resources=list(resources), slider_max=panel.slider_max,
                          invalid_count=affordability + 1, prior_real_purchases=purchases)
                return
            if panel.slider_max > 0:
                successful_purchase(run, army, panel, panel.slider_max, "explicit_native_stock_purchase_for_budget_test")
                purchases += 1
            else:
                run.call(run.purchase(panel, 1), "zero_stock_positive_count_rejected", expect_rejection=True)
            run.call(run.town_action("back"), "close_dwelling_after_budget_test")
        run.next_own_day(owner, day)
    raise base.ProbeFailure("Resource-limited recruitment was not reached within declared bounds")


def natural_boundary(run):
    base.ensure(run.args.fixture_kind == "natural-townless", "Natural boundary requires the declared townless fixture")
    while run.live():
        run.call(run.action("end_turn"), "explicit_turn_for_native_seven_day_rule")
    base.ensure(run.end.status == "completed" and run.end.outcomes == {"red": "loss", "blue": "win"}
                and run.end.rewards == {"red": -1, "blue": 1}, "Native terminal did not yield paired verified outcomes")
    run.check("native_v3_terminal_finalization", "passed", outcomes=run.end.outcomes, rewards=run.end.rewards)


def crash_boundary(run):
    run.open_town()
    run.open_hall()
    choice = next(choice for choice in run.panel("hall").buildings if choice.status == "allowed")
    details = run.open_building(choice)
    _, additions, result = run.call(run.town_action("construct", selection=details.ref), "native_build_before_process_crash")
    request_evidence(run, additions, result, "built")
    run.kill_owned_process()
    run.finish()
    base.ensure(run.end.status == "failed" and all(value is None for value in run.end.bootstrap.values()),
                "Completed process crash became an eligible cutoff")
    run.check("native_v3_crash_after_build_is_unscored", "passed")


def fixture_input(path):
    from validation.probes import create_boundary_fixture as boundary
    from validation.probes import create_town_fixtures as capacity
    path = Path(path).resolve()
    manifest = boundary.decode(path.read_bytes())
    source, target = Path(manifest["source"]["path"]), Path(manifest["fixture"]["path"])
    base.ensure(source.is_absolute() and target.is_absolute(), "Fixture paths must be absolute")
    base.ensure(base.sha(source) == manifest["source"]["sha256"]
                and base.sha(target) == manifest["fixture"]["sha256"], "Fixture bytes disagree with declaration")
    kind = manifest.get("kind")
    if manifest.get("schema") == capacity.SCHEMA:
        capacity.verify_fixture(target, kind=kind, player=manifest["player"], baseline=source)
    elif kind in boundary.SCHEMAS and manifest.get("schema") == boundary.SCHEMAS[kind]:
        boundary.verify_fixture(target, kind=kind, baseline=source)
    else:
        raise base.ProbeFailure("Unknown diagnostic fixture schema")
    return kind, target, manifest


CASES = {"construction": construction, "construction-budget": construction_budget,
         "recruitment": recruitment, "recruitment-upgrade": lambda run: recruitment(run, upgraded=True),
         "recruitment-budget": recruitment_budget,
         "capacity": capacity, "natural-boundary": natural_boundary, "crash-boundary": crash_boundary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="construction")
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--fixture", type=Path, help="Declared isolated fixture; overrides --map after exact verification")
    parser.add_argument("--player", choices=("red", "blue"), default="red")
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
    args.fixture_kind = None
    fixture = None
    if args.fixture is not None:
        args.fixture = args.fixture.expanduser().resolve()
        args.fixture_kind, args.map, fixture = fixture_input(args.fixture)
        if fixture.get("player", args.player) != args.player:
            parser.error("The diagnostic player must match the fixture player")
    else:
        from validation.probes.create_boundary_fixture import source_contents
        source_contents(args.map)  # undeclared map modifications are not diagnostic evidence
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
        "fixture_manifest": str(args.fixture) if args.fixture else None,
        "fixture_manifest_sha256": base.sha(args.fixture) if args.fixture else None,
        "fixture": fixture,
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
