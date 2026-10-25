"""Bounded native v6 army mechanics diagnostics; never policy or training data.

All setup uses explicit actions on the acting player's displayed controls. The
unchanged hero-management map supplies no diagnostic armies or resource grants.
Rejected Python calls, adapter notices and server acknowledgements are reported
separately. Native replay/tamper validation is a separate retained-artifact step.
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

from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner import contextual as c
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes import validate_town_native as town
from validation.probes.validate_hero_policy import hero_map_identity, retain_failed_evidence, wall_limit
from validation.probes.validate_hero_recruitment import HeroRun

INTERFACE = "adventure-army-management-v1"
ACTION_SCHEMA = "heroes-context-actions-v6"
CASES = ("stacks", "upgrades", "garrison", "exchange", "stack-cutoff", "exchange-cutoff")


def native_receipt(before, additions, result, expected):
    """One accepted native request is necessary, never sufficient, for success."""
    records = {kind: [frame for frame in additions if frame.kind == kind and frame.decision_seq == before.seq]
               for kind in ("consumed", "issued", "acknowledged", "result")}
    base.ensure(result is not None and result.accepted and result.status == expected
                and all(len(rows) == 1 for rows in records.values()), "Expected one complete native request receipt")
    consumed, issued, ack, end = (records[kind][0] for kind in ("consumed", "issued", "acknowledged", "result"))
    base.ensure(consumed.consumer == "engine" and ack.accepted and issued.command_index == ack.command_index
                and consumed.seq < issued.seq < ack.seq < end.seq and end.result == result,
                "Native request receipt is rejected, reordered or belongs to another command")
    return {"decision_seq": before.seq, "consumed_seq": consumed.seq, "issued_seq": issued.seq,
            "acknowledged_seq": ack.seq, "result_seq": end.seq, "native_rejection": False}


def rows(frame):
    window = frame.observation.window
    base.ensure(window is not None and window.armies, "Expected explicitly visible army rows")
    return tuple(window.armies)


def slots(armies):
    return {slot.ref: (army, slot) for army in armies for slot in army.slots}


def hero_state(heroes, speed_subjects=()):
    # The displayed movement maximum can legitimately change with army speed.
    # Current remaining movement, identity, position and garrison status cannot.
    return [{key: value for key, value in asdict(hero).items()
             if key != "movement_max" or hero.handle not in speed_subjects} for hero in heroes]


def expected_stack_operation(armies, action):
    """Independent exact slot arithmetic, including the final hero-army rule."""
    positions = slots(armies)
    base.ensure(action.selection in positions and action.destination in positions
                and action.selection != action.destination, "Slots must be distinct and currently displayed")
    source_army, source = positions[action.selection]
    destination_army, destination = positions[action.destination]
    base.ensure(source.count > 0, "Source stack is empty")
    expected = [asdict(army) for army in armies]
    mutable = {slot["ref"]: slot for army in expected for slot in army["slots"]}
    left, right = mutable[source.ref], mutable[destination.ref]
    if action.kind in ("split_stack", "merge_stacks"):
        count = action.count if action.kind == "split_stack" else source.count
        base.ensure(type(count) is int and 0 < count <= source.count
                    and destination.creature in (None, source.creature), "Invalid exact split or merge")
        if action.kind == "merge_stacks":
            base.ensure(destination.count > 0, "Merge requires two occupied matching stacks")
        base.ensure(destination.count + count <= 2**31 - 1, "Destination count exceeds native range")
        left["count"] -= count
        if not left["count"]: left["creature"] = None
        right["creature"], right["count"] = source.creature, destination.count + count
    elif action.kind == "swap_stacks":
        left["creature"], right["creature"] = destination.creature, source.creature
        left["count"], right["count"] = destination.count, source.count
    else:
        raise base.ProbeFailure("Not a declared stack operation")
    base.ensure(all(not army["requires_nonempty"] or any(slot["count"] for slot in army["slots"])
                    for army in expected), "Operation empties a hero's last required stack")
    return expected


def exact_stack_operation(before, after, action, additions, result):
    expected = expected_stack_operation(rows(before), action)
    locations = slots(rows(before))
    speed_subjects = {locations[ref][0].subject for ref in (action.selection, action.destination)}
    proof = native_receipt(before, additions, result,
                           {"split_stack": "split", "merge_stacks": "merged", "swap_stacks": "swapped"}[action.kind])
    base.ensure([asdict(army) for army in rows(after)] == expected, "Native operation changed an unexpected army, slot or count")
    base.ensure(before.observation.context == after.observation.context
                and before.observation.resources == after.observation.resources
                and hero_state(before.observation.heroes, speed_subjects) == hero_state(after.observation.heroes, speed_subjects)
                and before.observation.towns == after.observation.towns,
                "Stack operation changed context, resources or owned object identity")
    return {**proof, "operation": action.kind, "source": action.selection, "destination": action.destination,
            "count": action.count, "before": [asdict(army) for army in rows(before)], "after": expected}


def exact_upgrade(parent, before, after, action, additions, result):
    panel = before.observation.window.stack_panel
    base.ensure(panel is not None and panel.affordable is True and panel.selected_upgrade_ref == action.selection,
                "Upgrade must name its displayed affordable confirmation")
    offer = next(item for item in panel.offers if item.ref == action.selection)
    expected = [asdict(army) for army in rows(parent)]
    changed = [slot for army in expected for slot in army["slots"] if slot["ref"] == panel.source.ref]
    base.ensure(len(changed) == 1 and changed[0]["count"] == panel.source.count
                and changed[0]["creature"] == panel.source.creature, "Upgrade source differs from its parent window")
    changed[0]["creature"] = offer.creature
    proof = native_receipt(before, additions, result, "upgraded")
    base.ensure([asdict(army) for army in rows(after)] == expected
                and after.observation.context == panel.parent_context
                and parent.observation.resources == before.observation.resources
                and tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == panel.total_cost
                and hero_state(before.observation.heroes, (before.observation.context.subject,))
                    == hero_state(after.observation.heroes, (before.observation.context.subject,))
                and before.observation.towns == after.observation.towns,
                "Upgrade changed another stack, count, identity or resource amount")
    return {**proof, "source": panel.source.ref, "old_creature": panel.source.creature,
            "new_creature": offer.creature, "count": panel.source.count, "total_cost": list(panel.total_cost)}


def army_contents(army):
    return tuple((slot.slot, slot.creature, slot.count) for slot in army.slots)


def expected_simple_garrison_merge(town_army, visitor):
    """Exact native getSlotFor order for probes with a matching or free slot."""
    expected = [list(item) for item in army_contents(visitor)]
    for slot in town_army.slots:
        if not slot.count: continue
        target = next((item for item in expected if item[1] == slot.creature), None)
        if target is None: target = next((item for item in expected if item[2] == 0), None)
        base.ensure(target is not None, "This probe does not model native full-army duplicate consolidation")
        target[1], target[2] = slot.creature, target[2] + slot.count
    return tuple(tuple(item) for item in expected)


def exact_garrison(before, after, additions, result):
    old, new = rows(before), rows(after)
    old_by = {army.role: army for army in old}
    new_by = {army.role: army for army in new}
    proof = native_receipt(before, additions, result, "garrison_swapped")
    visitor, garrison = old_by.get("visiting_hero"), old_by.get("garrison_hero")
    expected_flags = {hero.handle: hero.in_garrison for hero in before.observation.heroes}
    if visitor is not None and garrison is None:
        base.ensure(set(new_by) == {"garrison_hero"} and new_by["garrison_hero"].subject == visitor.subject
                    and army_contents(new_by["garrison_hero"]) == expected_simple_garrison_merge(old_by["town_garrison"], visitor),
                    "Garrison entry did not preserve the exact native town-to-hero consolidation")
        expected_flags[visitor.subject] = True
        operation = "enter"
    elif garrison is not None and visitor is None:
        base.ensure(set(new_by) == {"town_garrison", "visiting_hero"}
                    and new_by["visiting_hero"].subject == garrison.subject
                    and army_contents(new_by["visiting_hero"]) == army_contents(garrison)
                    and not any(slot.count for slot in new_by["town_garrison"].slots),
                    "Garrison exit changed the hero army or left hidden town troops")
        expected_flags[garrison.subject] = False
        operation = "exit"
    else:
        base.ensure(visitor is not None and garrison is not None and set(new_by) == {"garrison_hero", "visiting_hero"},
                    "Expected occupied native garrison swap")
        base.ensure(new_by["garrison_hero"].subject == visitor.subject
                    and new_by["visiting_hero"].subject == garrison.subject
                    and army_contents(new_by["garrison_hero"]) == army_contents(visitor)
                    and army_contents(new_by["visiting_hero"]) == army_contents(garrison), "Occupied swap changed armies or heroes")
        expected_flags[visitor.subject], expected_flags[garrison.subject] = True, False
        operation = "swap"
    speed_subjects = {army.subject for army in old if army.role != "town_garrison"}
    before_heroes = hero_state(before.observation.heroes, speed_subjects)
    after_heroes = hero_state(after.observation.heroes, speed_subjects)
    for hero in before_heroes: hero["in_garrison"] = expected_flags[hero["handle"]]
    base.ensure(before_heroes == after_heroes and before.observation.resources == after.observation.resources
                and before.observation.context == after.observation.context
                and before.observation.towns == after.observation.towns, "Garrison changed unrelated owned state or resources")
    return {**proof, "operation": operation, "before": [asdict(army) for army in old], "after": [asdict(army) for army in new]}


class ArmyRun(HeroRun):
    def expected_offers(self, frame):
        return c.Offers.from_observation(frame.observation, schema=ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return c.Action(kind, schema=ACTION_SCHEMA, **parameters)

    def armies(self):
        return rows(self.inspect_request())

    def army(self, *, role=None, subject=None):
        matches = [army for army in self.armies() if (role is None or army.role == role)
                   and (subject is None or army.subject == subject)]
        base.ensure(len(matches) == 1, "Expected exactly one displayed army for this explicit operation")
        return matches[0]

    def manipulate(self, kind, source, destination, count=None):
        action = self.action(kind, subject=self.inspect_request().observation.context.subject,
                             selection=source, destination=destination, count=count)
        before, additions, result = self.call(action, "explicit_" + kind)
        self.check("native_" + kind + "_exact_effects", "passed",
                   **exact_stack_operation(before, self.inspect_request(), action, additions, result))

    def garrison(self):
        before, additions, result = self.call(self.town_action("swap_garrison"), "explicit_swap_garrison")
        self.check("native_garrison_exact_effects", "passed", **exact_garrison(before, self.inspect_request(), additions, result))

    def construct(self, text_fragment):
        hall = self.open_hall()
        choice = next((item for item in hall.buildings if text_fragment in item.name_text_id), None)
        base.ensure(choice is not None and choice.status == "allowed", "Declared native construction is not currently allowed")
        details = self.open_building(choice)
        before, additions, result = self.call(self.town_action("construct", selection=details.ref), "explicit_prepare_building")
        town.request_evidence(self, additions, result, "built")
        base.ensure(tuple(a - b for a, b in zip(before.observation.resources, self.inspect_request().observation.resources))
                    == details.cost, "Preparation construction cost changed")

    def buy(self, creature, count):
        self.open_recruitment()
        panel = self.select_creature(creature)
        base.ensure(panel.slider_max >= count, "Declared exact creature purchase is not affordable/available")
        before, additions, result = self.call(self.purchase(panel, count), "explicit_prepare_creature_purchase")
        town.request_evidence(self, additions, result, "recruited")
        base.ensure(tuple(a - b for a, b in zip(before.observation.resources, self.inspect_request().observation.resources))
                    == tuple(count * unit for unit in panel.unit_cost), "Preparation recruitment cost changed")
        # Accepting native recruitment closes the panel itself. A second back
        # action would address an absent control in the returned overview.
        self.panel("overview")

    def hire(self):
        panel = self.open_tavern()
        base.ensure(panel.recruit_status == "allowed", "Legitimate native second-hero hire is unavailable")
        action = self.town_action("hire_hero", selection=panel.selected_offer_ref)
        before, additions, result = self.call(action, "explicit_prepare_second_hero")
        proof = native_receipt(before, additions, result, "hired")
        after = self.inspect_request()
        old = {hero.handle for hero in before.observation.heroes}
        added = [hero for hero in after.observation.heroes if hero.handle not in old]
        base.ensure(len(added) == 1 and len(after.observation.heroes) == len(old) + 1
                    and self.army(role="visiting_hero").subject == added[0].handle
                    and tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources))
                    == (0, 0, 0, 0, 0, 0, panel.gold_cost), "Second hero hire changed the wrong visitor or resource amount")
        self.check("native_preparation_hire", "passed", hero=added[0].handle, **proof)
        return added[0].handle


def stacks_case(run):
    run.open_town()
    run.construct(".dwellingUpLvl1.")
    run.buy(0, 7)
    run.buy(1, 7)
    upper, hero = run.army(role="town_garrison"), run.army(role="visiting_hero")
    base_slot = next(slot.ref for slot in upper.slots if slot.creature == 0)
    upgraded = next(slot.ref for slot in upper.slots if slot.creature == 1)
    first, empty, spare = (hero.slots[index].ref for index in (0, 1, 2))
    for changes in ({"selection": "not-displayed"}, {"destination": "not-displayed"}, {"count": 8}, {"subject": "not-owned"}):
        parameters = {"subject": run.inspect_request().observation.context.subject,
                      "selection": base_slot, "destination": empty, "count": 1, **changes}
        run.reject(run.action("split_stack", **parameters), "invalid_split_rejected_before_send")
    run.manipulate("swap_stacks", base_slot, first)
    run.manipulate("swap_stacks", base_slot, first)
    run.manipulate("split_stack", base_slot, empty, 3)
    run.manipulate("split_stack", empty, base_slot, 1)
    run.manipulate("split_stack", first, spare, 5)
    run.manipulate("merge_stacks", spare, first)
    run.manipulate("swap_stacks", upgraded, empty)
    run.manipulate("swap_stacks", upgraded, empty)
    run.manipulate("split_stack", base_slot, empty, 5)
    run.manipulate("merge_stacks", empty, first)
    count = next(slot.count for army in run.armies() for slot in army.slots if slot.ref == first)
    run.reject(run.town_action("split_stack", selection=first, destination=base_slot, count=count),
               "required_last_hero_stack_rejected_before_send")
    run.reject(run.town_action("split_stack", selection=first, destination=upgraded, count=1),
               "unlike_creature_split_rejected_before_send")
    run.manipulate("split_stack", first, base_slot, 1)
    run.manipulate("merge_stacks", base_slot, first)
    remembered = run.armies()
    run.call(run.action("close"), "explicit_close_after_army_changes")
    run.open_town()
    base.ensure(run.armies() == remembered, "Closed/reopened army panel changed exact stacks")


def prepare_upgrade_confirmation(run):
    run.open_town()
    run.construct(".dwellingUpLvl1.")
    parent = run.inspect_request()
    source = next(slot for slot in run.army(role="visiting_hero").slots if slot.creature == 0)
    run.call(run.town_action("open_stack", selection=source.ref), "explicit_open_upgradeable_stack")
    panel = run.inspect_request().observation.window.stack_panel
    offer = next((item for item in panel.offers if item.creature == 1), None)
    base.ensure(offer is not None, "Native Pikeman stack lacks its available Halberdier upgrade")
    stack_subject = run.inspect_request().observation.context.subject
    run.reject(run.action("select_upgrade", subject=stack_subject, selection="not-displayed"),
               "foreign_upgrade_rejected_before_send")
    run.call(run.action("select_upgrade", subject=stack_subject, selection=offer.ref), "explicit_select_native_upgrade")
    selected = run.inspect_request().observation.window.stack_panel
    base.ensure(selected.affordable and selected.total_cost and any(selected.total_cost), "Expected native nonzero affordable cost")
    return parent, source, offer, stack_subject


def stack_cutoff_case(run):
    prepare_upgrade_confirmation(run)
    run.check("cutoff_begins_at_real_upgrade_confirmation", "passed")


def upgrades_case(run):
    parent, source, offer, stack_subject = prepare_upgrade_confirmation(run)
    run.call(run.action("back", subject=stack_subject), "explicit_cancel_upgrade_confirmation")
    base.ensure(run.inspect_request().observation.context.panel == "overview", "Cancellation did not return to stack overview")
    run.call(run.action("select_upgrade", subject=stack_subject, selection=offer.ref), "explicit_reselect_native_upgrade")
    action = run.action("upgrade_stack", subject=stack_subject, selection=offer.ref)
    before, additions, result = run.call(action, "explicit_upgrade_stack")
    run.check("native_upgrade_exact_effects", "passed", **exact_upgrade(parent, before, run.inspect_request(), action, additions, result))
    run.call(run.town_action("open_stack", selection=source.ref), "explicit_open_now_upgraded_stack")
    base.ensure(not run.inspect_request().observation.window.stack_panel.offers, "Fully upgraded baseline stack still offers an upgrade")
    run.call(run.action("back", subject=stack_subject), "explicit_close_stack_overview")


def garrison_case(run):
    original = run.inspect_request().observation.heroes[0].handle
    run.open_town()
    run.buy(0, 14)
    run.garrison()
    run.garrison()
    run.construct(".tavern.")
    run.garrison()
    second = run.hire()
    run.garrison()
    run.call(run.action("close"), "explicit_close_after_occupied_garrison_swap")
    run.reject(run.action("move_to", subject=second, x=0, y=0), "garrisoned_hero_move_rejected_before_send")
    run.depart(original, "explicit_original_visitor_departure")
    run.open_town()
    run.garrison()
    run.call(run.action("close"), "explicit_close_after_garrison_release")
    run.depart(second, "explicit_released_hero_departure")
    run.open_town()
    run.reject(run.town_action("swap_garrison"), "absent_empty_town_garrison_control_rejected_before_send")


def prepare_exchange(run):
    original = run.inspect_request().observation.heroes[0].handle
    run.open_town()
    run.construct(".tavern.")
    run.call(run.action("close"), "explicit_close_before_first_departure")
    run.depart(original, "explicit_first_hero_departure")
    run.open_town()
    second = run.hire()
    run.call(run.action("close"), "explicit_close_before_second_departure")
    run.depart(second, "explicit_second_hero_departure")
    destination = next(hero.position for hero in run.inspect_request().observation.heroes if hero.handle == second)
    before, additions, result = run.call(run.action("move_to", subject=original, x=destination[0], y=destination[1]),
                                        "explicit_visit_visible_owned_hero")
    after = run.inspect_request()
    base.ensure(result is not None and result.accepted and result.status == "interaction"
                and after.observation.context.kind == "exchange"
                and {after.observation.context.subject, after.observation.context.selection} == {original, second},
                "Native own-hero visit did not enter its exact exchange context")
    requests = [frame for frame in additions if frame.event is not None and frame.event.kind == "army_exchange_requested"]
    completed = [frame for frame in additions if frame.kind == "result" and frame.decision_seq == before.seq]
    opened = [frame for frame in additions if frame.kind == "context_opened"]
    base.ensure(len(requests) == len(completed) == len(opened) == 1
                and requests[0].seq < completed[0].seq < opened[0].seq,
                "Exchange request, parent-movement result and unsolicited context are not ordered")
    run.check("native_owned_exchange_query_opened", "passed", parent_decision=before.seq,
               query_event=requests[0].seq, parent_result=completed[0].seq, context_opened=opened[0].seq)
    return original, second


def exchange_cutoff_case(run):
    prepare_exchange(run)
    run.check("cutoff_begins_at_real_exchange_query", "passed")


def exchange_case(run):
    original, second = prepare_exchange(run)
    first, other = run.army(subject=original), run.army(subject=second)
    source = next(slot for slot in first.slots if slot.count > 1)
    target = next((slot for slot in other.slots if not slot.count), None)
    base.ensure(target is not None, "Native recruited hero lacks the declared free exchange slot")
    run.manipulate("split_stack", source.ref, target.ref, 1)
    run.manipulate("merge_stacks", target.ref, source.ref)
    before, additions, result = run.call(run.action("close"), "explicit_close_native_hero_exchange_query")
    proof = native_receipt(before, additions, result, "closed")
    base.ensure(run.inspect_request().observation.context.kind == "map"
                and before.observation.resources == run.inspect_request().observation.resources,
                "Closing actual exchange failed to release movement into the map")
    run.check("native_owned_exchange_query_closed", "passed", **proof)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--player", choices=(*base.COLORS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=104001)
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--case-seconds", type=float, default=120)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147483630 or not 1 <= args.max_days <= 20 or not 1 <= args.max_decisions <= 256 or any(
            not math.isfinite(value) or not 0 < value <= 180 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, days, actions and wall limits")
    for name in ("binary", "data", "output", "map"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    base.ensure(not args.output.exists(), "Preserve prior attempts; output already exists")
    scenario = hero_map_identity(args.map)
    environment.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    args.interface = INTERFACE
    cases, colors = CASES if args.case == "all" else (args.case,), base.COLORS if args.player == "both" else (args.player,)
    schedule = [{"case": case, "player": color, "seed": args.seed + 2 * CASES.index(case) + base.COLORS.index(color)}
                for case in cases for color in colors]
    args.output.mkdir(parents=True)
    implementation = implementation_identity()
    implementation["experiments/explorer/create_hero_scenario.py"] = digest(ROOT / "experiments/explorer/create_hero_scenario.py")
    native = runtime_identity(args.binary.parent)
    plan = {"schema": "heroes-army-management-native-plan-v1", "interface": INTERFACE, "map": scenario,
            "implementation": implementation, "native_runtime": native, "binary_sha256": digest(args.binary),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "schedule": schedule, "bounds": {key: getattr(args, key) for key in
                ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
            "training_input": False, "training_performed": False, "ordinary_outcome_claim": False,
            "rejection_scope": "Python pre-send, adapter visible notice and actual server acknowledgements are distinct"}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-army-management-native-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "training_input": False, "training_performed": False,
              "ordinary_outcome_claim": False, "optimizer_steps": 0}
    def check_identity():
        current = implementation_identity()
        current["experiments/explorer/create_hero_scenario.py"] = digest(ROOT / "experiments/explorer/create_hero_scenario.py")
        base.ensure(current == implementation and runtime_identity(args.binary.parent) == native
                    and hero_map_identity(args.map) == scenario and digest(args.binary) == plan["binary_sha256"]
                    and digest(args.output / "plan.json") == plan_hash, "Source/runtime/scenario changed during validation")
    for row in schedule:
        args.player = row["player"]
        run = None
        attempt = {**row, "status": "running", "outcomes": dict.fromkeys(base.COLORS), "rewards": dict.fromkeys(base.COLORS)}
        report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        try:
            with wall_limit(args.case_seconds):
                check_identity()
                run = ArmyRun(row["case"] + "-" + row["player"], args, environment, c, auditor, row["seed"])
                run.start()
                {"stacks": stacks_case, "upgrades": upgrades_case, "garrison": garrison_case,
                 "exchange": exchange_case, "stack-cutoff": stack_cutoff_case,
                 "exchange-cutoff": exchange_cutoff_case}[row["case"]](run)
                run.finish()
                run.persist_and_audit()
                base.ensure(run.end.status == "truncated", "Mechanics diagnostic did not reach its declared unscored cutoff")
                attempt.update(status="passed", mechanics=run.summary(), episode_directory=str(run.env.episode_dir))
                check_identity()
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if run is not None:
                retain_failed_evidence(run.env, None, attempt, error)
                if run.env.result is not None:
                    run.end = run.state = run.env.result
                    if not run.audit_attempted:
                        try:
                            with wall_limit(30): run.persist_and_audit()
                        except Exception as secondary:
                            attempt["failure_history_audit_error"] = f"{type(secondary).__name__}: {secondary}"
                attempt["mechanics"] = run.summary(attempt["error"])
        finally:
            if run is not None:
                try: run.env.close()
                except Exception as error: attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            try: check_identity()
            except Exception as error: attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
            report["not_started"] = len(schedule) - len(report["cases"])
            write_json(args.output / "report.json", report)
        if attempt["status"] == "failed": break
    report["status"] = "passed" if not report["not_started"] and all(case["status"] == "passed" for case in report["cases"]) else "failed"
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "attempts": len(report["cases"]), "not_started": report["not_started"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
