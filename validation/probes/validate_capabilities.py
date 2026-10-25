"""Bounded v8 real-engine capability acceptance; never training experience.

Operators use only the active player's displayed controls. Authored fixture
state is used to check effects, never as a policy observation. All attempts and
missing random branches are retained; each successful capture must also pass
independent player-history audit and native replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explorer import create_capability_scenarios as fixtures
from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.contextual_replay import verify_episode
from experiments.learner import contextual as c
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes.validate_adventure_dialogs import DialogRun, explicit_choice, notice_templates
from validation.probes.validate_army_management import native_receipt, rows as army_rows
from validation.probes.validate_hero_policy import retain_failed_evidence, wall_limit

INTERFACE = "adventure-capabilities-v1"
ACTION_SCHEMA = "heroes-context-actions-v8"
CASES = {
    "marketplace": "town_controls", "purchases": "town_controls", "artifacts": "artifacts",
    "artifact-transfer": "artifacts", "adventure-spells": "adventure_spells", "level-up": "level_up",
    "neutral-overflow": "neutral_overflow", "stables": "stables", "eagle-eye": "eagle_eye",
    "fly-movement": "fly_movement", "water-walk-movement": "water_walk_movement",
    "visions": "visions",
}


def resource_effect(before, after, choice, count=1):
    """Validate the exact displayed native exchange/purchase resource debit."""
    base.ensure(len(before) == len(after) == 7, "Resource snapshot must contain all seven resources")
    expected = [0] * 7
    if choice.kind == "trade_resource":
        base.ensure(type(count) is int and 1 <= count <= choice.max_count, "Trade quantity outside displayed lots")
        base.ensure(choice.resource_from != choice.resource_to, "Displayed exchange sells and buys the same resource")
        expected[choice.resource_from] -= choice.give * count
        expected[choice.resource_to] += choice.receive * count
    else:
        base.ensure(choice.cost is not None and len(choice.cost) == 7, "Purchase lacks its displayed resource cost")
        expected = [-value for value in choice.cost]
    actual = [new - old for old, new in zip(before, after)]
    base.ensure(actual == expected, "Native resource effect differs from the displayed offer")
    return actual


def inventory_counts(items):
    # Locked slots are constituent reservations of a combination artifact;
    # counting them as independently owned objects would duplicate ownership.
    return Counter((item.subject, item.id) for item in items if item.kind == "artifact" and not item.locked)


def schedule(cases, colors, seed, repeats):
    return [{"case": case, "fixture": CASES[case], "player": color, "repeat": repeat,
             "seed": seed + 128 * tuple(CASES).index(case) + 2 * repeat + fixtures.COLORS.index(color)}
            for case in cases for repeat in range(repeats) for color in colors]


class CapabilityRun(DialogRun):
    def expected_offers(self, frame):
        return c.Offers.from_observation(frame.observation, schema=ACTION_SCHEMA)

    def action(self, kind, **parameters):
        return c.Action(kind, schema=ACTION_SCHEMA, **parameters)

    def live(self):
        if self.state is not None and not self.state.done:
            frame = self.inspect_request()
            panel = getattr(frame.observation.window, "capability_panel", None)
            if panel is not None and panel.mandatory and (
                    self.commands >= self.args.max_decisions or time.monotonic() - self.started > self.args.case_seconds):
                raise base.ProbeFailure("Diagnostic bound reached in a mandatory capability query")
        return super().live()

    def inspect_request(self):
        frame = super().inspect_request()
        window = frame.observation.window
        panel = getattr(window, "capability_panel", None)
        if frame.observation.context.kind == "capability":
            base.ensure(panel is not None and panel.kind == frame.observation.context.panel,
                        "Active capability context lacks its own displayed panel")
        else:
            base.ensure(panel is None, "Closed capability panel refreshed its private contents")
        return frame

    def capability(self, expected=None):
        frame = self.inspect_request()
        panel = getattr(frame.observation.window, "capability_panel", None)
        base.ensure(frame.observation.context.kind == "capability" and panel is not None,
                    "Explicitly open the capability panel before reading its contents")
        base.ensure(expected is None or panel.kind == expected, "Wrong capability panel is active")
        return panel

    def open_capability(self, kind):
        before = self.inspect_request()
        base.ensure(before.observation.context.kind in ("hero", "town", "exchange"),
                    "Capability must open from a legitimate hero/town/exchange window")
        self.call(self.action("open_capability", subject=before.observation.context.subject, panel=kind),
                  "explicit_open_" + kind)
        panel = self.capability(kind)
        base.ensure(panel.parent_context == before.observation.context and not panel.mandatory,
                    "Manual capability lost its exact parent context")
        self.check("explicit_capability_window", "passed", kind=kind, parent=asdict(panel.parent_context))
        return panel

    def close_capability(self):
        before = self.inspect_request()
        panel = self.capability()
        base.ensure(not panel.mandatory, "Cannot close a mandatory native choice without answering")
        self.call(self.action("close_capability", subject=before.observation.context.subject),
                  "explicit_close_" + panel.kind)
        after = self.inspect_request()
        base.ensure(after.observation.context == panel.parent_context
                    and getattr(after.observation.window, "capability_panel", None) is None,
                    "Closing capability failed to restore its parent without refreshing private contents")
        self.check("closed_capability_fields_absent", "passed", kind=panel.kind)

    def choose(self, choice, *, count=None):
        before = self.inspect_request()
        panel = self.capability()
        base.ensure(choice in panel.choices, "Only currently displayed native controls may be selected")
        action = self.action(choice.kind, subject=before.observation.context.subject,
                             selection=choice.ref, count=count)
        _, additions, result = self.call(action, "explicit_" + choice.kind)
        base.ensure(result is not None and result.accepted, "Displayed native capability action did not complete")
        proof = native_receipt(before, additions, result, result.status)
        self.check("native_" + choice.kind, "passed", choice=asdict(choice), count=count, **proof)
        return before, additions, result

    def reject_foreign_choice(self, kind):
        self.reject(self.action(kind, subject=self.inspect_request().observation.context.subject,
                                selection="not-a-displayed-capability-choice", count=1 if kind == "trade_resource" else None),
                    "foreign_capability_choice_rejected_before_send")

    def close_to_map(self):
        if self.inspect_request().observation.context.kind == "capability":
            self.close_capability()
        if self.inspect_request().observation.context.kind != "map":
            self.call(self.action("close"), "explicit_close_parent_window")
        base.ensure(self.inspect_request().observation.window is None, "Closed parent window exposed fields")

    def open_hero_capability(self, kind):
        self.call(self.action("open_hero", subject=self.own_hero()), "explicit_open_actor")
        return self.open_capability(kind)


def marketplace_case(run):
    run.open_town()
    panel = run.open_capability("marketplace")
    run.reject_foreign_choice("trade_resource")
    choices = [choice for choice in panel.choices if choice.kind == "trade_resource" and choice.max_count >= 1]
    base.ensure(choices, "Prebuilt native marketplace has no affordable displayed exchange")
    # Cover both native rounding and quantity bounds using the actual offer.
    for _ in range(2):
        choice = next(choice for choice in run.capability().choices if choice.kind == "trade_resource" and choice.max_count >= 1)
        run.reject(run.action("trade_resource", subject=run.inspect_request().observation.context.subject,
                              selection=choice.ref, count=choice.max_count + 1),
                   "trade_above_displayed_max_rejected")
        count = min(2, choice.max_count)
        before, _, _ = run.choose(choice, count=count)
        delta = resource_effect(before.observation.resources, run.inspect_request().observation.resources, choice, count)
        run.check("trade_exact_native_resources", "passed", delta=delta, lots=count)
    remembered = run.capability().choices[0]
    run.close_capability()
    run.reject(run.action("trade_resource", subject=run.inspect_request().observation.context.subject,
                          selection=remembered.ref, count=1), "closed_market_choice_rejected")
    run.close_to_map()


def purchases_case(run):
    run.open_town()
    run.open_capability("purchases")
    run.reject_foreign_choice("buy_artifact")
    purchased = []
    for _ in range(4):
        choices = [choice for choice in run.capability().choices if choice.kind == "buy_artifact"]
        if not choices:
            break
        choice = choices[0]
        before, _, _ = run.choose(choice)
        delta = resource_effect(before.observation.resources, run.inspect_request().observation.resources, choice)
        purchased.append(choice.artifact)
        base.ensure(all(item.artifact != choice.artifact for item in run.capability().choices if item.kind == "buy_artifact"),
                    "Already owned spellbook or war machine remains purchasable")
        run.check("purchase_exact_native_cost", "passed", artifact=choice.artifact, delta=delta)
    base.ensure(len(set(purchased)) >= 2, "Spellbook and Castle war machine were not both purchased")
    run.close_to_map()
    inventory = run.open_hero_capability("artifacts")
    actual = {item.id for item in inventory.items if item.kind == "artifact"}
    base.ensure(set(purchased) <= actual, "Paid native artifacts absent after explicitly inspecting hero equipment")
    run.check("purchases_visible_only_after_inventory_open", "passed", artifacts=purchased)
    run.close_to_map()


def artifacts_case(run):
    panel = run.open_hero_capability("artifacts")
    run.reject_foreign_choice("move_artifact")
    before_counts = inventory_counts(panel.items)
    backpack_move = next((choice for choice in panel.choices if choice.kind == "move_artifact"
                          and choice.slot is not None and choice.slot >= 19 and choice.destination_slot < 19), None)
    base.ensure(backpack_move is not None, "Fixture lacks a displayed backpack-to-equipment move")
    run.choose(backpack_move)
    base.ensure(inventory_counts(run.capability().items) == before_counts,
                "Equipping an artifact created or lost ownership")
    reverse = next((choice for choice in run.capability().choices if choice.kind == "move_artifact"
                    and choice.artifact == backpack_move.artifact and choice.slot == backpack_move.destination_slot
                    and choice.destination_slot == backpack_move.slot), None)
    base.ensure(reverse is not None, "Equipped artifact lacks its displayed backpack destination")
    run.choose(reverse)
    # If the first move swapped a combination component, explicitly restore it
    # using the currently displayed equipment choices before assembling.
    for _ in range(8):
        choices = run.capability().choices
        assembly = next((choice for choice in choices if choice.kind == "assemble_artifact"), None)
        if assembly is not None:
            break
        restore = next((choice for choice in choices if choice.kind == "move_artifact" and choice.slot >= 19
                        and choice.destination_slot < 19 and choice.artifact != backpack_move.artifact), None)
        base.ensure(restore is not None, "Native combination components cannot be restored from displayed choices")
        run.choose(restore)
    else:
        raise base.ProbeFailure("Native artifact assembly remained unavailable")
    before_assembly = inventory_counts(run.capability().items)
    run.choose(assembly)
    disassembly = next((choice for choice in run.capability().choices if choice.kind == "disassemble_artifact"), None)
    base.ensure(disassembly is not None, "Assembled native artifact does not offer disassembly")
    run.choose(disassembly)
    base.ensure(inventory_counts(run.capability().items) == before_assembly,
                "Native assembly/disassembly roundtrip changed artifact ownership")
    run.check("native_inventory_and_combination_roundtrip", "passed", owned_before=[
        {"subject": subject, "artifact": artifact, "count": count}
        for (subject, artifact), count in sorted(before_counts.items())])
    run.close_to_map()


def artifact_transfer_case(run):
    original = run.own_hero()
    partner = next(hero for hero in run.inspect_request().observation.heroes if hero.handle != original)
    run.call(run.action("move_to", subject=original, x=partner.position[0], y=partner.position[1]),
             "explicit_visit_owned_partner")
    base.ensure(run.inspect_request().observation.context.kind == "exchange", "Native own-hero visit did not open exchange")
    panel = run.open_capability("artifacts")
    choice = next((choice for choice in panel.choices if choice.kind == "move_artifact"
                   and choice.subject != choice.destination and choice.destination_slot >= 19), None)
    base.ensure(choice is not None, "Real exchange lacks displayed inter-hero artifact transfer")
    counts = inventory_counts(panel.items)
    run.choose(choice)
    expected = counts.copy()
    expected[(choice.subject, choice.artifact)] -= 1
    expected[(choice.destination, choice.artifact)] += 1
    base.ensure(+inventory_counts(run.capability().items) == +expected,
                "Native inter-hero artifact transfer changed unrelated ownership")
    run.check("native_artifact_transfer_exact_owners", "passed", artifact=choice.artifact,
              source=choice.subject, destination=choice.destination)
    run.close_to_map()


def adventure_spells_case(run):
    prior = run.hero_window()
    panel = run.open_hero_capability("spellbook")
    run.reject_foreign_choice("cast_adventure_spell")
    offered = {item.id for item in panel.items if item.kind == "spell"}
    cast = set()
    target_cancel_checked = False
    for _ in range(len(fixtures.ADVENTURE_SPELLS) * 2):
        choices = [choice for choice in run.capability().choices if choice.kind == "cast_adventure_spell"
                   and choice.spell not in cast]
        if not choices:
            break
        # Choose a legal displayed target. No hidden terrain, route cost, native
        # map file or server state is consulted by this operator.
        choice = choices[0]
        mana_before = run.inspect_request().observation.window.mana
        cost = next(item.amount for item in run.capability().items if item.kind == "spell" and item.id == choice.spell)
        _, additions, _ = run.choose(choice)
        frame = run.inspect_request()
        cancelled = False
        if frame.observation.context.kind == "capability":
            result_panel = run.capability()
            if result_panel.kind == "spell_target":
                base.ensure(result_panel.mandatory, "Native Town Portal target selection must remain mandatory")
                run.reject(run.action("close_capability", subject=frame.observation.context.subject),
                           "mandatory_spell_target_close_rejected")
                run.reject_foreign_choice("choose_spell_target")
                operation = "cancel_spell_target" if not target_cancel_checked else "choose_spell_target"
                target = next((candidate for candidate in result_panel.choices if candidate.kind == operation), None)
                base.ensure(target is not None, "Native target query lacks its displayed " + operation)
                run.choose(target)
                cancelled = operation == "cancel_spell_target"
                target_cancel_checked |= cancelled
            elif result_panel.kind == "world_view":
                base.ensure(not result_panel.mandatory and not result_panel.choices,
                            "Delivered world-view payload invented mandatory choices")
                base.ensure(any(item.kind in ("world_object", "world_terrain") for item in result_panel.items),
                            "Fixture world-view spell delivered no visible information")
                run.check("native_delivered_world_view", "passed", spell=choice.spell,
                          object_count=sum(item.kind == "world_object" for item in result_panel.items),
                          terrain_count=sum(item.kind == "world_terrain" for item in result_panel.items))
                run.close_capability()
            else:
                raise base.ProbeFailure("Adventure cast opened an unexpected capability panel: " + result_panel.kind)
        base.ensure(run.inspect_request().observation.context.kind == "map", "Adventure spell failed to return to map")
        run.open_hero_capability("spellbook")
        mana_after = run.inspect_request().observation.window.mana
        if cancelled:
            base.ensure(mana_after == mana_before, "Cancelled native spell target query spent mana")
            run.check("native_spell_target_cancel", "passed", spell=choice.spell, mana=mana_after)
            continue
        # Failed native effects can show a legitimate informational notice and
        # spend no mana. Preserve that gap rather than labeling it a cast.
        if mana_before - mana_after == cost:
            cast.add(choice.spell)
            run.check("native_spell_exact_mana", "passed", spell=choice.spell, cost=cost,
                      mana_before=mana_before, mana_after=mana_after)
        else:
            run.check("native_spell_effect", "unverified", spell=choice.spell, displayed_cost=cost,
                      mana_before=mana_before, mana_after=mana_after, notices=notice_templates(additions))
            break
    run.close_to_map()
    current = run.hero_window()
    base.ensure(cast and current.mana < prior.mana, "No real adventure spell consumed native mana")
    run.check("native_adventure_casts", "passed", spell_ids=sorted(cast),
              initial_displayed_spell_ids=sorted(offered), mana_before=prior.mana, mana_after=current.mana)
    # A panel can legitimately display an unavailable spell; absence of a
    # castable target is recorded as a coverage gap, never silently passed.
    if offered - cast:
        run.check("adventure_spell_target_coverage", "unverified", uncast_spell_ids=sorted(offered - cast))


def level_up_case(run):
    previous = run.hero_window()
    visited = set()
    for _ in range(9):
        targets = run.targets(101, visited)  # Native Obj::TREASURE_CHEST.
        if not targets:
            break
        target = targets[0]
        run.visit(target)
        visited.add(target.handle)
        if run.inspect_request().observation.context.kind != "dialog":
            continue
        panel = run.inspect_request().observation.window.dialog_panel
        index = next((index for index, item in enumerate(panel.components) if item.kind == "experience"), None)
        base.ensure(index is not None, "Native chest has no displayed experience component")
        choice = explicit_choice(panel, "select_component", index)
        run.call(run.action("choose_dialog", subject=run.inspect_request().observation.context.subject,
                            selection=choice.ref), "explicit_take_displayed_experience")
        decisions = []
        while run.inspect_request().observation.context.kind == "capability":
            panel = run.capability("level_up")
            base.ensure(panel.mandatory, "Native level-up query is not mandatory")
            run.reject(run.action("close_capability", subject=run.inspect_request().observation.context.subject),
                       "mandatory_levelup_close_rejected")
            run.reject_foreign_choice("choose_level_up")
            choice = next((item for item in panel.choices if item.kind == "choose_level_up"), None)
            base.ensure(choice is not None, "Native level-up lacks displayed skill options")
            run.choose(choice)
            decisions.append({"skill": choice.skill, "level": choice.skill_level})
        current = run.hero_window()
        base.ensure(decisions and current.level > previous.level,
                    "Native experience did not produce a policy-selected level-up and higher hero level")
        run.check("native_levelup_skill_choice", "passed", before=previous.level, after=current.level, choices=decisions)
        return
    run.check("native_levelup_branch", "unverified", reason="Predeclared chest sequence did not offer experience")


def movement_spell_case(run):
    before = run.inspect_request().observation
    actor = next(hero for hero in before.heroes if hero.handle == run.own_hero())
    target = (actor.position[0], actor.position[1] + 4, actor.position[2])
    tiles = {tile.position: tile for tile in before.tiles}
    base.ensure(target in tiles and not tiles[target].blocked, "Far-bank land target is not visible and selectable")
    # Native core terrain ID8 is water. The operator reads only these already
    # delivered tiles; no fixture file or hidden route knowledge enters actions.
    water = {position for position, tile in tiles.items() if tile.terrain == 8}
    base.ensure(tiles[target].terrain != 8 and all((target[0], row, target[2]) in water
                                                 for row in range(actor.position[1] + 1, target[1])),
                "Displayed crossing lacks visible water and a land destination")
    _, initial_frames, result = run.call(run.action("move_to", subject=actor.handle, x=target[0], y=target[1]),
                                         "explicit_attempt_crossing_without_spell")
    base.ensure(result is not None and result.status == "no_route_known" and result.position == actor.position
                and not any(frame.kind == "issued" for frame in initial_frames),
                "Crossing fixture is reachable without its isolated movement spell")
    panel = run.open_hero_capability("spellbook")
    spells = [item for item in panel.items if item.kind == "spell"]
    base.ensure(len(spells) == 1 and spells[0].id in (6, 7), "Movement fixture must isolate Fly or Water Walk")
    choice = next(candidate for candidate in panel.choices if candidate.kind == "cast_adventure_spell")
    run.choose(choice)
    base.ensure(run.inspect_request().observation.context.kind == "map", "Movement spell did not close its window")
    _, additions, result = run.call(run.action("move_to", subject=actor.handle, x=target[0], y=target[1]),
                                    "explicit_native_crossing_after_spell")
    base.ensure(result is not None and result.destination_reached and result.position == target,
                "Movement spell was cast but native route failed to reach the visible far bank")
    positions = {hero.position for frame in additions if frame.observation is not None
                 for hero in frame.observation.heroes if hero.handle == actor.handle}
    crossed = sorted(positions & water)
    issued = [frame for frame in additions if frame.kind == "issued"]
    acknowledged = [frame for frame in additions if frame.kind == "acknowledged"]
    base.ensure(crossed and len(issued) == len(acknowledged) >= 4 and all(frame.accepted for frame in acknowledged),
                "Crossing lacks actual water positions and accepted native movement commands")
    run.check("native_movement_spell_crossing", "passed", spell=choice.spell, target=list(target),
              water_positions=[list(position) for position in crossed], native_steps=len(issued),
              movement_before=actor.movement, movement_after=result.movement)


def visions_case(run):
    run.call(run.action("open_hero", subject=run.own_hero()), "explicit_open_actor_before_visions")
    run.reject(run.action("open_capability", subject=run.own_hero(), panel="visions"),
               "visions_inspection_unavailable_before_cast")
    run.close_to_map()
    book = run.open_hero_capability("spellbook")
    spell = next((choice for choice in book.choices if choice.kind == "cast_adventure_spell" and choice.spell == 2), None)
    base.ensure(spell is not None, "Native Visions spell is not offered by its isolated fixture")
    run.choose(spell)
    panel = run.open_hero_capability("visions")
    base.ensure(panel.inspected_subject is None and not panel.items,
                "Visions panel leaked object contents before an explicit inspection")
    run.reject_foreign_choice("inspect_object")
    for kind, expected_item in ((54, "monster_intent"), (34, "hero_stat"), (98, "town_stat")):
        observation = run.inspect_request().observation
        objects = {obj.handle: obj for obj in observation.objects}
        choice = next((choice for choice in run.capability().choices
                       if choice.destination in objects and objects[choice.destination].kind == kind
                       and objects[choice.destination].owner != "self"), None)
        base.ensure(choice is not None, "Native Visions lacks a displayed enemy/neutral object category " + str(kind))
        before = run.inspect_request()
        _, additions, result = run.call(run.action("inspect_object", subject=before.observation.context.subject,
                                                  selection=choice.ref), "explicit_visions_inspection")
        panel = run.capability("visions")
        base.ensure(result is not None and result.status == "inspected" and result.accepted
                    and not any(frame.kind == "issued" for frame in additions),
                    "Read-only Visions inspection unexpectedly issued a native mutation")
        base.ensure(panel.inspected_subject == choice.destination and panel.items
                    and all(item.subject == choice.destination for item in panel.items)
                    and any(item.kind == expected_item for item in panel.items),
                    "Visions did not expose exactly its explicitly selected object's native information")
        if kind in (54, 34):
            base.ensure(any(item.kind == "creature_exact" for item in panel.items)
                        and not any(item.kind == "creature_quantity" for item in panel.items),
                        "Expert Visions did not expose native exact creature counts")
        run.check("native_visions_inspection", "passed", object_kind=kind, subject=choice.destination,
                  item_kinds=sorted({item.kind for item in panel.items}), native_issued=0)
        run.close_capability()
        panel = run.open_capability("visions")
        base.ensure(panel.inspected_subject is None and not panel.items,
                    "Reopening Visions refreshed a previous target without inspection")
    run.close_to_map()


def neutral_overflow_case(run):
    target = next(iter(run.targets(54)), None)
    base.ensure(target is not None, "Compliant diagnostic neutral is not visible")
    prior = run.hero_window()
    run.visit(target)
    panel = run.inspect_request().observation.window.dialog_panel
    base.ensure(panel.family == "neutral_free_join", "Compliant neutral did not offer native free joining")
    choice = explicit_choice(panel, "accept")
    _, additions, _ = run.call(run.action("choose_dialog", subject=run.inspect_request().observation.context.subject,
                                        selection=choice.ref), "explicit_accept_full_army_join")
    frame = run.inspect_request()
    base.ensure(frame.observation.context.kind == "exchange", "Native full-army joining did not open its exchange query")
    armies = army_rows(frame)
    neutral = next((army for army in armies if army.role == "joining_neutral"), None)
    own = next((army for army in armies if army.subject == run.own_hero()), None)
    base.ensure(neutral is not None and own is not None, "Overflow window lacks its exact delivered joining/hero armies")
    base.ensure(any(item.event is not None and item.event.kind == "army_join_requested" for item in additions),
                "Overflow ownership was not authorized by a real native query")
    source = next(slot for slot in neutral.slots if slot.count)
    destination = next(slot for slot in own.slots if slot.count)
    # Native garrison exchange decides how to resolve the full army; one swap
    # exercises the real query without deleting or inventing stacks in Python.
    run.call(run.action("swap_stacks", subject=frame.observation.context.subject,
                        selection=source.ref, destination=destination.ref), "explicit_choose_overflow_stack")
    run.call(run.action("close"), "explicit_close_native_join_exchange")
    base.ensure(run.inspect_request().observation.context.kind == "map", "Native overflow query did not finish")
    current = run.hero_window()
    base.ensure(current.army != prior.army and any(stack.creature == source.creature and stack.count == source.count
                                                 for stack in current.army),
                "Chosen joining stack is absent from the hero's reopened army")
    run.check("native_full_army_join_resolved", "passed", creature=source.creature, count=source.count)


def stables_case(run):
    before = run.inspect_request()
    town = before.observation.towns[0]
    _, additions, _ = run.call(run.action("move_to", subject=run.own_hero(), x=town.position[0], y=town.position[1]),
                              "explicit_enter_castle_stables")
    templates = notice_templates(additions)
    base.ensure("core.genrltxt.580" in templates, "Castle Stables bonus did not emit its native informational notice")
    run.check("native_castle_stables_notice", "passed", templates=templates)
    if run.inspect_request().observation.context.kind != "map":
        run.close_to_map()


def eagle_eye_case(run):
    observation = run.inspect_request().observation
    enemy = next((obj for obj in observation.objects if obj.kind == 34), None)
    base.ensure(enemy is not None, "Prepared spellcasting enemy is not currently visible")
    _, additions, _ = run.call(run.action("move_to", subject=run.own_hero(), x=enemy.position[0], y=enemy.position[1]),
                              "explicit_fight_visible_spellcaster")
    learned = [frame.event.notice for frame in additions if frame.event is not None and frame.event.kind == "notice"
               and any(token.text_id == "core.genrltxt.221" for token in frame.event.notice.text)]
    if learned:
        run.check("native_eagle_eye_notice", "passed", notices=[asdict(notice) for notice in learned])
    else:
        run.check("native_eagle_eye_notice", "unverified", templates=notice_templates(additions),
                  reason="Native automatic combat or Eagle Eye chance did not trigger a learning notice")


RUN_CASE = {"marketplace": marketplace_case, "purchases": purchases_case, "artifacts": artifacts_case,
            "artifact-transfer": artifact_transfer_case, "adventure-spells": adventure_spells_case,
            "level-up": level_up_case, "neutral-overflow": neutral_overflow_case,
            "stables": stables_case, "eagle-eye": eagle_eye_case,
            "fly-movement": movement_spell_case, "water-walk-movement": movement_spell_case,
            "visions": visions_case}


def fixture_identity(fixture, player, maps):
    path = maps / (fixtures.scenario_id(fixture, player) + ".vmap")
    fixtures.verify_scenario(path, fixture=fixture, player=player)
    manifest = path.with_suffix(".manifest.json")
    declared = json.loads(manifest.read_text())
    base.ensure(declared["map_sha256"] == digest(path) and declared["diagnostic_only"] is True,
                "Capability fixture metadata differs from its exact map")
    return {"path": str(path), "sha256": digest(path), "manifest_sha256": digest(manifest),
            "fixture": fixture, "player": player, "training_input": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "replay-binary", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maps", type=Path, default=fixtures.MAPS)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--player", choices=(*fixtures.COLORS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=158001)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--max-days", type=int, default=12)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--case-seconds", type=float, default=180)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147480000 or not 1 <= args.seed_count <= 16 or not 1 <= args.max_days <= 32 or not 1 <= args.max_decisions <= 1024 or any(
            not math.isfinite(value) or not 0 < value <= 600 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, days, decisions and wall limits")
    for name in ("binary", "replay_binary", "data", "output", "maps"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    base.ensure(not args.output.exists(), "Preserve every previous run; output already exists")
    args.interface = INTERFACE
    c.profile_for(INTERFACE)
    cases = tuple(CASES) if args.case == "all" else (args.case,)
    colors = fixtures.COLORS if args.player == "both" else (args.player,)
    identities = {(CASES[case], color): fixture_identity(CASES[case], color, args.maps) for case in cases for color in colors}
    for identity in identities.values():
        environment.VcmiLauncher(args.binary, args.data, Path(identity["path"])).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    implementation = implementation_identity()
    for name in ("experiments/explorer/create_capability_scenarios.py", "validation/probes/validate_capabilities.py",
                 "validation/probes/validate_contextual_native.py"):
        implementation[name] = digest(ROOT / name)
    native = runtime_identity(args.binary.parent)
    args.output.mkdir(parents=True)
    plan = {"schema": "heroes-capability-diagnostic-plan-v1", "interface": INTERFACE,
            "schedule": schedule(cases, colors, args.seed, args.seed_count), "fixtures": list(identities.values()),
            "implementation": implementation, "native_runtime": native,
            "binary_sha256": digest(args.binary), "replay_binary_sha256": digest(args.replay_binary),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
            "training_performed": False, "training_input": False, "optimizer_steps": 0,
            "missing_branches": "Retained as unverified; no adaptive seed replacement or native outcome injection."}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-capability-native-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "not_started": len(plan["schedule"]),
              "training_performed": False, "training_input": False, "optimizer_steps": 0}

    def check_identity():
        for name, expected in implementation.items():
            base.ensure(digest(ROOT / name) == expected, "Diagnostic source changed during capture: " + name)
        base.ensure(runtime_identity(args.binary.parent) == native and digest(args.binary) == plan["binary_sha256"]
                    and digest(args.replay_binary) == plan["replay_binary_sha256"] and digest(args.output / "plan.json") == plan_hash,
                    "Diagnostic runtime or plan changed")
        for (fixture, color), expected in identities.items():
            base.ensure(fixture_identity(fixture, color, args.maps) == expected, "Capability fixture changed")

    for row in plan["schedule"]:
        run = None
        attempt = {**row, "status": "running"}
        report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        try:
            with wall_limit(args.case_seconds):
                check_identity()
                args.player, args.map = row["player"], Path(identities[(row["fixture"], row["player"])]["path"])
                run = CapabilityRun(f"{row['case']}-{row['player']}-{row['repeat']:02d}", args, environment, c, auditor, row["seed"])
                run.start()
                RUN_CASE[row["case"]](run)
                run.finish()
                run.persist_and_audit()
                base.ensure(run.end.status in ("completed", "truncated"), "Mechanics ended as an invalid native failure")
                replay = verify_episode(run.env.episode_dir, binary=args.replay_binary, data=args.data,
                                        timeout=min(180, args.case_seconds))
                base.ensure(replay["verified"], "Independent native replay rejected the capability capture: " + str(replay.get("error")))
                run.check("independent_native_replay", "passed")
                summary = run.summary()
                attempt.update(status=summary["probe_status"], mechanics=summary, replay_verified=True,
                               episode_directory=str(run.env.episode_dir))
                check_identity()
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if run is not None:
                retain_failed_evidence(run.env, None, attempt, error)
                if run.env.result is not None:
                    run.end = run.state = run.env.result
                    if not run.audit_attempted:
                        try:
                            with wall_limit(30):
                                run.persist_and_audit()
                        except Exception as secondary:
                            attempt["failure_history_audit_error"] = f"{type(secondary).__name__}: {secondary}"
                attempt["mechanics"] = run.summary(attempt["error"])
        finally:
            if run is not None:
                try:
                    run.env.close()
                except Exception as error:
                    attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            try:
                check_identity()
            except Exception as error:
                attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
            report["not_started"] = len(plan["schedule"]) - len(report["cases"])
            write_json(args.output / "report.json", report)
        if attempt["status"] == "failed":
            break
    report["status"] = "failed" if report["not_started"] or any(row["status"] == "failed" for row in report["cases"]) else (
        "unverified" if any(row["status"] == "unverified" for row in report["cases"]) else "passed")
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "attempts": len(report["cases"]), "not_started": report["not_started"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
