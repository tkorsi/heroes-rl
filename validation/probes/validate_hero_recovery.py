#!/usr/bin/env python3
"""Ordinary-scenario hero-loss/replacement mechanics; never training or strategy.

Both players receive explicit diagnostic commands. The prospective defender
buys fourteen displayed pikemen and transfers exactly fourteen to its visitor.
The prospective loser constructs a Tavern, approaches only currently displayed
tiles/enemy towns, and attacks through the unchanged automatic battle system.
A loss is never assumed or forced: failure to reach it stays unverified. Only a
real battle loss, retained own town and exact native hire prove recovery.
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
from validation.probes import validate_hero_recruitment as hire
from validation.probes import validate_town_capture as capture
from validation.probes import validate_town_native as town
from validation.probes.validate_town_transfer import exact_transfer


def verified_loss(frame, history, hero_ref, town_ref):
    """Require permitted actual loss events and a live town-only observation."""
    base.ensure(frame.kind == "decision" and not frame.observation.heroes
                and town_ref in {item.handle for item in frame.observation.towns},
                "Recovery requires an actual live town-only decision")
    lost = [item for item in history if item.seq < frame.seq and item.event is not None
            and item.event.kind == "hero_lost" and item.event.subject == hero_ref]
    defeated = [item for item in history if item.seq < frame.seq and item.event is not None
                and item.event.kind == "battle_complete" and item.event.won is False]
    starts = [item for item in history if item.seq < frame.seq and item.event is not None
              and item.event.kind == "battle_start" and item.event.side == "attacker"]
    base.ensure(len(lost) == 1 and len(defeated) == 1 and starts,
                "Town-only state lacks one observed original-hero loss and actual automatic battle defeat")
    battle = starts[-1]
    base.ensure(any(item.observation is not None and item.seq < battle.seq
                    and hero_ref in {hero.handle for hero in item.observation.heroes}
                    and town_ref in {town.handle for town in item.observation.towns} for item in history),
                "Recovery lacks the original owned hero and retained town before the actual battle")
    base.ensure(battle.seq < lost[0].seq and battle.seq < defeated[0].seq,
                "Hero loss/battle defeat is not after the visible attacking battle start")
    base.ensure(not any(item.kind in ("terminal", "finalized") for item in history if item.seq <= frame.seq),
                "Hero loss was incorrectly treated as the whole episode ending")
    return {"hero": hero_ref, "retained_town": town_ref, "day": frame.day,
            "battle_start_seq": battle.seq, "battle_defeat_seq": defeated[0].seq,
            "hero_lost_seq": lost[0].seq, "town_only_decision_seq": frame.seq,
            "episode_terminal": False}


class RecoveryRun(hire.HeroRun):
    def start(self):
        # Both roles need their own first turn; do not skip the other player's
        # preparation when the requested recovery side is Blue.
        base.CaseRun.start(self)


def construct_tavern(run):
    run.open_town()
    hall = run.open_hall()
    choice = next((item for item in hall.buildings if ".tavern." in item.name_text_id), None)
    base.ensure(choice is not None and choice.status == "allowed", "Recovery Tavern is not buildable natively")
    detail = run.open_building(choice)
    before, additions, result = run.call(run.town_action("construct", selection=detail.ref),
                                        "explicit_build_recovery_tavern")
    town.request_evidence(run, additions, result, "built")
    after = run.inspect_request()
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources, after.observation.resources)) == detail.cost,
                "Recovery Tavern construction changed a different resource amount")
    run.check("native_recovery_tavern_constructed", "passed", price=list(detail.cost))
    run.unwind()


def reinforce_defender(run):
    run.open_town()
    rows = run.panel("overview").armies
    base.ensure(rows is not None and rows.destination is not None and rows.source.role == "town_garrison",
                "Defender lacks actual town/visitor army rows")
    original_count = sum(slot.count for slot in rows.destination.slots)
    before_army = run.inspect_request().observation.window.army
    panel = run.open_recruitment()
    selected = run.select_creature(0)  # Explicit mechanics fixture: displayed Pikeman card.
    base.ensure(selected.slider_max >= 14, "Native defender does not offer fourteen affordable pikemen")
    town.successful_purchase(run, before_army, selected, 14, "explicit_buy_fourteen_defender_pikemen")
    run.call(run.town_action("back"), "explicit_close_defender_recruitment")
    rows = run.panel("overview").armies
    source = next((slot for slot in rows.source.slots if slot.creature == 0 and slot.count == 14), None)
    destination = next((slot for slot in rows.destination.slots if slot.creature == 0), None)
    base.ensure(source is not None and destination is not None, "Displayed native defender slots do not match the declared transfer")
    action = run.town_action("transfer_stack", selection=source.ref, destination=destination.ref, count=14)
    before, additions, result = run.call(action, "explicit_transfer_fourteen_to_defender")
    effect = exact_transfer(before, run.inspect_request(), action, additions, result)
    base.ensure(sum(slot.count for slot in run.panel("overview").armies.destination.slots) == original_count + 14,
                "Actual defender did not gain precisely fourteen native creatures")
    run.check("native_defender_reinforcement_fixture", "passed", original_count=original_count,
              final_count=original_count + 14, transfer=effect)
    run.unwind()


def recovery_case(run):
    loser = run.args.player
    defender = "blue" if loser == "red" else "red"
    prepared = set()
    original_hero = retained_town = None
    prefix = None
    attempted = set()
    lost_evidence = None
    while run.live():
        frame = run.inspect_request()
        obs = frame.observation
        color = run.state.color
        base.ensure(obs.context.kind == "map", "Recovery controls unexpectedly left the map")
        if color not in prepared:
            base.ensure(len(obs.heroes) == len(obs.towns) == 1,
                        "Declared ordinary initial hero/town counts changed")
            if color == loser:
                original_hero, retained_town = obs.heroes[0].handle, obs.towns[0].handle
                prefix = run.history(color)
                construct_tavern(run)
            else:
                reinforce_defender(run)
            prepared.add(color)
            run.call(run.action("end_turn"), "explicit_turn_after_role_preparation")
            continue
        if color == loser and not obs.heroes:
            history = [contract.decode_frame(item) for item in run.history(color)]
            lost_evidence = verified_loss(frame, history, original_hero, retained_town)
            run.check("native_hero_loss_keeps_episode_live_and_town_owned", "passed", **lost_evidence)
            break
        if color == loser and prepared == {loser, defender} and obs.heroes[0].movement:
            base.ensure(len(obs.heroes) == 1 and obs.heroes[0].handle == original_hero,
                        "Recovery approach unexpectedly switched heroes")
            target, reason = capture.attacker_destination(obs, attempted)
            if target is None:
                run.check("native_original_hero_loss", "unverified", reason=reason)
                break
            attempted.add(target)
            run.call(run.action("move_to", subject=original_hero, x=target[0], y=target[1]),
                     "diagnostic_recovery_approach_" + reason)
        else:
            run.call(run.action("end_turn"), "explicit_turn_for_actual_movement_reset_or_defender_wait")
    if lost_evidence is None:
        run.check("native_original_hero_loss", "unverified", reason="No verified battle loss within the declared ordinary-game bounds")
        return
    base.ensure(run.state.color == loser and run.state.reward == 0, "Ordinary hero loss assigned a terminal reward")
    run.open_town()
    base.ensure(run.inspect_request().observation.context.subject == retained_town,
                "Recovery opened a different town")
    rows = run.panel("overview").armies
    base.ensure(rows.destination is None, "Lost hero remained the current visitor")
    tavern = run.open_tavern()
    base.ensure(tavern.recruit_status == "allowed" and tavern.offers,
                "Native no-hero owner cannot currently recruit an offered replacement")
    # Select a displayed card explicitly, even if it is the native default.
    selected = run.select_hero(tavern.offers[0].ref)
    action = run.town_action("hire_hero", selection=selected.selected_offer_ref)
    before, additions, result = run.call(action, "explicit_hire_replacement_after_real_battle_loss")
    effect = hire.exact_hire(before, run.inspect_request(), action, additions, result)
    base.ensure(effect["heroes_before"] == 0 and effect["heroes_after"] == 1
                and run.panel("overview").armies.source == rows.source,
                "Replacement changed original town creatures or was not a zero-to-one native hero recovery")
    run.check("native_replacement_hero_after_real_loss", "passed", loss=lost_evidence, hire=effect)
    replacement = effect["new_hero"]
    run.call(run.action("close"), "explicit_close_recovery_town")
    run.call(run.action("open_hero", subject=replacement), "explicit_inspect_replacement_hero")
    window = run.inspect_request().observation.window
    base.ensure(window.context.subject == replacement and window.army, "Replacement hero has no selectable native army/window")
    run.check("native_replacement_window", "passed", hero=replacement, army=[asdict(stack) for stack in window.army])
    run.call(run.action("close"), "explicit_close_replacement_hero")
    run.depart(replacement, "explicit_move_replacement_hero_after_loss")
    base.ensure(run.history(loser)[:len(prefix)] == prefix, "Recovery mutated the original player history")
    run.check("ordinary_scenario_recovery_sequence_complete", "passed", player=loser, original_hero=original_hero,
              replacement=replacement, own_town=retained_town, training_input=False, reward=0,
              outcome_frequency_claim=False, automatic_battle_unchanged=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=base.COLORS, required=True)
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=102101)
    parser.add_argument("--max-days", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--case-seconds", type=float, default=180)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147483647 or not 1 <= args.max_days <= 30 or not 1 <= args.max_decisions <= 300 or any(
            not math.isfinite(value) or not 0 < value <= 180 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, choices, days and finite time limits")
    for name in ("map", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; every attempted game must be preserved")
    verify_hero_scenario(args.map)
    args.interface = contract.HEROES_ENVIRONMENT
    env = base.load("heroes_recovery_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_recovery_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    implementation = implementation_identity()
    from experiments.headless.contextual_evidence import runtime_identity
    base.write(args.output / "plan.json", {
        "schema": "heroes-hero-recovery-validation-v1", "player": args.player, "interface": args.interface,
        "seed": args.seed, "map": str(args.map), "map_sha256": base.sha(args.map),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation": implementation, "native_runtime": runtime_identity(args.binary.parent),
        "binary_sha256": base.sha(args.binary), "probe_sha256": base.sha(Path(__file__)),
        "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog")},
        "training_input": False, "training_performed": False, "ordinary_outcome_frequency_claim": False,
        "scripted_mechanics_only": True, "native_rules_and_automatic_battle_unchanged": True,
        "declared_commands": "Build loser's Tavern; native buy and exact transfer fourteen defender Pikemen; visible-map approach and automatic combat; recruit/move only after an observed actual hero loss"})
    run = RecoveryRun("recovery", args, env, contract, auditor, args.seed)
    error = None
    try:
        run.start()
        recovery_case(run)
        run.finish()
        run.persist_and_audit()
        base.ensure(implementation_identity() == implementation, "Authored implementation changed during native validation")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("native_hero_recovery_execution", "failed", error=error)
        hire.finalize_diagnostic_failure(run, error)
    finally:
        hire.close_diagnostic(run)
    report = run.summary(error)
    report.update(interface=args.interface, player=args.player, training_input=False,
                  ordinary_outcome_frequency_claim=False, scripted_mechanics_only=True)
    base.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "error": error, "report": str(args.output / "report.json")}))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
