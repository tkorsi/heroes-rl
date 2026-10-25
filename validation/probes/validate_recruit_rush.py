#!/usr/bin/env python3
"""Bounded hand-scripted recruitment/rush diagnostics, never learner experience.

Only current player-visible observations choose actions. The script knows the
public map rectangle, but neither reads map objects nor controls Nullkiller.
Every typed interface command counts, including navigation and acknowledgements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless.builtin_env import BuiltinHeadlessEnv
from experiments.headless.builtin_replay import verify_episode
from experiments.headless.env import Limits
from experiments.headless.process_deadline import DeadlineVcmiLauncher
from experiments.learner import contextual as c
from validation.probes.validate_builtin_boundaries import until, write


def diagnostic_accepted(report):
    return (report.get("replay", {}).get("verified") is True
        and report.get("owned_processes_closed") is True
        and report.get("status") in ("completed", "truncated")
        and not report.get("error") and not report.get("cleanup_error"))


def summarize_episode(directory, actions, color):
    """Bind observed effects and battles to sent actions, including terminal replies."""
    path = directory / "agents" / color / "history-v8.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    by_decision = {entry["decision_seq"]: entry["index"] for entry in actions}
    results, battles, current_action, active_battle = [], [], None, None
    consumed = []
    for frame in frames:
        if frame["kind"] == "consumed":
            current_action = by_decision.get(frame["decision_seq"])
            consumed.append(current_action)
        elif frame["kind"] == "result":
            results.append({"action_index": by_decision.get(frame["decision_seq"]),
                "history_seq": frame["seq"], "day": frame["day"], **frame["result"]})
        elif frame["kind"] == "event" and frame["event"]["kind"] == "battle_start":
            active_battle = {"start_action_index": current_action, "start_history_seq": frame["seq"],
                "day": frame["day"], "side": frame["event"]["side"]}
            battles.append(active_battle)
        elif frame["kind"] == "event" and frame["event"]["kind"] == "battle_complete":
            if active_battle is None:
                raise ValueError("Battle completion has no captured start")
            active_battle.update(completion_action_index=current_action,
                completion_history_seq=frame["seq"], won=frame["event"]["won"])
            active_battle = None
    return {"typed_actions_attempted": len(actions), "consumed_actions": consumed,
        "consumed_action_count": len(consumed), "result_status_counts": dict(Counter(x["status"] for x in results)),
        "rejected_results": [x for x in results if not x["accepted"]], "results": results, "battles": battles,
        "superseded": [{"action_index": by_decision.get(x["decision_seq"]), "reason": x["reason"],
            "history_seq": x["seq"]} for x in frames if x["kind"] == "superseded"],
        "terminal": [{"history_seq": x["seq"], "day": x["day"], "outcome": x["outcome"]}
            for x in frames if x["kind"] == "terminal"],
        "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Rush:
    """An explicitly authored diagnostic plan, not a neural policy."""

    def __init__(self):
        self.stage = "open"
        self.recruited_controls = set()
        self.hero = None
        self.town = None
        self.direction = None
        self.previous_move = None
        self.preparation_actions = None
        self.actions = []

    def action(self, kind, **kwargs):
        return c.Action(kind, schema=c.EXPANDED_ACTION_SCHEMA, **kwargs)

    def choose(self, frame):
        observation = frame.observation
        context = observation.context
        if self.hero is None:
            self.hero = observation.heroes[0].handle
            self.town = observation.towns[0].handle
            self.direction = 1 if observation.heroes[0].position[0] < observation.size[0] / 2 else -1
        if context.kind == "dialog":
            panel = observation.window.dialog_panel
            # A displayed modal query is mandatory. Prefer its affirmative
            # button, otherwise the first actual visible choice.
            choice = next((x for x in panel.choices if x.kind in ("accept", "pursue")), panel.choices[0])
            return self.action("choose_dialog", subject=context.subject, selection=choice.ref), "acknowledge_displayed_dialog"
        if context.kind == "capability":
            panel = observation.window.capability_panel
            if not panel.mandatory:
                return self.action("close_capability", subject=context.subject), "close_optional_capability"
            choice = panel.choices[0]
            return self.action(choice.kind, subject=context.subject, selection=choice.ref), "answer_displayed_mandatory_capability"
        if self.stage == "open":
            self.stage = "purchase"
            return self.action("open_town", subject=self.town), "open_own_town"
        if self.stage == "purchase":
            panel = observation.window.town_panel
            if context.panel == "recruitment":
                recruit = panel.recruitment
                if recruit.control_ref not in self.recruited_controls and recruit.slider_max:
                    self.recruited_controls.add(recruit.control_ref)
                    return self.action("recruit", subject=context.subject, selection=recruit.selected_offer_ref,
                        count=recruit.slider_max, destination=recruit.destination.ref), "buy_displayed_available_troops"
                self.recruited_controls.add(recruit.control_ref)
                return self.action("back", subject=context.subject), "back_to_town_armies"
            control = next((x for x in panel.controls if x.panel == "recruitment" and x.ref not in self.recruited_controls), None)
            if control:
                return self.action("open_panel", subject=context.subject, panel="recruitment", selection=control.ref), "open_displayed_dwelling"
            armies = observation.window.armies
            source = next((x for x in armies if x.role in ("town_garrison", "garrison_hero")), None)
            visitor = next((x for x in armies if x.role == "visiting_hero"), None)
            exchange = c.ArmyExchange(source, visitor) if source else None
            if exchange and exchange.destination:
                for source in exchange.source.slots:
                    if not source.count:
                        continue
                    destinations = sorted(exchange.destination.slots,
                        key=lambda slot: (slot.creature != source.creature, slot.slot))
                    for destination in destinations:
                        if exchange.maximum(source.ref, destination.ref) == source.count:
                            kind = "merge_stacks" if destination.count else "swap_stacks"
                            return self.action(kind, subject=context.subject,
                                selection=source.ref, destination=destination.ref), "give_bought_troops_to_starting_hero"
            self.stage = "rush"
            self.preparation_actions = len(self.actions) + 1
            return self.action("close"), "close_town_and_start_rush"
        if context.kind != "map":
            return self.action("close"), "close_window_during_rush"
        hero = next((x for x in observation.heroes if x.handle == self.hero), None)
        if hero is None:
            # After losing the attacking hero, do not improvise a second army.
            # Natural opponent play may still take the remaining town.
            return self.action("end_turn"), "wait_for_natural_outcome_after_hero_loss"
        if hero.movement == 0:
            return self.action("end_turn"), "replenish_movement"
        if self.previous_move == (frame.day, hero.position, hero.movement):
            return self.action("end_turn"), "end_turn_after_route_made_no_progress"
        enemies = [x for x in observation.objects if x.owner == "opponent" and x.kind in (34, 98)]
        # Prefer a visible enemy hero; defeating it is distinct from winning
        # the whole game. A town is targeted only while it is actually seen.
        enemies.sort(key=lambda x: (x.kind != 34, abs(x.position[0] - hero.position[0]) + abs(x.position[1] - hero.position[1])))
        if enemies:
            target = enemies[0].position
            label = "attack_visible_enemy_hero" if enemies[0].kind == 34 else "capture_visible_enemy_town"
        else:
            # Explore the current visible frontier in the direction away from
            # our starting edge. No target coordinate comes from hidden state.
            tiles = [x for x in observation.tiles if not x.blocked and not x.visitable and x.position != hero.position]
            target = max(tiles, key=lambda x: (self.direction * x.position[0], -abs(x.position[1] - hero.position[1]))).position
            label = "explore_visible_frontier"
        self.previous_move = (frame.day, hero.position, hero.movement)
        return self.action("move_to", subject=hero.handle, x=target[0], y=target[1]), label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--hard-stop", type=float, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "artifacts/runs") or "runs" in output.parts:
        raise ValueError("Diagnostic output must be outside training run directories")
    output.mkdir(parents=True, exist_ok=False)
    deadline_wall = min(time.time() + args.seconds, args.hard_stop)
    deadline = time.monotonic() + deadline_wall - time.time()
    report = {"schema": "heroes-recruit-rush-diagnostic-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False, "training_samples": 0, "optimization_eligible": False,
        "policy_kind": "hand-scripted-diagnostic", "seat": args.seat, "seed": args.seed,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(), "deadline_wall": deadline_wall,
        "max_actions": args.max_actions, "status": "started", "minimum_actions_claim": False}
    write(output / "report.json", report)
    launcher = DeadlineVcmiLauncher(args.runtime / "vcmiclient", args.data, args.map,
        deadline_monotonic=deadline, deadline_wall=deadline_wall, evidence_path=output / "native-deadline.jsonl")
    color = ("red", "blue")[args.seat]
    env = BuiltinHeadlessEnv(launcher, output, learner_player=args.seat,
        interface=c.EXPANDED_ENVIRONMENT,
        limits=Limits(max_days=365, max_actions=args.max_actions, startup_seconds=25,
            watchdog_seconds=25, shutdown_seconds=10, bridge_timeout=25),
        agents={color: {"kind": "hand-scripted-recruit-rush-diagnostic", "training": False}})
    plan = Rush()
    state = None
    try:
        with until(deadline, deadline_wall, reserve=35):
            state = env.reset(1, args.seed)
            with (output / "actions.jsonl").open("x") as log:
                while not state.done:
                    frame = c.decode_frame(state.record)
                    if len(plan.actions) >= args.max_actions:
                        state = env.truncate("collection_window")
                        break
                    action, label = plan.choose(frame)
                    if not frame.offers.permits(action):
                        raise ValueError("Script proposed an unoffered action: " + str(action))
                    entry = {"index": len(plan.actions) + 1, "day": frame.day, "decision_seq": frame.seq,
                        "label": label, "action": asdict(action), "context": asdict(frame.observation.context),
                        "heroes": [asdict(x) for x in frame.observation.heroes],
                        "visible_opponents": [asdict(x) for x in frame.observation.objects if x.owner == "opponent"],
                        "window": asdict(frame.observation.window) if frame.observation.window else None,
                        "resources": frame.observation.resources, "training_input": False}
                    plan.actions.append(entry)
                    log.write(json.dumps(entry, sort_keys=True) + "\n")
                    log.flush()
                    state = env.step(action)
            report.update(status=state.status, termination_reason=state.termination_reason,
                outcomes=state.outcomes, rewards=state.rewards)
        with until(deadline, deadline_wall, reserve=3):
            replay = verify_episode(output / "episode-000001", binary=args.runtime / "vcmireplay",
                data=args.data, timeout=max(1, min(deadline - time.monotonic(), deadline_wall - time.time()) - 5))
        report["replay"] = {key: replay.get(key) for key in ("verified", "error", "native_opponent_requests", "native")}
        report["action_evidence"] = summarize_episode(output / "episode-000001", plan.actions, color)
        if not replay.get("verified"):
            raise ValueError("Native replay did not verify; no diagnostic outcome is accepted")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "diagnostic_failed"
        if state is not None and not state.done:
            try:
                env._note_problem("protocol_error", "Scripted diagnostic failed: " + report["error"])
                state = env._finish(None)
                report["native_status"] = state.status
            except Exception as final_error:
                report["finalization_error"] = f"{type(final_error).__name__}: {final_error}"
    finally:
        try:
            env.close()
            launcher.close_owned_groups(timeout=2)
            launcher.assert_closed()
            report["owned_processes_closed"] = True
        except Exception as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "diagnostic_failed"
        report.update(completed_at=datetime.now(timezone.utc).isoformat(),
            typed_actions=len(plan.actions), preparation_actions=plan.preparation_actions,
            action_counts=dict(Counter(x["action"]["kind"] for x in plan.actions)),
            first_enemy_attack_action=next((x["index"] for x in plan.actions if x["label"] == "attack_visible_enemy_hero"), None),
            first_town_attack_action=next((x["index"] for x in plan.actions if x["label"] == "capture_visible_enemy_town"), None))
        write(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "replay"}, sort_keys=True))
    return 0 if diagnostic_accepted(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
