#!/usr/bin/env python3
"""Bounded, real-VCMI validation of the contextual interface; never training.

Mechanics cases select coordinates only from their player's observations to
exercise particular interface conditions. They are evaluator probes, not a
product policy, demonstrations, exploration strategy or learner dataset. The
separate random cohort selects uniform action types and uniform parameters;
no probe action, engine-only fact, reward shaping or PPO update enters it.

This driver deliberately requires the native contextual implementation. It does
not replace an old engine, emulate windows, fabricate histories, or mock VCMI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("red", "blue")
DEFAULT_CASES = ("navigation", "mechanics", "day-cutoff", "kill-after-action", "kill-before-truncate")
CASES = DEFAULT_CASES + ("closed-window-battle",)


class ProbeFailure(RuntimeError):
    pass


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure(condition, message):
    if not condition:
        raise ProbeFailure(message)


class CaseRun:
    def __init__(self, name, args, env_module, contract, auditor, seed):
        self.name, self.args, self.api, self.contract, self.auditor = name, args, env_module, contract, auditor
        self.directory = args.output / name
        self.directory.mkdir()
        self.started = time.monotonic()
        self.checks = []
        self.attempts = []
        self.snapshots = {color: [] for color in COLORS}
        self.initial = {}
        self.state = None
        self.end = None
        self.audit_attempted = False
        self.history_audit_passed = False
        self.route_stops = []
        self.commands = 0
        self.seed = seed
        self.records = self.directory / "attempts.jsonl"
        self.records.touch(exist_ok=False)
        days = 1 if name == "day-cutoff" else args.max_days
        limits = env_module.Limits(max_days=days, max_actions=args.max_decisions,
                                   watchdog_seconds=args.watchdog, startup_seconds=args.startup_seconds,
                                   shutdown_seconds=args.watchdog, bridge_timeout=min(300, max(10, int(args.watchdog))))
        self.env = env_module.HeadlessSelfPlayEnv(
            env_module.VcmiLauncher(args.binary, args.data, args.map, "info"), self.directory,
            limits=limits, ledger=True, replay=True, interface=contract.ENVIRONMENT,
            agents={color: {"kind": "contextual-interface-validation", "training": False,
                            "case": name, "seed": seed} for color in COLORS})

    def check(self, name, status, **evidence):
        ensure(status in ("passed", "failed", "unverified"), "Invalid probe status")
        self.checks.append({"check": name, "status": status, **evidence})

    def start(self):
        self.state = self.env.reset(episode=1, seed=self.seed)
        if self.state.done:
            self.end = self.state
            raise ProbeFailure("Engine ended before the first contextual decision")
        self.inspect_request()

    def history(self, color):
        frames = self.env.history_v2(color)
        ensure(type(frames) is list and all(type(frame) is dict for frame in frames),
               "history_v2 must return JSON frames, not raw engine messages")
        return frames

    def inspect_request(self):
        request = self.state
        ensure(not request.done, "Expected a live request")
        ensure(request.color in COLORS and request.player == COLORS.index(request.color), "Cross-player request")
        ensure(request.reward == 0, "Ordinary request must have zero reward")
        ensure(type(request.frames) is list and request.frames, "Request must include contextual history frames")
        decoded = [self.contract.decode_frame(raw) for raw in request.frames]
        ensure(all(frame.player == request.player for frame in decoded), "Request mixed player histories")
        frame = decoded[-1]
        ensure(frame.kind == "decision", "Request.frames must end with its actual decision")
        full = self.history(request.color)
        ensure(full and self.contract.decode_frame(full[-1]) == frame, "Request does not bind current player history")
        previous = self.snapshots[request.color]
        ensure(len(full) >= len(previous) and full[:len(previous)] == previous, "Delivered history prefix changed")
        self.snapshots[request.color] = json.loads(json.dumps(full))
        self.initial.setdefault(request.color, frame)
        ensure(frame.offers == self.contract.Offers.from_observation(frame.observation), "Offers differ from public support")
        ensure(frame.observation.context.kind != "map" or frame.observation.window is None, "Closed window refreshed in map observation")
        return frame

    def call(self, action, label, *, expect_rejection=False):
        frame = self.inspect_request()
        request = self.state
        before = self.history(request.color)
        entry = {"index": len(self.attempts), "label": label, "player": request.player, "color": request.color,
                 "decision_seq": frame.seq, "day": frame.day, "context": asdict(frame.observation.context),
                 "action": asdict(action), "training_input": False}
        self.attempts.append(entry)
        try:
            self.commands += 1
            next_state = self.env.step(action)
            self.state = next_state
            if expect_rejection:
                if next_state.done:
                    self.end = next_state
                raise ProbeFailure("Invalid/stale action was accepted by the environment API")
            if next_state.done:
                self.end = next_state
            else:
                self.inspect_request()
            after = self.history(request.color)
            ensure(after[:len(before)] == before, "Submitted action mutated its previously delivered prefix")
            additions = [self.contract.decode_frame(raw) for raw in after[len(before):]]
            results = [record.result for record in additions if record.kind == "result" and record.decision_seq == frame.seq]
            entry.update(api_rejected=False, appended_frames=len(additions),
                         result=[asdict(result) for result in results],
                         native_issued=sum(record.kind == "issued" for record in additions),
                         native_accepted=sum(record.kind == "acknowledged" and record.accepted is True for record in additions),
                         events=[asdict(record.event) for record in additions if record.event])
            result = results[-1] if results else None
            if action.kind == "move_to":
                stop_evidence = route_stop_evidence(additions, result)
                self.route_stops.append(stop_evidence)
                entry["route_stop_evidence"] = stop_evidence
                for stop_kind, evidence in stop_evidence.items():
                    if evidence["status"] == "failed":
                        self.check(f"native_{stop_kind}_stop_evidence", "failed", evidence=evidence,
                                   decision_seq=frame.seq)
            return frame, additions, result
        except ValueError as error:
            entry.update(api_rejected=True, error=str(error))
            if not expect_rejection:
                raise ProbeFailure(f"Valid offered action raised ValueError: {error}") from error
            ensure(self.env.request is request, "Rejected call changed current request identity")
            ensure(self.history(request.color) == before, "Rejected call changed player history")
            self.check(label, "passed", decision_seq=frame.seq, error=str(error))
            return frame, [], None
        finally:
            with self.records.open("a") as stream:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def live(self):
        if self.state is None or self.state.done:
            return False
        if self.commands >= self.args.max_decisions:
            self.check("probe_decision_budget", "unverified", limit=self.args.max_decisions)
            self.end = self.env.truncate("collection_window")
            self.state = self.end
            return False
        if time.monotonic() - self.started > self.args.case_seconds:
            # An expired watchdog is a technical failure, never a fabricated horizon.
            self.check("probe_wall_watchdog", "failed", limit_seconds=self.args.case_seconds)
            self.kill_owned_process()
            self.end = self.env.truncate("collection_window")
            self.state = self.end
            ensure(self.end.status == "failed", "Watchdog failure was misclassified as cutoff")
            return False
        return True

    def kill_owned_process(self):
        process = self.env.process
        ensure(process is not None and process.poll() is None, "Failure injection requires live owned engine process")
        os.kill(process.pid, signal.SIGKILL)
        exited = process.wait(timeout=15)
        ensure(exited == -signal.SIGKILL, "Owned engine was not killed as expected")
        self.check("owned_process_killed", "passed", pid=process.pid, returncode=exited)

    def finish(self):
        if self.state is not None and not self.state.done:
            self.end = self.env.truncate("collection_window")
            self.state = self.end
        if self.state is not None and self.state.done:
            self.end = self.state

    def persist_and_audit(self):
        ensure(not self.audit_attempted, "Evidence audit already attempted; preserve original files")
        self.audit_attempted = True
        ensure(self.end is not None, "No finalized episode is available")
        history_paths = {}
        for color in COLORS:
            frames = self.history(color)
            previous = self.snapshots[color]
            ensure(frames[:len(previous)] == previous, f"Finalization mutated the {color} delivered prefix")
            # Preserve independently read API output separately; never repair producer files or its manifest.
            path = self.directory / f"observed-history-v2-{color}.jsonl"
            with path.open("x") as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, sort_keys=True, separators=(",", ":")) + "\n")
            history_paths[color] = path
        manifest_path = self.env.episode_dir / "manifest.json"
        ensure(manifest_path.is_file(), "Environment did not persist its manifest")
        manifest = json.loads(manifest_path.read_text())
        declarations = manifest.get("players", {})
        producer_paths = {}
        for color in COLORS:
            declaration = declarations.get(color, {}).get("history_v2", {})
            named = declaration.get("path")
            candidates = ([self.env.episode_dir / named] if isinstance(named, str) else []) + [
                self.env.episode_dir / "agents" / color / "history-v2.jsonl",
                self.env.episode_dir / f"history-v2-{color}.jsonl",
            ]
            matches = [path for path in candidates if path.is_file() and sha(path) == declaration.get("sha256")]
            ensure(matches, f"No persisted {color} history file matches producer manifest")
            producer = matches[0]
            producer_records = [json.loads(line) for line in producer.read_text().splitlines()]
            ensure(producer_records == self.history(color), f"Persisted {color} history differs from API")
            producer_paths[color] = producer
        report = self.auditor.audit(producer_paths["red"], producer_paths["blue"], manifest_path)
        write(self.directory / "history-audit.json", report)
        ensure(report["consistent"] and report["manifest_bound"], "Independent history/manifest audit failed")
        self.check("independent_history_manifest_audit", "passed", report=str(self.directory / "history-audit.json"))
        if self.end.status == "completed":
            ensure(set(self.end.rewards.values()) == {-1, 1}, "Natural result lacks complementary terminal rewards")
            ensure(report["scoring_binding_verified"], "Natural reward lacks evidence binding")
        else:
            ensure(all(value is None for value in self.end.rewards.values()), "Unscored episode has rewards")
            ensure(all(value is None for value in self.end.outcomes.values()), "Unscored episode has outcomes")
            if self.end.status == "failed":
                ensure(not report["learner_eligible"], "Technical/unsupported failure became learner eligible")
                ensure(all(player["transitions"]["effective_bootstrap_allowed"] == 0 for player in report["players"].values()),
                       "Technical failure bootstraps value")
        self.history_audit_passed = True
        return manifest_path, report

    def summary(self, error=None):
        status = "failed" if error or any(row["status"] == "failed" for row in self.checks) else (
            "unverified" if any(row["status"] == "unverified" for row in self.checks) else "passed")
        return {"case": self.name, "probe_status": status, "engine_seed": self.seed,
                "episode_status": self.end.status if self.end is not None else None,
                "termination_reason": self.end.termination_reason if self.end is not None else None,
                "error": error, "checks": self.checks, "attempts": len(self.attempts),
                "history_audit_passed": self.history_audit_passed,
                "reported_outcomes": self.end.outcomes if self.end is not None else None,
                "reported_rewards": self.end.rewards if self.end is not None else None,
                "elapsed_seconds": round(time.monotonic() - self.started, 3), "training_performed": False,
                "manifest": str(self.env.episode_dir / "manifest.json") if self.env.episode_dir else None}


def route_stop_evidence(frames, result):
    """Check one policy action's persisted cause and cessation, not its label alone."""
    report = {name: {"status": "unverified", "reason": "No matching completed action/cause"}
              for name in ("encounter", "interruption")}
    matches = [index for index, frame in enumerate(frames)
               if frame.kind == "result" and result is not None and frame.result == result]
    if not matches:
        return report
    result_index = matches[-1]
    result_frame = frames[result_index]
    decision = result_frame.decision_seq
    starts = [index for index, frame in enumerate(frames[:result_index])
              if frame.kind == "consumed" and frame.decision_seq == decision]
    start = starts[-1] if starts else 0
    causes = {
        "encounter": [index for index in range(start, result_index)
                      if frames[index].event is not None and frames[index].event.kind in
                      ("visit", "battle_start", "battle_complete", "notice")],
        "interruption": [index for index in range(start, result_index)
                         if frames[index].kind == "acknowledged" and frames[index].decision_seq == decision
                         and frames[index].accepted is False],
    }
    for name, indices in causes.items():
        expected = "interaction" if name == "encounter" else "interrupted"
        if not indices:
            report[name]["reported_status"] = result.status
            continue
        cause = indices[0]
        issued_after = [frame.seq for frame in frames[cause + 1:]
                        if frame.kind == "issued" and frame.decision_seq == decision]
        status = "failed" if issued_after else "passed" if result.status == expected else "unverified"
        report[name] = {"status": status, "cause_seq": frames[cause].seq, "result_seq": result_frame.seq,
                        "reported_status": result.status, "issued_after_cause": issued_after,
                        "reason": "Native route issued another step after a stop cause" if issued_after else
                                  "Observed cause and no further route issuance" if status == "passed" else
                                  "Observed cause has a different result status; stop coverage not established"}
    return report


def stop_coverage(reports, kind):
    values = [report[kind] for report in reports]
    status = "failed" if any(value["status"] == "failed" for value in values) else (
        "passed" if any(value["status"] == "passed" for value in values) else "unverified")
    return status, [value for value in values if value["status"] != "unverified"][:4]


def validated_natural_count(rows):
    """Count accepted native evidence, separately from reported episode outcomes."""
    return sum(row.get("case", "").startswith("random-") and row.get("episode_status") == "completed"
               and row.get("probe_status") == "passed" and row.get("history_audit_passed") is True for row in rows)


def paired_comparison_status(rows, comparisons):
    """A paired equality claim requires both complete independently audited runs."""
    if any(row.get("probe_status") == "failed" for row in rows) or any(
            comparison.get("status") == "failed" for comparison in comparisons):
        return "failed"
    if len(rows) != 2 or not comparisons or any(row.get("probe_status") != "passed"
            or row.get("history_audit_passed") is not True for row in rows) or any(
            comparison.get("status") != "passed" for comparison in comparisons):
        return "unverified"
    return "passed"


def full_support(run, frame):
    obs, offers = frame.observation, frame.offers
    if obs.context.kind != "map":
        return
    for hero in obs.heroes:
        ensure(all(offers.permits(run.contract.Action("move_to", hero.handle, x, y))
                   for y in range(obs.size[1]) for x in range(obs.size[0])), "Coordinate support contains a reachability mask")
    if obs.heroes:
        visible = {tile.position[:2] for tile in obs.tiles}
        unknown = next(((x, y) for y in range(obs.size[1]) for x in range(obs.size[0]) if (x, y) not in visible), None)
        blocked = next((tile.position[:2] for tile in obs.tiles if tile.blocked), None)
        run.check("full_coordinate_support", "passed", player=frame.player, coordinates=obs.size[0] * obs.size[1],
                  unknown_included=unknown, blocked_included=blocked,
                  zero_movement=any(hero.movement == 0 for hero in obs.heroes))


def navigation(run):
    phase = {color: 0 for color in COLORS}
    opened = {color: {} for color in COLORS}
    opening_refs = {}
    incomes = set()
    first_resources = {}
    movement_changes = set()
    for color in COLORS:
        run.check(f"{color}_underlying_window_change_while_closed", "unverified",
                  reason="This navigation case does not edit armies or inject hidden state; field absence/prefix immutability are checked separately.")
    while run.live():
        frame = run.inspect_request()
        color, obs = run.state.color, frame.observation
        first_resources.setdefault(color, obs.resources)
        if obs.resources != first_resources[color]:
            incomes.add(color)
        if phase[color] == 0:
            full_support(run, frame)
            ensure(obs.heroes and obs.towns, "Navigation fixture requires an owned hero and town")
            invalid = run.contract.Action("move_to", obs.heroes[0].handle, obs.size[0], 0)
            run.call(invalid, "out_of_bounds_rejected_unchanged", expect_rejection=True)
            opening_refs[color] = run.contract.Action("open_hero", obs.heroes[0].handle)
            run.call(opening_refs[color], "map_to_hero")
            phase[color] = 1
        elif phase[color] == 1:
            ensure(obs.context.kind == "hero", "Open hero did not create hero context")
            opened[color]["hero"] = asdict(obs.window)
            run.call(opening_refs[color], "stale_map_action_rejected_in_hero", expect_rejection=True)
            run.call(run.contract.Action("close"), "hero_to_map")
            phase[color] = 2
        elif phase[color] == 2:
            ensure(obs.window is None, "Closed hero window remains current")
            run.call(run.contract.Action("open_town", obs.towns[0].handle), "map_to_town")
            phase[color] = 3
        elif phase[color] == 3:
            ensure(obs.context.kind == "town", "Open town did not create town context")
            opened[color]["town"] = asdict(obs.window)
            run.call(run.contract.Action("close"), "town_to_map")
            phase[color] = 4
        elif phase[color] == 4:
            ensure(obs.window is None, "Closed town window remains current")
            run.check(f"{color}_map_hero_map_town_map", "passed", window_contexts=sorted(opened[color]))
            run.call(run.contract.Action("close"), "stale_close_rejected_on_map", expect_rejection=True)
            phase[color] = 5
        elif phase[color] == 5:
            hero = obs.heroes[0]
            target = next((tile.position for tile in obs.tiles
                           if not tile.blocked and not tile.visitable and tile.position != hero.position
                           and abs(tile.position[0] - hero.position[0]) + abs(tile.position[1] - hero.position[1]) == 1), None)
            if target is not None:
                _, _, result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "map_movement_after_close")
                if result is not None and result.movement != hero.movement:
                    movement_changes.add(color)
            phase[color] = 6
        else:
            ensure(obs.window is None, "Inactive window fields refreshed across map turns")
            if len(incomes) == 2 and len(movement_changes) == 2 and min(phase.values()) >= 6:
                break
            run.call(run.contract.Action("end_turn"), "advance_income_with_windows_closed")
    for color in COLORS:
        run.check(f"{color}_closed_window_current_fields_absent", "passed" if phase[color] >= 4 else "unverified")
        run.check(f"{color}_map_resources_refresh_when_closed", "passed" if color in incomes else "unverified")
        run.check(f"{color}_map_movement_refresh_when_closed", "passed" if color in movement_changes else "unverified")
    if set(run.initial) == set(COLORS):
        tiles = [{tile.position for tile in run.initial[color].observation.tiles} for color in COLORS]
        extent = run.initial["red"].observation.size
        ensure(tiles[0] != tiles[1] and all(len(seen) < extent[0] * extent[1] for seen in tiles),
               "Initial player fog views are identical or fully revealed")
        run.check("separate_native_initial_fog", "passed", visible_tiles=[len(seen) for seen in tiles])
    run.check("paired_hidden_world_invariance", "unverified", reason="Requires a separate paired native fixture/source audit; unequal initial fog alone is insufficient.")


def mechanics(run):
    target_color = run.state.color
    attempted_frontiers = set()
    revealed = set()
    explored = False
    continued = False
    partial = False
    zero = False
    zero_support = False
    partial_endpoint = None
    checked_continuation = False
    phase = "noop"
    movement_attempts = 0
    while run.live():
        frame = run.inspect_request()
        obs, color = frame.observation, run.state.color
        if color != target_color:
            run.call(run.contract.Action("end_turn"), "other_player_mechanical_turn")
            continue
        ensure(obs.context.kind == "map", "Movement probe unexpectedly left map context")
        if not obs.heroes:
            run.check("mechanics_hero_available", "unverified", reason="Owned hero became unavailable")
            break
        hero = obs.heroes[0]
        known = {tile.position for tile in obs.tiles}
        passable = [tile.position for tile in obs.tiles if not tile.blocked and not tile.visitable]
        full_support(run, frame)
        if phase == "noop":
            _, additions, result = run.call(run.contract.Action("move_to", hero.handle, *hero.position[:2]), "same_position_noop")
            ensure(result is not None and result.status == "reached" and result.destination_reached is True
                   and result.position == hero.position and result.movement == hero.movement,
                   "Same-position command did not preserve actual position/points")
            ensure(not any(record.kind == "issued" for record in additions), "Same-position noop issued a native move")
            run.check("same_position_noop", "passed")
            phase = "unknown"
            continue
        if phase == "unknown":
            target = next(((x, y, 0) for y in range(obs.size[1]) for x in range(obs.size[0]) if (x, y, 0) not in known), None)
            if target is None:
                run.check("unknown_target_no_route", "unverified", reason="No unknown coordinate in current view")
            else:
                _, additions, result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "unknown_target")
                ensure(result is not None and result.status == "no_route_known" and result.position == hero.position
                       and result.movement == hero.movement and not any(record.kind == "issued" for record in additions),
                       "Unknown target leaked/planned movement or changed actual state")
                run.check("unknown_target_no_route", "passed", target=list(target))
            phase = "frontier"
            continue
        if phase == "frontier":
            frontier = [pos for pos in passable if pos != hero.position and pos not in attempted_frontiers and any(
                0 <= pos[0] + dx < obs.size[0] and 0 <= pos[1] + dy < obs.size[1]
                and (pos[0] + dx, pos[1] + dy, 0) not in known
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
            if not frontier or len(attempted_frontiers) >= run.args.frontier_attempts:
                phase = "partial"
                continue
            target = sorted(frontier)[0]
            attempted_frontiers.add(target)
            _, additions, result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "known_frontier_exploration_probe")
            new = {tile.position for record in additions if record.observation for tile in record.observation.tiles} - known
            if new:
                explored, revealed = True, new
                phase = "continued"
                run.check("known_frontier_reveals_new_terrain", "passed", target=list(target), newly_observed=len(new))
        elif phase == "continued":
            candidates = sorted(pos for pos in passable if pos in revealed and pos != hero.position)
            if not candidates:
                run.check("newly_known_target_continued_exploration", "unverified", reason="New observations contained no supported passable target")
                phase = "partial"
                continue
            target = candidates[0]
            _, additions, result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "newly_known_destination_probe")
            continued = result is not None and result.position is not None and result.position != hero.position
            run.check("newly_known_target_continued_exploration", "passed" if continued else "unverified", target=list(target),
                      stop_status=result.status if result else None)
            phase = "partial"
        elif phase == "advance_after_partial":
            run.call(run.contract.Action("end_turn"), "discard_partial_route_before_next_turn")
            phase = "verify_after_partial"
            continue
        elif phase == "verify_after_partial":
            ensure(frame.day > partial_endpoint["day"], "End turn did not reach a later own decision day")
            ensure(hero.position == partial_endpoint["position"], "A discarded route moved automatically on the next turn")
            run.check("no_automatic_next_turn_route_continuation", "passed",
                      prior_target=list(partial_endpoint["target"]), actual_position=list(hero.position), next_day=frame.day)
            checked_continuation = True
            phase = "partial"
            continue
        elif phase == "partial":
            targets = sorted(pos for pos in passable if pos != hero.position)
            if not targets or movement_attempts >= run.args.movement_attempts:
                break
            # Alternate known endpoints to exercise point consumption. This finite
            # evaluator sequence is kept outside product policies and training.
            target = targets[-1] if movement_attempts % 2 == 0 else targets[0]
            movement_attempts += 1
            _, additions, result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "movement_point_boundary_probe")
            if result is not None:
                partial |= (result.status == "movement_limit" and result.destination_reached is False
                            and result.position != hero.position and any(record.kind == "acknowledged" and record.accepted for record in additions))
                zero |= result.movement == 0
                if partial and partial_endpoint is None:
                    partial_endpoint = {"position": result.position, "target": target, "day": frame.day}
                    phase = "advance_after_partial"
                if result.movement == 0:
                    next_frame = run.inspect_request() if not run.state.done and run.state.color == color else None
                    if next_frame:
                        full_support(run, next_frame)
                        zero_support = True
                        _, zero_additions, zero_result = run.call(run.contract.Action("move_to", hero.handle, *target[:2]), "zero_movement_target_support")
                        ensure(zero_result is not None and zero_result.status in ("reached", "movement_limit")
                               and not any(record.kind == "issued" for record in zero_additions),
                               "Zero-point movement planned/executed native commands")
                    if phase != "advance_after_partial":
                        break
        else:
            raise ProbeFailure("Unknown mechanics phase")
    for name, covered in (("known_frontier_exploration", explored), ("continued_exploration", continued),
                          ("partial_current_turn_route", partial), ("zero_point_all_coordinate_support", zero_support)):
        run.check(name, "passed" if covered else "unverified", bounded_attempts=movement_attempts)
    for label, kind in (("encounter_stops_route", "encounter"), ("native_interruption_stops_route", "interruption")):
        status, evidence = stop_coverage(run.route_stops, kind)
        run.check(label, status, evidence=evidence)
    if not checked_continuation:
        run.check("no_automatic_next_turn_route_continuation", "unverified",
                  reason="No surviving partial-route endpoint was observed across the next native own turn.")
    run.check("zero_movement_points_observed", "passed" if zero else "unverified")


def closed_window_battle(run):
    """A deliberate evaluator battle fixture; never a reusable agent strategy."""
    color = run.state.color
    first = run.inspect_request()
    ensure(first.observation.context.kind == "map" and first.observation.heroes,
           "Closed-window fixture needs an owned hero on the map")
    hero_handle = first.observation.heroes[0].handle
    run.call(run.contract.Action("open_hero", hero_handle), "battle_fixture_open_hero")
    if run.state.done:
        run.check("closed_window_battle_reopen", "unverified", reason="Episode ended while opening initial window")
        return
    opened = run.inspect_request()
    ensure(opened.observation.context == run.contract.Context("hero", hero_handle), "Wrong hero window opened")
    original_window = asdict(opened.observation.window)
    original_prefix = run.history(color)
    run.call(run.contract.Action("close"), "battle_fixture_close_hero")
    closed_start = len(run.history(color))
    frontier_attempts = 0
    battle_attempts = 0
    tried_frontiers = set()
    battle_completed = False
    needs_next_turn = False
    while run.live():
        frame = run.inspect_request()
        obs = frame.observation
        if run.state.color != color:
            run.call(run.contract.Action("end_turn"), "battle_fixture_other_player_turn")
            continue
        ensure(obs.window is None, "Closed hero fields refreshed during native battle fixture")
        hero = next((entry for entry in obs.heroes if entry.handle == hero_handle), None)
        if hero is None:
            run.check("closed_window_battle_reopen", "unverified", reason="Hero became unavailable; do not open a replacement")
            break
        if battle_completed:
            break
        if needs_next_turn:
            run.call(run.contract.Action("end_turn"), "battle_fixture_explicit_turn_end")
            needs_next_turn = False
            continue
        monsters = sorted((obj for obj in obs.objects if obj.kind == 54 and obj.owner == "neutral"),
                          key=lambda obj: obj.position)
        if monsters and battle_attempts < run.args.movement_attempts:
            target = monsters[0].position
            battle_attempts += 1
            _, additions, result = run.call(run.contract.Action("move_to", hero_handle, *target[:2]),
                                            "battle_fixture_observed_neutral")
            battle_completed |= any(record.event is not None and record.event.kind == "battle_complete" for record in additions)
            if result is not None and result.status == "movement_limit":
                needs_next_turn = True
            elif result is not None and result.status == "no_route_known":
                run.check("closed_window_battle_route", "unverified", reason="Observed neutral has no supported known route")
                break
        elif frontier_attempts < run.args.frontier_attempts:
            known = {tile.position for tile in obs.tiles}
            candidates = sorted(tile.position for tile in obs.tiles if not tile.blocked and not tile.visitable
                                and tile.position != hero.position and tile.position not in tried_frontiers and any(
                                    0 <= tile.position[0] + dx < obs.size[0] and 0 <= tile.position[1] + dy < obs.size[1]
                                    and (tile.position[0] + dx, tile.position[1] + dy, 0) not in known
                                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))))
            if not candidates:
                break
            target = candidates[0]
            tried_frontiers.add(target)
            frontier_attempts += 1
            _, additions, result = run.call(run.contract.Action("move_to", hero_handle, *target[:2]),
                                            "battle_fixture_known_frontier")
            battle_completed |= any(record.event is not None and record.event.kind == "battle_complete" for record in additions)
            if result is not None and result.status == "movement_limit":
                needs_next_turn = True
        else:
            break
    history = run.history(color)
    ensure(history[:len(original_prefix)] == original_prefix, "Battle rewrote the earlier open-window snapshot")
    closed_frames = [run.contract.decode_frame(raw) for raw in history[closed_start:]]
    ensure(all(frame.observation is None or frame.observation.window is None for frame in closed_frames),
           "Army/window data refreshed while the hero window was closed")
    run.check("closed_window_snapshot_immutable_through_battle_attempts", "passed", frontier_attempts=frontier_attempts,
              battle_attempts=battle_attempts, battle_completed=battle_completed)
    run.check("closed_window_native_battle_completed", "passed" if battle_completed else "unverified")
    encounter_status, encounter_evidence = stop_coverage(run.route_stops, "encounter")
    run.check("closed_window_encounter_stop_observed", encounter_status, evidence=encounter_evidence)
    if not battle_completed or run.state.done or run.state.color != color:
        run.check("closed_window_changed_army_then_reopen", "unverified",
                  reason="No surviving own post-battle decision available within the declared bounds")
        return
    current = run.inspect_request()
    if hero_handle not in {hero.handle for hero in current.observation.heroes}:
        run.check("closed_window_changed_army_then_reopen", "unverified", reason="Battle removed the hero")
        return
    run.call(run.contract.Action("open_hero", hero_handle), "battle_fixture_reopen_same_hero")
    if run.state.done:
        run.check("closed_window_changed_army_then_reopen", "unverified", reason="Episode ended before reopened window observation")
        return
    reopened = run.inspect_request()
    ensure(reopened.observation.context == run.contract.Context("hero", hero_handle), "Reopened wrong hero")
    new_window = asdict(reopened.observation.window)
    before_units = sum(stack["count"] for stack in original_window["army"])
    after_units = sum(stack["count"] for stack in new_window["army"])
    casualties = before_units - after_units
    ensure(run.history(color)[:len(original_prefix)] == original_prefix, "Reopening refreshed an old window frame")
    run.check("closed_window_changed_army_then_reopen", "passed" if casualties > 0 else "unverified",
              before_units=before_units, after_units=after_units, casualties=max(0, casualties),
              reason=None if casualties > 0 else "Real battle caused no observed casualties; changed-field freshness remains unverified")
    run.call(run.contract.Action("close"), "battle_fixture_return_to_map")


def normalize_permitted_frames(frames):
    """Rename player-local routing handles and episode identity only.

    Player sequence, revision, day, facts, events and command acknowledgements
    remain intact: a hidden-world-dependent difference in them must not vanish.
    """
    handles = {}

    def handle(value):
        if value not in handles:
            handles[value] = f"observed-{len(handles) + 1}"
        return handles[value]

    def walk(value, key=None):
        if key == "episode":
            return "paired-episode"
        if key in ("handle", "subject") and isinstance(value, str):
            return handle(value)
        if key in ("hero_refs", "town_refs") and isinstance(value, list):
            return [handle(item) for item in value]
        if isinstance(value, dict):
            return {name: walk(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(frames)


def visible_fixture_opponent(frame, positions):
    observation = frame.get("observation") if isinstance(frame, dict) else None
    if not observation:
        return False
    visible = {tuple(tile["position"]) for tile in observation["tiles"]}
    return any(obj["owner"] == "opponent" and obj["kind"] == 34
               and tuple(obj["position"]) in positions and tuple(obj["position"]) in visible
               for obj in observation["objects"])


def compare_permitted_prefixes(left, right, fixture_positions):
    normalized = [normalize_permitted_frames(frames) for frames in (left, right)]
    a, b = normalized
    index = next((index for index, (first, second) in enumerate(zip(a, b)) if first != second), min(len(a), len(b)))
    if index == len(a) == len(b):
        return {"status": "passed", "equal_frames": index, "comparison": "complete permitted prefixes equal",
                "normalized_sha256": hashlib.sha256(canonical(a).encode()).hexdigest()}
    # Stop only at a directly observed fixture opponent, not an arbitrary world
    # change or differing own route. Different own movement before that is a failure.
    differing = [frames[index] for frames in (left, right) if index < len(frames)]
    legitimate = any(visible_fixture_opponent(frame, fixture_positions) for frame in differing)
    if legitimate and len(differing) == 2 and all(frame.get("observation") for frame in differing):
        # New opponent visibility may change its own facts, but cannot excuse a
        # different own route, resources, terrain, context or command metadata.
        scrubbed = json.loads(json.dumps([left[:index + 1], right[:index + 1]]))
        for frames in scrubbed:
            observation = frames[-1]["observation"]
            observation["objects"] = [obj for obj in observation["objects"]
                                      if not (obj["owner"] == "opponent" and obj["kind"] == 34
                                              and tuple(obj["position"]) in fixture_positions)]
        legitimate = normalize_permitted_frames(scrubbed[0]) == normalize_permitted_frames(scrubbed[1])
    return {"status": "unverified" if legitimate else "failed", "equal_frames": index,
            "first_difference_seq": index, "left_frames": len(a), "right_frames": len(b),
            "comparison": "stopped at observed opponent visibility difference" if legitimate
                          else "permitted history diverged before any observed fixture opponent",
            "comparison_stopped_at_legitimate_visibility_difference": legitimate}


def hidden_variant(run, variant):
    ensure(run.state.color == "red", "Paired fixture requires Red's first normal turn")
    first = run.inspect_request()
    red_target = tuple(run.args.paired_red_target) + (0,)
    blue_targets = [tuple(run.args.paired_blue_targets[index:index + 2]) + (0,) for index in (0, 2)]
    red_tiles = {tile.position: tile for tile in first.observation.tiles}
    if (not first.observation.heroes or red_target not in red_tiles or red_tiles[red_target].blocked
            or red_tiles[red_target].visitable):
        run.check("paired_red_known_target_available", "unverified", target=list(red_target))
        return None
    red_hero = first.observation.heroes[0].handle
    run.call(run.contract.Action("end_turn"), "paired_fixture_red_initial_end_turn")
    if run.state.done or run.state.color != "blue":
        run.check("paired_blue_turn_available", "unverified")
        return None
    blue = run.inspect_request()
    if not blue.observation.heroes:
        run.check("paired_blue_hidden_hero_available", "unverified")
        return None
    blue_hero = blue.observation.heroes[0]
    blue_tiles = {tile.position: tile for tile in blue.observation.tiles}
    suitable = (blue_hero.position not in red_tiles and blue_targets[0] != blue_targets[1]
                and all(pos in blue_tiles and not blue_tiles[pos].blocked and not blue_tiles[pos].visitable
                        and pos not in red_tiles and pos != blue_hero.position
                        and max(abs(pos[0] - blue_hero.position[0]), abs(pos[1] - blue_hero.position[1])) == 1
                        for pos in blue_targets))
    if not suitable:
        run.check("paired_hidden_adjacent_fixture_available", "unverified", targets=[list(pos) for pos in blue_targets])
        return None
    target = blue_targets[variant]
    _, additions, result = run.call(run.contract.Action("move_to", blue_hero.handle, *target[:2]),
                                    f"paired_hidden_blue_variant_{variant}")
    issued = sum(record.kind == "issued" for record in additions)
    accepted = sum(record.kind == "acknowledged" and record.accepted is True for record in additions)
    interactions = [record.event.kind for record in additions if record.event is not None and record.event.kind in
                    ("visit", "battle_start", "battle_complete", "hero_lost", "notice")]
    if (result is None or result.status != "reached" or result.position != target or not result.destination_reached
            or issued != 1 or accepted != 1 or interactions):
        run.check("paired_hidden_single_native_move", "unverified", actual=asdict(result) if result else None,
                  issued=issued, accepted=accepted, interaction_events=interactions)
        return None
    run.check("paired_hidden_single_native_move", "passed", actual_position=list(target), native_commands=issued)
    if run.state.done or run.state.color != "blue":
        run.check("paired_blue_end_turn_available", "unverified")
        return None
    run.call(run.contract.Action("end_turn"), "paired_fixture_blue_end_turn")
    if run.state.done or run.state.color != "red":
        run.check("paired_red_successor_available", "unverified")
        return None
    current = run.inspect_request()
    current_tiles = {tile.position for tile in current.observation.tiles}
    if target in current_tiles or any(obj.owner == "opponent" and obj.position == target for obj in current.observation.objects):
        run.check("paired_opponent_still_hidden", "unverified", reason="Variant target became legitimately visible before comparison")
        return None
    ensure(red_hero in {hero.handle for hero in current.observation.heroes}, "Red's own hero unexpectedly changed")
    before = run.history("red")
    full_support(run, current)
    _, additions, result = run.call(run.contract.Action("move_to", red_hero, *red_target[:2]), "paired_identical_red_known_route")
    after = run.history("red")
    comparison = {"before": before, "after": after, "red_target": list(red_target),
                  "blue_actual_position": list(target), "red_result": asdict(result) if result else None,
                  "red_native_commands": sum(record.kind == "issued" for record in additions)}
    write(run.directory / "paired-red-permitted-prefixes.json", comparison)
    run.check("paired_red_route_completed", "passed" if result is not None else "unverified",
              result=asdict(result) if result else None)
    return comparison


def paired_hidden_routes(args, env_module, contract, auditor, seed):
    rows, snapshots = [], []
    fixed_map_sha256, fixed_binary_sha256 = sha(args.map), sha(args.binary)
    for variant in (0, 1):
        run = CaseRun(f"hidden-route-{'a' if variant == 0 else 'b'}", args, env_module, contract, auditor, seed)
        error, snapshot = None, None
        try:
            ensure(sha(args.map) == fixed_map_sha256 and sha(args.binary) == fixed_binary_sha256,
                   "Paired fixture map/binary changed between runs")
            run.start()
            snapshot = hidden_variant(run, variant)
            run.finish()
            run.persist_and_audit()
            ensure(sha(args.map) == fixed_map_sha256 and sha(args.binary) == fixed_binary_sha256,
                   "Paired fixture map/binary changed during native execution")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.check("paired_fixture_execution", "failed", error=error)
            try:
                run.finish()
                if run.end is not None and not run.audit_attempted:
                    run.persist_and_audit()
            except Exception as cleanup_error:
                run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup_error).__name__}: {cleanup_error}")
        finally:
            run.env.close()
        row = run.summary(error)
        write(run.directory / "probe-report.json", row)
        rows.append(row)
        snapshots.append(snapshot)
    result = {"fixture": "same-map-same-seed-hidden-opponent-adjacent-move", "engine_seed": seed,
              "map_sha256": fixed_map_sha256, "binary_sha256": fixed_binary_sha256, "training_input": False,
              "scope": "Hidden opponent movement variant, including its position, movement points and revealed fog; no hidden-terrain or guard invariance claim."}
    if any(snapshot is None for snapshot in snapshots):
        result.update(status="failed" if any(row["probe_status"] == "failed" for row in rows) else "unverified",
                      reason="Both comparable native fixture runs were not obtained")
    else:
        positions = {tuple(snapshot["blue_actual_position"]) for snapshot in snapshots}
        before = compare_permitted_prefixes(snapshots[0]["before"], snapshots[1]["before"], positions)
        after = compare_permitted_prefixes(snapshots[0]["after"], snapshots[1]["after"], positions)
        status = paired_comparison_status(rows, [before, after])
        result.update(status=status, before_identical_red_action=before, after_identical_red_action=after,
                      red_target=snapshots[0]["red_target"], blue_hidden_positions=[list(pos) for pos in sorted(positions)])
    write(args.output / "paired-hidden-route-comparison.json", result)
    return rows, result


def day_cutoff(run):
    while run.live():
        frame = run.inspect_request()
        ensure(frame.observation.context.kind == "map", "Day-cutoff fixture unexpectedly opened a window")
        run.call(run.contract.Action("end_turn"), "explicit_day_boundary")
    ensure(run.end is not None and run.end.status == "truncated", "Day horizon did not produce a cutoff")
    ensure(run.end.termination_reason == "game_day_limit", "Unexpected day cutoff reason")
    run.check("both_player_verified_day_cutoffs", "passed")


def injected_failure(run, before_truncate):
    frame = run.inspect_request()
    ensure(frame.observation.heroes, "Crash fixture needs an owned hero")
    run.call(run.contract.Action("end_turn"), "acknowledged_native_turn_before_crash")
    ensure(not run.state.done, "Episode ended before failure injection")
    records = [run.contract.decode_frame(raw) for raw in run.history(COLORS[frame.player])]
    ensure(any(record.kind == "consumed" for record in records)
           and any(record.kind == "acknowledged" and record.accepted is True for record in records)
           and any(record.kind == "result" for record in records),
           "Failure injected before an actually consumed/native-acknowledged/resolved action")
    run.kill_owned_process()
    if before_truncate:
        run.end = run.env.truncate("collection_window")
    else:
        # Public step must observe the pre-existing process exit before sending a
        # response. This is deliberately distinct from the truncate finalizer.
        current = run.inspect_request()
        action = run.contract.Action("end_turn") if current.observation.context.kind == "map" else run.contract.Action("close")
        run.call(action, "drain_preexisting_crash_via_step")
        run.end = run.state if run.state.done else None
    run.state = run.end
    ensure(run.end is not None and run.end.status == "failed", "Native crash was not classified as technical failure")
    ensure(all(value is None for value in run.end.rewards.values()), "Crash received a reward")
    ensure(all(value is None for value in run.end.bootstrap.values()), "Crash retained bootstrap observations")
    run.check("preexisting_crash_before_truncate" if before_truncate else "post_consumed_action_crash", "passed")


def random_action(frame, rng, contract):
    obs = frame.observation
    if obs.context.kind != "map":
        return contract.Action("close")
    kinds = ["end_turn"]
    if obs.heroes:
        kinds += ["move_to", "open_hero"]
    if obs.towns:
        kinds.append("open_town")
    kind = rng.choice(kinds)
    if kind == "move_to":
        return contract.Action(kind, rng.choice(obs.heroes).handle, rng.randrange(obs.size[0]), rng.randrange(obs.size[1]))
    if kind == "open_hero":
        return contract.Action(kind, rng.choice(obs.heroes).handle)
    if kind == "open_town":
        return contract.Action(kind, rng.choice(obs.towns).handle)
    return contract.Action(kind)


def random_episode(run, seeds):
    rngs = {color: random.Random(seeds[color]) for color in COLORS}
    while run.live():
        frame = run.inspect_request()
        run.call(random_action(frame, rngs[run.state.color], run.contract), "uniform_type_and_parameter_sample")
    run.check("uniform_random_cohort_finished", "passed", policy_seeds=seeds,
              episode_status=run.end.status if run.end else None)


def run_case(name, args, env_module, contract, auditor, seed, policy_seeds=None):
    run = CaseRun(name, args, env_module, contract, auditor, seed)
    error = None
    try:
        run.start()
        if name == "navigation":
            navigation(run)
        elif name == "mechanics":
            mechanics(run)
        elif name == "closed-window-battle":
            closed_window_battle(run)
        elif name == "day-cutoff":
            day_cutoff(run)
        elif name in ("kill-after-action", "kill-before-truncate"):
            injected_failure(run, name == "kill-before-truncate")
        else:
            random_episode(run, policy_seeds)
        run.finish()
        run.persist_and_audit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("probe_execution", "failed", error=error)
        # Retain whatever native evidence exists; never rewrite the producer's
        # result to make an assertion pass. Normal cleanup owns only this engine.
        try:
            run.finish()
            if run.end is not None and not run.audit_attempted:
                run.persist_and_audit()
        except Exception as cleanup_error:
            run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup_error).__name__}: {cleanup_error}")
    finally:
        run.env.close()
    summary = run.summary(error)
    write(run.directory / "probe-report.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=ROOT)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    result.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwritten")
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    result.add_argument("--paired-hidden-routes", action="store_true", help="Optional two-game evaluator fixture, same map and seed")
    result.add_argument("--paired-red-target", nargs=2, type=int, default=[1, 10], metavar=("X", "Y"))
    result.add_argument("--paired-blue-targets", nargs=4, type=int, default=[15, 8, 16, 8], metavar=("AX", "AY", "BX", "BY"))
    result.add_argument("--random-episodes", type=int, default=2)
    result.add_argument("--seed", type=int, default=81001)
    result.add_argument("--red-policy-seed", type=int, default=91001)
    result.add_argument("--blue-policy-seed", type=int, default=101001)
    result.add_argument("--max-days", type=int, default=20)
    result.add_argument("--max-decisions", type=int, default=300)
    result.add_argument("--case-seconds", type=float, default=180)
    result.add_argument("--watchdog", type=float, default=30)
    result.add_argument("--startup-seconds", type=float, default=120)
    result.add_argument("--frontier-attempts", type=int, default=12)
    result.add_argument("--movement-attempts", type=int, default=32)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    for name in ("source", "binary", "data", "map", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (args.random_episodes < 0 or args.random_episodes > 20 or len(set(args.cases)) != len(args.cases)
            or any(value < 0 for value in args.paired_red_target + args.paired_blue_targets)
            or not all(math.isfinite(value) for value in (args.case_seconds, args.watchdog, args.startup_seconds))
            or not 1 <= args.seed <= 2147483000 or min(args.max_days, args.max_decisions, args.case_seconds,
                                                    args.watchdog, args.startup_seconds, args.frontier_attempts,
                                                    args.movement_attempts) <= 0):
        cli.error("Use unique cases, 0..20 random episodes, valid seed and positive finite bounds")
    if args.output.exists():
        cli.error("Output exists; preserved native evidence must not be overwritten")
    sys.path.insert(0, str(args.source))
    from experiments.learner import contextual as contract
    env_module = load("heroes_contextual_probe_env", args.source / "experiments/headless/env.py")
    auditor = load("heroes_contextual_probe_auditor", args.source / "validation/probes/audit_contextual_histories.py")
    if "interface" not in inspect.signature(env_module.HeadlessSelfPlayEnv).parameters or not hasattr(env_module.HeadlessSelfPlayEnv, "history_v2"):
        cli.exit(2, "Native contextual environment API is not implemented: require interface= and history_v2(). No game launched.\n")
    env_module.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(args.source / "tools/prepare_vcmi.py"), "--check"], cwd=args.source, check=True)
    args.output.mkdir()
    schedule = [{"case": name, "engine_seed": args.seed + index} for index, name in enumerate(args.cases)]
    for index in range(args.random_episodes):
        schedule.append({"case": f"random-{index + 1:03d}", "engine_seed": args.seed + 100 + index,
                         "policy_seeds": {"red": args.red_policy_seed + index, "blue": args.blue_policy_seed + index}})
    plan = {"schema": "heroes-contextual-native-probe-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": contract.ENVIRONMENT, "history_schema": contract.SCHEMA, "action_schema": contract.ACTION_SCHEMA,
            "schedule": schedule,
            "paired_hidden_fixture": {"engine_seed": args.seed + 200, "red_target": args.paired_red_target,
                                      "blue_targets": args.paired_blue_targets} if args.paired_hidden_routes else None,
            "bounds": {name: getattr(args, name) for name in
                                               ("max_days", "max_decisions", "case_seconds", "watchdog", "frontier_attempts", "movement_attempts")},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip(),
            "binary_sha256": sha(args.binary), "map_sha256": sha(args.map), "probe_sha256": sha(Path(__file__)),
            "training_performed": False, "paid_compute": False,
            "scope": "Bounded native interface probes and a separate seeded uniform random cohort. No learned policy or training data collection."}
    write(args.output / "plan.json", plan)
    rows = [run_case(row["case"], args, env_module, contract, auditor, row["engine_seed"], row.get("policy_seeds")) for row in schedule]
    paired_report = None
    if args.paired_hidden_routes:
        paired_rows, paired_report = paired_hidden_routes(args, env_module, contract, auditor, args.seed + 200)
        rows.extend(paired_rows)
    raw_natural = sum(row["episode_status"] == "completed" for row in rows if row["case"].startswith("random-"))
    natural = validated_natural_count(rows)
    failed = any(row["probe_status"] == "failed" for row in rows) or paired_report is not None and paired_report["status"] == "failed"
    unverified = (any(row["probe_status"] == "unverified" for row in rows) or natural == 0
                  or paired_report is not None and paired_report["status"] == "unverified")
    report = {**plan, "cases": rows, "probe_status": "failed" if failed else "unverified" if unverified else "passed",
              "random_natural_completions": natural, "random_reported_completed_episodes": raw_natural,
              "random_natural_count_definition": "Completed random episodes with passed probe and independently bound history audit",
              "paired_hidden_route_comparison": paired_report,
              "natural_outcomes_gate": "passed" if natural else "unverified",
              "native_source_visibility_audit_required": True}
    write(args.output / "validation.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "report": str(args.output / "validation.json"),
                      "random_natural_completions": natural, "cases": [{k: row[k] for k in
                          ("case", "probe_status", "episode_status", "termination_reason")} for row in rows]}))
    return 1 if failed else 2 if unverified else 0


if __name__ == "__main__":
    raise SystemExit(main())
