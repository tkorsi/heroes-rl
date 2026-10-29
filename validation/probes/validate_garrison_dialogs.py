"""Bounded real-engine map-garrison checks; scripted diagnostics, never PPO.

Every selected operation uses a delivered player-visible control. Authored
fixture state is only an effect oracle. Captures retain native replay and an
independent history audit; no training, learned checkpoint or reward is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explorer import create_garrison_scenarios as fixtures
from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.contextual_replay import verify_episode
from experiments.headless.process_deadline import (DeadlineVcmiLauncher, guarded_popen,
                                                   close_owned_process, cleanup_from_evidence)
from experiments.learner import contextual as c
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes.validate_army_management import (army_contents, exact_stack_operation,
    exact_exchange_stack_roundtrip, rows, native_receipt)
from validation.probes.validate_capabilities import CapabilityRun, capability_implementation_identity, INTERFACE
from validation.probes.validate_hero_policy import retain_failed_evidence, wall_limit
from tools.isolated_diagnostic import diagnostic_lease


class GarrisonRun(CapabilityRun):
    def __init__(self, *positional, **keywords):
        super().__init__(*positional, **keywords)
        args = self.args
        self.env.launcher = DeadlineVcmiLauncher(args.binary, args.data, args.map, "info",
            deadline_wall=min(args.deadline_wall, time.time() + args.case_seconds),
            deadline_monotonic=min(args.deadline_monotonic, time.monotonic() + args.case_seconds),
            evidence_path=self.directory / "native-deadline.jsonl")


def implementation_identity():
    identity = capability_implementation_identity()
    # The local launch/restore controller cannot execute in this native-only
    # diagnostic. It later pins this acceptance, so binding it here would form
    # a receipt/pin cycle. Keep every actual adapter, replay and engine source.
    identity.pop("experiments/learner/local_script_training.py", None)
    for name in ("experiments/explorer/create_garrison_scenarios.py",
                 "tools/isolated_diagnostic.py",
                 "validation/probes/validate_garrison_dialogs.py"):
        identity[name] = digest(ROOT / name)
    return dict(sorted(identity.items()))


def schedule(cases, colors, seed):
    return [{"case": case, "player": player, "seed": seed + 2 * fixtures.CASES.index(case) + fixtures.COLORS.index(player)}
            for case in cases for player in colors]


def fixture_identity(case, player, maps):
    path = maps / (fixtures.scenario_id(case, player) + ".vmap")
    fixtures.verify_scenario(path, case, player)
    manifest = path.with_suffix(".manifest.json")
    declared = json.loads(manifest.read_text())
    base.ensure(declared["map_sha256"] == digest(path) and declared["diagnostic_only"]
                and not declared["training_input"], "Garrison map manifest mismatch")
    return {"path": str(path), "sha256": digest(path), "manifest_sha256": digest(manifest),
            "case": case, "player": player}


def exchange_rows(run):
    frame = run.inspect_request()
    base.ensure(frame.observation.context.kind == "exchange", "Native visit failed to open the actual garrison query")
    hero = next((army for army in rows(frame) if army.role == "hero"), None)
    guard = next((army for army in rows(frame) if army.role in ("object_garrison", "locked_garrison")), None)
    base.ensure(hero is not None and guard is not None and hero.subject == run.own_hero()
                and hero.requires_nonempty and not guard.requires_nonempty,
                "Garrison query lacks its exact actor/object armies and last-stack rules")
    return frame, hero, guard


def check_open(run, target, additions, locked):
    frame, hero, guard = exchange_rows(run)
    expected_event = "garrison_locked_requested" if locked else "garrison_requested"
    matching = [item for item in additions if item.event is not None
                and item.event.kind == expected_event and item.event.subject == hero.subject
                and item.event.partner == guard.subject and item.event.object_kind == target.kind
                and item.event.owner == "self"]
    opened = [item for item in additions if item.kind == "context_opened" and item.reason == "object_garrison"]
    base.ensure(len(matching) == len(opened) == 1 and matching[0].seq < opened[0].seq < frame.seq
                and guard.subject == target.handle
                and guard.role == ("locked_garrison" if locked else "object_garrison"),
                "Garrison event/permission/context opening is missing or reordered")
    base.ensure(not frame.offers.capability_controls, "Nonhero garrison exposed hero artifact controls")
    run.check("native_object_garrison_open", "passed", object_kind=target.kind, owner="self",
              locked=locked, query_event_seq=matching[0].seq, opened_seq=opened[0].seq,
              hero=hero.subject, partner=guard.subject)


def operate(run, kind, source, destination, count=None):
    before = run.inspect_request()
    action = run.action(kind, subject=before.observation.context.subject,
                        selection=source.ref, destination=destination.ref, count=count)
    _, additions, result = run.call(action, "explicit_garrison_" + kind)
    proof = exact_stack_operation(before, run.inspect_request(), action, additions, result)
    run.check("garrison_exact_stack_effect", "passed", **proof)


def stack_roundtrip(run):
    before, hero, guard = exchange_rows(run)
    slot = next(slot for slot in hero.slots if slot.count)
    _, additions, result = run.call(run.action("open_stack", subject=before.observation.context.subject, selection=slot.ref),
                                    "explicit_inspect_garrison_hero_stack")
    opened = run.inspect_request()
    _, closing, closed = run.call(run.action("back", subject=opened.observation.context.subject),
                                 "explicit_back_to_garrison_query")
    proof = exact_exchange_stack_roundtrip(before, opened, run.inspect_request(), slot.ref,
                                           (additions, result), (closing, closed))
    run.check("garrison_stack_details_return_to_query", "passed", **proof)


def close_exchange(run):
    before, hero, guard = exchange_rows(run)
    source = next(slot for slot in hero.slots if slot.count)
    destination = guard.slots[0]
    _, additions, result = run.call(run.action("close"), "explicit_close_map_garrison_query")
    native_receipt(before, additions, result, "closed")
    after = run.inspect_request()
    base.ensure(after.observation.context.kind == "map" and after.observation.window is None,
                "Closed native query retained hidden garrison rows or a stale context")
    run.reject(run.action("split_stack", subject=hero.subject, selection=source.ref,
                          destination=destination.ref, count=1), "closed_garrison_control_rejected")
    run.check("native_garrison_close_returned_to_map", "passed")


def diagnostic(run, case):
    kind = 53 if case.startswith("mine-") else (219 if case.startswith("vertical-") else 33)
    target = next(iter(run.targets(kind)), None)
    base.ensure(target is not None, "Authored garrison is not visible to the actor")
    if case == "mine-repeat":
        base.ensure(target.owner == "neutral", "Repeat fixture must initially be a neutral mine")
        run.visit(target)
        base.ensure(run.inspect_request().observation.context.kind == "map", "First mine capture unexpectedly opened a garrison")
        target = next(obj for obj in run.inspect_request().observation.objects if obj.handle == target.handle)
        base.ensure(target.owner == "self", "Native mine capture did not transfer ownership")
        additions = run.revisit(target)
        run.check("neutral_capture_then_owned_revisit", "passed")
    else:
        additions = run.visit(target)
    locked = case.endswith("-locked")
    check_open(run, target, additions, locked)
    stack_roundtrip(run)
    before, hero, guard = exchange_rows(run)
    source = next(slot for slot in hero.slots if slot.count)
    if case == "garrison-full":
        base.ensure(all(slot.count for army in (hero, guard) for slot in army.slots),
                    "Full-row fixture did not load all fourteen occupied slots")
        # Closing a full exchange must be possible without discarding troops.
        old = {army.subject: army_contents(army) for army in (hero, guard)}
        close_exchange(run)
        additions = run.revisit(target)
        check_open(run, target, additions, False)
        current = exchange_rows(run)
        base.ensure({army.subject: army_contents(army) for army in current[1:]} == old,
                    "Close/revisit of a full query modified native troops")
    else:
        empty = next(slot for slot in guard.slots if not slot.count)
        if locked:
            same = next(slot for slot in guard.slots if slot.creature == source.creature and slot.count)
            base.ensure(same.slot != source.slot, "Locked occupied deposit must cover different native slot numbers")
            operate(run, "split_stack", source, same, 1)
            _, hero, guard = exchange_rows(run)
            source = next(slot for slot in hero.slots if slot.count)
            empty = next(slot for slot in guard.slots if not slot.count)
            for kind in ("split_stack", "merge_stacks", "swap_stacks"):
                run.reject(run.action(kind, subject=hero.subject, selection=same.ref,
                                      destination=source.ref, count=1 if kind == "split_stack" else None),
                           "locked_garrison_withdrawal_rejected_" + kind)
        # A single-stack hero must retain one creature, including deposits.
        run.reject(run.action("split_stack", subject=hero.subject, selection=source.ref,
                              destination=empty.ref, count=source.count), "hero_last_stack_deposit_rejected")
        operate(run, "split_stack", source, empty, 1)
        _, hero, guard = exchange_rows(run)
        deposited = next(slot for slot in guard.slots if slot.ref == empty.ref)
        if locked:
            other_empty = next(slot for slot in guard.slots if not slot.count)
            operate(run, "swap_stacks", deposited, other_empty)
        else:
            original = next(slot for slot in hero.slots if slot.ref == source.ref)
            operate(run, "merge_stacks", deposited, original)
    close_exchange(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "replay-binary", "data", "output", "maps"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", choices=(*fixtures.CASES, "all"), default="all")
    parser.add_argument("--player", choices=(*fixtures.COLORS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=83010000)
    parser.add_argument("--max-days", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=96)
    parser.add_argument("--case-seconds", type=float, default=120)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    parser.add_argument("--official-lock", type=Path,
                        default=Path.home() / "Projects/Heroes-artifacts/local-script-training/owner.lock")
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--isolated-diagnostic", action="store_true",
                        help="Use a private diagnostic lease and fresh investigation output; never touch official state")
    parser.add_argument("--supervision", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--supervision-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.seed <= 2147480000 or not 60 <= args.wall_seconds <= 1800 or not 1 <= args.max_days <= 16 or not 1 <= args.max_decisions <= 256 or any(
            not math.isfinite(value) or not 0 < value <= 180 for value in (args.case_seconds, args.watchdog, args.startup_seconds)):
        parser.error("Use bounded positive seeds, days, decisions and wall limits")
    for name in ("binary", "replay_binary", "data", "output", "maps"):
        value = getattr(args, name).expanduser()
        setattr(args, name, value.absolute() if name == 'output' and args.isolated_diagnostic else value.resolve())
    base.ensure(not args.output.exists(), "Preserve previous attempts; output exists")
    if args.supervision is None:
        return supervise(args, list(sys.argv[1:] if argv is None else argv))
    base.ensure(args.supervision_sha256 and digest(args.supervision) == args.supervision_sha256,
                "Supervisor plan binding changed")
    supervision = json.loads(args.supervision.read_text())
    base.ensure(supervision["harness_sha256"] == digest(__file__)
                and args.output == Path(supervision["execution_output"])
                and args.isolated_diagnostic == supervision.get("isolated_diagnostic", False),
                "Worker supervision scope differs")
    args.deadline_wall = supervision["absolute_deadline_unix"] - 10
    args.deadline_monotonic = supervision["hard_deadline_monotonic"] - 10
    base.ensure(min(args.deadline_wall - time.time(), args.deadline_monotonic - time.monotonic()) > 0,
                "Diagnostic window already expired")
    args.interface = INTERFACE
    cases = fixtures.CASES if args.case == "all" else (args.case,)
    colors = fixtures.COLORS if args.player == "both" else (args.player,)
    identities = {(case, player): fixture_identity(case, player, args.maps) for case in cases for player in colors}
    for identity in identities.values():
        environment.VcmiLauncher(args.binary, args.data, Path(identity["path"])).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    implementation, native = implementation_identity(), runtime_identity(args.binary.parent)
    args.output.mkdir(parents=True)
    plan = {"schema": "heroes-garrison-diagnostic-plan-v1", "interface": INTERFACE,
            "schedule": schedule(cases, colors, args.seed), "fixtures": list(identities.values()),
            "implementation": implementation, "native_runtime": native,
            "binary_sha256": digest(args.binary), "replay_binary_sha256": digest(args.replay_binary),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
            "training_performed": False, "training_input": False, "optimizer_steps": 0,
            "scripted_mechanics_only": True, "seed_replacement": False}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-garrison-native-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "not_started": len(plan["schedule"]),
              "training_performed": False, "training_input": False, "optimizer_steps": 0}

    def check_identity():
        for name, expected in implementation.items():
            base.ensure(digest(ROOT / name) == expected, "Diagnostic source changed: " + name)
        base.ensure(runtime_identity(args.binary.parent) == native and digest(args.binary) == plan["binary_sha256"]
                    and digest(args.replay_binary) == plan["replay_binary_sha256"] and digest(args.output / "plan.json") == plan_hash,
                    "Diagnostic runtime or plan changed")
        for (case, color), identity in identities.items():
            base.ensure(fixture_identity(case, color, args.maps) == identity, "Diagnostic map changed")

    for row in plan["schedule"]:
        run, attempt = None, {**row, "status": "running"}
        report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        try:
            with wall_limit(args.case_seconds):
                check_identity()
                args.player, args.map = row["player"], Path(identities[(row["case"], row["player"])]["path"])
                run = GarrisonRun(f"{row['case']}-{row['player']}", args, environment, c, auditor, row["seed"])
                run.start()
                diagnostic(run, row["case"])
                run.finish()
                run.persist_and_audit()
                base.ensure(run.end.status in ("completed", "truncated"), "Native mechanics failed")
                replay = verify_episode(run.env.episode_dir, binary=args.replay_binary, data=args.data, timeout=min(120, args.case_seconds))
                base.ensure(replay["verified"], "Native replay failed: " + str(replay.get("error")))
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
    report["status"] = "passed" if not report["not_started"] and all(row["status"] == "passed" for row in report["cases"]) else "failed"
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "attempts": len(report["cases"]), "not_started": report["not_started"]}))
    return 0 if report["status"] == "passed" else 1


def supervise(args, argv):
    """One immutable window, an explicit ownership lease and an external guard."""
    with diagnostic_lease(args.output, isolated=getattr(args, 'isolated_diagnostic', False),
                          official_lock=args.official_lock) as lease:
        args.output.mkdir(parents=True, exist_ok=False)
        wall, mono = time.time(), time.monotonic()
        plan = {"schema": "heroes-garrison-diagnostic-supervision-v1", "harness_sha256": digest(__file__),
                "started_unix": wall, "started_monotonic": mono,
                "absolute_deadline_unix": wall + args.wall_seconds,
                "hard_deadline_monotonic": mono + args.wall_seconds,
                "execution_output": str(args.output / "execution"), "official_lock": str(args.official_lock),
                "isolated_diagnostic": getattr(args, 'isolated_diagnostic', False), "diagnostic_lease": lease,
                "training_performed": False, "optimizer_steps": 0}
        supervision = args.output / "supervision.json"
        write_json(supervision, plan)
        command = [sys.executable, __file__, *argv, "--output", plan["execution_output"],
                   "--supervision", str(supervision), "--supervision-sha256", digest(supervision)]
        process = None
        code, errors = None, []
        with (args.output / "worker.log").open("w") as log:
            try:
                process = guarded_popen(command, deadline_wall=plan["absolute_deadline_unix"] - 5,
                    deadline_monotonic=plan["hard_deadline_monotonic"] - 5,
                    evidence_path=args.output / "deadline.jsonl", cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait(timeout=args.wall_seconds)
            finally:
                limits = dict(deadline_wall=plan["absolute_deadline_unix"],
                              deadline_monotonic=plan["hard_deadline_monotonic"])
                if process is not None:
                    try:
                        close_owned_process(process, timeout=2, **limits)
                    except Exception as error:
                        errors.append(f"worker: {type(error).__name__}: {error}")
                for evidence in args.output.rglob("*deadline.jsonl"):
                    try:
                        cleanup_from_evidence(evidence, timeout=2, **limits)
                    except Exception as error:
                        errors.append(f"{evidence}: {type(error).__name__}: {error}")
                write_json(args.output / "shutdown.json", {"finished_unix": time.time(), "returncode": code,
                    "owned_cleanup_verified": not errors, "errors": errors,
                    "within_hard_deadline": time.time() <= plan["absolute_deadline_unix"],
                    "training_performed": False, "optimizer_steps": 0})
        print(json.dumps({"returncode": code, "owned_cleanup_verified": not errors,
                          "report": str(args.output / "execution/report.json")}))
        return code if not errors and code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
