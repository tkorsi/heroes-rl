#!/usr/bin/env python3
"""Four isolated v8 paired cutoff controls from the frozen 1728 policy.

Fixture navigation is explicitly scripted. Only mandatory level-up answers use
the unchanged policy. These mixed controls are never training trajectories,
ordinary policy evaluation, or optimizer admission capabilities.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import benchmark_copied_training as copies
from tools import run_selfplay_training as campaign

BASE = campaign.ARTIFACTS / "selfplay-cutoff-diagnostics"
SCHEMA = "heroes-selfplay-cutoff-controls-v1"
INTERFACE = "adventure-capabilities-v1"
GUARD = "visible-interaction-budget-revealed-v1"
SEEDS = tuple(range(1_946_000_000, 1_946_000_004))
POLICY_SEEDS = tuple(range(1_947_000_000, 1_947_000_004))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schedule(offset=0):
    require(type(offset) is int and 0 <= offset <= 100_000 and offset % 4 == 0, "Invalid diagnostic seed offset")
    return [dict(case=case, fixture="artifacts" if case == "optional-capability" else "level_up",
                 player=color, engine_seed=SEEDS[index] + offset, policy_seed=POLICY_SEEDS[index] + offset)
            for index, (case, color) in enumerate((case, color)
                for case in ("optional-capability", "mandatory-level-up") for color in ("red", "blue"))]


def reserve(output, identity, *, base=BASE):
    """Consume four fresh seeds before launch, including unstarted failed cases."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "namespace.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path = base / "seed-reservations.json"
        ledger = campaign.read(ledger_path) if ledger_path.exists() else dict(schema=SCHEMA + "-seeds", reservations=[])
        require(ledger.get("schema") == SCHEMA + "-seeds" and type(ledger.get("reservations")) is list,
                "Unknown diagnostic seed ledger")
        for index, item in enumerate(ledger["reservations"]):
            require(item["offset"] == index * 4 and item["schedule"] == schedule(index * 4)
                    and item["consumed_including_unstarted"] is True, "Diagnostic seed ledger changed")
        require(all(item["output"] != str(output) for item in ledger["reservations"]), "Output already consumed seeds")
        offset = len(ledger["reservations"]) * 4
        rows = schedule(offset)
        value = dict(schema=SCHEMA + "-reservation", output=str(output), source=identity,
                     offset=offset, schedule=rows, consumed_including_unstarted=True)
        ledger["reservations"].append(value)
        # Commit consumption first. A crash before the immutable receipt is
        # written still consumes these seeds, so the next attempt advances.
        campaign.write(ledger_path, ledger)
        path = base / f"reserved-{rows[0]['engine_seed']}-{rows[-1]['engine_seed']}.json"
        campaign.write(path, value, new=True)
    return campaign.binding(path)


def source_identity():
    from validation.probes.validate_capabilities import capability_implementation_identity
    from experiments.learner.inference import source_identity as learner_identity
    implementation = capability_implementation_identity()
    for path in (Path(__file__).resolve(), ROOT / "tools/run_selfplay_training.py",
                 ROOT / "tools/benchmark_copied_training.py"):
        implementation[str(path.relative_to(ROOT))] = campaign.binding(path)["sha256"]
    return dict(checkout=copies.checkout_identity(ROOT), source=learner_identity(INTERFACE), implementation=implementation)


def parent_state(descriptor):
    from experiments.learner.local_script_training import restore_parent
    state = restore_parent(descriptor)
    require(state.progress.updates == 1728, "Cutoff controls require exact checkpoint 1728")
    return state


def prepare(args):
    from experiments.headless.contextual_evidence import runtime_identity
    from validation.probes.validate_capabilities import fixture_identity
    from experiments.explorer.create_capability_scenarios import MAPS
    output = copies.private_path(args.output, fresh=True)
    identity = source_identity()
    pointer = campaign.read(campaign.ORIGINAL / "latest-checkpoint.json")
    protected = campaign.protected_inputs(pointer)
    state = parent_state(pointer["parent"])
    original = campaign.read(campaign.bound(pointer["parent"]["plan"]))
    require(original.get("policy_support") == GUARD, "Parent policy support differs")
    runtime = runtime_identity(Path(original["binary"]).parent)
    require(runtime == original["runtime"], "Actual engine runtime changed")
    fixtures = {f"{row['fixture']}-{row['player']}": fixture_identity(row["fixture"], row["player"], MAPS)
                for row in schedule()}
    campaign.verify_protected(protected)
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
                   cwd=ROOT, check=True, timeout=60)
    output.mkdir(parents=True, mode=0o700)
    reservation = reserve(output, identity)
    rows = campaign.read(campaign.bound(reservation))["schedule"]
    seconds = 4 * (args.case_seconds + 40) + 120
    plan = dict(schema=SCHEMA + "-plan", output=str(output), source=identity, protected=protected,
                parent_descriptor=pointer["parent"], parent_checkpoint=state.checkpoint,
                parent_weights_sha256=state.weights_sha256, policy_support=GUARD,
                full_model_adam_rng_progress_sampler_pool_restored=True,
                binary=original["binary"], data=original["data"], replay_binary=original["replay_binary"],
                native_runtime=runtime, fixtures=fixtures, schedule=rows, reservation=reservation,
                case_seconds=args.case_seconds, max_days=12, max_decisions=256, max_neural_answers=8,
                hard_deadline_monotonic=time.monotonic() + seconds, hard_deadline_unix=time.time() + seconds,
                training_input=False, optimizer_updates=0, mixed_scripted_and_neural_history=True)
    campaign.write(output / "plan.json", plan, new=True)
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    from validation.probes.validate_capabilities import fixture_identity
    from experiments.explorer.create_capability_scenarios import MAPS
    require(source_identity() == plan["source"], "Cutoff diagnostic source changed")
    require(runtime_identity(Path(plan["binary"]).parent) == plan["native_runtime"], "Cutoff native runtime changed")
    campaign.bound(plan["replay_binary"])
    campaign.bound(plan["reservation"])
    for row in plan["schedule"]:
        key = f"{row['fixture']}-{row['player']}"
        require(fixture_identity(row["fixture"], row["player"], MAPS) == plan["fixtures"][key], "Fixture changed")
    campaign.verify_protected(plan["protected"])


def prepare_level_up(run):
    """Use visible fixture chests until one delivers its actual level-up query."""
    from validation.probes.validate_adventure_dialogs import explicit_choice
    visited = set()
    for _ in range(9):
        targets = run.targets(101, visited)
        if not targets:
            break
        target = targets[0]
        run.visit(target)
        visited.add(target.handle)
        frame = run.inspect_request()
        if frame.observation.context.kind != "dialog":
            continue
        panel = frame.observation.window.dialog_panel
        component = next((index for index, value in enumerate(panel.components) if value.kind == "experience"), None)
        require(component is not None, "Fixture chest did not offer displayed experience")
        selected = explicit_choice(panel, "select_component", component)
        run.call(run.action("choose_dialog", subject=frame.observation.context.subject, selection=selected.ref),
                 "scripted_fixture_take_displayed_experience")
        frame = run.inspect_request()
        panel = frame.observation.window.capability_panel if frame.observation.window is not None else None
        if panel is not None and panel.kind == "level_up" and panel.mandatory:
            return frame
        raise ValueError("Declared experience control did not reach a mandatory level-up")
    raise ValueError("Predeclared native fixture did not produce the required level-up branch")


def neural_evidence(session, seed):
    from experiments.learner import contextual as c
    from experiments.learner.contextual_rollout import DecisionEstimate
    return dict(schema=SCHEMA + "-mixed-policy", training_input=False, purpose="diagnostic",
                mixed_scripted_and_neural_history=True, fully_policy_controlled=False, optimizer_updates=0,
                checkpoint=session.checkpoint, weights_sha256=session.weights_sha256,
                player=session.history.player, policy_seed=seed, policy_support=GUARD,
                history=[c.encode_frame(frame) for frame in session.history.frames],
                neural_decisions=[item.to_dict() for item in session._choices.values()],
                estimates=[DecisionEstimate(seq, value, session._prefixes[seq]).to_dict()
                           for seq, value in session._values.items()])


def reconstruct_mixed(raw, model):
    """Read saved evidence afresh; replay all history and sample only labelled answers."""
    from experiments.learner import contextual as c
    from experiments.learner.contextual_model import weights_sha256
    from experiments.learner.contextual_rollout import ContextualSession
    require(raw["schema"] == SCHEMA + "-mixed-policy" and raw["training_input"] is False
            and raw["fully_policy_controlled"] is False and raw["mixed_scripted_and_neural_history"] is True
            and raw["policy_support"] == GUARD and raw["weights_sha256"] == weights_sha256(model),
            "Expected explicitly mixed fixed-policy diagnostic evidence")
    frames = tuple(c.decode_frame(frame) for frame in raw["history"])
    require(frames and all(frame.player == raw["player"] and frame.schema == c.EXPANDED_SCHEMA for frame in frames),
            "Mixed diagnostic history crossed players or interfaces")
    choices = {record["decision_seq"]: record for record in raw["neural_decisions"]}
    require(len(choices) == len(raw["neural_decisions"]) and 1 <= len(choices) <= 8, "Invalid neural answer count")
    session = ContextualSession(model, frames[0].episode, raw["player"], raw["checkpoint"], "scripted-fixture-operator",
                                schema=c.EXPANDED_SCHEMA, seed=raw["policy_seed"], guard_version=GUARD)
    for frame in frames:
        session.consume((frame,))
        if frame.seq in choices:
            panel = frame.offers.capability_panel if frame.kind == "decision" else None
            require(panel is not None and panel.kind == "level_up" and panel.mandatory
                    and session.history.requires_choice, "Neural choice escaped the mandatory level-up query")
            require(session.choose(frame.seq).to_dict() == choices[frame.seq], "Mixed neural answer failed exact reconstruction")
    actual = neural_evidence(session, raw["policy_seed"])
    require(actual == raw, "Mixed diagnostic saved history, values or choices changed")
    return dict(passed=True, actions_resampled=len(choices), values_recomputed=len(session._values),
                maximum_absolute_difference=0, private_memory_reset=True, weights_unchanged=True,
                fully_policy_controlled=False, training_input=False)


def mandatory_cutoff(run, state, row):
    from experiments.learner import contextual as c
    from experiments.learner.contextual_rollout import ContextualSession
    frame = prepare_level_up(run)
    require(frame.player == ("red", "blue").index(row["player"]), "Wrong fixture actor")
    session = ContextualSession(state.model, frame.episode, frame.player, state.checkpoint, "scripted-fixture-operator",
                                schema=c.EXPANDED_SCHEMA, seed=row["policy_seed"], guard_version=GUARD)
    path = run.directory / "mixed-policy-evidence.json"
    run.env.defer_dialog_cutoff("collection_window")
    for _ in range(8):
        session.consume(run.history(row["player"]))
        decision = run.inspect_request()
        panel = decision.offers.capability_panel
        require(session.history.requires_choice and panel is not None and panel.kind == "level_up" and panel.mandatory,
                "Mandatory cutoff returned an unexpected query")
        selected = session.choose(decision.seq)
        require(selected.action.kind == "choose_level_up", "Policy did not sample a real level-up answer")
        campaign.write(path, neural_evidence(session, row["policy_seed"]))
        run.call(selected.action, "neural_full_distribution_mandatory_level_up")
        if run.state.done:
            run.end = run.state
            require(run.end.status == "truncated", "Mandatory control did not finish at the verified cutoff")
            session.consume(run.history(row["player"]))
            campaign.write(path, neural_evidence(session, row["policy_seed"]))
            reconstructed = reconstruct_mixed(campaign.read(path), state.model)
            campaign.write(run.directory / "mixed-policy-reconstruction.json", reconstructed, new=True)
            return reconstructed
    raise ValueError("Mandatory native query exceeded the unchanged eight-answer chain bound")


def continuation_evidence(run, row):
    """Bind the native finalizer replies to their absence from both frozen prefixes."""
    from experiments.learner import contextual as c
    declaration = campaign.read(run.env.episode_dir / "contextual-finalization.json")["continuation"]
    require(declaration["allowed_actions"] == ["back", "close", "close_capability", "end_turn"]
            and declaration["limit_per_player"] == 8 and declaration["learner_excluded"] is True,
            "Finalizer declaration does not bind both neutral closes and their exclusion")
    result = {}
    for color in ("red", "blue"):
        path = run.env.episode_dir / "agents" / color / "bridge.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        responses = [record["message"] for record in records if record["direction"] == "policy"
                     and record["message"].get("outside_learner_prefix") is True]
        actions = [response["action"]["kind"] for response in responses]
        require(all(kind in ("close_capability", "close", "back", "end_turn") for kind in actions),
                "Technical continuation contained a strategic choice")
        frozen = tuple(c.decode_frame(frame) for frame in run.history(color))
        live = run.env.sides[("red", "blue").index(color)].validator.live.frames
        cutoff = next(frame for frame in frozen if frame.kind == "cutoff")
        require(all(frame.kind != "submitted" for frame in frozen[cutoff.seq + 1:]),
                "Frozen history includes technical finalizer actions")
        prefix_submitted = sum(frame.kind == "submitted" for frame in frozen)
        live_submitted = sum(frame.kind == "submitted" for frame in live)
        require(live_submitted - prefix_submitted == len(responses), "Continuation replies differ from live/frozen history")
        require(declaration["actions"][("red", "blue").index(color)] == len(responses) <= 8,
                "Continuation action count differs from its native finalization declaration")
        result[color] = dict(actions=actions, outside_learner_prefix=True,
                             frozen_submissions=prefix_submitted, live_submissions=live_submitted,
                             declared_allowed_actions=declaration["allowed_actions"], limit_per_player=8)
    if row["case"] == "optional-capability":
        require("close_capability" in result[row["player"]]["actions"],
                "Required native capability finalizer branch was not executed")
    return result


def worker(plan_path, digest):
    import torch
    from experiments.headless import env as environment, contextual_replay
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from validation.probes import audit_contextual_histories as auditor
    from validation.probes.validate_capabilities import CapabilityRun
    from validation.probes.validate_hero_policy import wall_limit, retain_failed_evidence
    from experiments.learner import contextual as c
    require(campaign.binding(plan_path)["sha256"] == digest, "Diagnostic plan changed")
    plan = campaign.read(plan_path)
    output = Path(plan["output"])
    torch.set_num_threads(1)
    report = dict(schema=SCHEMA, status="running", attempts=[], training_input=False, training_samples=0,
                  optimizer_updates=0, plan=campaign.binding(plan_path), source=plan["source"],
                  native_runtime=plan["native_runtime"], parent_descriptor=plan["parent_descriptor"],
                  owned_cleanup_verified=False, protected_inputs_unchanged=False)
    campaign.write(output / "report.json", report, new=True)
    try:
        state = parent_state(plan["parent_descriptor"])
        require(state.weights_sha256 == plan["parent_weights_sha256"], "Restored policy identity changed")
        for row in plan["schedule"]:
            run = launcher = None
            attempt = dict(**row, status="running", training_input=False, cleanup_verified=False)
            report["attempts"].append(attempt)
            campaign.write(output / "report.json", report)
            started = time.monotonic()
            try:
                with wall_limit(plan["case_seconds"]):
                    check_identity(plan)
                    args = SimpleNamespace(output=output, player=row["player"], interface=INTERFACE,
                        binary=Path(plan["binary"]), data=Path(plan["data"]),
                        map=Path(plan["fixtures"][f"{row['fixture']}-{row['player']}"]["path"]),
                        max_days=12, max_decisions=256, watchdog=30, startup_seconds=30,
                        case_seconds=plan["case_seconds"])
                    run = CapabilityRun(f"{row['case']}-{row['player']}", args, environment, c, auditor, row["engine_seed"])
                    launcher = DeadlineVcmiLauncher(args.binary, args.data, args.map, "info",
                        deadline_monotonic=min(time.monotonic() + plan["case_seconds"] + 30, plan["hard_deadline_monotonic"]),
                        deadline_wall=min(time.time() + plan["case_seconds"] + 30, plan["hard_deadline_unix"]),
                        evidence_path=run.directory / "native-process-deadline.jsonl")
                    run.env.launcher = launcher
                    run.start()
                    if row["case"] == "optional-capability":
                        panel = run.open_hero_capability("artifacts")
                        require(not panel.mandatory, "Optional fixture unexpectedly requires a choice")
                        run.end = run.state = run.env.truncate("collection_window")
                    else:
                        attempt["mixed_policy_reconstruction"] = mandatory_cutoff(run, state, row)
                    require(run.end.status == "truncated", "Control did not reach paired cutoff")
                    run.persist_and_audit()
                    attempt["continuation"] = continuation_evidence(run, row)
                    remaining = plan["case_seconds"] - (time.monotonic() - started)
                    require(remaining > 0, "No time remains for exact native replay")
                    replay = contextual_replay.verify_episode(run.env.episode_dir,
                        binary=Path(plan["replay_binary"]["path"]), data=args.data, timeout=remaining)
                    require(replay.get("verified") is True, "Independent native replay rejected cutoff")
                    check_identity(plan)
                    attempt.update(status="passed", history_audit_verified=True, replay_verified=True,
                                   episode_status=run.end.status, termination_reason=run.end.termination_reason,
                                   episode_directory=str(run.env.episode_dir), mechanics=run.summary())
                    attempt["evidence"] = {name: campaign.binding(run.directory / name) for name in ("history-audit.json",)}
                    attempt["evidence"].update(manifest=campaign.binding(run.env.episode_dir / "manifest.json"),
                        replay=campaign.binding(run.env.episode_dir / "contextual-verification/contextual-replay-verification.json"))
            except Exception as error:
                attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
                if run is not None:
                    retain_failed_evidence(run.env, None, attempt, error)
            finally:
                cleanup_errors = []
                if run is not None:
                    try:
                        run.env.close()
                    except Exception as error:
                        cleanup_errors.append(f"environment: {type(error).__name__}: {error}")
                if launcher is not None:
                    try:
                        launcher.close_owned_groups(timeout=10)
                        launcher.assert_closed()
                    except Exception as error:
                        cleanup_errors.append(f"native ownership: {type(error).__name__}: {error}")
                attempt.update(cleanup_verified=not cleanup_errors, cleanup_errors=cleanup_errors,
                               elapsed_seconds=time.monotonic() - started)
                if cleanup_errors:
                    attempt["status"] = "failed"
                campaign.write(output / "report.json", report)
            if attempt["status"] != "passed":
                break
        report["status"] = "passed" if len(report["attempts"]) == 4 and all(
            row["status"] == "passed" for row in report["attempts"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["not_started"] = 4 - len(report["attempts"])
        report["owned_cleanup_verified"] = bool(report["attempts"]) and all(row["cleanup_verified"] for row in report["attempts"])
        try:
            report["protected_inputs_unchanged"] = campaign.verify_protected(plan["protected"])
        except Exception as error:
            report.update(status="failed", protection_error=str(error))
        campaign.write(output / "report.json", report)
    return report


def supervise(plan):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output, process, errors = Path(plan["output"]), None, []
    try:
        with (output / "worker.log").open("xb") as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), "_worker", "--plan", str(output / "plan.json"),
                "--sha256", campaign.binding(output / "plan.json")["sha256"]],
                deadline_monotonic=plan["hard_deadline_monotonic"], deadline_wall=plan["hard_deadline_unix"],
                evidence_path=output / "worker-process-deadline.jsonl", cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            process.wait(timeout=max(1, min(plan["hard_deadline_monotonic"] - time.monotonic(),
                                           plan["hard_deadline_unix"] - time.time())) + 1)
    finally:
        try:
            if process is not None:
                close_owned_process(process, timeout=10)
            for evidence in output.glob("*/native-process-deadline.jsonl"):
                cleanup_from_evidence(evidence, timeout=10)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        protected = campaign.verify_protected(plan["protected"])
        campaign.write(output / "shutdown.json", dict(owned_cleanup_verified=not errors, errors=errors,
            protected_inputs_unchanged=protected, returncode=process.returncode if process else None), new=True)
    report = campaign.read(output / "report.json") if (output / "report.json").exists() else {"status": "failed"}
    require(process is not None and process.returncode == 0 and not errors and report.get("status") == "passed",
            "Cutoff controls failed; preserve all evidence and never replay this seed reservation")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-seconds", type=float, default=180)
    parser.add_argument("command", nargs="?", choices=("_worker",))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        require(args.plan is not None and args.sha256 is not None, "Worker requires a bound plan")
        return int(worker(args.plan, args.sha256)["status"] != "passed")
    require(args.output is not None and math.isfinite(args.case_seconds) and 60 <= args.case_seconds <= 600,
            "Use a fresh private output and a finite 60..600-second case bound")
    with campaign.activity_lease():
        report = supervise(prepare(args))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
