#!/usr/bin/env python3
"""Bounded passive-army native controls, never a training run.

Check mode is read-only. Explicit --run reserves every seed before starting an
independently supervised worker. Each attempt is retained, including failures;
there is no retry. Scripted mechanics controls and a 64-choice frozen-policy
prefix are deliberately unscored diagnostics, not completed policy games.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_selfplay_training as campaign
from experiments.learner import contextual as c
from experiments.learner.interaction_guard import RECRUIT_ONLY_BLOCKED_KINDS, RECRUIT_ONLY_VERSION

SCHEMA = "heroes-passive-armies-native-v1"
BASE = campaign.ARTIFACTS / "passive-army-diagnostics"
ENGINE_SEED, POLICY_SEED = 1_978_000_000, 1_979_000_000
CASES = ("legacy-red", "original-red", "original-blue", "artifacts-red", "artifacts-blue", "tiny-prefix")
NORMAL_CHOICES, RESERVED_CHOICES = 64, 16
require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound


def source_implementation():
    """The same source inventory as the manual launcher, before acceptance exists."""
    from experiments.learner.inference import source_identity
    from validation.probes.validate_capabilities import capability_implementation_identity
    result = capability_implementation_identity()
    result.update(source_identity(campaign.INTERFACE)["source_sha256"])
    for name in ("tools/run_selfplay_training.py", "tools/benchmark_copied_training.py"):
        result[name] = binding(ROOT / name)["sha256"]
    return result


def remaining(plan, reserve=10):
    return campaign.remaining(plan, reserve)


def validate_passive(observation, *, enabled=True):
    """Check complete owned identity coverage, including empty actual town armies."""
    if not enabled:
        require(observation.army_information is None and not observation.own_armies,
                "Legacy observation unexpectedly acquired passive army fields")
        return {}
    require(observation.army_information == c.OWN_ARMY_INFORMATION, "Passive army marker is absent")
    expected = {hero.handle: "hero" for hero in observation.heroes}
    expected.update({town.handle: "town_garrison" for town in observation.towns})
    actual = {army.subject: army.role for army in observation.own_armies}
    require(actual == expected and len(actual) == len(observation.own_armies),
            "Passive armies omit an owner, duplicate a garrison hero, or expose a foreign army")
    return {army.subject: army for army in observation.own_armies}


def composition(army):
    result = Counter()
    for stack in army.slots:
        result[stack.creature] += stack.count
    return result


def visible_composition(stacks):
    return tuple((stack.slot, stack.creature, stack.count) for stack in stacks if stack.count)


def validate_policy_proofs(policies, collected):
    require(len(policies) == len(collected.rollouts) == 2, "Both policy reconstructions are required")
    for player, (proof, rollout) in enumerate(zip(policies, collected.rollouts)):
        require(proof.get("player") == player and proof.get("checkpoint") == rollout.behavior_checkpoint
                and proof.get("maximum_absolute_difference") == 0
                and proof.get("private_memory_reset") is True and proof.get("weights_unchanged") is True
                and proof.get("actions_resampled") == len(rollout.decisions) > 0
                and proof.get("values_recomputed") == len(rollout.estimates),
                "Policy reconstruction omitted choices, values, or exact unchanged weights")


def mutate_own_attack(records):
    """Mutate native tape data directly; do not merely invalidate a file digest."""
    result = deepcopy(records)
    for row in result:
        if row.get("kind") != "message":
            continue
        for record in row["message"].get("records", []):
            observation = record.get("observation") or {}
            for army in observation.get("own_armies", []):
                if army["slots"]:
                    army["slots"][0]["attack"] += 1
                    return result
    raise ValueError("No captured own stack is available for the native mutation control")


def _factory(plan, *, enabled):
    from experiments.headless import env
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    launchers = []
    def launcher(*args):
        value = DeadlineVcmiLauncher(*args, deadline_monotonic=plan["hard_deadline_monotonic"] - 10,
                                     deadline_wall=plan["hard_deadline_unix"] - 10)
        launchers.append(value)
        return value
    def environment(*args, **kwargs):
        return env.HeadlessSelfPlayEnv(*args, **kwargs,
            **({"army_information": c.OWN_ARMY_INFORMATION} if enabled else {}))
    return SimpleNamespace(Limits=env.Limits, VcmiLauncher=launcher, HeadlessSelfPlayEnv=environment), launchers


def original_controls(run):
    initial = run.inspect_request().observation
    require(initial.context.kind == "map" and initial.window is None, "Initial army evidence required a click")
    initial_armies = validate_passive(initial)
    actor, town = run.own_hero(), initial.towns[0].handle
    require(initial_armies[actor].slots, "Original hero has no occupied army slot")
    run.check("map_passive_own_composition_without_clicks", "passed", armies=len(initial_armies))
    window = run.hero_window()
    require(visible_composition(window.army) == visible_composition(initial_armies[actor].slots)
            and run.inspect_request().observation.own_armies == initial.own_armies,
            "Explicit hero inspection differs from passive composition or changes current stats")
    require(run.inspect_request().observation.context.kind == "map", "Explicit inspection did not close to map")
    run.check("inspection_preserves_passive_army_and_stats", "passed")
    run.open_town()
    before = validate_passive(run.inspect_request().observation)
    panel = run.open_recruitment()
    require(panel.slider_max >= 1, "Original map cannot recruit one displayed creature")
    creature = next(card.creature for card in panel.offers if card.ref == panel.selected_offer_ref)
    old_frame, additions, result = run.call(run.purchase(panel, 1), "explicit_buy_one_creature")
    from validation.probes.validate_town_native import request_evidence
    request_evidence(run, additions, result, "recruited")
    run.panel("overview")
    after = validate_passive(run.inspect_request().observation)
    expected = composition(before[town]); expected[creature] += 1
    require(composition(after[town]) == expected and after[actor] == before[actor],
            "Recruitment passive composition has the wrong destination or count")
    require(tuple(a - b for a, b in zip(old_frame.observation.resources,
            run.inspect_request().observation.resources)) == panel.unit_cost, "Recruitment cost changed")
    run.check("recruitment_updates_actual_town_army", "passed", creature=creature, count=1)
    from validation.probes.validate_army_management import exact_garrison
    expected_combined = composition(after[actor]) + composition(after[town])
    for operation in ("enter", "exit"):
        old, additions, result = run.call(run.town_action("swap_garrison"), "explicit_garrison_" + operation)
        current = run.inspect_request()
        proof = exact_garrison(old, current, additions, result)
        armies = validate_passive(current.observation)
        require(proof["operation"] == operation and composition(armies[actor]) == expected_combined
                and not armies[town].slots, "Passive garrison rows duplicated or lost recruited troops")
        require(visible_composition(armies[actor].slots) == visible_composition(next(
            row for row in current.observation.window.armies if row.subject == actor).slots),
            "Passive garrison army differs from explicit current window")
        run.check("passive_garrison_" + operation, "passed", operation=operation)
    run.close_to_map()


def artifact_controls(run):
    before = run.inspect_request().observation
    actor = run.own_hero()
    armies = validate_passive(before)
    skills_before = run.hero_window().primary_skills
    panel = run.open_hero_capability("artifacts")
    # The declared fixture equips Sword of Judgement in native right-hand slot3.
    # Select only its currently displayed move to a currently offered backpack slot.
    choice = next((choice for choice in panel.choices if choice.kind == "move_artifact"
        and choice.subject == actor == choice.destination and choice.slot == 3
        and choice.destination_slot >= 19), None)
    require(choice is not None, "Authored stat artifact has no displayed unequip choice")
    run.choose(choice)
    after = validate_passive(run.inspect_request().observation)
    run.close_to_map()
    skills_after = run.hero_window().primary_skills
    attack_delta, defense_delta = (skills_after[i] - skills_before[i] for i in (0, 1))
    require(attack_delta == defense_delta == -5, "Declared artifact did not change native primary stats by -5")
    expected = tuple(replace(stack, attack=stack.attack + attack_delta, defense=stack.defense + defense_delta)
                     for stack in armies[actor].slots)
    require(after[actor].slots == expected and all(after[key] == value for key, value in armies.items() if key != actor),
            "Passive current stack stats fail to follow the artifact's native owner-only effect")
    run.check("artifact_current_combat_stats_update", "passed", attack_delta=attack_delta,
              defense_delta=defense_delta, affected_slots=len(expected))


def replay_case(folder, plan):
    from experiments.headless import contextual_replay
    report = contextual_replay.verify_episode(folder, binary=bound(plan["replay_binary"]),
        data=Path(plan["data"]), timeout=remaining(plan))
    require(report.get("verified") is True and report.get("histories", {}).get("verified") is True,
            "Exact native replay or independent history reconstruction failed")
    return report


def mechanics_case(case, plan):
    from validation.probes import validate_capabilities as probes
    from validation.probes.validate_hero_recruitment import finalize_diagnostic_failure
    from experiments.explorer import create_capability_scenarios as fixtures
    color = case.rsplit("-", 1)[1]
    enabled = not case.startswith("legacy-")
    game_map = Path(plan["game_map"])
    if case.startswith("artifacts-"):
        game_map = Path(plan["output"]) / "maps" / (case + ".vmap")
        fixture = fixtures.create_scenario(game_map, fixture="artifacts", player=color)
        write(game_map.with_suffix(".manifest.json"), fixture, new=True)
        fixtures.verify_scenario(game_map, fixture="artifacts", player=color)
    module, launchers = _factory(plan, enabled=enabled)
    args = SimpleNamespace(output=Path(plan["output"]), binary=Path(plan["binary"]), data=Path(plan["data"]),
        map=game_map, player=color, max_days=365, max_decisions=64, watchdog=20, startup_seconds=30,
        case_seconds=min(90, remaining(plan)), interface=campaign.INTERFACE)
    class PassiveRun(probes.CapabilityRun):
        def inspect_request(self):
            frame = super().inspect_request()
            validate_passive(frame.observation, enabled=enabled)
            return frame
    run = PassiveRun(case, args, module, c, probes.auditor, plan["seeds"][case])
    error = None
    try:
        run.start()
        if case.startswith("legacy-"):
            run.call(run.action("end_turn"), "legacy_explicit_turn")
            run.check("legacy_runtime_optional_fields_absent_both_seats", "passed")
        elif case.startswith("original-"):
            original_controls(run)
        else:
            artifact_controls(run)
        run.finish(); run.persist_and_audit()
        require(run.end.status == "truncated", "Technical prefix unexpectedly completed a game")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        finalize_diagnostic_failure(run, error)
    finally:
        try:
            if getattr(run, "env", None) is not None:
                run.env.close()
        finally:
            for launcher in launchers:
                launcher.close_owned_groups(timeout=min(5, remaining(plan, 0)))
                launcher.assert_closed()
    summary = run.summary(error)
    write(run.directory / "controls.json", summary, new=True)
    require(summary["probe_status"] == "passed", f"{case} failed: {error}")
    report = replay_case(run.env.episode_dir, plan)
    summary["native_replay_verified"] = report["verified"]
    summary["episode_dir"] = str(run.env.episode_dir)
    return summary


def tiny_prefix(plan):
    from experiments.headless import env as native
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.contextual_storage import save_collected
    from experiments.learner.selfplay_training import reconstruct_policies
    from validation.probes.validate_hero_policy import retain_failed_evidence
    from tools import run_tiny_selfplay_one_game as tiny
    models = tiny.frozen_models(plan)
    folder = Path(plan["output"]) / "tiny-prefix"
    launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["game_map"]),
        deadline_monotonic=plan["hard_deadline_monotonic"] - 10, deadline_wall=plan["hard_deadline_unix"] - 10)
    collector = ContextualCollector(models, (tiny.CHECKPOINT,) * 2, learner_player=0,
        opponent_pool=campaign.POOL, interface=campaign.INTERFACE, purpose="diagnostic", origin="native",
        policy_seeds=tuple(plan["policy_seeds"]), guard_version=RECRUIT_ONLY_VERSION)
    env = native.HeadlessSelfPlayEnv(launcher, folder,
        native.Limits(max_days=365, max_actions=NORMAL_CHOICES + RESERVED_CHOICES, watchdog_seconds=20, startup_seconds=30,
                      shutdown_seconds=15, bridge_timeout=30), ledger=True, replay=True,
        interface=campaign.INTERFACE, army_information=c.OWN_ARMY_INFORMATION,
        agents={color: dict(checkpoint=tiny.CHECKPOINT, kind="frozen-neural", purpose="diagnostic")
                for color in ("red", "blue")})
    try:
        collected = collector.run_episode(env, episode=1, seed=plan["seeds"]["tiny-prefix"],
            max_requests_per_player=NORMAL_CHOICES, max_dialog_requests_per_player=RESERVED_CHOICES)
        save_collected(env.episode_dir / "neural-rollout.json", collected)
    except Exception as error:
        attempt = dict(status="failed", error=f"{type(error).__name__}: {error}", training_input=False)
        retain_failed_evidence(env, collector, attempt, error, cleanup_seconds=min(15, remaining(plan, 0)))
        write(folder / "capture-failure.json", attempt, new=True)
        raise
    finally:
        try:
            env.close()
        finally:
            launcher.close_owned_groups(timeout=min(5, remaining(plan, 0))); launcher.assert_closed()
    require(all(rollout.purpose == "diagnostic" and not rollout.optimization_eligible
                and 0 < len(rollout.decisions) <= NORMAL_CHOICES + RESERVED_CHOICES
                and all(decision.action.kind not in RECRUIT_ONLY_BLOCKED_KINDS for decision in rollout.decisions)
                for rollout in collected.rollouts), "Both seats need nonempty diagnostic choices with the selected support")
    manifest = read(env.episode_dir / "manifest.json")
    require(manifest["status"] == "truncated" and all(value is None for value in manifest["rewards_by_player"].values()),
            "Frozen technical prefix was unexpectedly scored as a complete game")
    policies = reconstruct_policies(collected, tiny.frozen_models(plan), plan["policy_seeds"], lambda: remaining(plan))
    validate_policy_proofs(policies, collected)
    replay = replay_case(env.episode_dir, plan)
    result = dict(case="tiny-prefix", probe_status="passed", policies=policies,
        native_replay_verified=replay["verified"], episode_dir=str(env.episode_dir),
        choices_by_player=[len(rollout.decisions) for rollout in collected.rollouts],
        normal_choices_per_player=NORMAL_CHOICES, reserved_dialog_choices_per_player=RESERVED_CHOICES,
        inference_backend="python", optimizer_updates=0, training_samples_admitted=0)
    write(folder / "controls.json", result, new=True)
    return result


def native_mutant(folder, plan):
    from experiments.headless import contextual_replay as replay
    from experiments.headless import replay as legacy
    from experiments.headless.process_deadline import guarded_popen, close_owned_process
    output = Path(plan["output"]) / "native-own-attack-mutant"
    output.mkdir()
    manifest = read(folder / "manifest.json")
    tape, _ = replay.records(folder / "engine/contextual-replay.jsonl", cutoff=True)
    server, _ = replay.records(folder / "engine/replay.jsonl", cutoff=True)
    server, _ = legacy.complete_replay_records(server, cut_short=True)
    legacy.write_jsonl(output / "replay.jsonl", server)
    legacy.write_jsonl(output / "contextual-replay.jsonl", mutate_own_attack(tape))
    environment, _ = legacy.replay_profile(output, folder, Path(plan["data"]), manifest["launch"]["requested_settings"])
    inherited = {key: value for key, value in os.environ.items() if not key.startswith(
        ("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_"))}
    report_path = output / "native-report.json"
    process = None
    try:
        with (output / "native.log").open("xb") as stream:
            process = guarded_popen([str(bound(plan["replay_binary"])), "--log", str(output / "replay.jsonl"),
                "--contextual", str(output / "contextual-replay.jsonl"), "--report", str(report_path), "--prefix"],
                deadline_monotonic=plan["hard_deadline_monotonic"] - 10,
                deadline_wall=plan["hard_deadline_unix"] - 10,
                evidence_path=output / "native-process-deadline.jsonl", cwd=Path(plan["binary"]).parent,
                env={**inherited, **environment}, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait(timeout=remaining(plan))
        report = read(report_path)
        require(code == 3 and report.get("schema") == replay.REPORT_SCHEMA and report.get("verified") is False
                and "Contextual replay differs at record " in report.get("error", ""),
                "Native replay accepted a fabricated current own-stack attack statistic")
        return dict(verified=True, returncode=code, native_report=binding(report_path),
                    mutation="first_own_stack_attack_plus_one", layer="native_regenerated_observation")
    finally:
        if process is not None:
            close_owned_process(process, timeout=min(5, remaining(plan, 0)))
        shutil.rmtree(output / "profile", ignore_errors=True)


def check_inputs(args):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.episodes.scoring import engine_recipe_digests
    from tools import run_tiny_selfplay_one_game as tiny
    runtime = args.runtime.resolve()
    freeze = read(runtime / "freeze-receipt.json")
    require(freeze.get("source_recipe_check") is True and freeze.get("frozen_bin") == str(runtime / "bin"),
            "Runtime must first pass the independent freeze recipe check")
    require(all(freeze["source_inputs"].get(key, {}).get("sha256") == value
                for key, value in engine_recipe_digests(ROOT).items()), "Runtime recipe differs from current canonical engine")
    game_map = bound(freeze["map"])
    preparation = read(bound(dict(path=str(tiny.PREPARATION), sha256=tiny.PREPARATION_SHA,
                                 bytes=tiny.PREPARATION.stat().st_size)))
    require(preparation["policy"]["sha256"] == tiny.POLICY_SHA, "Tiny initial checkpoint differs")
    protected = {**preparation["protected"], preparation["policy"]["path"]: preparation["policy"]}
    campaign.verify_protected(protected)
    require(args.data.resolve().is_dir(), "Game data directory is absent")
    return dict(binary=str(runtime / "bin/vcmiclient"), replay_binary=binding(runtime / "bin/vcmireplay"),
        runtime=runtime_identity(runtime / "bin"), runtime_freeze=binding(runtime / "freeze-receipt.json"),
        game_map=str(game_map), map=binding(game_map), data=str(args.data.resolve()),
        source_implementation=source_implementation(), harness=binding(__file__), protected=protected,
        tiny_policy=preparation["policy"], weights_sha256=tiny.WEIGHTS_SHA,
        model_checkpoint=tiny.CHECKPOINT, tiny_metadata=tiny.expected_metadata())


def recheck(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    require(source_implementation() == plan["source_implementation"] and binding(__file__) == plan["harness"],
            "Diagnostic implementation changed during capture")
    require(runtime_identity(Path(plan["binary"]).parent) == plan["runtime"], "Frozen runtime changed")
    bound(plan["map"]); bound(plan["runtime_freeze"]); bound(plan["replay_binary"])
    campaign.verify_protected(plan["protected"])


def _seed_row(sequence, output):
    return dict(schema=SCHEMA + "-seed-reservation", sequence=sequence, output=str(output),
        seeds={case: ENGINE_SEED + sequence * 16 + index for index, case in enumerate(CASES)},
        policy_seeds=[POLICY_SEED + sequence * 16 + i for i in (0, 1)],
        purpose="diagnostic", consumed_including_unstarted=True)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_json(left, right):
    # JSON distinguishes booleans from integer seed/sequence values.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def reserve_seeds(output):
    """Burn all seeds before creating an attempt, with independent durable receipts.

    A missing ledger, orphan receipt, interrupted write or altered schedule fails
    closed. An unstarted attempt is allowed, but its reserved sequence is consumed.
    """
    BASE.mkdir(parents=True, exist_ok=True)
    _fsync_directory(BASE.parent)
    require(output.resolve() == output and output.parent == BASE.resolve()
            and not output.exists(), "Seed reservation requires a fresh private attempt path")
    with (BASE / "seed-ledger.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = BASE / "seed-ledger.jsonl"
        receipts = BASE / "reservation-receipts"
        require(not ledger.is_symlink() and not receipts.is_symlink(), "Seed evidence must not be symlinked")
        if not ledger.exists():
            require(set(BASE.iterdir()) == {BASE / "seed-ledger.lock"},
                    "Missing seed ledger with prior diagnostic evidence; refuse seed reuse")
        entries = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        require(all(type(row) is dict and row.keys() == _seed_row(0, output).keys()
                    and type(row["output"]) is str for row in entries), "Malformed diagnostic seed ledger record")
        allowed_paths = {BASE / name for name in ("seed-ledger.lock", "seed-ledger.jsonl", "reservation-receipts")}
        allowed_paths.update(Path(row["output"]) for row in entries)
        require(set(BASE.iterdir()) <= allowed_paths, "Diagnostic evidence exists outside its seed ledger")
        expected_receipts = {receipts / f"{index:06d}.json" for index in range(len(entries))}
        require((set(receipts.iterdir()) if receipts.exists() else set()) == expected_receipts,
                "Seed ledger and independent reservation receipt inventory disagree")
        outputs = set()
        for index, prior in enumerate(entries):
            path = Path(prior["output"])
            require(path.resolve() == path and path.parent == BASE.resolve() and str(path) not in outputs,
                    "Seed ledger reused or changed a private attempt path")
            receipt = receipts / f"{index:06d}.json"
            require(not receipt.is_symlink() and _same_json(prior, _seed_row(index, path))
                    and _same_json(read(receipt), prior),
                    "Seed schedule differs from its independent receipt or deterministic namespace")
            attempt_receipt = path / "seed-reservation.json"
            if path.exists():
                require(attempt_receipt.is_file() and not attempt_receipt.is_symlink()
                        and _same_json(read(attempt_receipt), prior),
                        "Attempt seed receipt is absent or differs from central reservation")
            outputs.add(str(path))
        require(str(output) not in outputs, "A previously reserved attempt cannot restart")
        sequence = len(entries)
        require(sequence < 10_000, "Diagnostic seed namespace exhausted")
        row = _seed_row(sequence, output)
        receipts.mkdir(exist_ok=True)
        _fsync_directory(BASE)
        write(receipts / f"{sequence:06d}.json", row, new=True)
        with ledger.open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode()); stream.flush(); os.fsync(stream.fileno())
        _fsync_directory(BASE)
        return row


def worker(path, sha):
    from validation.probes.validate_hero_policy import wall_limit
    require(binding(path)["sha256"] == sha, "Worker plan binding changed")
    plan = read(path); output = Path(plan["output"])
    require(os.getppid() == plan["supervisor_pid"], "Only the lease-owning supervisor can start this worker")
    write(output / "worker-claim.json", dict(pid=os.getpid(), plan_sha256=sha), new=True)
    report = dict(schema=SCHEMA, status="running", runtime=plan["runtime"],
        source_implementation=plan["source_implementation"], army_information=c.OWN_ARMY_INFORMATION,
        policy_support=RECRUIT_ONLY_VERSION, players=[0, 1], controls=0, cases=[],
        scope="bounded_technical_prefixes", full_policy_games=0, optimizer_updates=0,
        training_samples_admitted=0, native_replay_verified=False, both_policies_reconstructed=False,
        owned_cleanup_verified=False, plan=binding(path))
    write(output / "report.json", report, new=True)
    try:
        with wall_limit(remaining(plan)):
            recheck(plan)
            import torch
            torch.set_num_threads(1)
            # Copy only the immutable initial policy. Official artifacts are read-only.
            policy_copy = output / "initial-policy.pt"
            shutil.copyfile(bound(plan["tiny_policy"]), policy_copy)
            plan = {**plan, "tiny_policy": binding(policy_copy)}
            for case in CASES:
                remaining(plan)
                result = tiny_prefix(plan) if case == "tiny-prefix" else mechanics_case(case, plan)
                report["cases"].append(result)
                report["controls"] += len(result.get("checks", []))
                write(output / "report.json", report)
            original = next(row for row in report["cases"] if row["case"] == "original-red")
            report["native_stat_mutant"] = native_mutant(Path(original["episode_dir"]), plan)
            recheck(plan)
            report.update(status="passed", native_replay_verified=True, both_policies_reconstructed=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        write(output / "report.json", report)
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--runtime", type=Path, default=campaign.ARTIFACTS / "runtime/local-v8-passive-armies-v1")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sha", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_plan:
        return worker(args.worker_plan, args.worker_sha)
    require(args.data is not None and 60 <= args.seconds <= 600 and not (args.run and args.check),
            "Provide --data, 60..600 seconds, and either --check or --run")
    plan = check_inputs(args)
    if not args.run:
        print(json.dumps(dict(status="ready_for_explicit_diagnostic_run", cases=CASES,
            total_seconds=args.seconds, gameplay_started=False, training_started=False,
            runtime=plan["runtime"], army_information=c.OWN_ARMY_INFORMATION), sort_keys=True))
        return 0
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output = (args.output or BASE / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                    + "-" + secrets.token_hex(3))).resolve()
    require(output.is_relative_to(BASE) and not output.exists(), "Require a fresh private diagnostic output")
    started = time.monotonic()
    with campaign.activity_lease():
        reservation = reserve_seeds(output)
        output.mkdir(parents=True)
        plan.update(output=str(output), seeds=reservation["seeds"], policy_seeds=reservation["policy_seeds"],
                    supervisor_pid=os.getpid(), hard_deadline_monotonic=started + args.seconds,
                    hard_deadline_unix=time.time() + args.seconds)
        write(output / "seed-reservation.json", reservation, new=True)
        path = output / "plan.json"; write(path, plan, new=True)
        process, supervision_error = None, None
        try:
            with (output / "worker.log").open("xb") as stream:
                process = guarded_popen([sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(path),
                    "--worker-sha", binding(path)["sha256"]],
                    deadline_monotonic=plan["hard_deadline_monotonic"] - 5,
                    deadline_wall=plan["hard_deadline_unix"] - 5,
                    evidence_path=output / "worker-process-deadline.jsonl", cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                process.wait(timeout=remaining(plan, 4))
        except Exception as error:
            supervision_error = f"{type(error).__name__}: {error}"
        finally:
            report_path = output / "report.json"
            report = read(report_path) if report_path.is_file() else dict(schema=SCHEMA, status="failed",
                error="Worker ended without a report", training_samples_admitted=0, optimizer_updates=0)
            if supervision_error or process is None or process.returncode != 0 or report.get("status") != "passed":
                report.update(status="failed", supervision_error=supervision_error,
                              worker_exit_code=process.returncode if process is not None else None)
            try:
                if process is not None:
                    close_owned_process(process, timeout=min(3, remaining(plan, 0)))
                groups = []
                for receipt in sorted(output.rglob("*-process-deadline.jsonl")):
                    groups.extend(cleanup_from_evidence(receipt, timeout=min(2, remaining(plan, 0))))
                report.update(owned_cleanup_verified=True, process_groups=groups)
            except Exception as error:
                report.update(status="failed", owned_cleanup_verified=False,
                              cleanup_error=f"{type(error).__name__}: {error}")
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            try:
                recheck(plan)
            except Exception as error:
                report.update(status="failed", preservation_error=f"{type(error).__name__}: {error}")
            write(report_path, report)
    print(json.dumps(dict(status=report["status"], report=str(output / "report.json"))))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
