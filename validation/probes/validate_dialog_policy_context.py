"""Frozen-policy dialog samples after explicit isolated-fixture navigation.

This mixed operator/neural diagnostic is excluded from training and ordinary
self-play evaluation. Only navigation is scripted. Every actual dialog answer
is sampled from the unchanged v7 policy, saved before sending, and resampled
independently from the complete player-local history and declared policy seed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.explorer.create_dialog_scenarios import MAPS
from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint
from experiments.learner.contextual_rollout import BehaviorDecision, ContextualSession, DecisionEstimate, RecurrentPrefix
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes.validate_adventure_dialogs import DialogRun, OBJECT_KINDS, dialog_open_proof, fixture_identity
from validation.probes.validate_army_management import native_receipt
from validation.probes.validate_hero_policy import retain_failed_evidence, wall_limit
from validation.probes.validate_hero_policy_context import checkpoint_identity

FAMILIES = {"guarded_mine": "guarded_mine", "neutral_free_join": "neutral_free",
            "neutral_paid_join": "neutral_paid", "neutral_flee": "neutral_flee",
            "chest": "chest", "arena": "arena", "school_magic": "school_magic", "school_war": "school_war"}
NEURAL_LABEL = "neural_full_distribution_dialog_choice"
EVIDENCE_SCHEMA = "heroes-v7-neural-dialog-context-evidence-v1"


@dataclass(frozen=True)
class Settings:
    engine_seed: int = 109101
    policy_seed: int = 109201
    max_choices: int = 8
    case_seconds: float = 120.0

    def __post_init__(self):
        for seed in (self.engine_seed, self.policy_seed):
            base.ensure(type(seed) is int and 1 <= seed <= 2147483000, "Seeds must permit the entire declared schedule")
        base.ensure(type(self.max_choices) is int and 1 <= self.max_choices <= 8, "Use at most eight real dialog choices")
        base.ensure(type(self.case_seconds) in (int, float) and math.isfinite(self.case_seconds)
                    and 0 < self.case_seconds <= 180, "Use a finite case wall limit of at most 180 seconds")

    def schedule(self, family="all", player="both"):
        base.ensure(family in (*FAMILIES, "all") and player in (*base.COLORS, "both"), "Unknown diagnostic family or player")
        return [{"family": name, "fixture": FAMILIES[name], "player": color,
                 "engine_seed": self.engine_seed + 8 * tuple(FAMILIES).index(name) + base.COLORS.index(color),
                 "policy_seeds": [self.policy_seed + 16 * tuple(FAMILIES).index(name) + 2 * base.COLORS.index(color) + p
                                  for p in (0, 1)]}
                for name in (FAMILIES if family == "all" else (family,))
                for color in (base.COLORS if player == "both" else (player,))]


def choice_evidence(sessions):
    return {"schema": EVIDENCE_SCHEMA, "training_input": False, "purpose": "diagnostic",
            "mixed_scripted_and_neural_history": True, "self_play": False, "optimizer_steps": 0,
            "players": [{"player": session.history.player, "checkpoint": session.checkpoint,
                "policy_seed": session.diagnostic_seed, "model_config": session.model.config.to_dict(),
                "history": [c.encode_frame(frame) for frame in session.history.frames],
                "neural_decisions": [choice.to_dict() for choice in session._choices.values()],
                "estimates": [DecisionEstimate(seq, value, session._prefixes[seq]).to_dict()
                              for seq, value in session._values.items()]} for session in sessions]}


def replay_choices(raw, model):
    """Fresh memory and RNG for each side; never sample operator decisions."""
    base.ensure(raw["schema"] == EVIDENCE_SCHEMA and raw["training_input"] is False
                and raw["purpose"] == "diagnostic" and raw["mixed_scripted_and_neural_history"] is True
                and raw["self_play"] is False and raw["optimizer_steps"] == 0, "Expected mixed diagnostic evidence")
    base.ensure([row["player"] for row in raw["players"]] == [0, 1], "Expected separate ordered player histories")
    reports = []
    for row in raw["players"]:
        history = tuple(c.decode_frame(frame) for frame in row["history"])
        base.ensure(history and all(frame.schema == c.DIALOGS_SCHEMA and frame.player == row["player"] for frame in history),
                    "Expected complete player-local v7 history")
        base.ensure(row["model_config"] == model.config.to_dict(), "Diagnostic model configuration changed")
        decisions = tuple(BehaviorDecision.from_dict(item, history[item["decision_seq"]]) for item in row["neural_decisions"])
        by_seq = {item.decision_seq: item for item in decisions}
        base.ensure(len(by_seq) == len(decisions), "Duplicate neural choices")
        session = ContextualSession(model, history[0].episode, row["player"], row["checkpoint"],
                                    "scripted-context-operator", schema=c.DIALOGS_SCHEMA, seed=row["policy_seed"])
        for frame in history:
            session.consume((frame,))
            if frame.seq in by_seq:
                base.ensure(frame.kind == "decision" and frame.observation.context.kind == "dialog"
                            and by_seq[frame.seq].action.kind == "choose_dialog", "Neural choice escaped its real dialog")
                base.ensure(session.choose(frame.seq) == by_seq[frame.seq],
                            "Neural action, value, full support, prefix or conditional probabilities changed")
        expected = tuple(DecisionEstimate(item["decision_seq"], item["value"], RecurrentPrefix(**item["prefix"]))
                         for item in row["estimates"])
        actual = tuple(DecisionEstimate(seq, value, session._prefixes[seq]) for seq, value in session._values.items())
        base.ensure(actual == expected and len(session._choices) == len(decisions), "Saved own-decision values or neural choices changed")
        reports.append({"player": row["player"], "actions_resampled": len(decisions), "values_recomputed": len(actual),
                        "conditionals_recomputed": sum(len(item.conditionals) for item in decisions)})
    return {"passed": True, "players": reports, "maximum_absolute_difference": 0,
            "complete_own_histories": True, "separate_private_memory_resets": True}


def consume_histories(run, sessions):
    for color, session in zip(base.COLORS, sessions):
        session.consume(run.history(color))


def sample_dialog_chain(run, sessions, maximum):
    """Defer a cutoff without choosing an answer; sample every chained query."""
    first = run.inspect_request()
    base.ensure(first.observation.context.kind == "dialog", "Neural control requires an actual dialog")
    run.env.defer_dialog_cutoff("collection_window")
    observed = []
    for _ in range(maximum):
        frame = run.inspect_request()
        base.ensure(frame.observation.context.kind == "dialog", "Deferred cutoff returned a non-dialog live request")
        consume_histories(run, sessions)
        session = sessions[run.state.player]
        choice = session.choose(frame.seq)
        base.ensure(choice.action.kind == "choose_dialog", "Policy produced a non-dialog choice")
        panel = frame.observation.window.dialog_panel
        selected = next(item for item in panel.choices if item.ref == choice.action.selection)
        # A failed native send still leaves its exact original sample on disk.
        write_json(run.directory / "neural-context-evidence.json", choice_evidence(sessions))
        before, additions, result = run.call(choice.action, NEURAL_LABEL)
        receipt = native_receipt(before, additions, result, "dialog_answered")
        observed.append({"player": frame.player, "decision_seq": frame.seq, "family": panel.family,
                         "choice": asdict(selected), "receipt": receipt})
        if run.state.done:
            base.ensure(run.state.status == "truncated", "Deferred diagnostic did not finish at an unscored paired cutoff")
            return observed
        proof = dialog_open_proof(before, run.inspect_request(), additions, result)
        run.check("neural_answer_opened_native_chained_dialog", "passed", **proof)
    raise base.ProbeFailure("Real dialog chain exceeded its declared policy-choice bound; no answer was fabricated")


def coverage(expected, observed):
    seen = sorted({item["family"] for item in observed})
    missing = [] if observed and observed[0]["family"] == expected else [expected]
    return {"status": "unverified" if missing else "passed", "expected_first_dialog": expected,
            "observed_families": seen, "missing": missing, "sampled_choices": len(observed),
            "all_alternative_answers_claimed": False}


def validate_drain(manifest, observed):
    drain = manifest.get("dialog_cutoff_drain")
    if not observed:
        base.ensure(drain is None, "An unsampled dialog cutoff was deferred")
        return {"status": "unverified", "reason": "No first dialog was observed"}
    expected = [{"player": item["player"], "decision_seq": item["decision_seq"]} for item in observed]
    base.ensure(drain is not None and drain["reason"] == "collection_window"
                and drain["limit"] == 8 and drain["learner_included"] is True
                and {key: drain["trigger"][key] for key in expected[0]} == expected[0]
                and drain["choices"] == expected, "Deferred cutoff omitted or replaced an actual policy answer")
    return {"status": "passed", "real_choices_before_cutoff": len(expected),
            "dialog_cutoff_drain": drain, "training_input": False}


def run(args):
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    schedule = settings.schedule(args.family, args.player)
    torch.set_num_threads(1)
    for name in ("binary", "data", "output", "maps", "checkpoint"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    base.ensure(not args.output.exists(), "Output already exists; retain every attempt")
    fixtures = {(row["fixture"], row["player"]): fixture_identity(row["fixture"], row["player"], args.maps) for row in schedule}
    for identity in fixtures.values():
        environment.VcmiLauncher(args.binary, args.data, Path(identity["path"])).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    loaded = load_policy_checkpoint(args.checkpoint, interface=c.DIALOGS_ENVIRONMENT)
    loaded.model.requires_grad_(False)
    frozen = checkpoint_identity(loaded, args.checkpoint)
    implementation = implementation_identity()
    implementation["experiments/explorer/create_dialog_scenarios.py"] = digest(ROOT / "experiments/explorer/create_dialog_scenarios.py")
    native = runtime_identity(args.binary.parent)
    args.output.mkdir(parents=True)
    args.interface, args.max_days, args.max_decisions = c.DIALOGS_ENVIRONMENT, 8, 96
    args.watchdog, args.startup_seconds = 30, 30
    plan = {"schema": "heroes-v7-neural-dialog-context-plan-v1", "settings": asdict(settings), "schedule": schedule,
            "checkpoint": frozen, "checkpoint_source": loaded.source_metadata, "implementation": implementation,
            "native_runtime": native, "binary_sha256": digest(args.binary), "fixtures": list(fixtures.values()),
            "bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "watchdog", "startup_seconds")},
            "script_control": "Visit one currently visible fixture target through explicit native navigation; no scripted dialog answers",
            "neural_control": "Unchanged policy answers the first dialog and its actual chain; defer collection cutoff until a real non-dialog boundary",
            "history": "Two separate complete own histories including operator actions; no opponent history inputs",
            "missing_branches": "Unverified; no adaptive resampling, alternate targets or seeds",
            "self_play": False, "training_input": False, "training_performed": False, "optimizer_steps": 0,
            "ordinary_outcome_claim": False, "failure_finalization_seconds": 30}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-v7-neural-dialog-context-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "not_started": len(schedule), "frozen_before": frozen,
              "self_play": False, "training_input": False, "training_performed": False, "optimizer_steps": 0,
              "ordinary_outcome_claim": False}

    def check_identity():
        current = implementation_identity()
        current["experiments/explorer/create_dialog_scenarios.py"] = digest(ROOT / "experiments/explorer/create_dialog_scenarios.py")
        base.ensure(current == implementation and runtime_identity(args.binary.parent) == native
                    and digest(args.binary) == plan["binary_sha256"] and digest(args.output / "plan.json") == plan_hash
                    and checkpoint_identity(loaded, args.checkpoint) == frozen, "Diagnostic source/runtime/checkpoint changed")
        for (fixture, color), expected in fixtures.items():
            base.ensure(fixture_identity(fixture, color, args.maps) == expected, "Diagnostic map changed")

    for row in schedule:
        case, sessions = None, None
        attempt = {**row, "status": "running"}
        report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        try:
            with wall_limit(settings.case_seconds):
                check_identity()
                args.player, args.map = row["player"], Path(fixtures[(row["fixture"], row["player"])]["path"])
                case = DialogRun(row["family"] + "-" + args.player, args, environment, c, auditor, row["engine_seed"])
                case.start()
                sessions = tuple(ContextualSession(loaded.model, "1", player, loaded.checkpoint,
                    "scripted-context-operator", schema=c.DIALOGS_SCHEMA, seed=row["policy_seeds"][player]) for player in (0, 1))
                for player, session in enumerate(sessions): session.diagnostic_seed = row["policy_seeds"][player]
                targets = case.targets(OBJECT_KINDS[row["fixture"]])
                observed = []
                if targets:
                    case.visit(targets[0])
                    if case.family() is not None:
                        observed = sample_dialog_chain(case, sessions, settings.max_choices)
                case.finish()
                manifest_path, _ = case.persist_and_audit()
                base.ensure(case.end.status == "truncated", "Expected an unscored diagnostic cutoff")
                consume_histories(case, sessions)
                evidence_path = case.directory / "neural-context-evidence.json"
                evidence = choice_evidence(sessions)
                write_json(evidence_path, evidence)
                reloaded = load_policy_checkpoint(args.checkpoint, interface=c.DIALOGS_ENVIRONMENT)
                reconstruction = replay_choices(json.loads(evidence_path.read_text()), reloaded.model)
                base.ensure(all(evidence["players"][p]["history"] == case.history(color) for p, color in enumerate(base.COLORS)),
                            "Policy evidence omitted or changed actual own history")
                labels = [{"control": "neural" if item["label"] == NEURAL_LABEL else "scripted_diagnostic", **item} for item in case.attempts]
                write_json(case.directory / "diagnostic-actions.json", labels)
                branch = coverage(row["family"], observed)
                drain = validate_drain(json.loads(manifest_path.read_text()), observed)
                mechanics = case.summary()
                base.ensure(mechanics["probe_status"] != "failed", "Native diagnostic mechanics checks failed")
                status = "unverified" if "unverified" in (branch["status"], mechanics["probe_status"]) else "passed"
                attempt.update(status=status, coverage=branch, sampled_dialogs=observed, cutoff_deferral=drain,
                    reconstruction=reconstruction, mechanics=mechanics, evidence_sha256=digest(evidence_path),
                    actions_sha256=digest(case.directory / "diagnostic-actions.json"), manifest_sha256=digest(manifest_path),
                    episode_directory=str(case.env.episode_dir), weights_unchanged=True)
                check_identity()
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if case is not None:
                retain_failed_evidence(case.env, SimpleNamespace(sessions=sessions) if sessions else None, attempt, error)
                if case.env.result is not None:
                    case.end = case.state = case.env.result
                    if not case.audit_attempted:
                        try:
                            with wall_limit(30): case.persist_and_audit()
                        except Exception as secondary: attempt["failure_history_audit_error"] = f"{type(secondary).__name__}: {secondary}"
                attempt["mechanics"] = case.summary(attempt["error"])
        finally:
            if case is not None:
                try: case.env.close()
                except Exception as error: attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            try: check_identity()
            except Exception as error: attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
            report["not_started"] = len(schedule) - len(report["cases"])
            write_json(args.output / "report.json", report)
        if attempt["status"] == "failed": break
    report["status"] = "failed" if report["not_started"] or any(row["status"] == "failed" for row in report["cases"]) else (
        "unverified" if any(row["status"] == "unverified" for row in report["cases"]) else "passed")
    report["frozen_after"] = checkpoint_identity(loaded, args.checkpoint)
    write_json(args.output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maps", type=Path, default=MAPS)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--player", choices=(*base.COLORS, "both"), default="both")
    for name, default in asdict(Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float if name == "case_seconds" else int, default=default)
    report = run(parser.parse_args(argv))
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "optimizer_steps": 0}))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
