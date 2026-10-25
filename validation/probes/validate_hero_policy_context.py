"""Live Tavern policy diagnostics after explicitly scripted context preparation.

This is not self-play, ordinary-policy evaluation, a demonstration dataset or
training. The script builds a native Tavern and moves its visitor away. A fixed
random-initialized policy then samples its full unmodified distribution at at
most eight real Tavern decisions. Back navigation is followed by an explicitly
scripted reopening. If hiring precedes selection, the hired visitor departs
through explicit native controls before reopening. Missing branches stay gaps.
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

from experiments.headless import env as environment
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint
from experiments.learner.contextual_model import weights_sha256
from experiments.learner.contextual_rollout import BehaviorDecision, ContextualSession, DecisionEstimate, RecurrentPrefix
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, write_json
from validation.probes import audit_contextual_histories as auditor
from validation.probes import validate_contextual_native as base
from validation.probes import validate_town_native as town
from validation.probes.validate_hero_policy import hero_map_identity, retain_failed_evidence, wall_limit
from validation.probes.validate_hero_recruitment import HeroRun, exact_hire


@dataclass(frozen=True)
class Settings:
    engine_seed: int = 103101
    policy_seed: int = 103104
    max_choices: int = 8
    case_seconds: float = 90.0

    def __post_init__(self):
        for seed in (self.engine_seed, self.policy_seed):
            base.ensure(type(seed) is int and 1 <= seed < 2147483647, "Seeds must permit both predeclared colors")
        base.ensure(type(self.max_choices) is int and 1 <= self.max_choices <= 8, "Use at most eight neural choices")
        base.ensure(type(self.case_seconds) in (int, float) and math.isfinite(self.case_seconds)
                    and 0 < self.case_seconds <= 120, "Use a finite case wall limit of at most 120 seconds")

    def schedule(self):
        return [{"player": color, "engine_seed": self.engine_seed + player,
                 "policy_seed": self.policy_seed + player} for player, color in enumerate(base.COLORS)]


def checkpoint_identity(loaded, path):
    base.ensure(not loaded.model.training and all(not p.requires_grad and p.device.type == "cpu"
                for p in loaded.model.parameters()), "Diagnostic weights must be frozen on CPU")
    base.ensure(weights_sha256(loaded.model) == loaded.weights_sha256, "Diagnostic policy weights changed")
    return {"file_sha256": digest(path), "weights_sha256": loaded.weights_sha256,
            "checkpoint": loaded.checkpoint, "provenance": loaded.provenance}


def prepare_tavern(run):
    """Operator controls exercise only currently displayed native mechanics."""
    frame = run.inspect_request()
    base.ensure(len(frame.observation.heroes) == len(frame.observation.towns) == 1,
                "Expected the exact ordinary scenario's one initial hero and town")
    hero = frame.observation.heroes[0].handle
    run.open_town()
    hall = run.open_hall()
    choice = next((item for item in hall.buildings if ".tavern." in item.name_text_id), None)
    base.ensure(choice is not None and choice.status == "allowed", "Native Tavern is not buildable")
    detail = run.open_building(choice)
    before, additions, result = run.call(run.town_action("construct", selection=detail.ref),
                                        "explicit_prepare_construct_tavern")
    town.request_evidence(run, additions, result, "built")
    base.ensure(tuple(a - b for a, b in zip(before.observation.resources,
                run.inspect_request().observation.resources)) == detail.cost, "Tavern construction price changed")
    run.call(run.action("close"), "explicit_prepare_close_town")
    run.depart(hero, "explicit_prepare_visitor_departure")
    run.open_town()
    panel = run.open_tavern()
    base.ensure(panel.recruit_status == "allowed", "Prepared native Tavern cannot hire")


def choice_evidence(session):
    return {"schema": "heroes-v5-neural-tavern-context-evidence-v1", "training_input": False,
            "purpose": "diagnostic", "mixed_scripted_and_neural_history": True,
            "checkpoint": session.checkpoint, "policy_seed": session.diagnostic_seed,
            "model_config": session.model.config.to_dict(),
            "history": [c.encode_frame(frame) for frame in session.history.frames],
            "neural_decisions": [choice.to_dict() for choice in session._choices.values()],
            "estimates": [DecisionEstimate(seq, value, session._prefixes[seq]).to_dict()
                          for seq, value in session._values.items()]}


def replay_choices(raw, model):
    """Rebuild all own memory, sampling only the labelled neural decision IDs."""
    base.ensure(raw["training_input"] is False and raw["purpose"] == "diagnostic"
                and raw["mixed_scripted_and_neural_history"] is True, "Expected mixed diagnostic evidence")
    history = tuple(c.decode_frame(frame) for frame in raw["history"])
    base.ensure(history and all(frame.schema == c.HEROES_SCHEMA for frame in history), "Expected v5 history")
    base.ensure(raw["model_config"] == model.config.to_dict(), "Diagnostic model configuration changed")
    decisions = tuple(BehaviorDecision.from_dict(item, history[item["decision_seq"]])
                      for item in raw["neural_decisions"])
    by_seq = {item.decision_seq: item for item in decisions}
    base.ensure(len(by_seq) == len(decisions), "Duplicate neural choices")
    session = ContextualSession(model, history[0].episode, history[0].player, raw["checkpoint"],
                                "scripted-context-operator", schema=c.HEROES_SCHEMA, seed=raw["policy_seed"])
    for frame in history:
        session.consume((frame,))
        if frame.seq in by_seq:
            base.ensure(frame.kind == "decision" and frame.observation.context.kind == "town"
                        and frame.observation.context.panel == "tavern", "Neural choice escaped its Tavern context")
            base.ensure(session.choose(frame.seq) == by_seq[frame.seq],
                        "Neural action, value, full support, prefix or conditional probabilities changed")
    expected_estimates = tuple(DecisionEstimate(item["decision_seq"], item["value"], RecurrentPrefix(**item["prefix"]))
                              for item in raw["estimates"])
    actual_estimates = tuple(DecisionEstimate(seq, value, session._prefixes[seq])
                            for seq, value in session._values.items())
    base.ensure(actual_estimates == expected_estimates and len(session._choices) == len(decisions),
                "A saved own-decision value or neural action was not reconstructed")
    return {"passed": True, "actions_resampled": len(decisions), "values_recomputed": len(actual_estimates),
            "conditionals_recomputed": sum(len(item.conditionals) for item in decisions),
            "maximum_absolute_difference": 0, "complete_own_history": True, "private_memory_reset": True}


def branch_coverage(decisions):
    counts = {kind: sum(item.action.kind == kind for item in decisions) for kind in ("back", "select_hero", "hire_hero")}
    missing = [kind for kind in ("select_hero", "hire_hero") if not counts[kind]]
    return {"status": "unverified" if missing else "passed", "counts": counts, "missing": missing}


def sample_context(run, session, maximum):
    """No masking, action override, RNG reset or retry of an undesired draw."""
    color = run.args.player
    witnessed = set()
    for index in range(maximum):
        frame = run.inspect_request()
        base.ensure(run.state.color == color and frame.observation.context.kind == "town"
                    and frame.observation.context.panel == "tavern", "Neural sample requires the actual own Tavern")
        session.consume(run.history(color))
        choice = session.choose(frame.seq)
        # Persist the sampled action before sending it, including a failed send.
        write_json(run.directory / "neural-context-evidence.json", choice_evidence(session))
        before, additions, result = run.call(choice.action, "neural_full_distribution_tavern_choice")
        if choice.action.kind == "hire_hero":
            proof = exact_hire(before, run.inspect_request(), choice.action, additions, result)
            run.check("native_neural_hire_exact_effects", "passed", **proof)
        else:
            base.ensure(choice.action.kind in ("back", "select_hero") and result is not None and result.accepted,
                        "Unexpected Tavern action or unsuccessful native result")
        witnessed.add(choice.action.kind)
        if {"select_hero", "hire_hero"} <= witnessed or index + 1 == maximum:
            return
        if choice.action.kind == "hire_hero":
            run.call(run.action("close"), "explicit_prepare_close_after_neural_hire")
            run.depart(proof["new_hero"], "explicit_prepare_neurally_hired_visitor_departure")
            run.open_town()
            run.open_tavern()
        elif choice.action.kind == "back":
            run.open_tavern()  # explicitly scripted setup for the next unchanged policy draw


def run(args):
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    torch.set_num_threads(1)
    for name in ("binary", "data", "output", "map", "checkpoint"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    base.ensure(not args.output.exists(), "Output already exists; retain every attempt")
    scenario = hero_map_identity(args.map)
    environment.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True, timeout=60)
    loaded = load_policy_checkpoint(args.checkpoint, interface=c.HEROES_ENVIRONMENT)
    loaded.model.requires_grad_(False)
    frozen = checkpoint_identity(loaded, args.checkpoint)
    implementation = implementation_identity()
    implementation["experiments/explorer/create_hero_scenario.py"] = digest(ROOT / "experiments/explorer/create_hero_scenario.py")
    native = runtime_identity(args.binary.parent)
    args.output.mkdir(parents=True)
    args.interface, args.max_days, args.max_decisions = c.HEROES_ENVIRONMENT, 5, 64
    args.watchdog, args.startup_seconds = 20, 30
    plan = {"schema": "heroes-v5-neural-tavern-context-plan-v1", "settings": asdict(settings),
            "schedule": settings.schedule(), "checkpoint": frozen, "checkpoint_source": loaded.source_metadata,
            "implementation": implementation, "native_runtime": native, "map": scenario,
            "environment_bounds": {key: getattr(args, key) for key in ("max_days", "max_decisions", "watchdog", "startup_seconds")},
            "script_control": "Build Tavern, depart initial visitor, open Tavern; reopen after neural back; if hire precedes selection, explicitly depart that visitor and reopen without granting resources",
            "neural_control": "Full unchanged distribution; stop when both selection and hire witnessed or declared choice bound; never retry seeds",
            "history": "Complete own history including clearly labelled scripted actions; no opponent history",
            "self_play": False, "training_input": False, "training_performed": False, "optimizer_steps": 0,
            "ordinary_outcome_claim": False, "failure_finalization_seconds": 30}
    write_json(args.output / "plan.json", plan)
    plan_hash = digest(args.output / "plan.json")
    report = {"schema": "heroes-v5-neural-tavern-context-validation-v1", "plan_sha256": plan_hash,
              "status": "running", "cases": [], "self_play": False, "training_performed": False,
              "training_input": False, "optimizer_steps": 0, "ordinary_outcome_claim": False, "frozen_before": frozen}
    def check_identity():
        current = implementation_identity()
        current["experiments/explorer/create_hero_scenario.py"] = digest(ROOT / "experiments/explorer/create_hero_scenario.py")
        base.ensure(current == implementation and runtime_identity(args.binary.parent) == native
                    and hero_map_identity(args.map) == scenario and digest(args.output / "plan.json") == plan_hash
                    and checkpoint_identity(loaded, args.checkpoint) == frozen, "Diagnostic source/runtime/checkpoint changed")
    for row in settings.schedule():
        args.player = row["player"]
        attempt = {**row, "status": "running", "outcomes": dict.fromkeys(base.COLORS), "rewards": dict.fromkeys(base.COLORS)}
        report["cases"].append(attempt)
        write_json(args.output / "report.json", report)
        case, session = None, None
        try:
            with wall_limit(settings.case_seconds):
                check_identity()
                case = HeroRun("context-" + args.player, args, environment, c, auditor, row["engine_seed"])
                case.start()
                session = ContextualSession(loaded.model, "1", base.COLORS.index(args.player), loaded.checkpoint,
                                            "scripted-context-operator", schema=c.HEROES_SCHEMA, seed=row["policy_seed"])
                session.diagnostic_seed = row["policy_seed"]
                prepare_tavern(case)
                sample_context(case, session, settings.max_choices)
                case.finish()
                case.persist_and_audit()
                base.ensure(case.end.status == "truncated", "Expected an unscored diagnostic cutoff")
                session.consume(case.history(args.player))
                evidence_path = case.directory / "neural-context-evidence.json"
                write_json(evidence_path, choice_evidence(session))
                reloaded = load_policy_checkpoint(args.checkpoint, interface=c.HEROES_ENVIRONMENT)
                reconstruction = replay_choices(json.loads(evidence_path.read_text()), reloaded.model)
                base.ensure(json.loads(evidence_path.read_text())["history"] == case.history(args.player),
                            "Neural evidence omitted or changed the actual own history")
                labels = [{"control": "neural" if item["label"] == "neural_full_distribution_tavern_choice" else "scripted_diagnostic",
                           **item} for item in case.attempts]
                write_json(case.directory / "diagnostic-actions.json", labels)
                coverage = branch_coverage(tuple(session._choices.values()))
                attempt.update(status=coverage["status"], coverage=coverage, reconstruction=reconstruction,
                               mechanics=case.summary(), evidence_sha256=digest(evidence_path),
                               actions_sha256=digest(case.directory / "diagnostic-actions.json"),
                               episode_directory=str(case.env.episode_dir), weights_unchanged=True)
                check_identity()
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if case is not None:
                retain_failed_evidence(case.env, SimpleNamespace(sessions=(session,)) if session else None, attempt, error)
        finally:
            if case is not None:
                try: case.env.close()
                except Exception as error: attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            try: check_identity()
            except Exception as error: attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}")
            report["not_started"] = len(settings.schedule()) - len(report["cases"])
            report["status"] = "failed" if attempt["status"] == "failed" else "running"
            write_json(args.output / "report.json", report)
        if attempt["status"] == "failed": break
    if report["status"] != "failed":
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "unverified"
        report["frozen_after"] = checkpoint_identity(loaded, args.checkpoint)
    write_json(args.output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    for name, default in asdict(Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float if name == "case_seconds" else int, default=default)
    report = run(parser.parse_args(argv))
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "optimizer_steps": 0}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
