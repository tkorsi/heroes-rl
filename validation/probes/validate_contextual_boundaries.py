#!/usr/bin/env python3
"""Run one bounded real-VCMI boundary fixture, with independent history audit.

Generate input with create_boundary_fixture.py, then pass its fixture.json.
These explicit mechanical evaluator actions are excluded from training and
random-policy outcome coverage. No engine, policy, or terminal is emulated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.learner import contextual as contract
from validation.probes import validate_contextual_native as probe
from validation.probes.create_boundary_fixture import SCHEMAS, decode, sha, verify_fixture


def fixture_input(path):
    """Authenticate declared map bytes and recheck their exact field isolation."""
    path = Path(path).resolve()
    manifest = decode(path.read_bytes())
    kinds = [kind for kind, schema in SCHEMAS.items() if manifest.get("schema") == schema]
    if len(kinds) != 1 or manifest.get("kind", kinds[0]) != kinds[0]:
        raise ValueError("Expected a supported boundary fixture manifest")
    kind = kinds[0]
    source, game_map = Path(manifest["source"]["path"]), Path(manifest["fixture"]["path"])
    if not source.is_absolute() or not game_map.is_absolute():
        raise ValueError("Fixture source and map paths must be absolute")
    if sha(source.read_bytes()) != manifest["source"]["sha256"] or sha(game_map.read_bytes()) != manifest["fixture"]["sha256"]:
        raise ValueError("Fixture/source hash disagrees with its manifest")
    verify_fixture(game_map, kind=kind, baseline=source)
    return kind, game_map, manifest


def natural_case(run):
    while run.live():
        frame = run.inspect_request()
        probe.ensure(frame.observation.context.kind == "map", "Townless fixture left map context")
        run.call(contract.Action("end_turn"), "explicit_native_turn_for_seven_day_rule")
    probe.ensure(run.end is not None and run.end.status == "completed", "Native townless ending was not verified")
    probe.ensure(set(run.initial) == {"red", "blue"}, "Both real initial player views are required")
    red, blue = run.initial["red"].observation, run.initial["blue"].observation
    probe.ensure(len(red.heroes) == 1 and not red.towns and len(blue.heroes) == 1 and len(blue.towns) == 1,
                 "Real initial views do not match the townless fixture")
    probe.ensure(run.end.outcomes == {"red": "loss", "blue": "win"}
                 and run.end.rewards == {"red": -1, "blue": 1}, "Unexpected native complementary outcomes/rewards")
    run.check("native_townless_natural_boundary", "passed", outcomes=run.end.outcomes, rewards=run.end.rewards)


def unsupported_case(run):
    first = run.inspect_request()
    owner = run.state.player
    targets = [obj for obj in first.observation.objects if obj.kind == 97 and obj.subtype == 0]
    probe.ensure(len(targets) == 1, "Exactly one Den must actually be visible to the acting player")
    target = targets[0].position  # observed target only; fixture metadata never supplies the action coordinate
    while run.live():
        frame = run.inspect_request()
        probe.ensure(frame.observation.context.kind == "map", "Unsupported fixture unexpectedly opened a supported window")
        if run.state.player != owner:
            action = contract.Action("end_turn")
        else:
            probe.ensure(frame.observation.heroes, "Evaluator's owned hero disappeared before the Den visit")
            hero = frame.observation.heroes[0]
            action = contract.Action("end_turn") if hero.movement == 0 else contract.Action("move_to", hero.handle, *target[:2])
        run.call(action, "real_observed_den_window_attempt")
    probe.ensure(run.end is not None and run.end.status == "failed", "Unsupported window did not fail closed")
    probe.ensure(run.end.termination_reason == "unsupported_interface", "Failure is not identified as unsupported interface")
    probe.ensure("unsupported_interaction: thieves guild" in (run.end.manifest.get("failure_detail") or ""),
                 "Failure did not identify the actual Den callback")
    probe.ensure(all(value is None for value in run.end.rewards.values())
                 and all(value is None for value in run.end.outcomes.values()), "Unsupported interaction was scored")
    probe.ensure(all(value is None for value in run.end.bootstrap.values()), "Unsupported interaction retained bootstrap")
    run.check("real_unsupported_den_boundary", "passed", termination_reason=run.end.termination_reason,
              observed_target=list(target))


def terminal_view_coverage(run):
    """Optional event coverage; its absence never downgrades a valid boundary."""
    coverage = {}
    for color in ("red", "blue"):
        frames = [contract.decode_frame(raw) for raw in run.history(color)]
        terminals = [frame for frame in frames if frame.kind == "terminal"]
        later = [frame for frame in frames if terminals and frame.seq > terminals[0].seq and frame.kind == "view"]
        coverage[color] = {"status": "observed" if later else "not_observed",
                           "terminal_seq": terminals[0].seq if terminals else None,
                           "postterminal_view_sequences": [frame.seq for frame in later],
                           "final_observed_owned_hero_count": len(later[-1].observation.heroes) if later else None,
                           "required_for_boundary_success": False}
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="fixture.json from create_boundary_fixture.py")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; all attempts retained")
    parser.add_argument("--seed", type=int, default=89001)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--case-seconds", type=float, default=30)
    parser.add_argument("--watchdog", type=float, default=20)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (not 1 <= args.seed <= 2147483647 or not 1 <= args.max_decisions <= 200
            or any(not math.isfinite(value) or not 0 < value <= 60
                   for value in (args.case_seconds, args.watchdog, args.startup_seconds))):
        parser.error("Use a signed-32-bit positive seed, 1..200 decisions and finite 0..60 second bounds")
    for name in ("fixture", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; never overwrite preserved native evidence")
    kind, args.map, fixture = fixture_input(args.fixture)
    args.source, args.max_days = ROOT, 10 if kind == "natural-townless" else 3
    env = probe.load("heroes_boundary_probe_env", ROOT / "experiments/headless/env.py")
    auditor = probe.load("heroes_boundary_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    probe.write(args.output / "plan.json", {
        "schema": "heroes-contextual-boundary-validation-v1", "kind": kind,
        "fixture_manifest": str(args.fixture), "fixture_manifest_sha256": sha(args.fixture.read_bytes()),
        "fixture": fixture["fixture"], "engine_seed": args.seed, "binary_sha256": probe.sha(args.binary),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "probe_sha256": probe.sha(Path(__file__)),
        "bounds": {name: getattr(args, name) for name in ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
        "training_performed": False, "random_baseline_outcome_claim": False,
        "scope": "Real native evaluator fixture; explicit mechanics actions are not training experience"})
    run = probe.CaseRun(kind, args, env, contract, auditor, args.seed)
    error = None
    coverage = None
    try:
        run.start()
        (natural_case if kind == "natural-townless" else unsupported_case)(run)
        run.persist_and_audit()
        coverage = terminal_view_coverage(run)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.check("fixture_execution", "failed", error=error)
        try:
            run.finish()
            if run.end is not None and not run.audit_attempted:
                run.persist_and_audit()
        except Exception as cleanup:
            run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup).__name__}: {cleanup}")
    finally:
        run.env.close()
    report = run.summary(error)
    report.update(scope="Fixture-only native boundary validation; no random-baseline or training claim",
                  postterminal_view_coverage=coverage, fixture_manifest=str(args.fixture),
                  random_baseline_outcome_claim=False)
    probe.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "report": str(args.output / "report.json")}, indent=2))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
