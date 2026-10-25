#!/usr/bin/env python3
"""Bounded real neutral-battle terminal diagnostic; no training or baseline claim.

The fixture coordinates and hidden stack sizes never select an action. The
explicit evaluator uses only current map views. A natural battle loss alone
does not prove the neutral request origin: that requires the native receipt.
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
from experiments.headless.contextual_evidence import terminal_barrier, terminal_capture_close
from experiments.learner import contextual as contract
from validation.probes import validate_contextual_native as probe
from validation.probes.create_neutral_terminal_fixture import PLAYERS, fixture_input, sha


def diagnostic_action(frame, *, losing_player):
    """Fixture evaluator only: a currently visible neutral, otherwise visible gold.

    Distance breaks ties among current displayed objects, never unexplored tiles
    or fixture metadata. This function is not installed in any policy/collector.
    """
    obs = frame.observation
    probe.ensure(obs.context.kind == "map", "Neutral diagnostic requires current map view")
    if frame.player != losing_player:
        return contract.Action("end_turn")
    probe.ensure(len(obs.heroes) == 1, "Expected exactly one owned hero before terminal battle")
    hero = obs.heroes[0]
    if hero.movement == 0:
        return contract.Action("end_turn")
    targets = [item for item in obs.objects if item.kind == 54 and item.owner == "neutral"]
    if not targets:
        targets = [item for item in obs.objects if item.kind == 79 and item.subtype == 6]
    if not targets:
        return None  # No hidden-coordinate fallback or unbounded exploration.
    target = min(targets, key=lambda item: (
        abs(item.position[0] - hero.position[0]) + abs(item.position[1] - hero.position[1]), item.handle))
    return contract.Action("move_to", hero.handle, *target.position[:2])


def classify_neutral_boundary(histories, *, player, outcomes, request):
    """Coverage classification of already audited records, not independent replay."""
    frames = tuple(contract.decode_frame(record) for record in histories[player])
    terminal = [frame for frame in frames if frame.kind == "terminal"]
    battle_starts = [frame for frame in frames if frame.event is not None and frame.event.kind == "battle_start"]
    defeats = [frame for frame in frames if frame.event is not None
               and frame.event.kind == "battle_complete" and frame.event.won is False]
    lost = [frame for frame in frames if frame.event is not None and frame.event.kind == "hero_lost"]
    expected = {color: "loss" if color == player else "win" for color in PLAYERS}
    final_views = [frame for frame in frames if frame.observation is not None]
    natural_battle = bool(outcomes == expected and len(terminal) == 1 and terminal[0].day < 7
                         and battle_starts and defeats and lost and final_views
                         and not final_views[-1].observation.heroes and not final_views[-1].observation.towns)
    # Exact neutral identity/type is authenticated by terminal_barrier + native
    # emission; independent native replay remains a separate required report.
    neutral = natural_battle and request.get("player") == -1 and request.get("type") == 198
    return {"natural_battle_loss": natural_battle, "neutral_request_origin": neutral,
            "request": request, "terminal_day": terminal[0].day if terminal else None,
            "battle_start_sequences": [frame.seq for frame in battle_starts],
            "battle_defeat_sequences": [frame.seq for frame in defeats],
            "hero_lost_sequences": [frame.seq for frame in lost],
            "independent_native_replay_verified": False}


def neutral_case(run, player):
    # Both agents must have an actual consumed decision before the scorer can
    # accept a self-play episode. Otherwise a day-one loss can legitimately be
    # classified agent_never_acted despite an authentic native terminal receipt.
    acted = set()
    while run.live() and len(acted) < 2:
        frame = run.inspect_request()
        acted.add(frame.player)
        run.call(contract.Action("end_turn"), "diagnostic_both_players_initial_turn")
    while run.live():
        frame = run.inspect_request()
        action = diagnostic_action(frame, losing_player=PLAYERS.index(player))
        if action is None:
            run.check("visible_neutral_approach", "unverified", reason="No current visible guard or gold; no hidden fallback")
            run.finish()
            break
        run.call(action, "diagnostic_currently_visible_neutral_approach")
    if run.end is None or run.end.status != "completed":
        run.check("neutral_battle_natural_terminal", "failed" if run.end and run.end.status == "failed" else "unverified",
                  reason="A verified natural ending was not reached within the declared bounds")
        return None
    proof = terminal_barrier(run.env.episode_dir, episode=run.end.manifest["episode"],
                             interface=run.interface, outcomes=[run.end.outcomes[color] for color in PLAYERS])
    terminal_capture_close(run.env.episode_dir, proof)
    evidence = classify_neutral_boundary({color: run.history(color) for color in PLAYERS}, player=player,
                                         outcomes=run.end.outcomes, request=proof["server"]["request"])
    run.check("native_last_hero_battle_loss", "passed" if evidence["natural_battle_loss"] else "failed", evidence=evidence)
    run.check("authenticated_neutral_make_action_terminal", "passed" if evidence["neutral_request_origin"] else "unverified",
              request=evidence["request"], required_player=-1, required_type=198)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=98501)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--case-seconds", type=float, default=60)
    parser.add_argument("--watchdog", type=float, default=30)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (not 1 <= args.seed <= 2147483647 or not 1 <= args.max_decisions <= 100
            or any(not math.isfinite(value) or not 0 < value <= 120
                   for value in (args.case_seconds, args.watchdog, args.startup_seconds))):
        parser.error("Use a positive int32 seed, 1..100 decisions and finite 0..120 second limits")
    for name in ("fixture", "binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output exists; preserve every attempt")
    player, args.map, fixture = fixture_input(args.fixture)
    args.source, args.max_days = ROOT, 4
    env = probe.load("heroes_neutral_probe_env", ROOT / "experiments/headless/env.py")
    auditor = probe.load("heroes_neutral_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    probe.write(args.output / "plan.json", {
        "schema": "heroes-contextual-neutral-terminal-validation-v1", "player": player,
        "fixture_manifest": str(args.fixture), "fixture_manifest_sha256": sha(args.fixture.read_bytes()),
        "fixture": fixture["fixture"], "engine_seed": args.seed, "binary_sha256": probe.sha(args.binary),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "probe_sha256": probe.sha(Path(__file__)),
        "bounds": {name: getattr(args, name) for name in ("max_days", "max_decisions", "case_seconds", "watchdog", "startup_seconds")},
        "training_performed": False, "random_baseline_outcome_claim": False,
        "scope": "Isolated real native battle fixture; current-view mechanical evaluator only"})
    run = probe.CaseRun(f"neutral-terminal-{player}", args, env, contract, auditor, args.seed)
    error, evidence = None, None
    try:
        run.start()
        evidence = neutral_case(run, player)
        run.persist_and_audit()
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
    report.update(scope="Fixture-only native neutral battle validation; not training or random-baseline evidence",
                  neutral_boundary=evidence, fixture_manifest=str(args.fixture), random_baseline_outcome_claim=False,
                  independent_native_replay_verified=False)
    probe.write(args.output / "report.json", report)
    print(json.dumps({"probe_status": report["probe_status"], "episode_status": report["episode_status"],
                      "report": str(args.output / "report.json")}, indent=2))
    return 0 if report["probe_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
