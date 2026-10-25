#!/usr/bin/env python3
"""Controlled town-capture mechanics on the unchanged baseline; never training.

The defender explicitly vacates its town and then ends turns. The attacker uses
only its current observation to select a displayed enemy town, or a displayed
empty tile near the public map centre to reveal it. These declared evaluator
controls test reachability; they are not a policy, demonstrations, ordinary-game
outcome-frequency evidence, or permission to change the map and game rules.
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
from validation.probes import validate_contextual_native as base
from validation.probes.create_boundary_fixture import source_contents


def empty_tiles(observation):
    """Current displayed plain tiles only; no hidden map or native route oracle."""
    occupied = {item.position for item in (*observation.objects, *observation.heroes, *observation.towns)}
    return [tile.position for tile in observation.tiles
            if not tile.blocked and not tile.visitable and tile.position not in occupied]


def centre_distance(position, size):
    # Doubled coordinates preserve exact mirrored geometry for even map sizes.
    return sum((2 * position[i] - (size[i] - 1)) ** 2 for i in (0, 1))


def defender_destination(observation):
    """One local, observed relocation; choose the outward side of its own town."""
    if len(observation.heroes) != 1 or len(observation.towns) != 1:
        return None
    origin = observation.heroes[0].position
    choices = [pos for pos in empty_tiles(observation)
               if 1 <= max(abs(pos[i] - origin[i]) for i in (0, 1)) <= 2]
    return min(choices, key=lambda pos: (-centre_distance(pos, observation.size), pos[1], pos[0])) if choices else None


def attacker_destination(observation, attempted):
    """A currently visible enemy town takes precedence over a plain view tile."""
    towns = [item.position for item in observation.objects if item.kind == 98 and item.owner == "opponent"]
    if towns:
        return min(towns), "currently_observed_enemy_town"
    choices = [pos for pos in empty_tiles(observation) if pos not in attempted]
    if not choices:
        return None, "no_untried_displayed_empty_tile"
    return min(choices, key=lambda pos: (centre_distance(pos, observation.size), pos[1], pos[0])), "displayed_empty_centre_tile"


class CaptureRun(base.CaseRun):
    def expected_offers(self, frame):
        return contract.Offers.from_observation(frame.observation, schema=contract.profile_for(self.interface).action_schema)

    def action(self, kind, **parameters):
        return contract.Action(kind, schema=contract.profile_for(self.interface).action_schema, **parameters)


def capture_case(run, attacker):
    defender = "blue" if attacker == "red" else "red"
    relocated = False
    relocation_attempted = False
    captured = False
    defender_townless_alive = []
    attempted = set()
    while run.live():
        frame = run.inspect_request()
        obs = frame.observation
        color = run.state.color
        base.ensure(obs.context.kind == "map", "Capture controls unexpectedly left the map")
        if color == defender and not obs.towns and obs.heroes:
            defender_townless_alive.append({"day": frame.day, "seq": frame.seq, "hero_count": len(obs.heroes)})
        if color == attacker and len(obs.towns) == 2 and not captured:
            captured = True
            run.check("observed_native_town_capture", "passed", player=color, day=frame.day,
                      decision_seq=frame.seq, own_town_count=2)
        if color == defender and not relocated and not relocation_attempted:
            target = defender_destination(obs)
            if target is None:
                run.check("defender_relocation", "unverified", reason="No displayed local empty target")
                break
            relocation_attempted = True
            _, additions, result = run.call(run.action("move_to", subject=obs.heroes[0].handle,
                                                       x=target[0], y=target[1]), "explicit_observed_defender_relocation")
            relocated = result is not None and result.status == "reached" and result.destination_reached
            base.ensure(not any(record.event and record.event.kind in ("battle_start", "hero_lost") for record in additions),
                        "Defender relocation unexpectedly encountered a battle")
            run.check("defender_relocation", "passed" if relocated else "unverified", observed_target=list(target),
                      result_status=result.status if result else None)
            if not relocated:
                break
        elif color == attacker and relocated and not captured and obs.heroes and obs.heroes[0].movement:
            target, reason = attacker_destination(obs, attempted)
            if target is None:
                run.check("visible_capture_target", "unverified", reason=reason)
                break
            attempted.add(target)
            run.call(run.action("move_to", subject=obs.heroes[0].handle, x=target[0], y=target[1]), reason)
        else:
            run.call(run.action("end_turn"), "explicit_turn_for_relocation_or_standard_seven_day_rule")
    run.finish()
    base.ensure(set(run.initial) == {"red", "blue"}, "Both actual initial player decisions are required")
    base.ensure(all(len(frame.observation.heroes) == len(frame.observation.towns) == 1
                    for frame in run.initial.values()), "Initial owned counts differ from the unchanged baseline")
    base.ensure(run.end.status != "failed", "Native capture attempt failed technically")
    if not captured:
        run.check("observed_native_town_capture", "unverified", reason="Capture condition not reached within bounds")
    if run.end.status != "completed":
        run.check("standard_capture_then_seven_day_ending", "unverified", reason="No verified natural ending within bounds")
        return
    base.ensure(captured and defender_townless_alive, "Natural ending did not follow the declared capture and surviving defender")
    expected = {attacker: "win", defender: "loss"}
    base.ensure(run.end.outcomes == expected and run.end.rewards == {attacker: 1, defender: -1},
                "Natural ending lacks expected complementary outcomes/rewards")
    frames = [contract.decode_frame(raw) for raw in run.history(defender)]
    terminal = next(record for record in frames if record.kind == "terminal")
    before = [record for record in frames if record.seq < terminal.seq and record.observation is not None]
    base.ensure(before and len(before[-1].observation.heroes) == 1 and not before[-1].observation.towns,
                "Defender did not retain its hero until the standard terminal")
    base.ensure(not any(record.event and record.event.kind == "hero_lost" for record in frames if record.seq < terminal.seq),
                "A preterminal hero loss invalidates seven-day coverage")
    notices = [token.text_id for record in frames if record.event and record.event.notice
               for token in record.event.notice.text]
    base.ensure("heroes.baseline.seven_days_without_town" in notices, "Native seven-day notice was not observed")
    from experiments.headless.contextual_evidence import terminal_barrier, terminal_capture_close
    barrier = terminal_barrier(run.env.episode_dir, episode=terminal.episode,
                               interface=run.interface, outcomes=[expected[color] for color in base.COLORS])
    close = terminal_capture_close(run.env.episode_dir, barrier)
    run.check("standard_capture_then_seven_day_ending", "passed", outcomes=expected,
              surviving_defender_decisions=defender_townless_alive,
              terminal_day=terminal.day, terminal_seq=terminal.seq,
              terminal_request=barrier["client"]["request"], terminal_capture_close=close)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=98201)
    parser.add_argument("--interface", choices=(contract.TOWN_ENVIRONMENT, contract.ARMIES_ENVIRONMENT),
                        default=contract.TOWN_ENVIRONMENT)
    parser.add_argument("--attacker", choices=("red", "blue", "both"), default="both")
    parser.add_argument("--max-days", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--case-seconds", type=float, default=60)
    parser.add_argument("--watchdog", type=float, default=20)
    parser.add_argument("--startup-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if (not 1 <= args.seed <= 2147483647 or not 1 <= args.max_days <= 30 or not 1 <= args.max_decisions <= 200
            or any(not math.isfinite(value) or not 0 < value <= 60
                   for value in (args.case_seconds, args.watchdog, args.startup_seconds))):
        parser.error("Require bounded positive seed, days, decisions and finite time limits")
    for name in ("binary", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        parser.error("Output already exists; preserve every attempt")
    args.map = ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"
    source_contents(args.map)
    env = base.load("heroes_capture_probe_env", ROOT / "experiments/headless/env.py")
    auditor = base.load("heroes_capture_probe_auditor", ROOT / "validation/probes/audit_contextual_histories.py")
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    colors = ("red", "blue") if args.attacker == "both" else (args.attacker,)
    base.write(args.output / "plan.json", {
        "schema": "heroes-town-capture-mechanics-v1", "attacker_colors": colors, "engine_seed": args.seed,
        "map": str(args.map), "map_sha256": base.sha(args.map), "binary_sha256": base.sha(args.binary),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "probe_sha256": base.sha(Path(__file__)), "interface": args.interface,
        "bounds": {name: getattr(args, name) for name in ("max_days", "max_decisions", "case_seconds", "watchdog")},
        "training_input": False, "training_performed": False, "ordinary_outcome_frequency_claim": False})
    reports = []
    for color in colors:
        run = CaptureRun("capture-" + color, args, env, contract, auditor, args.seed)
        error = None
        try:
            run.start()
            capture_case(run, color)
            run.persist_and_audit()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.check("capture_execution", "failed", error=error)
            try:
                run.finish()
                if run.end is not None and not run.audit_attempted:
                    run.persist_and_audit()
            except Exception as cleanup:
                run.check("failure_evidence_finalization", "failed", error=f"{type(cleanup).__name__}: {cleanup}")
        finally:
            run.env.close()
        report = run.summary(error)
        report.update(attacker=color, training_input=False, ordinary_outcome_frequency_claim=False)
        reports.append(report)
        base.write(run.directory / "report.json", report)
    base.write(args.output / "report.json", {"cases": reports, "training_performed": False,
                                            "ordinary_outcome_frequency_claim": False})
    print(json.dumps({"report": str(args.output / "report.json"),
                      "statuses": [row["probe_status"] for row in reports]}, indent=2))
    return 0 if all(row["probe_status"] == "passed" for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
