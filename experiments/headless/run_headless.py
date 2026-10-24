#!/usr/bin/env python3
"""Run a predetermined series of headless self-play episodes on the real engine.

Both sides are seeded uniform-random agents over the same player-visible action
interface. The seed schedule is a pure function of the arguments and is written
to the summary before the first episode, so a series can be reproduced and an
interrupted series can be compared against what was planned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


env_module = _load("heroes_headless_env", HERE / "env.py")
agents_module = _load("heroes_headless_agents", HERE / "agents.py")
scoring = env_module.scoring

DEFAULT_MAP = ROOT / "experiments/explorer/maps/random-baseline-v1.vmap"
DEFAULT_DATA = Path.home() / ("Library/Application Support/vcmi" if sys.platform == "darwin" else ".local/share/vcmi")


def host_description() -> dict:
    """Hardware and OS facts for the benchmark record; never derived from guesses."""
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count()}
    try:
        if sys.platform == "darwin":
            def sysctl(key):
                return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            info.update(cpu=sysctl("machdep.cpu.brand_string"), memory_bytes=int(sysctl("hw.memsize")),
                        performance_cores=int(sysctl("hw.perflevel0.physicalcpu")),
                        efficiency_cores=int(sysctl("hw.perflevel1.physicalcpu")),
                        os_version=subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip())
        else:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            names = [line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")]
            meminfo = Path("/proc/meminfo").read_text().split()
            info.update(cpu=names[0] if names else None, memory_bytes=int(meminfo[meminfo.index("MemTotal:") + 1]) * 1024)
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        pass
    return info


def build_type(binary: Path) -> str | None:
    cache = binary.parent.parent / "CMakeCache.txt"
    try:
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:"):
                return line.split("=", 1)[1]
    except OSError:
        pass
    return None


def schedule(args) -> list[dict]:
    """The frozen per-episode seed schedule."""
    return [{"episode": args.first_episode + index,
             "engine_seed": args.seed + index * args.seed_stride,
             "red_policy_seed": args.red_policy_seed + index,
             "blue_policy_seed": args.blue_policy_seed + index}
            for index in range(args.episodes)]


def child_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def episode_row(end, plan: dict, wall: float, logs_bytes: int) -> dict:
    manifest = end.manifest
    players = manifest["players"]
    row = {"episode": plan["episode"], "engine_seed": plan["engine_seed"], "status": end.status,
           "termination_reason": end.termination_reason, "outcomes": end.outcomes, "rewards": end.rewards,
           "wall_seconds": round(wall, 6), "env_steps": manifest["env_steps"],
           "startup_seconds": manifest["timing"]["startup_seconds"],
           "shutdown_seconds": manifest["timing"]["shutdown_seconds"],
           "engine_wait_seconds": manifest["timing"]["engine_wait_seconds"],
           "agent_seconds": manifest["timing"]["agent_seconds"],
           "step_latency_seconds": manifest["step_latency_seconds"], "log_bytes": logs_bytes,
           "failure_detail": manifest.get("failure_detail")}
    for color, player in players.items():
        coverage = player["coverage"]
        row[color] = {"actions_sent": player["actions_sent"], "resolved": player["actions_accepted"] + player["actions_rejected"],
                      "accepted": player["actions_accepted"], "rejected": player["actions_rejected"],
                      "decisions": player["decisions_sent"], "turns": player["turns_completed"],
                      "last_day": player["last_day"], "battles": player["battles_started"],
                      "end_turns": coverage["end_turns"], "moves": coverage["successful_moves"],
                      "town_only_turns": coverage["town_only_turns"], "dialogs": coverage["dialogs"],
                      "coverage": coverage,
                      "visible_updates": player["visible_updates"], "handles": player["handles_assigned"]}
    return row


def aggregate(rows: list[dict], wall_seconds: float) -> dict:
    statuses = {}
    for row in rows:
        key = f"{row['status']}:{row['termination_reason']}"
        statuses[key] = statuses.get(key, 0) + 1
    resolved = sum(row[c]["resolved"] for row in rows for c in ("red", "blue"))
    accepted = sum(row[c]["accepted"] for row in rows for c in ("red", "blue"))
    rejected = sum(row[c]["rejected"] for row in rows for c in ("red", "blue"))
    stepping = sum(row["engine_wait_seconds"] + row["agent_seconds"] for row in rows)
    natural = sum(1 for row in rows if row["status"] == "completed")
    totals = {key: sum(row[c][key] for row in rows for c in ("red", "blue"))
              for key in ("decisions", "turns", "battles", "end_turns", "moves", "town_only_turns", "dialogs")}
    coverage = {key: sum(row[c]["coverage"][key] for row in rows for c in ("red", "blue"))
                for key in rows[0]["red"]["coverage"]} if rows else {}
    return {"episodes": len(rows), "status_counts": statuses, "natural_completions": natural,
            "truncations": sum(1 for row in rows if row["status"] == "truncated"),
            "failures": sum(1 for row in rows if row["status"] == "failed"),
            "resolved_actions": resolved, "accepted_actions": accepted, "rejected_attempts": rejected, **totals,
            "coverage_totals": coverage, "wall_seconds": round(wall_seconds, 6),
            "end_to_end_resolved_actions_per_second": round(resolved / wall_seconds, 3) if wall_seconds else None,
            "end_to_end_accepted_actions_per_second": round(accepted / wall_seconds, 3) if wall_seconds else None,
            "end_to_end_rejected_attempts_per_second": round(rejected / wall_seconds, 3) if wall_seconds else None,
            "steady_state_resolved_actions_per_second": round(resolved / stepping, 3) if stepping else None,
            "natural_games_per_second": round(natural / wall_seconds, 6) if wall_seconds else None,
            "mean_episode_seconds": round(sum(r["wall_seconds"] for r in rows) / len(rows), 6) if rows else None,
            "mean_days_per_episode": round(sum(max(r["red"]["last_day"], r["blue"]["last_day"]) for r in rows) / len(rows), 3) if rows else None,
            "mean_resolved_actions_per_episode": round(resolved / len(rows), 3) if rows else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New series directory")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Directory that contains the game's Data folder")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--first-episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=40001, help="Engine seed of the first episode")
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--red-policy-seed", type=int, default=1)
    parser.add_argument("--blue-policy-seed", type=int, default=100001)
    parser.add_argument("--max-days", type=int, default=40)
    parser.add_argument("--max-actions", type=int, default=2000, help="Adventure actions per player")
    parser.add_argument("--watchdog", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="Collection window in seconds; the episode in progress at its end is truncated")
    parser.add_argument("--log-level", default="info", choices=["trace", "debug", "info"])
    parser.add_argument("--profile", action="store_true", help="Ask ExplorerRL for per-message timing fields")
    parser.add_argument("--ledger", default="full", choices=["full", "none"],
                        help="full: write each side's bridge stream (internal evidence with engine IDs)")
    parser.add_argument("--skip-check", action="store_true", help="Skip tools/prepare_vcmi.py --check")
    args = parser.parse_args()
    if args.episodes < 1 or args.first_episode < 1 or args.seed < 1 or args.seed_stride < 1:
        parser.error("Use positive episode counts, episode numbers, seeds and stride")
    if args.seed + (args.episodes - 1) * args.seed_stride > 2147483647:
        parser.error("The seed schedule exceeds the signed 32-bit range")

    launcher = env_module.VcmiLauncher(args.binary, args.data, args.map, args.log_level)
    try:
        launcher.preflight()
        if not args.skip_check:
            subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], check=True)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Preflight failed: {error}\n")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = schedule(args)
    limits = env_module.Limits(max_days=args.max_days, max_actions=args.max_actions, watchdog_seconds=args.watchdog)
    summary = {
        "schema_version": 1, "kind": "headless-self-play-series", "created_at": datetime.now(timezone.utc).isoformat(),
        "host": host_description(), "build_type": build_type(launcher.binary),
        "identity": {**launcher.identity(), "engine": env_module._git_state(ROOT / "vcmi"),
                     "project": env_module._git_state(ROOT), "engine_recipe": scoring.engine_recipe_digests(ROOT)},
        "workload": {"map": str(args.map), "agents": "independent seeded uniform-random agents, both colors",
                     "limits": {"max_days": args.max_days, "max_actions_per_player": args.max_actions,
                                "watchdog_seconds": args.watchdog, "collection_window_seconds": args.duration},
                     "log_level": args.log_level, "ledger": args.ledger, "profile": args.profile},
        "schedule": plan, "episodes": [], "aggregate": None, "interrupted": None,
    }
    summary_path = output / "summary.json"
    scoring.write_json(summary_path, summary)
    env = env_module.HeadlessSelfPlayEnv(launcher, output / "episodes", limits, ledger=args.ledger == "full",
                                         profile=args.profile)
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    try:
        for entry in plan:
            if deadline and time.monotonic() >= deadline:
                break
            players = {"red": agents_module.UniformRandomAgent(entry["red_policy_seed"]),
                       "blue": agents_module.UniformRandomAgent(entry["blue_policy_seed"])}
            env.agent_descriptions = {color: agent.description for color, agent in players.items()}
            episode_started = time.monotonic()
            step = env.reset(entry["episode"], entry["engine_seed"])
            while not step.done:
                if deadline and time.monotonic() >= deadline:
                    step = env.truncate("collection_window")
                    break
                step = env.step(players[step.color].act(step))
            wall = time.monotonic() - episode_started
            row = episode_row(step, entry, wall, directory_bytes(env.episode_dir / "logs"))
            summary["episodes"].append(row)
            summary["aggregate"] = aggregate(summary["episodes"], time.monotonic() - started)
            summary["aggregate"]["child_peak_rss_bytes"] = child_peak_rss_bytes()
            scoring.write_json(summary_path, summary)
            print(json.dumps({"episode": row["episode"], "status": row["status"],
                              "termination": row["termination_reason"], "rewards": row["rewards"],
                              "resolved": row["red"]["resolved"] + row["blue"]["resolved"],
                              "seconds": row["wall_seconds"]}), flush=True)
    except KeyboardInterrupt:
        summary["interrupted"] = "keyboard_interrupt"
    finally:
        env.close()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary["aggregate"] = aggregate(summary["episodes"], time.monotonic() - started)
        summary["aggregate"].update(child_peak_rss_bytes=child_peak_rss_bytes(),
                                    child_cpu_seconds=round(usage.ru_utime + usage.ru_stime, 3),
                                    harness_cpu_seconds=round(sum(resource.getrusage(resource.RUSAGE_SELF)[:2]), 3))
        scoring.write_json(summary_path, summary)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
