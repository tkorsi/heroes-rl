#!/usr/bin/env python3
"""Attribute a profiled headless series' time to startup, actions, battles,
observations, transport, logging, policy and reset.

Input is a ``run_headless.py --profile --ledger full`` series directory. The
native fields come from ExplorerRL's per-message profile (``perf``) in each
side's bridge ledger; the adapter fields come from each episode manifest.

Categories overlap in wall time: the two players' bridges and the engine's
threads run concurrently, and ``engine_wait`` already contains everything the
engine did while the adapter waited. They are reported as totals and per
decision cycle, not as shares that add up to 100%. The engine's own info log
is not timed natively; compare series with ``--ledger none`` and ``--replay
none`` for the evidence-writing cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BATTLE_EVENTS = {"battle_start", "battle_complete"}  # as ExplorerRL emits them


def _completed_battles(state) -> int | None:
    value = state.get("completed_battles") if isinstance(state, dict) else None
    return value if type(value) is int else None


def episode_costs(episode: Path) -> dict:
    manifest = json.loads((episode / "manifest.json").read_text())
    timing = manifest["timing"]
    native = dict.fromkeys(("visible_capture_us", "visible_captures", "observation_build_us", "observations",
                            "action_request_us", "actions", "battle_request_us", "battle_actions", "encode_us",
                            "send_us", "message_bytes", "messages"), 0)
    for color in ("red", "blue"):
        ledger = episode / "agents" / color / "bridge.jsonl"
        if not ledger.is_file():
            continue
        # A battle inside an action's request can hand its events to a decision
        # the engine raised meanwhile (a level-up after the fight, for example),
        # so every engine message since the side's last submitted action counts.
        battle_events = False
        for line in ledger.read_text().splitlines():
            entry = json.loads(line)
            message = entry["message"]
            if entry["direction"] == "policy":
                battle_events = battle_events and "action" not in message
                continue
            battle_events = battle_events or (message.get("type") in {"decision", "action_result"} and any(
                event.get("kind") in BATTLE_EVENTS for event in message.get("events", [])))
            if "perf" not in message:
                continue
            perf = message["perf"]
            native["visible_capture_us"] += perf.get("visible_capture_us", 0)
            native["visible_captures"] += perf.get("visible_captures", 0)
            native["encode_us"] += perf.get("previous_message_encode_us", 0)
            native["send_us"] += perf.get("previous_message_send_us", 0)
            native["message_bytes"] += perf.get("previous_message_bytes", 0)
            native["messages"] += 1
            if message["type"] == "observation":
                native["observation_build_us"] += perf.get("build_us", 0)
                native["observations"] += 1
            elif message["type"] == "action_result":
                # The bridge's own battle counter, where the states carry it, needs no events at all.
                before, after = (_completed_battles(message.get(key)) for key in ("before", "after"))
                battle = battle_events or (None not in (before, after) and after > before)
                native["battle_request_us" if battle else "action_request_us"] += perf.get("request_us", 0)
                native["battle_actions" if battle else "actions"] += 1
    players = manifest["players"].values()
    # The environment's wait for the server's replay log before it stops the
    # engine. With both terminals delivered it already lies in shutdown_seconds.
    tail = ((manifest.get("replay") or {}).get("tail") or {}).get("waited_seconds", 0.0)
    return {"episode": manifest["episode"], "status": manifest["status"],
            "decision_cycles": sum(p["actions_accepted"] + p["actions_rejected"] for p in players),
            "battles": sum(p["battles_started"] for p in players), "wall_seconds": timing["wall_seconds"],
            "startup_seconds": timing["startup_seconds"] or 0.0, "shutdown_seconds": timing["shutdown_seconds"] or 0.0,
            "replay_tail_seconds": tail if timing["shutdown_seconds"] is None else 0.0,
            "engine_wait_seconds": timing["engine_wait_seconds"], "agent_seconds": timing["agent_seconds"],
            "adapter_seconds": timing.get("adapter_seconds", {}), "native": native}


def breakdown(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    rows = {row["episode"]: row for row in summary["episodes"]}
    episodes = [episode_costs(path.parent) for path in sorted((run / "episodes").glob("episode-*/manifest.json"))]
    episodes = [e for e in episodes if e["episode"] in rows]
    cycles = sum(e["decision_cycles"] for e in episodes)
    totals = {key: sum(e[key] for e in episodes) for key in ("wall_seconds", "startup_seconds", "shutdown_seconds",
                                                               "replay_tail_seconds", "engine_wait_seconds",
                                                               "agent_seconds")}
    # The runner's own time between episodes: profile setup, launch and teardown outside the manifest window.
    totals["runner_reset_seconds"] = sum(rows[e["episode"]]["wall_seconds"] - e["wall_seconds"] for e in episodes)
    adapter = {}
    for e in episodes:
        for key, value in e["adapter_seconds"].items():
            adapter[key] = adapter.get(key, 0.0) + value
    native = {key: sum(e["native"][key] for e in episodes) for key in episodes[0]["native"]} if episodes else {}

    def per_cycle(seconds: float):
        return round(seconds * 1e6 / cycles, 1) if cycles else None

    categories = {
        "startup (launch to both bridges ready)": totals["startup_seconds"],
        "reset (last terminal to exit, replay tail wait, runner relaunch)":
            totals["shutdown_seconds"] + totals["replay_tail_seconds"] + totals["runner_reset_seconds"],
        "actions (engine request round trip, no battle)": native.get("action_request_us", 0) / 1e6,
        "battles (request round trips that contained a battle)": native.get("battle_request_us", 0) / 1e6,
        "observations (native capture and message build)": native.get("observation_build_us", 0) / 1e6
        + native.get("visible_capture_us", 0) / 1e6,
        "observations (adapter validation and history fold)": adapter.get("ingest", 0.0),
        "actions (adapter history fold of submitted actions)": adapter.get("act", 0.0),
        "transport (native encode and send)": (native.get("encode_us", 0) + native.get("send_us", 0)) / 1e6,
        "transport (adapter decode and response send)": adapter.get("decode", 0.0) + adapter.get("respond", 0.0),
        "logging (adapter bridge ledger)": adapter.get("ledger", 0.0),
        "policy (seeded uniform-random agents)": totals["agent_seconds"],
    }
    return {"schema_version": 1, "kind": "headless-cost-breakdown", "run": str(run),
            "workload": summary.get("workload"), "host": summary.get("host"), "episodes": len(episodes),
            "decision_cycles": cycles, "battles": sum(e["battles"] for e in episodes), "totals": totals,
            "native_totals": native, "adapter_totals": {k: round(v, 6) for k, v in adapter.items()},
            "categories_seconds": {k: round(v, 6) for k, v in categories.items()},
            "categories_microseconds_per_decision_cycle": {k: per_cycle(v) for k, v in categories.items()},
            "note": "Categories overlap (two bridges and engine threads run concurrently); they do not sum to wall time. "
                    "The engine info log is untimed; compare series with and without evidence writing."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="A run_headless.py --profile --ledger full series directory")
    parser.add_argument("--output", type=Path, help="Write the breakdown JSON here")
    args = parser.parse_args(argv)
    result = breakdown(args.run.expanduser().resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
