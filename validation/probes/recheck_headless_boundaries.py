#!/usr/bin/env python3
"""Controlled boundary probes; no engine process, real game, or training.

Run from any directory with --source pointing at the reviewed PR checkout and
--expected-head pinning its commit. Imports that checkout's production adapter
and existing tests/test_headless.py message builders, injects socket/process
failures in memory, and writes JSON to --output (boundary.json beside this script
by default). Nonzero exit means probe setup failed; finding statuses distinguish
bug reproduction from desired behavior passing. No learner import is needed.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import threading
import time

LOG = ("Experiment player 0 won. Terminal received.\n"
       "Experiment player 1 lost. Terminal received.\n"
       "Experiment self-play complete. Ending game.\n")


def load_reviewed_source(source):
    global env, history, observation, columns, obj, update, OWN
    fixture_path = source / "tests/test_headless.py"
    spec = importlib.util.spec_from_file_location("boundary_probe_fixtures", fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixtures
    spec.loader.exec_module(fixtures)
    env = fixtures.env_module
    history, observation, columns = fixtures.history, fixtures.observation, fixtures.columns
    obj, update, OWN = fixtures.obj, fixtures.update, fixtures.OWN


def action_consumption():
    player_history = history()
    hero = obj(1, 0, 0, owner=0)
    player_history.ingest(observation(0, columns(0, 2), [hero], [
        update(0, appeared=[hero], revealed=columns(0, 2), own=OWN)]))
    validator = env.BridgeValidator(0, "1")
    validator.ready = True
    validator.pending = ("action", 0, [True] * 9)
    side = SimpleNamespace(validator=validator, connection=Mock(), history=player_history,
                           actions_sent=0, decisions_sent=0, write_error=None)
    side.connection.sendall.side_effect = BrokenPipeError("engine closed after terminal")
    instance = env.HeadlessSelfPlayEnv.__new__(env.HeadlessSelfPlayEnv)
    instance.request = SimpleNamespace(player=0)
    instance.sides = {0: side}
    instance.request_time = time.monotonic()
    instance.agent_wait = 0
    instance.costs = {"respond": 0, "act": 0}
    instance.env_step = 0
    instance.step_latencies = []
    instance._write_ledger = lambda *args: None
    instance._advance = lambda: None
    before = len(player_history.v1.frames)
    instance.step(2)
    executions = {color: {"ready": True, "actions_sent": 1, "error": None,
                          "terminal": {"outcome": outcome},
                          "write_error": side.write_error if color == "red" else None}
                  for color, outcome in (("red", "win"), ("blue", "loss"))}
    classification = env.scoring.classify_headless_self_play(0, None, LOG, executions)
    added = player_history.v1.frames[before:]
    reproduced = side.actions_sent == 1 and any(f["kind"] == "action" for f in added)
    return {
        "finding": "P1 action consumption", "bug_reproduced": reproduced,
        "desired_behavior_passed": side.actions_sent == 0 and not any(f["kind"] == "action" for f in added),
        "setup": "Valid pending Red action; sendall raises BrokenPipeError. Actual step(), history fold and classifier execute. Only transport, next advance and unrelated ledger I/O are stubbed.",
        "observed": {"write_error": side.write_error, "actions_sent_delta": side.actions_sent,
                     "added_frames": added, "classification_with_write_error": classification},
        "expected": "A failed or unconsumed response must not become a taken-action sample or receive terminal credit. Preserve the preceding unacted decision, reconcile consumption, or exclude uncertain learner data. A separately verified natural outcome may remain reportable.",
        "source": {"experiments/headless/env.py": [367, 380],
                   "experiments/episodes/scoring.py": [92, 94],
                   "engine/ExplorerRL/ExplorerRL.cpp": [888, 890]},
    }


def finalization_order(order):
    with tempfile.TemporaryDirectory(prefix="heroes-boundary-recheck-") as directory:
        instance = env.HeadlessSelfPlayEnv(None, Path(directory), replay=False)
        instance.episode_dir = Path(directory)
        (instance.episode_dir / "logs").mkdir()
        instance.log = (instance.episode_dir / "logs/process.log").open("w")
        instance.log.write(LOG)
        instance.process = Mock()
        instance.process.wait.return_value = 0
        instance.started = time.monotonic()
        instance.reason = instance.detail = None
        instance.stopping = threading.Event()
        instance.shutdown_deadline = None
        instance.episode, instance.seed, instance.created_at = 1, 7, "controlled probe"
        instance.engine_wait = instance.agent_wait = 0
        instance.costs, instance.step_latencies, instance.launch_info = {}, [], {}
        instance.sides = {
            player: SimpleNamespace(color=color, reader=None, ledger=None, cutoff=None,
                                    terminal_at=0, ready_at=0, decode_seconds=0,
                                    unread_from=0, unread_frames_from=0,
                                    validator=SimpleNamespace(ready=True, terminal=outcome),
                                    history=SimpleNamespace(records=[], v1=SimpleNamespace(frames=[])))
            for player, (color, outcome) in enumerate((("red", "win"), ("blue", "loss")))
        }
        instance._execution = lambda side: {
            "ready": True, "actions_sent": 1, "error": None,
            "terminal": {"outcome": "win" if side.color == "red" else "loss"}}
        instance._counters = lambda side: {}
        instance._identity = lambda: {}
        instance._engine_evidence = lambda text: ({}, None)
        events = {
            "exit": ("exit", None, 0, time.monotonic()),
            "bridge_error": ("bridge_error", 0,
                             "ProtocolError: Bridge stream ended inside a message", time.monotonic()),
        }
        for kind in order:
            instance.events.put(events[kind])
        # The process is a Mock; this patch guarantees the failure branch cannot
        # address a real process group. All queue handling and scoring are real.
        with patch.object(env, "stop_process_group", lambda process: None):
            result = instance._advance()
        return {"queued_order": order, "status": result.status,
                "termination_reason": result.termination_reason, "rewards": result.rewards,
                "failure_detail": result.manifest["failure_detail"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Reviewed PR checkout root")
    parser.add_argument("--expected-head", required=True, help="Exact expected Git commit")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("boundary.json"))
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise SystemExit(f"Expected exact reviewed head {args.expected_head}, found {head}")
    load_reviewed_source(source)
    action = action_consumption()
    exit_first = finalization_order(["exit", "bridge_error"])
    error_first = finalization_order(["bridge_error", "exit"])
    stream = {
        "finding": "P2 stream finalization",
        "bug_reproduced": exit_first["status"] == "completed" and error_first["status"] == "failed",
        "desired_behavior_passed": all(result["status"] == "failed" and all(v is None for v in result["rewards"].values())
                                       for result in (exit_first, error_first)),
        "setup": "Actual _advance/_finish/scoring with identical terminal/log evidence and queued exit plus reader error; only event ordering changes. Process, unrelated reporting, and metadata are stubbed. No process is launched.",
        "observed": {"exit_first": exit_first, "reader_error_first": error_first},
        "expected": "Both orders fail closed with null rewards; a queued reader protocol error must survive final draining.",
        "source": {"experiments/headless/env.py": [[483, 485], [523, 536], [646, 661]]},
    }
    sources = ["experiments/headless/env.py", "experiments/episodes/scoring.py", "engine/ExplorerRL/ExplorerRL.cpp"]
    output = {"schema_version": 1, "reviewed_commit": head,
              "command": shlex.join([sys.executable, *sys.argv]),
              "source": str(source),
              "scope": "Controlled in-memory injection only; no source/test edits, real games, build, benchmark interference or training.",
              "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in sources},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "findings": [action, stream]}
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reviewed_commit": head, "output": str(destination),
                      "findings": [{"finding": f["finding"], "bug_reproduced": f["bug_reproduced"],
                                    "desired_behavior_passed": f["desired_behavior_passed"]}
                                   for f in output["findings"]]}, indent=2))


if __name__ == "__main__":
    main()
