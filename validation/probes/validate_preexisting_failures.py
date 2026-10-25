#!/usr/bin/env python3
"""Validate exclusion of failures present before truncate(); no game is launched.

The source checkout and expected commit are explicit arguments. A successful
script exit only means both controlled scenarios executed: inspect each probe's
desired_behavior_passed field for the required failure/exclusion behavior.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probe(fixture, learner, scenario):
    env = fixture.env_module
    with tempfile.TemporaryDirectory(prefix="heroes-horizon-failure-probe-") as directory:
        instance = env.HeadlessSelfPlayEnv(None, Path(directory), replay=False)
        instance.episode_dir = Path(directory)
        (instance.episode_dir / "logs").mkdir()
        instance.log = (instance.episode_dir / "logs/process.log").open("w")
        instance.process = Mock()
        instance.process.wait.return_value = -11 if scenario == "already_crashed" else -15
        instance.process.poll.return_value = -11 if scenario == "already_crashed" else None
        instance.started = time.monotonic()
        instance.reason = instance.detail = None
        instance.stopping = threading.Event()
        instance.shutdown_deadline = None
        instance.episode, instance.seed, instance.created_at = 1, 7, "controlled probe"
        instance.engine_wait = instance.agent_wait = 0
        instance.costs, instance.step_latencies, instance.launch_info = {}, [], {}
        sides = {}
        for player, color in enumerate(("red", "blue")):
            history = env.PlayerHistory(player, color, "1")
            history.start([20, 20, 1])
            hero = fixture.obj(1, 0, 0, owner=player)
            message = fixture.observation(0, fixture.columns(0, 2), [hero], [
                fixture.update(0, appeared=[hero], revealed=fixture.columns(0, 2), own=fixture.OWN)])
            message.update(player=player, width=20, height=20)
            history.ingest(message)
            history.act(2, 0)
            history.ingest({"type": "action_result", "action_id": 0, "action": 2, "accepted": False,
                            "before": message["state"], "after": message["state"], "events": []})
            next_message = {**message, "action_id": 1, "events": []}
            record = history.ingest(next_message)
            sides[player] = SimpleNamespace(color=color, player=player, reader=None, connection=None,
                                             ledger=None, cutoff=None, cutoff_seq=None, cutoff_reason=None,
                                             terminal_at=None, ready_at=0, decode_seconds=0, limit=None,
                                             unread_from=len(history.records), unread_frames_from=len(history.v1.frames),
                                             history=history, validator=SimpleNamespace(ready=True, terminal=None))
            if player == 0:
                instance.request = env.Request(color, player, "action", record, [record], [], mask=record["action_mask"])
        instance.sides = sides
        instance._execution = lambda side: {"ready": True, "actions_sent": 1, "error": None,
                                           "limit": side.limit, "terminal": None}
        instance._counters = lambda side: {"history_v1": side.history.v1.summary()}
        instance._identity = lambda: {}
        instance._engine_evidence = lambda text: ({}, None)
        event_time = time.monotonic()
        if scenario == "already_crashed":
            instance.events.put(("exit", None, -11, event_time))
        else:
            instance.events.put(("bridge_error", 1, "ProtocolError: malformed JSON before truncate", event_time))
        truncate_started = time.monotonic()
        with patch.object(env, "stop_process_group", lambda process: None):
            result = instance.truncate("collection_window")
        frame_seq = result.bootstrap_v1_seq.get("red")
        prefix_valid = False
        if frame_seq is not None:
            cursor = learner.HistoryCursor("1", 0)
            frames = sides[0].history.v1.frames[:frame_seq + 1]
            for frame in frames:
                cursor.consume(learner.decode_frame(frame))
            prefix_valid = cursor.decision is not None and cursor.decision.seq == frame_seq
        return {"scenario": scenario, "failure_queued_before_truncate": event_time < truncate_started,
                "process_already_exited_before_truncate": scenario == "already_crashed",
                "observed": {"status": result.status, "reason": result.termination_reason,
                             "returncode": result.manifest["returncode"], "rewards": result.rewards,
                             "bootstrap_v1_seq": result.bootstrap_v1_seq, "valid_red_cutoff_prefix": prefix_valid},
                "expected": "Technical failure excludes this trajectory: failed status and no bootstrap. These events predate truncate(), so they were not caused by intentional process shutdown.",
                "bug_reproduced": result.status == "truncated" and frame_seq is not None and prefix_valid,
                "desired_behavior_passed": (result.status == "failed"
                    and all(value is None for value in result.rewards.values())
                    and all(value is None for value in result.bootstrap.values())
                    and all(value is None for value in result.bootstrap_v1_seq.values()))}


def function_lines(path, name, indentation):
    """Find current source lines instead of retaining an old revision's offsets."""
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.startswith(" " * indentation + "def " + name + "("))
    boundary = re.compile(r"^ {" + str(indentation) + r"}(?:def |class |@)")
    end = next((index for index in range(start + 1, len(lines)) if boundary.match(lines[index])), len(lines))
    return [start + 1, end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--learner-contract", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("boundary-horizon-validation.json"))
    args = parser.parse_args()
    source = args.source.resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise SystemExit(f"Expected {args.expected_head}, found {head}")
    fixture = load("horizon_probe_fixtures", source / "tests/test_headless.py")
    learner = load("horizon_probe_learner_contract", args.learner_contract.resolve())
    output = {"reviewed_commit": head,
              "command": shlex.join([sys.executable, *sys.argv]),
              "scope": "Actual truncate/finalization/classification and actual learner decoding; synthetic valid history, mock process/evidence/transport. No process, game or training launched.",
              "source": {"experiments/headless/env.py": [
                             function_lines(source / "experiments/headless/env.py", "truncate", 4),
                             function_lines(source / "experiments/headless/env.py", "_finish", 4)],
                         "experiments/episodes/scoring.py": function_lines(
                             source / "experiments/episodes/scoring.py", "classify_headless_self_play", 0)},
              "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in ("experiments/headless/env.py", "experiments/headless/player_history.py",
                                             "experiments/episodes/scoring.py")},
              "learner_contract_sha256": hashlib.sha256(args.learner_contract.read_bytes()).hexdigest(),
              "probes": [probe(fixture, learner, mode) for mode in ("already_crashed", "already_queued_reader_error")],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
