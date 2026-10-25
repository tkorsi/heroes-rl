#!/usr/bin/env python3
"""Bounded real-game boundary checks; random agents only, no training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_learner(end, histories, contract, trajectory):
    """Validate actual histories; failure targets are contract checks, never training data."""
    results = {}
    for player, color in enumerate(("red", "blue")):
        frames = tuple(contract.decode_frame(raw) for raw in histories[color])
        row = {"frames": len(frames), "taken_actions": sum(f.kind == "action" for f in frames)}
        if end.status == "truncated":
            seq = end.bootstrap_v1_seq[color]
            if seq is None:
                raise AssertionError(f"{color} has no cutoff sequence")
            prefix = frames[:seq + 1]
            cursor = contract.HistoryCursor(prefix[0].episode, player)
            for frame in prefix:
                cursor.consume(frame)
            cutoff = prefix[-1]
            assert cutoff.kind == "decision" and cutoff.player == player and cutoff.seq == seq
            assert cutoff.day == 2 and cursor.decision == cutoff
            row.update(cutoff_seq=seq, cutoff_day=cutoff.day, cutoff_is_own_unacted_decision=True)
        elif end.status == "failed":
            if frames:
                cursor = contract.HistoryCursor(frames[0].episode, player)
                for frame in frames:
                    cursor.consume(frame)
            records = []
            checkpoint, opponent = f"uniform-random-probe-player-{player}", f"uniform-random-probe-player-{1-player}"
            for action in (f for f in frames if f.kind == "action"):
                decision = frames[action.decision_seq]
                records.append(trajectory.DecisionRecord(
                    frame=decision, action=action.action, behavior_log_prob=-math.log(sum(decision.mask)),
                    # Required finite field; ignored by TechnicalFailure. This is not a critic prediction.
                    behavior_value=0.0, behavior_checkpoint=checkpoint, opponent_checkpoint=opponent,
                    recurrent_prefix=trajectory.recurrent_provenance(frames[:decision.seq + 1], checkpoint)))
            targets = ()
            if records:
                item = trajectory.PlayerTrajectory(frames[0].episode, player, checkpoint, opponent,
                                                   "learner", frames, tuple(records),
                                                   trajectory.TechnicalFailure(end.termination_reason))
                targets = item.transition_targets()
                assert not item.eligible
            assert all(not target.eligible and target.reward is None and target.bootstrap_value is None
                       and target.return_target is None for target in targets)
            row.update(targets=[asdict(target) for target in targets], eligible_targets=0,
                       behavior_value_is_ignored_contract_sentinel=True)
        else:
            raise AssertionError(f"Unexpected episode result: {end.status}")
        results[color] = row
    return results


def run_case(name, args, env_module, agent_module, contract, trajectory):
    output = args.output / name
    agents = {color: agent_module.UniformRandomAgent(seed) for color, seed in (("red", 1), ("blue", 100001))}
    launcher = env_module.VcmiLauncher(args.binary, args.data, args.map)
    launcher.preflight()
    limits = env_module.Limits(max_days=1 if name == "day-horizon" else 40, max_actions=500,
                              watchdog_seconds=10, startup_seconds=45, shutdown_seconds=10, bridge_timeout=30)
    env = env_module.HeadlessSelfPlayEnv(launcher, output, limits, ledger=True, replay=True,
                                       agents={color: agent.description for color, agent in agents.items()})
    detail = {"case": name, "passed": False, "injection": None}
    started = time.monotonic()
    request_count = 0
    try:
        episode = {"day-horizon": 1, "technical-kill": 2, "technical-kill-before-cutoff": 3}[name]
        state = env.reset(episode=episode, seed=40000 + episode)
        while not state.done:
            if time.monotonic() - started > args.wall_seconds or request_count >= 2000:
                raise RuntimeError("Bounded probe budget exhausted before the required boundary")
            assert state.reward == 0
            acknowledged = sum(side.accepted + side.rejected for side in env.sides.values())
            if name.startswith("technical-kill") and state.kind == "action" and acknowledged >= 1:
                process = env.process
                assert process is not None and process.poll() is None and env.request is state
                assert Path(process.args[0]).resolve() == args.binary
                # VcmiLauncher uses start_new_session=True and useProcess=False. Never find or signal other PIDs.
                group = os.getpgid(process.pid)
                assert group == process.pid and group != os.getpgrp()
                detail["injection"] = {"pid": process.pid, "private_pgid": group, "signal": "SIGKILL",
                                       "target": "Only the Popen process returned by this case's VcmiLauncher",
                                       "acknowledged_actions_before_kill": acknowledged,
                                       "pending_request_player": state.player,
                                       "pending_action_id": state.record.get("action_id"),
                                       "pending_request_kind": state.kind}
                process.kill()
                if name == "technical-kill-before-cutoff":
                    # The process is demonstrably dead before the public collection cutoff.
                    # A later truncate() must not turn its existing crash into an eligible boundary.
                    exited = process.wait(timeout=10)
                    detail["injection"].update(exit_before_truncate=exited,
                                                finalization_call="env.truncate() after process.wait()")
                    assert exited == -signal.SIGKILL
                    state = env.truncate()
                else:
                    # There is no public wait-only API. Drain the real exit through its normal finalizer,
                    # without sending an additional policy response or declaring a truncation.
                    state = env._advance()
                assert state.done
                break
            state = env.step(agents[state.color].act(state))
            request_count += 1
        end = state
        histories = {color: env.history_v1(color) for color in agents}
        for color, frames in histories.items():
            path = env.episode_dir / f"history-v1-{color}.jsonl"
            path.write_text("".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames))
        detail.update(status=end.status, termination_reason=end.termination_reason, rewards=end.rewards,
                      outcomes=end.outcomes, bootstrap=end.bootstrap, bootstrap_v1_seq=end.bootstrap_v1_seq,
                      manifest_path=str(env.episode_dir / "manifest.json"),
                      manifest_sha256=sha(env.episode_dir / "manifest.json"),
                      returncode=end.manifest["returncode"], ordinary_requests=request_count)
        assert all(value is None for value in end.rewards.values())
        assert all(value is None for value in end.outcomes.values())
        if name == "day-horizon":
            assert end.status == "truncated" and end.termination_reason == "game_day_limit"
            assert all(end.bootstrap[color] is not None and end.bootstrap[color]["day"] == 2 for color in agents)
        else:
            assert detail["injection"] is not None and end.status == "failed"
            assert end.manifest["returncode"] == -signal.SIGKILL
            assert all(value is None for value in end.bootstrap.values())
            assert all(value is None for value in end.bootstrap_v1_seq.values())
        detail["learner_contract"] = check_learner(end, histories, contract, trajectory)
        detail["passed"] = True
    except Exception as error:
        detail["error"] = f"{type(error).__name__}: {error}"
    finally:
        env.close()  # The environment owns only the private process/session it launched.
    output.mkdir(parents=True, exist_ok=True)
    detail["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write(output / "details.json", detail)
    return {key: value for key, value in detail.items() if key != "bootstrap"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "learner-source", "binary", "data", "map", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--wall-seconds", type=float, default=90)
    parser.add_argument("--case", action="append", choices=("day-horizon", "technical-kill", "technical-kill-before-cutoff"),
                        help="Repeat to select cases; defaults to day-horizon and technical-kill")
    args = parser.parse_args()
    for name in ("source", "learner_source", "binary", "data", "map", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.wall_seconds <= 0:
        parser.error("--wall-seconds must be positive")
    if args.output.exists():
        parser.error("--output must be new; existing evidence is never overwritten")
    sys.path[:0] = [str(args.source / "experiments/headless"), str(args.learner_source / "experiments")]
    import env as env_module
    import agents as agent_module
    from learner import contract, trajectory

    args.output.mkdir(parents=True)
    selected = args.case or ["day-horizon", "technical-kill"]
    if len(set(selected)) != len(selected):
        parser.error("Each --case may be selected only once")
    rows = [run_case(name, args, env_module, agent_module, contract, trajectory) for name in selected]
    result = {"schema_version": 1, "kind": "bounded-native-boundaries", "cases": rows,
              "passed": all(row["passed"] for row in rows), "training": False,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip(),
              "learner_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.learner_source, text=True).strip(),
              "binary_sha256": sha(args.binary), "probe_sha256": sha(Path(__file__)),
              "scope": f"{len(rows)} native random-agent episodes. Day cutoff and owned-process failure injection only. No value inference, parameter updates or replay verification."}
    write(args.output / "boundaries.json", result)
    print(json.dumps({"passed": result["passed"], "cases": [{k: row.get(k) for k in
                     ("case", "passed", "status", "termination_reason", "error")} for row in rows],
                      "report": str(args.output / "boundaries.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
