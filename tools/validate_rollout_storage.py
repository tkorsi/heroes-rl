#!/usr/bin/env python3
"""Bounded, offline rollout-storage comparison on copied completed evidence.

No game, optimizer, seed reservation or admission capability is created. The
original capture and checkpoint inputs are hashed before and after the worker.
Raw copies and measurements belong in a fresh private artifacts directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_copied_training import binding, private_path, require, write_new

SCHEMA = "heroes-rollout-storage-comparison-v1"


def timed(report, name, operation):
    print(json.dumps(dict(phase=name, status="started")), flush=True)
    started = time.monotonic()
    value = operation()
    report["seconds"][name] = time.monotonic() - started
    print(json.dumps(dict(phase=name, status="passed", seconds=report["seconds"][name])), flush=True)
    return value


def source_files():
    names = ("experiments/learner/contextual_storage.py", "experiments/learner/contextual_rollout.py",
             "experiments/learner/selfplay_training.py", "experiments/headless/contextual_replay.py",
             "experiments/headless/contextual_replay_history.py", "validation/probes/validate_hero_policy.py",
             "tools/validate_rollout_storage.py")
    return {name: binding(ROOT / name) for name in names}


def worker(run, output, seconds):
    import torch
    from experiments.headless import contextual_evidence, contextual_replay
    from experiments.learner import contextual as c, selfplay_training as training
    from experiments.learner.contextual_rollout import CollectedEpisode
    from experiments.learner.contextual_storage import load_collected, save_collected
    from tools.run_selfplay_one_game import frozen_models
    from tools.run_selfplay_training import read, write

    started = time.monotonic()
    deadline = started + seconds
    report = dict(schema=SCHEMA, status="running", seconds={}, gameplay_started=False,
                  optimizer_updates=0, samples_admitted=0, seeds_reserved=False,
                  source=source_files(), run=str(run), output=str(output))
    originals = {}
    try:
        accepted = read(run / "report.json")
        shutdown = read(run / "shutdown.json")
        require(accepted.get("status") == "passed" and accepted.get("native_replay_verified") is True
                and accepted.get("both_policies_reconstructed") is True
                and accepted.get("training_input") is False
                and accepted.get("optimizer_updates") == 0
                and shutdown.get("owned_cleanup_verified") is True and shutdown.get("returncode") == 0,
                "Only a completed verified diagnostic may be copied")
        plan_path = Path(accepted["plan"]["path"])
        require(binding(plan_path) == accepted["plan"], "Accepted plan changed")
        plan = read(plan_path)
        source = Path(accepted["episode_directory"]).resolve()
        require(source.is_relative_to(run), "Source episode escaped the completed run")
        files = [p for p in source.rglob("*") if p.is_file()]
        files.extend((run / "report.json", run / "shutdown.json", plan_path, Path(accepted["proof"]["path"])))
        files.extend(Path(value["path"]) for value in plan["protected"].values())
        files.extend(Path(plan["parent"]["checkpoint"][key]["path"]) for key in ("policy", "resume"))
        originals = timed(report, "hash_originals_before", lambda: {str(p): binding(p) for p in set(files)})
        write_new(output / "originals-before.json", originals)
        folder = output / "episode"
        timed(report, "copy_complete_episode", lambda: shutil.copytree(source, folder))
        raw_path = folder / "neural-rollout.json"
        legacy_path = folder / "legacy-neural-rollout.json"
        raw_path.rename(legacy_path)

        def legacy_load():
            raw = read(legacy_path)
            collected = CollectedEpisode.from_dict(raw)
            require(collected.to_dict() == raw, "Legacy full roundtrip differs")
            return collected

        legacy = timed(report, "legacy_load_and_roundtrip", legacy_load)
        gc.collect()
        timed(report, "legacy_pretty_write", lambda: write(folder / "legacy-rewritten.json", legacy.to_dict(), new=True))
        gc.collect()
        timed(report, "external_seal_write", lambda: save_collected(raw_path, legacy))
        external = timed(report, "external_load_and_roundtrip", lambda: load_collected(raw_path))
        require(legacy == external, "Decoded histories/actions/supports/values/samples differ")
        report["exact_decoded_equality"] = True
        report["players"] = [dict(player=r.player, frames=len(r.history), decisions=len(r.decisions),
            estimates=len(r.estimates), samples=len(r.samples), history_sha256=r.history_sha256,
            weights_sha256=r.weights_sha256, boundary=r.boundary, optimization_eligible=r.optimization_eligible)
            for r in external.rollouts]
        old_binding, new_binding = binding(legacy_path), binding(raw_path)
        require(binding(folder / "legacy-rewritten.json")["sha256"] == old_binding["sha256"],
                "Legacy rewrite bytes differ from the recorded serializer")
        report["bytes"] = dict(legacy_rollout=old_binding["bytes"], external_envelope=new_binding["bytes"],
            source_episode=sum(p.stat().st_size for p in source.rglob("*") if p.is_file()))
        report["bytes"]["projected_episode"] = report["bytes"]["source_episode"] - old_binding["bytes"] + new_binding["bytes"]
        del legacy
        gc.collect()
        torch.set_num_threads(1)
        models = frozen_models(plan)
        report["policies"] = timed(report, "exact_both_policy_reconstruction", lambda: training.reconstruct_policies(
            external, models, plan["schedule"]["policy_seeds"], lambda: deadline - time.monotonic()))
        require(all(not r.optimization_eligible for r in external.rollouts), "Diagnostic became optimizer eligible")
        del external, models
        gc.collect()
        native = timed(report, "independent_transport_and_native_replay", lambda: contextual_replay.verify_episode(
            folder, binary=Path(plan["replay_binary"]["path"]), data=Path(plan["data"]),
            output=output / "fresh-verification", timeout=min(180, deadline - time.monotonic())))
        require(native.get("verified") is True and native.get("histories", {}).get("verified") is True,
                "Copied native replay failed: " + str(native.get("error")))
        report["replay"] = native
        require(source_files() == report["source"], "Benchmark source changed during execution")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            report["originals_unchanged"] = all(binding(name) == value for name, value in originals.items())
            report["original_files_checked"] = len(originals)
            if not report["originals_unchanged"]:
                report["status"] = "failed"
        except BaseException as error:
            report.update(status="failed", preservation_error=f"{type(error).__name__}: {error}")
        report["elapsed_seconds"] = time.monotonic() - started
        report["limitations"] = ["One archived long cutoff game, one timing measurement per operation; caches were not flushed.",
            "No full-game or full-training throughput claim; engine and policy semantics are unchanged.",
            "The projected episode size replaces only the duplicated rollout; raw replay/history evidence is retained."]
        write_new(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(math.isfinite(args.seconds) and 60 <= args.seconds <= 900, "Use a 60..900-second offline deadline")
    run = args.run.resolve()
    output = private_path(args.output, fresh=not args.worker)
    require(not output.is_relative_to(run) and not run.is_relative_to(output), "Use a separate diagnostic copy")
    if args.worker:
        report = worker(run, output, args.seconds)
        return 0 if report["status"] == "passed" else 1
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    output.mkdir(parents=True, mode=0o700)
    process = None
    try:
        with (output / "worker.log").open("xb") as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--run", str(run),
                "--output", str(output), "--seconds", str(args.seconds - 10)],
                deadline_monotonic=time.monotonic() + args.seconds, deadline_wall=time.time() + args.seconds,
                evidence_path=output / "deadline.jsonl", cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            process.wait(timeout=args.seconds + 1)
    finally:
        errors = []
        if process is not None:
            try:
                close_owned_process(process, timeout=5)
            except Exception as error:
                errors.append(str(error))
        try:
            cleanup_from_evidence(output / "deadline.jsonl", timeout=5)
        except Exception as error:
            errors.append(str(error))
        write_new(output / "shutdown.json", dict(owned_cleanup_verified=not errors, errors=errors,
            returncode=process.returncode if process else None))
    require(process is not None and process.returncode == 0 and not errors, "Offline comparison failed; retain private evidence")
    print(json.dumps(dict(status="passed", result=str(output / "result.json")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
