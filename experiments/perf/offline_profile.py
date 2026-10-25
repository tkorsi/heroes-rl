"""Attribute post-collection costs on a COPY of one retained contextual episode.

Read-only with respect to its input: run it on a copied episode folder, never on
an original scientific artifact. No game, native replay or optimizer is started.
Each step is timed (wall and process CPU) and optionally profiled with cProfile:

- audit: the independent persisted-history audit used by collection and replay;
- decode: decoding both persisted histories into strict PlayerHistory objects;
- digest: one canonical JSON digest pass over each decoded history;
- transitions: one trajectory derivation per history;
- rollout_from_dict: rebuilding the persisted neural rollout with its checks;
- rollout_to_dict: re-encoding and serializing it, as persistence does;
- behavior: offline behavior analysis;
- reconstruction: the full fixed-weight policy reconstruction (needs --checkpoint).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = ("audit", "decode", "digest", "transitions", "rollout_from_dict", "rollout_to_dict", "behavior", "reconstruction")


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="A copied episode directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--interface", default="adventure-dialogs-v1")
    parser.add_argument("--policy-seeds", type=int, nargs=2, help="Red and blue policy seeds for reconstruction")
    parser.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from experiments.learner import contextual as c
    from experiments.learner.contextual_rollout import CollectedEpisode
    from validation.probes.audit_contextual_histories import audit
    folder = args.episode.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("Output exists; never overwrite an earlier profile")
    output.mkdir(parents=True)
    manifest = json.loads((folder / "manifest.json").read_text())
    profile = c.profile_for(args.interface)
    paths = [folder / manifest["players"][color][profile.history_key]["path"] for color in ("red", "blue")]
    state = {}
    report = {"schema": "heroes-performance-offline-profile-v1", "episode": str(folder),
              "manifest_status": manifest.get("status"), "termination_reason": manifest.get("termination_reason"),
              "history_bytes": [path.stat().st_size for path in paths], "steps": {}, "training_performed": False}

    def decode():
        histories = []
        for player, path in enumerate(paths):
            frames = [c.decode_frame(json.loads(line)) for line in path.read_text().splitlines()]
            history = c.PlayerHistory(frames[0].episode, player, schema=frames[0].schema)
            for frame in frames:
                history.append(frame)
            histories.append(history)
        state["histories"] = histories
        return {"frames": [len(h.frames) for h in histories]}

    def collected():
        raw = json.loads((folder / "neural-rollout.json").read_text())
        state["collected"] = CollectedEpisode.from_dict(raw)
        return {"decisions": [len(r.decisions) for r in state["collected"].rollouts]}

    def to_dict():
        text = json.dumps(state["collected"].to_dict(), sort_keys=True, indent=2, allow_nan=False)
        return {"bytes": len(text)}

    def behavior():
        from experiments.learner.evaluation_behavior import analyze_episode
        return {"status": analyze_episode(folder).get("status")}

    def reconstruction():
        from experiments.learner.checkpoints import load_policy_checkpoint
        from validation.probes.validate_hero_policy import audit_persisted_policy
        if args.checkpoint is None or args.policy_seeds is None:
            return {"skipped": "requires --checkpoint and --policy-seeds"}
        loaded = load_policy_checkpoint(args.checkpoint, interface=args.interface, allowed_provenance=("random_initialization",))
        loaded.model.eval().requires_grad_(False)
        result = audit_persisted_policy(folder, loaded, list(args.policy_seeds), interface=args.interface)
        return {"players": [{key: p[key] for key in ("player", "actions_resampled", "values_recomputed", "conditionals_recomputed")}
                            for p in result["players"]]}

    actions = {
        "audit": lambda: {"consistent": audit(*paths, folder / "manifest.json")["consistent"]},
        "decode": decode,
        "digest": lambda: {"sha256": [h.digest() for h in state["histories"]]},
        "transitions": lambda: {"transitions": [len(h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA)) for h in state["histories"]]},
        "rollout_from_dict": collected,
        "rollout_to_dict": to_dict,
        "behavior": behavior,
        "reconstruction": reconstruction,
    }
    for name in STEPS:
        if name not in args.steps:
            continue
        if name in ("digest", "transitions") and "histories" not in state:
            decode()
        if name == "rollout_to_dict" and "collected" not in state:
            collected()
        profiler = cProfile.Profile() if args.cprofile else None
        started, cpu = time.perf_counter(), _cpu()
        if profiler:
            profiler.enable()
        try:
            detail = actions[name]()
            error = None
        except Exception as exc:  # retained: a failed step is part of the evidence
            detail, error = None, f"{type(exc).__name__}: {exc}"
        finally:
            if profiler:
                profiler.disable()
        report["steps"][name] = {"wall_seconds": round(time.perf_counter() - started, 6),
                                 "cpu_seconds": round(_cpu() - cpu, 6), "detail": detail, "error": error,
                                 "cprofile": bool(profiler)}
        if profiler:
            profiler.dump_stats(output / f"{name}.prof")
            for order in ("tottime", "cumulative"):
                text = io.StringIO()
                pstats.Stats(profiler, stream=text).sort_stats(order).print_stats(45)
                (output / f"{name}-{order}.txt").write_text(text.getvalue())
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(name, json.dumps(report["steps"][name]), flush=True)
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["pid"] = os.getpid()
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
