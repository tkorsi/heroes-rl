"""Engine-free replay of the built-in collector's controller over recorded real games.

Feeds each persisted own history to a fresh ContextualSession exactly as
BuiltinCollector.act receives it (wire dicts, one request per recorded choice,
then the final frames) and requires every choice, value and prefix to equal
the persisted real-game rollout bit for bit. Reports controller time.

Usage: python tools/acceleration/controller_replay.py [--typed] EPISODE_DIR...
It imports the checkout it sits in, or the one HEROES_ROOT names (to time the
base code). --typed also hands over the typed frames, as the environments of
the accelerated code do (the base code has no typed-frame path, so time it
without --typed). With --repeat N the same model replays the games N times;
later passes show what the model's inference memos save, as when training
audits a game with the model that collected it.
"""
import argparse, cProfile, json, pstats, sys, time
from pathlib import Path
sys.path.insert(0, __import__("os").environ.get("HEROES_ROOT") or str(Path(__file__).resolve().parents[2]))
import torch
torch.set_num_threads(1)
from experiments.learner import contextual as c
from experiments.learner.builtin_rollout import load_builtin_episode, BuiltinCollector
from experiments.headless import builtin_evidence as evidence
from validation.probes.validate_embedded_engine import random_policy


TYPED = False


def requests(collected):
    chosen = sorted(item.decision_seq for item in collected.decisions)
    wire = [json.loads(c.canonical_frame(frame)) for frame in collected.history]
    start, chunks, typed = 0, [], []
    for seq in chosen:
        chunks.append(wire[start:seq + 1]); typed.append(tuple(collected.history[start:seq + 1])); start = seq + 1
    return chunks, wire[start:], typed, tuple(collected.history[start:])


def replay(folder, model, checkpoint, profiler=None):
    """BuiltinCollector.act for every recorded request, then finish, as in collection."""
    from types import SimpleNamespace
    from experiments.headless.builtin_env import COLORS
    collected = load_builtin_episode(folder / "builtin-rollout.json")
    manifest = collected.manifest
    profile = evidence.manifest_profile(manifest)
    collector = BuiltinCollector(model, checkpoint, learner_player=collected.player,
                                 opponent_identity=collected.opponent_identity, policy_seed=collected.policy_seed,
                                 interface=profile.environment)
    chunks, final, typed, final_typed = requests(collected)
    color = COLORS[collected.player]
    extra = lambda frames: {"typed_frames": frames} if TYPED else {}
    asked = [SimpleNamespace(done=False, kind="contextual", player=collected.player, color=color, past_horizon=False,
                             reward=0, frames=frames, record=frames[-1], **extra(t)) for frames, t in zip(chunks, typed)]
    end = SimpleNamespace(done=True, final_frames={color: final}, manifest=manifest,
                          **({"final_typed_frames": {color: final_typed}} if TYPED else {}))
    if profiler: profiler.enable()
    begun = time.perf_counter()
    collector.start(collected.history[0].episode)
    actions = [collector.act(request) for request in asked]
    result = collector.finish(end, manifest_path=folder / "manifest.json")
    elapsed = time.perf_counter() - begun
    if profiler: profiler.disable()
    for same, what in ((result.decisions == collected.decisions, "choices"),
                       (result.estimates == collected.estimates, "estimates"),
                       (result.history == collected.history, "history"),
                       (tuple(actions) == tuple(item.action for item in collected.decisions), "actions")):
        if not same:
            raise SystemExit(f"{folder.name}: {what} differ from the persisted rollout")
    return {"episode": folder.name, "frames": len(collected.history), "decisions": len(chunks), "seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--interface", default=c.DIALOGS_ENVIRONMENT)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--typed", action="store_true", help="hand the persisted typed frames over as an environment does")
    args = parser.parse_args()
    global TYPED
    TYPED = args.typed
    loaded, _ = random_policy(args.seed, args.interface)
    profiler = cProfile.Profile() if args.profile else None
    results, passes = [], []
    for _ in range(args.repeat):
        done = [replay(folder, loaded.model, loaded.checkpoint, profiler) for folder in args.episodes]
        results += done
        passes.append(1e3 * sum(r["seconds"] for r in done) / sum(r["decisions"] for r in done))
    total = sum(r["seconds"] for r in results); decisions = sum(r["decisions"] for r in results)
    frames = sum(r["frames"] for r in results)
    print(json.dumps({"runs": len(results), "decisions": decisions, "frames": frames, "seconds": total,
                      "ms_per_decision": 1e3 * total / decisions, "ms_per_frame": 1e3 * total / frames,
                      "ms_per_decision_per_pass": passes, "identical": True}))
    if profiler:
        profiler.dump_stats(args.profile)
        pstats.Stats(str(args.profile)).sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()
