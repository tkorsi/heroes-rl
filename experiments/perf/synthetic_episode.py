"""Profile collection bookkeeping on a SYNTHETIC paired episode of realistic size.

This is a cloud-side measurement aid, never native or gameplay evidence. It
builds two strict v7 dialog-profile player histories whose map observations
grow to a full 20x20 known map (real long-cohort frames average about 9 KB),
drives them through the real ContextualCollector with randomly initialized
v7-sized models, and times each post-collection step separately:

- collect: the collector loop (encoding, forward_step, sampling bookkeeping);
- finish: ContextualCollector.finish (origin "synthetic", so no file audit);
- to_dict / dumps: rollout persistence as the evaluators write it;
- loads / from_dict: rebuilding the persisted rollout for reconstruction;
- replay_inference: a fresh session folding the complete history again.

Actions are scripted (move_to/end_turn) so that no window simulation is
needed; the model still scores every frame and evaluates every chosen action.
Only ContextualDistribution.sample is replaced, and only inside this process.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import sys
import time
from types import SimpleNamespace
import hashlib

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments/headless"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

COLORS = ("red", "blue")


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class ScriptedEpisode:
    """Two player histories with growing map knowledge and scripted actions."""

    def __init__(self, c, *, decisions, tiles_start, tiles_end, objects, episode):
        self.c, self.decisions, self.episode = c, decisions, episode
        self.tiles_start, self.tiles_end, self.objects = tiles_start, tiles_end, objects
        self.interface = c.DIALOGS_ENVIRONMENT
        self.pair = None
        self.unread = [0, 0]
        self.taken = [0, 0]
        self.positions = [[(1, 1), (2, 1)], [(18, 18), (17, 18)]]
        self.forced = {}
        self.turn = 0

    def observation(self, player):
        c = self.c
        progress = self.taken[player] / max(1, self.decisions)
        known = int(self.tiles_start + (self.tiles_end - self.tiles_start) * progress)
        order = sorted(((x, y) for x in range(20) for y in range(20)),
                       key=lambda p: abs(p[0] - (1 if player == 0 else 18)) + abs(p[1] - (1 if player == 0 else 18)))
        tiles = tuple(c.Tile((x, y, 0), (x * 7 + y) % 9, 1 if (x + y) % 5 == 0 else 0, (x * y) % 11 == 3, (x + 2 * y) % 13 == 1)
                      for x, y in sorted(order[:known]))
        objects = tuple(c.ObjectView(f"p{player}-object-{index}", order[index * 3 % known] + (0,), 5 + index % 40,
                                     index % 7, "neutral" if index % 3 else "self") for index in range(min(self.objects, known // 3)))
        heroes = tuple(c.Hero(f"p{player}-hero-{index}", position + (0,), 1500 - 10 * (self.taken[player] % 50), 1500, False)
                       for index, position in enumerate(self.positions[player]))
        towns = (c.Town(f"p{player}-town", ((2, 2) if player == 0 else (17, 17)) + (0,)),)
        return c.Observation((20, 20, 1), tiles, objects, (5, 5, 10, 10, 2, 2, 2500 + self.taken[player]), heroes, towns)

    def decide(self, player):
        c, history = self.c, self.pair[player]
        obs = self.observation(player)
        frame = history.emit("decision", observation=obs, revision=history.revision + 1,
                             offers=c.Offers.from_observation(obs, schema=c.DIALOGS_ACTION_SCHEMA))
        step = self.taken[player]
        if step % 10 == 9:
            action = c.Action("end_turn", schema=c.DIALOGS_ACTION_SCHEMA)
        else:
            hero = step % 2
            x, y = self.positions[player][hero]
            target = ((x + 1 + step % 3) % 20, (y + step % 2) % 20)
            action = c.Action("move_to", f"p{player}-hero-{hero}", *target, schema=c.DIALOGS_ACTION_SCHEMA)
        self.forced[(player, frame.seq)] = action
        return frame

    def request(self, player):
        from contextual_env import encoded
        history = self.pair[player]
        frames = encoded(history, self.unread[player])
        self.unread[player] = len(history.frames)
        return SimpleNamespace(done=False, color=COLORS[player], player=player, kind="contextual", past_horizon=False,
                               reward=0, frames=frames, record=frames[-1])

    def reset(self, *, episode, seed):
        c = self.c
        self.pair = tuple(c.PlayerHistory(str(episode), player, schema=c.DIALOGS_SCHEMA) for player in (0, 1))
        for player in (0, 1):
            self.decide(player)
        return self.request(0)

    def step(self, selected):
        c = self.c
        player = self.turn
        history = self.pair[player]
        seq = history.decision.seq
        history.emit("submitted", decision_seq=seq, action=selected)
        history.emit("consumed", decision_seq=seq, consumer="engine")
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        if selected.kind == "move_to":
            hero = int(selected.subject.rsplit("-", 1)[1])
            self.positions[player][hero] = (selected.x, selected.y)
            movement = 1500 - 10 * ((self.taken[player] + 1) % 50)
            history.emit("result", decision_seq=seq, result=c.Result("reached", True, (selected.x, selected.y, 0), movement, True))
        else:
            history.emit("result", decision_seq=seq, result=c.Result("turn_ended", True))
            history.day += 1
        self.taken[player] += 1
        history.emit("view", observation=self.observation(player))
        self.decide(player)
        self.turn = 1 - player
        return self.request(self.turn)

    def defer_dialog_cutoff(self, reason):
        raise AssertionError("Scripted map episode has no dialog")

    def truncate(self, reason):
        c = self.c
        for history in self.pair:
            history.emit("cutoff", decision_seq=history.decision.seq, reason=reason)
        c.finalize_cutoff(self.pair, evidence_ref="synthetic-not-engine-evidence", histories_complete=(True, True),
                          failure_free=True, shutdown_verified=True, continuation_limit=8, continuation_steps=(0, 0))
        return self.end()

    def end(self):
        c = self.c
        profile = self.pair[0].profile
        players, bootstraps = {}, {}
        for color, history in zip(COLORS, self.pair):
            prefix = history.frames[:-1]
            digest = hashlib.sha256(b"".join(json.dumps(c.encode_frame(f), sort_keys=True, separators=(",", ":")).encode() + b"\n"
                                             for f in prefix)).hexdigest()
            players[color] = {profile.history_key: {"canonical_sha256": history.digest(), "prefix_sha256": digest,
                                                    "boundary": "cutoff", "complete": True}}
            cutoff = next(frame for frame in history.frames if frame.kind == "cutoff")
            bootstraps[color] = c.encode_frame(history.frames[cutoff.decision_seq])
        outcomes = {color: None for color in COLORS}
        manifest = {"schema_version": 1, "kind": "contextual-headless-self-play", "environment": profile.environment,
                    "history_schema": profile.history_schema, "action_schema": profile.action_schema, "episode": self.pair[0].episode,
                    "status": "truncated", "evidence_ref": "synthetic-not-engine-evidence", "outcomes_by_player": outcomes,
                    "rewards_by_player": dict(outcomes), "returncode": -15, "players": players,
                    "termination_reason": "collection_window", "dialog_cutoff_drain": None}
        return SimpleNamespace(done=True, status="truncated", outcomes=outcomes, rewards=dict(outcomes),
                               bootstrap=bootstraps, manifest=manifest)

    def history(self, color):
        from contextual_env import encoded
        return encoded(self.pair[COLORS.index(color)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=400, help="Scripted choices per player")
    parser.add_argument("--tiles-start", type=int, default=92)
    parser.add_argument("--tiles-end", type=int, default=400)
    parser.add_argument("--objects", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from experiments.learner import contextual as c
    from experiments.learner import contextual_model as m
    from experiments.learner import contextual_rollout as r
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("Output exists; never overwrite an earlier profile")
    output.mkdir(parents=True)
    torch.manual_seed(20261025)
    model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).eval().requires_grad_(False)
    env = ScriptedEpisode(c, decisions=args.decisions, tiles_start=args.tiles_start, tiles_end=args.tiles_end,
                          objects=args.objects, episode="synthetic-performance-episode")
    original_sample = m.ContextualDistribution.sample
    collector = r.ContextualCollector((model, model), ("synthetic-red", "synthetic-blue"), learner_player=0,
        opponent_pool="synthetic-performance", interface=c.DIALOGS_ENVIRONMENT, purpose="synthetic", origin="synthetic",
        policy_seeds=(1, 2))

    def scripted_sample(self, generator=None):
        session = collector.sessions[env.turn]
        return env.forced[(env.turn, session.history.decision.seq)]
    m.ContextualDistribution.sample = scripted_sample
    report = {"schema": "heroes-performance-synthetic-episode-v1", "synthetic": True, "native_evidence": False,
              "decisions_per_player": args.decisions, "tiles": [args.tiles_start, args.tiles_end], "objects": args.objects,
              "torch_threads": torch.get_num_threads(), "python": sys.version, "torch": torch.__version__, "steps": {}}
    state = {}
    finish = collector.finish

    def timed(name, function):
        profiler = cProfile.Profile() if args.cprofile else None
        started, cpu = time.perf_counter(), _cpu()
        if profiler: profiler.enable()
        try:
            value = function()
        finally:
            if profiler: profiler.disable()
        report["steps"][name] = {"wall_seconds": round(time.perf_counter() - started, 6), "cpu_seconds": round(_cpu() - cpu, 6)}
        if profiler:
            profiler.dump_stats(output / f"{name}.prof")
            for order in ("tottime", "cumulative"):
                text = io.StringIO()
                pstats.Stats(profiler, stream=text).sort_stats(order).print_stats(40)
                (output / f"{name}-{order}.txt").write_text(text.getvalue())
        print(name, report["steps"][name], flush=True)
        return value

    def deferred_finish(end, *, histories, manifest_path=None):
        state["end"], state["histories"] = end, histories
        return None
    collector.finish = deferred_finish
    timed("collect", lambda: collector.run_episode(env, episode=env.episode, seed=1, max_requests=2 * args.decisions))
    collector.finish = finish
    m.ContextualDistribution.sample = original_sample
    frames = [len(h.frames) for h in env.pair]
    sizes = [sum(len(record) + 1 for record in __import__("contextual_env").canonical_records(h)) for h in env.pair]
    report.update(frames=frames, history_bytes=sizes)
    print("frames", frames, "history bytes", sizes, flush=True)
    episode = timed("finish", lambda: finish(state["end"], histories=state["histories"]))
    raw = timed("to_dict", episode.to_dict)
    text = timed("dumps", lambda: json.dumps(raw, sort_keys=True, indent=2, allow_nan=False))
    report["rollout_bytes"] = len(text)
    report["rollout_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    loaded = timed("loads", lambda: json.loads(text))
    rebuilt = timed("from_dict", lambda: r.CollectedEpisode.from_dict(loaded))
    if rebuilt != episode:
        raise SystemExit("Rebuilt rollout differs")
    if "audit" not in args.skip:
        from contextual_env import persist_history
        from validation.probes.audit_contextual_histories import audit_player
        paths = []
        for color, history in zip(COLORS, env.pair):
            paths.append(output / f"history-{color}.jsonl")
            persist_history(paths[-1], history)
        reports = timed("audit_players", lambda: [audit_player(path, player)[0] for player, path in enumerate(paths)])
        report["audit_sha256"] = [(item["canonical_sha256"], item["prefix_sha256"], not item["failures"]) for item in reports]
        for path in paths:
            path.unlink()  # synthetic scratch data; only the digests are retained
    if "replay_inference" not in args.skip:
        def replay():
            session = r.ContextualSession(model, env.episode, 0, "synthetic-red", "synthetic-blue", schema=c.DIALOGS_SCHEMA, seed=1)
            session.consume(rebuilt.rollouts[0].history)
            return session
        timed("replay_inference", replay)
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["history_sha256"] = [h.digest() for h in env.pair]
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
