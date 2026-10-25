"""Bounded matched runs of the cold and embedded built-in backends.

One invocation runs one scenario sequentially in this process. The cold
backend launches one game process per row; the embedded backend loads one
in-process engine once and reuses it for every row. Both use the unchanged
BuiltinHeadlessEnv contract, the same map, limits, seat/seed schedule and
controller, and every row can be verified by the independent replay tool.

Controllers are evaluation-only: ``policy`` samples an unchanged frozen v7
checkpoint through BuiltinCollector; ``end_turn`` mirrors the reusable
NativeSession benchmark (end turn, first displayed dialog choice); ``visit``
moves an owned hero to a currently visible object of one declared kind and
answers its dialog with the first displayed choice; ``stall`` withholds the
first answer beyond the bridge timeout to prove failure poisoning. None of
them is a learner, reward, demonstration or training input.

Times are per phase: reset through the first decision, engine transition
work, controller (policy inference) time, finalization, replay and audit.
They overlap where the backends overlap work (the cold reader thread decodes
while the engine runs), so they are reported as totals, not shares.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless import embedded_evidence as embedded  # noqa: E402
from experiments.headless.contextual_evidence import runtime_identity, strict_json  # noqa: E402
from experiments.learner import contextual as c  # noqa: E402
from validation.probes.validate_builtin_boundaries import binding, require, until, write  # noqa: E402

SCHEMA = "heroes-embedded-engine-validation-v1"
COLORS = ("red", "blue")
CONTROLLERS = ("policy", "end_turn", "visit", "stall")
BACKENDS = ("cold", "embedded")
SCOPE = ("Bounded evaluation-only rows; no training, rewards, demonstrations or map changes. "
         "Controllers other than policy are explicit test workloads, never learner inputs.")


def schedule(count, seats, engine_seed, policy_seed):
    return [{"index": index, "episode": index + 1, "learner_player": seats[index % len(seats)],
             "engine_seed": engine_seed + index, "policy_seed": policy_seed + index} for index in range(count)]


def percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "p50": pick(.5), "p90": pick(.9),
            "p99": pick(.99), "max": ordered[-1]}


def max_rss_bytes(who):
    value = resource.getrusage(who).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def host_resources(engine=None):
    """Host process state; the embedded module adds native RSS, threads and descriptors."""
    snapshot = {"python_threads": threading.active_count(), "peak_self_bytes": max_rss_bytes(resource.RUSAGE_SELF),
                "peak_children_bytes": max_rss_bytes(resource.RUSAGE_CHILDREN)}
    if engine is not None:
        snapshot.update(engine.resources())
    return snapshot


class Timed:
    def __init__(self):
        self.seconds, self.calls, self.samples = 0.0, 0, []

    def wrap(self, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.seconds += elapsed
                self.calls += 1
                self.samples.append(elapsed)
        return timed

    def summary(self):
        return {"seconds": self.seconds, "calls": self.calls, "latency": percentiles(self.samples)}


class ScriptedRunner:
    """BuiltinCollector.run_episode semantics for explicit test controllers."""

    def __init__(self, kind, *, learner_player, visit_kind=None, stall_seconds=0.0):
        self.kind, self.learner_player = kind, learner_player
        self.visit_kind, self.stall_seconds = visit_kind, stall_seconds
        self.session = None  # retain_failed_builtin_evidence keeps no policy evidence for scripts
        self.dialog_answers, self.visit_moves, self.stalled = [], 0, False

    def act(self, request):
        frame = c.decode_frame(request.frames[-1])
        require(frame.kind == "decision" and frame.player == self.learner_player, "Scripted controller needs an own decision")
        observation, offers = frame.observation, frame.offers
        kind = observation.context.kind
        if self.kind == "stall" and not self.stalled:
            self.stalled = True
            time.sleep(self.stall_seconds)
        if kind == "dialog":
            choice = offers.dialog_panel.choices[0]
            self.dialog_answers.append({"day": frame.day, "selection": choice.ref})
            return c.Action("choose_dialog", subject=offers.context.subject, selection=choice.ref,
                            schema=c.DIALOGS_ACTION_SCHEMA)
        require(kind == "map", "Scripted workload reached an unsupported window; no fabricated action")
        if self.kind == "visit" and not self.dialog_answers and self.visit_moves < 2 * frame.day:
            targets = [obj for obj in observation.objects if obj.kind == self.visit_kind and obj.owner != "self"]
            heroes = [hero for hero in observation.heroes if hero.handle in offers.movable_hero_refs and hero.movement > 0]
            if targets and heroes:
                target = targets[0].position
                hero = min(heroes, key=lambda h: max(abs(h.position[0] - target[0]), abs(h.position[1] - target[1])))
                self.visit_moves += 1
                return c.Action("move_to", subject=hero.handle, x=target[0], y=target[1], schema=c.DIALOGS_ACTION_SCHEMA)
        return c.Action("end_turn", schema=c.DIALOGS_ACTION_SCHEMA)

    def run_episode(self, env, *, episode, seed, max_requests):
        request, choices = env.reset(episode=episode, seed=seed), 0
        while not request.done:
            if choices >= max_requests:
                if c.decode_frame(request.frames[-1]).observation.context.kind == "dialog":
                    env.defer_dialog_cutoff("collection_window")
                    request = env.step(self.act(request))
                    choices += 1
                else:
                    request = env.truncate("collection_window")
            else:
                request = env.step(self.act(request))
                choices += 1
        return request


def history_summary(folder, manifest):
    """Action mix, results and visible events from the persisted own history."""
    color = COLORS[manifest["learner_player"]]
    declaration = manifest["players"][color]["history_v7"]
    frames = [c.decode_frame(json.loads(line)) for line in (folder / declaration["path"]).read_text().splitlines() if line]
    submitted = Counter(f.action.kind for f in frames if f.kind == "submitted")
    results = Counter(f.result.status for f in frames if f.kind == "result")
    acknowledged = Counter("accepted" if f.accepted else "rejected" for f in frames if f.kind == "acknowledged")
    events = Counter(f.event.kind for f in frames if f.kind == "event")
    return {"frames": len(frames), "decisions": sum(f.kind == "decision" for f in frames),
            "submitted": sum(submitted.values()), "action_mix": dict(sorted(submitted.items())),
            "result_mix": dict(sorted(results.items())),
            "results_accepted": sum(f.result.accepted for f in frames if f.kind == "result"),
            "results_rejected": sum(not f.result.accepted for f in frames if f.kind == "result"),
            "commands": dict(acknowledged), "visible_events": dict(sorted(events.items())),
            "last_day": max((f.day for f in frames), default=None)}


def canonical_history(folder, manifest):
    color = COLORS[manifest["learner_player"]]
    return (folder / manifest["players"][color]["history_v7"]["path"]).read_bytes().splitlines()


def delta(before, after):
    return {key: after[key] - before[key] for key in after if isinstance(after[key], (int, float))}


def make_launcher(args):
    if args.backend == "embedded":
        from experiments.headless.embedded_env import EmbeddedLauncher
        return EmbeddedLauncher(args.binary, args.data, args.map)
    from experiments.headless.env import VcmiLauncher
    return VcmiLauncher(args.binary, args.data, args.map)


def run(args):
    import torch
    torch.set_num_threads(1)
    from experiments.headless.builtin_env import BuiltinHeadlessEnv
    from experiments.headless.builtin_replay import verify_episode
    from experiments.headless.env import Limits
    from experiments.learner.builtin_rollout import (
        BuiltinCollector, audit_persisted_builtin_policy, retain_failed_builtin_evidence)
    from experiments.learner.evaluate_builtin_opponent import load_fixed, native_opponent
    started_process = time.monotonic()
    for name in ("binary", "replay_binary", "data", "map", "output", "checkpoint"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    output = args.output
    require(not output.exists(), "Preserve every earlier run; use a new output directory")
    require(args.controller != "policy" or args.checkpoint is not None, "The policy controller needs --checkpoint")
    require(args.controller != "visit" or args.visit_kind is not None, "The visit controller needs --visit-kind")
    seats = [int(seat) for seat in args.seats.split(",")]
    require(seats and all(seat in (0, 1) for seat in seats), "Seats are 0 and/or 1")
    rows = schedule(args.episodes, seats, args.engine_seed, args.policy_seed)
    runtime = runtime_identity(args.binary.parent)
    loaded = fixed = None
    if args.controller == "policy":
        loaded, fixed = load_fixed(args.checkpoint)
    output.mkdir(parents=True)
    limits = Limits(max_days=args.max_days, max_actions=args.choices + 8, startup_seconds=60,
                    watchdog_seconds=args.watchdog, shutdown_seconds=15, bridge_timeout=args.bridge_timeout)
    plan = {"schema": SCHEMA, "backend": args.backend, "controller": args.controller, "rows": rows,
            "map": binding(args.map), "binary": str(args.binary), "replay_binary": str(args.replay_binary),
            "data": str(args.data), "runtime": runtime, "checkpoint": binding(args.checkpoint) if args.checkpoint else None,
            "frozen": fixed, "limits": {"max_days": args.max_days, "choices": args.choices, "hard_choices": args.choices + 8,
            "watchdog_seconds": args.watchdog, "bridge_timeout": args.bridge_timeout, "row_seconds": args.row_seconds},
            "visit_kind": args.visit_kind, "stall_seconds": args.stall_seconds, "replay": args.replay,
            "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                         "torch": torch.__version__, "cpus": os.cpu_count()},
            "scope": SCOPE, "training_performed": False}
    write(output / "plan.json", plan)
    report = {"schema": SCHEMA + "-result", "plan": binding(output / "plan.json"), "status": "running", "rows": []}
    launcher = make_launcher(args)
    engine = None
    if args.backend == "embedded":
        from experiments.headless.embedded_env import EmbeddedEngine
        loading = time.perf_counter()
        engine = EmbeddedEngine.acquire(launcher, output / "embedded-worker")
        report["engine_load_seconds"] = time.perf_counter() - loading
        report["engine_timings_after_load"] = engine.timings()
    report["resources_start"] = host_resources(engine)
    for row in rows:
        seat = row["learner_player"]
        stall = args.controller == "stall" and row["index"] == 0
        controller = args.controller if args.controller != "stall" or stall else "end_turn"
        result = {**row, "controller": controller, "status": "failed"}
        env = runner = None
        policy, step, finish = Timed(), Timed(), Timed()
        native_before = engine.timings() if engine else None
        row_started = time.perf_counter()
        try:
            with until(time.monotonic() + args.row_seconds, time.time() + args.row_seconds, 0):
                agents = {COLORS[seat]: {"kind": f"embedded-validation-{controller}", "evaluation_only": True}}
                options = dict(learner_player=seat, limits=limits, agents=agents)
                if args.backend == "embedded":
                    from experiments.headless.embedded_env import EmbeddedBuiltinEnv
                    env = EmbeddedBuiltinEnv(launcher, output, engine=engine, **options)
                else:
                    env = BuiltinHeadlessEnv(launcher, output, **options)
                reset = env.reset
                reset_seconds = []

                def timed_reset(*a, **k):
                    begun = time.perf_counter()
                    try:
                        return reset(*a, **k)
                    finally:
                        reset_seconds.append(time.perf_counter() - begun)
                env.reset = timed_reset
                env.step = step.wrap(env.step)
                env._finish = finish.wrap(env._finish)
                if controller == "policy":
                    opponent = native_opponent(runtime, seat)

                    class TimedCollector(BuiltinCollector):
                        act = policy.wrap(BuiltinCollector.act)
                    runner = TimedCollector(loaded.model, loaded.checkpoint, learner_player=seat,
                                            opponent_identity=opponent, policy_seed=row["policy_seed"])
                    collected = runner.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                                   max_requests=args.choices)
                    write(env.episode_dir / "builtin-rollout.json", collected.to_dict())
                    manifest = collected.manifest
                else:
                    runner = ScriptedRunner(controller, learner_player=seat, visit_kind=args.visit_kind,
                                            stall_seconds=args.stall_seconds)
                    runner.act = policy.wrap(runner.act)
                    end = runner.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                             max_requests=args.choices)
                    manifest = end.manifest
                    result["dialog_answers"] = runner.dialog_answers
                result.update(status="collected", episode_status=manifest["status"],
                              termination_reason=manifest["termination_reason"],
                              outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                              reset_seconds=reset_seconds[0] if reset_seconds else None)
        except Exception as error:
            result.update(error=f"{type(error).__name__}: {error}")
            if env is not None and runner is not None:
                retain_failed_builtin_evidence(env, runner if controller == "policy" else None, result, error,
                                               cleanup_seconds=30)
        collected_at = time.perf_counter()
        result["collection_seconds"] = collected_at - row_started
        result["step"] = step.summary()
        result["controller_time"] = policy.summary()
        result["finalize"] = finish.summary()
        if env is not None:
            result.update(engine_wait_seconds=env.engine_wait, agent_wait_seconds=env.agent_wait,
                          adapter_seconds=dict(env.costs), step_latency=percentiles(env.step_latencies))
            folder = env.episode_dir
            if folder is not None:
                result["episode_directory"] = str(folder)
                if (folder / "manifest.json").is_file():
                    manifest = strict_json((folder / "manifest.json").read_bytes())
                    result.setdefault("episode_status", manifest["status"])
                    result.setdefault("termination_reason", manifest["termination_reason"])
                    try:
                        result["history"] = history_summary(folder, manifest)
                    except Exception as error:
                        result["history_error"] = f"{type(error).__name__}: {error}"
                    final_path = folder / "builtin-finalization.json"
                    if final_path.is_file():
                        final = strict_json(final_path.read_bytes())
                        result["native_close"] = final.get("native_close")
                        result["returncode"] = final.get("returncode")
            try:
                env.close()
            except Exception as error:
                result["close_error"] = f"{type(error).__name__}: {error}"
        if engine is not None:
            result["native_timings"] = delta(native_before, engine.timings())
            result["poisoned_after"] = engine.poisoned
        verify = args.replay == "every" or (args.replay == "first" and row["index"] == 0)
        if result["status"] == "collected" and verify:
            begun = time.perf_counter()
            replay = verify_episode(Path(result["episode_directory"]), binary=args.replay_binary, data=args.data,
                                    timeout=args.row_seconds)
            result["replay_seconds"] = time.perf_counter() - begun
            result["replay_verified"] = replay.get("verified") is True
            result["replay_error"] = replay.get("error")
            if replay.get("embedded_backend") or replay.get("identity_check", {}).get("embedded_backend"):
                result["replay_embedded_identity"] = True
            if controller == "policy" and result["replay_verified"]:
                begun = time.perf_counter()
                try:
                    rebuilt, _ = load_fixed(args.checkpoint)
                    audit = audit_persisted_builtin_policy(Path(result["episode_directory"]), rebuilt.model,
                        rebuilt.checkpoint, row["policy_seed"], native_opponent(runtime, seat))
                    result["policy_audit"] = audit.get("passed") is True
                except Exception as error:
                    result["policy_audit"] = False
                    result["policy_audit_error"] = f"{type(error).__name__}: {error}"
                result["audit_seconds"] = time.perf_counter() - begun
        if result["status"] == "collected":
            natural = result["episode_status"] == "completed"
            result["status"] = ("passed" if (not verify or result.get("replay_verified"))
                                and result.get("policy_audit", True) else "failed")
            result["natural"] = natural
        result["row_seconds"] = time.perf_counter() - row_started
        result["resources_after"] = host_resources(engine)
        report["rows"].append(result)
        write(output / "report.json", report)
        if engine is not None and engine.poisoned:
            # A failure poisons the in-process engine: the next reset must be refused.
            refusal = None
            try:
                engine.require_ready()
            except ValueError as error:
                refusal = str(error)
            report["poisoned"] = {"after_episode": row["episode"], "next_reset_refused": refusal is not None,
                                  "refusal": refusal}
            break
    report["summary"] = summarize(report["rows"], args.backend)
    report["status"] = "finished"
    report["process_seconds"] = time.monotonic() - started_process
    report["resources_end"] = host_resources(engine)
    if args.checkpoint is not None:
        report["checkpoint_after"] = binding(args.checkpoint)
        require(report["checkpoint_after"] == plan["checkpoint"], "Checkpoint changed during validation")
    write(output / "report.json", report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return report


def summarize(rows, backend):
    collected = [row for row in rows if row.get("episode_status") is not None]
    statuses = Counter((row.get("episode_status"), row.get("termination_reason")) for row in rows)
    collection = sum(row["collection_seconds"] for row in rows)
    submitted = sum(row.get("history", {}).get("submitted", 0) for row in rows)
    actions, results, events = Counter(), Counter(), Counter()
    for row in rows:
        actions.update(row.get("history", {}).get("action_mix", {}))
        results.update(row.get("history", {}).get("result_mix", {}))
        events.update(row.get("history", {}).get("visible_events", {}))
    natural = [row for row in rows if row.get("episode_status") == "completed"]
    native = Counter()
    for row in rows:
        native.update({key: value for key, value in row.get("native_timings", {}).items()})
    return {"backend": backend, "rows": len(rows), "passed": sum(row["status"] == "passed" for row in rows),
            "episode_statuses": {f"{s}/{t}": n for (s, t), n in sorted(statuses.items(), key=str)},
            "natural_completed": len(natural), "replay_verified": sum(bool(row.get("replay_verified")) for row in rows),
            "collection_seconds": collection, "submitted_decisions": submitted,
            "decisions_per_second": submitted / collection if collection else None,
            "completed_games_per_second": len(natural) / collection if collection else None,
            "rows_per_second": len(collected) / collection if collection else None,
            "reset": percentiles([row["reset_seconds"] for row in rows if row.get("reset_seconds") is not None]),
            "step_seconds": sum(row["step"]["seconds"] for row in rows),
            "controller_seconds": sum(row["controller_time"]["seconds"] for row in rows),
            "finalize_seconds": sum(row["finalize"]["seconds"] for row in rows),
            "engine_wait_seconds": sum(row.get("engine_wait_seconds", 0) for row in rows),
            "adapter_seconds": dict(sum((Counter(row.get("adapter_seconds", {})) for row in rows), Counter())),
            "replay_seconds": sum(row.get("replay_seconds", 0) for row in rows),
            "audit_seconds": sum(row.get("audit_seconds", 0) for row in rows),
            "action_mix": dict(sorted(actions.items())), "result_mix": dict(sorted(results.items())),
            "results_rejected": sum(row.get("history", {}).get("results_rejected", 0) for row in rows),
            "visible_events": dict(sorted(events.items())), "native_timings": dict(native) if native else None,
            "rss_trajectory": [row["resources_after"].get("resident_bytes") for row in rows],
            "thread_trajectory": [row["resources_after"].get("threads") for row in rows],
            "descriptor_trajectory": [row["resources_after"].get("open_descriptors") for row in rows]}


def compare(args):
    """Matched rows: outcome agreement and the identical own-history prefix."""
    left, right = (strict_json(Path(path, "report.json").read_bytes()) for path in (args.left, args.right))
    rows = []
    for a, b in zip(left["rows"], right["rows"]):
        require(all(a[key] == b[key] for key in ("episode", "learner_player", "engine_seed", "policy_seed", "controller")),
                "Rows are not matched")
        entry = {key: a[key] for key in ("episode", "learner_player", "engine_seed", "controller")}
        entry.update(left_status=a.get("episode_status"), right_status=b.get("episode_status"),
                     outcome_equal=a.get("outcomes") == b.get("outcomes"))
        try:
            histories = []
            for row in (a, b):
                folder = Path(row["episode_directory"])
                histories.append(canonical_history(folder, strict_json((folder / "manifest.json").read_bytes())))
            prefix = 0
            for x, y in zip(*histories):
                if x != y:
                    break
                prefix += 1
            entry.update(left_frames=len(histories[0]), right_frames=len(histories[1]), identical_prefix_frames=prefix,
                         identical=histories[0] == histories[1])
            if prefix < min(map(len, histories)):
                entry["first_difference"] = [json.loads(h[prefix]).get("kind") for h in histories]
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        rows.append(entry)
    result = {"schema": SCHEMA + "-comparison", "left": binding(Path(args.left, "report.json")),
              "right": binding(Path(args.right, "report.json")), "rows": rows,
              "note": "A shared prefix shows equal learner observations under equal inputs; autonomous AI threads "
                      "make later divergence possible and do not by themselves indicate a rule difference."}
    write(args.output, result)
    print(json.dumps(rows, indent=2))
    return result


def codec(args):
    """Encode/decode cost of recorded player messages: socket line codec vs direct conversion."""
    import importlib.util
    from experiments.headless.contextual_protocol import loads
    binary = args.binary.expanduser().resolve()
    path = embedded.module_path(binary)
    spec = importlib.util.spec_from_file_location(embedded.MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    texts = []
    for ledger in sorted(Path(args.episode).expanduser().resolve().glob("agents/*/bridge.jsonl")):
        for line in ledger.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("direction") == "engine":
                texts.append(json.dumps(entry["message"], separators=(",", ":"), ensure_ascii=False))
    require(texts, "No recorded engine messages")
    native = module.codec_seconds(texts, args.repeat)
    encoded = [text.encode() for text in texts]
    begun = time.perf_counter()
    for _ in range(args.repeat):
        for text in encoded:
            loads(text)
    python = time.perf_counter() - begun
    result = {**native, "python_loads_seconds": python, "per_message_us": {
        "socket_encode": 1e6 * native["jsonwriter_seconds"] / (native["messages"] * args.repeat),
        "socket_python_decode": 1e6 * python / (native["messages"] * args.repeat),
        "embedded_direct": 1e6 * native["to_python_seconds"] / (native["messages"] * args.repeat)},
        "note": "Socket transfer itself is not included; the cold reader thread decodes concurrently with the engine."}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        write(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--backend", choices=BACKENDS, required=True)
    runner.add_argument("--controller", choices=CONTROLLERS, required=True)
    runner.add_argument("--binary", type=Path, required=True, help="vcmiclient in the prepared build/bin")
    runner.add_argument("--replay-binary", type=Path, required=True)
    runner.add_argument("--data", type=Path, required=True)
    runner.add_argument("--map", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--checkpoint", type=Path)
    runner.add_argument("--episodes", type=int, default=2)
    runner.add_argument("--seats", default="0,1")
    runner.add_argument("--engine-seed", type=int, default=121902)
    runner.add_argument("--policy-seed", type=int, default=121903)
    runner.add_argument("--max-days", type=int, default=12)
    runner.add_argument("--choices", type=int, default=256)
    runner.add_argument("--watchdog", type=float, default=60)
    runner.add_argument("--bridge-timeout", type=float, default=30)
    runner.add_argument("--row-seconds", type=float, default=300)
    runner.add_argument("--visit-kind", type=int)
    runner.add_argument("--stall-seconds", type=float, default=0.0)
    runner.add_argument("--replay", choices=("every", "first", "none"), default="every")
    matched = commands.add_parser("compare")
    matched.add_argument("--left", type=Path, required=True)
    matched.add_argument("--right", type=Path, required=True)
    matched.add_argument("--output", type=Path, required=True)
    measure = commands.add_parser("codec")
    measure.add_argument("--binary", type=Path, required=True)
    measure.add_argument("--episode", type=Path, required=True)
    measure.add_argument("--repeat", type=int, default=20)
    measure.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare, "codec": codec}[args.command](args)


if __name__ == "__main__":
    main()
