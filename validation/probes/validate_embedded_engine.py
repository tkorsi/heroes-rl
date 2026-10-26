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
first answer beyond the bridge timeout to prove failure poisoning;
``recorded`` resubmits, in order, the actions an earlier run of the same
schedule delivered, so the engine and adapter are timed without inference
(``compare`` then proves the replayed histories identical). None of them is a
learner, reward, demonstration or training input.

Times are per phase: reset through the first decision, engine transition
work, controller (policy inference) time, finalization, replay and audit.
They overlap where the backends overlap work (the cold reader thread decodes
while the engine runs), so they are reported as totals, not shares. A game
that ends inside step() or reset() finalizes inside that call; the nested
finalization is reported next to the totals instead of being subtracted.

A row passes only when its game ended naturally or by a declared cutoff, its
replay (where requested) verified and, for policy rows, its policy audit
(unless --audit none) held. The stall row passes only when the controller
actually withheld its answer, the game failed and, in-process, the engine was
poisoned; the run then also requires the next reset to be refused in Python
and natively. Rows reuse one environment per seat and controller, as
consecutive BuiltinCollector games do, unless --env-per-row is given.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.headless import embedded_evidence as embedded  # noqa: E402
from experiments.headless.contextual_evidence import file_identity, runtime_identity, strict_json  # noqa: E402
from experiments.learner import contextual as c  # noqa: E402
from validation.probes.validate_builtin_boundaries import binding, require, until, write  # noqa: E402

SCHEMA = "heroes-embedded-engine-validation-v1"
COLORS = ("red", "blue")
CONTROLLERS = ("policy", "end_turn", "visit", "stall", "recorded")
BACKENDS = ("cold", "embedded")
SCOPE = ("Bounded evaluation-only rows; no training, rewards, demonstrations or map changes. "
         "Controllers other than policy are explicit test workloads, never learner inputs.")
ENDED = ("completed", "truncated")  # manifest statuses of a game that did not fail
# Plan fields two runs must share before their rows count as matched.
MATCHED_PLAN = ("controller", "rows", "limits", "visit_kind", "stall_seconds", "random_policy_seed", "interface")
POLICY_PLAN = ("controller", "random_policy_seed")  # exempt when one run resubmits the other's actions
PLAN_DEFAULTS = {"interface": c.DIALOGS_ENVIRONMENT}  # plans written before the field existed
ABSOLUTE_PATH = re.compile(r"(?<![\w.~-])(?:/[^\s'\"<>|:;,()\[\]{}]+){2,}")


def schedule(count, seats, engine_seed, policy_seed, step=1):
    return [{"index": index, "episode": index + 1, "learner_player": seats[index % len(seats)],
             "engine_seed": engine_seed + step * index, "policy_seed": policy_seed + step * index}
            for index in range(count)]


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

    def add(self, elapsed):
        self.seconds += elapsed
        self.calls += 1
        self.samples.append(elapsed)

    def wrap(self, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(time.perf_counter() - started)
        return timed

    def summary(self):
        return {"seconds": self.seconds, "calls": self.calls, "latency": percentiles(self.samples)}


class Phases:
    """One row's reset/step/truncate/finalize times on a possibly reused environment."""

    def __init__(self, env):
        self.games = 0
        self.new_row()
        reset, step, truncate, finish = env.reset, env.step, env.truncate, env._finish
        env.reset = self._outer("reset", reset)
        env.step = self._outer("step", step)
        env.truncate = self._outer("truncate", truncate)

        def timed_finish(*args, **kwargs):
            started = time.perf_counter()
            try:
                return finish(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.timers["finalize"].add(elapsed)
                if self.inside is not None:
                    self.nested[self.inside] += elapsed
        env._finish = timed_finish

    def new_row(self):
        self.timers = {name: Timed() for name in ("reset", "step", "truncate", "finalize")}
        self.nested, self.inside = Counter(), None

    def _outer(self, name, function):
        def timed(*args, **kwargs):
            outer, self.inside = self.inside, self.inside or name
            try:
                return self.timers[name].wrap(function)(*args, **kwargs)
            finally:
                self.inside = outer
        return timed

    def summary(self):
        finalize = self.timers["finalize"].summary()
        finalize["nested_seconds"] = dict(self.nested)  # already inside the reset/step/truncate totals
        return {"step": self.timers["step"].summary(), "truncate": self.timers["truncate"].summary(),
                "finalize": finalize, "reset_samples": list(self.timers["reset"].samples)}


class ScriptedRunner:
    """BuiltinCollector.run_episode semantics for explicit test controllers."""

    def __init__(self, kind, *, learner_player, visit_kind=None, stall_seconds=0.0, recorded=None,
                 interface=c.DIALOGS_ENVIRONMENT):
        self.kind, self.learner_player = kind, learner_player
        self.schema = c.profile_for(interface).action_schema
        self.visit_kind, self.stall_seconds = visit_kind, stall_seconds
        self.recorded = deque(recorded or ())
        self.session = None  # retain_failed_builtin_evidence keeps no policy evidence for scripts
        self.dialog_answers, self.visit_moves, self.stalled = [], 0, False

    def act(self, request):
        frame = c.decode_frame(request.frames[-1])
        require(frame.kind == "decision" and frame.player == self.learner_player, "Scripted controller needs an own decision")
        observation, offers = frame.observation, frame.offers
        kind = observation.context.kind
        if self.kind == "recorded":
            require(self.recorded, "Recorded actions ran out before the game ended: the replayed game diverged")
            # The recorded schema is kept: offers of another interface refuse it.
            return c.Action(**self.recorded.popleft())
        if self.kind == "stall" and not self.stalled:
            self.stalled = True
            time.sleep(self.stall_seconds)
        if kind == "dialog":
            choice = offers.dialog_panel.choices[0]
            self.dialog_answers.append({"day": frame.day, "selection": choice.ref})
            return c.Action("choose_dialog", subject=offers.context.subject, selection=choice.ref,
                            schema=self.schema)
        panel = offers.capability_panel
        if kind == "capability" and panel is not None and panel.mandatory:
            # v8 skill or spell-target query: the first displayed choice, as for dialogs.
            choice = panel.choices[0]
            self.dialog_answers.append({"day": frame.day, "selection": choice.ref, "capability": panel.kind})
            return c.Action(choice.kind, subject=choice.subject, selection=choice.ref, schema=self.schema)
        require(kind == "map", "Scripted workload reached an unsupported window; no fabricated action")
        if self.kind == "visit" and not self.dialog_answers and self.visit_moves < 2 * frame.day:
            targets = [obj for obj in observation.objects if obj.kind == self.visit_kind and obj.owner != "self"]
            heroes = [hero for hero in observation.heroes if hero.handle in offers.movable_hero_refs and hero.movement > 0]
            if targets and heroes:
                target = targets[0].position
                hero = min(heroes, key=lambda h: max(abs(h.position[0] - target[0]), abs(h.position[1] - target[1])))
                self.visit_moves += 1
                return c.Action("move_to", subject=hero.handle, x=target[0], y=target[1], schema=self.schema)
        return c.Action("end_turn", schema=self.schema)

    def run_episode(self, env, *, episode, seed, max_requests):
        request, choices = env.reset(episode=episode, seed=seed), 0
        while not request.done:
            if choices >= max_requests:
                if env.sides[env.learner_player].validator.live.requires_choice:
                    env.defer_dialog_cutoff("collection_window")
                    request = env.step(self.act(request))
                    choices += 1
                else:
                    request = env.truncate("collection_window")
            else:
                request = env.step(self.act(request))
                choices += 1
        return request


def recorded_actions(run, row):
    """The actions the learner seat delivered in the matched row of an earlier run."""
    folder = run / f"episode-{row['episode']:06d}"
    manifest = strict_json((folder / "manifest.json").read_bytes())
    require(manifest["learner_player"] == row["learner_player"] and manifest["seed"] == row["engine_seed"],
            "Recorded row differs in seat or engine seed")
    color = ("red", "blue")[row["learner_player"]]
    actions = []
    # The schema travels with each line; run() requires the source run's interface.
    for line in (folder / "agents" / color / "bridge.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["direction"] == "policy":
            require(entry["message"]["delivered"] is True, "Recorded run has an undelivered response")
            actions.append(entry["message"]["action"])
    require(actions, "Recorded row has no delivered actions")
    return actions


def history_declaration(manifest):
    from experiments.headless.builtin_evidence import manifest_profile
    return manifest["players"][COLORS[manifest["learner_player"]]][manifest_profile(manifest).history_key]


def history_summary(folder, manifest):
    """Action mix, results and visible events from the persisted own history."""
    declaration = history_declaration(manifest)
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
    return (folder / history_declaration(manifest)["path"]).read_bytes().splitlines()


def delta(before, after):
    return {key: after[key] - before[key] for key in after if isinstance(after[key], (int, float))}


def random_policy(seed, interface=c.DIALOGS_ENVIRONMENT):
    """A labelled, freshly initialized model: never a trained or the frozen checkpoint.

    Used where the frozen checkpoint cannot load (its exact Torch release is
    bound); it exercises the same model, sampling and engine paths.
    """
    from types import SimpleNamespace
    import torch
    from experiments.learner import contextual_model as m
    torch.manual_seed(seed)
    profile = c.profile_for(interface)
    model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(profile.history_schema)))
    model.eval().requires_grad_(False)
    from experiments.learner.builtin_rollout import weights_sha256
    label = f"random-initialization-v{profile.version}-seed-{seed}-not-a-trained-or-frozen-checkpoint"
    return SimpleNamespace(model=model, checkpoint=label), {"label": label, "weights_sha256": weights_sha256(model),
                                                             "model_config": model.config.to_dict()}


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
    require(args.controller != "policy" or (args.checkpoint is None) != (args.random_policy_seed is None),
            "The policy controller needs exactly one of --checkpoint or --random-policy-seed")
    require(args.controller != "visit" or args.visit_kind is not None, "The visit controller needs --visit-kind")
    require((args.controller == "recorded") == (args.recorded is not None), "The recorded controller needs --recorded <run>")
    require(args.controller != "stall" or args.stall_seconds > args.bridge_timeout,
            "The stall controller must withhold its first answer beyond --bridge-timeout")
    seats = [int(seat) for seat in args.seats.split(",")]
    require(seats and all(seat in (0, 1) for seat in seats), "Seats are 0 and/or 1")
    require(args.episodes >= 1, "--episodes must be at least 1")
    rows = schedule(args.episodes, seats, args.engine_seed, args.policy_seed, args.seed_step)
    require(all(1 <= row["engine_seed"] <= 2147483647 for row in rows), "Engine seeds must be signed 32-bit and positive")
    if args.recorded is not None:
        args.recorded = args.recorded.expanduser().resolve()
        source_plan = strict_json((args.recorded / "plan.json").read_bytes())
        source = source_plan["rows"]
        require(all(any(all(row[key] == other[key] for key in ("episode", "learner_player", "engine_seed")) for other in source)
                    for row in rows), "The recorded run used a different episode, seat or engine-seed schedule")
        require(plan_value(source_plan, "interface") == args.interface, "The recorded run used another interface")
    runtime = runtime_identity(args.binary.parent)
    loaded = fixed = None
    if args.controller == "policy":
        loaded, fixed = (load_fixed(args.checkpoint) if args.checkpoint
                         else random_policy(args.random_policy_seed, args.interface))
    output.mkdir(parents=True)
    reserve = getattr(args, "dialog_reserve", 8)
    require(type(reserve) is int and reserve >= 0, "Dialog reserve must be a nonnegative integer")
    limits = Limits(max_days=args.max_days, max_actions=args.choices + reserve, startup_seconds=60,
                    watchdog_seconds=args.watchdog, shutdown_seconds=15, bridge_timeout=args.bridge_timeout)
    plan = {"schema": SCHEMA, "backend": args.backend, "controller": args.controller, "rows": rows,
            "interface": args.interface,
            "map": binding(args.map), "binary": str(args.binary), "replay_binary": str(args.replay_binary),
            "data": str(args.data), "runtime": runtime, "checkpoint": binding(args.checkpoint) if args.checkpoint else None,
            "random_policy_seed": args.random_policy_seed,
            "frozen": fixed, "limits": {"max_days": args.max_days, "choices": args.choices, "hard_choices": args.choices + reserve,
            "watchdog_seconds": args.watchdog, "bridge_timeout": args.bridge_timeout, "row_seconds": args.row_seconds},
            "visit_kind": args.visit_kind, "stall_seconds": args.stall_seconds, "replay": args.replay,
            "audit": args.audit,
            "reuse_environment": not args.env_per_row,
            "recorded": {"run": args.recorded.name, "report_sha256": file_identity(args.recorded / "report.json")["sha256"]}
                        if args.recorded is not None else None,
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
    environments = {}  # (seat, controller) -> (env, phases): consecutive games on one environment

    def environment(seat, controller):
        key = (seat, controller)
        if key not in environments or args.env_per_row:
            agents = {COLORS[seat]: {"kind": f"embedded-validation-{controller}", "evaluation_only": True}}
            options = dict(learner_player=seat, limits=limits, agents=agents, interface=args.interface)
            if args.backend == "embedded":
                from experiments.headless.embedded_env import EmbeddedBuiltinEnv
                env = EmbeddedBuiltinEnv(launcher, output, engine=engine, **options)
            else:
                env = BuiltinHeadlessEnv(launcher, output, **options)
            environments[key] = (env, Phases(env))
        env, phases = environments[key]
        phases.new_row()
        phases.games += 1
        return env, phases

    def close_environment(env, result):
        try:
            env.close()
        except Exception as error:
            result["close_error"] = f"{type(error).__name__}: {error}"

    for row in rows:
        seat = row["learner_player"]
        stall = args.controller == "stall" and row["index"] == 0
        controller = args.controller if args.controller != "stall" or stall else "end_turn"
        result = {**row, "controller": controller, "status": "failed"}
        env = runner = phases = previous = None
        policy = Timed()
        native_before = engine.timings() if engine else None
        row_started = time.perf_counter()
        try:
            with until(time.monotonic() + args.row_seconds, time.time() + args.row_seconds, 0):
                env, phases = environment(seat, controller)
                previous = env.episode_dir
                result["environment_game"] = phases.games
                if controller == "policy":
                    opponent = native_opponent(runtime, seat)

                    class TimedCollector(BuiltinCollector):
                        act = policy.wrap(BuiltinCollector.act)
                    runner = TimedCollector(loaded.model, loaded.checkpoint, learner_player=seat,
                                            opponent_identity=opponent, policy_seed=row["policy_seed"],
                                            interface=args.interface)
                    collected = runner.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                                   max_requests=args.choices)
                    write(env.episode_dir / "builtin-rollout.json", collected.to_dict())
                    manifest = collected.manifest
                else:
                    runner = ScriptedRunner(controller, learner_player=seat, visit_kind=args.visit_kind,
                                            stall_seconds=args.stall_seconds, interface=args.interface,
                                            recorded=recorded_actions(args.recorded, row) if args.recorded else None)
                    runner.act = policy.wrap(runner.act)
                    try:
                        end = runner.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                                 max_requests=args.choices)
                    finally:
                        result["stalled"] = runner.stalled
                    manifest = end.manifest
                    result["dialog_answers"] = runner.dialog_answers
                result.update(status="collected", episode_status=manifest["status"],
                              termination_reason=manifest["termination_reason"],
                              outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"])
        except Exception as error:
            result.update(error=f"{type(error).__name__}: {error}")
            if env is not None and runner is not None:
                retain_failed_builtin_evidence(env, runner if controller == "policy" else None, result, error,
                                               cleanup_seconds=30)
        collected_at = time.perf_counter()
        result["collection_seconds"] = collected_at - row_started
        result["controller_time"] = policy.summary()
        if phases is not None:
            timing = phases.summary()
            samples = timing.pop("reset_samples")
            result.update(timing, reset_seconds=samples[0] if samples else None)
        if env is not None:
            folder = env.episode_dir
            # Only a game this row started owns the environment's metrics and episode directory; a reset that
            # failed earlier leaves the previous row's (retain_failed_builtin_evidence would name its folder).
            started = folder is not None and folder != previous and folder.name == f"episode-{row['episode']:06d}"
            if not started:
                for key in ("episode_directory", "episode_status", "termination_reason", "failure_finalization"):
                    result.pop(key, None)
            if started:
                result.update(engine_wait_seconds=env.engine_wait, agent_wait_seconds=env.agent_wait,
                              adapter_seconds=dict(env.costs), step_latency=percentiles(env.step_latencies))
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
            if args.env_per_row or result["status"] != "collected":
                close_environment(env, result)
                environments.pop((seat, controller), None)
        if engine is not None:
            result["native_timings"] = delta(native_before, engine.timings())
            result["poisoned_after"] = engine.poisoned
        result["resources_collected"] = host_resources(engine)
        ended = result.get("episode_status") in ENDED
        verify = not stall and ended and (args.replay == "every" or (args.replay == "first" and row["index"] == 0))
        if result["status"] == "collected" and verify:
            begun = time.perf_counter()
            replay = verify_episode(Path(result["episode_directory"]), binary=args.replay_binary, data=args.data,
                                    timeout=args.row_seconds)
            result["replay_seconds"] = time.perf_counter() - begun
            result["replay_verified"] = replay.get("verified") is True
            result["replay_error"] = replay.get("error")
            if replay.get("embedded_backend") or replay.get("identity_check", {}).get("embedded_backend"):
                result["replay_embedded_identity"] = True
        # The audit reads only the persisted rollout, manifest and history; it does not depend on the replay.
        audit = controller == "policy" and ended and args.audit == "every"
        if result["status"] == "collected" and audit:
            begun = time.perf_counter()
            try:
                rebuilt, _ = (load_fixed(args.checkpoint) if args.checkpoint
                              else random_policy(args.random_policy_seed, args.interface))
                audited = audit_persisted_builtin_policy(Path(result["episode_directory"]), rebuilt.model,
                    rebuilt.checkpoint, row["policy_seed"], native_opponent(runtime, seat))
                result["policy_audit"] = audited.get("passed") is True
            except Exception as error:
                result["policy_audit"] = False
                result["policy_audit_error"] = f"{type(error).__name__}: {error}"
            result["audit_seconds"] = time.perf_counter() - begun
        if stall:
            result["expected"] = "failed"
        result["status"] = row_status(result, stall=stall, verify=verify, audit=audit, embedded=engine is not None)
        result["natural"] = result.get("episode_status") == "completed"
        result["replay_requested"], result["audit_requested"] = verify, audit
        result["row_seconds"] = time.perf_counter() - row_started
        result["resources_after"] = host_resources(engine)
        report["rows"].append(result)
        write(output / "report.json", report)
        if engine is not None and engine.poisoned:
            report["poisoned"] = refusal_check(engine, launcher, output, row, limits)
            break
        if getattr(args, "stop_on_failure", False) and result["status"] != "passed":
            break
    report["unstarted_rows"] = rows[len(report["rows"]):]
    for env, _ in environments.values():
        close_environment(env, report.setdefault("final_close", {}))
    report["summary"] = summarize(report["rows"], args.backend)
    report["checks"] = run_checks(args, report, engine)
    report["verdict"] = ("passed" if all(report["checks"].values()) and report["summary"]["passed"] == len(report["rows"])
                         else "failed")
    report["status"] = "finished"
    report["process_seconds"] = time.monotonic() - started_process
    report["resources_end"] = host_resources(engine)
    if args.checkpoint is not None:
        report["checkpoint_after"] = binding(args.checkpoint)
        require(report["checkpoint_after"] == plan["checkpoint"], "Checkpoint changed during validation")
    write(output / "report.json", report)
    print(json.dumps({"verdict": report["verdict"], "checks": report["checks"], **report["summary"]}, indent=2, sort_keys=True))
    return report


def row_status(result, *, stall, verify, audit, embedded):
    """Passed only for a game that ended as intended and whose requested evidence held."""
    if stall:
        # The answer must actually have been withheld and must fail the game; in-process it must poison the engine.
        return ("passed" if result.get("stalled") is True and result.get("episode_status") == "failed"
                and (not embedded or result.get("poisoned_after") is True) else "failed")
    if result["status"] != "collected" or result.get("episode_status") not in ENDED:
        return "failed"
    return ("passed" if (not verify or result.get("replay_verified") is True)
            and (not audit or result.get("policy_audit") is True) else "failed")


def refusal_check(engine, launcher, output, row, limits):
    """A poisoned engine must refuse the next game both in the environment and natively."""
    from experiments.headless.embedded_env import EmbeddedBuiltinEnv
    folder = output / "refusal-check"
    result = {"after_episode": row["episode"], "environment_refusal": None, "native_refusal": None}
    try:
        env = EmbeddedBuiltinEnv(launcher, folder, engine=engine, learner_player=row["learner_player"], limits=limits,
                                 agents={COLORS[row["learner_player"]]: {"kind": "embedded-validation-refusal",
                                                                         "evaluation_only": True}})
        env.reset(episode=row["episode"] + 1, seed=row["engine_seed"] + 1)
        result["environment_started_game"] = True
    except engine.errors as error:
        result["environment_refusal"] = f"{type(error).__name__}: {error}"
    try:
        # Bypasses every Python precheck: SessionHealth itself must refuse before any mutation.
        engine.engine.reset(episode="refusal-check", map="Maps/headless" + launcher.game_map.suffix.lower(),
                            capture_directory=str(folder / "native-capture"), seed=1, learner=row["learner_player"],
                            policy_timeout=limits.bridge_timeout, log_path="")
        result["native_started_game"] = True
        engine.engine.finish(natural=False, timeout=30)
    except engine.errors as error:
        result["native_refusal"] = f"{type(error).__name__}: {error}"
    result["next_reset_refused"] = (result["environment_refusal"] is not None and result["native_refusal"] is not None
                                    and not (folder / "native-capture").exists())
    return result


def run_checks(args, report, engine):
    rows = report["rows"]
    checks = {"all_rows_ran": len(rows) == len(report_plan_rows(args)) or args.controller == "stall"}
    if args.controller == "stall":
        checks["stall_row_failed"] = bool(rows) and rows[0].get("stalled") is True and rows[0].get("episode_status") == "failed"
        if engine is not None:
            checks["poisoned_and_refused"] = (report.get("poisoned") or {}).get("next_reset_refused") is True
            checks["stopped_after_poisoning"] = len(rows) == 1
        else:
            checks["cold_recovered"] = len(rows) > 1 and all(row["status"] == "passed" for row in rows[1:])
    elif engine is not None:
        checks["never_poisoned"] = "poisoned" not in report
    return checks


def report_plan_rows(args):
    return schedule(args.episodes, [int(seat) for seat in args.seats.split(",")], args.engine_seed, args.policy_seed,
                    args.seed_step)


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
    # Only games that passed count as completed throughput.
    natural = [row for row in rows if row["status"] == "passed" and row.get("episode_status") == "completed"]
    native, nested = Counter(), Counter()
    for row in rows:
        native.update({key: value for key, value in row.get("native_timings", {}).items()})
        nested.update(row.get("finalize", {}).get("nested_seconds", {}))
    phase = lambda name: sum(row.get(name, {}).get("seconds", 0) for row in rows)
    return {"backend": backend, "rows": len(rows), "passed": sum(row["status"] == "passed" for row in rows),
            "episode_statuses": {f"{s}/{t}": n for (s, t), n in sorted(statuses.items(), key=str)},
            "natural_completed": len(natural), "replay_verified": sum(bool(row.get("replay_verified")) for row in rows),
            "replay_requested": sum(bool(row.get("replay_requested")) for row in rows),
            "collection_seconds": collection, "submitted_decisions": submitted,
            "decisions_per_second": submitted / collection if collection else None,
            "completed_games_per_second": len(natural) / collection if collection else None,
            "rows_per_second": len(collected) / collection if collection else None,
            "reset": percentiles([row["reset_seconds"] for row in rows if row.get("reset_seconds") is not None]),
            "step_seconds": phase("step"), "truncate_seconds": phase("truncate"),
            "controller_seconds": phase("controller_time"), "finalize_seconds": phase("finalize"),
            "finalize_nested_seconds": dict(nested),
            "engine_wait_seconds": sum(row.get("engine_wait_seconds", 0) for row in rows),
            "adapter_seconds": dict(sum((Counter(row.get("adapter_seconds", {})) for row in rows), Counter())),
            "replay_seconds": sum(row.get("replay_seconds", 0) for row in rows),
            "audit_seconds": sum(row.get("audit_seconds", 0) for row in rows),
            "action_mix": dict(sorted(actions.items())), "result_mix": dict(sorted(results.items())),
            "results_rejected": sum(row.get("history", {}).get("results_rejected", 0) for row in rows),
            "visible_events": dict(sorted(events.items())), "native_timings": dict(native) if native else None,
            "rss_trajectory": [row["resources_collected"].get("resident_bytes") for row in rows],
            "thread_trajectory": [row["resources_collected"].get("threads") for row in rows],
            "descriptor_trajectory": [row["resources_collected"].get("open_descriptors") for row in rows]}


def plan_value(plan, key):
    return plan.get(key, PLAN_DEFAULTS.get(key))


def run_identity(folder):
    """A run by name and content, never by its absolute location."""
    return {"run": folder.name, "report_sha256": file_identity(folder / "report.json")["sha256"],
            "plan_sha256": file_identity(folder / "plan.json")["sha256"]}


def compare(args):
    """Matched rows of equally configured runs: outcome agreement and the identical own history."""
    folders = [Path(path).expanduser().resolve() for path in (args.left, args.right)]
    left, right = (strict_json((folder / "report.json").read_bytes()) for folder in folders)
    plans = [strict_json((folder / "plan.json").read_bytes()) for folder in folders]
    identities = [run_identity(folder) for folder in folders]
    for plan, report, folder in zip(plans, (left, right), folders):
        require(report["plan"]["sha256"] == file_identity(folder / "plan.json")["sha256"], "Report does not bind its plan")
    # A recorded run resubmits exactly the other run's delivered actions; only its policy fields differ.
    recorded = any((plan.get("recorded") or {}) == {"run": other["run"], "report_sha256": other["report_sha256"]}
                   for plan, other in ((plans[1], identities[0]), (plans[0], identities[1])))
    keys = [key for key in MATCHED_PLAN if not (recorded and key in POLICY_PLAN)]
    differing = [key for key in keys if plan_value(plans[0], key) != plan_value(plans[1], key)]
    differing += [] if plans[0]["map"]["sha256"] == plans[1]["map"]["sha256"] else ["map"]
    if not recorded and (plans[0].get("checkpoint") or {}).get("sha256") != (plans[1].get("checkpoint") or {}).get("sha256"):
        differing.append("checkpoint")
    require(not differing, f"Runs are not equally configured: {', '.join(differing)}")
    rows = []
    for a, b in zip(left["rows"], right["rows"]):
        require(all(a[key] == b[key] for key in ("episode", "learner_player", "engine_seed")), "Rows are not matched")
        entry = {key: a[key] for key in ("episode", "learner_player", "engine_seed")}
        entry.update(controllers=[a["controller"], b["controller"]], left_status=a["status"], right_status=b["status"],
                     left_episode_status=a.get("episode_status"), right_episode_status=b.get("episode_status"),
                     outcome_equal=all(a.get(key) == b.get(key) for key in ("episode_status", "termination_reason", "outcomes")))
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
            entry["error"] = scrub(f"{type(error).__name__}: {error}", local_roots(*folders))
        rows.append(entry)
    matched = min(len(left["rows"]), len(right["rows"]))
    result = {"schema": SCHEMA + "-comparison", "left": identities[0], "right": identities[1],
              "backends": [plan["backend"] for plan in plans], "recorded": recorded, "rows": rows,
              "unmatched_rows": {"left": [row["episode"] for row in left["rows"][matched:]],
                                 "right": [row["episode"] for row in right["rows"][matched:]]},
              "all_identical": bool(rows) and len(left["rows"]) == len(right["rows"])
                               and all(row.get("identical") is True and row["outcome_equal"] for row in rows),
              "runtime_equal": plans[0]["runtime"] == plans[1]["runtime"],
              "note": "A shared prefix shows equal learner observations under equal inputs; autonomous AI threads "
                      "make later divergence possible and do not by themselves indicate a rule difference."}
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ("all_identical", "unmatched_rows", "recorded", "runtime_equal")}))
    return result


def codec(args):
    """Encode/decode cost of recorded player messages: socket line codec vs direct conversion."""
    import importlib.util
    from experiments.headless.contextual_protocol import loads
    binary = args.binary.expanduser().resolve()
    path = embedded.module_path(binary.parent if binary.is_file() else binary)
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
    # What the cold reader receives: VCMI's JsonWriter text with its line breaks removed.
    lines = [module.parsed_text(text).replace("\n", "").replace("\r", "").encode() for text in texts]
    require(sum(len(line) + 1 for line in lines) == native["wire_bytes"], "Socket lines differ from the measured encoding")
    begun = time.perf_counter()
    for _ in range(args.repeat):
        for line in lines:
            loads(line)
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


ROW_FIELDS = ("episode", "learner_player", "engine_seed", "controller", "status", "expected", "episode_status",
              "termination_reason", "outcomes", "error", "reset_seconds", "collection_seconds", "row_seconds",
              "replay_seconds", "audit_seconds", "replay_requested", "replay_verified", "replay_error", "policy_audit",
              "audit_requested", "returncode", "poisoned_after", "dialog_answers", "environment_game", "stalled")
RESOURCE_FIELDS = ("resident_bytes", "threads", "open_descriptors", "python_threads", "peak_self_bytes",
                   "peak_children_bytes")


def local_roots(*paths):
    """Known machine locations, longest first; they may contain spaces the path pattern stops at."""
    roots = {str(Path.home()), str(ROOT)}
    for path in paths:
        if path:
            path = Path(path)
            roots.update(str(part) for part in (path, *path.parents) if len(part.parts) > 2)
    return tuple(sorted(roots, key=len, reverse=True))


def scrub(value, roots=()):
    """Replace absolute paths in free text; the record names runs, never machine locations."""
    if isinstance(value, str):
        # Error messages quote filenames that may contain spaces. Remove the
        # whole quoted absolute path before the unquoted-token fallback.
        value = re.sub(r"(['\"])/[^\r\n]*?\1", lambda match: match[1] + "<path>" + match[1], value)
        for root in roots:
            value = value.replace(root, "<path>")
        return ABSOLUTE_PATH.sub("<path>", value)
    if isinstance(value, list):
        return [scrub(item, roots) for item in value]
    if isinstance(value, dict):
        return {scrub(key, roots): scrub(item, roots) for key, item in value.items()}
    return value


def compact_row(row):
    result = {key: row[key] for key in ROW_FIELDS if key in row}
    history = row.get("history", {})
    result.update(submitted=history.get("submitted"), last_day=history.get("last_day"),
                  action_mix=history.get("action_mix"), results_rejected=history.get("results_rejected"),
                  step_seconds=row.get("step", {}).get("seconds"), controller_seconds=row.get("controller_time", {}).get("seconds"),
                  finalize_seconds=row.get("finalize", {}).get("seconds"),
                  finalize_nested_seconds=row.get("finalize", {}).get("nested_seconds"),
                  engine_wait_seconds=row.get("engine_wait_seconds"), native_timings=row.get("native_timings"),
                  step_latency=row.get("step_latency"), close_clean=(row.get("native_close") or {}).get("close_clean"),
                  # Children peaks are maxima over every waited child so far: game processes and replay verifiers.
                  resources_collected={key: row.get("resources_collected", {}).get(key) for key in RESOURCE_FIELDS},
                  resources_after={key: row.get("resources_after", {}).get(key) for key in RESOURCE_FIELDS})
    return result


def collect(args):
    """One compact, path-free JSON of several run reports for the repository record."""
    runs, locations = [], []
    for folder in args.runs:
        folder = Path(folder).expanduser().resolve()
        report, plan = (strict_json((folder / name).read_bytes()) for name in ("report.json", "plan.json"))
        locations += [folder, *(plan.get(key) for key in ("binary", "replay_binary", "data"))]
        runs.append({**run_identity(folder), "backend": plan["backend"], "controller": plan["controller"],
                     "map_sha256": plan["map"]["sha256"], "checkpoint_sha256": (plan.get("checkpoint") or {}).get("sha256"),
                     "random_policy_seed": plan.get("random_policy_seed"), "frozen": plan.get("frozen"),
                     "interface": plan_value(plan, "interface"),
                     "limits": plan["limits"], "platform": plan["platform"], "rows_planned": len(plan["rows"]),
                     "reuse_environment": plan.get("reuse_environment"), "recorded": plan.get("recorded"),
                     "status": report["status"], "verdict": report.get("verdict"), "checks": report.get("checks"),
                     "engine_load_seconds": report.get("engine_load_seconds"),
                     "process_seconds": report.get("process_seconds"), "poisoned": report.get("poisoned"),
                     "resources_start": report.get("resources_start"), "resources_end": report.get("resources_end"),
                     "summary": report.get("summary"), "rows": [compact_row(row) for row in report["rows"]]})
    extra = {}
    for path in args.attach or ():
        path = Path(path).expanduser().resolve()
        locations.append(path)
        extra[path.name] = strict_json(path.read_bytes())
    roots = local_roots(*locations)
    result = scrub({"schema": SCHEMA + "-compact", "runs": runs, "attached": extra, "scope": SCOPE}, roots)
    text = json.dumps(result)
    require(not ABSOLUTE_PATH.search(text) and not any(root in text for root in roots),
            "Compact record still names a local path")
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
    runner.add_argument("--random-policy-seed", type=int, help="labelled freshly initialized v7 policy instead of --checkpoint")
    runner.add_argument("--episodes", type=int, default=2)
    runner.add_argument("--seats", default="0,1")
    runner.add_argument("--engine-seed", type=int, default=121902)
    runner.add_argument("--policy-seed", type=int, default=121903)
    runner.add_argument("--seed-step", type=int, default=1, help="0 repeats the base seeds on every row")
    runner.add_argument("--max-days", type=int, default=12)
    runner.add_argument("--choices", type=int, default=256)
    runner.add_argument("--watchdog", type=float, default=60)
    runner.add_argument("--bridge-timeout", type=int, default=30, help="whole seconds, as EXPLORER_RL_TIMEOUT requires")
    runner.add_argument("--row-seconds", type=float, default=300)
    runner.add_argument("--visit-kind", type=int)
    runner.add_argument("--stall-seconds", type=float, default=0.0)
    runner.add_argument("--replay", choices=("every", "first", "none"), default="every")
    runner.add_argument("--audit", choices=("every", "none"), default="every",
                        help="reconstruct the frozen policy's choices of every ended policy row")
    runner.add_argument("--env-per-row", action="store_true", help="a new environment per game instead of one per seat")
    runner.add_argument("--interface", default=c.DIALOGS_ENVIRONMENT,
                        choices=(c.DIALOGS_ENVIRONMENT, *((c.EXPANDED_ENVIRONMENT,) if hasattr(c, "EXPANDED_ENVIRONMENT") else ())))
    runner.add_argument("--recorded", type=Path, help="earlier run directory whose delivered actions the recorded controller resubmits")
    matched = commands.add_parser("compare")
    matched.add_argument("--left", type=Path, required=True)
    matched.add_argument("--right", type=Path, required=True)
    matched.add_argument("--output", type=Path, required=True)
    measure = commands.add_parser("codec")
    measure.add_argument("--binary", type=Path, required=True)
    measure.add_argument("--episode", type=Path, required=True)
    measure.add_argument("--repeat", type=int, default=20)
    measure.add_argument("--output", type=Path)
    gather = commands.add_parser("collect")
    gather.add_argument("--runs", type=Path, nargs="+", required=True)
    gather.add_argument("--attach", type=Path, nargs="*", help="Extra small JSON results (comparisons, codec, benchmark)")
    gather.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare, "codec": codec, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
