"""Bounded, staged performance benchmark of the current contextual pipeline.

Performance cohorts only. Rows come from configs/performance-benchmark-v1.json,
with seeds disjoint from every scientific cohort. The frozen v7 checkpoint is
loaded on CPU with one Torch thread and is never optimized. Every scheduled row
is retained, including failures and rows not started; there are no retries or
replacements. A campaign has one absolute wall deadline that covers worker
start-up, collection, policy reconstruction, independent native replay,
offline behavior and cleanup.

Each attempt is timed per stage, with process CPU and child (engine/verifier)
CPU alongside wall time. Timing wrappers only measure: they never change
arguments, results or exceptions, and they are installed in this harness
process only. Campaigns run with --cprofile are diagnostic and must not be used
for speed claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/performance-benchmark-v1.json"
SCHEMA = "heroes-performance-benchmark-v1"
COLORS = ("red", "blue")
DEFAULT_MAP = ROOT / "experiments/explorer/maps/hero-management-v1.vmap"
# Heavy processes that would contaminate a clean timing sample when they do not
# belong to this campaign. Matching is on the full command line.
FOREIGN_PATTERN = re.compile(r"vcmiclient|vcmiserver|vcmireplay|ninja|clang|evaluate_|validate_|unittest")
CLEANUP_RESERVE_SECONDS = 20
CLEANUP_GRACE_SECONDS = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def write(path, value):
    """Atomic replacement; an interrupted worker never leaves half a JSON file."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temp.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def config():
    value = json.loads(CONFIG.read_text())
    require(value.get("schema") == "heroes-performance-benchmark-config-v1"
            and value.get("purpose") == "performance_measurement_only"
            and value.get("optimizer_updates") == 0 and value.get("training_performed") is False,
            "Unexpected performance benchmark configuration")
    return value


def rows(cfg, count):
    """Deterministic rows; seat swaps keep each policy seed with its model."""
    generator = cfg["row_generator"]
    require(type(count) is int and 1 <= count <= generator["max_rows"], "Row count outside the frozen generator")
    result = []
    for index in range(count):
        pair, seat = divmod(index, 2)
        seeds = [generator["policy_seed_base"] + 2 * pair, generator["policy_seed_base"] + 2 * pair + 1]
        result.append({"row": index, "episode": index + 1, "engine_seed": generator["engine_seed_base"] + pair,
                       "learner_player": seat, "policy_seeds": seeds if seat == 0 else seeds[::-1]})
    return result


def assignment(row_count, workers):
    """Static round-robin; every row has exactly one worker."""
    require(type(workers) is int and 1 <= workers <= 32, "Use 1 to 32 workers")
    return {worker: list(range(worker, row_count, workers)) for worker in range(workers)}


# -- measurement -------------------------------------------------------------

def _cpu_self():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _cpu_children():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


class StageTimer:
    """Inclusive wall time and call counts per key; nested re-entry counts once.

    Wrapped functions run on the environment's main thread, so no lock is taken.
    """

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self._depth = defaultdict(int)

    def timed(self, function, key):
        seconds, calls, depths = self.seconds, self.calls, self._depth

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            depth = depths[key]
            depths[key] = depth + 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                depths[key] = depth
                calls[key] += 1
                if depth == 0:
                    seconds[key] += elapsed
        wrapper.__perf_original__ = function
        return wrapper

    def wrap(self, owner, name, key):
        original = getattr(owner, name)
        require(callable(original) and not hasattr(original, "__perf_original__"), f"Cannot time {name}")
        setattr(owner, name, self.timed(original, key))

    def snapshot(self):
        return {"seconds": dict(self.seconds), "calls": dict(self.calls),
                "cpu_self": _cpu_self(), "cpu_children": _cpu_children(), "wall": time.perf_counter()}

    @staticmethod
    def delta(before, after):
        keys = set(before["seconds"]) | set(after["seconds"]) | set(before["calls"]) | set(after["calls"])
        stages = {}
        for key in sorted(keys):
            calls = after["calls"].get(key, 0) - before["calls"].get(key, 0)
            seconds = after["seconds"].get(key, 0.0) - before["seconds"].get(key, 0.0)
            if calls:
                stages[key] = {"seconds": round(seconds, 6), "calls": calls}
        return {"wall_seconds": round(after["wall"] - before["wall"], 6),
                "cpu_self_seconds": round(after["cpu_self"] - before["cpu_self"], 6),
                "cpu_children_seconds": round(after["cpu_children"] - before["cpu_children"], 6),
                "stages": stages}


def wrapper_overhead_seconds(calls=50000):
    """Per-call cost of one timing wrapper, for the reported overhead estimate."""
    probe = StageTimer()
    plain = (lambda: None)
    timed = probe.timed(plain, "probe")
    started = time.perf_counter()
    for _ in range(calls):
        timed()
    wrapped = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(calls):
        plain()
    return max(0.0, (wrapped - (time.perf_counter() - started)) / calls)


TIMER = StageTimer()


def instrument_modules():
    """Wrap hot functions once per worker process; measurement only."""
    from experiments.learner import contextual as contract
    from experiments.learner import contextual_model as model_module
    from experiments.learner import contextual_rollout as rollout
    from experiments.headless import contextual_replay
    import validation.probes.audit_contextual_histories as history_audit
    import validation.probes.validate_hero_policy as hero_policy
    wraps = [
        (rollout.ContextualSession, "consume", "policy.consume"),
        (rollout.ContextualSession, "choose", "policy.choose"),
        (rollout, "weights_sha256", "policy.weights_sha256"),
        (rollout, "_frame_bytes", "policy.frame_bytes"),
        (model_module.ContextualEncoder, "encode", "model.encode"),
        (model_module.ContextualActorCritic, "forward_step", "model.forward_step"),
        (model_module.ContextualDistribution, "sample", "model.sample"),
        (model_module.ContextualDistribution, "evaluate", "model.evaluate"),
        (contract, "decode_frame", "contract.decode_frame"),
        (contract, "encode_frame", "contract.encode_frame"),
        (contract.PlayerHistory, "append", "contract.history_append"),
        (rollout.ContextualCollector, "finish", "collector.finish"),
        (rollout.ContextualRollout, "__post_init__", "collector.rollout_validation"),
        (history_audit, "audit", "audit.histories"),
        (hero_policy, "reconstruct_behavior", "reconstruction.behavior"),
        (rollout.CollectedEpisode, "to_dict", "rollout.to_dict"),
        (contextual_replay, "run_native", "replay.native_process"),
        (contextual_replay, "bound_evidence", "replay.bound_evidence"),
        (contextual_replay, "cross_check_wire", "replay.cross_check_wire"),
    ]
    for owner, name, key in wraps:
        TIMER.wrap(owner, name, key)
    # Names imported into other modules must be wrapped where they are looked up.
    hero_policy.audit = history_audit.audit
    from experiments.headless import contextual_replay_history
    TIMER.wrap(contextual_replay_history, "verify_histories", "replay.verify_histories")


def instrument_environment(env, launcher):
    """Instance-level wrappers on this one environment object."""
    for name, key in (("reset", "env.reset"), ("step", "env.step"), ("truncate", "env.truncate"),
                      ("defer_dialog_cutoff", "env.defer_dialog_cutoff"), ("_finish", "env.finalize"),
                      ("_assert_sources_unchanged", "env.reset_source_check"),
                      ("_preflight_contextual", "env.reset_preflight"), ("_identity", "env.identity"),
                      ("_issue", "env.issue_request")):
        setattr(env, name, TIMER.timed(getattr(env, name), key))
    module = sys.modules[type(env).__mro__[1].__module__]
    if not hasattr(module.encoded, "__perf_original__"):
        TIMER.wrap(module, "encoded", "env.encode_request_frames")
        TIMER.wrap(module, "persist_history", "env.persist_history")
        TIMER.wrap(module.ContextualProducer, "ingest", "env.producer_ingest")
    if not hasattr(launcher.launch, "__perf_original__"):
        launcher.launch = TIMER.timed(launcher.launch, "env.launch_spawn")


def environment_counters(env):
    sides = list(getattr(env, "sides", {}).values())
    return {"engine_wait_seconds": round(getattr(env, "engine_wait", 0.0), 6),
            "agent_wait_seconds": round(getattr(env, "agent_wait", 0.0), 6),
            "adapter_costs_seconds": {key: round(value, 6) for key, value in getattr(env, "costs", {}).items()},
            "reader_decode_seconds": round(sum(side.decode_seconds for side in sides), 6),
            "bytes_received": sum(side.bytes_received for side in sides),
            "replay_tail_wait_seconds": round(getattr(env, "replay_tail_wait", 0.0), 6),
            "engine_steps_timed": len(getattr(env, "step_latencies", [])),
            "env_steps": getattr(env, "env_step", 0)}


def current_rss_bytes(pid=None):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) * 1024 if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def peak_rss_bytes(who):
    value = resource.getrusage(who).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


# -- worker --------------------------------------------------------------------

def load_fixed(path, cfg, interface):
    from experiments.learner.checkpoints import load_policy_checkpoint
    from validation.probes.validate_hero_policy import frozen_fingerprints
    require(digest(path) == cfg["checkpoint_sha256"], "Only the frozen v7 benchmark checkpoint is permitted")
    loaded = load_policy_checkpoint(path, interface=interface, allowed_provenance=("random_initialization",))
    loaded.model.eval().requires_grad_(False)
    fixed = frozen_fingerprints(loaded, path)
    require(fixed["learner_weights_sha256"] == cfg["learner_weights_sha256"]
            and fixed["opponents"] == [cfg["opponent"]], "Frozen policy identities differ")
    return loaded, fixed


def source_compatibility(loaded, interface):
    """Record, never hide, whether the checkpoint's learner bytes still match."""
    from experiments.learner.inference import source_identity
    current = source_identity(interface)
    saved = loaded.source_metadata
    changed = sorted(path for path, sha in saved["source_sha256"].items() if current["source_sha256"].get(path) != sha)
    runtime = [key for key in ("torch", "python", "interface") if saved[key] != current[key]]
    return {"compatible": not changed and not runtime, "changed_learner_files": changed,
            "runtime_mismatch": runtime, "git_commit": current["git_commit"],
            "tracked_worktree_dirty": current["tracked_worktree_dirty"]}


class Phases:
    def __init__(self):
        self.result = {}

    @contextmanager
    def __call__(self, name):
        before = TIMER.snapshot()
        try:
            yield
        finally:
            self.result[name] = StageTimer.delta(before, TIMER.snapshot())


def compact_native(report):
    if not isinstance(report, dict):
        return None
    native = report.get("native") if isinstance(report.get("native"), dict) else {}
    return {"verified": report.get("verified") is True, "error": report.get("error"),
            "native_seconds": native.get("seconds"), "native_returncode": native.get("returncode"),
            "native_counts": {key: value for key, value in native.items() if type(value) is int}}


def history_counts(collected):
    result = {}
    for rollout in collected.rollouts:
        kinds = Counter(frame.kind for frame in rollout.history)
        result[COLORS[rollout.player]] = {"frames": len(rollout.history), "kinds": dict(sorted(kinds.items())),
            "sampled_choices": len(rollout.decisions), "estimates": len(rollout.estimates),
            "last_day": max((frame.day for frame in rollout.history), default=None),
            "history_sha256": rollout.history_sha256, "boundary": rollout.boundary}
    return result


def real_operations():
    """The production pipeline; tests substitute fakes for native/model work."""
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.contextual_replay import verify_episode
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.evaluation_behavior import analyze_episode
    from validation.probes.validate_hero_policy import audit_persisted_policy, frozen_fingerprints, retain_failed_evidence
    return {"env_class": HeadlessSelfPlayEnv, "limits_class": Limits, "collector_class": ContextualCollector,
            "verify_episode": verify_episode, "analyze_episode": analyze_episode,
            "audit_persisted_policy": audit_persisted_policy, "frozen_fingerprints": frozen_fingerprints,
            "retain_failed_evidence": retain_failed_evidence, "load_fixed": load_fixed,
            "instrument_environment": instrument_environment}


# Only these end states let a worker continue to its next row.
CONTINUABLE = ("passed", "native_failure")


def run_row(context, row, output):
    """One attempt. It is passed only after every check and both cleanups.

    A start marker is persisted before any native work and updated at every
    phase, so a hard-killed attempt stays visibly incomplete. Any exception or
    cleanup error makes the attempt failed and unscored, and stops the worker.
    """
    ops = context["ops"]
    cfg, plan, launcher = context["cfg"], context["plan"], context["launcher"]
    loaded, opponent = context["loaded"], context["opponent"]
    limit = plan["choice_limit"]
    max_total = limit + cfg["reserved_dialog_choices"]
    attempt = {**row, "status": "started", "phase": None, "outcomes": dict.fromkeys(COLORS), "training_samples": 0,
               "optimizer_steps": 0, "concurrent_load_free": None, "cprofile": bool(context.get("cprofile")),
               "started_unix": time.time()}
    phases = Phases()
    started = time.perf_counter()
    env = collector = None
    attempt_path = output / "attempts" / f"row-{row['row']:03d}.json"

    def mark(phase):
        attempt["phase"] = phase
        write(attempt_path, attempt)
    mark("setup")
    native_failure = False
    verified = manifest = None
    try:
        seat = row["learner_player"]
        opponent_id = cfg["opponent"][0]
        models = (loaded.model, opponent.model) if seat == 0 else (opponent.model, loaded.model)
        checkpoints = (loaded.checkpoint, opponent_id) if seat == 0 else (opponent_id, loaded.checkpoint)
        sampled_by_player = [0, 0]

        class BoundedCollector(ops["collector_class"]):
            sampled = 0

            def act(self, request):
                require(type(self.sampled) is int and 0 <= self.sampled < max_total,
                        "Total policy-choice budget exhausted; unresolved query remains unscored")
                require(request.record["day"] <= cfg["max_days"], "Day budget exhausted")
                action = super().act(request)
                self.sampled += 1
                sampled_by_player[request.player] += 1
                return action

        collector = BoundedCollector(models, checkpoints, learner_player=seat,
            opponent_pool="performance-benchmark-v1", interface=cfg["interface"], purpose="diagnostic",
            origin="native", policy_seeds=tuple(row["policy_seeds"]))
        collector.act = TIMER.timed(collector.act, "policy.act")
        lim = cfg["environment_limits"]
        limits = ops["limits_class"](max_days=cfg["max_days"], max_actions=max_total + 1,
            watchdog_seconds=lim["watchdog_seconds"], startup_seconds=lim["startup_seconds"],
            shutdown_seconds=lim["shutdown_seconds"], bridge_timeout=lim["bridge_timeout"])
        env = ops["env_class"](launcher, context["episodes"], limits=limits, ledger=True, replay=True,
            interface=cfg["interface"], agents={color: {"checkpoint": key, "kind": "fixed-untrained-recurrent",
                "evaluation_only": True, "performance_benchmark": True} for color, key in zip(COLORS, checkpoints)})
        ops["instrument_environment"](env, launcher)
        mark("collection")
        with phases("collection"):
            collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"], max_requests=limit)
        attempt["environment"] = environment_counters(env)
        folder = env.episode_dir
        attempt["episode_directory"] = str(folder)
        mark("persist_rollout")
        with phases("persist_rollout"):
            write(folder / "neural-rollout.json", collected.to_dict())
        manifest = json.loads((folder / "manifest.json").read_text())
        attempt.update(episode_status=manifest["status"], termination_reason=manifest["termination_reason"],
            sampled_choices=sum(len(r.decisions) for r in collected.rollouts),
            sampled_by_player=list(sampled_by_player), histories=history_counts(collected),
            actions={color: {key: manifest["players"][color].get(key) for key in
                             ("actions_sent", "actions_accepted", "actions_rejected", "responses_undelivered")}
                     | {"continuation_steps": manifest["players"][color]["live_evidence"]["continuation_steps"]}
                     for color in COLORS},
            manifest_wall_seconds=manifest.get("wall_seconds"), engine_seed_recorded=manifest.get("seed"))
        require(attempt["sampled_choices"] <= max_total, "Total policy-choice cap exceeded")
        require(all(not r.optimization_eligible for r in collected.rollouts), "Benchmark became optimizer eligible")
        if manifest["status"] == "failed":
            native_failure = True
            attempt["native_failure_detail"] = manifest.get("failure_detail")
        else:
            require(manifest["status"] in ("completed", "truncated"), "Unknown episode boundary")
            mark("reconstruction")
            with phases("reconstruction_load"):
                reloaded, _ = ops["load_fixed"](plan["checkpoint"]["path"], cfg, cfg["interface"])
            with phases("reconstruction"):
                reconstruction = ops["audit_persisted_policy"](folder, reloaded, row["policy_seeds"], interface=cfg["interface"])
                write(folder / "policy-reconstruction.json", reconstruction)
            attempt["reconstruction"] = [{key: player[key] for key in ("player", "actions_resampled",
                "values_recomputed", "conditionals_recomputed", "maximum_absolute_difference")}
                for player in reconstruction["players"]]
            remaining = context["deadline"] - time.monotonic() - CLEANUP_RESERVE_SECONDS
            require(remaining > 1, "No campaign time remains for native replay")
            mark("native_replay")
            with phases("native_replay"):
                replay = ops["verify_episode"](folder, binary=Path(plan["replay_binary"]["path"]), data=Path(plan["data"]),
                                               timeout=min(cfg["replay_timeout_seconds"], remaining))
            attempt["replay"] = compact_native(replay)
            require(replay.get("verified") is True, "Independent native replay failed")
        mark("behavior")
        with phases("behavior"):
            behavior = ops["analyze_episode"](folder)
            write(folder / "behavior.json", behavior)
        mark("weights_check")
        with phases("weights_check"):
            require(ops["frozen_fingerprints"](loaded, plan["checkpoint"]["path"]) == context["fixed"], "Weights changed")
        attempt["weights_unchanged"] = True
        verified = "native_failure" if native_failure else "passed"
    except Exception as error:
        verified = None
        attempt.update(error=f"{type(error).__name__}: {error}", failed_phase=attempt["phase"])
        try:
            ops["retain_failed_evidence"](env, collector, attempt, error,
                cleanup_seconds=max(1.0, min(30.0, context["deadline"] - time.monotonic() - 2)))
        except Exception as secondary:
            attempt["failure_retention_error"] = f"{type(secondary).__name__}: {secondary}"
    finally:
        attempt["phase"] = "cleanup"
        cleanup_errors = []
        with phases("cleanup"):
            # Independent paths: a failed env.close never skips group cleanup.
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    cleanup_errors.append(f"env.close: {type(error).__name__}: {error}")
            try:
                launcher.close_owned_groups()
            except Exception as error:
                cleanup_errors.append(f"close_owned_groups: {type(error).__name__}: {error}")
        attempt["cleanup_verified"] = not cleanup_errors
        if cleanup_errors:
            attempt["cleanup_errors"] = cleanup_errors
        if verified is None or cleanup_errors:
            attempt.update(status="failed", outcomes=dict.fromkeys(COLORS))
        else:
            attempt["status"] = verified
            if verified == "passed":
                attempt["outcomes"] = manifest["outcomes_by_player"]
        attempt["stop_worker"] = attempt["status"] not in CONTINUABLE
        attempt["phases"] = phases.result
        wrapped_calls = sum(stage["calls"] for data in phases.result.values() for stage in data["stages"].values())
        attempt["timing_wrapper_calls"] = wrapped_calls
        attempt["timing_overhead_estimate_seconds"] = round(wrapped_calls * context["wrapper_overhead"], 6)
        attempt["attempt_wall_seconds"] = round(time.perf_counter() - started, 6)
        attempt["worker_rss_bytes"] = current_rss_bytes()
        attempt["worker_peak_rss_bytes"] = peak_rss_bytes(resource.RUSAGE_SELF)
        attempt["children_peak_rss_bytes"] = peak_rss_bytes(resource.RUSAGE_CHILDREN)
        attempt["finished_unix"] = time.time()
        write(attempt_path, attempt)
    return attempt


# -- identities ------------------------------------------------------------------

def source_snapshot(root=ROOT):
    """Every tracked file's bytes as they are on disk now, plus the commit."""
    root = Path(root)
    names = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.split(b"\0")
    files = {}
    for name in sorted(item.decode() for item in names if item):
        path = root / name
        files[name] = digest(path) if path.is_file() else None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"git_commit": commit, "tracked_files": len(files), "tracked_sha256": combined, "files": files}


def native_snapshot(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    binary = Path(plan["binary"])
    return {"binary": binding(binary), "replay_binary": binding(plan["replay_binary"]["path"]),
            "map": binding(plan["map"]), "runtime": runtime_identity(binary.parent)}


def identity_drift(plan, plan_sha256, plan_path, root=ROOT):
    """Names of every bound identity that differs from the prepared plan."""
    drift = []
    try:
        if digest(plan_path) != plan_sha256:
            drift.append("plan")
        current = source_snapshot(root)
        if current["tracked_sha256"] != plan["source"]["tracked_sha256"] or current["git_commit"] != plan["source"]["git_commit"]:
            changed = sorted(name for name in set(current["files"]) | set(plan["source"]["files"])
                             if current["files"].get(name) != plan["source"]["files"].get(name))
            drift.append("source:" + ",".join(changed[:20]) if changed else "source:commit")
        if native_snapshot(plan) != plan["native_snapshot"]:
            drift.append("native_runtime")
    except Exception as error:  # an identity that cannot be read is not the planned one
        drift.append(f"unreadable:{type(error).__name__}: {error}")
    return drift


def worker(args, ops=None):
    process_started = time.perf_counter()
    plan_path = Path(args.plan)
    require(digest(plan_path) == args.plan_sha256, "Plan file differs from the supervisor's plan")
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    report = {"worker": args.worker, "pid": os.getpid(), "rows": plan["assignment"][str(args.worker)],
              "started_unix": time.time(), "init": {}, "attempts": []}
    report_path = output / f"worker-{args.worker:02d}.json"
    cprofile = None
    stop = threading.Event()

    def budget_signal(signum, frame):
        stop.set()
        raise TimeoutError("Campaign deadline is ending")
    signal.signal(signal.SIGUSR1, budget_signal)
    schedule = rows(plan["configuration"], plan["row_count"])
    pending = list(report["rows"])

    def skip_rest(reason):
        while pending:
            index = pending.pop(0)
            report["attempts"].append({**schedule[index], "status": "not_started", "reason": reason})
    try:
        before = time.perf_counter()
        import torch
        torch.set_num_threads(plan["configuration"]["torch_threads"])
        report["init"]["torch_import_seconds"] = round(time.perf_counter() - before, 6)
        report["init"]["torch_threads"] = torch.get_num_threads()
        before = time.perf_counter()
        ops = ops or real_operations()
        if ops.get("instrument_modules", True):
            instrument_modules()
        report["init"]["module_import_seconds"] = round(time.perf_counter() - before, 6)
        report["init"]["timing_wrapper_overhead_seconds_per_call"] = overhead = wrapper_overhead_seconds()
        cfg = plan["configuration"]
        before = time.perf_counter()
        loaded, fixed = ops["load_fixed"](plan["checkpoint"]["path"], cfg, cfg["interface"])
        require(fixed == plan["frozen"], "Frozen weights changed after planning")
        opponent = loaded.opponent_pool.get(cfg["opponent"][0])
        report["init"]["checkpoint_load_seconds"] = round(time.perf_counter() - before, 6)
        report["source_compatibility"] = ops.get("source_compatibility", source_compatibility)(loaded, cfg["interface"])
        episodes = output / f"worker-{args.worker:02d}"
        launcher = ops["launcher_factory"](plan, args, output) if "launcher_factory" in ops else default_launcher(plan, args, output)
        context = {"cfg": cfg, "plan": plan, "launcher": launcher, "loaded": loaded, "opponent": opponent, "ops": ops,
                   "fixed": fixed, "episodes": episodes, "deadline": args.deadline, "wrapper_overhead": overhead,
                   "cprofile": args.cprofile}
        report["init"]["worker_ready_seconds"] = round(time.perf_counter() - process_started, 6)
        write(report_path, report)
        check = ops.get("identity_drift", identity_drift)
        while pending:
            if stop.is_set() or args.deadline - time.monotonic() < CLEANUP_RESERVE_SECONDS + 10:
                skip_rest("campaign_deadline")
                break
            drift = check(plan, args.plan_sha256, plan_path)
            if drift:
                report["identity_drift"] = drift
                skip_rest("identity_drift_before_row")
                break
            row = schedule[pending.pop(0)]
            if args.cprofile:
                import cProfile
                cprofile = cProfile.Profile()
                cprofile.enable()
            attempt = run_row(context, row, output)
            if cprofile is not None:
                cprofile.disable()
                folder = output / "profiles"
                folder.mkdir(exist_ok=True)
                cprofile.dump_stats(folder / f"row-{row['row']:03d}.prof")
                cprofile = None
            drift = check(plan, args.plan_sha256, plan_path)
            if drift:
                attempt.update(status="failed", outcomes=dict.fromkeys(COLORS), stop_worker=True,
                               error="Identity drift during the attempt: " + "; ".join(drift))
                write(output / "attempts" / f"row-{row['row']:03d}.json", attempt)
                report["identity_drift"] = drift
            report["attempts"].append({key: attempt.get(key) for key in ("row", "status", "attempt_wall_seconds")})
            write(report_path, report)
            if attempt["stop_worker"]:
                skip_rest("worker_stopped_after_" + attempt["status"])
                break
        report["status"] = "finished"
    except TimeoutError as error:
        report.update(status="deadline", error=str(error))
        skip_rest("campaign_deadline")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        skip_rest("worker_failure")
    finally:
        if cprofile is not None:
            cprofile.disable()
        report["finished_unix"] = time.time()
        report["worker_wall_seconds"] = round(time.perf_counter() - process_started, 6)
        report["cpu_self_seconds"] = round(_cpu_self(), 6)
        report["cpu_children_seconds"] = round(_cpu_children(), 6)
        write(report_path, report)
    return report


def default_launcher(plan, args, output):
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    return DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["map"]),
        deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
        evidence_path=output / f"native-deadline-worker-{args.worker:02d}.jsonl")


# -- supervisor ----------------------------------------------------------------

def process_table():
    """{pid: (ppid, cpu, command)}, or None when inspection itself failed."""
    try:
        result = subprocess.run(["ps", "-axo", "pid=,ppid=,pcpu=,command="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    table = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2], parts[3])
    return table or None


def foreign_processes(supervisor_pid, owned_root, table=None):
    """Heavy processes that are not descendants of this supervisor, or None.

    Only the supervisor's own process tree is excluded. Another campaign or a
    build in the same checkout is foreign and is flagged as same_root.
    """
    table = process_table() if table is None else table
    if table is None:
        return None
    owned = {supervisor_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in table.items():
            if ppid in owned and pid not in owned:
                owned.add(pid)
                changed = True
    found = []
    for pid, (ppid, cpu, command) in sorted(table.items()):
        if pid in owned or command.startswith(("ps ", "grep ")):
            continue
        same_root = str(owned_root) in command
        if FOREIGN_PATTERN.search(command) or "contextual_benchmark" in command or same_root:
            found.append({"pid": pid, "cpu_percent": cpu, "command": command[:200], "same_root": same_root})
    return found


class LoadMonitor(threading.Thread):
    def __init__(self, supervisor_pid, owned_root, interval=5.0, inspect=foreign_processes):
        super().__init__(daemon=True, name="load-monitor")
        self.supervisor_pid, self.owned_root, self.interval, self.inspect = supervisor_pid, owned_root, interval, inspect
        self.samples, self.failed_samples, self.foreign = 0, 0, {}
        self.max_load = 0.0
        self.halt = threading.Event()

    def run(self):
        while not self.halt.is_set():
            self.samples += 1
            self.max_load = max(self.max_load, os.getloadavg()[0])
            found = self.inspect(self.supervisor_pid, self.owned_root)
            if found is None:
                self.failed_samples += 1
            for item in found or []:
                self.foreign.setdefault((item["pid"], item["command"]), item)
            self.halt.wait(self.interval)


def prepare(args, cfg):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Campaign output exists; never replace or retry an earlier campaign")
    checkpoint = binding(args.checkpoint)
    require(checkpoint["sha256"] == cfg["checkpoint_sha256"], "Only the frozen v7 benchmark checkpoint is permitted")
    game_map = binding(args.map)
    require(game_map["sha256"] == cfg["map_sha256"], "Benchmark scenario changed")
    launcher = VcmiLauncher(args.binary.resolve(), args.data.resolve(), args.map.resolve())
    launcher.preflight()
    checked = subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
    require(checked.returncode == 0, "Prepared VCMI dependency does not match the recipe: " + checked.stderr.strip())
    choice_limit = args.choice_limit if args.choice_limit is not None else cfg["ordinary_choice_limit"]
    require(type(choice_limit) is int and 1 <= choice_limit <= 4088, "Choice limit outside the evaluated range")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    require(not dirty, "Campaigns run only from a clean committed checkout")
    source = source_snapshot(ROOT)
    declared = {"implementation": args.implementation, "baseline_commit": args.baseline_commit}
    changed = subprocess.run(["git", "diff", "--name-only", args.baseline_commit, "HEAD"], cwd=ROOT,
                             capture_output=True, text=True)
    require(changed.returncode == 0, "Declared baseline commit is unknown: " + changed.stderr.strip())
    declared["files_changed_from_baseline"] = sorted(line for line in changed.stdout.splitlines() if line)
    runtime_changes = [name for name in declared["files_changed_from_baseline"]
                       if not name.startswith(("experiments/perf/", "tests/test_perf_", "configs/performance-"))]
    require((args.implementation == "baseline") == (not runtime_changes),
            "Declare --implementation optimized exactly when runtime sources differ from the baseline commit")
    declared["runtime_files_changed_from_baseline"] = runtime_changes
    plan = {"schema": SCHEMA + "-plan", "label": args.label, "configuration": cfg, "output": str(output),
            "row_count": args.rows, "rows": rows(cfg, args.rows), "workers": args.workers,
            "assignment": {str(k): v for k, v in assignment(args.rows, args.workers).items()},
            "choice_limit": choice_limit, "choice_limit_is_frozen_default": choice_limit == cfg["ordinary_choice_limit"],
            "campaign_seconds": args.campaign_seconds, "cleanup_grace_seconds": CLEANUP_GRACE_SECONDS,
            "cprofile": args.cprofile, "declared_source": declared,
            "checkpoint": checkpoint, "map": str(args.map.resolve()), "map_binding": game_map,
            "binary": str(args.binary.resolve()), "data": str(args.data.resolve()),
            "replay_binary": binding(args.replay_binary), "native": launcher.identity(),
            "runtime": runtime_identity(args.binary.resolve().parent), "prepared_check": checked.stdout.strip(),
            "source": source, "harness_sha256": digest(Path(__file__)),
            "host": {"platform": sys.platform, "machine": os.uname().machine, "cpu_count": os.cpu_count(),
                     "python": sys.version, "executable": sys.executable},
            "training_performed": False, "optimizer_updates": 0}
    plan["native_snapshot"] = native_snapshot(plan)
    return output, plan


def real_supervisor_operations():
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    return {"spawn": guarded_popen, "close_process": close_owned_process, "cleanup_evidence": cleanup_from_evidence,
            "monotonic": time.monotonic, "wall": time.time, "sleep": time.sleep, "inspect": foreign_processes,
            "kill": os.kill, "monitor_interval": 5.0}


def run(args):
    cfg = config()
    output, plan = prepare(args, cfg)
    import torch
    torch.set_num_threads(cfg["torch_threads"])
    _, plan["frozen"] = load_fixed(plan["checkpoint"]["path"], cfg, cfg["interface"])
    plan["torch"] = str(torch.__version__)
    output.mkdir(parents=True)
    (output / "attempts").mkdir()
    write(output / "plan.json", plan)
    return supervise(plan, output, real_supervisor_operations())


def supervise(plan, output, ops):
    """Launch, bound and clean up every worker, then always write the report.

    The whole campaign, including cleanup verification, ends by
    campaign_seconds + cleanup_grace_seconds. Cleanup shares one absolute
    budget; each wait gets only what remains of it.
    """
    plan_sha256 = digest(output / "plan.json")
    started, wall_started = ops["monotonic"](), ops["wall"]()
    deadline, wall_deadline = started + plan["campaign_seconds"], wall_started + plan["campaign_seconds"]
    cleanup_end = deadline + plan["cleanup_grace_seconds"]
    load_start = os.getloadavg()
    supervisor_pid = os.getpid()
    monitor = LoadMonitor(supervisor_pid, ROOT, interval=ops["monitor_interval"], inspect=ops["inspect"])
    initial_foreign = ops["inspect"](supervisor_pid, ROOT)
    monitor.start()
    processes = {}
    supervisor_errors = []
    signalled = False
    try:
        for worker_index in range(plan["workers"]):
            command = [sys.executable, "-m", "experiments.perf.contextual_benchmark", "worker",
                       "--plan", str(output / "plan.json"), "--plan-sha256", plan_sha256, "--worker", str(worker_index),
                       "--deadline", str(deadline - 2), "--wall-deadline", str(wall_deadline - 2)]
            if plan["cprofile"]:
                command.append("--cprofile")
            log = (output / f"worker-{worker_index:02d}.log").open("xb")
            try:
                process = ops["spawn"](command, deadline_monotonic=deadline - 1, deadline_wall=wall_deadline - 1,
                    evidence_path=output / f"worker-deadline-{worker_index:02d}.jsonl", cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            processes[worker_index] = (process, log)
        while any(process.poll() is None for process, _ in processes.values()):
            left = deadline - ops["monotonic"]()
            if not signalled and left <= CLEANUP_RESERVE_SECONDS:
                for process, _ in processes.values():
                    if process.poll() is None:
                        try:
                            ops["kill"](process.pid, signal.SIGUSR1)
                        except ProcessLookupError:
                            pass
                signalled = True
            if left <= 1:
                break
            ops["sleep"](0.2)
    except BaseException as error:
        supervisor_errors.append(f"supervisor: {type(error).__name__}: {error}")
    finally:
        def budget():
            return max(0.01, cleanup_end - ops["monotonic"]())
        for worker_index, (process, log) in processes.items():
            try:
                ops["close_process"](process, timeout=budget())
            except Exception as error:
                supervisor_errors.append(f"worker {worker_index}: {type(error).__name__}: {error}")
            log.close()
        for worker_index in processes:
            try:
                ops["cleanup_evidence"](output / f"native-deadline-worker-{worker_index:02d}.jsonl", timeout=budget())
            except Exception as error:
                supervisor_errors.append(f"native cleanup {worker_index}: {type(error).__name__}: {error}")
        campaign_wall = ops["monotonic"]() - started
        monitor.halt.set()
        monitor.join(timeout=10)
        workers = []
        for worker_index in range(plan["workers"]):
            path = output / f"worker-{worker_index:02d}.json"
            workers.append(read_json(path) or {"worker": worker_index, "status": "missing" if worker_index in processes else "not_launched"})
        attempts = collect_attempts(plan["rows"], output, workers, signalled)
        inspection_failed = initial_foreign is None or monitor.failed_samples > 0
        concurrent = bool(initial_foreign) or bool(monitor.foreign)
        clean = not concurrent and not inspection_failed and not plan["cprofile"] and not supervisor_errors
        for attempt in attempts:
            attempt["concurrent_load_free"] = None if inspection_failed else not concurrent
        report = {"schema": SCHEMA + "-result", "label": plan["label"], "plan_sha256": plan_sha256,
                  "workers": plan["workers"], "row_count": plan["row_count"], "choice_limit": plan["choice_limit"],
                  "campaign_seconds": plan["campaign_seconds"], "declared_source": plan["declared_source"],
                  "source_tracked_sha256": plan["source"]["tracked_sha256"], "git_commit": plan["source"]["git_commit"],
                  "campaign_wall_seconds": round(campaign_wall, 6), "deadline_signalled": signalled,
                  "supervisor_errors": supervisor_errors, "load_average_start": load_start,
                  "load_average_end": os.getloadavg(), "load_average_max_1m": monitor.max_load,
                  "foreign_processes_at_start": initial_foreign, "foreign_processes_during": list(monitor.foreign.values()),
                  "inspection_samples": monitor.samples, "inspection_failed_samples": monitor.failed_samples,
                  "inspection_failed": inspection_failed, "concurrent_load_detected": concurrent,
                  "clean_timing_sample": clean, "cprofile": plan["cprofile"],
                  "attempts": attempts, "worker_reports": workers, "summary": summarize_attempts(attempts, campaign_wall),
                  "training_performed": False, "optimizer_updates": 0}
        write(output / "campaign.json", report)
    return report


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def collect_attempts(plan_rows, output, workers, signalled):
    """Every scheduled row exactly once. A started row never becomes not_started."""
    skipped = {}
    for report in workers:
        for item in report.get("attempts") or []:
            if item.get("status") == "not_started":
                skipped[item["row"]] = item.get("reason")
    attempts = []
    for row in plan_rows:
        path = output / "attempts" / f"row-{row['row']:03d}.json"
        data = read_json(path) if path.exists() else None
        if path.exists() and data is None:
            attempts.append({**row, "status": "incomplete", "reason": "unreadable_attempt_record",
                             "outcomes": dict.fromkeys(COLORS)})
        elif data is None:
            reason = skipped.get(row["row"]) or ("campaign_deadline" if signalled else "worker_failure")
            attempts.append({**row, "status": "not_started", "reason": reason})
        elif data.get("status") in ("started", "running"):
            attempts.append({**data, "status": "incomplete", "reason": "attempt_not_finished",
                             "outcomes": dict.fromkeys(COLORS)})
        else:
            attempts.append(data)
    return attempts


# -- summaries -----------------------------------------------------------------

def _stage_total(attempt, phase, key=None):
    data = attempt.get("phases", {}).get(phase)
    if data is None:
        return None
    if key is None:
        return data["wall_seconds"]
    stage = data["stages"].get(key)
    return stage["seconds"] if stage else 0.0


def summarize_attempts(attempts, campaign_wall):
    statuses = Counter(attempt.get("status") for attempt in attempts)
    episode_statuses = Counter(attempt.get("episode_status") for attempt in attempts if attempt.get("episode_status"))
    done = [attempt for attempt in attempts if attempt.get("phases")]
    totals = defaultdict(float)
    for attempt in done:
        for phase, data in attempt["phases"].items():
            totals["phase." + phase + ".wall"] += data["wall_seconds"]
            totals["phase." + phase + ".cpu_self"] += data["cpu_self_seconds"]
            totals["phase." + phase + ".cpu_children"] += data["cpu_children_seconds"]
            for key, stage in data["stages"].items():
                totals[phase + "." + key] += stage["seconds"]
        for key, value in attempt.get("environment", {}).items():
            if isinstance(value, (int, float)):
                totals["environment." + key] += value
            elif isinstance(value, dict):
                for inner, number in value.items():
                    totals["environment." + key + "." + inner] += number
    choices = sum(attempt.get("sampled_choices") or 0 for attempt in done)
    accepted = sum((attempt.get("actions") or {}).get(color, {}).get("actions_accepted") or 0 for attempt in done for color in COLORS)
    rejected = sum((attempt.get("actions") or {}).get(color, {}).get("actions_rejected") or 0 for attempt in done for color in COLORS)
    passed = [attempt for attempt in attempts if attempt.get("status") == "passed"]
    return {"attempts": len(attempts), "status_counts": dict(statuses), "episode_status_counts": dict(episode_statuses),
            "natural_endings_verified": sum(1 for a in passed if a.get("episode_status") == "completed"),
            "cutoffs_verified": sum(1 for a in passed if a.get("episode_status") == "truncated"),
            "failures": sum(1 for a in attempts if a.get("status") not in ("passed", "not_started")),
            "incomplete": statuses.get("incomplete", 0), "not_started": statuses.get("not_started", 0),
            "sampled_choices": choices, "actions_accepted": accepted, "actions_rejected": rejected,
            "campaign_wall_seconds": round(campaign_wall, 6),
            "verified_attempts_per_hour": round(3600 * len(passed) / campaign_wall, 3) if campaign_wall else None,
            "sampled_choices_per_second_end_to_end": round(choices / campaign_wall, 3) if campaign_wall else None,
            "totals_seconds": {key: round(value, 6) for key, value in sorted(totals.items())}}


def campaign_group(report, plan):
    """Campaigns are comparable only with identical workload and implementation."""
    cfg_sha = hashlib.sha256(json.dumps(plan["configuration"], sort_keys=True).encode()).hexdigest()
    return (report["label"], report["workers"], plan["declared_source"]["implementation"], plan["source"]["tracked_sha256"],
            plan["row_count"], plan["choice_limit"], plan["campaign_seconds"], plan["cprofile"], cfg_sha,
            plan["native_snapshot"]["binary"]["sha256"], plan["checkpoint"]["sha256"])


def exclusion_reasons(report):
    reasons = []
    if report.get("cprofile"):
        reasons.append("profiled")
    if report.get("concurrent_load_detected"):
        reasons.append("concurrent_load")
    if report.get("inspection_failed", True):
        reasons.append("load_inspection_unknown")
    if report.get("supervisor_errors"):
        reasons.append("supervisor_errors")
    if report["summary"].get("failures") or report["summary"].get("not_started"):
        reasons.append("incomplete_or_failed_rows")
    return reasons


def summarize(args):
    """Median and spread per comparable group, over clean campaigns only."""
    groups = defaultdict(list)
    for path in args.campaigns:
        report = json.loads((Path(path) / "campaign.json").read_text())
        plan = json.loads((Path(path) / "plan.json").read_text())
        groups[campaign_group(report, plan)].append((str(path), report))
    result = []
    for key, members in sorted(groups.items(), key=lambda item: json.dumps(item[0], default=str)):
        label, workers, implementation, source_sha, row_count, choice_limit, seconds, profiled = key[:8]
        clean = [(path, report) for path, report in members if not exclusion_reasons(report)]
        entry = {"label": label, "workers": workers, "implementation": implementation, "source_tracked_sha256": source_sha,
                 "rows": row_count, "choice_limit": choice_limit, "campaign_seconds": seconds, "cprofile": profiled,
                 "campaigns": [path for path, _ in members], "clean_campaigns": [path for path, _ in clean],
                 "excluded": {path: exclusion_reasons(report) for path, report in members if exclusion_reasons(report)}}
        metrics = defaultdict(list)
        for _, report in clean:
            summary = report["summary"]
            metrics["campaign_wall_seconds"].append(summary["campaign_wall_seconds"])
            metrics["sampled_choices"].append(summary["sampled_choices"])
            metrics["verified_attempts_per_hour"].append(summary["verified_attempts_per_hour"])
            metrics["sampled_choices_per_second_end_to_end"].append(summary["sampled_choices_per_second_end_to_end"])
            for name, value in summary["totals_seconds"].items():
                metrics[name].append(value)
            for name in ("natural_endings_verified", "cutoffs_verified", "failures", "not_started",
                         "actions_accepted", "actions_rejected"):
                metrics[name].append(summary[name])
        entry["clean_metrics"] = {name: {"median": statistics.median(values), "min": min(values), "max": max(values),
                                         "n": len(values)} for name, values in sorted(metrics.items())
                                  if values and all(isinstance(v, (int, float)) for v in values)}
        histories = defaultdict(set)
        for _, report in members:
            for attempt in report["attempts"]:
                for color, data in (attempt.get("histories") or {}).items():
                    histories[(attempt["row"], color)].add(data["history_sha256"])
        entry["history_hash_sets"] = {f"{row}:{color}": sorted(values) for (row, color), values in sorted(histories.items())}
        result.append(entry)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        write(args.output, {"schema": SCHEMA + "-summary", "groups": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    campaign = sub.add_parser("run", help="Plan and execute one bounded campaign")
    for name in ("output", "checkpoint", "binary", "data", "replay-binary"):
        campaign.add_argument("--" + name, type=Path, required=True)
    campaign.add_argument("--map", type=Path, default=DEFAULT_MAP)
    campaign.add_argument("--label", required=True)
    campaign.add_argument("--rows", type=int, required=True)
    campaign.add_argument("--workers", type=int, default=1)
    campaign.add_argument("--campaign-seconds", type=float, required=True)
    campaign.add_argument("--choice-limit", type=int, default=None)
    campaign.add_argument("--cprofile", action="store_true")
    campaign.add_argument("--implementation", choices=("baseline", "optimized"), required=True)
    campaign.add_argument("--baseline-commit", default="9193db35857e07ddbc45e98a1dd7c3f4a8c07da0")
    work = sub.add_parser("worker")
    work.add_argument("--plan", type=Path, required=True)
    work.add_argument("--plan-sha256", required=True)
    work.add_argument("--worker", type=int, required=True)
    work.add_argument("--deadline", type=float, required=True)
    work.add_argument("--wall-deadline", type=float, required=True)
    work.add_argument("--cprofile", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("campaigns", nargs="+")
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        require(10 <= args.campaign_seconds <= 7200, "Campaign wall cap must be 10 to 7200 seconds")
        report = run(args)
        print(json.dumps({"label": report["label"], "workers": report["workers"],
                          "campaign_wall_seconds": report["campaign_wall_seconds"],
                          "concurrent_load_detected": report["concurrent_load_detected"],
                          "summary": {k: v for k, v in report["summary"].items() if k != "totals_seconds"}}, indent=2))
    elif args.mode == "worker":
        worker(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
