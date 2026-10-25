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


def run_row(context, row, output):
    import torch  # noqa: F401  (already imported; keeps the worker's dependency explicit)
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.contextual_replay import verify_episode
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.evaluation_behavior import analyze_episode
    from validation.probes.validate_hero_policy import audit_persisted_policy, frozen_fingerprints, retain_failed_evidence
    cfg, plan, launcher = context["cfg"], context["plan"], context["launcher"]
    loaded, opponent = context["loaded"], context["opponent"]
    limit = plan["choice_limit"]
    max_total = limit + cfg["reserved_dialog_choices"]
    attempt = {**row, "status": "running", "outcomes": dict.fromkeys(COLORS), "training_samples": 0,
               "optimizer_steps": 0, "concurrent_load_free": None}
    phases = Phases()
    started = time.perf_counter()
    env = collector = None
    attempt_path = output / "attempts" / f"row-{row['row']:03d}.json"
    try:
        seat = row["learner_player"]
        opponent_id = cfg["opponent"][0]
        models = (loaded.model, opponent.model) if seat == 0 else (opponent.model, loaded.model)
        checkpoints = (loaded.checkpoint, opponent_id) if seat == 0 else (opponent_id, loaded.checkpoint)
        sampled_by_player = [0, 0]

        class BoundedCollector(ContextualCollector):
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
        limits = Limits(max_days=cfg["max_days"], max_actions=max_total + 1, watchdog_seconds=lim["watchdog_seconds"],
                        startup_seconds=lim["startup_seconds"], shutdown_seconds=lim["shutdown_seconds"],
                        bridge_timeout=lim["bridge_timeout"])
        env = HeadlessSelfPlayEnv(launcher, context["episodes"], limits=limits, ledger=True, replay=True,
            interface=cfg["interface"], agents={color: {"checkpoint": key, "kind": "fixed-untrained-recurrent",
                "evaluation_only": True, "performance_benchmark": True} for color, key in zip(COLORS, checkpoints)})
        instrument_environment(env, launcher)
        with phases("collection"):
            collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"], max_requests=limit)
        attempt["environment"] = environment_counters(env)
        folder = env.episode_dir
        attempt["episode_directory"] = str(folder)
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
            attempt.update(status="native_failure", error=manifest.get("failure_detail"))
        else:
            require(manifest["status"] in ("completed", "truncated"), "Unknown episode boundary")
            with phases("reconstruction_load"):
                reloaded, _ = load_fixed(plan["checkpoint"]["path"], cfg, cfg["interface"])
            with phases("reconstruction"):
                reconstruction = audit_persisted_policy(folder, reloaded, row["policy_seeds"], interface=cfg["interface"])
                write(folder / "policy-reconstruction.json", reconstruction)
            attempt["reconstruction"] = [{key: player[key] for key in ("player", "actions_resampled",
                "values_recomputed", "conditionals_recomputed", "maximum_absolute_difference")}
                for player in reconstruction["players"]]
            remaining = context["deadline"] - time.monotonic() - CLEANUP_RESERVE_SECONDS
            require(remaining > 1, "No campaign time remains for native replay")
            with phases("native_replay"):
                replay = verify_episode(folder, binary=Path(plan["replay_binary"]["path"]), data=Path(plan["data"]),
                                        timeout=min(cfg["replay_timeout_seconds"], remaining))
            attempt["replay"] = compact_native(replay)
            require(replay.get("verified") is True, "Independent native replay failed")
            attempt.update(status="passed", outcomes=manifest["outcomes_by_player"])
        with phases("behavior"):
            behavior = analyze_episode(folder)
            write(folder / "behavior.json", behavior)
        with phases("weights_check"):
            require(frozen_fingerprints(loaded, plan["checkpoint"]["path"]) == context["fixed"], "Weights changed")
        attempt["weights_unchanged"] = True
    except Exception as error:
        attempt.update(status="failed" if attempt["status"] == "running" else attempt["status"],
                       error=f"{type(error).__name__}: {error}", outcomes=dict.fromkeys(COLORS))
        try:
            retain_failed_evidence(env, collector, attempt, error,
                                   cleanup_seconds=max(1.0, min(30.0, context["deadline"] - time.monotonic() - 2)))
        except Exception as secondary:
            attempt["failure_retention_error"] = f"{type(secondary).__name__}: {secondary}"
    finally:
        with phases("cleanup"):
            try:
                if env is not None:
                    env.close()
                launcher.close_owned_groups()
                attempt["cleanup_verified"] = True
            except Exception as error:
                attempt.update(cleanup_verified=False, cleanup_error=f"{type(error).__name__}: {error}")
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


def worker(args):
    process_started = time.perf_counter()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(plan["output"])
    report = {"worker": args.worker, "pid": os.getpid(), "rows": plan["assignment"][str(args.worker)],
              "started_unix": time.time(), "init": {}}
    report_path = output / f"worker-{args.worker:02d}.json"
    cprofile = None
    stop = threading.Event()

    def budget_signal(signum, frame):
        stop.set()
        raise TimeoutError("Campaign deadline is ending")
    signal.signal(signal.SIGUSR1, budget_signal)
    try:
        before = time.perf_counter()
        import torch
        torch.set_num_threads(plan["configuration"]["torch_threads"])
        report["init"]["torch_import_seconds"] = round(time.perf_counter() - before, 6)
        report["init"]["torch_threads"] = torch.get_num_threads()
        before = time.perf_counter()
        instrument_modules()
        from experiments.headless.process_deadline import DeadlineVcmiLauncher
        report["init"]["module_import_seconds"] = round(time.perf_counter() - before, 6)
        report["init"]["timing_wrapper_overhead_seconds_per_call"] = overhead = wrapper_overhead_seconds()
        cfg = plan["configuration"]
        before = time.perf_counter()
        loaded, fixed = load_fixed(plan["checkpoint"]["path"], cfg, cfg["interface"])
        require(fixed == plan["frozen"], "Frozen weights changed after planning")
        opponent = loaded.opponent_pool.get(cfg["opponent"][0])
        report["init"]["checkpoint_load_seconds"] = round(time.perf_counter() - before, 6)
        report["source_compatibility"] = source_compatibility(loaded, cfg["interface"])
        episodes = output / f"worker-{args.worker:02d}"
        launcher = DeadlineVcmiLauncher(Path(plan["binary"]), Path(plan["data"]), Path(plan["map"]),
            deadline_monotonic=args.deadline, deadline_wall=args.wall_deadline,
            evidence_path=output / f"native-deadline-worker-{args.worker:02d}.jsonl")
        context = {"cfg": cfg, "plan": plan, "launcher": launcher, "loaded": loaded, "opponent": opponent,
                   "fixed": fixed, "episodes": episodes, "deadline": args.deadline, "wrapper_overhead": overhead}
        report["init"]["worker_ready_seconds"] = round(time.perf_counter() - process_started, 6)
        write(report_path, report)
        schedule = rows(cfg, plan["row_count"])
        report["attempts"] = []
        for index in report["rows"]:
            row = schedule[index]
            if stop.is_set() or args.deadline - time.monotonic() < CLEANUP_RESERVE_SECONDS + 10:
                report["attempts"].append({**row, "status": "not_started", "reason": "campaign_deadline"})
                continue
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
            report["attempts"].append({key: attempt.get(key) for key in ("row", "status", "attempt_wall_seconds")})
            write(report_path, report)
        report["status"] = "finished"
    except TimeoutError as error:
        report.update(status="deadline", error=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if cprofile is not None:
            cprofile.disable()
        report["finished_unix"] = time.time()
        report["worker_wall_seconds"] = round(time.perf_counter() - process_started, 6)
        report["cpu_self_seconds"] = round(_cpu_self(), 6)
        report["cpu_children_seconds"] = round(_cpu_children(), 6)
        write(report_path, report)
    return report


# -- supervisor ----------------------------------------------------------------

def foreign_processes(owned_root, own_pids):
    try:
        listing = subprocess.run(["ps", "-axo", "pid=,ppid=,pcpu=,command="], capture_output=True, text=True,
                                 timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = []
    for line in listing.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, cpu, command = int(parts[0]), int(parts[1]), parts[2], parts[3]
        if pid in own_pids or ppid in own_pids or str(owned_root) in command or "contextual_benchmark" in command:
            continue
        if FOREIGN_PATTERN.search(command) and not command.startswith(("ps ", "grep ")):
            found.append({"pid": pid, "cpu_percent": cpu, "command": command[:200]})
    return found


class LoadMonitor(threading.Thread):
    def __init__(self, owned_root, own_pids, interval=5.0):
        super().__init__(daemon=True, name="load-monitor")
        self.owned_root, self.own_pids, self.interval = owned_root, own_pids, interval
        self.samples, self.foreign = 0, {}
        self.max_load = 0.0
        self.halt = threading.Event()

    def run(self):
        while not self.halt.is_set():
            self.samples += 1
            self.max_load = max(self.max_load, os.getloadavg()[0])
            for item in foreign_processes(self.owned_root, set(self.own_pids)) or []:
                self.foreign.setdefault(item["command"], item)
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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    plan = {"schema": SCHEMA + "-plan", "label": args.label, "configuration": cfg, "output": str(output),
            "row_count": args.rows, "rows": rows(cfg, args.rows), "workers": args.workers,
            "assignment": {str(k): v for k, v in assignment(args.rows, args.workers).items()},
            "choice_limit": choice_limit, "choice_limit_is_frozen_default": choice_limit == cfg["ordinary_choice_limit"],
            "campaign_seconds": args.campaign_seconds, "cprofile": args.cprofile,
            "checkpoint": checkpoint, "map": str(args.map.resolve()), "map_binding": game_map,
            "binary": str(args.binary.resolve()), "data": str(args.data.resolve()),
            "replay_binary": binding(args.replay_binary), "native": launcher.identity(),
            "runtime": runtime_identity(args.binary.resolve().parent), "prepared_check": checked.stdout.strip(),
            "source": {"git_commit": commit, "tracked_worktree_dirty": bool(dirty), "root": str(ROOT)},
            "host": {"platform": sys.platform, "machine": os.uname().machine, "cpu_count": os.cpu_count(),
                     "python": sys.version, "executable": sys.executable},
            "training_performed": False, "optimizer_updates": 0}
    return output, plan


def run(args):
    cfg = config()
    output, plan = prepare(args, cfg)
    import torch
    from validation.probes.validate_hero_policy import frozen_fingerprints  # noqa: F401
    torch.set_num_threads(cfg["torch_threads"])
    _, plan["frozen"] = load_fixed(plan["checkpoint"]["path"], cfg, cfg["interface"])
    plan["torch"] = str(torch.__version__)
    output.mkdir(parents=True)
    (output / "attempts").mkdir()
    write(output / "plan.json", plan)
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    started, wall_started = time.monotonic(), time.time()
    deadline, wall_deadline = started + args.campaign_seconds, wall_started + args.campaign_seconds
    load_start = os.getloadavg()
    own_pids = [os.getpid()]
    monitor = LoadMonitor(ROOT, own_pids)
    initial_foreign = foreign_processes(ROOT, set(own_pids))
    monitor.start()
    processes = {}
    for worker_index in range(args.workers):
        command = [sys.executable, "-m", "experiments.perf.contextual_benchmark", "worker",
                   "--plan", str(output / "plan.json"), "--worker", str(worker_index),
                   "--deadline", str(deadline - 2), "--wall-deadline", str(wall_deadline - 2)]
        if args.cprofile:
            command.append("--cprofile")
        log = (output / f"worker-{worker_index:02d}.log").open("xb")
        process = guarded_popen(command, deadline_monotonic=deadline - 1, deadline_wall=wall_deadline - 1,
            evidence_path=output / f"worker-deadline-{worker_index:02d}.jsonl", cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes[worker_index] = (process, log)
        own_pids.append(process.pid)
    signalled = False
    while any(process.poll() is None for process, _ in processes.values()):
        left = deadline - time.monotonic()
        if not signalled and left <= CLEANUP_RESERVE_SECONDS:
            for process, _ in processes.values():
                if process.poll() is None:
                    try:
                        os.kill(process.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
            signalled = True
        if left <= 1:
            break
        time.sleep(0.2)
    supervisor_errors = []
    for worker_index, (process, log) in processes.items():
        try:
            close_owned_process(process, timeout=2.0)
        except Exception as error:
            supervisor_errors.append(f"worker {worker_index}: {type(error).__name__}: {error}")
        log.close()
        try:
            cleanup_from_evidence(output / f"native-deadline-worker-{worker_index:02d}.jsonl", timeout=2.0)
        except FileNotFoundError:
            pass
        except Exception as error:
            supervisor_errors.append(f"native cleanup {worker_index}: {type(error).__name__}: {error}")
    campaign_wall = time.monotonic() - started
    monitor.halt.set()
    monitor.join(timeout=10)
    attempts = []
    for row in plan["rows"]:
        path = output / "attempts" / f"row-{row['row']:03d}.json"
        attempts.append(json.loads(path.read_text()) if path.exists()
                        else {**row, "status": "not_started", "reason": "campaign_deadline" if signalled else "worker_failure"})
    workers = []
    for worker_index in range(args.workers):
        path = output / f"worker-{worker_index:02d}.json"
        workers.append(json.loads(path.read_text()) if path.exists() else {"worker": worker_index, "status": "missing"})
    concurrent = bool(initial_foreign) or bool(monitor.foreign)
    for attempt in attempts:
        attempt["concurrent_load_free"] = not concurrent
    report = {"schema": SCHEMA + "-result", "label": args.label, "plan_sha256": digest(output / "plan.json"),
              "workers": args.workers, "row_count": args.rows, "choice_limit": plan["choice_limit"],
              "campaign_wall_seconds": round(campaign_wall, 6), "deadline_signalled": signalled,
              "supervisor_errors": supervisor_errors, "load_average_start": load_start,
              "load_average_end": os.getloadavg(), "load_average_max_1m": monitor.max_load,
              "foreign_processes_at_start": initial_foreign, "foreign_processes_during": list(monitor.foreign.values()),
              "concurrent_load_detected": concurrent, "cprofile": args.cprofile,
              "attempts": attempts, "worker_reports": workers, "summary": summarize_attempts(attempts, campaign_wall),
              "training_performed": False, "optimizer_updates": 0}
    write(output / "campaign.json", report)
    return report


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
            "not_started": statuses.get("not_started", 0),
            "sampled_choices": choices, "actions_accepted": accepted, "actions_rejected": rejected,
            "campaign_wall_seconds": round(campaign_wall, 6),
            "verified_attempts_per_hour": round(3600 * len(passed) / campaign_wall, 3) if campaign_wall else None,
            "sampled_choices_per_second_end_to_end": round(choices / campaign_wall, 3) if campaign_wall else None,
            "totals_seconds": {key: round(value, 6) for key, value in sorted(totals.items())}}


def summarize(args):
    """Group campaigns by label and worker count; report median and spread."""
    groups = defaultdict(list)
    for path in args.campaigns:
        report = json.loads((Path(path) / "campaign.json").read_text())
        groups[(report["label"], report["workers"])].append(report)
    result = []
    for (label, workers), reports in sorted(groups.items()):
        entry = {"label": label, "workers": workers, "campaigns": len(reports),
                 "concurrent_load_detected": [r["concurrent_load_detected"] for r in reports],
                 "cprofile": [r["cprofile"] for r in reports]}
        metrics = defaultdict(list)
        for report in reports:
            summary = report["summary"]
            metrics["campaign_wall_seconds"].append(summary["campaign_wall_seconds"])
            metrics["sampled_choices"].append(summary["sampled_choices"])
            metrics["verified_attempts_per_hour"].append(summary["verified_attempts_per_hour"])
            metrics["sampled_choices_per_second_end_to_end"].append(summary["sampled_choices_per_second_end_to_end"])
            for key, value in summary["totals_seconds"].items():
                metrics[key].append(value)
            for key in ("natural_endings_verified", "cutoffs_verified", "failures", "not_started",
                        "actions_accepted", "actions_rejected"):
                metrics[key].append(summary[key])
        entry["metrics"] = {key: {"median": statistics.median(values), "min": min(values), "max": max(values),
                                  "n": len(values)} for key, values in sorted(metrics.items())
                            if all(isinstance(v, (int, float)) for v in values)}
        histories = defaultdict(set)
        for report in reports:
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
    work = sub.add_parser("worker")
    work.add_argument("--plan", type=Path, required=True)
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
