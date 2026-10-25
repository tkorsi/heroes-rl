"""Bounded explicitly versioned fixed-weight diagnostics on the exact Tavern-enabled scenario.

Exactly two games exchange the fixed learner/opponent seats. No optimizer is
imported or constructed. Every started attempt is retained; errors stop the
schedule without retries. This is interface validation, not a training run or
an outcome-frequency estimate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.explorer.create_hero_scenario import SCENARIO_ID, verify_hero_scenario
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.contextual_replay import verify_episode
from experiments.headless.env import HeadlessSelfPlayEnv, Limits, VcmiLauncher
from experiments.learner import contextual as c
from experiments.learner.checkpoints import FrozenOpponentPool, load_policy_checkpoint, save_policy_checkpoint
from experiments.learner.contextual_model import ContextualActorCritic, ContextualEncoder, ModelConfig, model_schema_for, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector
from experiments.learner.first_run import implementation_identity
from experiments.learner.inference import digest, history_coverage, source_identity, write_json
from validation.probes.audit_contextual_histories import audit

INTERFACE = c.HEROES_ENVIRONMENT
COLORS = ("red", "blue")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Settings:
    engine_seed: int = 103001
    initialization_seed: int = 103002
    opponent_seed: int = 103003
    policy_seed: int = 103004
    max_requests: int = 128
    max_days: int = 10
    case_seconds: float = 120.0

    def __post_init__(self):
        for name in ("engine_seed", "initialization_seed", "opponent_seed", "policy_seed"):
            value = getattr(self, name)
            require(type(value) is int and 1 <= value <= 2147483647, "Use positive signed-32-bit seeds")
        require(self.initialization_seed != self.opponent_seed, "Use distinct policy initialization seeds")
        require(type(self.max_requests) is int and 1 <= self.max_requests <= 512, "Use 1..512 choices per game")
        require(type(self.max_days) is int and 1 <= self.max_days <= 30, "Use 1..30 game days")
        require(type(self.case_seconds) in (int, float) and math.isfinite(self.case_seconds)
                and 0 < self.case_seconds <= 180, "Use a finite wall limit of at most 180 seconds")

    def schedule(self):
        return [{"episode": seat + 1, "engine_seed": self.engine_seed, "learner_player": seat,
                 "policy_seeds": [self.policy_seed + 2 * seat, self.policy_seed + 2 * seat + 1]}
                for seat in (0, 1)]


def hero_map_identity(path):
    """Separate exact verifier; never weaken ordinary_map_identity's baseline."""
    path = Path(path).resolve()
    verify_hero_scenario(path)
    return {"scenario": SCENARIO_ID, "path": str(path), "sha256": digest(path),
            "layout_seed": 0, "semantics_verified": True, "fixture": False}


@contextmanager
def wall_limit(seconds):
    """Bound a main-thread case; cleanup uses the environment's finite waits."""
    require(hasattr(signal, "setitimer"), "Wall-bounded probe requires POSIX setitimer")
    previous = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "An existing wall timer is active")
    def expired(signum, frame):
        raise TimeoutError("Fixed-weight diagnostic exceeded its declared case wall limit")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def prepare_policy(path, settings, source, *, interface=INTERFACE):
    """Save and reload fresh random weights without consuming caller RNG."""
    with torch.random.fork_rng(devices=[]):
        config = ModelConfig(schema=model_schema_for(interface))
        torch.manual_seed(settings.initialization_seed)
        learner = ContextualActorCritic(config).eval().requires_grad_(False)
        torch.manual_seed(settings.opponent_seed)
        opponent = ContextualActorCritic(config).eval().requires_grad_(False)
        pool = FrozenOpponentPool()
        pool.add("hero-opponent-" + weights_sha256(opponent)[:16], opponent)
        checkpoint = "hero-policy-" + weights_sha256(learner)[:16]
        save_policy_checkpoint(path, learner, checkpoint=checkpoint, source_metadata=source,
                               provenance="random_initialization", opponent_pool=pool, interface=interface)
        loaded = load_policy_checkpoint(path, expected_source_metadata=source, interface=interface)
        require(loaded.weights_sha256 == weights_sha256(learner)
                and loaded.opponent_pool.identities == pool.identities, "Saved random checkpoint roundtrip changed weights")
    loaded.model.requires_grad_(False)
    return loaded


def frozen_fingerprints(loaded, checkpoint):
    require(loaded.opponent_pool is not None and len(loaded.opponent_pool.identities) == 1,
            "Exactly one fixed opponent is required")
    loaded.opponent_pool.assert_unchanged()
    require(not loaded.model.training and all(not p.requires_grad and p.device.type == "cpu"
            for p in loaded.model.parameters()), "Policy must remain frozen on CPU")
    require(weights_sha256(loaded.model) == loaded.weights_sha256, "Learner weights changed")
    return {"checkpoint_file_sha256": digest(checkpoint), "learner_weights_sha256": loaded.weights_sha256,
            "opponents": [list(item) for item in loaded.opponent_pool.identities]}


def reconstruct_behavior(history, decisions, estimates, model, seed, *, schema=c.HEROES_SCHEMA):
    """Independently restart one player's memory and RNG; replay every frame."""
    require(history and all(frame.schema == schema for frame in history), "Expected the explicitly selected history version")
    before = weights_sha256(model)
    require(all(item.prefix.weights_sha256 == before for item in (*decisions, *estimates)),
            "Behavior belongs to different fixed weights")
    encoder = ContextualEncoder(history[0].episode, history[0].player, schema)
    state = model.initial_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_choice = {item.decision_seq: item for item in decisions}
    by_value = {item.decision_seq: item for item in estimates}
    require(len(by_choice) == len(decisions) and len(by_value) == len(estimates), "Duplicate behavior decisions")
    checked_actions = checked_values = checked_conditionals = 0
    with torch.no_grad():
        for frame in history:
            step = model.forward_step(encoder.encode(frame), state)
            state = step.state
            if frame.seq in by_value:
                require(step.decision is not None and step.decision.value.item() == by_value[frame.seq].value,
                        "Saved player value is not exactly reconstructed")
                checked_values += 1
            if frame.seq in by_choice:
                saved = by_choice[frame.seq]
                require(step.decision is not None and step.decision.distribution.support == saved.support,
                        "Offered policy support changed")
                distribution = step.decision.distribution
                require(distribution.sample(generator=generator) == saved.action,
                        "Saved action is not reproduced by its separate player RNG")
                evaluated = distribution.evaluate(saved.action)
                require(evaluated.log_prob.item() == saved.log_prob, "Joint action probability changed")
                require(len(evaluated.conditionals) == len(saved.conditionals), "Conditional probability count changed")
                for actual, expected in zip(evaluated.conditionals, saved.conditionals):
                    require((actual.name, actual.selected_index, actual.support_size, actual.log_prob.item()) ==
                            (expected.name, expected.selected_index, expected.support_size, expected.log_prob),
                            "Conditional action probability or support changed")
                    checked_conditionals += 1
                checked_actions += 1
    require(checked_actions == len(decisions) and checked_values == len(estimates), "Unreconstructed behavior record")
    require(weights_sha256(model) == before, "Behavior reconstruction changed weights")
    return {"actions_resampled": checked_actions, "values_recomputed": checked_values,
            "conditionals_recomputed": checked_conditionals, "maximum_absolute_difference": 0,
            "weights_unchanged": True, "private_memory_reset": True}


def audit_persisted_policy(folder, loaded, policy_seeds, *, interface=INTERFACE):
    """Read the saved artifact and checkpoint anew, independently of collection."""
    profile = c.profile_for(interface)
    raw = json.loads((folder / "neural-rollout.json").read_text())
    collected = CollectedEpisode.from_dict(raw)
    require(collected.to_dict() == raw, "Persisted rollout does not roundtrip exactly")
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest == json.loads(collected.manifest_json), "Rollout embeds another manifest")
    paths = tuple((folder / manifest["players"][color][profile.history_key]["path"]).resolve()
                  for color in COLORS)
    require(all(folder.resolve() in path.parents for path in paths), "History escapes its episode")
    independent = audit(*paths, folder / "manifest.json")
    require(independent["consistent"] and independent["manifest_bound"], "Independent history audit failed")
    models = {loaded.checkpoint: loaded.model}
    models.update({key: loaded.opponent_pool.get(key).model for key, _ in loaded.opponent_pool.identities})
    players = []
    for rollout, path, config in zip(collected.rollouts, paths, collected.model_configs_json):
        require(rollout.origin == "native" and rollout.purpose == "diagnostic"
                and not rollout.optimization_eligible, "Diagnostic rollout became optimizer eligible")
        require((rollout.role == "learner") == (rollout.behavior_checkpoint == loaded.checkpoint),
                "Checkpoint and player roles disagree")
        model = models[rollout.behavior_checkpoint]
        require(model.config.to_dict() == json.loads(config), "Saved model configuration changed")
        require(tuple(c.decode_frame(json.loads(line)) for line in path.read_text().splitlines()) == rollout.history,
                "Rollout changed or omitted persisted visible history")
        result = reconstruct_behavior(rollout.history, rollout.decisions, rollout.estimates,
                                      model, policy_seeds[rollout.player], schema=profile.history_schema)
        players.append({"player": rollout.player, "checkpoint": rollout.behavior_checkpoint,
                        "optimization_eligible": False, **result})
    return {"passed": True, "players": players, "history_audit": independent}


def retain_failed_evidence(env, collector, attempt, error, *, cleanup_seconds=30.0):
    """Best-effort failure evidence without inventing a cutoff or losing the cause.

    Collection can be interrupted outside the environment's normal watchdog.
    Finalize that live process as a technical failure before close() kills it.
    An already finalized native episode remains immutable even when a later
    policy/replay audit fails. The caller always performs process cleanup.
    """
    if env is None:
        return
    folder = env.episode_dir
    if folder is not None:
        attempt["episode_directory"] = str(folder)
    if env.process is not None and env.result is None:
        try:
            with wall_limit(cleanup_seconds):
                reason = "watchdog_timeout" if isinstance(error, TimeoutError) else "protocol_error"
                env._note_problem(reason, "Fixed-weight diagnostic failed: " + attempt["error"])
                end = env._finish(None)
                require(end.status == "failed" and end.outcomes == dict.fromkeys(COLORS)
                        and end.rewards == dict.fromkeys(COLORS),
                        "Failure cleanup produced a scored or truncated boundary")
                attempt.update(episode_status=end.status, termination_reason=end.termination_reason,
                               failure_finalization="persisted")
        except Exception as secondary:
            attempt["failure_finalization_error"] = f"{type(secondary).__name__}: {secondary}"
    if folder is not None and collector is not None and collector.sessions is not None:
        try:
            write_json(folder / "partial-policy-evidence.json", {
                "incomplete": True, "training_input": False,
                "players": [{"player": session.history.player,
                             "decisions": [item.to_dict() for item in session._choices.values()],
                             "history": [c.encode_frame(frame) for frame in session.history.frames]}
                            for session in collector.sessions]})
        except Exception as secondary:
            attempt["partial_evidence_error"] = f"{type(secondary).__name__}: {secondary}"


def run(args):
    interface = getattr(args, "interface", INTERFACE)
    require(interface in (c.HEROES_ENVIRONMENT, c.MANAGEMENT_ENVIRONMENT), "Unsupported diagnostic interface")
    profile = c.profile_for(interface)
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    torch.set_num_threads(1)
    output, game_map, binary, data, replay_binary = (
        Path(getattr(args, name)).expanduser().resolve()
        for name in ("output", "map", "binary", "data", "replay_binary"))
    require(not output.exists(), "Output already exists; preserve all earlier attempts")
    scenario = hero_map_identity(game_map)
    launcher = VcmiLauncher(binary, data, game_map)
    launcher.preflight()
    require(replay_binary.is_file(), "Missing independent native replay binary")
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"], cwd=ROOT,
                   check=True, timeout=60)
    output.mkdir(parents=True)
    source = source_identity(interface)
    implementation = implementation_identity()
    implementation["experiments/explorer/create_hero_scenario.py"] = digest(
        ROOT / "experiments/explorer/create_hero_scenario.py")
    native, runtime = launcher.identity(), runtime_identity(binary.parent)
    checkpoint = output / "fixed-policy.pt"
    loaded = prepare_policy(checkpoint, settings, source, interface=interface)
    fixed = frozen_fingerprints(loaded, checkpoint)
    opponent_id = loaded.opponent_pool.identities[0][0]
    limits = Limits(max_days=settings.max_days, max_actions=settings.max_requests + 1,
                    watchdog_seconds=30, startup_seconds=30, shutdown_seconds=30, bridge_timeout=30)
    plan = {"schema": f"heroes-v{profile.version}-fixed-policy-plan-v1", "interface": interface, "settings": asdict(settings),
            "schedule": settings.schedule(), "source": source, "implementation": implementation, "map": scenario,
            "native": native, "runtime": runtime, "replay_binary_sha256": digest(replay_binary),
            "frozen": fixed, "environment_limits": asdict(limits), "training_performed": False,
            "optimizer_updates": 0, "optimizer_steps": 0, "collector_purpose": "diagnostic",
            "wall_scope": "Each collection, reconstruction and replay share one case deadline; failed-episode finalization has a separate 30-second limit, then finite process cleanup follows"}
    write_json(output / "plan.json", plan)
    plan_hash = digest(output / "plan.json")
    report = {"schema": f"heroes-v{profile.version}-fixed-policy-validation-v1", "plan_sha256": plan_hash, "status": "running",
              "attempts": [], "training_performed": False, "optimizer_updates": 0, "optimizer_steps": 0,
              "training_samples": 0, "ordinary_outcome_claim": False, "frozen_before": fixed}
    write_json(output / "report.json", report)

    def check_identity():
        current = implementation_identity()
        current["experiments/explorer/create_hero_scenario.py"] = digest(ROOT / "experiments/explorer/create_hero_scenario.py")
        require(current == implementation and hero_map_identity(game_map) == scenario, "Source or exact map changed")
        require(launcher.identity() == native and runtime_identity(binary.parent) == runtime
                and digest(replay_binary) == plan["replay_binary_sha256"], "Native runtime changed")
        require(digest(output / "plan.json") == plan_hash and frozen_fingerprints(loaded, checkpoint) == fixed,
                "Plan, checkpoint or fixed weights changed")

    for row in settings.schedule():
        attempt = {**row, "status": "running", "outcomes": dict.fromkeys(COLORS), "rewards": dict.fromkeys(COLORS)}
        report["attempts"].append(attempt)
        write_json(output / "report.json", report)
        started, env, collector = time.monotonic(), None, None
        try:
            with wall_limit(settings.case_seconds):
                check_identity()
                opponent = loaded.opponent_pool.get(opponent_id)
                models = (loaded.model, opponent.model) if row["learner_player"] == 0 else (opponent.model, loaded.model)
                checkpoints = (loaded.checkpoint, opponent_id) if row["learner_player"] == 0 else (opponent_id, loaded.checkpoint)
                collector = ContextualCollector(models, checkpoints, learner_player=row["learner_player"],
                    opponent_pool="hero-fixed-diagnostic-v1", interface=interface, purpose="diagnostic", origin="native",
                    policy_seeds=tuple(row["policy_seeds"]))
                env = HeadlessSelfPlayEnv(launcher, output, limits=limits, ledger=True, replay=True, interface=interface,
                    agents={color: {"checkpoint": key, "kind": "fixed-untrained-recurrent"}
                            for color, key in zip(COLORS, checkpoints)})
                collected = collector.run_episode(env, episode=row["episode"], seed=row["engine_seed"],
                                                  max_requests=settings.max_requests)
                folder = env.episode_dir
                write_json(folder / "neural-rollout.json", collected.to_dict())
                manifest = json.loads((folder / "manifest.json").read_text())
                attempt.update(episode_directory=str(folder), episode_status=manifest["status"],
                               termination_reason=manifest["termination_reason"],
                               coverage=history_coverage(tuple(env.history(color) for color in COLORS)))
                require(manifest["status"] in ("completed", "truncated"), "Native game ended with a technical failure")
                require(manifest["identity"]["map_sha256"] == scenario["sha256"], "Native game used another map")
                require(collector.sessions[0]._state.data_ptr() != collector.sessions[1]._state.data_ptr(),
                        "Players share recurrent memory storage")
                reloaded = load_policy_checkpoint(checkpoint, expected_source_metadata=source, interface=interface)
                reconstruction = audit_persisted_policy(folder, reloaded, row["policy_seeds"], interface=interface)
                write_json(folder / "policy-reconstruction.json", reconstruction)
                remaining = settings.case_seconds - (time.monotonic() - started)
                require(remaining > 0, "No time remains for independent native replay")
                replay = verify_episode(folder, binary=replay_binary, data=data, timeout=remaining)
                require(replay.get("verified") is True, "Independent contextual replay failed")
                check_identity()
                attempt.update(status="passed", outcomes=manifest["outcomes_by_player"], rewards=manifest["rewards_by_player"],
                               reconstruction=reconstruction["players"], replay_verified=True,
                               weights_unchanged=True, private_player_memories=True,
                               files={str(path.relative_to(output)): digest(path) for path in (
                                   folder / "manifest.json", folder / "neural-rollout.json",
                                   folder / "policy-reconstruction.json",
                                   folder / "contextual-verification/contextual-replay-verification.json")})
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}",
                           outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
            retain_failed_evidence(env, collector, attempt, error)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    attempt.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}",
                                   outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
            attempt["wall_seconds"] = time.monotonic() - started
            try:
                check_identity()
            except Exception as error:
                attempt.update(status="failed", identity_error=f"{type(error).__name__}: {error}",
                               outcomes=dict.fromkeys(COLORS), rewards=dict.fromkeys(COLORS))
            report["not_started"] = len(settings.schedule()) - len(report["attempts"])
            report["status"] = "failed" if attempt["status"] != "passed" else "running"
            write_json(output / "report.json", report)
        if attempt["status"] != "passed":
            break
    if all(item["status"] == "passed" for item in report["attempts"]) and not report["not_started"]:
        report["status"] = "passed"
        report["frozen_after"] = frozen_fingerprints(loaded, checkpoint)
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "data", "replay-binary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--map", type=Path, default=ROOT / "experiments/explorer/maps/hero-management-v1.vmap")
    for name, default in asdict(Settings()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float if name == "case_seconds" else int, default=default)
    parser.add_argument("--interface", choices=(c.HEROES_ENVIRONMENT, c.MANAGEMENT_ENVIRONMENT), default=INTERFACE)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "attempts": len(report["attempts"]),
                      "not_started": report["not_started"], "optimizer_steps": 0}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
