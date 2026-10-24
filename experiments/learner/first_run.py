"""Prepare an immutable first-update proposal; never launch a game or optimizer.

The readiness block records reviewed, source-bound validation. Preparation can
produce an incomplete proposal, but only all required checks permit requesting
launch approval. That approval is always a separate human decision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/first-training-v1.json"
SCHEMA = "heroes-first-training-plan-v1"
INTERFACE = "adventure-armies-v1"
READINESS_CHECKS = ("full_tests", "prepared", "transfer_red", "transfer_blue", "natural_red", "natural_blue",
                    "failure_after_ack", "v3_checkpoint_compatibility", "dormant_executor_review")
# Fixed first proposal: changing seeds, budgets or coefficients requires a new
# reviewed configuration/version, not a larger invocation-time override.
_EXPECTED = {
    "schema": "heroes-first-training-config-v1", "interface": INTERFACE,
    "initialization_seed": 99001, "opponent_seed": 99002, "training_engine_seeds": [99101],
    "evaluation_engine_seeds": [99201, 99202], "policy_seed": 99003,
    "max_requests": 256, "max_days": 30, "total_wall_seconds": 3600, "torch_threads": 1,
    "max_updates": 1, "max_optimizer_steps": 16, "adam": {"lr": 0.0003},
    "ppo": {"gamma": 1.0, "gae_lambda": 1.0, "clip_ratio": 0.2, "value_clip": 0.2,
            "value_coefficient": 0.5, "entropy_coefficient": 0.0, "max_grad_norm": 0.5,
            "epochs": 1, "decisions_per_window": 16, "windows_per_minibatch": 2,
            "normalize_advantages": True},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_config(value):
    """Reject unknown fields, numeric aliases and changes to the reviewed budget."""
    require(type(value) is dict, "Configuration must be an object")
    def exact_types(actual, expected):
        if type(actual) is not type(expected):
            return False
        if type(expected) is dict:
            return actual.keys() == expected.keys() and all(exact_types(actual[k], v) for k, v in expected.items())
        if type(expected) is list:
            return len(actual) == len(expected) and all(exact_types(a, b) for a, b in zip(actual, expected))
        return True
    require(exact_types(value, _EXPECTED), "Configuration contains unsupported fields or types")
    try:
        require(canonical(value) == canonical(_EXPECTED), "Configuration differs from the bounded first-run proposal")
    except (TypeError, OverflowError) as error:
        raise ValueError("Configuration must contain only finite JSON values") from error
    return deepcopy(value)


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read_json(path):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(Path(path).read_text())


def implementation_identity():
    """Bind the complete authored engine, adapter, verifier, learner and tests."""
    from experiments.episodes.scoring import engine_recipe_digests
    result = engine_recipe_digests(ROOT)
    patterns = ("experiments/headless/*.py", "experiments/learner/*.py", "experiments/episodes/*.py",
                "validation/probes/*.py", "tests/*.py", "tests/native/*", "dependencies/*")
    paths = {ROOT / "tools/prepare_vcmi.py", CONFIG}
    for pattern in patterns:
        paths.update(ROOT.glob(pattern))
    for path in sorted(paths):
        if path.is_file():
            result[str(path.relative_to(ROOT))] = file_identity(path)["sha256"]
    return dict(sorted(result.items()))


def check_readiness(evidence_path, *, source, implementation, native_identity, native_runtime, map_identity):
    """Check retained review evidence; this performs no new game or optimization."""
    if evidence_path is None:
        return {"ready_to_request_launch": False, "reason": "Required validation evidence was not supplied",
                "checks": {}, "remaining_scope": ["All version-four validation and executor review must be bound before launch approval."]}
    evidence_path = Path(evidence_path).expanduser().resolve()
    evidence = read_json(evidence_path)
    require(type(evidence) is dict and evidence.get("schema") == "heroes-town-transfer-validation-v1",
            "Expected the versioned troop-transfer validation evidence")
    block = evidence.get("first_training_readiness")
    expected_keys = {"interface", "source_sha256", "implementation_sha256", "native_identity", "native_runtime",
                     "map_identity", "checks", "remaining_scope"}
    require(type(block) is dict and set(block) == expected_keys, "Readiness evidence fields differ")
    expected = {"interface": INTERFACE, "source_sha256": source["source_sha256"],
                "implementation_sha256": implementation, "native_identity": native_identity,
                "native_runtime": native_runtime, "map_identity": map_identity}
    for key, value in expected.items():
        require(canonical(block[key]) == canonical(value), "Readiness identity changed: " + key)
    require(type(block["checks"]) is dict and set(block["checks"]) == set(READINESS_CHECKS),
            "Readiness needs every specifically named validation and review")
    for name, check in block["checks"].items():
        require(type(check) is dict and set(check) == {"status", "artifacts"} and check["status"] == "passed"
                and type(check["artifacts"]) is list and check["artifacts"], "Readiness check is not passed: " + name)
        for item in check["artifacts"]:
            require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}
                    and type(item["path"]) is str and type(item["bytes"]) is int and item["bytes"] > 0
                    and type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                    "Malformed readiness artifact: " + name)
            relative = Path(item["path"])
            target = (ROOT / relative).resolve()
            require(not relative.is_absolute() and ".." not in relative.parts and target.is_relative_to(ROOT),
                    "Readiness artifact must remain within this workspace")
            require(file_identity(target) == {k: item[k] for k in ("sha256", "bytes")},
                    "Readiness artifact changed: " + item["path"])
    require(type(block["remaining_scope"]) is list and block["remaining_scope"]
            and all(type(item) is str and item.strip() for item in block["remaining_scope"]),
            "Readiness must preserve explicit unverified scope")
    return {"ready_to_request_launch": True, "evidence": {"path": str(evidence_path), **file_identity(evidence_path)},
            "checks": deepcopy(block["checks"]), "remaining_scope": list(block["remaining_scope"]),
            "meaning": "Required retained checks and their current identities match; execution still lacks human authorization."}


def prepare_checkpoint(path, configuration, source):
    """Create fresh version-four random actor/value and an independent frozen opponent."""
    import torch
    from .checkpoints import FrozenOpponentPool, load_policy_checkpoint, save_policy_checkpoint
    from .contextual_model import ContextualActorCritic, ModelConfig, model_schema_for, weights_sha256
    configuration = validate_config(configuration)
    with torch.random.fork_rng(devices=[]):
        config = ModelConfig(schema=model_schema_for(INTERFACE))
        torch.manual_seed(configuration["initialization_seed"])
        model = ContextualActorCritic(config).eval()
        learner_digest = weights_sha256(model)
        torch.manual_seed(configuration["opponent_seed"])
        opponent = ContextualActorCritic(config).eval()
        pool = FrozenOpponentPool()
        pool.add("random-v4-" + weights_sha256(opponent)[:16], opponent)
        checkpoint_id = "random-v4-" + learner_digest[:16]
        save_policy_checkpoint(path, model, checkpoint=checkpoint_id, source_metadata=source,
                               opponent_pool=pool, interface=INTERFACE)
        loaded = load_policy_checkpoint(path, expected_source_metadata=source, interface=INTERFACE)
        require(loaded.weights_sha256 == learner_digest and loaded.opponent_pool.identities == pool.identities,
                "Fresh checkpoint roundtrip changed policy or opponent")
        loaded.opponent_pool.assert_unchanged()
    return {"path": str(Path(path).resolve()), **file_identity(path), "checkpoint": checkpoint_id,
            "weights_sha256": learner_digest, "model_config": config.to_dict(),
            "provenance": "random_initialization", "opponents": [list(item) for item in pool.identities]}


def schedule(configuration):
    rows = []
    for phase, seeds in (("before", configuration["evaluation_engine_seeds"]),
                         ("training", configuration["training_engine_seeds"]),
                         ("after", configuration["evaluation_engine_seeds"])):
        index = 0
        for seed in seeds:
            for seat in (0, 1):
                offset = 1000 if phase == "training" else 2000
                rows.append({"phase": phase, "engine_seed": seed, "learner_player": seat,
                             "policy_seeds": [configuration["policy_seed"] + offset + 2 * index + side
                                              for side in (0, 1)],
                             "optimizer_eligible": phase == "training"})
                index += 1
    return rows


def prepare(args):
    import torch
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.headless.env import VcmiLauncher
    from .inference import ordinary_map_identity, source_identity
    config_path = args.config.expanduser().resolve()
    require(config_path == CONFIG, "Use the tracked first-training configuration")
    configuration = validate_config(read_json(config_path))
    torch.set_num_threads(configuration["torch_threads"])
    binary, data, game_map, replay = (path.expanduser().resolve() for path in
                                     (args.binary, args.data, args.game_map, args.replay_binary))
    launcher = VcmiLauncher(binary, data, game_map)
    launcher.preflight()
    require(replay == binary.parent / "vcmireplay", "Verifier must belong to the same native runtime")
    contents = launcher.ai_library.read_bytes()
    require(all(marker in contents for marker in (b"adventure-armies-v1", b"heroes-player-history-v4", b"heroes-context-actions-v4")),
            "Native module does not implement the prepared version-four profile")
    require(b"heroes-contextual-terminal-barrier-v1" in binary.read_bytes(), "Client lacks native terminal drainage")
    checked = subprocess.run([sys.executable, str(ROOT / "tools/prepare_vcmi.py"), "--check"],
                             cwd=ROOT, text=True, capture_output=True)
    require(checked.returncode == 0, "Prepared native sources do not exactly match the authored recipe: " + checked.stderr)
    source, implementation = source_identity(INTERFACE), implementation_identity()
    native, runtime, map_identity = launcher.identity(), runtime_identity(binary.parent), ordinary_map_identity(game_map)
    readiness = check_readiness(args.readiness_evidence, source=source, implementation=implementation,
                               native_identity=native, native_runtime=runtime, map_identity=map_identity)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "prepared-check.log").write_text(checked.stdout + checked.stderr)
    checkpoint = prepare_checkpoint(output / "initial-policy.pt", configuration, source)
    require(source_identity(INTERFACE)["source_sha256"] == source["source_sha256"]
            and implementation_identity() == implementation and launcher.identity() == native
            and runtime_identity(binary.parent) == runtime, "Implementation changed during preparation")
    plan = {"schema": SCHEMA, "authorization": "not_granted", "status": "prepared_not_executed",
            "training_performed": False, "optimizer_updates": 0, "configuration": configuration,
            "configuration_file": {"path": str(config_path), **file_identity(config_path)},
            "source": source, "implementation_sha256": implementation, "native_identity": native,
            "native_runtime": runtime, "map_identity": map_identity,
            "binary": str(binary), "data": str(data), "game_map": str(game_map), "replay_binary": str(replay),
            "initial_checkpoint": checkpoint, "readiness": readiness, "attempt_schedule": schedule(configuration),
            "stop_rules": ["No failed or invalid attempt is replaced or admitted to PPO.",
                           "Any history, replay, behavior-recompute, native or ownership error stops execution.",
                           "The global time budget includes collection, verification, optimization and evaluation.",
                           "Zero natural outcomes alone does not cancel an otherwise valid update; it supplies no observed victory signal."],
            "resume": {"boundary": "between_rollouts_only", "immutable_initial": True,
                       "optimizer_checkpoint": "resume.pt", "resume_execution_implemented": False,
                       "note": "Executor must verify a save/load roundtrip; a stopped process is never automatically restarted."}}
    plan_path = output / "plan.json"
    with plan_path.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    command = [sys.executable, "-m", "experiments.learner.gameplay", "--plan", str(plan_path),
               "--approved-plan-sha256", file_identity(plan_path)["sha256"], "--output", str(output / "approved-execution")]
    proposal = {"schema": "heroes-first-training-preparation-result-v1", "training_performed": False,
                "ready_to_request_launch": readiness["ready_to_request_launch"],
                "plan": {"path": str(plan_path), **file_identity(plan_path)},
                "proposed_command_requires_future_human_authorization": shlex.join(command)}
    with (output / "preparation.json").open("x") as stream:
        json.dump(proposal, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    return proposal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "vcmi/build/bin/vcmiclient")
    parser.add_argument("--replay-binary", type=Path, default=ROOT / "vcmi/build/bin/vcmireplay")
    parser.add_argument("--data", type=Path, default=Path.home() / "Library/Application Support/vcmi")
    parser.add_argument("--map", dest="game_map", type=Path, default=ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")
    parser.add_argument("--readiness-evidence", type=Path)
    print(json.dumps(prepare(parser.parse_args(argv)), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
