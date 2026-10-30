"""Fresh two-policy v8 capture, verification and single-use learner admission.

Serialized reports are evidence, never admission capabilities. The coordinator
creates a binding before launch, observes its owned child completion and verifies
both policies independently before the unchanged PPO kernel sees learner samples.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random

import torch

from experiments.headless import contextual_evidence as evidence, contextual_replay
from . import checkpoints as cp, contextual as c, interaction_guard
from .contextual_model import weights_sha256
from .contextual_rollout import CollectedEpisode, behavior_guard_version
from .inference import source_identity
from .ppo import PPOConfig, _admit

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "heroes-selfplay-batch-v1"
COMPLETION_SCHEMA = "heroes-selfplay-capture-complete-v1"
RECEIPT_SCHEMA = "heroes-selfplay-training-receipt-v1"
PROVENANCE = cp.SELFPLAY_GAMEPLAY_PROVENANCE
POOL = "selfplay-batch-start-v1"
_BINDINGS, _LAUNCHES, _COMPLETIONS, _RECEIPTS = {}, {}, {}, {}
_USED = set()
_SLOTS = set()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def binding(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Expected a regular bound file")
    path = path.resolve()
    return {"path": str(path), **evidence.file_identity(path)}


def checked(value):
    require(type(value) is dict and set(value) == {"path", "sha256", "bytes"}
            and binding(value["path"]) == value, "Bound self-play input changed")
    return Path(value["path"])


def read(path):
    return evidence.strict_json(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")


def time_left(remaining):
    value = remaining()
    if type(value) not in (int, float) or not 0 < value < float("inf"):
        raise TimeoutError("Self-play verification/update deadline exhausted")
    return value


def metadata(plan_sha256, initial_checkpoint, opponent_checkpoint, *, campaign_id, opponent_pool=POOL, purpose="training"):
    value = dict(plan_sha256=plan_sha256, initial_checkpoint=dict(initial_checkpoint),
                 opponent_checkpoint=dict(opponent_checkpoint), campaign_id=campaign_id,
                 opponent_pool=opponent_pool, policy_support=interaction_guard.REVEALED_VERSION,
                 learner_only_ppo=True, frozen_opponent=True, purpose=purpose)
    cp._check_selfplay_gameplay_metadata(value, c.EXPANDED_ENVIRONMENT)
    checked(initial_checkpoint); checked(opponent_checkpoint)
    return value


def load_policy(path, expected_source_metadata, *, allowed_provenance=(PROVENANCE,)):
    require(type(allowed_provenance) is tuple and allowed_provenance
            and set(allowed_provenance) <= {PROVENANCE, cp.BUILTIN_GAMEPLAY_PROVENANCE},
            "Self-play imports only explicit verified gameplay provenance")
    return cp.load_policy_checkpoint(path, expected_source_metadata=expected_source_metadata,
        allowed_provenance=allowed_provenance, interface=c.EXPANDED_ENVIRONMENT)


def implementation_identity():
    from validation.probes.validate_capabilities import capability_implementation_identity
    expected = capability_implementation_identity()
    for name in ("tools/run_selfplay_training.py", "tools/benchmark_copied_training.py", "tools/validate_selfplay_cutoffs.py"):
        expected[name] = binding(ROOT / name)["sha256"]
    return expected


def _plan(path, digest):
    require(binding(path)["sha256"] == digest, "Self-play plan changed")
    value = read(path)
    require(value.get("schema") == SCHEMA and value.get("mode") in ("training", "diagnostic")
            and value.get("policy_support") == interaction_guard.REVEALED_VERSION
            and value.get("opponent_pool") == POOL, "Unsupported self-play plan contract")
    require(value.get("source") == source_identity(c.EXPANDED_ENVIRONMENT)
            and value["source"].get("tracked_worktree_dirty") is False,
            "Self-play source must match its clean committed plan")
    expected = implementation_identity()
    require(value.get("implementation") == expected, "Self-play implementation inventory differs")
    for name, digest in value["implementation"].items():
        path = evidence.inside(ROOT, name)
        require(binding(path)["sha256"] == digest, "Self-play implementation changed: " + name)
    checked(value["checkpoint"]); checked(value["opponent_checkpoint"]); checked(value["replay_binary"])
    cfg, rows = value["configuration"], value["schedule"]
    require(type(rows) is list and len(rows) == 4 and cfg["training_games"] == 4
            and cfg["allow_verified_training_prefix"] is False,
            "Self-play requires all four planned games; partial batches cannot train")
    require([row.get("episode") for row in rows] == [1, 2, 3, 4]
            and [row.get("learner_player") for row in rows] == [0, 1, 0, 1]
            and all(row.get("phase") == value["mode"] for row in rows), "Self-play schedule changed")
    require(all(type(row.get("engine_seed")) is int and 0 < row["engine_seed"] < 2**31
            and type(row.get("policy_seeds")) is list and len(row["policy_seeds"]) == 2
            and all(type(seed) is int and 0 <= seed < 2**63 for seed in row["policy_seeds"]) for row in rows)
            and len({row["engine_seed"] for row in rows}) == 4
            and len({seed for row in rows for seed in row["policy_seeds"]}) == 8,
            "Self-play needs distinct explicit game and per-player policy seeds")
    require(cfg["ordinary_choice_limit"] == 4960 and cfg["reserved_dialog_choices"] == 160
            and cfg["max_total_choices"] == 5120 and cfg["max_optimizer_steps"] == 640,
            "Self-play must preserve the original collection/update limits")
    return value


@dataclass(frozen=True)
class CaptureBinding:
    plan_path: str
    plan_sha256: str
    schedule_index: int
    folder: str
    plan_json: str

    @property
    def plan(self): return json.loads(self.plan_json)

    @property
    def row(self): return self.plan["schedule"][self.schedule_index]

    def recheck(self):
        require(_BINDINGS.get(id(self)) is self and canonical(_plan(self.plan_path, self.plan_sha256)) == self.plan_json,
                "Self-play binding is not this coordinator's unchanged prelaunch capability")


def create_capture_binding(plan_path, plan_sha256, schedule_index, folder):
    plan = _plan(plan_path, plan_sha256)
    require(type(schedule_index) is int and 0 <= schedule_index < 4, "Invalid self-play schedule index")
    folder = Path(folder).resolve()
    expected = Path(plan["execution_output"]).resolve() / f"episode-{plan['schedule'][schedule_index]['episode']:06d}"
    require(folder == expected and not folder.exists(), "Self-play capture must start in its absent planned folder")
    require((plan_sha256, schedule_index) not in _SLOTS, "Self-play capture slot was already bound")
    for key, source, identity, weights in (
            ("checkpoint", "checkpoint_source", "learner_checkpoint", "learner_weights_sha256"),
            ("opponent_checkpoint", "opponent_checkpoint_source", "opponent_checkpoint_id", "opponent_weights_sha256")):
        loaded = load_policy(checked(plan[key]), plan[source])
        require(loaded.checkpoint == plan[identity] and loaded.weights_sha256 == plan[weights]
                and loaded.source_metadata["purpose"] == plan["mode"],
                "Self-play policy file identity or purpose differs from its plan")
    value = CaptureBinding(str(Path(plan_path).resolve()), plan_sha256, schedule_index, str(folder), canonical(plan))
    _BINDINGS[id(value)] = value
    _SLOTS.add((plan_sha256, schedule_index))
    return value


def register_capture_launch(bound, process, token):
    bound.recheck()
    require(id(bound) not in _LAUNCHES and type(token) is str and len(token) == 64
            and all(x in "0123456789abcdef" for x in token)
            and process.poll() is None and type(process.pid) is int and process.pid > 0,
            "Register one live owned child and private completion token")
    _LAUNCHES[id(bound)] = (process, process.pid, token)


def accept_capture_completion(bound, message):
    bound.recheck()
    require(id(bound) in _LAUNCHES and id(bound) not in _COMPLETIONS, "Capture completion is not fresh")
    process, pid, token = _LAUNCHES[id(bound)]
    require(process.pid == pid and process.poll() == 0, "Capture child did not finish successfully")
    require(type(message) is dict and set(message) == {"schema", "token", "episode", "files"}
            and message["schema"] == COMPLETION_SCHEMA and message["token"] == token
            and type(message["episode"]) is int and message["episode"] == bound.row["episode"],
            "Capture completion differs from its actual child launch")
    expected = {name: evidence.file_identity(evidence.inside(bound.folder, name))
                for name in ("manifest.json", "neural-rollout.json")}
    require(message["files"] == expected, "Capture changed after child completion")
    _COMPLETIONS[id(bound)] = canonical(expected)


def _files(folder, manifest):
    names = {"manifest.json", "neural-rollout.json", "contextual-finalization.json", "logs/process.log",
             "engine/experiment-metadata.json", *manifest["replay"]["files"]}
    for color in ("red", "blue"):
        declaration = manifest["players"][color]
        names.update((f"agents/{color}/bridge.jsonl", declaration[c.EXPANDED_PROFILE.history_key]["path"],
                      declaration["live_evidence"]["path"]))
    return {name: evidence.file_identity(evidence.inside(folder, name)) for name in sorted(names)}


def _check_collected(bound, collected, models):
    plan, row = bound.plan, bound.row
    manifest = json.loads(collected.manifest_json)
    require(manifest == read(Path(bound.folder) / "manifest.json")
            and manifest["episode"] == str(row["episode"]) and manifest["seed"] == row["engine_seed"]
            and manifest["environment"] == c.EXPANDED_ENVIRONMENT
            and manifest["status"] in ("completed", "truncated"), "Self-play capture identity/boundary differs")
    require(manifest["identity"].get("native_runtime_sha256") == plan["runtime"]
            and all(manifest["identity"].get(key) == value for key, value in plan["native"].items()),
            "Self-play capture used another engine/map/runtime")
    require(models[0] is not models[1] and not ({p.data_ptr() for p in models[0].parameters()}
            & {p.data_ptr() for p in models[1].parameters()}), "Learner and frozen opponent must own separate tensors")
    cfg = plan["configuration"]
    for player, (rollout, model, config) in enumerate(zip(collected.rollouts, models, collected.model_configs_json)):
        learner = player == row["learner_player"]
        identity = plan["learner_checkpoint" if learner else "opponent_checkpoint_id"]
        digest = plan["learner_weights_sha256" if learner else "opponent_weights_sha256"]
        require(rollout.player == player and rollout.role == ("learner" if learner else "frozen_opponent")
                and rollout.behavior_checkpoint == identity and rollout.weights_sha256 == digest == weights_sha256(model)
                and json.loads(config) == model.config.to_dict() and rollout.opponent_pool == POOL
                and rollout.purpose == plan["mode"] and rollout.origin == "native",
                "Self-play policy, role, weights, or purpose differs")
        require((behavior_guard_version(rollout.decisions) == plan["policy_support"]
                or not rollout.decisions and not rollout.samples)
                and len(rollout.decisions) <= cfg["max_total_choices"]
                and all(rollout.history[d.decision_seq].day <= cfg["max_days"] for d in rollout.decisions),
                "Self-play support or per-player choice/day budget differs")
        require(not learner or plan["mode"] != "training" or rollout.optimization_eligible,
                "Failed or incomplete learner trajectory cannot train")
        require(learner or not rollout.optimization_eligible, "Frozen opponent became optimizer eligible")
    return manifest


@dataclass(frozen=True)
class VerifiedTrainingReceipt:
    binding: CaptureBinding
    collected: CollectedEpisode
    files_json: str
    verification_json: str

    @property
    def rollout(self): return self.collected.rollouts[self.binding.row["learner_player"]]

    def to_dict(self):
        return dict(schema=RECEIPT_SCHEMA, plan_sha256=self.binding.plan_sha256,
                    schedule_index=self.binding.schedule_index, folder=self.binding.folder,
                    learner_player=self.rollout.player, samples=len(self.rollout.samples),
                    files=json.loads(self.files_json), verification=json.loads(self.verification_json),
                    admission_from_file=False, learner_only_ppo=True)


def reconstruct_policies(collected, models, seeds, remaining):
    """Restart each player's memory and dedicated RNG independently."""
    from validation.probes.validate_hero_policy import reconstruct_behavior
    policies = []
    for player, (rollout, model) in enumerate(zip(collected.rollouts, models)):
        time_left(remaining)
        result = reconstruct_behavior(rollout.history, rollout.decisions, rollout.estimates, model,
                                      seeds[player], schema=c.EXPANDED_SCHEMA)
        require(result.get("maximum_absolute_difference") == 0 and result.get("private_memory_reset") is True
                and result.get("weights_unchanged") is True
                and result.get("actions_resampled") == len(rollout.decisions)
                and result.get("values_recomputed") == len(rollout.estimates), "Two-policy exact reconstruction failed")
        policies.append(dict(player=player, checkpoint=rollout.behavior_checkpoint, **result))
    return policies


def verify_training_capture(bound, learner_model, opponent_model, *, replay_binary, data, remaining):
    from experiments.headless.contextual_replay_history import verify_histories
    time_left(remaining); bound.recheck()
    require(id(bound) in _COMPLETIONS and not any(receipt.binding is bound for receipt in _RECEIPTS.values()),
            "Only a freshly completed capture can be verified once")
    folder, plan, row = Path(bound.folder), bound.plan, bound.row
    require(Path(replay_binary).resolve() == checked(plan["replay_binary"]).resolve()
            and Path(data).resolve() == Path(plan["data"]).resolve(), "Self-play verifier runtime changed")
    require(canonical({name: evidence.file_identity(folder / name) for name in
            ("manifest.json", "neural-rollout.json")}) == _COMPLETIONS[id(bound)], "Completed capture changed")
    raw = read(folder / "neural-rollout.json")
    collected = CollectedEpisode.from_dict(raw)
    require(collected.to_dict() == raw, "Paired self-play artifact does not roundtrip exactly")
    models = (learner_model, opponent_model) if row["learner_player"] == 0 else (opponent_model, learner_model)
    manifest = _check_collected(bound, collected, models)
    original = _files(folder, manifest)
    final = contextual_replay.bound_evidence(folder, manifest, Path(replay_binary).resolve())
    histories = verify_histories(folder, manifest, final)
    require(histories.get("verified") is True and set(histories["players"]) == {"red", "blue"},
            "Both player histories must reconstruct")
    policies = reconstruct_policies(collected, models, row["policy_seeds"], remaining)
    output = folder / "training-verification"
    replay = contextual_replay.verify_episode(folder, binary=replay_binary, data=data,
                                             output=output, timeout=time_left(remaining))
    require(replay.get("schema") == contextual_replay.REPORT_SCHEMA and replay.get("verified") is True
            and replay.get("histories") == histories, "Exact two-player native replay failed")
    require(read(output / "contextual-replay-verification.json") == replay, "Replay report changed")
    native = read(output / "native-report.json")
    require(native == {key: value for key, value in replay["native"].items() if key not in ("returncode", "seconds")},
            "Executed native report changed")
    contextual_replay.validate_native_report(replay["native"], replay["wire"], final,
                                            cutoff=manifest["status"] == "truncated")
    bound.recheck(); _check_collected(bound, collected, models); time_left(remaining)
    require(_files(folder, manifest) == original, "Capture changed during verification")
    proof = dict(histories=histories, policies=policies, replay_verified=True,
                 replay_binary=binding(replay_binary), learner_only_ppo=True)
    write_new(folder / "training-policy-reconstruction.json", proof)
    for name in ("training-policy-reconstruction.json", "training-verification/contextual-replay-verification.json",
                 "training-verification/native-report.json"):
        original[name] = evidence.file_identity(evidence.inside(folder, name))
    receipt = VerifiedTrainingReceipt(bound, collected, canonical(original), canonical(proof))
    write_new(folder / "training-receipt.json", receipt.to_dict())
    _RECEIPTS[id(receipt)] = receipt
    return receipt


def _admit_selfplay(receipts, model, *, plan_sha256):
    require(type(receipts) is tuple and len(receipts) == 4 and len({id(r) for r in receipts}) == 4,
            "Self-play admission requires exactly four unique verified games")
    require(all(type(r) is VerifiedTrainingReceipt and _RECEIPTS.get(id(r)) is r
                and (r.binding.plan_sha256, r.binding.schedule_index) not in _USED
                for r in receipts), "Self-play receipt is forged, diagnostic or already consumed")
    require([r.binding.schedule_index for r in receipts] == [0, 1, 2, 3], "Self-play receipt order changed")
    for receipt in receipts:
        bound = receipt.binding; bound.recheck()
        require(bound.plan_sha256 == plan_sha256 and bound.plan["mode"] == "training", "Self-play receipt plan/purpose differs")
        folder = Path(bound.folder)
        require(read(folder / "training-receipt.json") == receipt.to_dict(), "Self-play receipt evidence changed")
        for name, expected in json.loads(receipt.files_json).items():
            require(evidence.file_identity(evidence.inside(folder, name)) == expected, "Admitted capture evidence changed")
        require(receipt.rollout.weights_sha256 == weights_sha256(model), "Self-play learner is off-policy")
    rollouts = tuple(receipt.rollout for receipt in receipts)
    _admit(rollouts, model, native=True)
    return rollouts


def release_receipts(receipts):
    """Discard large live capabilities after use; persisted evidence cannot remint them."""
    for receipt in receipts:
        require(type(receipt) is VerifiedTrainingReceipt, "Expected a typed self-play receipt")
        _RECEIPTS.pop(id(receipt), None)
        key = id(receipt.binding)
        for registry in (_BINDINGS, _LAUNCHES, _COMPLETIONS):
            registry.pop(key, None)


def native_update(model, optimizer, receipts, *, plan_sha256, config, generator, remaining, max_optimizer_steps=640):
    from .builtin_gameplay import transactional_update
    time_left(remaining)
    rollouts = _admit_selfplay(receipts, model, plan_sha256=plan_sha256)
    cfg = receipts[0].binding.plan["configuration"]
    require(type(config) is PPOConfig and config.to_dict() == cfg["ppo"]
            and type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1
            and all(canonical(optimizer.param_groups[0][key]) == canonical(value) for key, value in cfg["adam"].items())
            and max_optimizer_steps == cfg["max_optimizer_steps"], "Self-play changed PPO or Adam settings")
    windows = sum((len(item.samples) + config.decisions_per_window - 1) // config.decisions_per_window for item in rollouts)
    require(config.epochs * ((windows + config.windows_per_minibatch - 1) // config.windows_per_minibatch)
            <= max_optimizer_steps, "Complete self-play update exceeds the declared cap")
    _USED.update((r.binding.plan_sha256, r.binding.schedule_index) for r in receipts)
    release_receipts(receipts)
    result = transactional_update(model, optimizer, rollouts, config=config, generator=generator,
        remaining=lambda: time_left(remaining), max_optimizer_steps=max_optimizer_steps, enable_training=True)
    return replace(result, purpose=PROVENANCE), rollouts


def save_checkpoint_pair(output, model, optimizer, *, checkpoint, ppo_config, progress, metadata,
                         opponent_pool, sampler_state, generator):
    """Write new policy/resume files and verify every complete resume component."""
    from .gameplay import assert_same_state
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    policy, resume = output / "after-policy.pt", output / "after-resume.pt"
    before = deepcopy(dict(model=dict(model.state_dict()), adam=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), python_rng=random.getstate(), minibatch_rng=generator.get_state(),
        progress=vars(progress), sampler=sampler_state, pool=opponent_pool.state_dict()))
    kwargs = dict(checkpoint=checkpoint, source_metadata=metadata, opponent_pool=opponent_pool,
                  generators={"minibatch": generator}, interface=c.EXPANDED_ENVIRONMENT, provenance=PROVENANCE)
    cp.save_policy_checkpoint(policy, model, **kwargs)
    cp.save_checkpoint(resume, model, optimizer, ppo_config=ppo_config, progress=progress,
                       sampler_state=sampler_state, at_rollout_boundary=True, **kwargs)
    try:
        loaded = load_policy(policy, metadata)
        trial = torch.optim.Adam(loaded.model.parameters(), **optimizer.defaults)
        rng = torch.Generator()
        state = cp.load_checkpoint(resume, loaded.model, trial, expected_ppo_config=ppo_config,
            expected_source_metadata=metadata, generators={"minibatch": rng}, at_rollout_boundary=True,
            interface=c.EXPANDED_ENVIRONMENT, allowed_provenance=(PROVENANCE,))
        after = dict(model=dict(loaded.model.state_dict()), adam=trial.state_dict(),
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(), minibatch_rng=rng.get_state(),
            progress=vars(state.progress), sampler=state.sampler_state, pool=state.opponent_pool.state_dict())
        assert_same_state(before, after)
        require(state.checkpoint == loaded.checkpoint == checkpoint and loaded.weights_sha256 == weights_sha256(model),
                "Self-play policy/resume identity differs")
        return dict(policy=binding(policy), resume=binding(resume), checkpoint=checkpoint,
                    weights_sha256=loaded.weights_sha256, exact_model_adam_rng_progress_pool=True,
                    progress=vars(progress), sampler_state=sampler_state, metadata=metadata)
    finally:
        torch.set_rng_state(before["torch_rng"])
        random.setstate(before["python_rng"])
        generator.set_state(before["minibatch_rng"])


def restore_checkpoint_pair(pair, *, expected_metadata, expected_ppo_config, expected_optimizer_steps):
    """Restore a previously published training pair after the caller audits its chain.

    This does not admit saved trajectories or apply an update. The controller
    must independently bind publication, source compatibility and runtime before
    calling. Success installs the checkpoint RNGs; failure preserves caller RNGs.
    """
    from .builtin_training_resume import ParentState, _exact
    original_torch, original_python = torch.get_rng_state().clone(), random.getstate()
    try:
        fields = {"policy", "resume", "checkpoint", "weights_sha256", "exact_model_adam_rng_progress_pool",
                  "progress", "sampler_state", "metadata"}
        require(type(pair) is dict and set(pair) == fields
                and pair["exact_model_adam_rng_progress_pool"] is True
                and type(expected_metadata) is dict and expected_metadata.get("purpose") == "training"
                and canonical(pair["metadata"]) == canonical(expected_metadata)
                and type(expected_ppo_config) is dict
                and type(expected_optimizer_steps) is int and expected_optimizer_steps > 0,
                "Restore requires an exact published training checkpoint descriptor")
        policy_path, resume_path = checked(pair["policy"]), checked(pair["resume"])
        options = dict(expected_source_metadata=expected_metadata, allowed_provenance=(PROVENANCE,),
                       interface=c.EXPANDED_ENVIRONMENT)
        policy_raw = cp._read(policy_path, "policy", **options)
        resume_raw = cp._read(resume_path, "optimizer", **options)
        # A matching label or weight digest is insufficient: the paired policy
        # must also carry the identical archived pool and full sampling state.
        for name in policy_raw.keys() - {"kind", "resume"}:
            _exact(policy_raw[name], resume_raw[name])
        require(policy_raw["checkpoint"] == pair["checkpoint"]
                and policy_raw["weights_sha256"] == pair["weights_sha256"]
                and resume_raw["resume"]["progress"] == pair["progress"]
                and resume_raw["resume"]["sampler_state"] == pair["sampler_state"]
                and type(pair["sampler_state"].get("optimizer_steps")) is int
                and pair["sampler_state"]["optimizer_steps"] == expected_optimizer_steps,
                "Published checkpoint identity, progress or optimizer count differs")
        policy = load_policy(policy_path, expected_metadata)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters())
        generator = torch.Generator(device="cpu")
        restored = cp.load_checkpoint(resume_path, model, optimizer, expected_ppo_config=expected_ppo_config,
            generators={"minibatch": generator}, at_rollout_boundary=True, **options)
        expected_rng = resume_raw["rng"]
        require(set(expected_rng["torch_generators"]) == {"minibatch"}
                and expected_rng["python_generators"] == {}, "Unexpected self-play RNG inventory")
        for actual, expected in (
                (dict(model.state_dict()), resume_raw["model_state"]),
                (optimizer.state_dict(), resume_raw["resume"]["optimizer_state"]),
                (restored.opponent_pool.state_dict(), resume_raw["opponent_pool"]),
                (torch.get_rng_state(), expected_rng["torch_cpu"]),
                (random.getstate(), expected_rng["python"]),
                (generator.get_state(), expected_rng["torch_generators"]["minibatch"]),
                (vars(restored.progress), pair["progress"]),
                (restored.sampler_state, pair["sampler_state"])):
            _exact(actual, expected)
        require(restored.checkpoint == pair["checkpoint"] and weights_sha256(model) == pair["weights_sha256"]
                and optimizer.state and any(item["step"].item() > 0 for item in optimizer.state.values())
                and all(item["step"].item() <= expected_optimizer_steps for item in optimizer.state.values()),
                "Restored model/Adam differs from the published complete update")
        checked(pair["policy"]); checked(pair["resume"])
        model.eval().requires_grad_(False)
        return ParentState(model, optimizer, generator, restored.opponent_pool, restored.progress,
            deepcopy(restored.sampler_state), deepcopy(expected_metadata), restored.checkpoint,
            pair["weights_sha256"], {"policy": dict(pair["policy"]), "resume": dict(pair["resume"]),
                "exact_model_adam_rng_progress_pool": True}, expected_optimizer_steps)
    except BaseException:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)
        raise
