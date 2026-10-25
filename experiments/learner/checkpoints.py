"""Strict local checkpoints and frozen opponents; no environment or optimizer loop.

Resume is supported only between rollouts. Engine processes, sockets and live
recurrent memories are deliberately not resumable checkpoint contents.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch

SCHEMA = "heroes-context-checkpoint-v1"
BUILTIN_GAMEPLAY_PROVENANCE = "approved_builtin_gameplay_v1"
OPTIMIZER_PROVENANCES = ("synthetic_optimizer_test", "approved_gameplay", BUILTIN_GAMEPLAY_PROVENANCE)
PROVENANCES = ("random_initialization", *OPTIMIZER_PROVENANCES)


def _contracts(interface):
    from . import contextual as contract
    from .contextual_model import distribution_schema_for, model_schema_for
    profile = contract.profile_for(interface)
    if interface != profile.environment:
        raise ValueError("Checkpoint interface must name its explicit environment")
    return {"environment": profile.environment, "history_schema": profile.history_schema,
            "action_schema": profile.action_schema, "encoder_schema": model_schema_for(interface),
            "distribution_schema": distribution_schema_for(interface)}


def _json(value):
    def check(node):
        if type(node) is dict:
            if any(type(key) is not str for key in node):
                raise ValueError("Metadata keys must be strings")
            for child in node.values(): check(child)
        elif type(node) in (tuple, list):
            for child in node: check(child)
        elif node is not None and type(node) not in (str, int, float, bool):
            raise ValueError("Checkpoint metadata must be JSON values")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("Checkpoint identity must be a nonempty string")


def _model_api():
    from .contextual_model import ContextualActorCritic, ModelConfig, weights_sha256
    return ContextualActorCritic, ModelConfig, weights_sha256


def model_weight_digest(model):
    return _model_api()[2](model)


def _weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validate_weights(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Checkpoint model keys differ")
    for name, value in state.items():
        target = expected[name]
        if (type(value) is not torch.Tensor or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all().item()):
            raise ValueError("Invalid model tensor: " + name)


def _model_from_payload(payload):
    model_type, config_type, digest = _model_api()
    # Loading fixed weights must not consume the caller's sampling RNG.
    rng = torch.get_rng_state()
    try:
        model = model_type(config=config_type.from_dict(payload["model_config"]))
    finally:
        torch.set_rng_state(rng)
    _validate_weights(model, payload["model_state"])
    model.load_state_dict(payload["model_state"], strict=True)
    if digest(model) != payload["weights_sha256"]:
        raise ValueError("Checkpoint weights do not match their identity")
    return model


@dataclass(frozen=True)
class FrozenOpponent:
    checkpoint: str
    weights_sha256: str
    model: object

    def assert_unchanged(self):
        if (self.model.training or any(p.requires_grad for p in self.model.parameters())
                or model_weight_digest(self.model) != self.weights_sha256):
            raise ValueError("Frozen opponent was modified")


class FrozenOpponentPool:
    """Explicit checkpoint order, uniformly sampled with a caller-owned RNG."""

    def __init__(self):
        self._entries = {}

    def add(self, checkpoint_id, model):
        _text(checkpoint_id)
        if checkpoint_id in self._entries:
            raise ValueError("Duplicate frozen checkpoint identity")
        frozen = deepcopy(model).eval()
        frozen.requires_grad_(False)
        entry = FrozenOpponent(checkpoint_id, model_weight_digest(frozen), frozen)
        self._entries[checkpoint_id] = entry
        return entry

    def get(self, checkpoint_id):
        entry = self._entries[checkpoint_id]
        entry.assert_unchanged()
        return entry

    def sample(self, generator):
        if type(generator) is not random.Random or not self._entries:
            raise ValueError("Sampling requires a nonempty pool and a dedicated Python RNG")
        self.assert_unchanged()
        return self.get(generator.choice(tuple(self._entries)))

    @property
    def identities(self):
        self.assert_unchanged()
        return tuple((entry.checkpoint, entry.weights_sha256) for entry in self._entries.values())

    def assert_unchanged(self):
        for entry in self._entries.values(): entry.assert_unchanged()

    def state_dict(self):
        self.assert_unchanged()
        return [{"checkpoint": entry.checkpoint, "weights_sha256": entry.weights_sha256,
                 "model_config": entry.model.config.to_dict(), "model_state": _weights(entry.model)}
                for entry in self._entries.values()]

    @classmethod
    def from_state_dict(cls, state):
        if type(state) is not list:
            raise ValueError("Invalid frozen pool state")
        pool = cls()
        for entry in state:
            if type(entry) is not dict or set(entry) != {"checkpoint", "weights_sha256", "model_config", "model_state"}:
                raise ValueError("Invalid frozen opponent checkpoint")
            pool.add(entry["checkpoint"], _model_from_payload(entry))
        return pool


def _rng_state(generators, python_generators):
    for name, generator in generators.items():
        _text(name)
        if type(generator) is not torch.Generator or generator.device.type != "cpu":
            raise ValueError("This checkpoint version supports CPU sampling generators only")
    for name, generator in python_generators.items():
        _text(name)
        if type(generator) is not random.Random:
            raise ValueError("Expected dedicated Python sampling RNG")
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
            "torch_generators": {name: generator.get_state().clone() for name, generator in generators.items()},
            "python_generators": {name: generator.getstate() for name, generator in python_generators.items()}}


def _validate_rng(state, generators, python_generators):
    if type(state) is not dict or set(state) != {"python", "torch_cpu", "torch_generators", "python_generators"}:
        raise ValueError("Invalid checkpoint RNG fields")
    if set(state["torch_generators"]) != set(generators) or set(state["python_generators"]) != set(python_generators):
        raise ValueError("Named sampling RNG identities differ")
    _rng_state(generators, python_generators)
    random.Random().setstate(state["python"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    for value in state["python_generators"].values(): random.Random().setstate(value)
    for value in state["torch_generators"].values(): torch.Generator(device="cpu").set_state(value)


def _restore_rng(state, generators, python_generators):
    _validate_rng(state, generators, python_generators)
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    for name, generator in generators.items(): generator.set_state(state["torch_generators"][name])
    for name, generator in python_generators.items(): generator.setstate(state["python_generators"][name])


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Checkpoints are immutable; use a new output path")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic creation without ever replacing another checkpoint.
        os.link(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return str(path)


def _check_gameplay_provenance(provenance, source_metadata, interface):
    if provenance == BUILTIN_GAMEPLAY_PROVENANCE:
        _check_builtin_gameplay_metadata(source_metadata, interface)
        return
    if provenance != "approved_gameplay":
        return
    value = source_metadata.get("plan_sha256") if type(source_metadata) is dict else None
    if (interface != "adventure-armies-v1" or type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Gameplay checkpoint requires the exact approved v4 preparation-plan identity")


def _sha256(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _check_builtin_gameplay_metadata(metadata, interface):
    """A native opponent is identified explicitly, never by an archived model."""
    from experiments.headless.builtin_evidence import profile_for
    profile_for(interface)
    if (type(metadata) is not dict
            or not _sha256(metadata.get("plan_sha256"))):
        raise ValueError("Built-in gameplay checkpoint requires its exact preparation-plan identity")
    initial = metadata.get("initial_checkpoint")
    if (type(initial) is not dict or set(initial) != {"path", "sha256", "bytes"}
            or type(initial["path"]) is not str or not Path(initial["path"]).is_absolute()
            or not _sha256(initial["sha256"]) or type(initial["bytes"]) is not int or initial["bytes"] <= 0):
        raise ValueError("Built-in gameplay checkpoint requires its bound initial checkpoint")
    opponent = metadata.get("native_opponent")
    if (type(opponent) is not dict or set(opponent) != {"kind", "module", "difficulty", "binary_sha256"}
            or opponent["kind"] != "builtin" or opponent["module"] != "Nullkiller"
            or type(opponent["difficulty"]) is not int or opponent["difficulty"] != 0
            or not _sha256(opponent["binary_sha256"])):
        raise ValueError("Built-in gameplay checkpoint requires the actual Nullkiller Easy identity")
    if metadata.get("archived_neural_pool_executed") is not False:
        raise ValueError("The archived neural pool must not represent the executed native opponent")


def _payload(model, checkpoint, provenance, source_metadata, generators, python_generators, pool, interface):
    _text(checkpoint)
    if provenance not in PROVENANCES:
        raise ValueError("Unsupported checkpoint provenance")
    _check_gameplay_provenance(provenance, source_metadata, interface)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Only the validated CPU checkpoint path is enabled")
    if provenance == BUILTIN_GAMEPLAY_PROVENANCE and any(
            value.dtype != torch.float32 or value.device.type != "cpu" for value in model.state_dict().values()):
        raise ValueError("Built-in gameplay checkpoint requires CPU float32 model tensors")
    contracts = _contracts(interface)
    if model.config.schema != contracts["encoder_schema"] or pool is not None and any(
            pool.get(identity).model.config.schema != contracts["encoder_schema"] for identity, _ in pool.identities):
        raise ValueError("Checkpoint model/opponent versions must match the explicit interface")
    _validate_weights(model, model.state_dict())
    return {"schema": SCHEMA, "kind": "policy", "checkpoint": checkpoint, "provenance": provenance,
            "contracts": contracts,
            "torch_version": str(torch.__version__), "model_config": model.config.to_dict(),
            "weights_sha256": model_weight_digest(model), "model_state": _weights(model),
            "source_metadata": _json(source_metadata), "rng": _rng_state(generators, python_generators),
            "opponent_pool": None if pool is None else pool.state_dict(), "resume": None}


def _read(path, kind, expected_source_metadata, allowed_provenance, interface, historical_torch_version=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {"schema", "kind", "checkpoint", "provenance", "contracts", "torch_version", "model_config", "weights_sha256",
                "model_state", "source_metadata", "rng", "opponent_pool", "resume"}
    if type(payload) is not dict or set(payload) != expected or payload["schema"] != SCHEMA or payload["kind"] != kind:
        raise ValueError("Unsupported checkpoint schema or purpose")
    if kind == "optimizer" and payload["provenance"] not in OPTIMIZER_PROVENANCES:
        raise ValueError("Optimizer checkpoints require optimizer provenance")
    if (type(allowed_provenance) is not tuple or not allowed_provenance
            or any(value not in PROVENANCES for value in allowed_provenance)):
        raise ValueError("Explicit supported checkpoint provenance allowlist required")
    expected_version = str(torch.__version__)
    if historical_torch_version is not None:
        # Explicit loader-only CPU wheel relocation, never a release upgrade.
        versions = (historical_torch_version, expected_version)
        if (any(type(v) is not str or ('+' in v and not v.endswith('+cpu')) for v in versions)
                or historical_torch_version.removesuffix('+cpu') != expected_version.removesuffix('+cpu')
                or expected_source_metadata is None
                or expected_source_metadata.get('torch') != historical_torch_version):
            raise ValueError("Portable checkpoint requires the same public CPU Torch release")
        expected_version = historical_torch_version
    if payload["torch_version"] != expected_version or payload["provenance"] not in allowed_provenance:
        raise ValueError("Checkpoint version or provenance differs")
    if payload["contracts"] != _contracts(interface):
        raise ValueError("Checkpoint observation/action/encoder/distribution versions differ")
    expected_model = payload["contracts"]["encoder_schema"]
    pool = payload["opponent_pool"]
    if pool is not None and (type(pool) is not list or any(type(entry) is not dict for entry in pool)):
        raise ValueError("Invalid frozen opponent pool")
    configurations = [payload["model_config"], *(entry.get("model_config") for entry in pool or ())]
    if any(type(config) is not dict or config.get("schema") != expected_model for config in configurations):
        raise ValueError("Checkpoint model or frozen pool has an incompatible encoder version")
    _text(payload["checkpoint"])
    _json(payload["source_metadata"])
    _check_gameplay_provenance(payload["provenance"], payload["source_metadata"], interface)
    if payload["provenance"] == BUILTIN_GAMEPLAY_PROVENANCE and expected_source_metadata is None:
        raise ValueError("Built-in gameplay load requires explicit expected plan and opponent metadata")
    if expected_source_metadata is not None and payload["source_metadata"] != _json(expected_source_metadata):
        raise ValueError("Checkpoint source metadata differs")
    return payload


@dataclass(frozen=True)
class LoadedPolicy:
    model: object
    checkpoint: str
    weights_sha256: str
    provenance: str
    source_metadata: dict
    opponent_pool: FrozenOpponentPool | None


def save_policy_checkpoint(path, model, *, checkpoint, provenance="random_initialization", source_metadata,
                           generators=None, python_generators=None, opponent_pool=None, interface="adventure-town-v1"):
    return _write(path, _payload(model, checkpoint, provenance, source_metadata,
                                generators or {}, python_generators or {}, opponent_pool, interface))


def load_policy_checkpoint(path, *, expected_source_metadata=None, allowed_provenance=("random_initialization",),
                           restore_rng=False, generators=None, python_generators=None, interface="adventure-town-v1",
                           historical_torch_version=None):
    payload = _read(path, "policy", expected_source_metadata, allowed_provenance, interface, historical_torch_version)
    if payload["resume"] is not None:
        raise ValueError("Policy-only checkpoint contains optimizer resume state")
    model = _model_from_payload(payload).eval()
    pool = None if payload["opponent_pool"] is None else FrozenOpponentPool.from_state_dict(payload["opponent_pool"])
    if restore_rng:
        _restore_rng(payload["rng"], generators or {}, python_generators or {})
    return LoadedPolicy(model, payload["checkpoint"], payload["weights_sha256"], payload["provenance"],
                        payload["source_metadata"], pool)


@dataclass(frozen=True)
class TrainingProgress:
    updates: int = 0
    rollouts: int = 0
    decisions: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.updates, self.rollouts, self.decisions)):
            raise ValueError("Progress counters must be nonnegative integers")


@dataclass(frozen=True)
class ResumeState:
    checkpoint: str
    progress: TrainingProgress
    sampler_state: dict
    opponent_pool: FrozenOpponentPool


def _optimizer_layout(model, optimizer):
    """The first resume format deliberately supports CPU Adam only."""
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("This checkpoint version supports CPU Adam only")
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    layout, found = [], []
    for group in optimizer.param_groups:
        row = []
        for parameter in group["params"]:
            if id(parameter) not in names or parameter.device.type != "cpu":
                raise ValueError("Optimizer parameters must belong to the supplied CPU model")
            found.append(id(parameter)); row.append(names[id(parameter)])
        layout.append(row)
    if set(found) != set(names) or len(found) != len(names):
        raise ValueError("Optimizer must own each trainable model parameter exactly once")
    return layout


def _validate_adam_state(optimizer, state, *, strict_serialized_dtypes=False):
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Invalid Adam checkpoint state")
    if strict_serialized_dtypes:
        # Adam.load_state_dict casts moments to the parameter dtype. Validate
        # their original bytes first for the explicit built-in resume contract.
        groups = state["param_groups"]
        if type(groups) is not list or len(groups) != len(optimizer.param_groups) or type(state["state"]) is not dict:
            raise ValueError("Invalid serialized Adam parameter groups")
        identifiers = []
        for saved, live in zip(groups, optimizer.param_groups):
            if (type(saved) is not dict or type(saved.get("params")) is not list
                    or len(saved["params"]) != len(live["params"])):
                raise ValueError("Serialized Adam parameters differ")
            for identifier, parameter in zip(saved["params"], live["params"]):
                if type(identifier) is not int or identifier < 0:
                    raise ValueError("Invalid serialized Adam parameter identifier")
                identifiers.append(identifier)
                current = state["state"].get(identifier, {})
                if type(current) is not dict:
                    raise ValueError("Invalid serialized Adam parameter state")
                for name, value in current.items():
                    dtype = torch.float32 if name == "step" else parameter.dtype
                    if type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != dtype:
                        raise ValueError("Serialized Adam tensor dtype/device differs")
        if len(set(identifiers)) != len(identifiers) or set(state["state"]) - set(identifiers):
            raise ValueError("Serialized Adam state has duplicate or unknown parameters")
    trial = deepcopy(optimizer)
    trial.load_state_dict(deepcopy(state))
    for group in trial.param_groups:
        for key in ("lr", "eps", "weight_decay"):
            value = group[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid Adam hyperparameter")
        if (type(group["betas"]) not in (tuple, list) or len(group["betas"]) != 2
                or any(type(value) not in (int, float) or not 0 <= value < 1 for value in group["betas"])
                or type(group["amsgrad"]) is not bool):
            raise ValueError("Invalid Adam beta/AMSGrad configuration")
        for parameter in group["params"]:
            current = trial.state.get(parameter, {})
            if not current: continue
            required = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["amsgrad"] else set())
            if set(current) != required:
                raise ValueError("Unexpected Adam state fields")
            for name, value in current.items():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
                    raise ValueError("Nonfinite or untyped Adam state")
                if name == "step":
                    if value.ndim != 0 or value.item() < 0 or value.item() != int(value.item()):
                        raise ValueError("Adam step must be a nonnegative scalar integer")
                elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("Adam moment does not match its named model parameter")
                elif name != "exp_avg" and (value < 0).any().item():
                    raise ValueError("Adam second moment cannot be negative")
    # Extra state keys not linked to a parameter cannot silently survive load.
    if any(not isinstance(parameter, torch.Tensor) for parameter in trial.state):
        raise ValueError("Adam state contains an unknown parameter identifier")


def save_checkpoint(path, model, optimizer, *, checkpoint, ppo_config, progress, source_metadata,
                    opponent_pool, sampler_state, generators=None, python_generators=None,
                    at_rollout_boundary=False, interface="adventure-town-v1", provenance="synthetic_optimizer_test"):
    if provenance not in OPTIMIZER_PROVENANCES:
        raise ValueError("Optimizer checkpoints require optimizer provenance")
    if at_rollout_boundary is not True or type(progress) is not TrainingProgress or optimizer is None:
        raise ValueError("Optimizer checkpoint requires an explicit between-rollout boundary")
    if type(opponent_pool) is not FrozenOpponentPool or not opponent_pool.identities:
        raise ValueError("Resume requires its nonempty frozen opponent pool")
    layout = _optimizer_layout(model, optimizer)
    _validate_adam_state(optimizer, optimizer.state_dict(),
                         strict_serialized_dtypes=provenance == BUILTIN_GAMEPLAY_PROVENANCE)
    payload = _payload(model, checkpoint, provenance, source_metadata,
                       generators or {}, python_generators or {}, opponent_pool, interface)
    payload["kind"] = "optimizer"
    payload["resume"] = {"ppo_config": _json(ppo_config), "progress": vars(progress),
                         "sampler_state": _json(sampler_state), "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                         "optimizer_state": deepcopy(optimizer.state_dict()), "parameter_names": layout,
                         "at_rollout_boundary": True}
    return _write(path, payload)


def load_checkpoint(path, model, optimizer, *, expected_ppo_config, expected_source_metadata,
                    generators=None, python_generators=None, at_rollout_boundary=False, interface="adventure-town-v1",
                    allowed_provenance=("synthetic_optimizer_test",), historical_torch_version=None):
    if at_rollout_boundary is not True or optimizer is None:
        raise ValueError("Resume requires an explicit between-rollout boundary")
    layout = _optimizer_layout(model, optimizer)
    payload = _read(path, "optimizer", expected_source_metadata, allowed_provenance, interface, historical_torch_version)
    resume = payload["resume"]
    if (type(resume) is not dict or set(resume) != {"ppo_config", "progress", "sampler_state", "optimizer_class", "optimizer_state", "parameter_names", "at_rollout_boundary"}
            or resume["at_rollout_boundary"] is not True or resume["ppo_config"] != _json(expected_ppo_config)
            or payload["model_config"] != model.config.to_dict() or resume["parameter_names"] != layout
            or resume["optimizer_class"] != type(optimizer).__module__ + "." + type(optimizer).__qualname__):
        raise ValueError("Optimizer resume configuration differs")
    progress = TrainingProgress(**resume["progress"])
    sampler = _json(resume["sampler_state"])
    candidate = _model_from_payload(payload)
    _validate_weights(model, candidate.state_dict())
    pool = FrozenOpponentPool.from_state_dict(payload["opponent_pool"])
    if not pool.identities: raise ValueError("Resume requires frozen opponent identities")
    _validate_rng(payload["rng"], generators or {}, python_generators or {})
    # Validate compatibility before modifying the caller's live objects.
    _validate_adam_state(optimizer, resume["optimizer_state"],
                         strict_serialized_dtypes=payload["provenance"] == BUILTIN_GAMEPLAY_PROVENANCE)
    model.load_state_dict(candidate.state_dict(), strict=True)
    optimizer.load_state_dict(resume["optimizer_state"])
    _restore_rng(payload["rng"], generators or {}, python_generators or {})
    return ResumeState(payload["checkpoint"], progress, sampler, pool)
