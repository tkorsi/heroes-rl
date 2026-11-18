"""Training state on disk: one file per update, plus a one-time legacy import.

A checkpoint holds everything a run needs to continue exactly where it stopped:
the model, the Adam optimizer, the generator that shuffles PPO minibatches and
the progress counters. Files are written whole and renamed into place, and
``latest.json`` is switched last, so an interrupted run always resumes from a
complete checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch

from .model.policy import ContextualActorCritic, ModelConfig, weights_sha256

FORMAT = "heroes-checkpoint-v1"
LEGACY_FORMAT = "heroes-context-checkpoint-v1"


@dataclass
class State:
    model: ContextualActorCritic
    optimizer: torch.optim.Adam
    generator: torch.Generator  # shuffles PPO minibatches
    progress: dict  # updates, games, decisions, optimizer_steps
    origin: str | None = None  # where the first checkpoint of this state directory came from

    @property
    def updates(self):
        return self.progress["updates"]


def _state(model, learning_rate):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def fresh(*, learning_rate, seed=0):
    """A randomly initialized policy that has never been trained."""
    torch.manual_seed(seed)
    model, optimizer = _state(ContextualActorCritic(ModelConfig()), learning_rate)
    return State(model, optimizer, torch.Generator().manual_seed(seed),
                 {"updates": 0, "games": 0, "decisions": 0, "optimizer_steps": 0}, "random initialization")


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(directory, state):
    """Write the checkpoint of ``state.updates`` and make it the latest one."""
    directory = Path(directory)
    path = directory / "checkpoints" / f"update-{state.updates:06d}.pt"
    digest = weights_sha256(state.model)
    _write(path, {
        "format": FORMAT, "model_config": state.model.config.to_dict(),
        "model_state": {name: value.detach().clone() for name, value in state.model.state_dict().items()},
        "weights_sha256": digest, "optimizer_state": state.optimizer.state_dict(),
        "shuffle_rng": state.generator.get_state(), "progress": dict(state.progress), "origin": state.origin,
    })
    pointer = directory / "latest.json"
    temporary = pointer.with_name("latest.json.tmp")
    temporary.write_text(json.dumps({"checkpoint": str(path.relative_to(directory)), "weights_sha256": digest,
                                     **state.progress}, indent=2) + "\n")
    os.replace(temporary, pointer)
    return path


def latest(directory):
    """Path of the newest checkpoint in a state directory, or None when it has none."""
    pointer = Path(directory) / "latest.json"
    if not pointer.is_file():
        return None
    return Path(directory) / json.loads(pointer.read_text())["checkpoint"]


def load_model(path):
    """Only the policy of a checkpoint, in evaluation mode. This is all an actor needs."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ContextualActorCritic(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    if weights_sha256(model) != payload["weights_sha256"]:
        raise ValueError(f"Checkpoint weights do not match their recorded digest: {path}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def load(path, *, learning_rate):
    """The complete training state of a checkpoint written by ``save``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} file: {path}")
    model, optimizer = _state(load_model(path), learning_rate)
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer.param_groups[0]["lr"] = learning_rate
    generator = torch.Generator()
    generator.set_state(payload["shuffle_rng"])
    return State(model, optimizer, generator, dict(payload["progress"]), payload.get("origin"))


# -- one-time import of the previous pipeline's checkpoints --------------------

def find_legacy(directory):
    """The newest policy/optimizer pair the previous pipeline left in ``directory``.

    Returns ``(policy.pt, resume.pt)`` or None. The previous pipeline pointed at
    its newest pair from ``latest-checkpoint.json`` while a run was active and
    from ``retained/state.json`` after it had cleaned up.
    """
    directory = Path(directory)
    candidates = []
    for pointer in (directory / "latest-checkpoint.json", directory / "retained" / "state.json"):
        try:
            pair = json.loads(pointer.read_text())["checkpoint"]
            files = [_relocated(pair[kind]["path"], directory) for kind in ("policy", "resume")]
            if all(file is not None for file in files):
                candidates.append((pair["progress"]["updates"], files))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return tuple(max(candidates, key=lambda item: item[0])[1]) if candidates else None


def _relocated(recorded, directory):
    """A recorded absolute path, or the same file under ``directory`` when the folder was moved."""
    recorded = Path(recorded)
    if recorded.is_file():
        return recorded
    parts = recorded.parts
    if directory.name in parts:
        moved = directory.joinpath(*parts[len(parts) - parts[::-1].index(directory.name):])
        if moved.is_file():
            return moved
    return None


def import_legacy(policy_path, resume_path, *, learning_rate):
    """Continue from a policy/optimizer pair of the previous pipeline.

    Weights, Adam moments and step counts, the minibatch shuffle generator and
    the progress counters carry over unchanged. The legacy files are only read.
    """
    policy = torch.load(policy_path, map_location="cpu", weights_only=True)
    resume = torch.load(resume_path, map_location="cpu", weights_only=True)
    if (policy.get("schema") != LEGACY_FORMAT or resume.get("schema") != LEGACY_FORMAT
            or policy.get("kind") != "policy" or resume.get("kind") != "optimizer"):
        raise ValueError("Expected a legacy policy.pt and its resume.pt")
    if policy["weights_sha256"] != resume["weights_sha256"] or policy["checkpoint"] != resume["checkpoint"]:
        raise ValueError("The legacy policy and optimizer files belong to different checkpoints")
    model = ContextualActorCritic(ModelConfig.from_dict(policy["model_config"]))
    model.load_state_dict(policy["model_state"])
    if weights_sha256(model) != policy["weights_sha256"]:
        raise ValueError("Legacy weights do not match their recorded digest")
    model, optimizer = _state(model, learning_rate)
    saved = resume["resume"]
    if saved["parameter_names"] != [[name for name, _ in model.named_parameters()]]:
        raise ValueError("The legacy optimizer state is laid out for different parameters")
    optimizer.load_state_dict(saved["optimizer_state"])
    optimizer.param_groups[0]["lr"] = learning_rate
    generator = torch.Generator()
    generator.set_state(resume["rng"]["torch_generators"]["minibatch"])
    progress = {"updates": saved["progress"]["updates"], "games": saved["progress"]["rollouts"],
                "decisions": saved["progress"]["decisions"],
                "optimizer_steps": saved["sampler_state"]["optimizer_steps"]}
    return State(model, optimizer, generator, progress, "legacy checkpoint " + policy["checkpoint"])
