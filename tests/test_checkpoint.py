"""Checkpoints: the round trip and the one-time import of the previous pipeline's files."""
import json

import pytest
import torch

from heroes import checkpoint
from heroes.model.policy import weights_sha256


def _trained_a_little(state):
    """Give the optimizer real moments and move the generator, as one update would."""
    loss = sum(parameter.square().sum() for parameter in state.model.parameters())
    loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad()
    torch.randperm(5, generator=state.generator)
    state.progress = {"updates": 3, "games": 30, "decisions": 4000, "optimizer_steps": 120}
    return state


def _same_optimizer(left, right):
    a, b = left.state_dict()["state"], right.state_dict()["state"]
    return a.keys() == b.keys() and all(
        torch.equal(torch.as_tensor(a[key][name]), torch.as_tensor(b[key][name])) for key in a for name in a[key])


def test_a_saved_state_loads_back_unchanged(tmp_path):
    state = _trained_a_little(checkpoint.fresh(learning_rate=3e-4))
    assert checkpoint.latest(tmp_path) is None
    path = checkpoint.save(tmp_path, state)
    assert path == checkpoint.latest(tmp_path) == tmp_path / "checkpoints" / "update-000003.pt"
    loaded = checkpoint.load(path, learning_rate=1e-4)
    assert weights_sha256(loaded.model) == weights_sha256(state.model)
    assert _same_optimizer(loaded.optimizer, state.optimizer)
    assert loaded.optimizer.param_groups[0]["lr"] == 1e-4  # the configured rate wins over the stored one
    assert loaded.generator.get_state().equal(state.generator.get_state())
    assert (loaded.progress, loaded.origin) == (state.progress, "random initialization")
    assert json.loads((tmp_path / "latest.json").read_text())["updates"] == 3
    actor_model = checkpoint.load_model(path)
    assert not actor_model.training and not any(parameter.requires_grad for parameter in actor_model.parameters())


def test_changed_weights_are_refused(tmp_path):
    path = checkpoint.save(tmp_path, checkpoint.fresh(learning_rate=3e-4))
    payload = torch.load(path, weights_only=True)
    next(iter(payload["model_state"].values())).add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="digest"):
        checkpoint.load_model(path)


def write_legacy(directory, *, updates=1784, name="update-001784"):
    """A policy/optimizer pair laid out like the previous pipeline's, with its pointer file."""
    state = _trained_a_little(checkpoint.fresh(learning_rate=3e-4, seed=5))
    folder = directory / "retained" / name
    folder.mkdir(parents=True)
    digest = weights_sha256(state.model)
    common = {"schema": checkpoint.LEGACY_FORMAT, "checkpoint": name, "weights_sha256": digest}
    torch.save({**common, "kind": "policy", "model_config": state.model.config.to_dict(),
                "model_state": state.model.state_dict()}, folder / "policy.pt")
    torch.save({**common, "kind": "optimizer", "rng": {"torch_generators": {"minibatch": state.generator.get_state()}},
                "resume": {"parameter_names": [[n for n, _ in state.model.named_parameters()]],
                           "optimizer_state": state.optimizer.state_dict(),
                           "progress": {"updates": updates, "rollouts": 9822, "decisions": 12926530},
                           "sampler_state": {"optimizer_steps": 406884}}}, folder / "resume.pt")
    (directory / "retained" / "state.json").write_text(json.dumps({"checkpoint": {
        "policy": {"path": str(folder / "policy.pt")}, "resume": {"path": str(folder / "resume.pt")},
        "progress": {"updates": updates}}}))
    return state


def test_the_previous_pipelines_checkpoint_continues_unchanged(tmp_path):
    legacy = tmp_path / "selfplay"
    original = write_legacy(legacy)
    pair = checkpoint.find_legacy(legacy)
    assert [path.name for path in pair] == ["policy.pt", "resume.pt"]
    state = checkpoint.import_legacy(*pair, learning_rate=3e-4)
    assert weights_sha256(state.model) == weights_sha256(original.model)
    assert _same_optimizer(state.optimizer, original.optimizer)
    assert state.generator.get_state().equal(original.generator.get_state())
    assert state.progress == {"updates": 1784, "games": 9822, "decisions": 12926530, "optimizer_steps": 406884}
    assert state.origin == "legacy checkpoint update-001784"
    # Saved in the new format, it is checkpoint 1784 of the new state folder.
    assert checkpoint.save(tmp_path / "state", state).name == "update-001784.pt"


def test_a_moved_legacy_folder_is_still_found(tmp_path):
    write_legacy(tmp_path / "selfplay")
    moved = tmp_path / "elsewhere" / "selfplay"
    moved.parent.mkdir()
    (tmp_path / "selfplay").rename(moved)
    pair = checkpoint.find_legacy(moved)  # the pointer still records the old absolute paths
    assert pair is not None and all(path.is_file() and moved in path.parents for path in pair)
    assert checkpoint.find_legacy(tmp_path / "nothing-here") is None


def test_mismatched_legacy_files_are_refused(tmp_path):
    write_legacy(tmp_path / "a")
    write_legacy(tmp_path / "b", name="update-000100")
    policy, _ = checkpoint.find_legacy(tmp_path / "a")
    _, resume = checkpoint.find_legacy(tmp_path / "b")
    with pytest.raises(ValueError, match="different checkpoints"):
        checkpoint.import_legacy(policy, resume, learning_rate=3e-4)
