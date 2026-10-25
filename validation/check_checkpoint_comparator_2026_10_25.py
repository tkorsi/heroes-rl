"""Verify the repaired comparator against the saved run without resuming it."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_checkpoint, load_policy_checkpoint
from experiments.learner.contextual_model import ContextualActorCritic, weights_sha256
from experiments.learner.first_run import implementation_identity
from experiments.learner.gameplay import assert_same_state, read_plan

BASE = ROOT / "runs/first-training-prepared-2026-10-25"
OUTPUT = BASE / "approved-execution"
PLAN_HASH = "ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6"


def artifact(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    torch.set_num_threads(1)
    assert artifact(BASE / "plan.json")["sha256"] == PLAN_HASH
    plan = json.loads((BASE / "plan.json").read_text())
    original_evidence = ROOT / "validation/first-gameplay-2026-10-25.json"
    assert artifact(original_evidence)["sha256"] == "2ebc629c03d6fc864ad878c63747d0235cd73689712bd319e37dca08ed28ab7b"
    before = artifact(OUTPUT / "resume.pt")
    assert before["sha256"] == "df8834bac83c7444ceff5396f06015ab9f14b7f90aea4dd62c1b7a6b3bb2f089"
    current = implementation_identity()
    changed = {name: {"execution": digest, "post_fix": current.get(name)}
               for name, digest in plan["implementation_sha256"].items() if current.get(name) != digest}
    assert set(current) == set(plan["implementation_sha256"])
    assert set(changed) == {"experiments/learner/gameplay.py", "tests/test_gameplay_checkpoint.py"}
    try:
        read_plan(BASE / "plan.json", PLAN_HASH)
    except ValueError as error:
        stale_plan_rejection = str(error)
        assert "source/runtime changed" in stale_plan_rejection
    else:
        raise AssertionError("Changed implementation must not reuse the old execution approval")

    cfg = plan["configuration"]
    initial = load_policy_checkpoint(BASE / "initial-policy.pt", interface=c.ARMIES_ENVIRONMENT)
    payload = torch.load(OUTPUT / "resume.pt", map_location="cpu", weights_only=True)
    expected = ContextualActorCritic(initial.model.config)
    expected.load_state_dict(payload["model_state"])
    model = ContextualActorCritic(initial.model.config)
    optimizer = torch.optim.Adam(model.parameters(), **cfg["adam"])
    generators = {"minibatch": torch.Generator(device="cpu")}
    python_generators = {"schedule": random.Random()}
    restored = load_checkpoint(OUTPUT / "resume.pt", model, optimizer,
        expected_ppo_config=cfg["ppo"],
        expected_source_metadata={"plan_sha256": PLAN_HASH, "source": plan["source"], "configuration": cfg},
        generators=generators, python_generators=python_generators,
        at_rollout_boundary=True, interface=c.ARMIES_ENVIRONMENT, allowed_provenance=("approved_gameplay",))
    # Both state dictionaries have the same OrderedDict type used by execute().
    assert_same_state(expected.state_dict(), model.state_dict())
    assert_same_state(payload["resume"]["optimizer_state"], optimizer.state_dict())
    assert_same_state(payload["rng"]["torch_cpu"], torch.get_rng_state())
    assert_same_state(payload["rng"]["python"], random.getstate())
    assert_same_state(payload["rng"]["torch_generators"]["minibatch"], generators["minibatch"].get_state())
    assert_same_state(payload["rng"]["python_generators"]["schedule"], python_generators["schedule"].getstate())
    assert restored.opponent_pool.identities == initial.opponent_pool.identities
    assert weights_sha256(model) == payload["weights_sha256"]
    assert vars(restored.progress) == {"updates": 1, "rollouts": 2, "decisions": 294}
    different = deepcopy(model.state_dict())
    next(iter(different.values())).reshape(-1)[0] += 1
    try:
        assert_same_state(model.state_dict(), different)
    except ValueError as error:
        assert str(error) == "Resume tensor differs"
    else:
        raise AssertionError("Changed model tensor was not detected")
    assert before == artifact(OUTPUT / "resume.pt")
    evidence = {
        "schema": "heroes-checkpoint-comparator-fix-v1", "date": "2026-10-25", "status": "passed",
        "scope": "Read-only load and comparison of the preserved checkpoint after a comparator-only repair. No game, PPO update, optimizer step, checkpoint rewrite or execution continuation.",
        "original_execution_evidence": artifact(original_evidence), "checkpoint": before,
        "changed_implementation_files": changed, "original_plan_rejected_after_source_change": stale_plan_rejection,
        "checks": {"ordered_model_state": True, "adam_state": True, "global_and_sampler_rng": True,
                   "progress": True, "frozen_opponent": True, "altered_tensor_rejected": True,
                   "checkpoint_bytes_unchanged": True},
        "regression_before": {"result": "failed", "tests": 4, "errors": 2,
            "artifact": artifact(ROOT / "runs/checkpoint-comparator-fix-2026-10-25/regression-before.log")},
        "regression_after": {"result": "passed", "tests": 9,
            "artifact": artifact(ROOT / "runs/checkpoint-comparator-fix-2026-10-25/regression-after.log")},
        "verifier": artifact(Path(__file__).resolve()), "additional_gameplay_attempts": 0,
        "additional_gameplay_optimizer_steps": 0, "after_evaluation_performed": False,
    }
    destination = ROOT / "validation/checkpoint-comparator-fix-2026-10-25.json"
    with destination.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checkpoint_unchanged": True,
                      "old_plan_rejected": stale_plan_rejection, "gameplay_resumed": False}))


if __name__ == "__main__":
    main()
