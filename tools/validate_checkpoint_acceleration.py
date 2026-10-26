"""Bounded, frozen update412 validation; no PPO update or training admission.

The general embedded probe retains its original v7 checkpoint gate. Only this
standalone diagnostic process temporarily supplies its separately pinned v8
loader, and restores the original function even on failure. The same wrapper
can run with either reviewed source tree; runtime/learner code is not patched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INTERFACE = "adventure-capabilities-v1"
PINS = {
    "after-policy.pt": "1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88",
    "after-resume.pt": "09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321",
    "plan.json": "4ceb55e2f477c8406e90eaa9a2b79c4a13514bd37b984b626e2ea79f24f95e23",
    "report.json": "20c070ce2390315176073db880498ad1a7c86ddca2f4451bd82ad2c979327110",
}
MAPS = {
    0: "7d88860234664580c564fc2c71554fd813519ace9904ff368dc30eef3c352019",
    1: "75303cc2b19c1d0ddc5fbac35315a951dfc9bb6e780b4245508f8a2a78e691f1",
}
WEIGHTS = "c310ae6b0b295149641802577fa4469c7a1662d530108c0289e3ce8f9c726d84"
PROGRESS = {"updates": 412, "rollouts": 1648, "decisions": 95836}
DEPENDENCY = "9aeac52025a9bd0cd307303e8b06f732aefa640795ccf9e4fd9d7ba1f0ba65bd"


def require(value, message):
    if not value:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def pinned(path, digest):
    result = binding(path)
    require(result["sha256"] == digest, "Pinned input changed: " + Path(path).name)
    return result


def validate_configuration(plan, report):
    cfg = plan["configuration"]
    require(plan["source"]["interface"] == INTERFACE and cfg["interface"] == INTERFACE,
            "The original trained v8 interface is required")
    expected = {"ordinary_choice_limit": 4960, "reserved_dialog_choices": 160,
                "max_total_choices": 5120, "max_days": 365, "torch_threads": 1}
    require(all(type(cfg.get(k)) is int and cfg[k] == v for k, v in expected.items()),
            "The original curriculum choice/day budget changed")
    require(report["cumulative_progress"] == PROGRESS,
            "Only the final update412 checkpoint is permitted")
    require(report["plan_sha256"] == PINS["plan.json"], "Final plan/report binding changed")
    require(report["after_checkpoint"]["weights_sha256"] == WEIGHTS, "Final tensor digest changed")
    return cfg


class Frozen412:
    def __init__(self, root):
        import torch
        from experiments.learner.builtin_preservation import verify_pair
        from experiments.learner.builtin_training_resume import _metadata
        self.folder = Path(root).resolve() / "checkpoint"
        self.inputs = {name: pinned(self.folder / name, digest) for name, digest in PINS.items()}
        self.plan, self.report = [json.loads((self.folder / name).read_text()) for name in ("plan.json", "report.json")]
        self.configuration = validate_configuration(self.plan, self.report)
        self.dependency_path = self.folder.parent / "dependencies/update411-after-policy.pt"
        self.dependency = pinned(self.dependency_path, DEPENDENCY)
        require(self.plan["checkpoint"]["sha256"] == DEPENDENCY, "Initial checkpoint dependency changed")
        self.metadata = _metadata(self.plan, PINS["plan.json"])
        historical = self.plan["source"]["torch"]
        require(historical == "2.9.1+cpu" and str(torch.__version__) in ("2.9.1", "2.9.1+cpu"),
                "Only the existing same-release CPU Torch relocation is permitted")
        self.options = {"historical_torch_version": historical}
        self.restore = verify_pair(self.folder / "after-resume.pt", self.folder / "after-policy.pt",
            self.plan, self.report, migration=SimpleNamespace(checkpoint_options=self.options))
        require(self.restore["progress"] == PROGRESS and self.restore["weights_sha256"] == WEIGHTS
                and self.restore["exact_model_adam_rng_progress_pool"] is True, "Exact checkpoint restoration failed")
        self.relocation = {"historical_torch": historical, "current_torch": str(torch.__version__),
            "current_python": sys.version, "historical_python": self.plan["source"]["python"],
            "mode": "existing strict same-public-release CPU loader; original bytes unchanged",
            "cross_platform_policy_drift_verified": False}

    def unchanged(self):
        require(self.inputs == {name: pinned(self.folder / name, digest) for name, digest in PINS.items()},
                "A preserved checkpoint input changed")
        require(pinned(self.dependency_path, DEPENDENCY) == self.dependency, "Initial dependency changed")

    def load(self, path, metadata=None):
        from experiments.learner import checkpoints as cp
        from validation.probes.validate_hero_policy import frozen_fingerprints
        require(Path(path).resolve() == self.folder / "after-policy.pt", "Unexpected policy path")
        require(metadata is None or metadata == self.metadata, "Unexpected requested source metadata")
        self.unchanged()
        loaded = cp.load_policy_checkpoint(path, expected_source_metadata=self.metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE, **self.options)
        loaded.model.eval().requires_grad_(False)
        require(loaded.weights_sha256 == WEIGHTS, "Loaded weights differ from update412")
        return loaded, frozen_fingerprints(loaded, path)


@contextmanager
def scoped_loader(loader):
    from experiments.learner import evaluate_builtin_opponent
    original = evaluate_builtin_opponent.load_fixed
    evaluate_builtin_opponent.load_fixed = loader
    try:
        yield
    finally:
        evaluate_builtin_opponent.load_fixed = original


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run(args):
    import torch
    from experiments.learner.inference import source_identity
    from validation.probes import validate_embedded_engine as probe
    torch.set_num_threads(1)
    root, output = args.input.resolve(), args.output.resolve()
    require(not output.exists(), "Use a fresh output; all prior attempts are preserved")
    require(type(args.episodes) is int and 1 <= args.episodes <= 4, "At most four frozen games per process")
    require(args.engine_seed == 710001 and args.policy_seed == 720001, "Use the preregistered fixed seed schedule")
    require(0 < args.row_seconds <= 300 and 0 < args.watchdog <= 60, "Bounded diagnostic timeouts required")
    frozen = Frozen412(root)
    map_path = root / "maps" / f"curriculum-3-{args.seat}.vmap"
    map_binding = pinned(map_path, MAPS[args.seat])
    source = source_identity(INTERFACE)
    require(source["tracked_worktree_dirty"] is False, "Commit the validation release before gameplay")
    own_source = {str(p.relative_to(ROOT)): binding(p) for p in
        (Path(__file__).resolve(), ROOT / "validation/probes/validate_embedded_engine.py")}
    output.mkdir(parents=True)
    receipt = {"schema": "heroes-checkpoint412-acceleration-validation-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(), "phase": "prepared",
        "training_performed": False, "training_samples": 0, "optimizer_updates": 0, "optimizer_steps": 0,
        "checkpoint_inputs": frozen.inputs, "initial_dependency": frozen.dependency,
        "restore": frozen.restore, "relocation": frozen.relocation,
        "source": source, "harness": own_source, "map": map_binding, "seat": args.seat,
        "capture_verification_requested": "HEROES_VERIFY_NATIVE_CAPTURE" in os.environ,
        "scope": "Frozen validation only; same-host comparison is distinct from Linux-to-Mac policy drift."}
    write(output / "validation-receipt.json", receipt)
    if args.prepare_only:
        return receipt
    options = SimpleNamespace(backend=args.backend, controller="policy", binary=args.binary,
        replay_binary=args.replay_binary, data=args.data, map=map_path, output=output / "games",
        checkpoint=frozen.folder / "after-policy.pt", random_policy_seed=None,
        episodes=args.episodes, seats=str(args.seat), engine_seed=args.engine_seed,
        policy_seed=args.policy_seed, seed_step=1, max_days=365, choices=4960, dialog_reserve=160,
        watchdog=args.watchdog, bridge_timeout=30, row_seconds=args.row_seconds,
        visit_kind=None, stall_seconds=0.0, replay="every", audit="every", env_per_row=False,
        interface=INTERFACE, recorded=None, stop_on_failure=True)
    try:
        with scoped_loader(frozen.load):
            result = probe.run(options)
        actual_plan = json.loads((output / "games/plan.json").read_text())
        require(actual_plan["limits"]["choices"] == 4960 and actual_plan["limits"]["hard_choices"] == 5120
                and actual_plan["limits"]["max_days"] == 365, "Actual probe limits differ")
        receipt.update(phase="finished", verdict=result["verdict"], summary=result["summary"],
            unstarted_rows=result.get("unstarted_rows", []),
            probe_report=binding(output / "games/report.json"), probe_plan=binding(output / "games/plan.json"))
    except BaseException as error:
        receipt.update(phase="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        frozen.unchanged()
        require(pinned(map_path, MAPS[args.seat]) == map_binding, "Map changed during validation")
        require(source_identity(INTERFACE) == source, "Learner source changed during validation")
        require(own_source == {name: binding(ROOT / name) for name in own_source}, "Harness changed during validation")
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), inputs_unchanged=True)
        write(output / "validation-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=("cold", "embedded"), required=True)
    for name in ("binary", "replay-binary", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--engine-seed", type=int, default=710001)
    parser.add_argument("--policy-seed", type=int, default=720001)
    parser.add_argument("--row-seconds", type=float, default=300)
    parser.add_argument("--watchdog", type=float, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({k: result.get(k) for k in ("phase", "verdict", "training_samples", "optimizer_updates")}, sort_keys=True))


if __name__ == "__main__":
    main()
