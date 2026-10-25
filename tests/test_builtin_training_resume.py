"""Synthetic checkpoint continuation only; no game, replay or native admission."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import builtin_training_resume as resume
from experiments.learner import checkpoints as cp, contextual_model as model_api
from experiments.learner.ppo import PPOConfig


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def synthetic_step(model, optimizer):
    """Populate Adam state; deliberately not a gameplay or PPO-loss fixture."""
    model.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    sum(p.square().mean() for p in model.parameters()).backward()
    optimizer.step()


class ParentResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        before_torch, before_python = torch.get_rng_state(), random.getstate()
        self.addCleanup(torch.set_rng_state, before_torch)
        self.addCleanup(random.setstate, before_python)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.git("init", "-q")
        self.git("config", "user.name", "Synthetic test")
        self.git("config", "user.email", "synthetic@example.invalid")
        (self.root / ".gitignore").write_text("runs/\nbuild/\nmap.vmap\n")
        self.core = self.root / "experiments/learner/contextual_model.py"
        self.core.parent.mkdir(parents=True)
        self.core.write_text("# Synthetic committed source identity, not executable model code.\n")
        self.commit("initial synthetic source")
        self.source = self.source_identity()
        self.runtime = {"libvcmi": "a" * 64}
        binary = self.root / "build/vcmiclient"
        binary.parent.mkdir(); binary.write_bytes(b"synthetic binary fixture")
        self.game_map = self.root / "map.vmap"
        self.game_map.write_bytes(b"synthetic map fixture")
        self.native = {"binary": str(binary), "binary_sha256": resume.binding(binary)["sha256"],
                       "map": {"name": "map.vmap", "sha256": resume.binding(self.game_map)["sha256"]}}
        self.patch_root = patch.object(resume, "ROOT", self.root)
        self.patch_root.start(); self.addCleanup(self.patch_root.stop)
        self.patch_runtime = patch.object(resume, "runtime_identity", return_value=self.runtime)
        self.patch_runtime.start(); self.addCleanup(self.patch_runtime.stop)
        torch.manual_seed(1551); random.seed(1552)
        self.model = model_api.ContextualActorCritic(model_api.ModelConfig(
            schema=model_api.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.pool = cp.FrozenOpponentPool(); self.pool.add("archived-not-executed", self.model)
        self.generator = torch.Generator().manual_seed(1553)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.cfg = {"adam": {"lr": 0.0003}, "ppo": PPOConfig(epochs=1).to_dict()}
        self.run = self.root / "runs/pilot"; self.run.mkdir(parents=True)
        initial = self.run / "original-random.pt"
        cp.save_policy_checkpoint(initial, self.model, checkpoint="synthetic-random", source_metadata=self.source,
                                  opponent_pool=self.pool, interface=resume.INTERFACE)
        self.initial = resume.binding(initial)
        self.plan = {"schema": resume.PILOT + "-plan", "source": self.source, "runtime": self.runtime,
            "native": self.native, "game_map": str(self.game_map), "checkpoint": self.initial,
            "configuration": self.cfg, "execution_output": str(self.run / "execution"),
            "native_opponents": [{"kind": "builtin", "module": "Nullkiller", "difficulty": 0,
                                  "binary_sha256": "b" * 64, "player": 1-seat} for seat in (0, 1)],
            "schedule": [{"episode": i+1, "phase": phase} for i, phase in
                         enumerate(("before", "training", "training", "after"))]}
        self.plan_path = self.run / "plan.json"; write(self.plan_path, self.plan)
        self.metadata = resume._metadata(self.plan, resume.binding(self.plan_path)["sha256"])
        synthetic_step(self.model, self.optimizer)
        self.progress = cp.TrainingProgress(1, 2, 4)
        self.sampler = {"next_schedule_index": 3, "optimizer_steps": 1}
        self.report_path, self.report = self.save_run(self.plan_path, self.plan, self.metadata,
            self.model, self.optimizer, self.pool, self.generator, self.progress, self.sampler)
        self.raw = torch.load(self.report["after_checkpoint"]["resume"]["path"], weights_only=True)
        added = self.root / "experiments/learner/new_runner.py"
        added.write_text("# Additive synthetic continuation runner.\n")
        self.commit("add synthetic continuation runner")
        self.current = self.source_identity()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True, stderr=subprocess.DEVNULL).strip()

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)

    def source_identity(self):
        return {"git_commit": self.git("rev-parse", "HEAD"), "tracked_worktree_dirty": False,
                "torch": str(torch.__version__), "python": sys.version, "interface": resume.INTERFACE,
                "source_sha256": {str(p.relative_to(self.root)): resume.binding(p)["sha256"]
                                  for p in sorted(self.core.parent.glob("*.py"))}}

    def save_run(self, plan_path, plan, metadata, model, optimizer, pool, generator, progress, sampler):
        output = Path(plan["execution_output"]); output.mkdir()
        options = {"checkpoint": "synthetic-completed-" + str(progress.updates), "source_metadata": metadata,
                   "opponent_pool": pool, "generators": {"minibatch": generator}, "interface": resume.INTERFACE,
                   "provenance": cp.BUILTIN_GAMEPLAY_PROVENANCE}
        cp.save_checkpoint(output / "after-resume.pt", model, optimizer, ppo_config=self.cfg["ppo"],
            progress=progress, sampler_state=sampler, at_rollout_boundary=True, **options)
        cp.save_policy_checkpoint(output / "after-policy.pt", model, **options)
        report = {"schema": plan["schema"].removesuffix("-plan") + "-result", "status": "finished",
            "update_status": "checkpoint_verified", "worker_complete": True, "supervisor_cleanup_verified": True,
            "within_shared_budget": True, "training_performed": True, "optimizer_updates": 1,
            "optimizer_steps": 1, "training_samples": 4, "plan_sha256": resume.binding(plan_path)["sha256"],
            "attempts": [{**row, "status": "passed", "completion_receipt": True, "cleanup_verified": True,
                          "identity_verified": True, "replay_verified": True,
                          "training_samples": 2 if row["phase"] == "training" else 0} for row in plan["schedule"]],
            "after_checkpoint": {"policy": resume.binding(output / "after-policy.pt"),
                "resume": resume.binding(output / "after-resume.pt"),
                "weights_sha256": cp.model_weight_digest(model), "exact_model_adam_rng_progress_pool": True}}
        if plan["schema"] == resume.CONTINUATION + "-plan":
            report.update(parent=plan["parent"], cumulative_progress=asdict(progress))
        path = output / "report.json"; write(path, report)
        return path, report

    def load(self, **changes):
        options = {"current_runtime": self.runtime, "current_native": self.native,
                   "expected_plan_sha256": resume.binding(self.plan_path)["sha256"],
                   "expected_report_sha256": resume.binding(self.report_path)["sha256"]}
        options.update(changes)
        return resume.load_parent(self.plan_path, self.report_path, self.current, **options)

    def assert_reject_preserves_rng(self, **changes):
        before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
        with self.assertRaises((ValueError, KeyError)):
            self.load(**changes)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(random.getstate(), before_python)

    def test_exact_restore_preserves_adam_rng_progress_and_never_executes_archived_pool(self):
        with patch.object(cp.FrozenOpponentPool, "sample", side_effect=AssertionError("Never execute archived pool")):
            parent = self.load()
        resume._exact(dict(parent.model.state_dict()), self.raw["model_state"])
        resume._exact(parent.optimizer.state_dict(), self.raw["resume"]["optimizer_state"])
        resume._exact(parent.pool.state_dict(), self.pool.state_dict())
        self.assertEqual(parent.progress, self.progress)
        self.assertEqual(parent.sampler_state, self.sampler)
        self.assertEqual(parent.metadata, self.metadata)
        self.assertEqual(parent.optimizer_steps, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.raw["rng"]["torch_cpu"]))
        self.assertEqual(random.getstate(), self.raw["rng"]["python"])
        self.assertTrue(torch.equal(parent.generator.get_state(), self.raw["rng"]["torch_generators"]["minibatch"]))
        self.assertFalse(parent.model.training)
        self.assertTrue(all(not p.requires_grad for p in parent.model.parameters()))
        self.assertEqual(set(parent.bindings["source_proof"]["added_sources"]), {"experiments/learner/new_runner.py"})
        self.assertEqual(resume.binding(self.initial["path"]), self.initial)

    def test_unverified_or_unknown_completion_never_loads(self):
        for key, value in (("status", "stopped"), ("update_status", "interrupted_unknown"),
                           ("worker_complete", False), ("supervisor_cleanup_verified", False),
                           ("optimizer_steps", None), ("optimizer_updates", True)):
            with self.subTest(key=key):
                changed = deepcopy(self.report); changed[key] = value; write(self.report_path, changed)
                self.assert_reject_preserves_rng()
        changed = deepcopy(self.report); changed["attempts"][1]["status"] = "failed"
        write(self.report_path, changed); self.assert_reject_preserves_rng()

    def test_hash_native_runtime_and_python_drift_reject(self):
        self.assert_reject_preserves_rng(expected_report_sha256="f" * 64)
        self.assert_reject_preserves_rng(expected_plan_sha256="f" * 64)
        self.assert_reject_preserves_rng(current_runtime={"libvcmi": "c" * 64})
        self.assert_reject_preserves_rng(current_native={**self.native, "binary_sha256": "d" * 64})
        self.current = {**self.current, "python": "another-python"}
        self.assert_reject_preserves_rng()

    def test_committed_existing_source_change_or_uncommitted_addition_reject(self):
        self.core.write_text("# Changed model implementation.\n")
        self.commit("change synthetic core")
        self.current = self.source_identity()
        self.assert_reject_preserves_rng()

    def test_uncommitted_source_cannot_claim_clean_provenance(self):
        added = self.core.parent / "uncommitted.py"; added.write_text("# uncommitted\n")
        self.current = self.source_identity()
        self.assert_reject_preserves_rng()

    def test_mutated_checkpoint_and_reset_or_wrong_progress_adam_reject(self):
        target = Path(self.report["after_checkpoint"]["resume"]["path"])
        for mode in ("missing_adam", "progress", "moment_dtype", "native_metadata"):
            with self.subTest(mode=mode):
                payload = deepcopy(self.raw)
                if mode == "missing_adam": payload["resume"]["optimizer_state"]["state"] = {}
                elif mode == "progress": payload["resume"]["progress"]["decisions"] += 1
                elif mode == "moment_dtype":
                    value = next(iter(payload["resume"]["optimizer_state"]["state"].values()))
                    value["exp_avg"] = value["exp_avg"].double()
                else: payload["source_metadata"]["native_opponent"]["binary_sha256"] = "e" * 64
                torch.save(payload, target)
                changed = deepcopy(self.report); changed["after_checkpoint"]["resume"] = resume.binding(target)
                write(self.report_path, changed)
                self.assert_reject_preserves_rng()

    def test_next_batch_binds_new_metadata_and_accumulates_without_reset(self):
        parent = self.load()
        run = self.root / "runs/continuation"; run.mkdir()
        plan = {**self.plan, "schema": resume.CONTINUATION + "-plan", "source": self.current,
            "parent": parent.bindings, "execution_output": str(run / "execution"),
            "checkpoint": parent.bindings["policy"],
            "schedule": [{"episode": i+1, "phase": "training"} for i in range(2)]}
        plan["checkpoint_metadata"] = {**self.current, "initial_checkpoint": parent.bindings["policy"],
            "native_opponent": self.metadata["native_opponent"], "archived_neural_pool_executed": False,
            "parent_plan_sha256": parent.bindings["plan"]["sha256"],
            "parent_report_sha256": parent.bindings["report"]["sha256"]}
        path = run / "plan.json"; write(path, plan)
        metadata = resume._metadata(plan, resume.binding(path)["sha256"])
        self.assertNotEqual(metadata["plan_sha256"], parent.metadata["plan_sha256"])
        before_steps = [value["step"].item() for value in parent.optimizer.state.values()]
        synthetic_step(parent.model, parent.optimizer)
        self.assertEqual([value["step"].item() for value in parent.optimizer.state.values()], [v+1 for v in before_steps])
        progress = cp.TrainingProgress(2, 4, 8)
        sampler = {"next_schedule_index": 2, "optimizer_steps": 2, "batch_optimizer_steps": 1}
        report_path, report = self.save_run(path, plan, metadata, parent.model, parent.optimizer,
                                           parent.pool, parent.generator, progress, sampler)
        following = resume.load_parent(path, report_path, self.current, current_runtime=self.runtime,
            current_native=self.native, expected_plan_sha256=resume.binding(path)["sha256"],
            expected_report_sha256=resume.binding(report_path)["sha256"])
        self.assertEqual(following.progress, progress)
        self.assertEqual(following.optimizer_steps, 2)
        self.assertEqual(following.metadata, metadata)
        resume._exact(following.optimizer.state_dict(), parent.optimizer.state_dict())
        resume._exact(following.pool.state_dict(), self.pool.state_dict())
        self.assertEqual(resume.binding(self.initial["path"]), self.initial)


if __name__ == "__main__":
    unittest.main()
