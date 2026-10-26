"""Controller integration with synthetic reports; never run native games or PPO."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.headless import process_deadline
from experiments.learner import builtin_preservation, local_script_training as adapter
from experiments.learner.builtin_training_resume import binding
from tools import run_local_training as runner


class ControllerIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / "run"; self.run.mkdir()
        initial = self.root / "initial"; initial.mkdir()
        for name in ("plan", "report", "policy", "resume", "initial_checkpoint"):
            (initial / name).write_text("synthetic initial " + name)
        self.parent = {name: binding(initial / name)
                       for name in ("plan", "report", "policy", "resume", "initial_checkpoint")}
        self.session = dict(run_dir=str(self.run), run_root=str(self.root), artifact_root=str(self.root),
            initial_parent=self.parent, supervisor_pid=os.getppid(), next_sequence=266,
            max_updates=1, target_games=None, evaluation_interval_updates=5,
            profile="easier8x8", **runner.budget(1))
        self.session_path = self.run / "session.json"; runner.atomic(self.session_path, self.session)
        self.started = []; self.preserved = []; self.restore_calls = []
        self.bad_training = False; self.preservation_failure = False; self.restore_copy_failure = False
        self.stop_after = None; self.plans = {}
        self.progress_by_policy = {self.parent["policy"]["path"]: 412}
        self.addCleanup(patch.stopall)
        patch.object(adapter, "prepare_batch", side_effect=self.prepare).start()
        patch.object(adapter, "restore_parent", side_effect=self.restore).start()
        patch.object(builtin_preservation, "preserve", side_effect=self.preserve).start()
        patch.object(builtin_preservation, "verify_pair", side_effect=self.verify_copy).start()
        patch.object(process_deadline, "guarded_popen", side_effect=self.spawn).start()
        self.close = patch.object(process_deadline, "close_owned_process").start()
        patch.object(process_deadline, "cleanup_from_evidence").start()
        self.enough = patch.object(runner, "enough", return_value=True).start()

    def restore(self, parent):
        self.restore_calls.append(deepcopy(parent))
        updates = self.progress_by_policy[parent["policy"]["path"]]
        return SimpleNamespace(progress=SimpleNamespace(updates=updates, rollouts=updates * 4,
                               decisions=95836 + (updates - 412) * 20), optimizer_steps=3000 + updates - 412)

    def prepare(self, session_path, parent, sequence, seat, *, evaluate, label):
        folder = self.run / "batches" / f"{label}-{sequence}-{seat}"; folder.mkdir(parents=True)
        execution = folder / "execution"; execution.mkdir()
        for name in ("after-policy.pt", "after-resume.pt"):
            (execution / name).write_text("synthetic resulting state " + name)
        updates = self.progress_by_policy[parent["policy"]["path"]] + (0 if evaluate else 1)
        self.progress_by_policy[str(execution / "after-policy.pt")] = updates
        plan = {"checkpoint_metadata": {"initial_checkpoint": self.parent["policy"]},
                "mode": "evaluation" if evaluate else "training", "synthetic_updates": updates}
        path = folder / "plan.json"; runner.atomic(path, plan)
        report = dict(status="finished", worker_complete=True, supervisor_cleanup_verified=True,
            within_shared_budget=True, attempts=[dict(status="passed", completion_receipt=True,
                cleanup_verified=True, identity_verified=True, replay_verified=True,
                outcomes={"red": "win", "blue": "loss"}) for _ in range(4)],
            optimizer_updates=0 if evaluate else 1, optimizer_steps=0 if evaluate else 2,
            training_samples=0 if evaluate else 20, update_status="checkpoint_verified",
            after_checkpoint={"policy": binding(execution / "after-policy.pt"),
                              "resume": binding(execution / "after-resume.pt")})
        if not evaluate and self.bad_training:
            report["attempts"][-1].update(status="failed", replay_verified=False)
            report["worker_complete"] = False
        runner.atomic(execution / "report.json", report)
        self.plans[str(path)] = {"mode": label, "sequence": sequence, "seat": seat}
        return path

    def spawn(self, command, **kwargs):
        path = command[command.index("--plan") + 1]
        row = self.plans[path]; self.started.append(row)
        def wait(timeout):
            if self.stop_after == row["mode"]:
                (self.run / "stop-requested").touch()
            return 0
        return SimpleNamespace(wait=wait)

    def preserve(self, plan_path, report_path, destination, *, root):
        self.preserved.append(str(destination))
        if self.preservation_failure:
            raise OSError("synthetic full-disk preservation failure")
        destination.mkdir(parents=True)
        for name in ("after-policy.pt", "after-resume.pt"):
            (destination / name).write_bytes((plan_path.parent / "execution" / name).read_bytes())
        updates = runner.read(plan_path)["synthetic_updates"]
        return {"restore": {"progress": {"updates": updates, "rollouts": updates * 4,
                "decisions": 95836 + (updates - 412) * 20}}}

    def verify_copy(self, *args, **kwargs):
        if self.restore_copy_failure:
            raise ValueError("synthetic independent copy mismatch")
        return {"exact_model_adam_rng_progress_pool": True}

    def result(self):
        return runner.read(self.run / "summary.json")

    def target(self, games, evaluation_interval_updates=5):
        self.session.update(target_games=games, max_updates=None,
                            evaluation_interval_updates=evaluation_interval_updates,
                            profile="original")
        runner.atomic(self.session_path, self.session)

    def test_baseline_then_training_preservation_and_latest_pointer(self):
        self.assertEqual(runner.worker(self.session_path), 0)
        summary = self.result()
        self.assertEqual([r["mode"] for r in self.started], ["baseline", "baseline", "train", "final", "final"])
        self.assertEqual([r["seat"] for r in self.started[:2]], [0, 1])
        self.assertEqual(summary["new_ppo_updates"], 1)
        self.assertEqual(summary["admitted_training_games"], 4)
        self.assertEqual(summary["evaluation_games"], 16)
        self.assertEqual(summary["training_outcomes"], {"win": 4})
        self.assertEqual(len(self.preserved), 1)
        latest = runner.read(self.root / "latest-checkpoint.json")
        self.assertEqual(latest["parent"], summary["latest_parent"])
        self.assertEqual(latest["progress"]["updates"], 413)
        self.assertEqual(self.restore_calls[-1], latest["parent"])
        self.assertTrue(summary["initial_state_unchanged"])
        self.assertTrue(summary["owned_cleanup_verified"])
        self.assertEqual(runner.read(self.root / "sequence-ledger.json")["next_sequence"], 267)

    def test_partial_or_failed_batch_is_retained_but_never_admitted(self):
        self.bad_training = True
        self.assertEqual(runner.worker(self.session_path), 1)
        summary = self.result()
        self.assertEqual(summary["status"], "stopped_failure")
        self.assertEqual(summary["new_ppo_updates"], 0)
        self.assertEqual(summary["latest_parent"], self.parent)
        self.assertEqual(len(summary["failed_batches"]), 1)
        self.assertEqual(summary["failed_batches"][0]["completed_games"], 3)
        self.assertFalse((self.root / "latest-checkpoint.json").exists())
        self.assertEqual(self.preserved, [])
        self.assertEqual([r["mode"] for r in self.started], ["baseline", "baseline", "train"])
        self.assertEqual(runner.read(self.root / "sequence-ledger.json")["next_sequence"], 267)

    def test_failed_preservation_does_not_advance_existing_latest(self):
        existing = {"parent": self.parent, "marker": "previous verified state"}
        runner.atomic(self.root / "latest-checkpoint.json", existing)
        self.preservation_failure = True
        self.assertEqual(runner.worker(self.session_path), 1)
        self.assertEqual(runner.read(self.root / "latest-checkpoint.json"), existing)
        self.assertEqual(self.result()["latest_parent"], self.parent)
        self.assertEqual(self.result()["new_ppo_updates"], 0)
        self.assertIn("full-disk", self.result()["error"])

    def test_failed_independent_restore_does_not_publish_latest(self):
        self.restore_copy_failure = True
        self.assertEqual(runner.worker(self.session_path), 1)
        self.assertFalse((self.root / "latest-checkpoint.json").exists())
        self.assertEqual(self.result()["new_ppo_updates"], 0)
        self.assertEqual(self.result()["latest_parent"], self.parent)
        self.assertIn("copy mismatch", self.result()["error"])

    def test_stop_requested_before_start_prevents_any_baseline(self):
        (self.run / "stop-requested").touch()
        self.assertEqual(runner.worker(self.session_path), 0)
        self.assertEqual(self.result()["status"], "stopped_requested")
        self.assertEqual(self.started, [])
        self.assertEqual(self.result()["evaluation_games"], 0)
        self.assertFalse((self.root / "sequence-ledger.json").exists())

    def test_stop_during_first_baseline_prevents_second_baseline(self):
        self.stop_after = "baseline"
        self.assertEqual(runner.worker(self.session_path), 0)
        self.assertEqual(self.result()["status"], "stopped_requested")
        self.assertEqual([r["mode"] for r in self.started], ["baseline"])
        self.assertEqual(self.result()["new_ppo_updates"], 0)

    def test_stop_during_valid_training_preserves_it_but_starts_no_final(self):
        self.stop_after = "train"
        self.assertEqual(runner.worker(self.session_path), 0)
        self.assertEqual(self.result()["status"], "stopped_requested")
        self.assertEqual([r["mode"] for r in self.started], ["baseline", "baseline", "train"])
        self.assertEqual(self.result()["new_ppo_updates"], 1)
        self.assertEqual(self.result()["final"], [])
        self.assertEqual(runner.read(self.root / "latest-checkpoint.json")["progress"]["updates"], 413)

    def test_eight_game_target_excludes_evaluation_and_skips_periodic_at_target(self):
        self.target(8, evaluation_interval_updates=2)
        self.assertEqual(runner.worker(self.session_path), 0)
        summary = self.result()
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["stop_reason"], "training_game_target_reached")
        self.assertTrue(summary["target_games_met"])
        self.assertEqual(summary["target_games"], 8)
        self.assertEqual(summary["profile"], "original")
        self.assertEqual(summary["new_ppo_updates"], 2)
        self.assertEqual(summary["admitted_training_games"], 8)
        self.assertEqual(summary["evaluation_games"], 16)
        self.assertEqual([r["mode"] for r in self.started],
                         ["baseline", "baseline", "train", "train", "final", "final"])
        self.assertEqual(len(self.preserved), 2)
        self.assertEqual(runner.read(self.root / "sequence-ledger.json")["next_sequence"], 268)
        self.assertEqual(runner.read(self.root / "latest-checkpoint.json")["progress"]["updates"], 414)
        final_parents = [r["parent"] for r in summary["final"]]
        self.assertEqual(final_parents, [summary["latest_parent"], summary["latest_parent"]])

    def test_twelve_game_target_uses_configured_periodic_interval_without_extra_ppo(self):
        self.target(12, evaluation_interval_updates=1)
        self.assertEqual(runner.worker(self.session_path), 0)
        summary = self.result()
        self.assertEqual(summary["stop_reason"], "training_game_target_reached")
        self.assertTrue(summary["target_games_met"])
        self.assertEqual(summary["new_ppo_updates"], 3)
        self.assertEqual(summary["admitted_training_games"], 12)
        self.assertEqual(summary["evaluation_games"], 32)
        self.assertEqual([r["mode"] for r in self.started], [
            "baseline", "baseline", "train", "periodic", "periodic",
            "train", "periodic", "periodic", "train", "final", "final"])
        self.assertEqual([r["sequence"] for r in self.started if r["mode"] == "train"], [266, 267, 268])
        self.assertEqual(runner.read(self.root / "latest-checkpoint.json")["progress"]["updates"], 415)

    def test_budget_stop_before_target_reports_partial_progress_without_extra_training(self):
        self.target(12)
        self.enough.side_effect = lambda session, durations, gamesets=1: (
            gamesets != 3 or sum(row["mode"] == "train" for row in self.started) < 1)
        runner.worker(self.session_path)
        summary = self.result()
        self.assertEqual(summary["status"], "stopped_budget")
        self.assertEqual(summary["stop_reason"], "compute_budget_reserve")
        self.assertFalse(summary["target_games_met"])
        self.assertEqual(summary["target_games"], 12)
        self.assertEqual(summary["admitted_training_games"], 4)
        self.assertEqual(summary["new_ppo_updates"], 1)
        self.assertEqual(sum(row["mode"] == "train" for row in self.started), 1)
        self.assertEqual(runner.read(self.root / "sequence-ledger.json")["next_sequence"], 267)

    def test_failed_target_batch_never_counts_partial_games_toward_target(self):
        self.target(8); self.bad_training = True
        self.assertEqual(runner.worker(self.session_path), 1)
        summary = self.result()
        self.assertEqual(summary["status"], "stopped_failure")
        self.assertFalse(summary["target_games_met"])
        self.assertEqual(summary["admitted_training_games"], 0)
        self.assertEqual(summary["new_ppo_updates"], 0)
        self.assertEqual(summary["failed_batches"][0]["completed_games"], 3)
        self.assertEqual(summary["evaluation_games"], 8)


if __name__ == "__main__":
    unittest.main()
