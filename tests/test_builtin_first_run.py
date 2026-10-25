"""Bounded pilot orchestration and synthetic checkpoint fixtures; no native games."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.learner import builtin_first_run as pilot


class PilotPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def plan(self):
        cfg = pilot.config()
        return {"schema": pilot.SCHEMA + "-plan", "authorization": deepcopy(pilot.AUTHORIZATION),
            "configuration": cfg, "schedule": pilot.schedule(cfg),
            "execution_output": str(self.root / "execution"),
            "checkpoint": {"sha256": pilot.CHECKPOINT_SHA}}

    def test_inclusive_dialog_reserve_and_complete_batch_fit_optimizer_cap(self):
        cfg, rows = pilot.config(), pilot.schedule(pilot.config())
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["phase"] for r in rows], ["before"] * 2 + ["training"] * 4 + ["after"] * 2)
        self.assertEqual([r["episode"] for r in rows], list(range(1, 9)))
        self.assertEqual([r["learner_player"] for r in rows], [0, 1] * 4)
        for before, after in zip(rows[:2], rows[6:]):
            self.assertEqual({k: before[k] for k in ("engine_seed", "policy_seed", "learner_player")},
                             {k: after[k] for k in ("engine_seed", "policy_seed", "learner_player")})
        self.assertFalse({r["engine_seed"] for r in rows[2:6]} & {r["engine_seed"] for r in rows[:2]})
        self.assertFalse({r["policy_seed"] for r in rows[2:6]} & {r["policy_seed"] for r in rows[:2]})
        self.assertEqual(cfg["ordinary_choice_limit"] + cfg["reserved_dialog_choices"], 256)
        self.assertEqual(cfg["max_total_choices"], 256)
        windows = 4 * math.ceil(cfg["max_total_choices"] / cfg["ppo"]["decisions_per_window"])
        steps = cfg["ppo"]["epochs"] * math.ceil(windows / cfg["ppo"]["windows_per_minibatch"])
        self.assertEqual(steps, cfg["max_optimizer_steps"])
        self.assertEqual((steps, cfg["max_updates"]), (32, 1))
        self.assertLess(cfg["phase_deadlines_seconds"]["update"], cfg["phase_deadlines_seconds"]["after"])
        self.assertGreaterEqual(cfg["shared_wall_seconds"] - cfg["phase_deadlines_seconds"]["update"], 600)

    def test_rehashed_plan_still_rejects_schedule_scope_and_numeric_alias_tampering(self):
        path = self.root / "plan.json"
        base = self.plan()
        pilot.write(path, base)
        self.assertEqual(pilot.read_plan(path, pilot.digest(path)), base)
        for change in (lambda p: p["schedule"][0].update(learner_player=False),
                       lambda p: p["schedule"][0].update(episode=1.0),
                       lambda p: p["schedule"].pop(),
                       lambda p: p["schedule"].reverse(),
                       lambda p: p["configuration"].update(max_total_choices=264),
                       lambda p: p["configuration"]["ppo"].update(epochs=2),
                       lambda p: p["checkpoint"].update(sha256="a" * 64),
                       lambda p: p["authorization"].update(verified_direct_human_instruction=False),
                       lambda p: p.update(execution_output=str(self.root / "elsewhere"))):
            data = deepcopy(base); change(data); pilot.write(path, data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.read_plan(path, pilot.digest(path))
        pilot.write(path, base)
        saved = pilot.digest(path)
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            pilot.read_plan(path, saved)

    def test_config_cannot_raise_scope_or_use_boolean_integer_aliases(self):
        path = self.root / "config.json"
        base = pilot.config()
        for change in (lambda p: p.update(max_updates=True), lambda p: p.update(max_optimizer_steps=33),
                       lambda p: p.update(shared_wall_seconds=3601),
                       lambda p: p["ppo"].update(gamma=0.99),
                       lambda p: p["ppo"].update(entropy_coefficient=0.01)):
            data = deepcopy(base); change(data); pilot.write(path, data)
            with patch.object(pilot, "CONFIG", path), self.assertRaises(ValueError):
                pilot.config()

    def test_phase_and_attempt_clocks_use_earlier_absolute_clock(self):
        with patch.object(pilot.time, "monotonic", return_value=100), patch.object(pilot.time, "time", return_value=1000):
            shared = pilot.Budget(200, 1050)
            self.assertEqual(shared.remaining(), 50)
            attempt = shared.capped(80)
            self.assertEqual((attempt.deadline, attempt.wall_deadline), (180, 1050))
            self.assertEqual(attempt.remaining(), 50)
        with patch.object(pilot.time, "monotonic", return_value=110), patch.object(pilot.time, "time", return_value=1051):
            with self.assertRaises(TimeoutError):
                attempt.remaining()


class PilotSupervisorTests(PilotPlanTests):
    def execute(self, *, interrupted=False, spawn_error=None, poll_error=None):
        from experiments.headless import process_deadline
        plan = {**self.plan(), "wall_scope": "synthetic-clock-test", "comparison_scope": "synthetic-only"}
        args = SimpleNamespace(plan=self.root / "plan.json", sha256="a" * 64)
        process = SimpleNamespace(pid=123456, poll=Mock(side_effect=poll_error) if poll_error
                                  else lambda: 1 if interrupted else 0)
        def spawn(*args, **kwargs):
            if spawn_error:
                raise spawn_error
            report = json.loads((Path(plan["execution_output"]) / "report.json").read_text())
            report.update(worker_complete=not interrupted, status="stopped" if interrupted else "finished")
            if interrupted:
                report.update(update_status="started", optimizer_updates=1, optimizer_steps=2)
                report["attempts"][0].update(status="pending_verification", outcomes={"red": "win", "blue": "loss"},
                                             rewards={"red": 1, "blue": -1}, training_samples=10)
            pilot.write(Path(plan["execution_output"]) / "worker-report.json", report)
            return process
        with patch.object(pilot, "read_plan", return_value=plan), \
             patch.object(process_deadline, "guarded_popen", side_effect=spawn) as launched, \
             patch.object(process_deadline, "close_owned_process") as closed, \
             patch.object(process_deadline, "cleanup_from_evidence", return_value=()) as cleaned:
            report = pilot.execute(args)
        return report, plan, args, launched, closed, cleaned

    def test_failed_worker_pending_rows_remain_unscored_and_update_unknown(self):
        report, _, _, launched, closed, _ = self.execute(interrupted=True)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["worker_complete"])
        self.assertEqual(len(report["attempts"]), 8)
        first = report["attempts"][0]
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(first["outcomes"], {"red": None, "blue": None})
        self.assertEqual(first["rewards"], {"red": None, "blue": None})
        self.assertEqual(first["training_samples"], 0)
        self.assertEqual(report["update_status"], "interrupted_unknown")
        self.assertIsNone(report["optimizer_steps"])
        self.assertIsNone(report["optimizer_updates"])
        self.assertIsNone(report["training_performed"])
        self.assertTrue(all(row["status"] == "not_started" for row in report["attempts"][1:]))
        launched.assert_called_once(); closed.assert_called_once()

    def test_failed_spawn_retains_all_rows_and_output_cannot_be_reexecuted(self):
        report, plan, args, launched, closed, _ = self.execute(spawn_error=OSError("synthetic-spawn-failure"))
        self.assertEqual(report["status"], "stopped")
        self.assertEqual([row["status"] for row in report["attempts"]], ["not_started"] * 8)
        self.assertEqual(report["optimizer_steps"], 0)
        launched.assert_called_once(); closed.assert_not_called()
        with patch.object(pilot, "read_plan", return_value=plan), self.assertRaises(FileExistsError):
            pilot.execute(args)

    def test_poll_failure_reads_latest_durable_worker_journal_before_cleanup(self):
        report, _, _, launched, closed, _ = self.execute(
            interrupted=True, poll_error=OSError("synthetic-poll-failure"))
        self.assertEqual(report["update_status"], "interrupted_unknown")
        self.assertIsNone(report["optimizer_steps"])
        self.assertEqual(report["attempts"][0]["status"], "incomplete")
        self.assertEqual(report["attempts"][0]["rewards"], dict.fromkeys(pilot.COLORS))
        self.assertIn("synthetic-poll-failure", report["supervisor_error"])
        self.assertTrue(report["supervisor_cleanup_verified"])
        launched.assert_called_once(); closed.assert_called_once()

    def test_worker_passed_rows_need_exact_identity_and_full_durable_completion(self):
        plan = self.plan()
        path = self.root / "worker-report.json"
        report = {"plan_sha256": "a" * 64,
                  "attempts": [pilot.empty_attempt(row) for row in plan["schedule"]]}
        report["attempts"][0].update(status="passed", completion_receipt=True,
            cleanup_verified=True, identity_verified=True, replay_verified=True)
        pilot.write(path, report)
        self.assertEqual(pilot.read_worker_report(path, plan, "a" * 64), report)
        for field in ("completion_receipt", "cleanup_verified", "identity_verified", "replay_verified"):
            for value in (False, 1, None):
                changed = deepcopy(report); changed["attempts"][0][field] = value
                pilot.write(path, changed)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    pilot.read_worker_report(path, plan, "a" * 64)
        for field, value in (("learner_player", False), ("episode", 1.0)):
            changed = deepcopy(report); changed["attempts"][0][field] = value
            pilot.write(path, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                pilot.read_worker_report(path, plan, "a" * 64)


class PilotFrozenCheckpointTests(unittest.TestCase):
    def test_roundtrip_frozen_policy_with_empty_and_populated_synthetic_adam(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("Optional PyTorch is not installed")
        from experiments.learner import checkpoints as cp, contextual as c, contextual_model as m
        from tests.test_builtin_gameplay_checkpoint import equal_state, synthetic_adam_step

        threads, python_rng = torch.get_num_threads(), random.getstate()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(random.setstate, python_rng)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            torch.manual_seed(181); random.seed(182)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA,
                                                         hidden_size=8, feature_size=4))
            optimizer = torch.optim.Adam(model.parameters(), **pilot.config()["adam"])
            generator = torch.Generator().manual_seed(183)
            policy_rng = torch.Generator().manual_seed(184)
            pool = cp.FrozenOpponentPool(); pool.add("synthetic-archive-not-executed", model)
            initial = root / "initial.pt"
            cp.save_policy_checkpoint(initial, model, checkpoint="synthetic-initial",
                source_metadata={"purpose": "synthetic-roundtrip-no-native"}, opponent_pool=pool,
                interface=c.DIALOGS_ENVIRONMENT)
            metadata = {"purpose": "synthetic-roundtrip-no-native", "plan_sha256": "a" * 64,
                "initial_checkpoint": pilot.binding(initial),
                "native_opponent": {"kind": "builtin", "module": "Nullkiller", "difficulty": 0,
                                    "binary_sha256": "b" * 64},
                "archived_neural_pool_executed": False}
            archive = deepcopy(pool.state_dict())
            for step, name in enumerate(("before", "after")):
                if step:
                    model.train().requires_grad_(True)
                    synthetic_adam_step(model, optimizer, policy_rng, generator)
                    self.assertTrue(optimizer.state)
                model.eval().requires_grad_(False)
                expected = deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch": torch.get_rng_state(), "python": random.getstate(), "minibatch": generator.get_state()})
                loaded, evidence = pilot.checkpoint_roundtrip(root, model, optimizer, pool, generator,
                    metadata, pilot.config(), cp.TrainingProgress(step, step, 2 * step),
                    {"next_schedule_index": 6 if step else 0, "optimizer_steps": step}, name)
                equal_state(self, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch": torch.get_rng_state(), "python": random.getstate(),
                    "minibatch": generator.get_state()}, expected)
                equal_state(self, loaded.model.state_dict(), expected["model"])
                equal_state(self, pool.state_dict(), archive)
                equal_state(self, loaded.opponent_pool.state_dict(), archive)
                self.assertEqual(loaded.provenance, cp.BUILTIN_GAMEPLAY_PROVENANCE)
                self.assertTrue(evidence["exact_model_adam_rng_progress_pool"])
                self.assertFalse(model.training or loaded.model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertFalse(any(p.requires_grad for p in loaded.model.parameters()))
            with patch.object(cp, "save_checkpoint", side_effect=OSError("synthetic-save-failure")), \
                 self.assertRaisesRegex(OSError, "synthetic-save-failure"):
                pilot.checkpoint_roundtrip(root, model, optimizer, pool, generator, metadata, pilot.config(),
                    cp.TrainingProgress(), {}, "failed")
            self.assertFalse(any(p.requires_grad for p in model.parameters()))


class PilotAttemptCleanupTests(PilotPlanTests):
    def run_attempt(self, *, env_close_error=False, cleanup_write_error=False, old_training_folder=False):
        from experiments.headless import builtin_env, builtin_replay, process_deadline
        from experiments.learner import builtin_behavior, builtin_gameplay, builtin_rollout, contextual_model
        from validation.probes import validate_hero_policy
        cfg = pilot.config()
        plan = {**self.plan(), "binary": str(self.root / "vcmiclient"), "data": str(self.root / "data"),
            "game_map": str(self.root / "map.vmap"), "runtime": {}, "source": {"interface": pilot.INTERFACE}, "native": {},
            "replay_binary": {"path": str(self.root / "vcmireplay")},
            "checkpoint": {"path": str(self.root / "initial.pt"), "sha256": "c" * 64}}
        output = Path(plan["execution_output"]); output.mkdir()
        index = 2 if old_training_folder else 0
        row = plan["schedule"][index]
        folder = output / f"episode-{row['episode']:06d}"
        if old_training_folder:
            folder.mkdir(); (folder / "builtin-rollout.json").write_text('{"purpose":"diagnostic"}')
        calls, snapshots = [], []
        manifest = {"status": "completed", "termination_reason": "natural", "outcomes_by_player": {"red": "win", "blue": "loss"},
                    "rewards_by_player": {"red": 1, "blue": -1}}
        class Launcher:
            def __init__(self, *args, **kwargs): pass
            def close_owned_groups(self, **kwargs): calls.append("groups")
        class Env:
            def __init__(self, *args, **kwargs): calls.append("env")
            def close(self):
                calls.append("env.close")
                if env_close_error: raise OSError("synthetic-close-failure")
        class Collector:
            def __init__(self, *args, **kwargs): calls.append("collector")
            def run_episode(self, *args, **kwargs):
                calls.append("collect")
                folder.mkdir(); pilot.write(folder / "manifest.json", manifest)
                return SimpleNamespace(decisions=(object(),), to_dict=lambda: {"synthetic": True})
        def save(record):
            snapshots.append(deepcopy(record))
            if cleanup_write_error and record.get("stage") == "cleanup":
                raise OSError("synthetic-progress-write-failure")
        with ExitStack() as stack:
            for owner, name, value in (
                (pilot, "check_identity", lambda plan: None),
                (pilot, "native_opponent", lambda runtime, seat: {"synthetic": True}),
                (contextual_model, "weights_sha256", lambda model: "b" * 64),
                (process_deadline, "DeadlineVcmiLauncher", Launcher),
                (builtin_env, "BuiltinHeadlessEnv", Env),
                (builtin_rollout, "BuiltinCollector", Collector),
                (builtin_rollout, "audit_persisted_builtin_policy", lambda *a, **k: {"passed": True}),
                (builtin_rollout, "retain_failed_builtin_evidence", lambda *a, **k: calls.append("retained")),
                (builtin_replay, "verify_episode", lambda *a, **k: {"verified": True}),
                (builtin_behavior, "analyze_episode", lambda *a, **k: {"synthetic": True}),
                (validate_hero_policy, "wall_limit", lambda *a, **k: nullcontext()),
            ):
                stack.enter_context(patch.object(owner, name, value))
            fresh = stack.enter_context(patch.object(builtin_gameplay, "TrainingCollector"))
            result = pilot.run_attempt(plan, self.root / "plan.json", "a" * 64, index, object(), "synthetic-policy",
                                       pilot.Budget(pilot.time.monotonic() + 240, pilot.time.time() + 240), save)
        return result, calls, snapshots, fresh

    def test_environment_close_failure_does_not_skip_owned_group_cleanup(self):
        (attempt, receipt), calls, _, _ = self.run_attempt(env_close_error=True)
        self.assertIn("groups", calls)
        self.assertEqual(attempt["status"], "failed")
        self.assertIsNone(receipt)
        self.assertEqual(attempt["outcomes"], {"red": None, "blue": None})
        self.assertFalse(attempt["completion_receipt"])

    def test_progress_write_failure_cannot_prevent_cleanup_or_claim_success(self):
        (attempt, receipt), calls, _, _ = self.run_attempt(cleanup_write_error=True)
        self.assertIn("env.close", calls)
        self.assertIn("groups", calls)
        self.assertEqual(attempt["status"], "failed")
        self.assertIsNone(receipt)
        self.assertFalse(attempt["completion_receipt"])

    def test_existing_diagnostic_folder_is_never_imported_as_training_capture(self):
        (attempt, receipt), calls, _, fresh = self.run_attempt(old_training_folder=True)
        self.assertEqual(attempt["status"], "failed")
        self.assertIn("previously absent", attempt["error"])
        self.assertNotIn("collect", calls)
        self.assertIsNone(receipt)
        fresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
