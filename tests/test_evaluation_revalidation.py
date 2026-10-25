"""New cohort provenance and timing, without games or loading model weights."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import evaluate_long_dialogs as neural
from experiments.learner import evaluate_builtin_opponent as builtin
from experiments.learner import evaluation_revalidation as r


class RevalidationTests(unittest.TestCase):
    def common(self, root):
        cfg = neural.config()
        source = {"git_commit": "synthetic-repaired-source", "tracked_worktree_dirty": False,
                  "python": "test-python", "torch": "test-torch", "interface": neural.INTERFACE,
                  "source_sha256": {"learner.py": "unchanged"}}
        return {"source": source, "implementation": {"both-configs-and-code": "unchanged"},
            "native": {"binary_sha256": "client"}, "runtime": {"libvcmi": "rules", "AI/libNullkiller": "ai"},
            "map": {"sha256": cfg["map_sha256"]},
            "checkpoint": {"path": str(root / "unused.pt"), "bytes": 1, "sha256": cfg["checkpoint_sha256"]},
            "frozen": {"learner_weights_sha256": cfg["learner_weights_sha256"]},
            "checkpoint_source": source, "checkpoint_provenance": "random_initialization",
            "binary": str(root / "unused-client"), "data": str(root / "unused-data"),
            "game_map": str(root / "unused-map"),
            "replay_binary": {"path": str(root / "unused-replay"), "sha256": "replay", "bytes": 1}}

    def save(self, module, folder, plan, *, status="stopped"):
        folder.mkdir(parents=True, exist_ok=True)
        module.write(folder / "plan.json", plan)
        report = {"schema": module.result_schema(plan), "status": status,
            "plan_sha256": module.digest(folder / "plan.json"),
            "attempts": [module.empty_attempt(row) for row in plan["schedule"]],
            "training_performed": False, "optimizer_updates": 0, "optimizer_steps": 0,
            "training_samples": 0, "frozen_before": plan["frozen"],
            "frozen_after_file_sha256": plan["checkpoint"]["sha256"]}
        execution = folder / "execution"
        execution.mkdir(exist_ok=True)
        module.write(execution / "report.json", report)
        return folder / "plan.json", execution / "report.json"

    def old(self, root, module, label):
        folder = root / label
        cfg = module.config()
        plan = {**self.common(root), "schema": module.SCHEMA + "-plan", "configuration": cfg,
            "schedule": module.schedule(cfg), "execution_output": str(folder / "execution"),
            "training_performed": False, "authorization": {"historical": True}}
        return self.save(module, folder, plan)

    def new(self, root, module, old_paths, label, *, reference=None):
        cfg = module.config()
        folder = root / label
        args = SimpleNamespace(output=folder, revalidation_v2=True,
            prior_plan=old_paths[0], prior_report=old_paths[1])
        provenance = r.prepare_provenance(args, reader=module.read_plan, legacy_schema=module.SCHEMA,
            configuration=cfg, wanted_schedule=module.schedule(cfg))
        plan = {**self.common(root), "schema": module.V2_SCHEMA + "-plan", "configuration": cfg,
            "schedule": module.schedule(cfg), "execution_output": str(folder / "execution"),
            "training_performed": False, "authorization": dict(r.AUTHORIZATION), "revalidation": provenance}
        if reference is not None:
            plan["reference"] = builtin.reference(*reference)
        return plan, self.save(module, folder, plan)

    def read(self, module, path):
        return module.read_plan(path, module.digest(path))

    def test_v1_reader_remains_historical_without_current_source_or_seed_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            for module, label in ((neural, "old-a"), (builtin, "old-b")):
                paths = self.old(Path(tmp).resolve(), module, label)
                before = [p.read_bytes() for p in paths]
                value = self.read(module, paths[0])
                self.assertEqual(module.plan_version(value), 1)
                self.assertEqual(module.result_schema(value), module.SCHEMA + "-result")
                self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_v2_reader_binds_reused_seeds_and_preserves_old_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old = self.old(root, neural, "old-a")
            before = [p.read_bytes() for p in old]
            plan, paths = self.new(root, neural, old, "new-a")
            actual = self.read(neural, paths[0])
            self.assertEqual(actual, plan)
            self.assertEqual(neural.plan_version(actual), 2)
            self.assertEqual(neural.result_schema(actual), neural.V2_SCHEMA + "-result")
            self.assertNotIn("seed_freshness", actual)
            self.assertFalse(actual["revalidation"]["held_out"])
            self.assertEqual(actual["revalidation"]["prior"]["schedule"], plan["schedule"])
            self.assertEqual(before, [p.read_bytes() for p in old])

    def test_reused_seed_authorization_and_preservation_claim_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan, paths = self.new(root, neural, self.old(root, neural, "old-a"), "new-a")
            mutations = [lambda p: p["revalidation"].update(held_out=True),
                lambda p: p["revalidation"].update(seed_use="fresh_seeds"),
                lambda p: p["revalidation"].update(replaces_prior_attempts=True),
                lambda p: p.update(seed_freshness={"sha256": "stale-proof"}),
                lambda p: p["authorization"].update(human_turn="old-authorization"),
                lambda p: p["source"].update(tracked_worktree_dirty=True),
                lambda p: p["configuration"].update(optimizer_steps=False),
                lambda p: p["configuration"].update(max_days=200.0),
                lambda p: p["schedule"][0].update(learner_player=False),
                lambda p: p["revalidation"]["prior"]["plan"].update(sha256="0" * 64),
                lambda p: p["schedule"][0].update(engine_seed=123)]
            for mutate in mutations:
                changed = copy.deepcopy(plan)
                mutate(changed)
                neural.write(paths[0], changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    self.read(neural, paths[0])

    def test_later_mutation_of_prior_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old = self.old(root, neural, "old-a")
            _, paths = self.new(root, neural, old, "new-a")
            report = json.loads(old[1].read_text())
            report["historical_extra"] = "changed even though outcome fields were retained"
            neural.write(old[1], report)
            with self.assertRaisesRegex(ValueError, "prior cohort artifacts changed"):
                self.read(neural, paths[0])

    def test_missing_explicit_mode_or_old_output_overlap_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old = self.old(root, neural, "old-a")
            common = dict(reader=neural.read_plan, legacy_schema=neural.SCHEMA,
                configuration=neural.config(), wanted_schedule=neural.schedule(neural.config()))
            for args in (SimpleNamespace(output=root / "new", prior_plan=old[0], prior_report=old[1]),
                    SimpleNamespace(output=root / "new", revalidation_v2=True),
                    SimpleNamespace(output=old[0].parent / "replacement", revalidation_v2=True,
                        prior_plan=old[0], prior_report=old[1])):
                with self.assertRaises(ValueError):
                    r.prepare_provenance(args, **common)
            self.assertIsNone(r.prepare_provenance(SimpleNamespace(), **common))

    def test_builtin_v2_requires_new_matching_neural_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old_a = self.old(root, neural, "old-a")
            old_b = self.old(root, builtin, "old-b")
            _, new_a = self.new(root, neural, old_a, "new-a")
            plan, new_b = self.new(root, builtin, old_b, "new-b", reference=new_a)
            self.assertEqual(self.read(builtin, new_b[0]), plan)
            for mutate in (lambda p: p.update(reference=builtin.reference(*old_a)),
                    lambda p: p["runtime"].update(libvcmi="changed-rules"),
                    lambda p: p["runtime"].update({"AI/libNullkiller": "changed-ai"}),
                    lambda p: p["source"].update(git_commit="different-code"),
                    lambda p: p["implementation"].update(test="changed-source"),
                    lambda p: p["replay_binary"].update(sha256="diagnostic-verifier-only-drift"),
                    lambda p: p["checkpoint"].update(path="different-checkpoint-path"),
                    lambda p: p["schedule"][0].update(policy_seed=120602)):
                changed = copy.deepcopy(plan)
                mutate(changed)
                builtin.write(new_b[0], changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    self.read(builtin, new_b[0])

    def test_builtin_reference_rejects_running_wrong_version_or_rewritten_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old_a = self.old(root, neural, "old-a")
            _, new_a = self.new(root, neural, old_a, "new-a")
            _, new_b = self.new(root, builtin, self.old(root, builtin, "old-b"), "new-b", reference=new_a)
            report = json.loads(new_a[1].read_text())
            for mutate in (lambda p: p.update(status="running"), lambda p: p.update(schema=neural.SCHEMA + "-result"),
                    lambda p: p.update(status="finished"), lambda p: p["attempts"][0].update(rewards={"red": -1, "blue": 1}),
                    lambda p: p["attempts"][0].update(learner_player=False),
                    lambda p: p["attempts"][0].update(episode=1.0),
                    lambda p: p.update(training_samples=False), lambda p: p.update(extra="changed")):
                changed = copy.deepcopy(report)
                mutate(changed)
                neural.write(new_a[1], changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    self.read(builtin, new_b[0])

    def test_reference_frozen_scalar_types_are_exact_before_preparing_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old = self.old(root, neural, "old-a")
            plan, paths = self.new(root, neural, old, "new-a")
            plan["frozen"]["numeric_fixture"] = 0
            paths = self.save(neural, paths[0].parent, plan)
            report = json.loads(paths[1].read_text())
            report["frozen_before"]["numeric_fixture"] = False
            neural.write(paths[1], report)
            with self.assertRaisesRegex(ValueError, "frozen evaluation declaration"):
                builtin.reference(*paths)

    def test_both_implementation_inventories_bind_both_configs(self):
        with patch("experiments.learner.first_run.implementation_identity", side_effect=lambda: {}):
            self.assertEqual(neural.implementation(2), builtin.implementation(2))
            self.assertNotEqual(neural.implementation(1), builtin.implementation(1))
            self.assertIn("configs/long-dialog-evaluation-v1.json", builtin.implementation(2))
            self.assertIn("configs/builtin-opponent-evaluation-v1.json", neural.implementation(2))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Locked learner Python is required for preparation imports")
    def test_prepare_v2_uses_new_authorization_without_old_freshness_and_matches_both_cohorts(self):
        """Real preparation orchestration, with engine/model work mocked out."""
        from contextlib import ExitStack
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            common = self.common(root)
            old_a, old_b = self.old(root, neural, "old-a"), self.old(root, builtin, "old-b")
            reference = None
            for module, old, label in ((neural, old_a, "new-a"), (builtin, old_b, "new-b")):
                args = SimpleNamespace(output=root / label, checkpoint=Path(common["checkpoint"]["path"]),
                    binary=Path(common["binary"]), data=Path(common["data"]), map=Path(common["game_map"]),
                    replay_binary=Path(common["replay_binary"]["path"]), revalidation_v2=True,
                    prior_plan=old[0], prior_report=old[1])
                if reference is not None:
                    args.reference_plan, args.reference_report = reference
                loaded = SimpleNamespace(source_metadata=common["checkpoint_source"],
                    provenance=common["checkpoint_provenance"])
                launcher = Mock()
                launcher.identity.return_value = common["native"]
                def artifact(path):
                    for key in ("checkpoint", "replay_binary"):
                        if str(path) == common[key]["path"]:
                            return copy.deepcopy(common[key])
                    return r.binding(path)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(module, "binding", side_effect=artifact))
                    stack.enter_context(patch.object(module, "load_fixed", return_value=(loaded, common["frozen"])))
                    stack.enter_context(patch.object(module, "implementation", return_value=common["implementation"]))
                    stack.enter_context(patch.object(module.subprocess, "run", return_value=SimpleNamespace(stdout="synthetic recipe check")))
                    stack.enter_context(patch("experiments.headless.env.VcmiLauncher", return_value=launcher))
                    stack.enter_context(patch("experiments.headless.contextual_evidence.runtime_identity", return_value=common["runtime"]))
                    stack.enter_context(patch("experiments.learner.inference.source_identity", return_value=common["source"]))
                    stack.enter_context(patch("validation.probes.validate_hero_policy.hero_map_identity", return_value=common["map"]))
                    if module is builtin:
                        stack.enter_context(patch.object(module, "native_opponent", return_value={"synthetic": True}))
                    prepared = module.prepare(args)
                path = Path(prepared["plan"])
                plan = module.read_plan(path, prepared["sha256"])
                self.assertEqual(plan["authorization"], r.AUTHORIZATION)
                self.assertEqual(module.plan_version(plan), 2)
                self.assertNotIn("seed_freshness", plan)
                self.assertEqual(plan["configuration"], module.config())
                reference = self.save(module, args.output, plan)
                with self.assertRaisesRegex(ValueError, "output exists"):
                    module.prepare(args)

    def test_phase_timings_accumulate_progress_and_include_failure_cleanup(self):
        times = iter((11.0, 15.0, 16.0, 18.0, 22.0, 23.0))
        timer = r.PhaseTimings(10.0, clock=lambda: next(times))
        snapshots = [timer.advance(p) for p in ("collection", "collection", "failure_retention", "cleanup", "finished", "finished")]
        self.assertEqual(snapshots[1][0], {"initialization": 1.0, "collection": 4.0})
        final, now = snapshots[-2]
        self.assertEqual(final, {"initialization": 1.0, "collection": 5.0, "failure_retention": 2.0, "cleanup": 4.0})
        self.assertEqual(sum(final.values()), now - 10.0)
        self.assertEqual(snapshots[-1][0], final)
        self.assertNotIn("native_replay", final)  # Unexecuted stages are not measured as zero.

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Locked learner Python is required for worker imports")
    def test_timing_write_failures_do_not_skip_retention_or_either_cleanup(self):
        """Run each actual worker with mock native/model boundaries only."""
        from contextlib import ExitStack
        from unittest.mock import Mock
        import time
        class FailingCollector:
            def __init__(self, *args, **kwargs):
                pass

            def run_episode(self, *args, **kwargs):
                raise ValueError("synthetic collection failure")

        for module in (neural, builtin):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp).resolve()
                cfg = module.config()
                plan = {**self.common(root), "schema": module.V2_SCHEMA + "-plan", "configuration": cfg,
                    "schedule": module.schedule(cfg), "execution_output": str(root),
                    "native_opponents": ["native-opponent", "native-opponent"]}
                env = Mock(episode_dir=None)
                env.close.side_effect = OSError("synthetic environment-close failure")
                launcher = Mock()
                loaded = SimpleNamespace(model=object(), checkpoint="frozen-model",
                    opponent_pool=SimpleNamespace(get=lambda _: SimpleNamespace(model=object())))
                args = SimpleNamespace(plan=root / "unused.json", sha256="unused", index=0,
                    deadline=time.monotonic() + 100, wall_deadline=time.time() + 100)
                real_write = module.write
                def failed_progress(path, value):
                    if value.get("phase") in ("failure_retention", "cleanup"):
                        raise OSError("synthetic progress fsync failure")
                    real_write(path, value)
                stack.enter_context(patch.object(module, "read_plan", return_value=plan))
                stack.enter_context(patch.object(module, "check_identity"))
                stack.enter_context(patch.object(module, "load_fixed", return_value=(loaded, plan["frozen"])))
                stack.enter_context(patch.object(module, "write", side_effect=failed_progress))
                stack.enter_context(patch.object(module.signal, "signal"))
                stack.enter_context(patch("experiments.headless.process_deadline.DeadlineVcmiLauncher", return_value=launcher))
                if module is neural:
                    stack.enter_context(patch("experiments.headless.env.HeadlessSelfPlayEnv", return_value=env))
                    stack.enter_context(patch("experiments.learner.contextual_rollout.ContextualCollector", FailingCollector))
                    retained = stack.enter_context(patch("validation.probes.validate_hero_policy.retain_failed_evidence"))
                else:
                    stack.enter_context(patch.object(module, "native_opponent", return_value="native-opponent"))
                    stack.enter_context(patch("experiments.headless.builtin_env.BuiltinHeadlessEnv", return_value=env))
                    stack.enter_context(patch("experiments.learner.builtin_rollout.BuiltinCollector", FailingCollector))
                    retained = stack.enter_context(patch("experiments.learner.builtin_rollout.retain_failed_builtin_evidence"))
                attempt = module.worker(args)
                retained.assert_called_once()
                env.close.assert_called_once()
                launcher.close_owned_groups.assert_called_once()
                self.assertEqual(attempt["status"], "failed")
                self.assertEqual(attempt["rewards"], {"red": None, "blue": None})
                self.assertEqual(len(attempt["progress_write_errors"]), 2)
                self.assertIn("environment_cleanup_error", attempt)
                self.assertTrue(attempt["cleanup_verified"])
                self.assertIn("cleanup", attempt["phase_wall_seconds"])
                saved = json.loads((root / "attempt-01.json").read_text())
                self.assertEqual(saved, attempt)
                self.assertEqual(saved["phase"], "finished")

    def test_v2_supervisor_keeps_same_absolute_deadlines_and_durable_rows(self):
        """Real short-lived subprocesses only; no VCMI, model or replay calls."""
        from experiments.headless import process_deadline as guard
        import sys
        real_spawn = guard.guarded_popen
        for module in (neural, builtin):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                checkpoint = root / "unused.pt"
                checkpoint.write_bytes(b"synthetic-not-a-model")
                cfg = {**module.config(), "shared_wall_seconds": 1.2,
                    "cleanup_reserve_seconds": .3, "hard_stop_reserve_seconds": .1}
                plan = {"schema": module.V2_SCHEMA + "-plan", "configuration": cfg,
                    "schedule": module.schedule(cfg), "execution_output": str(root / "execution"),
                    "frozen": {}, "checkpoint": {"path": str(checkpoint)}, "wall_scope": "test only"}
                received = []
                def stalled(command, **kwargs):
                    received.append((float(command[command.index("--deadline") + 1]), kwargs["deadline_monotonic"]))
                    return real_spawn([sys.executable, "-c", "import time;time.sleep(30)"], **kwargs)
                args = SimpleNamespace(plan=root / "plan.json", sha256="a" * 64)
                with patch.object(module, "read_plan", return_value=plan), patch.object(module, "check_identity"), \
                     patch.object(guard, "guarded_popen", side_effect=stalled):
                    result = module.execute(args)
                self.assertEqual(result["schema"], module.V2_SCHEMA + "-result")
                self.assertEqual(result["stop_reason"], "shared_deadline")
                self.assertEqual(len(received), 1)
                self.assertEqual(received[0][0], received[0][1])
                self.assertEqual(len(result["attempts"]), 8)
                self.assertTrue(all(row["status"] == "not_started" for row in result["attempts"][1:]))
                self.assertTrue(all(row["rewards"] == {"red": None, "blue": None} for row in result["attempts"]))
                self.assertLess(result["execution_wall_seconds"], 2)
                with self.assertRaises(FileExistsError), patch.object(module, "read_plan", return_value=plan), \
                     patch.object(module, "check_identity"):
                    module.execute(args)


if __name__ == "__main__":
    unittest.main()
