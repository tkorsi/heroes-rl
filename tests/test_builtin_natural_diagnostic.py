"""Exercise diagnostic admission and orchestration with no native execution."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.headless import builtin_env, builtin_replay, process_deadline
from experiments.learner import builtin_rollout
from validation.probes import validate_builtin_natural as probe
from validation.probes import validate_hero_policy
from validation.probes.create_boundary_fixture import create_fixture


class NaturalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "frozen.pt"
        self.checkpoint.write_bytes(b"Synthetic checkpoint; never loaded as a model")
        create_fixture(self.root / "fixture", kind="natural-townless")
        self.game_map = self.root / "fixture/natural-townless.vmap"
        self.fixture = probe.natural_fixture_identity(self.game_map)
        self.source = {"synthetic_source": "unchanged"}
        self.runtime = {"synthetic_runtime": "unchanged"}
        self.frozen = {"synthetic_weights": "unchanged"}
        self.calls = 0

    def plan(self, output):
        return {"schema": probe.SCHEMA, "schedule": deepcopy(probe.SCHEDULE),
            "output": str(output), "fixture": deepcopy(self.fixture),
            "checkpoint": probe.binding(self.checkpoint), "frozen": self.frozen,
            "implementation": self.source, "runtime": self.runtime,
            "binary": str(self.root / "bin/vcmiclient"),
            "replay_binary": str(self.root / "bin/vcmireplay"), "data": str(self.root / "data"),
            "case_seconds": 120, "ordinary_choices": 256, "hard_choices": 264, "max_days": 12}

    @contextmanager
    def identities(self):
        with patch.object(probe, "implementation_identity", return_value=self.source), \
             patch.object(probe, "runtime_identity", return_value=self.runtime):
            yield

    def run_worker(self, *, seat=0, status="completed", reason="natural_terminal", days=(1,),
                   replay_verified=True, audit_error=None, cleanup_error=None):
        self.calls += 1
        output = self.root / f"worker-{self.calls}"
        output.mkdir()
        plan = self.plan(output)
        plan_path = output / "plan.json"
        probe.write(plan_path, plan)
        args = SimpleNamespace(plan=plan_path, sha256=probe.binding(plan_path)["sha256"],
            index=seat, deadline=500.0, wall_deadline=1500.0)
        folder = output / f"episode-{seat + 1:06d}"
        folder.mkdir()
        manifest = {"status": status, "termination_reason": reason,
            "outcomes_by_player": {"red": "loss", "blue": "win"},
            "rewards_by_player": {"red": -1, "blue": 1}}
        probe.write(folder / "manifest.json", manifest)
        original_manifest = (folder / "manifest.json").read_bytes()
        env = SimpleNamespace(episode_dir=folder, close=Mock(side_effect=cleanup_error))
        launcher = SimpleNamespace(close_owned_groups=Mock())
        loaded = SimpleNamespace(model=object(), checkpoint="synthetic-frozen-identity")
        accounting = {"authoritative_inputs": 3, "terminal_unapplied": [], "final_terminal_input": {}}
        sampled, instances, run_arguments, timers = [], [], [], []

        class SyntheticCollector:
            def __init__(self, *args, **kwargs):
                self.session = None
                self.arguments = (args, kwargs)
                instances.append(self)

            def act(self, request):
                sampled.append(request.record["day"])
                return {"synthetic_sample": len(sampled)}

            def run_episode(self, environment, **kwargs):
                run_arguments.append((environment, kwargs))
                actions = [self.act(SimpleNamespace(record={"day": day})) for day in days]
                return SimpleNamespace(manifest=manifest,
                    decisions=[SimpleNamespace(decision_seq=i) for i in range(len(actions))],
                    history=[SimpleNamespace(day=day) for day in days],
                    to_dict=lambda: {"synthetic_only": True, "sampled_actions": actions})

        @contextmanager
        def timer(deadline, wall_deadline, reserve=0):
            timers.append((deadline, wall_deadline, reserve))
            yield

        def replay(*args, **kwargs):
            target = folder / "builtin-verification/builtin-replay-verification.json"
            target.parent.mkdir()
            value = {"verified": replay_verified, "wire": {"request_accounting": accounting}}
            probe.write(target, value)
            return value

        with ExitStack() as stack:
            stack.enter_context(self.identities())
            stack.enter_context(patch("torch.set_num_threads"))
            stack.enter_context(patch.object(probe, "until", side_effect=timer))
            stack.enter_context(patch.object(probe, "remaining", return_value=50.0))
            stack.enter_context(patch.object(probe, "load_fixed", return_value=(loaded, self.frozen)))
            stack.enter_context(patch.object(probe, "native_opponent", return_value={"synthetic_native": seat ^ 1}))
            environment = stack.enter_context(patch.object(builtin_env, "BuiltinHeadlessEnv", return_value=env))
            native = stack.enter_context(patch.object(process_deadline, "DeadlineVcmiLauncher", return_value=launcher))
            stack.enter_context(patch.object(builtin_rollout, "BuiltinCollector", SyntheticCollector))
            audit = stack.enter_context(patch.object(builtin_rollout, "audit_persisted_builtin_policy",
                return_value={"synthetic_exact_reconstruction": True}, side_effect=audit_error))
            retained = stack.enter_context(patch.object(builtin_rollout, "retain_failed_builtin_evidence"))
            verifier = stack.enter_context(patch.object(builtin_replay, "verify_episode", side_effect=replay))
            stack.enter_context(patch.object(validate_hero_policy, "frozen_fingerprints", return_value=self.frozen))
            probe.worker(args)
        result = json.loads((output / f"case-{seat + 1}.json").read_text())
        self.assertEqual((folder / "manifest.json").read_bytes(), original_manifest)
        return SimpleNamespace(result=result, folder=folder, manifest=manifest, env=env, launcher=launcher,
            sampled=sampled, instances=instances, runs=run_arguments, timers=timers,
            environment=environment, native=native, audit=audit, retained=retained, verifier=verifier)

    def test_both_colors_admit_only_reconstructed_and_replayed_natural_results(self):
        for seat in (0, 1):
            with self.subTest(seat=seat):
                run = self.run_worker(seat=seat, days=(1, 12))
                self.assertEqual(run.result["status"], "passed")
                self.assertEqual(run.result["outcomes"], run.manifest["outcomes_by_player"])
                self.assertTrue(run.result["replay_verified"])
                self.assertFalse(run.result["ordinary_cohort_member"])
                self.assertEqual(run.result["training_samples"], 0)
                self.assertEqual(run.result["optimizer_steps"], 0)
                self.assertEqual(run.sampled, [1, 12])
                self.assertEqual(run.runs[0][1], {"episode": seat + 1, "seed": 121902, "max_requests": 256})
                self.assertEqual(run.instances[0].arguments[1]["policy_seed"], 121903)
                self.assertEqual(run.instances[0].arguments[1]["learner_player"], seat)
                run.audit.assert_called_once()
                run.verifier.assert_called_once()
                run.retained.assert_not_called()
                run.env.close.assert_called_once()
                run.launcher.close_owned_groups.assert_called_once_with(timeout=1)

    def test_non_natural_end_is_retained_unscored_without_replay_admission(self):
        for status, reason in (("truncated", "collection_window"), ("failed", "protocol_error"),
                               ("completed", "watchdog_timeout")):
            with self.subTest(status=status, reason=reason):
                run = self.run_worker(status=status, reason=reason)
                self.assertEqual(run.result["status"], "failed")
                self.assertEqual(run.result["outcomes"], {"red": None, "blue": None})
                self.assertEqual(run.result["rewards"], {"red": None, "blue": None})
                self.assertTrue((run.folder / "builtin-rollout.json").exists())
                run.retained.assert_called_once()
                run.audit.assert_not_called()
                run.verifier.assert_not_called()

    def test_failed_reconstruction_or_replay_cannot_inherit_native_scores(self):
        for options in ({"audit_error": ValueError("reconstruction differs")}, {"replay_verified": False}):
            with self.subTest(options=options):
                run = self.run_worker(**options)
                self.assertEqual(run.result["status"], "failed")
                self.assertEqual(run.result["outcomes"], {"red": None, "blue": None})
                self.assertEqual(run.result["rewards"], {"red": None, "blue": None})
                run.retained.assert_called_once()
                if "audit_error" in options:
                    run.verifier.assert_not_called()

    def test_cleanup_failure_revokes_otherwise_verified_result(self):
        run = self.run_worker(cleanup_error=RuntimeError("synthetic cleanup failure"))
        self.assertEqual(run.result["status"], "failed")
        self.assertEqual(run.result["outcomes"], {"red": None, "blue": None})
        self.assertEqual(run.result["rewards"], {"red": None, "blue": None})
        self.assertIn("synthetic cleanup failure", run.result["environment_cleanup_error"])
        run.launcher.close_owned_groups.assert_called_once_with(timeout=1)

    def test_hard_choice_and_day_bounds_stop_before_another_model_action(self):
        for days, expected_count in (((1,) * 265, 264), ((13,), 0)):
            with self.subTest(count=len(days), day=days[0]):
                run = self.run_worker(days=days)
                self.assertEqual(len(run.sampled), expected_count)
                self.assertEqual(run.result["status"], "failed")
                self.assertIn("choice/day budget exhausted", run.result["error"])
                self.assertEqual(run.result["rewards"], {"red": None, "blue": None})
                run.retained.assert_called_once()
                run.verifier.assert_not_called()

    def test_worker_setup_replay_and_cleanup_share_absolute_deadlines(self):
        run = self.run_worker()
        self.assertEqual(run.timers, [(500.0, 1500.0, 10), (500.0, 1500.0, 1), (500.0, 1500.0, 1)])
        self.assertEqual(run.native.call_args.kwargs["deadline_monotonic"], 500.0)
        self.assertEqual(run.native.call_args.kwargs["deadline_wall"], 1500.0)
        self.assertEqual(run.verifier.call_args.kwargs["timeout"], 40.0)
        limits = run.environment.call_args.kwargs["limits"]
        self.assertEqual((limits.max_actions, limits.max_days), (264, 12))

    def test_bound_scope_fixture_checkpoint_source_and_runtime_reject_drift(self):
        plan = self.plan(self.root / "unused")
        with self.identities():
            probe.check(plan)
            for mutate in (lambda p: p["schedule"][0].update(engine_seed=999),
                           lambda p: p.update(case_seconds=121), lambda p: p.update(hard_choices=265),
                           lambda p: p.update(ordinary_choices=257), lambda p: p.update(max_days=13),
                           lambda p: p["fixture"]["manifest"].update(sha256="0" * 64),
                           lambda p: p["checkpoint"].update(sha256="0" * 64),
                           lambda p: p.update(implementation={"changed": True}),
                           lambda p: p.update(runtime={"changed": True})):
                changed = deepcopy(plan)
                mutate(changed)
                with self.subTest(plan=changed), self.assertRaises(ValueError):
                    probe.check(changed)
        self.assertFalse(self.fixture["ordinary_cohort_member"])
        self.assertTrue(self.fixture["natural_outcome_validation"])
        self.assertIn("no historical fixture action script", self.fixture["probe_plan"])

    def run_supervisor(self, *, first_status="passed", first_returncode=0, cleanup_error=None):
        output = self.root / "supervisor"
        args = SimpleNamespace(output=output, checkpoint=self.checkpoint, map=self.game_map,
            binary=self.root / "bin/vcmiclient", replay_binary=self.root / "bin/vcmireplay", data=self.root / "data")
        launches = []

        def spawn(command, **kwargs):
            index = int(command[command.index("--index") + 1])
            plan = json.loads((output / "plan.json").read_text())
            row = plan["schedule"][index]
            status = first_status if index == 0 else "passed"
            record = {**row, "finished": True, "status": status,
                "outcomes": {"red": "loss", "blue": "win"} if status == "passed" else dict.fromkeys(probe.COLORS),
                "rewards": {"red": -1, "blue": 1} if status == "passed" else dict.fromkeys(probe.COLORS)}
            probe.write(output / f"case-{row['episode']}.json", record)
            process = SimpleNamespace(returncode=first_returncode if index == 0 else 0, wait=Mock())
            launches.append((command, kwargs, process))
            return process

        with self.identities(), patch("torch.set_num_threads"), \
             patch.object(probe, "load_fixed", return_value=(None, self.frozen)), \
             patch.object(probe.time, "monotonic", return_value=100.0), \
             patch.object(probe.time, "time", return_value=200.0), \
             patch.object(process_deadline, "guarded_popen", side_effect=spawn), \
             patch.object(process_deadline, "close_owned_process") as close, \
             patch.object(process_deadline, "cleanup_from_evidence", return_value=[], side_effect=cleanup_error):
            result = probe.execute(args)
        return result, launches, close

    def test_supervisor_retains_failure_and_does_not_launch_second_case(self):
        result, launches, close = self.run_supervisor(first_status="failed")
        self.assertEqual(len(launches), 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["status"] for r in result["attempts"]], ["failed", "not_started"])
        self.assertEqual(result["attempts"][0]["outcomes"], dict.fromkeys(probe.COLORS))
        self.assertEqual(result["attempts"][0]["rewards"], dict.fromkeys(probe.COLORS))
        self.assertTrue((self.root / "supervisor/case-1.json").is_file())
        close.assert_called_once()

    def test_supervisor_crash_or_cleanup_failure_cannot_keep_worker_scores(self):
        for mode in ("crash", "cleanup"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as folder:
                original = self.root
                self.root = Path(folder)
                try:
                    result, launches, _ = self.run_supervisor(
                        first_returncode=3 if mode == "crash" else 0,
                        cleanup_error=ValueError("group still active") if mode == "cleanup" else None)
                finally:
                    self.root = original
                self.assertEqual(len(launches), 1)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["attempts"][0]["rewards"], dict.fromkeys(probe.COLORS))
                self.assertEqual(result["attempts"][1]["status"], "not_started")

    def test_supervisor_passes_exact_two_seats_with_guarded_fixed_case_budget(self):
        result, launches, close = self.run_supervisor()
        self.assertEqual(result["status"], "passed")
        self.assertEqual([r["learner_player"] for r in result["attempts"]], [0, 1])
        self.assertFalse(result["ordinary_cohort_member"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(close.call_count, 2)
        for command, kwargs, process in launches:
            self.assertEqual(kwargs["deadline_monotonic"], 220.0)
            self.assertEqual(kwargs["deadline_wall"], 320.0)
            self.assertEqual(float(command[command.index("--deadline") + 1]), 220.0)
            self.assertEqual(float(command[command.index("--wall-deadline") + 1]), 320.0)
            process.wait.assert_called_once_with(timeout=118.0)
        self.assertEqual(result["checkpoint_after"], probe.binding(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
