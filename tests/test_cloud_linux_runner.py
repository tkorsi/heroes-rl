"""Synthetic lifecycle checks; harmless guarded children, no VCMI games/training."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validation.probes import validate_cloud_linux as runner


def synthetic_evidence(output, result):
    """Write explicitly synthetic bytes for admission/lifecycle tests only."""
    from tests.test_builtin_replay import NativeReportTests
    from experiments.headless import builtin_replay
    fixture = NativeReportTests()
    fixture.setUp()
    folder = Path(output) / f"episode-{result['episode']:06d}"
    folder.mkdir(exist_ok=True)
    color, seat = runner.COLORS[result["learner_player"]], result["learner_player"]
    controller = builtin_replay.controllers(seat)
    fixture.wire["controllers"] = controller
    fixture.barrier.update(controllers=controller, explorer_terminal_callbacks=[seat],
                           outcomes=[result["outcomes"][key] for key in runner.COLORS])
    fixture.native["summary"].update(controllers=controller, terminal_barrier=copy.deepcopy(fixture.barrier))
    accounting = {"authoritative_inputs": 4, "terminal_unapplied": [], "final_terminal_input": None}
    fixture.wire["request_accounting"] = accounting
    fixture.native["summary"]["request_accounting"] = copy.deepcopy(accounting)
    if result["status"] == "verified_cutoff":
        fixture.native["summary"].update(captured_prefix_only=True, terminal_barrier=None,
            boundary={"kind": "cutoff", "barrier_id": None, "client_packets": 10})
    manifest = {"episode": str(result["episode"]), "learner_player": seat, "status": result["episode_status"],
        "termination_reason": result["termination_reason"], "outcomes_by_player": result["outcomes"],
        "rewards_by_player": result["rewards"]}
    for name in ("builtin-rollout.json", "behavior.json", "logs/process.log", "engine/experiment-metadata.json",
                 "engine/replay.jsonl", "engine/contextual-replay.jsonl", f"agents/{color}/bridge.jsonl",
                 f"agents/{color}/history-v7.jsonl", f"agents/{color}/live-history-v7.jsonl"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.write(path, {"synthetic": True, "gameplay": False})
    runner.write(folder / "manifest.json", manifest)
    runner.write(folder / "builtin-finalization.json", {"terminal_barrier": {"client": fixture.barrier}})
    runner.write(folder / "policy-reconstruction.json", result["reconstruction"])
    (folder / "builtin-verification").mkdir(exist_ok=True)
    runner.write(folder / "builtin-verification/native-report.json", fixture.native)
    replay = {"schema": builtin_replay.REPORT_SCHEMA, "kind": "builtin-captured-command-replay", "verified": True,
        "training_performed": False, "training_samples": 0, "optimization_eligible": False,
        "opponent_policy_reexecuted": False, "wire": fixture.wire, "native": fixture.native,
        "inputs": {"manifest": {k: runner.binding(folder / "manifest.json")[k] for k in ("sha256", "bytes")}}}
    runner.write(folder / "builtin-verification/builtin-replay-verification.json", replay)
    result.update(episode_directory=str(folder), manifest=runner.binding(folder / "manifest.json"),
        replay=runner.binding(folder / "builtin-verification/builtin-replay-verification.json"),
        behavior=runner.binding(folder / "behavior.json"), request_accounting=accounting,
        evidence_bindings=runner.capture_bindings(folder, seat))
    return result


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="heroes-linux-runner-test-")
        self.addCleanup(temporary.cleanup)
        # Completion receipts bind canonical paths, including on macOS TMPDIR.
        self.root = Path(temporary.name).resolve()
        self.sha = "a" * 64
        self.plan = {"schedule": copy.deepcopy(runner.planning.SCHEDULE),
            "configuration": {"shared_wall_seconds": 10, "cleanup_reserve_seconds": .3,
                              "hard_stop_reserve_seconds": .1, "torch_threads": 1},
            "execution_output": str(self.root / "execution")}
        for row in self.plan["schedule"]:
            row["case_wall_seconds"] = 4
        self.args = SimpleNamespace(plan=self.root / "plan.json", sha256=self.sha)

    def result(self, row, status="verified_natural"):
        value = {**runner.empty_attempt(row), "status": status, "phase": "finished", "plan_sha256": self.sha,
            "cleanup_verified": True, "identity_verified": True, "weights_unchanged": True,
            "replay_verified": True,
            "reconstruction": {"passed": True, "private_memory_reset": True, "weights_unchanged": True,
                               "maximum_absolute_difference": 0}}
        if status == "verified_natural":
            value.update(episode_status="completed", termination_reason="natural_terminal",
                         outcomes={"red": "loss", "blue": "win"}, rewards={"red": -1, "blue": 1})
        elif status == "verified_cutoff":
            value.update(episode_status="truncated", termination_reason="collection_window")
        return value

    def execute(self, spawn, check=None):
        with patch.object(runner.planning, "read_plan", return_value=self.plan), \
             patch.object(runner.planning, "check_identity", side_effect=check), \
             patch.object(runner.guard, "guarded_popen", side_effect=spawn):
            return runner.execute(self.args)

    def child(self, command, kwargs, result, suffix=""):
        index = int(command[command.index("--index") + 1])
        code = ("from pathlib import Path; import json,os,signal,time; "
                "from validation.probes import validate_cloud_linux as r; "
                "from tests.test_cloud_linux_runner import synthetic_evidence; "
                f"result=synthetic_evidence(Path({self.plan['execution_output']!r}), json.loads({json.dumps(result)!r})); "
                f"r.publish_completion(Path({self.plan['execution_output']!r}), result, "
                f"plan_sha256={self.sha!r}, index={index}); " + suffix)
        return self.real_spawn([sys.executable, "-c", code], **kwargs)

    def test_verified_cutoff_and_natural_rows_require_durable_receipts_and_keep_fixture_scope(self):
        self.real_spawn = runner.guard.guarded_popen
        seen = []
        def spawn(command, **kwargs):
            index = int(command[command.index("--index") + 1])
            seen.append(index)
            row = self.plan["schedule"][index]
            return self.child(command, kwargs, self.result(row, "verified_cutoff" if index == 0 else "verified_natural"))
        report = self.execute(spawn)
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertEqual(report["status"], "finished")
        self.assertEqual(report["attempts"][0]["rewards"], {"red": None, "blue": None})
        self.assertEqual([r["fixture"] for r in report["attempts"]], [True, True, False, False])
        with patch.object(runner.planning, "read_plan", return_value=self.plan):
            self.assertEqual(runner.read_report(self.args.plan, self.sha), report)
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["optimizer_steps"], 0)
        with patch.object(runner.planning, "read_plan", return_value=self.plan), self.assertRaises(FileExistsError):
            runner.execute(self.args)

    def test_worker_death_before_receipt_never_admits_persisted_candidate_score(self):
        self.real_spawn = runner.guard.guarded_popen
        def spawn(command, **kwargs):
            result = self.result(self.plan["schedule"][0])
            path, _ = runner.paths(self.plan["execution_output"], 1)
            # Reproduce death after a final-looking row exists but before its
            # final verification/receipt. This is a harmless Python process.
            code = ("from pathlib import Path; import os,signal; "
                    f"Path({str(path)!r}).write_text({json.dumps(result)!r}); "
                    "os.kill(os.getpid(), signal.SIGKILL)")
            return self.real_spawn([sys.executable, "-c", code], **kwargs)
        report = self.execute(spawn)
        row = report["attempts"][0]
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["rewards"], {"red": None, "blue": None})
        self.assertEqual(row["last_worker_result"]["status"], "verified_natural")
        self.assertTrue(all(x["status"] == "not_started" for x in report["attempts"][1:]))
        self.assertFalse((self.root / "execution/completion-01.json").exists())

    def test_nonzero_exit_after_receipt_cannot_admit_an_outcome(self):
        self.real_spawn = runner.guard.guarded_popen
        def spawn(command, **kwargs):
            return self.child(command, kwargs, self.result(self.plan["schedule"][0]), "os._exit(7)")
        report = self.execute(spawn)
        self.assertEqual(report["attempts"][0]["status"], "incomplete")
        self.assertEqual(report["attempts"][0]["outcomes"], {"red": None, "blue": None})
        self.assertTrue((self.root / "execution/completion-01.json").exists())

    def test_shared_deadline_retains_all_rows_and_stops_harmless_process(self):
        self.real_spawn = runner.guard.guarded_popen
        self.plan["configuration"].update(shared_wall_seconds=1.4, cleanup_reserve_seconds=.4,
                                           hard_stop_reserve_seconds=.2)
        processes = []
        def spawn(command, **kwargs):
            process = self.real_spawn([sys.executable, "-c", "import time;time.sleep(30)"], **kwargs)
            processes.append(process)
            return process
        started = time.monotonic()
        report = self.execute(spawn)
        self.assertEqual(len(processes), 1)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIsNotNone(processes[0].poll())
        self.assertEqual(len(report["attempts"]), 4)
        self.assertEqual(report["stop_reason"], "shared_deadline")
        self.assertTrue(all(row["rewards"] == {"red": None, "blue": None} for row in report["attempts"]))
        self.assertTrue(all(row["status"] == "not_started" for row in report["attempts"][1:]))

    def test_crashed_worker_releases_independently_guarded_child(self):
        self.real_spawn = runner.guard.guarded_popen
        evidence = self.root / "execution/native-deadline-01.jsonl"
        def spawn(command, **kwargs):
            code = ("from pathlib import Path; import os,sys,signal,time; "
                    "from experiments.headless.process_deadline import guarded_popen; "
                    f"p=guarded_popen([sys.executable,'-c','import time;time.sleep(30)'], "
                    f"deadline_monotonic=time.monotonic()+20,deadline_wall=time.time()+20,evidence_path=Path({str(evidence)!r})); "
                    "time.sleep(.2); os.kill(os.getpid(),signal.SIGKILL)")
            return self.real_spawn([sys.executable, "-c", code], **kwargs)
        report = self.execute(spawn)
        row = report["attempts"][0]
        self.assertEqual(row["status"], "incomplete")
        self.assertNotIn("supervisor_cleanup_errors", row)
        self.assertTrue(row["supervisor_native_groups_closed"])
        for group in row["supervisor_native_groups_closed"]:
            self.assertFalse(runner.guard.group_live_members(group))

    def test_post_campaign_identity_failure_withdraws_admission_and_keeps_worker_files(self):
        self.real_spawn = runner.guard.guarded_popen
        def spawn(command, **kwargs):
            index = int(command[command.index("--index") + 1])
            return self.child(command, kwargs, self.result(self.plan["schedule"][index]))
        checks = iter((None, ValueError("runtime changed")))
        def check(*args, **kwargs):
            value = next(checks)
            if value:
                raise value
        report = self.execute(spawn, check)
        self.assertEqual(report["stop_reason"], "final_identity_failed")
        self.assertTrue(all(row["status"] == "incomplete" and row["rewards"] == {"red": None, "blue": None}
                            for row in report["attempts"]))
        self.assertTrue((self.root / "execution/completion-04.json").exists())
        self.assertFalse((self.root / "execution/completion.json").exists())

    def test_worker_receipt_binds_plan_result_and_exact_schedule_types(self):
        output = self.root / "execution"
        output.mkdir()
        row = self.plan["schedule"][0]
        result = synthetic_evidence(output, self.result(row))
        runner.publish_completion(output, result, plan_sha256=self.sha, index=0)
        self.assertEqual(runner.read_completion(output, row, plan_sha256=self.sha, index=0, returncode=0), result)
        receipt_path = output / "completion-01.json"
        receipt = json.loads(receipt_path.read_text())
        for update in ({"plan_sha256": "b" * 64}, {"index": False}, {"cleanup_verified": 1},
                       {"result": {**receipt["result"], "sha256": "0" * 64}}):
            runner.write(receipt_path, {**receipt, **update})
            with self.subTest(update=update), self.assertRaises(ValueError):
                runner.read_completion(output, row, plan_sha256=self.sha, index=0, returncode=0)
        runner.write(receipt_path, receipt)
        result["learner_player"] = False
        runner.write(output / "attempt-01.json", result)
        receipt["result"] = runner.binding(output / "attempt-01.json")
        runner.write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "schedule"):
            runner.read_completion(output, row, plan_sha256=self.sha, index=0, returncode=0)

    def test_missing_final_verification_and_scored_cutoffs_are_rejected(self):
        row = self.plan["schedule"][0]
        for field in ("cleanup_verified", "identity_verified", "weights_unchanged"):
            result = self.result(row)
            result[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.check_result(result, row)
        for change in ({"private_memory_reset": False}, {"passed": False}, {"maximum_absolute_difference": 1e-9}):
            result = self.result(row)
            result["reconstruction"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.check_result(result, row)
        result = self.result(row, "verified_cutoff")
        result["rewards"] = {"red": 0, "blue": 0}
        with self.assertRaisesRegex(ValueError, "Unscored"):
            runner.check_result(result, row)
        result = self.result(row)
        result["fixture"] = False
        with self.assertRaisesRegex(ValueError, "Fixture"):
            runner.check_result(result, row)

    def test_completed_capture_deletion_mutation_and_unchecked_requests_are_rejected(self):
        output = self.root / "execution"
        output.mkdir()
        row = self.plan["schedule"][0]
        result = synthetic_evidence(output, self.result(row))
        runner.publish_completion(output, result, plan_sha256=self.sha, index=0)
        folder = Path(result["episode_directory"])
        for name in ("agents/red/history-v7.jsonl", "engine/contextual-replay.jsonl",
                     "builtin-verification/native-report.json", "builtin-verification/builtin-replay-verification.json"):
            path = folder / name
            saved = path.read_bytes()
            path.unlink()
            with self.subTest(deleted=name), self.assertRaises(ValueError):
                runner.read_completion(output, row, plan_sha256=self.sha, index=0, returncode=0)
            path.write_bytes(saved + b"\n")
            with self.subTest(changed=name), self.assertRaises(ValueError):
                runner.read_completion(output, row, plan_sha256=self.sha, index=0, returncode=0)
            path.write_bytes(saved)
        replay_path = Path(result["replay"]["path"])
        replay = json.loads(replay_path.read_text())
        replay["native"]["summary"]["unchecked"]["requests"] = 1
        runner.write(replay_path, replay)
        result["replay"] = runner.binding(replay_path)
        result["evidence_bindings"] = runner.capture_bindings(folder, row["learner_player"])
        # Even rehashing all changed files cannot make incomplete native
        # coverage a valid result under the saved report's semantics.
        with self.assertRaisesRegex(ValueError, "coverage"):
            runner.verify_saved_evidence(result, output)

    def test_durable_writer_refuses_replacement_and_expired_publication(self):
        output = self.root / "execution"
        output.mkdir()
        result = self.result(self.plan["schedule"][0])
        with self.assertRaises(runner.BudgetExpired):
            runner.publish_completion(output, result, plan_sha256=self.sha, index=0,
                deadline=time.monotonic()-1, wall_deadline=time.time()-1)
        self.assertEqual(list(output.iterdir()), [])
        runner.publish_completion(output, result, plan_sha256=self.sha, index=0)
        original = (output / "completion-01.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.publish_completion(output, result, plan_sha256=self.sha, index=0)
        self.assertEqual((output / "completion-01.json").read_bytes(), original)
        self.assertFalse(any(path.name.startswith(".") for path in output.iterdir()))

    def test_completed_campaign_report_corruption_or_missing_receipt_rejects(self):
        self.real_spawn = runner.guard.guarded_popen
        def spawn(command, **kwargs):
            index = int(command[command.index("--index") + 1])
            return self.child(command, kwargs, self.result(self.plan["schedule"][index]))
        self.execute(spawn)
        with patch.object(runner.planning, "read_plan", return_value=self.plan):
            path = self.root / "execution/report.json"
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                runner.read_report(self.args.plan, self.sha)
            path.write_bytes(original)
            receipt_path = self.root / "execution/completion.json"
            receipt = json.loads(receipt_path.read_text())
            deadline = json.loads(original)["deadline_unix"]
            runner.write(receipt_path, {**receipt, "finished_unix": deadline + 1})
            with self.assertRaisesRegex(ValueError, "shared clock"):
                runner.read_report(self.args.plan, self.sha)
            runner.write(receipt_path, receipt)
            (self.root / "execution/completion.json").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.read_report(self.args.plan, self.sha)

    def test_final_report_fsync_crossing_deadline_cannot_publish_campaign_admission(self):
        self.real_spawn = runner.guard.guarded_popen
        def spawn(command, **kwargs):
            index = int(command[command.index("--index") + 1])
            return self.child(command, kwargs, self.result(self.plan["schedule"][index]))
        offset = [0]
        real_mono, real_wall, real_write = time.monotonic, time.time, runner.write
        def write(path, value, **kwargs):
            result = real_write(path, value, **kwargs)
            if Path(path).name == "report.json" and value.get("status") == "finished":
                offset[0] = 20
            return result
        with patch.object(runner.time, "monotonic", side_effect=lambda: real_mono()+offset[0]), \
             patch.object(runner.time, "time", side_effect=lambda: real_wall()+offset[0]), \
             patch.object(runner, "write", side_effect=write):
            report = self.execute(spawn)
        self.assertTrue(report["deadline_exceeded"])
        self.assertEqual(report["stop_reason"], "shared_deadline")
        self.assertTrue(all(row["rewards"] == {"red": None, "blue": None} for row in report["attempts"]))
        self.assertFalse((self.root / "execution/completion.json").exists())

    def test_own_history_counts_transient_sighting_without_inventing_hidden_fate(self):
        def frame(seq, kind, handles=(), player=0):
            return SimpleNamespace(player=player, episode="1", kind=kind, seq=seq,
                observation=SimpleNamespace(objects=[SimpleNamespace(handle=h, owner="opponent") for h in handles]))
        frames = [frame(0, "decision"), frame(1, "view", ("seen",)), frame(2, "event"),
                  frame(3, "view"), frame(4, "decision")]
        result = runner.own_history_coverage(frames, player=0, episode=1)
        self.assertEqual(result["opponent_handle_sightings_absent_at_both_surrounding_decisions"], 1)
        self.assertEqual(result["events_after_first_decision"], 1)
        self.assertTrue(result["transient_sighting_observed"])
        self.assertFalse(runner.own_history_coverage(frames[:1], player=0, episode=1)["transient_sighting_observed"])
        with self.assertRaisesRegex(ValueError, "mixed"):
            runner.own_history_coverage([*frames, frame(5, "event", player=1)], player=0, episode=1)

    def test_existing_deadline_alarm_stops_blocking_work_without_new_budget(self):
        deadline, wall = time.monotonic()+.04, time.time()+.04
        with self.assertRaises(runner.BudgetExpired):
            with runner.until(deadline, wall):
                time.sleep(1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        with self.assertRaises(runner.BudgetExpired):
            with runner.until(deadline, wall):
                self.fail("Expired work must not start")


class WorkerLifecycleTests(unittest.TestCase):
    # These additional tests mock the model/game collaborators. They verify
    # ordering and evidence admission, never native behavior or game outcomes.
    setUp = RunnerTests.setUp
    result = RunnerTests.result
    def setup_worker(self, stack, *, fail_post_identity=False, collection_error=False):
        from experiments.headless import builtin_env, builtin_replay
        from experiments.learner import builtin_rollout, builtin_behavior
        from validation.probes import validate_hero_policy
        output = Path(self.plan["execution_output"])
        output.mkdir()
        self.plan.update(binary="unused", data="unused", checkpoint={"path": "unused"}, frozen={},
            maps={"fixture": {"path": "unused"}}, native_opponents=[{}, {}], replay_binary={"path": "unused"})
        row = self.plan["schedule"][0]
        calls = []
        fixture_manifest = {"episode": "1", "learner_player": 0,
            "status": "completed", "termination_reason": "natural_terminal",
            "outcomes_by_player": {"red": "loss", "blue": "win"},
            "rewards_by_player": {"red": -1, "blue": 1}}

        class Environment:
            def __init__(self, *args, **kwargs):
                self.episode_dir = output / "episode-000001"
                self.episode_dir.mkdir()
                self.process, self.result = object() if collection_error else None, None

            def _note_problem(self, reason, detail):
                calls.append("failure_noted")

            def _finish(self, unused):
                calls.append("failure_retained")
                runner.write(self.episode_dir / "synthetic-failed-evidence.json", {"incomplete": True})
                return SimpleNamespace(status="failed", outcomes={"red": None, "blue": None},
                    rewards={"red": None, "blue": None}, termination_reason="protocol_error")

            def close(self):
                calls.append("environment_closed")

        class Collector:
            def __init__(self, *args, **kwargs):
                self.session = SimpleNamespace(history=SimpleNamespace(player=0, episode="1", frames=[]), _choices={})
                self.sessions = (self.session,)
                self.opponent_identity = {}

            def run_episode(self, env, **kwargs):
                calls.append("collection")
                if collection_error:
                    raise RuntimeError("synthetic collection failure")
                runner.write(env.episode_dir / "manifest.json", fixture_manifest)
                return SimpleNamespace(to_dict=lambda: {}, manifest=fixture_manifest,
                    optimization_eligible=False, decisions=[], history=[])

        class Launcher:
            def __init__(self, *args, **kwargs):
                calls.append("launcher_created")

            def close_owned_groups(self, **kwargs):
                calls.append("native_closed")

        def check(*args, **kwargs):
            calls.append("identity")
            if fail_post_identity and calls.count("identity") == 2:
                raise ValueError("synthetic post-run identity change")

        def replay(folder, **kwargs):
            calls.append("replay")
            value = synthetic_evidence(output, self.result(row))
            return json.loads(Path(value["replay"]["path"]).read_text())

        def retain(env, collector, result, error, *, cleanup_seconds):
            # Exercise the real helper's wall-limit requirement: an already
            # active alarm here would prevent actual failure finalization.
            with validate_hero_policy.wall_limit(cleanup_seconds):
                result["failure_finalization"] = "persisted"
                runner.write(env.episode_dir / "synthetic-failed-evidence.json", {"incomplete": True})
            calls.append("failure_retained")

        for target, name, value in ((runner.planning, "read_plan", lambda *a: self.plan),
            (runner.planning, "check_identity", check),
            (runner.planning, "load_fixed", lambda *a: (SimpleNamespace(model=None, checkpoint="synthetic"), {})),
            (runner.guard, "DeadlineVcmiLauncher", Launcher), (builtin_env, "BuiltinHeadlessEnv", Environment),
            (builtin_rollout, "BuiltinCollector", Collector),
            (builtin_rollout, "audit_persisted_builtin_policy", lambda *a: self.result(row)["reconstruction"]),
            (builtin_rollout, "retain_failed_builtin_evidence", builtin_rollout.retain_failed_builtin_evidence if collection_error else retain),
            (builtin_replay, "verify_episode", replay),
            (builtin_behavior, "analyze_episode", lambda *a: {"synthetic": True}),
            (validate_hero_policy, "frozen_fingerprints", lambda *a: {})):
            stack.enter_context(patch.object(target, name, value))
        started, wall_started = time.monotonic(), time.time()
        launch = runner.launch_declaration(self.plan, self.sha, 0, started, wall_started)
        launch_path = output / "launch-01.json"
        runner.write(launch_path, launch, exclusive=True)
        report = {"schema": runner.SCHEMA, "plan_sha256": self.sha, "status": "running",
            "started_monotonic": started, "started_unix": wall_started,
            "deadline_monotonic": launch["campaign_deadline_monotonic"], "deadline_unix": launch["campaign_deadline_unix"],
            "attempts": [runner.empty_attempt(item) for item in self.plan["schedule"]]}
        report["attempts"][0].update(status="pending_verification", launch=runner.binding(launch_path))
        runner.write(output / "report.json", report)
        args = SimpleNamespace(**vars(self.args), index=0,
            **{key: launch[key] for key in ("deadline", "wall_deadline", "work_deadline", "work_wall_deadline")})
        return args, calls

    def test_worker_receipt_follows_both_cleanup_and_final_identity(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack)
            real_publish = runner.publish_completion
            def publish(*args, **kwargs):
                self.assertEqual(calls[-3:], ["environment_closed", "native_closed", "identity"])
                return real_publish(*args, **kwargs)
            stack.enter_context(patch.object(runner, "publish_completion", side_effect=publish))
            result = runner.worker(args)
        self.assertEqual(result["status"], "verified_natural")
        self.assertTrue((self.root / "execution/completion-01.json").exists())

    def test_worker_post_identity_failure_never_writes_completion_receipt_or_scores(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack, fail_post_identity=True)
            result = runner.worker(args)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rewards"], {"red": None, "blue": None})
        self.assertFalse((self.root / "execution/completion-01.json").exists())
        self.assertIn("synthetic post-run identity", result["identity_error"])

    def test_active_failure_retention_can_use_its_existing_finalization_alarm(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack, collection_error=True)
            result = runner.worker(args)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_finalization"], "persisted")
        self.assertIn("failure_retained", calls)
        self.assertNotIn("failure_retention_error", result)
        self.assertTrue((self.root / "execution/episode-000001/synthetic-failed-evidence.json").exists())

    def test_progress_write_failure_cannot_skip_cleanup_or_keep_scores(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack)
            real_write = runner.write
            def write(path, value, **kwargs):
                if Path(path).name == "attempt-01.json" and value.get("phase") == "cleanup":
                    raise OSError("synthetic progress storage failure")
                return real_write(path, value, **kwargs)
            stack.enter_context(patch.object(runner, "write", side_effect=write))
            result = runner.worker(args)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rewards"], {"red": None, "blue": None})
        self.assertEqual(calls[-3:], ["environment_closed", "native_closed", "identity"])
        self.assertTrue(result["progress_write_errors"])

    def test_standalone_or_expanded_worker_deadline_cannot_launch_native_work(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack)
            for key in ("deadline", "wall_deadline", "work_deadline", "work_wall_deadline"):
                altered = SimpleNamespace(**{**vars(args), key: getattr(args, key) + 3600})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "CLI clocks"):
                    runner.worker(altered)
            (self.root / "execution/launch-01.json").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.worker(args)
        self.assertNotIn("launcher_created", calls)
        self.assertNotIn("collection", calls)
        self.assertFalse((self.root / "execution/attempt-01.json").exists())

    def test_rehashed_launch_record_cannot_extend_shared_or_case_budget(self):
        with ExitStack() as stack:
            args, calls = self.setup_worker(stack)
            path = self.root / "execution/launch-01.json"
            report_path = self.root / "execution/report.json"
            launch = json.loads(path.read_text())
            report = json.loads(report_path.read_text())
            launch["row_deadline_monotonic"] += 1
            runner.write(path, launch)
            report["attempts"][0]["launch"] = runner.binding(path)
            runner.write(report_path, report)
            with self.assertRaisesRegex(ValueError, "case clock"):
                runner.worker(args)
        self.assertNotIn("launcher_created", calls)
        self.assertNotIn("collection", calls)


if __name__ == "__main__":
    unittest.main()
