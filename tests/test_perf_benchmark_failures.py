"""Failure paths of the performance harness: nothing unverified is ever scored."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.perf import contextual_benchmark as bench

ROOT = Path(__file__).resolve().parents[1]


# -- fakes for one attempt -----------------------------------------------------

def manifest(status="truncated"):
    player = {"actions_sent": 3, "actions_accepted": 2, "actions_rejected": 1, "responses_undelivered": 0,
              "live_evidence": {"continuation_steps": 0}}
    return {"status": status, "termination_reason": "choice_limit", "players": {"red": player, "blue": player},
            "outcomes_by_player": {"red": "cutoff", "blue": "cutoff"}, "wall_seconds": 1.0, "seed": 1,
            "failure_detail": "engine exited" if status == "failed" else None}


class FakeEnv:
    status = "truncated"
    close_error = None

    def __init__(self, launcher, episodes, **kwargs):
        self.calls = launcher.calls
        self.episode_dir = Path(episodes) / "episode"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "manifest.json").write_text(json.dumps(manifest(self.status)))

    def close(self):
        self.calls.append("env.close")
        if self.close_error:
            raise self.close_error


class FakeLauncher:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def close_owned_groups(self):
        self.calls.append("close_owned_groups")
        if self.error:
            raise self.error


class FakeCollector:
    on_run = None

    def __init__(self, models, checkpoints, **kwargs):
        pass

    def act(self, request):
        return None

    def run_episode(self, env, episode, seed, max_requests):
        if self.on_run:
            self.on_run()
        rollout = SimpleNamespace(player=0, history=[SimpleNamespace(kind="decision", day=1)], decisions=[1, 2],
                                  estimates=[], history_sha256="0" * 64, boundary="cutoff", optimization_eligible=False)
        return SimpleNamespace(rollouts=[rollout, SimpleNamespace(**{**vars(rollout), "player": 1})], to_dict=dict)


def operations(**overrides):
    calls = []
    ops = {"env_class": FakeEnv, "limits_class": dict, "collector_class": FakeCollector,
           "verify_episode": lambda folder, **kw: calls.append("replay") or {"verified": True, "native": {"returncode": 0}},
           "analyze_episode": lambda folder: calls.append("behavior") or {"status": "ok"},
           "audit_persisted_policy": lambda folder, loaded, seeds, interface: {"players": [
               {"player": p, "actions_resampled": 1, "values_recomputed": 1, "conditionals_recomputed": 1,
                "maximum_absolute_difference": 0.0} for p in (0, 1)]},
           "frozen_fingerprints": lambda loaded, path: {"fixed": True},
           "retain_failed_evidence": lambda env, collector, attempt, error, cleanup_seconds: calls.append("retained"),
           "load_fixed": lambda path, cfg, interface: (SimpleNamespace(model=None), {"fixed": True}),
           "instrument_environment": lambda env, launcher: None}
    ops.update(overrides)
    return ops, calls


def context(folder, ops, launcher=None):
    cfg = bench.config()
    plan = {"choice_limit": 4, "checkpoint": {"path": "checkpoint.pt"}, "replay_binary": {"path": "vcmireplay"},
            "data": "data"}
    (Path(folder) / "attempts").mkdir(exist_ok=True)
    return {"cfg": cfg, "plan": plan, "launcher": launcher or FakeLauncher(), "ops": ops, "fixed": {"fixed": True},
            "loaded": SimpleNamespace(model=None, checkpoint="learner"), "opponent": SimpleNamespace(model=None),
            "episodes": Path(folder) / "episodes", "deadline": time.monotonic() + 1000, "wrapper_overhead": 0.0}


def row(index=0):
    return {"row": index, "episode": index + 1, "engine_seed": 140001, "learner_player": 0,
            "policy_seeds": [140101, 140102]}


def kill_during_collection(folder):
    """Run one real attempt whose collection is ended by SIGKILL."""
    FakeCollector.on_run = lambda self=None: os.kill(os.getpid(), signal.SIGKILL)
    ops, _ = operations()
    bench.run_row(context(folder, ops), row(), Path(folder))


class AttemptFailureTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.folder)]))
        self.addCleanup(setattr, FakeEnv, "status", "truncated")
        self.addCleanup(setattr, FakeEnv, "close_error", None)

    def attempt(self, ops, launcher=None):
        launcher = launcher or FakeLauncher()
        result = bench.run_row(context(self.folder, ops, launcher), row(), self.folder)
        persisted = json.loads((self.folder / "attempts" / "row-000.json").read_text())
        self.assertEqual(persisted, result)
        return result, launcher.calls

    def test_fully_checked_attempt_passes_and_continues(self):
        ops, calls = operations()
        result, cleanup = self.attempt(ops)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["stop_worker"])
        self.assertEqual(calls, ["replay", "behavior"])
        self.assertEqual(cleanup, ["env.close", "close_owned_groups"])
        self.assertTrue(result["cleanup_verified"])
        self.assertEqual(result["outcomes"], {"red": "cutoff", "blue": "cutoff"})

    def test_late_failures_after_replay_fail_and_stop_with_both_cleanups(self):
        def broken(*args, **kwargs):
            raise RuntimeError("late")
        for key, phase in (("analyze_episode", "behavior"), ("frozen_fingerprints", "weights_check")):
            with self.subTest(key=key):
                ops, calls = operations(**{key: broken})
                result, cleanup = self.attempt(ops)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failed_phase"], phase)
                self.assertTrue(result["stop_worker"])
                self.assertIn("replay", calls)
                self.assertIn("retained", calls)
                self.assertEqual(cleanup, ["env.close", "close_owned_groups"])
                self.assertEqual(result["outcomes"], {"red": None, "blue": None})

    def test_changed_weights_fail(self):
        ops, _ = operations(frozen_fingerprints=lambda loaded, path: {"fixed": False})
        result, _ = self.attempt(ops)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Weights changed", result["error"])

    def test_cleanup_errors_fail_a_verified_attempt_and_never_skip_the_other_cleanup(self):
        FakeEnv.close_error = OSError("close")
        ops, _ = operations()
        result, cleanup = self.attempt(ops, FakeLauncher(error=OSError("groups")))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["stop_worker"])
        self.assertEqual(cleanup, ["env.close", "close_owned_groups"])
        self.assertFalse(result["cleanup_verified"])
        self.assertEqual(len(result["cleanup_errors"]), 2)
        self.assertEqual(result["outcomes"], {"red": None, "blue": None})

    def test_native_failure_is_retained_unscored_and_continuable(self):
        FakeEnv.status = "failed"
        ops, calls = operations()
        result, cleanup = self.attempt(ops)
        self.assertEqual(result["status"], "native_failure")
        self.assertFalse(result["stop_worker"])
        self.assertNotIn("replay", calls)
        self.assertEqual(result["outcomes"], {"red": None, "blue": None})
        self.assertEqual(cleanup, ["env.close", "close_owned_groups"])

    def test_a_hard_killed_attempt_stays_incomplete(self):
        code = "import sys; from tests.test_perf_benchmark_failures import kill_during_collection; kill_during_collection(sys.argv[1])"
        done = subprocess.run([sys.executable, "-c", code, str(self.folder)], cwd=ROOT, capture_output=True, timeout=120)
        self.assertEqual(done.returncode, -signal.SIGKILL, done.stderr.decode())
        marker = json.loads((self.folder / "attempts" / "row-000.json").read_text())
        self.assertEqual((marker["status"], marker["phase"]), ("started", "collection"))
        attempts = bench.collect_attempts([row(0), row(1)], self.folder, [{"attempts": []}], signalled=False)
        self.assertEqual(attempts[0]["status"], "incomplete")
        self.assertEqual(attempts[0]["outcomes"], {"red": None, "blue": None})
        self.assertEqual((attempts[1]["status"], attempts[1]["reason"]), ("not_started", "worker_failure"))
        summary = bench.summarize_attempts(attempts, 1.0)
        self.assertEqual((summary["incomplete"], summary["failures"], summary["not_started"]), (1, 1, 1))

    def test_unreadable_record_is_incomplete_and_skip_reasons_are_kept(self):
        (self.folder / "attempts").mkdir()
        (self.folder / "attempts" / "row-000.json").write_text("{")
        workers = [{"attempts": [{"row": 1, "status": "not_started", "reason": "identity_drift_before_row"}]}]
        attempts = bench.collect_attempts([row(0), row(1)], self.folder, workers, signalled=True)
        self.assertEqual([a["status"] for a in attempts], ["incomplete", "not_started"])
        self.assertEqual(attempts[1]["reason"], "identity_drift_before_row")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401  (the worker sets its thread count)
        except ModuleNotFoundError as error:
            if error.name != "torch": raise
            self.skipTest("Optional torch is not installed")
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.folder)]))
        (self.folder / "attempts").mkdir()
        cfg = bench.config()
        self.plan = {"configuration": cfg, "output": str(self.folder), "row_count": 3, "frozen": {"fixed": True},
                     "assignment": {"0": [0, 1, 2]}, "choice_limit": 4, "checkpoint": {"path": "checkpoint.pt"},
                     "replay_binary": {"path": "vcmireplay"}, "data": "data"}
        bench.write(self.folder / "plan.json", self.plan)
        self.threads = __import__("torch").get_num_threads()

    def tearDown(self):
        __import__("torch").set_num_threads(self.threads)

    def run_worker(self, drift_after=None, **overrides):
        ops, calls = operations(**overrides)
        checks = []

        def drift(plan, sha, path):
            checks.append(sha)
            return ["source:experiments/learner/contextual.py"] if drift_after is not None and len(checks) > drift_after else []
        loaded = SimpleNamespace(model=None, checkpoint="learner", opponent_pool={bench.config()["opponent"][0]: SimpleNamespace(model=None)})
        ops.update(instrument_modules=False, identity_drift=drift, launcher_factory=lambda plan, args, output: FakeLauncher(),
                   source_compatibility=lambda loaded, interface: {"compatible": True},
                   load_fixed=lambda path, cfg, interface: (loaded, {"fixed": True}))
        args = argparse.Namespace(plan=str(self.folder / "plan.json"), plan_sha256=bench.digest(self.folder / "plan.json"),
                                  worker=0, deadline=time.monotonic() + 1000, wall_deadline=time.time() + 1000, cprofile=False)
        with mock.patch.object(bench, "wrapper_overhead_seconds", return_value=0.0):
            report = bench.worker(args, ops)
        return report, calls

    def test_all_rows_run_when_nothing_fails(self):
        report, _ = self.run_worker()
        self.assertEqual([a["status"] for a in report["attempts"]], ["passed"] * 3)
        self.assertEqual(report["status"], "finished")

    def test_failed_attempt_stops_the_worker_and_records_the_rest(self):
        def broken(folder):
            raise RuntimeError("behavior")
        report, _ = self.run_worker(analyze_episode=broken)
        self.assertEqual([a["status"] for a in report["attempts"]], ["failed", "not_started", "not_started"])
        self.assertEqual(report["attempts"][1]["reason"], "worker_stopped_after_failed")

    def test_identity_drift_during_a_row_fails_it_and_stops(self):
        report, _ = self.run_worker(drift_after=1)
        self.assertEqual([a["status"] for a in report["attempts"]], ["failed", "not_started", "not_started"])
        persisted = json.loads((self.folder / "attempts" / "row-000.json").read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("Identity drift", persisted["error"])
        self.assertTrue(report["identity_drift"])

    def test_identity_drift_before_a_row_skips_it(self):
        report, _ = self.run_worker(drift_after=2)
        self.assertEqual([a["status"] for a in report["attempts"]], ["passed", "not_started", "not_started"])
        self.assertEqual(report["attempts"][1]["reason"], "identity_drift_before_row")

    def test_a_changed_plan_file_is_refused(self):
        args = argparse.Namespace(plan=str(self.folder / "plan.json"), plan_sha256="0" * 64, worker=0)
        with self.assertRaises(ValueError):
            bench.worker(args, {})


class IdentityDriftTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.root)]))
        git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "code.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "code.py"], cwd=self.root, check=True)
        subprocess.run(git + ["commit", "-qm", "c"], cwd=self.root, check=True)
        self.native = {"binary": {"sha256": "a"}, "runtime": {"libs": "b"}}
        self.plan = {"source": bench.source_snapshot(self.root), "native_snapshot": dict(self.native)}
        self.plan_path = self.root / "plan.json"
        bench.write(self.plan_path, self.plan)
        self.sha = bench.digest(self.plan_path)

    def drift(self):
        with mock.patch.object(bench, "native_snapshot", return_value=self.native):
            return bench.identity_drift(self.plan, self.sha, self.plan_path, root=self.root)

    def test_unchanged_identities_have_no_drift(self):
        self.assertEqual(self.drift(), [])

    def test_each_identity_change_is_named(self):
        (self.root / "code.py").write_text("x = 2\n")
        self.assertEqual(self.drift(), ["source:code.py"])
        (self.root / "code.py").write_text("x = 1\n")
        self.plan_path.write_text(self.plan_path.read_text() + " ")
        self.assertEqual(self.drift(), ["plan"])
        bench.write(self.plan_path, self.plan)
        self.native = {**self.native, "runtime": {"libs": "changed"}}
        self.assertEqual(self.drift(), ["native_runtime"])

    def test_unreadable_identity_is_drift(self):
        with mock.patch.object(bench, "native_snapshot", side_effect=OSError("gone")):
            drift = bench.identity_drift(self.plan, self.sha, self.plan_path, root=self.root)
        self.assertTrue(drift and drift[0].startswith("unreadable:"))


class ProcessInspectionTests(unittest.TestCase):
    TABLE = {1: (0, "0.0", "/sbin/launchd"),
             100: (1, "1.0", "python -m experiments.perf.contextual_benchmark run"),
             101: (100, "90.0", "python -m experiments.perf.contextual_benchmark worker /owned/plan.json"),
             102: (101, "95.0", "/build/bin/vcmiclient --headless"),
             200: (1, "80.0", "/other/bin/vcmiclient --headless"),
             300: (1, "50.0", "python -m experiments.perf.contextual_benchmark worker /owned/other-plan.json"),
             400: (1, "70.0", "ninja -C /owned/vcmi/build"),
             500: (1, "3.0", "/Applications/Safari.app/Contents/MacOS/Safari")}

    def test_only_the_supervisors_own_tree_is_excluded(self):
        found = {item["pid"]: item for item in bench.foreign_processes(100, "/owned", self.TABLE)}
        self.assertEqual(sorted(found), [200, 300, 400])
        self.assertFalse(found[200]["same_root"])
        self.assertTrue(found[300]["same_root"])
        self.assertTrue(found[400]["same_root"])

    def test_failed_inspection_is_unknown_not_clean(self):
        with mock.patch.object(bench, "process_table", return_value=None):
            self.assertIsNone(bench.foreign_processes(100, "/owned"))
        monitor = bench.LoadMonitor(100, "/owned", interval=0.01, inspect=lambda pid, root: None)
        monitor.start()
        time.sleep(0.05)
        monitor.halt.set()
        monitor.join(5)
        self.assertGreater(monitor.failed_samples, 0)
        self.assertEqual(monitor.foreign, {})


class FakeClock:
    def __init__(self, step=0.0):
        self.now, self.step = 1000.0, step

    def __call__(self):
        self.now += self.step
        return self.now


class FakeProcess:
    def __init__(self, pid, finished=False):
        self.pid, self.finished = pid, finished

    def poll(self):
        return 0 if self.finished else None


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.output)]))
        (self.output / "attempts").mkdir()
        self.plan = {"label": "t", "workers": 3, "rows": [row(i) for i in range(3)], "row_count": 3, "choice_limit": 4,
                     "campaign_seconds": 60.0, "cleanup_grace_seconds": 5.0, "cprofile": False,
                     "declared_source": {"implementation": "optimized"},
                     "source": {"tracked_sha256": "s", "git_commit": "c"}}
        bench.write(self.output / "plan.json", self.plan)

    def operations(self, clock, spawn, close_cost=0.0):
        self.timeouts = []

        def close(process, timeout):
            self.timeouts.append(("close", process.pid, timeout))
            clock.now += close_cost

        def evidence(path, timeout):
            self.timeouts.append(("evidence", path.name, timeout))
        return {"spawn": spawn, "close_process": close, "cleanup_evidence": evidence, "monotonic": clock,
                "wall": time.time, "sleep": lambda seconds: setattr(clock, "now", clock.now + seconds),
                "inspect": lambda pid, root: [], "kill": lambda pid, sig: None, "monitor_interval": 0.01}

    def test_a_failed_second_spawn_still_cleans_up_and_reports(self):
        clock, spawned = FakeClock(), []

        def spawn(command, **kwargs):
            if spawned:
                raise OSError("spawn")
            spawned.append(FakeProcess(10 + len(spawned)))
            return spawned[-1]
        report = bench.supervise(self.plan, self.output, self.operations(clock, spawn))
        self.assertTrue((self.output / "campaign.json").exists())
        self.assertEqual(report["supervisor_errors"], ["supervisor: OSError: spawn"])
        self.assertEqual([t[:2] for t in self.timeouts], [("close", 10), ("evidence", "native-deadline-worker-00.jsonl")])
        self.assertEqual([w["status"] for w in report["worker_reports"]], ["missing", "not_launched", "not_launched"])
        self.assertEqual([a["status"] for a in report["attempts"]], ["not_started"] * 3)
        self.assertFalse(report["clean_timing_sample"])
        self.assertIn("supervisor_errors", bench.exclusion_reasons(report))

    def test_cleanup_waits_share_one_budget_that_ends_at_the_grace_deadline(self):
        clock, pids = FakeClock(step=0.5), iter(range(10, 20))
        ops = self.operations(clock, lambda command, **kw: FakeProcess(next(pids)), close_cost=4.0)
        report = bench.supervise(self.plan, self.output, ops)
        self.assertTrue((self.output / "campaign.json").exists())
        self.assertTrue(report["deadline_signalled"])
        waits = [t[2] for t in self.timeouts]
        self.assertEqual(len(waits), 6)
        start = 1000.5  # the first monotonic reading
        cleanup_end = start + self.plan["campaign_seconds"] + self.plan["cleanup_grace_seconds"]
        self.assertLessEqual(waits[0], self.plan["cleanup_grace_seconds"] + 1)
        self.assertTrue(all(wait == 0.01 for wait in waits[2:]), waits)
        self.assertGreater(clock.now, cleanup_end)
        self.assertEqual([w["status"] for w in report["worker_reports"]], ["missing"] * 3)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.folder)]))

    def campaign(self, name, implementation="optimized", source="s1", wall=10.0, **report_overrides):
        path = self.folder / name
        path.mkdir()
        attempts = [{"row": 0, "status": "passed", "episode_status": "truncated",
                     "histories": {"red": {"history_sha256": "h0"}}}]
        summary = bench.summarize_attempts(attempts, wall)
        report = {"label": "bench", "workers": 1, "cprofile": False, "concurrent_load_detected": False,
                  "inspection_failed": False, "supervisor_errors": [], "attempts": attempts, "summary": summary}
        report.update(report_overrides)
        plan = {"declared_source": {"implementation": implementation}, "source": {"tracked_sha256": source},
                "row_count": 1, "choice_limit": 4, "campaign_seconds": 60.0, "cprofile": report["cprofile"],
                "configuration": {"a": 1}, "native_snapshot": {"binary": {"sha256": "b"}}, "checkpoint": {"sha256": "k"}}
        bench.write(path / "campaign.json", report)
        bench.write(path / "plan.json", plan)
        return str(path)

    def test_only_clean_matching_campaigns_enter_metrics(self):
        paths = [self.campaign("a", wall=10.0), self.campaign("b", wall=20.0),
                 self.campaign("c", wall=1.0, concurrent_load_detected=True),
                 self.campaign("d", wall=1.0, inspection_failed=True),
                 self.campaign("e", wall=2.0, implementation="baseline", source="s0")]
        with mock.patch("builtins.print"):
            groups = bench.summarize(argparse.Namespace(campaigns=paths, output=None))
        self.assertEqual(len(groups), 2)
        optimized = next(g for g in groups if g["implementation"] == "optimized")
        self.assertEqual(optimized["clean_campaigns"], paths[:2])
        self.assertEqual(optimized["excluded"], {paths[2]: ["concurrent_load"], paths[3]: ["load_inspection_unknown"]})
        self.assertEqual(optimized["clean_metrics"]["campaign_wall_seconds"], {"median": 15.0, "min": 10.0, "max": 20.0, "n": 2})
        self.assertEqual(optimized["history_hash_sets"], {"0:red": ["h0"]})
        baseline = next(g for g in groups if g["implementation"] == "baseline")
        self.assertEqual(baseline["clean_campaigns"], [paths[4]])

    def test_failed_or_unstarted_rows_exclude_a_campaign(self):
        attempts = [{"row": 0, "status": "incomplete"}, {"row": 1, "status": "not_started"}]
        report = {"cprofile": True, "inspection_failed": False, "summary": bench.summarize_attempts(attempts, 1.0)}
        self.assertEqual(bench.exclusion_reasons(report), ["profiled", "incomplete_or_failed_rows"])
        self.assertIn("load_inspection_unknown", bench.exclusion_reasons({"summary": {}}))


if __name__ == "__main__":
    unittest.main()
