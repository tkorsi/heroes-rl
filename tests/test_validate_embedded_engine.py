"""Validation-probe helpers over the scripted in-process engine; synthetic only."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.headless import builtin_evidence as e, embedded_env as ee, embedded_evidence as ev
from tests.test_embedded_env import FakeEngine, SyntheticEmbeddedLauncher
from validation.probes import validate_embedded_engine as probe


def scripted(root, player, boundary, *, kind="end_turn", engine=None, episode=1):
    launcher = SyntheticEmbeddedLauncher(root) if not (root / "bin").exists() else scripted.launcher
    scripted.launcher = launcher
    engine = engine or FakeEngine(boundary)
    engine.boundary = boundary
    identity = {"engine_revision": "synthetic-not-native", "map_sha256": "b" * 64,
                "native_runtime_sha256": launcher.runtime, "backend": {"kind": ev.BACKEND, "protocol": ev.PROTOCOL}}
    env = ee.EmbeddedBuiltinEnv(launcher, root / "episodes", learner_player=player, engine=engine)
    runner = probe.ScriptedRunner(kind, learner_player=player)
    with ExitStack() as stack:
        stack.enter_context(patch.object(env, "_identity", return_value=identity))
        stack.enter_context(patch.object(env, "_engine_evidence", return_value=({"synthetic": True}, None)))
        stack.enter_context(patch.object(e, "capture_files", return_value={"schema": e.REPLAY_SCHEMA, "recorded": True, "files": {}}))
        stack.enter_context(patch.object(ee.native, "terminal_capture_close"))
        try:
            end = runner.run_episode(env, episode=episode, seed=11, max_requests=1)
        finally:
            env.close()
    return end, env, engine


class ProbeHelperTests(unittest.TestCase):
    def test_end_turn_controller_natural_and_cutoff_history_summary(self):
        for player in (0, 1):
            for boundary in ("natural", "cutoff"):
                with self.subTest(player=player, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    end, env, engine = scripted(Path(tmp), player, boundary)
                    self.assertEqual([r["action"]["kind"] for r in engine.responses], ["end_turn"])
                    summary = probe.history_summary(env.episode_dir, end.manifest)
                    self.assertEqual(summary["action_mix"], {"end_turn": 1})
                    self.assertEqual(summary["result_mix"], {"turn_ended": 1})
                    self.assertEqual(summary["commands"], {"accepted": 1})
                    self.assertEqual((summary["results_accepted"], summary["results_rejected"]), (1, 0))
                    self.assertEqual(end.manifest["status"], "completed" if boundary == "natural" else "truncated")

    @staticmethod
    def write_run(folder, rows, planned=None, **plan_fields):
        folder.mkdir(parents=True, exist_ok=True)
        plan = {"backend": "embedded", "controller": "end_turn", "map": {"sha256": "a" * 64}, "checkpoint": None,
                "limits": {"choices": 4, "max_days": 12}, "visit_kind": None, "stall_seconds": 0.0,
                "random_policy_seed": None, "recorded": None, "runtime": {"vcmiclient": "c" * 64},
                "platform": {"system": "Linux"},
                "rows": [{key: row[key] for key in ("episode", "learner_player", "engine_seed")} for row in planned or rows],
                **plan_fields}
        (folder / "plan.json").write_text(json.dumps(plan))
        report = {"plan": {"sha256": probe.file_identity(folder / "plan.json")["sha256"]}, "rows": rows}
        (folder / "report.json").write_text(json.dumps(report))

    def matched_runs(self, root):
        rows = []
        for side in ("left", "right"):
            end, env, _ = scripted(root / side, 0, "natural")
            row = {"index": 0, "episode": 1, "learner_player": 0, "engine_seed": 11, "policy_seed": 12,
                   "controller": "end_turn", "status": "passed", "episode_status": end.manifest["status"],
                   "termination_reason": end.manifest["termination_reason"], "outcomes": end.manifest["outcomes_by_player"],
                   "episode_directory": str(env.episode_dir), "collection_seconds": 0.5, "reset_seconds": 0.1,
                   "step": {"seconds": 0.2}, "controller_time": {"seconds": 0.01},
                   "finalize": {"seconds": 0.05, "nested_seconds": {"step": 0.05}},
                   "history": probe.history_summary(env.episode_dir, end.manifest), "resources_collected": {}}
            rows.append(row)
        return rows

    def compare(self, root):
        return probe.compare(type("Args", (), {"left": root / "left", "right": root / "right",
                                               "output": root / "comparison.json"}))

    def test_matched_comparison_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = self.matched_runs(root)
            self.write_run(root / "left", [left], backend="cold")
            self.write_run(root / "right", [right])
            result = self.compare(root)
            self.assertTrue(result["rows"][0]["identical"] and result["rows"][0]["outcome_equal"])
            self.assertTrue(result["all_identical"] and result["runtime_equal"])
            self.assertEqual(result["backends"], ["cold", "embedded"])
            self.assertEqual(result["left"]["run"], "left")
            self.assertNotIn(tmp, json.dumps(result))
            summary = probe.summarize([left], "embedded")
            self.assertEqual(summary["natural_completed"], 1)
            self.assertEqual(summary["submitted_decisions"], 1)
            self.assertEqual(summary["decisions_per_second"], 2.0)
            self.assertEqual(summary["action_mix"], {"end_turn": 1})
            self.assertEqual(summary["finalize_nested_seconds"], {"step": 0.05})
            failed = {**left, "status": "failed"}
            self.assertEqual(probe.summarize([failed], "embedded")["natural_completed"], 0)

    def test_comparison_refuses_differently_configured_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = self.matched_runs(root)
            for field, value in (("limits", {"choices": 8, "max_days": 12}), ("map", {"sha256": "d" * 64}),
                                 ("checkpoint", {"sha256": "e" * 64}), ("random_policy_seed", 3), ("visit_kind", 101)):
                with self.subTest(field=field):
                    self.write_run(root / "left", [left])
                    self.write_run(root / "right", [right], **{field: value})
                    with self.assertRaisesRegex(ValueError, "not equally configured"):
                        self.compare(root)

    def test_comparison_reports_unmatched_rows_and_unequal_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = self.matched_runs(root)
            extra = {**left, "episode": 2, "learner_player": 1, "engine_seed": 12}
            self.write_run(root / "left", [left, extra])
            self.write_run(root / "right", [{**right, "episode_status": "truncated"}], planned=[left, extra])
            result = self.compare(root)
            self.assertEqual(result["unmatched_rows"], {"left": [2], "right": []})
            self.assertFalse(result["rows"][0]["outcome_equal"])
            self.assertFalse(result["all_identical"])

    def test_recorded_run_matches_its_source_despite_policy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = self.matched_runs(root)
            self.write_run(root / "left", [left], controller="policy", random_policy_seed=21)
            source = {"run": "left", "report_sha256": probe.file_identity(root / "left" / "report.json")["sha256"]}
            self.write_run(root / "right", [{**right, "controller": "recorded"}], controller="recorded", recorded=source)
            result = self.compare(root)
            self.assertTrue(result["recorded"] and result["all_identical"])
            self.write_run(root / "right", [{**right, "controller": "recorded"}], controller="recorded",
                           recorded={**source, "report_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "not equally configured"):
                self.compare(root)

    def test_row_status_needs_an_intended_ending_and_its_evidence(self):
        status = lambda result, **k: probe.row_status(result, **{"stall": False, "verify": False, "embedded": True, **k})
        collected = {"status": "collected", "episode_status": "completed"}
        self.assertEqual(status(collected), "passed")
        self.assertEqual(status({**collected, "episode_status": "truncated"}), "passed")
        self.assertEqual(status({**collected, "episode_status": "failed"}), "failed")
        self.assertEqual(status({"status": "failed", "episode_status": "completed"}), "failed")
        self.assertEqual(status(collected, verify=True), "failed")
        self.assertEqual(status({**collected, "replay_verified": True}, verify=True), "passed")
        self.assertEqual(status({**collected, "replay_verified": True, "policy_audit": False}, verify=True), "failed")
        stalled = {"status": "collected", "episode_status": "failed", "poisoned_after": True}
        self.assertEqual(status(stalled, stall=True), "passed")
        self.assertEqual(status({**stalled, "poisoned_after": False}, stall=True), "failed")
        self.assertEqual(status({**stalled, "poisoned_after": False}, stall=True, embedded=False), "passed")
        self.assertEqual(status({**stalled, "episode_status": "completed"}, stall=True), "failed")

    def test_phases_report_finalization_nested_in_step(self):
        class Env:
            def reset(self):
                return "reset"

            def step(self):
                return self._finish()

            def truncate(self):
                return self._finish()

            def _finish(self):
                return "finished"
        env = Env()
        phases = probe.Phases(env)
        env.reset(), env.step(), env.truncate()
        timing = phases.summary()
        self.assertEqual(timing["finalize"]["calls"], 2)
        self.assertEqual(set(timing["finalize"]["nested_seconds"]), {"step", "truncate"})
        self.assertEqual((timing["step"]["calls"], timing["truncate"]["calls"], len(timing["reset_samples"])), (1, 1, 1))
        phases.new_row()
        self.assertEqual(phases.summary()["finalize"]["calls"], 0)

    def test_collect_writes_compact_path_free_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run-a"
            row = {"episode": 1, "learner_player": 0, "engine_seed": 11, "status": "passed",
                   "episode_directory": "/private/x", "history": {"submitted": 3},
                   "error": f"FileExistsError: [Errno 17] File exists: '{tmp}/run-a/episode-000001'",
                   "replay_error": "TimeoutExpired: /home/someone/vcmi/build/bin/vcmireplay timed out",
                   "resources_collected": {"threads": 5}, "native_close": {"close_clean": True}}
            self.write_run(run, [row])
            report = json.loads((run / "report.json").read_text())
            report.update(status="finished", summary={"rows": 1, "episode_statuses": {"completed/natural_terminal": 1}})
            (run / "report.json").write_text(json.dumps(report))
            attachment = Path(tmp) / "comparison.json"
            attachment.write_text(json.dumps({"left": {"path": "/Users/someone/runs/x/report.json"}, "relative": "Maps/a.vmap"}))
            result = probe.collect(type("Args", (), {"runs": [run], "attach": [attachment], "output": Path(tmp) / "c.json"}))
            compact = result["runs"][0]["rows"][0]
            self.assertEqual((compact["submitted"], compact["close_clean"], compact["resources_collected"]["threads"]),
                             (3, True, 5))
            text = json.dumps(result)
            for leaked in ("/private", tmp, "/home/someone", "/Users/someone"):
                self.assertNotIn(leaked, text)
            self.assertIn("completed/natural_terminal", text)
            self.assertEqual(result["attached"]["comparison.json"]["relative"], "Maps/a.vmap")
            self.assertEqual(result["runs"][0]["run"], "run-a")

    def test_schedule_alternates_seats_with_fixed_seed_bases(self):
        rows = probe.schedule(3, [0, 1], 100, 200)
        self.assertEqual([(r["episode"], r["learner_player"], r["engine_seed"], r["policy_seed"]) for r in rows],
                         [(1, 0, 100, 200), (2, 1, 101, 201), (3, 0, 102, 202)])
        self.assertEqual({(r["engine_seed"], r["policy_seed"]) for r in probe.schedule(4, [0, 1], 7, 8, 0)}, {(7, 8)})


if __name__ == "__main__":
    unittest.main()
