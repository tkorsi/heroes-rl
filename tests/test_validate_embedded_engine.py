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

    def test_matched_comparison_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = []
            for side in ("left", "right"):
                end, env, _ = scripted(root / side, 0, "natural")
                row = {"index": 0, "episode": 1, "learner_player": 0, "engine_seed": 11, "policy_seed": 12,
                       "controller": "end_turn", "status": "passed", "episode_status": end.manifest["status"],
                       "termination_reason": end.manifest["termination_reason"], "outcomes": end.manifest["outcomes_by_player"],
                       "episode_directory": str(env.episode_dir), "collection_seconds": 0.5, "reset_seconds": 0.1,
                       "step": {"seconds": 0.2}, "controller_time": {"seconds": 0.01}, "finalize": {"seconds": 0.05},
                       "history": probe.history_summary(env.episode_dir, end.manifest), "resources_after": {}}
                (root / side / "report.json").write_text(json.dumps({"rows": [row]}))
                reports.append(row)
            result = probe.compare(type("Args", (), {"left": root / "left", "right": root / "right",
                                                      "output": root / "comparison.json"}))
            self.assertTrue(result["rows"][0]["identical"] and result["rows"][0]["outcome_equal"])
            summary = probe.summarize(reports[:1], "embedded")
            self.assertEqual(summary["natural_completed"], 1)
            self.assertEqual(summary["submitted_decisions"], 1)
            self.assertEqual(summary["decisions_per_second"], 2.0)
            self.assertEqual(summary["action_mix"], {"end_turn": 1})

    def test_collect_writes_compact_path_free_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run-a"
            run.mkdir()
            plan = {"backend": "embedded", "controller": "end_turn", "map": {"sha256": "a" * 64}, "checkpoint": None,
                    "limits": {"choices": 4}, "platform": {"system": "Linux"}, "rows": [{}]}
            row = {"episode": 1, "status": "passed", "episode_directory": "/private/x", "history": {"submitted": 3},
                   "resources_after": {"threads": 5}, "native_close": {"close_clean": True}}
            (run / "plan.json").write_text(json.dumps(plan))
            (run / "report.json").write_text(json.dumps({"status": "finished", "rows": [row], "summary": {"rows": 1}}))
            result = probe.collect(type("Args", (), {"runs": [run], "attach": [], "output": Path(tmp) / "c.json"}))
            compact = result["runs"][0]["rows"][0]
            self.assertEqual((compact["submitted"], compact["close_clean"], compact["resources_after"]["threads"]), (3, True, 5))
            self.assertNotIn("/private", json.dumps(result))

    def test_schedule_alternates_seats_with_fixed_seed_bases(self):
        rows = probe.schedule(3, [0, 1], 100, 200)
        self.assertEqual([(r["episode"], r["learner_player"], r["engine_seed"], r["policy_seed"]) for r in rows],
                         [(1, 0, 100, 200), (2, 1, 101, 201), (3, 0, 102, 202)])
        self.assertEqual({(r["engine_seed"], r["policy_seed"]) for r in probe.schedule(4, [0, 1], 7, 8, 0)}, {(7, 8)})


if __name__ == "__main__":
    unittest.main()
