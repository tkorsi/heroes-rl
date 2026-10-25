"""Counterfactual scorer checks on synthetic histories, never native game proof."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from tests.test_contextual_management import action, at_confirmation, begin, history_at
from validation.probes import validate_army_policy_support as p


class ArmyPolicySupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(301)
            self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.MANAGEMENT_MODEL_SCHEMA,
                                                              hidden_size=8, feature_size=4))
        self.model.eval().requires_grad_(False)

    def test_every_frame_and_recorded_upgrade_path_scores_without_sampling(self):
        history = at_confirmation()
        begin(history, action("upgrade_stack", "hero", selection="upgrade-1"))
        before = m.weights_sha256(self.model)
        with patch.object(m.ContextualDistribution, "sample", side_effect=AssertionError("Must not sample")):
            scored = p.score_history(history.frames, self.model)
        self.assertEqual(scored["frames_folded"], len(history.frames))
        self.assertEqual(scored["action_kinds_scored"],
                         {"open_hero": 1, "open_stack": 1, "select_upgrade": 1, "upgrade_stack": 1})
        self.assertEqual(scored["values_scored"], 4)
        self.assertEqual(scored["checkpoint_weights_sha256"], before)
        self.assertEqual(m.weights_sha256(self.model), before)
        self.assertEqual(scored["actions_sampled"], 0)
        self.assertFalse(scored["optimization_eligible"])
        self.assertEqual(scored["training_samples"], 0)
        for frame, record in zip(history.frames, scored["frames"]):
            self.assertEqual((record["seq"], record["kind"]), (frame.seq, frame.kind))
        for record in scored["decisions"]:
            self.assertIsNotNone(record["operator_action"])
            self.assertTrue(record["conditionals"])
            self.assertAlmostEqual(record["joint_log_prob"], sum(item["log_prob"] for item in record["conditionals"]), places=5)

    def test_independent_folds_reset_player_memory_and_preserve_exact_support(self):
        red = history_at().frames
        blue = tuple(replace(frame, player=1) for frame in red)
        red_first = p.score_history(red, self.model)
        blue_first = p.score_history(blue, self.model)
        blue_second = p.score_history(blue, self.model)
        red_second = p.score_history(red, self.model)
        for first, second in ((red_first, red_second), (blue_first, blue_second)):
            comparison = p.compare_folds(first, second)
            self.assertTrue(comparison["exact"])
            self.assertEqual(comparison["frames_compared"], len(red))
            self.assertEqual(comparison["maximum_absolute_difference"], 0)
        self.assertEqual(red_first["initial_memory_sha256"], blue_first["initial_memory_sha256"])
        self.assertIsNone(red_first["decisions"][-1]["operator_action"])
        self.assertEqual(red_first["decisions"][-1]["support"]["offers"]["schema"], c.MANAGEMENT_ACTION_SCHEMA)

    def test_exact_comparison_rejects_each_material_output_change(self):
        scored = p.score_history(history_at().frames, self.model)
        for field in ("value", "joint_log_prob", "support", "conditionals", "memory"):
            with self.subTest(field=field):
                changed = deepcopy(scored)
                if field == "memory": changed["frames"][-1]["memory_sha256"] = "0" * 64
                elif field == "support": changed["decisions"][0][field]["context"]["kind"] = "town"
                elif field == "conditionals": changed["decisions"][0][field][0]["log_prob"] += 1e-9
                else: changed["decisions"][0][field] += 1e-9
                with self.assertRaisesRegex(ValueError, "folds differ"):
                    p.compare_folds(scored, changed)

    def test_frozen_model_and_player_local_history_are_required(self):
        frames = history_at().frames
        self.model.train()
        with self.assertRaisesRegex(ValueError, "frozen"): p.score_history(frames, self.model)
        self.model.eval()
        with self.assertRaises(ValueError): p.score_history((*frames[:-1], replace(frames[-1], player=1)), self.model)
        with self.assertRaisesRegex(ValueError, "nonempty"): p.score_history((), self.model)

    def test_split_joint_probability_includes_explicit_quantity_choice(self):
        history = history_at()
        begin(history, action("split_stack", "town", selection="upper-0", destination="lower-0", count=5))
        scored = p.score_history(history.frames, self.model)
        selected = scored["decisions"][-1]
        self.assertEqual(selected["operator_action"]["count"], 5)
        self.assertGreater(len(selected["conditionals"]), 3)
        self.assertEqual(scored["action_kinds_scored"]["split_stack"], 1)

    def test_replay_gate_requires_exact_manifest_verified_native_v6(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {"episode": "synthetic-binding-only", "environment": c.MANAGEMENT_ENVIRONMENT,
                        "history_schema": c.MANAGEMENT_SCHEMA, "action_schema": c.MANAGEMENT_ACTION_SCHEMA,
                        "status": "truncated"}
            path.write_text(json.dumps(manifest))
            replay = {"schema": p.REPORT_SCHEMA, "kind": "contextual-native-replay", "verified": True,
                      "episode": manifest["episode"], "training_performed": False,
                      "inputs": {"manifest": p.evidence.file_identity(path)}}
            p.validate_replay_binding(path, manifest, replay)
            for field, value in (("verified", False), ("episode", "other"), ("training_performed", True), ("schema", "other")):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    p.validate_replay_binding(path, manifest, {**replay, field: value})
            with self.assertRaisesRegex(ValueError, "v6"):
                p.validate_replay_binding(path, {**manifest, "history_schema": c.HEROES_SCHEMA}, replay)
            path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "another manifest"):
                p.validate_replay_binding(path, manifest, replay)

    def test_v7_dialog_choices_score_exact_semantics_with_private_memory(self):
        from tests.test_contextual_dialogs import FAMILIES, history_at as dialog_history, panel, begin_answer, finish_answer
        with torch.random.fork_rng():
            torch.manual_seed(302)
            model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        model.eval().requires_grad_(False)
        for family in FAMILIES:
            history = dialog_history(panel(family))
            seq = begin_answer(history, 1); finish_answer(history, seq)
            with patch.object(m.ContextualDistribution, "sample", side_effect=AssertionError("Must not sample")):
                first = p.score_history(history.frames, model)
                second = p.score_history(history.frames, model)
            self.assertTrue(p.compare_folds(first, second)["exact"])
            self.assertEqual(first["schema"], p.DIALOGS_SCHEMA)
            self.assertEqual(first["action_kinds_scored"], {"move_to": 1, "choose_dialog": 1})
            self.assertEqual(first["decisions"][-1]["support"]["offers"]["dialog_panel"]["family"], family)
            with self.assertRaises(ValueError): p.score_history(history.frames, self.model)

    def test_bound_file_mutation_and_failed_attempt_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source"; source.write_text("one")
            bindings = {"source": {"path": str(source), **p.evidence.file_identity(source)}}
            p.check_bindings(bindings)
            source.write_text("two")
            with self.assertRaisesRegex(ValueError, "changed during scoring"): p.check_bindings(bindings)
            episode = directory / "episode"; episode.mkdir()
            (episode / "manifest.json").write_text("{}")
            replay = directory / "replay.json"; replay.write_text('{"verified":false}')
            args = SimpleNamespace(episode=episode, replay_report=replay, checkpoint=directory / "absent.pt",
                                   output=directory / "output")
            report = p.run(args)
            self.assertEqual(report["status"], "failed")
            self.assertIn("independent native replay", report["error"])
            self.assertEqual(json.loads((args.output / "report.json").read_text()), report)
            self.assertEqual(report["optimizer_updates"], 0)
            with self.assertRaisesRegex(ValueError, "preserve prior attempts"): p.run(args)


if __name__ == "__main__": unittest.main()
