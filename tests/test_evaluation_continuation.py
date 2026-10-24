"""Evaluation continuation guards with synthetic state; no engine or optimizer steps."""
from copy import deepcopy
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
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import evaluate_continuation as evaluation
from experiments.learner.checkpoints import FrozenOpponentPool, TrainingProgress, save_checkpoint
from experiments.learner.contextual_model import ARMIES_MODEL_SCHEMA, ContextualActorCritic, ModelConfig, weights_sha256
from experiments.learner.inference import digest, write_json


class EvaluationContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def plan(self, folder):
        folder = folder.resolve()
        return {"schema": evaluation.PLAN_SCHEMA, "purpose": "evaluation_only", "additional_optimizer_steps": 0,
                "attempt_schedule": evaluation.fixed_schedule(), "remaining_wall_seconds": evaluation.REMAINING_SECONDS,
                "limits": {"max_requests": 256, "max_days": 30, "torch_threads": 1},
                "execution_output": str(folder / "execution"), "source": {"synthetic": True},
                "original_report": {"path": str(folder / "original/report.json")},
                "original_plan": {}, "resume_checkpoint": {}, "authorization": {},
                **{key: "synthetic" for key in ("native_identity", "native_runtime", "map_identity",
                                                "binary", "replay_binary", "data", "game_map")}}

    def test_fixed_scope_rejects_extensions_new_updates_and_numeric_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.plan(Path(folder))
            # Native/evidence checks are separately bound; these mutations must
            # fail in the pure scope check before any external operation occurs.
            mutations = [lambda p: p["attempt_schedule"].append(deepcopy(p["attempt_schedule"][0])),
                         lambda p: p["attempt_schedule"].pop(),
                         lambda p: p["attempt_schedule"][0].update(episode=11),
                         lambda p: p["attempt_schedule"][1].update(learner_player=0),
                         lambda p: p["attempt_schedule"][0].update(learner_player=False),
                         lambda p: p["attempt_schedule"][0].update(policy_seeds=[1, 2]),
                         lambda p: p["limits"].update(torch_threads=True),
                         lambda p: p.update(additional_optimizer_steps=1),
                         lambda p: p.update(additional_optimizer_steps=False),
                         lambda p: p.update(remaining_wall_seconds=3600)]
            for mutate in mutations:
                changed = deepcopy(plan)
                mutate(changed)
                with self.subTest(plan=changed), patch.object(evaluation, "check_original") as original:
                    with self.assertRaisesRegex(ValueError, "Continuation scope"):
                        evaluation.validate_plan(changed)
                    original.assert_not_called()

    def test_old_plan_is_not_accepted_as_a_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            write_json(path, {"schema": "heroes-first-training-plan-v1"})
            with patch.object(evaluation, "check_identity") as identity:
                with self.assertRaisesRegex(ValueError, "Continuation scope"):
                    evaluation.read_plan(path, digest(path))
                with self.assertRaisesRegex(ValueError, "exact new continuation plan"):
                    evaluation.read_plan(path, "0" * 64)
                identity.assert_not_called()

    def test_artifact_and_current_source_mutations_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evidence = folder / "evidence.json"
            write_json(evidence, {"original": True})
            bound = evaluation.binding(evidence)
            write_json(evidence, {"original": False})
            with self.assertRaisesRegex(ValueError, "Bound artifact changed"):
                evaluation.check_binding(bound)
            plan = self.plan(folder)
            path = folder / "plan.json"
            write_json(path, plan)
            with patch.object(evaluation, "validate_plan"), patch.object(evaluation, "check_identity",
                    side_effect=ValueError("Prepared implementation inventory or file bytes changed")):
                with self.assertRaisesRegex(ValueError, "implementation inventory"):
                    evaluation.read_plan(path, digest(path))

    def test_readonly_saved_checkpoint_load_uses_no_optimizer_or_rng(self):
        model = ContextualActorCritic(ModelConfig(schema=ARMIES_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        model_hash = weights_sha256(model)
        pool = FrozenOpponentPool()
        pool.add("synthetic-frozen-opponent", model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        original = {"source": {"fixture": "synthetic checkpoint format test"}, "configuration": {"ppo": {"epochs": 1}}}
        metadata = {"plan_sha256": evaluation.ORIGINAL_PLAN_SHA256, **original}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, checkpoint="first-update-" + model_hash[:16],
                ppo_config=original["configuration"]["ppo"], progress=TrainingProgress(**evaluation.PROGRESS),
                source_metadata=metadata, opponent_pool=pool, sampler_state={"next_phase": "after", "next_episode": 7},
                at_rollout_boundary=True, interface=c.ARMIES_ENVIRONMENT, provenance="approved_gameplay")
            before = digest(path)
            rng = torch.get_rng_state().clone()
            with patch.object(evaluation, "LEARNER_SHA256", model_hash), patch.object(evaluation, "OPPONENT", pool.identities[0]), \
                    patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimizer construction")), \
                    patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("No optimizer step")):
                loaded = evaluation.decode_fixed_checkpoint(path, original)
                evaluation.check_fixed(loaded)
                self.assertEqual(weights_sha256(loaded.model), model_hash)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(digest(path), before)
                self.assertTrue(all(not parameter.requires_grad for parameter in loaded.model.parameters()))
                changed = deepcopy(loaded.model.state_dict())
                next(iter(changed.values())).view(-1)[0] += 1
                with self.assertRaisesRegex(ValueError, "Resume tensor differs"):
                    evaluation.assert_same_state(loaded.model.state_dict(), changed)
            payload = torch.load(path, weights_only=True)
            payload["resume"]["sampler_state"]["next_episode"] = 8
            invalid = Path(temp) / "invalid.pt"
            torch.save(payload, invalid)
            with self.assertRaisesRegex(ValueError, "missing after-evaluation phase"):
                evaluation.decode_fixed_checkpoint(invalid, original)

    def execute_mocked(self, folder, effect=None):
        plan = self.plan(folder)
        plan["resume_checkpoint"] = {"path": "synthetic saved checkpoint"}
        patches = [patch.object(evaluation, "read_plan", return_value=plan),
                   patch.object(evaluation, "check_original", return_value={}),
                   patch.object(evaluation, "decode_fixed_checkpoint", return_value=object()),
                   patch.object(evaluation, "check_integrity"), patch.object(evaluation, "check_fixed"),
                   patch.object(evaluation, "collect_attempt", side_effect=effect,
                                return_value={"status": "truncated", "termination_reason": "collection_window",
                                              "outcomes": {"red": None, "blue": None}, "rewards": {"red": None, "blue": None}}),
                   patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimizer construction")),
                   patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("No optimizer step"))]
        entered = []
        try:
            entered = [item.start() for item in patches]
            result = evaluation.execute(folder / "plan.json", "a" * 64)
            calls = entered[5].call_args_list
        finally:
            for item in reversed(patches): item.stop()
        return result, calls

    def test_executor_calls_only_the_four_missing_slots_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            result, calls = self.execute_mocked(folder)
            self.assertEqual(result["status"], "completed")
            self.assertEqual([call.args[2] for call in calls], evaluation.fixed_schedule())
            self.assertEqual([row["episode"] for row in result["attempts"]], [7, 8, 9, 10])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["additional_optimizer_steps"], 0)
            self.assertFalse(result["training_performed"])
            self.assertEqual(result["saved_progress"], evaluation.PROGRESS)
            self.assertEqual(result, json.loads((folder / "execution/report.json").read_text()))
            with self.assertRaises(FileExistsError):
                self.execute_mocked(folder)

    def test_executor_retains_first_failure_and_never_replaces_it(self):
        for error in (ValueError("Native replay failed"), RuntimeError("Cleanup failed"),
                      evaluation.BudgetExpired("Original shared wall-time budget expired")):
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                result, calls = self.execute_mocked(folder, effect=error)
                self.assertEqual(result["status"], "stopped")
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(result["attempts"]), 1)
                self.assertEqual(result["attempts"][0]["episode"], 7)
                self.assertEqual(result["attempts"][0]["status"], "collection_error")
                self.assertIsNone(result["attempts"][0]["outcomes"])
                self.assertIsNone(result["attempts"][0]["rewards"])
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertEqual(result, json.loads((folder / "execution/report.json").read_text()))

    def test_diagnostic_contract_rejects_training_and_excess_budget(self):
        spec = evaluation.fixed_schedule()[0]
        checkpoint = "first-update-" + evaluation.LEARNER_SHA256[:16]
        rows = []
        for player, identity in enumerate(((checkpoint, evaluation.LEARNER_SHA256), evaluation.OPPONENT)):
            other = evaluation.OPPONENT if player == 0 else (checkpoint, evaluation.LEARNER_SHA256)
            rows.append(SimpleNamespace(player=player, episode="7", role="learner" if player == 0 else "frozen_opponent",
                behavior_checkpoint=identity[0], weights_sha256=identity[1], opponent_checkpoint=other[0],
                opponent_weights_sha256=other[1], opponent_pool=evaluation.POOL, purpose="diagnostic", origin="native",
                optimization_eligible=False, decisions=[SimpleNamespace(decision_seq=0)], history=[SimpleNamespace(day=30)]))
        collected = SimpleNamespace(rollouts=rows)
        loaded = SimpleNamespace(checkpoint=checkpoint)
        self.assertEqual(evaluation.validate_collected(collected, spec, loaded), 2)
        for attribute, value in (("purpose", "training"), ("origin", "synthetic"), ("optimization_eligible", True)):
            old = getattr(rows[0], attribute)
            setattr(rows[0], attribute, value)
            with self.assertRaisesRegex(ValueError, "exclusion from PPO"):
                evaluation.validate_collected(collected, spec, loaded)
            setattr(rows[0], attribute, old)
        rows[0].history[0].day = 31
        with self.assertRaisesRegex(ValueError, "day budget"):
            evaluation.validate_collected(collected, spec, loaded)
        rows[0].history[0].day = 30
        rows[0].decisions *= 256
        with self.assertRaisesRegex(ValueError, "sampled-choice budget"):
            evaluation.validate_collected(collected, spec, loaded)


if __name__ == "__main__":
    unittest.main()
