"""Synthetic diagnostic-controller and exact recurrent evidence checks; no games."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, model_schema_for
from experiments.learner.contextual_rollout import ContextualSession
from tests.test_contextual_heroes import action, history_at_tavern
from validation.probes import validate_hero_policy_context as probe

torch.set_num_threads(1)


class HeroPolicyContextTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(614)
            self.model = ContextualActorCritic(ModelConfig(schema=model_schema_for(c.HEROES_ENVIRONMENT),
                                                          hidden_size=8, feature_size=4)).eval().requires_grad_(False)
        self.history = history_at_tavern()
        self.session = ContextualSession(self.model, self.history.episode, 0, "fixed-context-policy",
                                         "scripted-context-operator", schema=c.HEROES_SCHEMA, seed=503)
        self.session.diagnostic_seed = 503
        self.session.consume(self.history.frames)
        self.choice = self.session.choose(self.history.decision.seq)
        self.raw = probe.choice_evidence(self.session)

    def test_schedule_declares_both_colors_without_cherry_picking(self):
        settings = probe.Settings()
        self.assertEqual(settings.schedule(), [
            {"player": "red", "engine_seed": 103101, "policy_seed": 103104},
            {"player": "blue", "engine_seed": 103102, "policy_seed": 103105}])
        for key, value in (("max_choices", 0), ("max_choices", 9), ("max_choices", True),
                           ("engine_seed", 2147483647), ("policy_seed", 0),
                           ("case_seconds", float("nan")), ("case_seconds", 121)):
            with self.subTest(key=key), self.assertRaises(probe.base.ProbeFailure):
                replace(settings, **{key: value})

    def test_entire_own_scripted_history_reconstructs_unmodified_neural_draw(self):
        result = probe.replay_choices(self.raw, self.model)
        self.assertTrue(result["passed"])
        self.assertEqual(result["actions_resampled"], 1)
        self.assertEqual(result["values_recomputed"], 3)  # map, overview and Tavern
        self.assertEqual(result["maximum_absolute_difference"], 0)
        self.assertEqual(self.raw["history"], [c.encode_frame(frame) for frame in self.history.frames])
        self.assertFalse(self.raw["training_input"])
        self.assertEqual(self.choice.support.offers, self.history.decision.offers)

    def test_exact_value_probability_prefix_and_history_cannot_be_changed(self):
        for corrupt in ("value", "log_prob", "prefix", "history"):
            changed = deepcopy(self.raw)
            if corrupt == "history":
                changed["history"].pop(1)
            elif corrupt == "prefix":
                changed["neural_decisions"][0]["prefix"]["history_sha256"] = "f" * 64
            else:
                changed["neural_decisions"][0][corrupt] += 0.125
            with self.subTest(corrupt=corrupt), self.assertRaises((ValueError, probe.base.ProbeFailure, IndexError)):
                probe.replay_choices(changed, self.model)
        changed = deepcopy(self.raw)
        changed["estimates"][0]["value"] += 0.5
        with self.assertRaisesRegex(probe.base.ProbeFailure, "own-decision value"):
            probe.replay_choices(changed, self.model)

    def test_sampler_rng_is_reconstructed_without_sampling_scripted_decisions(self):
        differing = False
        for seed in range(10):
            changed = deepcopy(self.raw)
            changed["policy_seed"] = seed
            try: probe.replay_choices(changed, self.model)
            except probe.base.ProbeFailure:
                differing = True
                break
        self.assertTrue(differing)

    def test_missing_selection_or_hire_is_an_honest_coverage_gap(self):
        choices = lambda kinds: tuple(SimpleNamespace(action=SimpleNamespace(kind=kind)) for kind in kinds)
        for kinds, missing in (((), ["select_hero", "hire_hero"]),
                               (("back", "hire_hero"), ["select_hero"]),
                               (("select_hero",), ["hire_hero"])):
            result = probe.branch_coverage(choices(kinds))
            self.assertEqual(result["status"], "unverified")
            self.assertEqual(result["missing"], missing)
        self.assertEqual(probe.branch_coverage(choices(("select_hero", "hire_hero")))["status"], "passed")

    def test_context_controller_executes_draws_and_only_reopens_after_back(self):
        with tempfile.TemporaryDirectory() as directory:
            choices = [SimpleNamespace(action=action(kind, subject="town", **parameters)) for kind, parameters in (
                ("back", {}), ("select_hero", {"selection": "offer-b"}),
                ("hire_hero", {"selection": "offer-b"}))]
            session = SimpleNamespace(consume=Mock(), choose=Mock(side_effect=choices))
            run = SimpleNamespace(args=SimpleNamespace(player="red"), state=SimpleNamespace(color="red"),
                directory=Path(directory), inspect_request=Mock(return_value=self.history.decision),
                history=Mock(return_value=self.raw["history"]), open_tavern=Mock(), check=Mock(),
                call=Mock(return_value=(self.history.decision, (), SimpleNamespace(accepted=True))))
            with patch.object(probe, "choice_evidence", return_value={"training_input": False}), \
                 patch.object(probe, "exact_hire", return_value={"native_receipt": "synthetic-controller-check"}):
                probe.sample_context(run, session, 8)
            self.assertEqual(session.choose.call_count, 3)
            self.assertEqual([call.args[0] for call in run.call.call_args_list], [choice.action for choice in choices])
            run.open_tavern.assert_called_once()
            run.check.assert_called_once()

    def test_context_controller_does_not_add_choices_at_the_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            choice = SimpleNamespace(action=action("back", subject="town"))
            session = SimpleNamespace(consume=Mock(), choose=Mock(return_value=choice))
            run = SimpleNamespace(args=SimpleNamespace(player="red"), state=SimpleNamespace(color="red"),
                directory=Path(directory), inspect_request=Mock(return_value=self.history.decision),
                history=Mock(return_value=self.raw["history"]), open_tavern=Mock(), check=Mock(),
                call=Mock(return_value=(self.history.decision, (), SimpleNamespace(accepted=True))))
            with patch.object(probe, "choice_evidence", return_value={"training_input": False}):
                probe.sample_context(run, session, 8)
            self.assertEqual(session.choose.call_count, 8)
            self.assertEqual(run.open_tavern.call_count, 7)
            run.check.assert_not_called()

    def test_hire_before_selection_uses_explicit_departure_without_extra_neural_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            choices = [SimpleNamespace(action=action(kind, subject="town", **parameters)) for kind, parameters in (
                ("hire_hero", {"selection": "offer-a"}), ("back", {}),
                ("select_hero", {"selection": "offer-b"}))]
            session = SimpleNamespace(consume=Mock(), choose=Mock(side_effect=choices))
            run = SimpleNamespace(args=SimpleNamespace(player="red"), state=SimpleNamespace(color="red"),
                directory=Path(directory), inspect_request=Mock(return_value=self.history.decision),
                history=Mock(return_value=self.raw["history"]), open_tavern=Mock(), check=Mock(),
                open_town=Mock(), depart=Mock(), action=action,
                call=Mock(return_value=(self.history.decision, (), SimpleNamespace(accepted=True))))
            with patch.object(probe, "choice_evidence", return_value={"training_input": False}), \
                 patch.object(probe, "exact_hire", return_value={"new_hero": "real-receipt-handle"}):
                probe.sample_context(run, session, 8)
            self.assertEqual(session.choose.call_count, 3)
            neural = [call.args[0] for call in run.call.call_args_list
                      if call.args[1] == "neural_full_distribution_tavern_choice"]
            self.assertEqual(neural, [choice.action for choice in choices])
            run.depart.assert_called_once_with("real-receipt-handle", "explicit_prepare_neurally_hired_visitor_departure")
            run.open_town.assert_called_once()
            self.assertEqual(run.open_tavern.call_count, 2)


if __name__ == "__main__": unittest.main()
