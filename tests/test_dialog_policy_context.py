"""Synthetic frozen-policy reconstruction/controller checks; no native games."""
from copy import deepcopy
from dataclasses import replace
import json
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
from tests.test_contextual_dialogs import history_at, panel
from validation.probes import validate_dialog_policy_context as probe

torch.set_num_threads(1)


class DialogPolicyContextTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(915)
            self.model = ContextualActorCritic(ModelConfig(schema=model_schema_for(c.DIALOGS_ENVIRONMENT),
                hidden_size=8, feature_size=4)).eval().requires_grad_(False)

    def evidence(self, family="chest"):
        history = history_at(panel(family))
        sessions = []
        for player in (0, 1):
            own = tuple(replace(frame, player=player) for frame in history.frames)
            session = ContextualSession(self.model, history.episode, player, "fixed-dialog-policy",
                "scripted-context-operator", schema=c.DIALOGS_SCHEMA, seed=512 + player)
            session.diagnostic_seed = 512 + player
            session.consume(own)
            if player == 0: session.choose(history.decision.seq)
            sessions.append(session)
        return probe.choice_evidence(sessions), sessions

    def test_schedule_is_declared_for_all_eight_families_and_both_colors(self):
        settings = probe.Settings()
        rows = settings.schedule()
        self.assertEqual(len(rows), 16)
        self.assertEqual({(row["family"], row["player"]) for row in rows},
                         {(family, color) for family in probe.FAMILIES for color in probe.base.COLORS})
        self.assertEqual(len({row["engine_seed"] for row in rows}), 16)
        self.assertEqual(len({seed for row in rows for seed in row["policy_seeds"]}), 32)
        self.assertEqual(settings.schedule("chest", "blue"), [row for row in rows if row["family"] == "chest" and row["player"] == "blue"])
        for key, value in (("max_choices", 0), ("max_choices", 9), ("max_choices", True),
                           ("engine_seed", 2147483647), ("policy_seed", 0), ("case_seconds", float("nan"))):
            with self.subTest(key=key), self.assertRaises(probe.base.ProbeFailure):
                replace(settings, **{key: value})

    def test_all_families_resample_exact_draw_and_rebuild_both_own_memories(self):
        for family in probe.FAMILIES:
            with self.subTest(family=family):
                raw, sessions = self.evidence(family)
                result = probe.replay_choices(raw, self.model)
                self.assertEqual([row["actions_resampled"] for row in result["players"]], [1, 0])
                self.assertEqual([row["values_recomputed"] for row in result["players"]], [2, 2])
                self.assertEqual(result["maximum_absolute_difference"], 0)
                self.assertTrue(result["separate_private_memory_resets"])
                choice = next(iter(sessions[0]._choices.values()))
                self.assertEqual(choice.support.offers, sessions[0].history.decision.offers)
                self.assertFalse(raw["training_input"])
                self.assertFalse(raw["self_play"])

    def test_exact_values_probabilities_prefix_and_local_history_are_bound(self):
        raw, _ = self.evidence()
        for corrupt in ("value", "log_prob", "prefix", "estimate", "history", "player"):
            changed = deepcopy(raw)
            player = changed["players"][0]
            if corrupt == "history": player["history"].pop(1)
            elif corrupt == "prefix": player["neural_decisions"][0]["prefix"]["history_sha256"] = "f" * 64
            elif corrupt == "estimate": player["estimates"][0]["value"] += 0.5
            elif corrupt == "player": changed["players"][1]["history"] = player["history"]
            else: player["neural_decisions"][0][corrupt] += 0.125
            with self.subTest(corrupt=corrupt), self.assertRaises((ValueError, probe.base.ProbeFailure, IndexError)):
                probe.replay_choices(changed, self.model)

    def test_rng_is_replayed_without_sampling_operator_decisions(self):
        raw, _ = self.evidence()
        detected = False
        for seed in range(10):
            changed = deepcopy(raw); changed["players"][0]["policy_seed"] = seed
            try: probe.replay_choices(changed, self.model)
            except probe.base.ProbeFailure:
                detected = True
                break
        self.assertTrue(detected)

    def test_absent_or_different_first_branch_remains_unverified(self):
        self.assertEqual(probe.coverage("chest", [])["status"], "unverified")
        self.assertEqual(probe.coverage("neutral_free_join", [{"family": "neutral_flee"}])["missing"], ["neutral_free_join"])
        result = probe.coverage("chest", [{"family": "chest"}])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["all_alternative_answers_claimed"])

    def test_cutoff_manifest_binds_every_actual_choice_and_first_trigger(self):
        observed = [{"player": 0, "decision_seq": 9}, {"player": 0, "decision_seq": 17}]
        drain = {"reason": "collection_window", "limit": 8, "learner_included": True,
                 "trigger": {**observed[0], "actions_sent": [1, 0]}, "choices": observed}
        self.assertEqual(probe.validate_drain({"dialog_cutoff_drain": drain}, observed)["status"], "passed")
        for key, value in (("choices", observed[:1]), ("trigger", {"player": 0, "decision_seq": 8}), ("limit", 9)):
            with self.subTest(key=key), self.assertRaises(probe.base.ProbeFailure):
                probe.validate_drain({"dialog_cutoff_drain": {**drain, key: value}}, observed)
        self.assertEqual(probe.validate_drain({"dialog_cutoff_drain": None}, [])["status"], "unverified")

    def test_actual_sample_is_saved_before_send_and_done_skips_another_request(self):
        raw, sessions = self.evidence()
        frame = sessions[0].history.decision
        # Use an unchosen session at the same exact history for the controller.
        sessions[0]._choices.clear()
        with tempfile.TemporaryDirectory() as directory:
            run = SimpleNamespace(directory=Path(directory), env=SimpleNamespace(defer_dialog_cutoff=Mock(), truncate=Mock()),
                state=SimpleNamespace(player=0, done=False), inspect_request=Mock(return_value=frame), check=Mock(),
                history=lambda color: raw["players"][probe.base.COLORS.index(color)]["history"])
            def send(action, label):
                saved = json.loads((run.directory / "neural-context-evidence.json").read_text())
                self.assertEqual(saved["players"][0]["neural_decisions"][0]["action"], as_action(action))
                self.assertEqual(label, probe.NEURAL_LABEL)
                run.state = SimpleNamespace(done=True, status="truncated")
                return frame, (), SimpleNamespace(accepted=True, status="dialog_answered")
            run.call = Mock(side_effect=send)
            with patch.object(probe, "native_receipt", return_value={"synthetic_receipt": True}):
                observed = probe.sample_dialog_chain(run, sessions, 8)
            self.assertEqual(len(observed), 1)
            run.env.defer_dialog_cutoff.assert_called_once_with("collection_window")
            run.env.truncate.assert_not_called()
            self.assertEqual(run.inspect_request.call_count, 2)
            run.call.assert_called_once()

    def test_exhausted_chain_fails_without_automatic_answer_or_cutoff(self):
        raw, sessions = self.evidence()
        frame = sessions[0].history.decision
        with tempfile.TemporaryDirectory() as directory:
            run = SimpleNamespace(directory=Path(directory), env=SimpleNamespace(defer_dialog_cutoff=Mock(), truncate=Mock()),
                state=SimpleNamespace(player=0, done=False), inspect_request=Mock(return_value=frame), check=Mock(),
                history=lambda color: raw["players"][probe.base.COLORS.index(color)]["history"],
                call=Mock(return_value=(frame, (), SimpleNamespace(accepted=True, status="dialog_answered"))))
            with patch.object(probe, "native_receipt", return_value={}), patch.object(probe, "dialog_open_proof", return_value={}), \
                 self.assertRaisesRegex(probe.base.ProbeFailure, "choice bound"):
                probe.sample_dialog_chain(run, sessions, 1)
            run.call.assert_called_once()
            run.env.truncate.assert_not_called()


def as_action(action):
    from dataclasses import asdict
    return asdict(action)


if __name__ == "__main__": unittest.main()
