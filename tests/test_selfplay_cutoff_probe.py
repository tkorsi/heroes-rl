"""Cutoff probe accounting and exact mixed-policy controls; no native games."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.learner import contextual as c, contextual_model as m
from experiments.learner.contextual_rollout import ContextualSession
from tests.test_builtin_v8_cutoff import mandatory_history
from tests.test_contextual_capabilities import begin, decide, observation
from tests.test_contextual_capability_cutoff_replay import CapabilityFixture
from tools import validate_selfplay_cutoffs as probe


class SelfPlayCutoffProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_four_controls_and_failed_attempts_consume_the_entire_reservation(self):
        rows = probe.schedule()
        self.assertEqual([row["engine_seed"] for row in rows], list(range(1_946_000_000, 1_946_000_004)))
        self.assertEqual([(row["case"], row["player"]) for row in rows], [
            ("optional-capability", "red"), ("optional-capability", "blue"),
            ("mandatory-level-up", "red"), ("mandatory-level-up", "blue")])
        self.assertEqual(len(set(row["policy_seed"] for row in rows)), 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = probe.reserve(root / "first-attempt", {"synthetic": True}, base=root / "reservations")
            raw = probe.campaign.read(probe.campaign.bound(receipt))
            self.assertTrue(raw["consumed_including_unstarted"])
            self.assertEqual([row["engine_seed"] for row in raw["schedule"]], list(probe.SEEDS))
            # No result file is needed: an interrupted attempt still consumes
            # every reserved case, including cases it never launched.
            following = probe.reserve(root / "second-attempt", {"synthetic": True}, base=root / "reservations")
            second = probe.campaign.read(probe.campaign.bound(following))
            self.assertEqual([row["engine_seed"] for row in second["schedule"]], [seed + 4 for seed in probe.SEEDS])
            self.assertEqual([row["policy_seed"] for row in second["schedule"]], [seed + 4 for seed in probe.POLICY_SEEDS])
            with self.assertRaisesRegex(ValueError, "already consumed"):
                probe.reserve(root / "first-attempt", {"synthetic": True}, base=root / "reservations")
            self.assertEqual(probe.campaign.read(probe.campaign.bound(receipt)), raw)

    def mixed(self):
        torch.manual_seed(173)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.EXPANDED_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        history = mandatory_history()
        session = ContextualSession(model, history.episode, 0, "synthetic-fixed", "scripted-fixture-operator",
            schema=c.EXPANDED_SCHEMA, seed=19, guard_version=probe.GUARD)
        session.consume(history.frames)
        selected = session.choose(history.decision.seq)
        seq = begin(history, selected.action, "engine")
        history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision + 1)
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
        decide(history, observation())
        session.consume(history.frames)
        return probe.neural_evidence(session, 19), model

    def test_mixed_reconstruction_samples_only_labelled_mandatory_answer(self):
        raw, model = self.mixed()
        result = probe.reconstruct_mixed(raw, model)
        self.assertTrue(result["passed"])
        self.assertFalse(result["fully_policy_controlled"])
        self.assertFalse(result["training_input"])
        self.assertEqual(result["actions_resampled"], 1)
        self.assertEqual(result["values_recomputed"], 3)
        self.assertEqual(result["maximum_absolute_difference"], 0)

    def test_mixed_reconstruction_rejects_changed_provenance_support_and_answer(self):
        raw, model = self.mixed()
        for mutate in (lambda x: x.update(training_input=True), lambda x: x.update(fully_policy_controlled=True),
                       lambda x: x.update(policy_support=None), lambda x: x["neural_decisions"].clear(),
                       lambda x: x["neural_decisions"][0].update(log_prob=0.0),
                       lambda x: x["neural_decisions"][0].update(decision_seq=0),
                       lambda x: x["estimates"][0].update(value=123.0)):
            altered = deepcopy(raw)
            mutate(altered)
            with self.subTest(mutation=altered), self.assertRaises(ValueError):
                probe.reconstruct_mixed(altered, model)

    def test_both_colors_finalizer_actions_are_bound_outside_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            for player, color in enumerate(("red", "blue")):
                fixture = CapabilityFixture(Path(temp) / color, player)
                run = SimpleNamespace(env=SimpleNamespace(episode_dir=fixture.directory,
                    sides={p: SimpleNamespace(validator=producer) for p, producer in enumerate(fixture.producers)}),
                    history=lambda selected: [c.encode_frame(frame) for frame in fixture.producers[
                        ("red", "blue").index(selected)].history.frames])
                result = probe.continuation_evidence(run, {"case": "optional-capability", "player": color})
                self.assertEqual(result[color]["actions"], ["close_capability", "close", "end_turn"])
                self.assertTrue(all(item["outside_learner_prefix"] for item in result.values()))
                self.assertEqual(result[color]["live_submissions"] - result[color]["frozen_submissions"], 3)
                with self.assertRaisesRegex(ValueError, "not executed"):
                    probe.continuation_evidence(run, {"case": "optional-capability", "player": ("blue", "red")[player]})


if __name__ == "__main__":
    unittest.main()
