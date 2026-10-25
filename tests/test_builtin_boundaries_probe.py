"""Zero-action probe controls tested with synthetic requests, never native games."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experiments.learner import contextual as c
from validation.probes import validate_builtin_boundaries as probe
from validation.probes.create_boundary_fixture import create_fixture
from tests.test_builtin_env import initial
from tests.test_contextual_dialogs import history_at


class FakeEnv:
    def __init__(self, player=0, history=None):
        self.history_value = initial(player) if history is None else history
        self.step = Mock(side_effect=AssertionError("No learner action is allowed"))
        self.truncations = []
        self.player = player

    def reset(self, **kwargs):
        h = self.history_value
        record = c.encode_frame(h.decision)
        return SimpleNamespace(done=False, player=self.player, color=probe.COLORS[self.player],
            kind="contextual", reward=0, past_horizon=False, record=record,
            frames=[c.encode_frame(f) for f in h.frames])

    def truncate(self, reason):
        self.truncations.append(reason)
        h = self.history_value
        h.emit("cutoff", decision_seq=h.decision.seq, reason=reason)
        declaration = dict.fromkeys(("actions_sent", "actions_accepted", "actions_rejected", "responses_undelivered"), 0)
        return SimpleNamespace(done=True, status="truncated", termination_reason=reason,
            outcomes=dict.fromkeys(probe.COLORS), rewards=dict.fromkeys(probe.COLORS),
            manifest={"players": {probe.COLORS[self.player]: declaration}})

    def history(self, color):
        return [c.encode_frame(f) for f in self.history_value.frames]


class BuiltinBoundaryProbeTests(unittest.TestCase):
    def test_both_seats_freeze_actual_initial_decision_without_action(self):
        for seat in (0, 1):
            env = FakeEnv(seat)
            end, metrics = probe.held_cutoff(env, {"episode": seat+1, "engine_seed": 121901, "learner_player": seat})
            self.assertEqual(env.truncations, ["collection_window"])
            env.step.assert_not_called()
            self.assertEqual(metrics["held_decision_seq"], 0)
            self.assertEqual(metrics["cutoff_seq"], 1)
            self.assertEqual(metrics["policy_actions"], 0)
            self.assertEqual(end.rewards, {"red": None, "blue": None})

    def test_natural_before_own_decision_is_failed_diagnostic_without_fake_cutoff(self):
        env = FakeEnv()
        env.reset = Mock(return_value=SimpleNamespace(done=True, status="completed"))
        with self.assertRaisesRegex(ValueError, "ended before"):
            probe.held_cutoff(env, {"episode": 1, "engine_seed": 121901, "learner_player": 0})
        self.assertEqual(env.truncations, [])
        env.step.assert_not_called()

    def test_strategic_dialog_is_never_answered_or_truncated(self):
        env = FakeEnv(history=history_at())
        with self.assertRaisesRegex(ValueError, "safe held decision"):
            probe.held_cutoff(env, {"episode": 1, "engine_seed": 121901, "learner_player": 0})
        self.assertEqual(env.truncations, [])
        env.step.assert_not_called()

    def test_cutoff_that_changes_initial_decision_is_not_admitted(self):
        env = FakeEnv()
        original = env.truncate
        def changed(reason):
            h = env.history_value
            old = h.decision
            h.emit("superseded", decision_seq=old.seq, reason="visible_state_changed")
            h.emit("decision", observation=old.observation, revision=1, offers=old.offers)
            return original(reason)
        env.truncate = changed
        with self.assertRaisesRegex(ValueError, "initial held decision"):
            probe.held_cutoff(env, {"episode": 1, "engine_seed": 121901, "learner_player": 0})
        env.step.assert_not_called()

    def test_exact_fixture_bound_and_historical_script_explicitly_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "fixture"
            create_fixture(folder, kind="natural-townless")
            identity = probe.fixture_identity(folder / "natural-townless.vmap")
            self.assertFalse(identity["ordinary_cohort_member"])
            self.assertFalse(identity["natural_outcome_validation"])
            self.assertIn("not used", identity["probe_plan"])
            self.assertEqual(identity["manifest"], probe.binding(folder / "fixture.json"))
            with self.assertRaisesRegex(ValueError, "Fixture"):
                probe.fixture_identity(probe.ROOT / "experiments/explorer/maps/random-baseline-v1.vmap")


if __name__ == "__main__":
    unittest.main()
