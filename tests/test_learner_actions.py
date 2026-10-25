"""Candidate schemas describe permitted controls, not a catalog or strategy."""
from dataclasses import replace
import unittest

from experiments.learner.actions import (
    ActionArgument, ActionCandidate, CandidateSet, movement_candidates,
)


class ActionContractTests(unittest.TestCase):
    def test_movement_support_exactly_matches_attempt_mask(self):
        support = movement_candidates((False, True) + (False,) * 6 + (True,))
        self.assertEqual([c.key for c in support.candidates], ["1", "8"])
        self.assertEqual(support.candidates[0].arguments, (ActionArgument(0, "category", 1),))
        self.assertEqual(support.candidates[-1].arguments, ())

    def test_parameterized_support_is_variable_and_typed(self):
        support = CandidateSet((
            ActionCandidate("opaque-choice-a", 17, (
                ActionArgument(0, "category", 23), ActionArgument(1, "scalar", 7),
                ActionArgument(2, "position", (3, 4, 0)),
            )),
        ))
        self.assertEqual(len(support.candidates), 1)
        with self.assertRaises(ValueError):
            replace(support, schema="unsupported-hota-version")

    def test_aliases_and_duplicate_parameter_slots_rejected(self):
        candidate = ActionCandidate("a", 17, (ActionArgument(0, "category", 3),))
        with self.assertRaises(ValueError):
            CandidateSet((candidate, replace(candidate, key="b")))
        with self.assertRaises(ValueError):
            CandidateSet((candidate, replace(candidate, type_id=18)))
        with self.assertRaises(ValueError):
            replace(candidate, arguments=candidate.arguments * 2)

    def test_no_silent_vocabulary_expansion_or_malformed_support(self):
        for make in (
            lambda: ActionCandidate("a", 256),
            lambda: ActionArgument(0, "category", 4096),
            lambda: ActionArgument(0, "scalar", float("nan")),
            lambda: ActionArgument(0, "position", (20, 0, 0)),
            lambda: CandidateSet(()),
            lambda: movement_candidates((False,) * 9),
        ):
            with self.assertRaises(ValueError):
                make()


if __name__ == "__main__":
    unittest.main()
