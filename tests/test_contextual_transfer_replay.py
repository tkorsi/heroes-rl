"""Pure mutation selection tests; native semantic checks run separately."""
from copy import deepcopy
import unittest

from experiments.headless.contextual_replay import tamper_cases


class TransferReplayMutationTests(unittest.TestCase):
    def test_count_and_slots_change_independently_without_changing_original(self):
        slots = lambda prefix, creature, count: [
            {"ref": f"{prefix}-{index}", "slot": index, "creature": creature if index == 0 else None,
             "count": count if index == 0 else 0} for index in range(7)]
        armies = {"source": {"slots": slots("upper", 0, 3)},
                  "destination": {"slots": slots("lower", 0, 5)}}
        action = {"kind": "transfer_stack", "selection": "upper-0", "destination": "lower-0", "count": 2}
        tape = [{"kind": "init"},
                {"seq": 0, "kind": "message", "player": 1,
                 "message": {"records": [{"observation": {"window": {"town_panel": {"armies": armies}}}}]}},
                {"seq": 1, "kind": "response", "player": 1, "response": {"action": action}}]
        original = deepcopy(tape)
        cases = {name: rows for name, rows in tamper_cases(tape) if "transfer" in name}
        self.assertEqual(set(cases), {"exact_transfer_count", "explicit_transfer_source_slot",
                                     "explicit_transfer_destination_slot"})
        expected = {"exact_transfer_count": {**action, "count": 1},
                    "explicit_transfer_source_slot": {**action, "selection": "upper-1"},
                    "explicit_transfer_destination_slot": {**action, "destination": "lower-1"}}
        for name, changed in cases.items():
            self.assertEqual(changed[-1]["response"]["action"], expected[name])
            self.assertEqual(changed[:-1], tape[:-1])
        self.assertEqual(tape, original)

    def test_old_profile_tape_does_not_gain_transfer_claims(self):
        tape = [{"kind": "init"}, {"seq": 0, "kind": "response", "player": 0,
                "response": {"action": {"kind": "recruit", "count": 1}}}]
        self.assertFalse(any("transfer" in name for name, _ in tamper_cases(tape)))


if __name__ == "__main__":
    unittest.main()
