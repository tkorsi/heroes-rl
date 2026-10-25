"""v6 negative-evidence selection tests; these are not real-engine outcomes."""
from copy import deepcopy
import unittest

from experiments.headless.contextual_replay import tamper_cases
from experiments.learner import contextual as c


def fixture():
    slots = [{"ref": f"slot-{i}", "slot": i, "creature": 0 if i == 0 else None,
              "count": 9 if i == 0 else 0} for i in range(7)]
    observation = {"heroes": [{"in_garrison": True}], "window": {
        "context": {"kind": "exchange", "subject": "hero-a", "selection": "hero-b", "panel": None},
        "armies": [{"slots": slots}], "stack_panel": {"offers": [{"ref": "up-0", "creature": 1}],
        "selected_upgrade_ref": "up-0", "total_cost": [0, 0, 0, 0, 0, 0, 90], "affordable": True}}}
    profile = c.MANAGEMENT_PROFILE
    tape = [{"kind": "init", "schema": "heroes-contextual-replay-v1", "interface": profile.environment,
             "protocol_version": profile.protocol, "history_schema": profile.history_schema, "action_schema": profile.action_schema},
            {"kind": "message", "player": 0, "message": {"records": [
                {"observation": observation},
                {"event": {"kind": "army_exchange_requested", "subject": "hero-a", "partner": "hero-b"}},
                {"kind": "context_opened", "reason": "own_hero_exchange", "revision": 8,
                 "context": {"subject": "hero-a", "selection": "hero-b"}}]}}]
    for kind in ("split_stack", "merge_stacks", "swap_stacks", "swap_garrison", "select_upgrade", "upgrade_stack", "close"):
        tape.append({"kind": "response", "player": 0, "response": {"action": {
            "kind": kind, "subject": "hero-a", "selection": "slot-0" if "stack" in kind else "up-0",
            "destination": "slot-1", "count": 3}}})
    for seq, row in enumerate(tape[1:]): row["seq"] = seq
    return tape


EXPECTED = {"garrisoned_hero_flag", "displayed_army_count", "displayed_upgrade_creature",
            "selected_upgrade_reference", "upgrade_native_cost", "upgrade_affordability",
            "exchange_requested_partner", "exchange_opened_partner", "exchange_opened_revision",
            "split_stack_source", "split_stack_destination", "exact_split_count", "merge_stacks_source",
            "merge_stacks_destination", "swap_stacks_source", "swap_stacks_destination",
            "garrison_swap_subject", "select_upgrade_selection", "upgrade_stack_selection", "exchange_close_action"}


class ArmyMutationTests(unittest.TestCase):
    def cases(self, tape):
        return {name: changed for name, changed in tamper_cases(tape) if name in EXPECTED}

    def test_mutations_are_independent_copies_of_only_witnessed_fields(self):
        tape = fixture(); original = deepcopy(tape)
        cases = self.cases(tape)
        self.assertEqual(set(cases), EXPECTED)
        self.assertEqual(tape, original)
        for name, changed in cases.items():
            self.assertNotEqual(changed, tape, name)
            self.assertEqual(changed[0], tape[0])
            self.assertEqual([row["seq"] for row in changed[1:]], list(range(len(changed) - 1)))
        self.assertEqual(cases["exact_split_count"][2]["response"]["action"]["count"], 2)
        self.assertEqual(cases["split_stack_destination"][2]["response"]["action"]["destination"], "slot-2")

    def test_profile_is_exact_and_old_profiles_gain_no_new_claims(self):
        for key, value in (("interface", c.HEROES_ENVIRONMENT), ("protocol_version", 8.0),
                           ("history_schema", c.HEROES_SCHEMA), ("action_schema", c.HEROES_ACTION_SCHEMA)):
            tape = fixture(); tape[0][key] = value
            self.assertEqual(self.cases(tape), {})
        self.assertEqual(self.cases(fixture()[:1]), {})

    def test_other_player_cannot_supply_displayed_slots_or_exchange_context(self):
        tape = fixture(); tape[1]["player"] = 1
        cases = self.cases(tape)
        self.assertNotIn("split_stack_source", cases)
        self.assertNotIn("exchange_close_action", cases)
        self.assertIn("exact_split_count", cases)


if __name__ == "__main__": unittest.main()
