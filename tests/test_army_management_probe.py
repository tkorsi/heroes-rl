"""Independent arithmetic/receipt oracles for native v6 diagnostics; no games."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from experiments.learner import contextual as c
from validation.probes import validate_army_management as p


def army(subject, role, content):
    return c.ArmyView(subject + "-army", role, subject, role != "town_garrison",
        tuple(c.ArmySlot(subject + "-slot-" + str(index), index,
                        content[index][0] if index in content else None,
                        content[index][1] if index in content else 0) for index in range(7)))


def frame(armies, *, resources=5000, heroes=None, context=None, stack_panel=None):
    return NS(seq=10, observation=NS(window=NS(armies=tuple(armies), stack_panel=stack_panel),
        resources=(0, 0, 0, 0, 0, 0, resources), context=context or c.Context("town", "town", "overview"),
        heroes=heroes if heroes is not None else (c.Hero("hero", (1, 1, 0), 1000, 1500, False),),
        towns=(c.Town("town", (1, 1, 0)),)))


def receipt(status):
    result = c.Result(status, True)
    return (NS(kind="consumed", decision_seq=10, seq=11, consumer="engine"),
            NS(kind="issued", decision_seq=10, seq=12, command_index=0),
            NS(kind="acknowledged", decision_seq=10, seq=13, command_index=0, accepted=True),
            NS(kind="result", decision_seq=10, seq=14, result=result)), result


def action(kind, source, destination=None, count=None):
    return c.Action(kind, subject="town", selection=source, destination=destination,
                    count=count, schema=c.MANAGEMENT_ACTION_SCHEMA)


class ArmyManagementProbeTests(unittest.TestCase):
    def setUp(self):
        self.upper = army("town", "town_garrison", {0: (0, 14), 1: (1, 7)})
        self.hero = army("hero", "visiting_hero", {0: (0, 25)})
        self.before = frame((self.upper, self.hero))

    def test_purchase_returns_to_overview_without_sending_an_extra_back(self):
        # Regression for retained native mechanics-1: successful recruitment
        # already closed its panel, then the old probe sent an invalid back.
        selected = NS(slider_max=14, unit_cost=(0, 0, 0, 0, 0, 0, 60))
        purchase = c.Action("recruit", "town", selection="recruit-0", destination="army-0",
                            count=7, schema=c.MANAGEMENT_ACTION_SCHEMA)
        after = frame((self.upper, self.hero), resources=4580)
        after.observation.window.context = after.observation.context
        after.observation.window.town_panel = NS(kind="overview")
        additions, result = receipt("recruited")
        run = object.__new__(p.ArmyRun)
        run.open_recruitment = Mock()
        run.select_creature = Mock(return_value=selected)
        run.purchase = Mock(return_value=purchase)
        run.call = Mock(return_value=(self.before, additions, result))
        run.inspect_request = Mock(return_value=after)
        run.buy(0, 7)
        run.call.assert_called_once_with(purchase, "explicit_prepare_creature_purchase")
        run.purchase.assert_called_once_with(selected, 7)
        after.observation.window.context = c.Context("town", "town", "recruitment", "dwelling-0")
        after.observation.window.town_panel = NS(kind="recruitment")
        with self.assertRaisesRegex(p.base.ProbeFailure, "Wrong active town panel"):
            run.buy(0, 7)

    def test_exact_partial_split_checks_both_rows_and_every_untouched_slot(self):
        selected = action("split_stack", "town-slot-0", "hero-slot-1", 3)
        after = frame((army("town", "town_garrison", {0: (0, 11), 1: (1, 7)}),
                       army("hero", "visiting_hero", {0: (0, 25), 1: (0, 3)})))
        additions, result = receipt("split")
        proof = p.exact_stack_operation(self.before, after, selected, additions, result)
        self.assertEqual(proof["count"], 3)
        corrupted = frame((army("town", "town_garrison", {0: (0, 11), 1: (1, 6)}), after.observation.window.armies[1]))
        with self.assertRaisesRegex(p.base.ProbeFailure, "unexpected army"):
            p.exact_stack_operation(self.before, corrupted, selected, additions, result)

    def test_reverse_split_and_same_army_rearrangement_preserve_last_stack_rule(self):
        reverse = p.expected_stack_operation((self.upper, self.hero), action("split_stack", "hero-slot-0", "town-slot-0", 2))
        self.assertEqual(reverse[0]["slots"][0]["count"], 16)
        self.assertEqual(reverse[1]["slots"][0]["count"], 23)
        moved = p.expected_stack_operation((self.hero,), action("split_stack", "hero-slot-0", "hero-slot-2", 25))
        self.assertIsNone(moved[0]["slots"][0]["creature"])
        self.assertEqual(moved[0]["slots"][2]["count"], 25)
        with self.assertRaisesRegex(p.base.ProbeFailure, "last required stack"):
            p.expected_stack_operation((self.upper, self.hero), action("split_stack", "hero-slot-0", "town-slot-0", 25))

    def test_merge_empties_source_and_unlike_swap_exchanges_exact_counts(self):
        merged = p.expected_stack_operation((self.upper, self.hero), action("merge_stacks", "town-slot-0", "hero-slot-0"))
        self.assertEqual((merged[0]["slots"][0]["creature"], merged[0]["slots"][0]["count"]), (None, 0))
        self.assertEqual(merged[1]["slots"][0]["count"], 39)
        swapped = p.expected_stack_operation((self.upper, self.hero), action("swap_stacks", "town-slot-1", "hero-slot-0"))
        self.assertEqual((swapped[0]["slots"][1]["creature"], swapped[0]["slots"][1]["count"]), (0, 25))
        self.assertEqual((swapped[1]["slots"][0]["creature"], swapped[1]["slots"][0]["count"]), (1, 7))
        with self.assertRaises(p.base.ProbeFailure):
            p.expected_stack_operation((self.upper, self.hero), action("split_stack", "town-slot-1", "hero-slot-0", 1))

    def test_same_creature_swap_exchanges_unequal_counts_instead_of_merging(self):
        selected = action("swap_stacks", "town-slot-0", "hero-slot-0")
        after = frame((army("town", "town_garrison", {0: (0, 25), 1: (1, 7)}),
                       army("hero", "visiting_hero", {0: (0, 14)})))
        additions, result = receipt("swapped")
        p.exact_stack_operation(self.before, after, selected, additions, result)
        incorrectly_merged = frame((army("town", "town_garrison", {1: (1, 7)}),
                                    army("hero", "visiting_hero", {0: (0, 39)})))
        with self.assertRaisesRegex(p.base.ProbeFailure, "unexpected army"):
            p.exact_stack_operation(self.before, incorrectly_merged, selected, additions, result)

    def test_native_receipt_rejects_missing_rejected_or_reordered_ack(self):
        additions, result = receipt("split")
        for bad in (additions[:2] + additions[3:],
                    additions[:2] + (NS(**{**vars(additions[2]), "accepted": False}),) + additions[3:],
                    additions[:2] + (NS(**{**vars(additions[2]), "seq": 15}),) + additions[3:]):
            with self.assertRaises(p.base.ProbeFailure): p.native_receipt(self.before, bad, result, "split")

    def test_only_participating_heroes_may_recalculate_movement_maximum(self):
        extra = c.Hero("unrelated", (4, 4, 0), 900, 1500, False)
        before = frame((self.upper, self.hero), heroes=(*self.before.observation.heroes, extra))
        selected = action("swap_stacks", "town-slot-1", "hero-slot-0")
        after = frame((army("town", "town_garrison", {0: (0, 14), 1: (0, 25)}),
                       army("hero", "visiting_hero", {0: (1, 7)})),
                      heroes=(replace(before.observation.heroes[0], movement_max=1560), extra))
        additions, result = receipt("swapped")
        p.exact_stack_operation(before, after, selected, additions, result)
        after.observation.heroes = (after.observation.heroes[0], replace(extra, movement_max=1560))
        with self.assertRaisesRegex(p.base.ProbeFailure, "owned object identity"):
            p.exact_stack_operation(before, after, selected, additions, result)

    def test_upgrade_checks_selected_creature_full_cost_and_all_other_slots(self):
        panel = c.StackPanel(self.before.observation.context, self.hero.slots[0], (c.UpgradeChoice("upgrade-1", 1),),
                             "upgrade-1", (0, 0, 0, 0, 0, 0, 375), True)
        before = frame((), context=c.Context("stack", "hero", "upgrade", "hero-slot-0"), stack_panel=panel)
        after = frame((self.upper, army("hero", "visiting_hero", {0: (1, 25)})), resources=4625,
                      heroes=(replace(self.before.observation.heroes[0], movement_max=1560),))
        selected = c.Action("upgrade_stack", "hero", selection="upgrade-1", schema=c.MANAGEMENT_ACTION_SCHEMA)
        additions, result = receipt("upgraded")
        proof = p.exact_upgrade(self.before, before, after, selected, additions, result)
        self.assertEqual(proof["total_cost"], [0, 0, 0, 0, 0, 0, 375])
        self.assertEqual(proof["count"], 25)
        after.observation.resources = (0, 0, 0, 0, 0, 0, 4626)
        with self.assertRaisesRegex(p.base.ProbeFailure, "resource amount"):
            p.exact_upgrade(self.before, before, after, selected, additions, result)

    def test_garrison_entry_consolidates_and_exit_restores_empty_town(self):
        upper = army("town", "town_garrison", {0: (0, 14)})
        before = frame((upper, self.hero))
        garrison = army("hero", "garrison_hero", {0: (0, 39)})
        inside = frame((garrison,), heroes=(replace(before.observation.heroes[0], in_garrison=True),))
        additions, result = receipt("garrison_swapped")
        self.assertEqual(p.exact_garrison(before, inside, additions, result)["operation"], "enter")
        outside = frame((army("town", "town_garrison", {}), replace(garrison, role="visiting_hero")))
        self.assertEqual(p.exact_garrison(inside, outside, additions, result)["operation"], "exit")
        outside.observation.window.armies = (upper, replace(garrison, role="visiting_hero"))
        with self.assertRaisesRegex(p.base.ProbeFailure, "hidden town troops"):
            p.exact_garrison(inside, outside, additions, result)

    def test_occupied_garrison_swap_preserves_both_armies_and_hero_identities(self):
        garrison = army("second", "garrison_hero", {0: (2, 9), 2: (0, 10)})
        heroes = (self.before.observation.heroes[0], c.Hero("second", (1, 1, 0), 1000, 1500, True))
        before = frame((garrison, self.hero), heroes=heroes)
        after = frame((replace(self.hero, role="garrison_hero"), replace(garrison, role="visiting_hero")),
                      heroes=(replace(heroes[0], in_garrison=True), replace(heroes[1], in_garrison=False)))
        additions, result = receipt("garrison_swapped")
        self.assertEqual(p.exact_garrison(before, after, additions, result)["operation"], "swap")
        after.observation.heroes = (replace(heroes[0], in_garrison=True), replace(heroes[1], in_garrison=False, movement=999))
        with self.assertRaisesRegex(p.base.ProbeFailure, "unrelated owned state"):
            p.exact_garrison(before, after, additions, result)


if __name__ == "__main__": unittest.main()
