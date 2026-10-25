"""Check diagnostic accounting without launching an engine or optimizer."""
from types import SimpleNamespace
import unittest

from validation.probes import validate_capabilities as probe


class CapabilityProbeTests(unittest.TestCase):
    def test_trade_checks_native_lot_rounding_and_both_resources(self):
        offer = SimpleNamespace(kind="trade_resource", max_count=3, resource_from=6,
                                resource_to=0, give=250, receive=1)
        before = (10, 10, 10, 10, 10, 10, 1000)
        after = (12, 10, 10, 10, 10, 10, 500)
        self.assertEqual(probe.resource_effect(before, after, offer, 2), [2, 0, 0, 0, 0, 0, -500])
        for malformed in ((12, 10, 10, 10, 10, 10, 501), (12, 11, 10, 10, 10, 10, 500), after[:-1]):
            with self.assertRaises(probe.base.ProbeFailure):
                probe.resource_effect(before, malformed, offer, 2)
        for count in (0, 4, True):
            with self.assertRaises(probe.base.ProbeFailure):
                probe.resource_effect(before, after, offer, count)

    def test_purchase_checks_every_resource_against_displayed_native_cost(self):
        offer = SimpleNamespace(kind="buy_artifact", cost=(2, 0, 3, 0, 0, 0, 500))
        before = (10, 10, 10, 10, 10, 10, 1000)
        self.assertEqual(probe.resource_effect(before, (8, 10, 7, 10, 10, 10, 500), offer),
                         [-2, 0, -3, 0, 0, 0, -500])
        with self.assertRaises(probe.base.ProbeFailure):
            probe.resource_effect(before, (8, 10, 7, 10, 10, 10, 499), offer)

    def test_inventory_keeps_duplicates_and_ownership_without_counting_locks(self):
        def item(subject, identifier, *, locked=False, kind="artifact"):
            return SimpleNamespace(subject=subject, id=identifier, locked=locked, kind=kind)
        counts = probe.inventory_counts((item("self-1", 7), item("self-1", 7), item("self-2", 7),
                                         item("self-1", 8, locked=True), item("self-1", 9, kind="spell")))
        self.assertEqual(dict(counts), {("self-1", 7): 2, ("self-2", 7): 1})

    def test_fixed_schedule_never_replaces_failed_or_randomly_missing_branches(self):
        rows = probe.schedule(tuple(probe.CASES), probe.fixtures.COLORS, 158001, 16)
        self.assertEqual(len(rows), len(probe.CASES) * 2 * 16)
        self.assertEqual(len({row["seed"] for row in rows}), len(rows))
        self.assertEqual(rows, probe.schedule(tuple(probe.CASES), probe.fixtures.COLORS, 158001, 16))
        for row in rows:
            self.assertEqual(row["fixture"], probe.CASES[row["case"]])


if __name__ == "__main__":
    unittest.main()
