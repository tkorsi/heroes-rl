"""Check diagnostic accounting without launching an engine or optimizer."""
from types import SimpleNamespace
import unittest

from validation.probes import validate_capabilities as probe


class VisibleArtifactRun:
    """Small native-transfer fake with only delivered inventory and choices."""
    def __init__(self, extras=(), *, partner=True, offers=True, corrupt=False):
        self.layout = [("actor", 0, 36, False), ("actor", 16, 3, True), ("actor", 19, 7, False)]
        self.layout += [("actor", 20 + index, artifact, False) for index, artifact in enumerate(extras)]
        self.layout += [("partner", 19, 88, False)]
        self.heroes = [SimpleNamespace(handle="actor", position=(1, 1))]
        if partner:
            self.heroes.append(SimpleNamespace(handle="partner", position=(2, 1)))
        self.context = SimpleNamespace(kind="capability", subject="actor")
        self.exchange = False
        self.offers, self.corrupt = offers, corrupt
        self.moves, self.checks = [], []

    def inspect_request(self):
        return SimpleNamespace(observation=SimpleNamespace(context=self.context, heroes=self.heroes))

    def capability(self):
        visible = [row for row in self.layout if self.exchange or row[0] == "actor"]
        items = [SimpleNamespace(kind="artifact", subject=owner, slot=slot, id=artifact, locked=locked)
                 for owner, slot, artifact, locked in visible]
        choices = []
        if self.exchange and self.offers:
            for owner, slot, artifact, locked in visible:
                if slot < 19 or locked:
                    continue
                destination = "partner" if owner == "actor" else "actor"
                end = 19 + sum(row[0] == destination and row[1] >= 19 for row in visible)
                choices.append(SimpleNamespace(kind="move_artifact", subject=owner, slot=slot,
                                               artifact=artifact, destination=destination, destination_slot=end))
        return SimpleNamespace(items=items, choices=choices)

    def choose(self, choice):
        if choice not in self.capability().choices:
            raise AssertionError("Test operator selected an undisplayed choice")
        self.moves.append(choice)
        self.layout.remove((choice.subject, choice.slot, choice.artifact, False))
        self.layout = [(owner, slot - int(owner == choice.subject and slot > choice.slot and slot >= 19), artifact, locked)
                       for owner, slot, artifact, locked in self.layout]
        self.layout.append((choice.destination, choice.destination_slot, choice.artifact, False))
        if self.corrupt:
            self.layout.append(("partner", 5, 999, False))

    def action(self, kind, **parameters):
        return SimpleNamespace(kind=kind, **parameters)

    def call(self, action, label):
        if action.kind != "move_to" or (action.x, action.y) != (2, 1):
            raise AssertionError("Test operator used an undisplayed partner")
        self.context = SimpleNamespace(kind="exchange", subject="actor")
        self.exchange = True

    def close_to_map(self):
        self.context = SimpleNamespace(kind="map", subject=None)
        self.exchange = False

    def open_capability(self, kind):
        self.context = SimpleNamespace(kind="capability", subject="actor")
        return self.capability()

    def open_hero_capability(self, kind):
        self.exchange = False
        return self.open_capability(kind)

    def check(self, name, status, **evidence):
        self.checks.append((name, status, evidence))


class CapabilityProbeTests(unittest.TestCase):
    def test_fingerprint_keeps_runtime_dependencies_but_excludes_concurrent_launch_controller(self):
        identity = probe.capability_implementation_identity()
        self.assertNotIn("experiments/learner/builtin_v8_session.py", identity)
        self.assertFalse(any(name.startswith("tests/") for name in identity))
        for name in ("experiments/learner/contextual.py", "experiments/learner/contextual_model.py",
                     "experiments/learner/contextual_rollout.py", "experiments/headless/contextual_replay.py",
                     "validation/probes/validate_capabilities.py", "experiments/explorer/create_capability_scenarios.py"):
            self.assertEqual(identity[name], probe.digest(probe.ROOT / name))

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

    def test_singleton_fixture_needs_no_parking_or_extra_native_actions(self):
        run = VisibleArtifactRun()
        self.assertIsNone(probe.park_backpack_extras(run, "actor"))
        probe.restore_backpack_extras(run, None)
        self.assertEqual(run.moves, [])

    def test_observed_105_and_multiple_extras_preserve_exact_owners_slots_and_locks(self):
        for extras in ((105,), (105, 7, 105)):
            with self.subTest(extras=extras):
                run = VisibleArtifactRun(extras)
                initial = tuple(sorted(run.layout))
                parked = probe.park_backpack_extras(run, "actor")
                self.assertEqual([(item.id, item.slot) for item in run.capability().items if item.slot >= 19], [(7, 19)])
                self.assertEqual([move.slot for move in run.moves], list(reversed(range(20, 20 + len(extras)))))
                probe.restore_backpack_extras(run, parked)
                self.assertEqual(tuple(sorted(run.layout)), initial)
                self.assertEqual(len(run.moves), 2 * len(extras))
                self.assertEqual(run.checks[-1][0], "native_extra_artifact_parking_roundtrip")

    def test_parking_fails_without_authored_item_visible_partner_or_offered_transfer(self):
        missing_item = VisibleArtifactRun((105,))
        missing_item.layout.remove(("actor", 19, 7, False))
        for run, error in ((missing_item, "authored Centaur Axe"),
                           (VisibleArtifactRun((105,), partner=False), "visible owned parking partner"),
                           (VisibleArtifactRun((105,), offers=False), "lacks an offered temporary transfer")):
            with self.subTest(error=error), self.assertRaisesRegex(probe.base.ProbeFailure, error):
                probe.park_backpack_extras(run, "actor")
            self.assertEqual(run.moves, [])

    def test_parking_and_return_reject_unrelated_inventory_changes(self):
        with self.assertRaisesRegex(probe.base.ProbeFailure, "unrelated slots or ownership"):
            probe.park_backpack_extras(VisibleArtifactRun((105,), corrupt=True), "actor")
        run = VisibleArtifactRun((105,))
        parked = probe.park_backpack_extras(run, "actor")
        run.corrupt = True
        with self.assertRaisesRegex(probe.base.ProbeFailure, "unrelated slots or ownership"):
            probe.restore_backpack_extras(run, parked)

    def test_return_requires_current_offer_and_complete_original_layout(self):
        run = VisibleArtifactRun((105,))
        parked = probe.park_backpack_extras(run, "actor")
        run.offers = False
        with self.assertRaisesRegex(probe.base.ProbeFailure, "lacks its offered return"):
            probe.restore_backpack_extras(run, parked)
        run = VisibleArtifactRun((105,))
        parked = probe.park_backpack_extras(run, "actor")
        run.layout.remove(("actor", 0, 36, False))
        run.layout.append(("actor", 1, 36, False))
        with self.assertRaisesRegex(probe.base.ProbeFailure, "complete original inventory"):
            probe.restore_backpack_extras(run, parked)

    def test_fixed_schedule_never_replaces_failed_or_randomly_missing_branches(self):
        rows = probe.schedule(tuple(probe.CASES), probe.fixtures.COLORS, 158001, 16)
        self.assertEqual(len(rows), len(probe.CASES) * 2 * 16)
        # The fog-layout pair intentionally shares native RNG: only its hidden
        # landing occupancy may differ, never its candidate-observation seed.
        self.assertEqual(len({row["seed"] for row in rows}), len(rows) - 2 * 16)
        self.assertEqual(len({(row["seed"], row["case"]) for row in rows}), len(rows))
        self.assertEqual(rows, probe.schedule(tuple(probe.CASES), probe.fixtures.COLORS, 158001, 16))
        for row in rows:
            self.assertEqual(row["fixture"], probe.CASES[row["case"]])


if __name__ == "__main__":
    unittest.main()
