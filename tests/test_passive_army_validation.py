"""Offline safety controls for the bounded native diagnostic harness."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_passive_army_information import informed_observation
from tests.test_contextual_capabilities import observation
from tools import validate_passive_armies as probe


class PassiveArmyValidationTests(unittest.TestCase):
    def test_ownership_inventory_includes_empty_town_and_never_duplicates_garrison_hero(self):
        obs = informed_observation()
        self.assertEqual(set(probe.validate_passive(obs)), {hero.handle for hero in obs.heroes}
                         | {town.handle for town in obs.towns})
        with self.assertRaises(ValueError):
            probe.validate_passive(SimpleNamespace(army_information=obs.army_information,
                heroes=obs.heroes, towns=obs.towns, own_armies=(*obs.own_armies, obs.own_armies[0])))
        with self.assertRaises(ValueError):
            probe.validate_passive(SimpleNamespace(army_information=obs.army_information,
                heroes=obs.heroes, towns=obs.towns,
                own_armies=(*obs.own_armies, replace(obs.own_armies[0], subject="visible-enemy"))))

    def test_disabled_extension_rejects_any_own_army_rows(self):
        self.assertEqual(probe.validate_passive(observation("map"), enabled=False), {})
        with self.assertRaises(ValueError):
            probe.validate_passive(informed_observation(), enabled=False)
        with self.assertRaises(ValueError):
            probe.validate_passive(observation("map"))

    def test_mutation_preserves_original_and_changes_only_one_native_attack(self):
        from experiments.learner.contextual import encode_observation, EXPANDED_SCHEMA
        records = [{"kind": "header", "army_information": "own-army-details-v1"},
                   {"kind": "message", "message": {"records": [{"observation":
                        encode_observation(informed_observation(), schema=EXPANDED_SCHEMA)}]}}]
        original = deepcopy(records)
        mutant = probe.mutate_own_attack(records)
        self.assertEqual(records, original)
        target = mutant[1]["message"]["records"][0]["observation"]["own_armies"][0]["slots"][0]
        target["attack"] -= 1
        self.assertEqual(mutant, original)
        with self.assertRaises(ValueError):
            probe.mutate_own_attack([{"kind": "message", "message": {"records": []}}])

    def test_unstarted_reserved_seeds_are_consumed_and_cannot_be_replayed(self):
        with TemporaryDirectory() as temporary, patch.object(probe, "BASE", Path(temporary).resolve()):
            first = probe.reserve_seeds(Path(temporary).resolve() / "failed-before-native-start")
            second = probe.reserve_seeds(Path(temporary).resolve() / "next-attempt")
            self.assertEqual(set(first["seeds"]), set(probe.CASES))
            self.assertTrue(first["consumed_including_unstarted"])
            all_first = set(first["seeds"].values()) | set(first["policy_seeds"])
            all_second = set(second["seeds"].values()) | set(second["policy_seeds"])
            self.assertFalse(all_first & all_second)
            rows = [json.loads(line) for line in (Path(temporary).resolve() / "seed-ledger.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [first, second])

    def test_nonsequential_ledger_fails_before_reservation(self):
        with TemporaryDirectory() as temporary, patch.object(probe, "BASE", Path(temporary).resolve()):
            path = Path(temporary).resolve() / "seed-ledger.jsonl"
            path.write_text('{"sequence": 8}\n')
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                probe.reserve_seeds(Path(temporary).resolve() / "must-not-run")
            self.assertEqual(path.read_bytes(), before)

    def test_missing_ledger_with_prior_receipt_refuses_to_reuse_any_seed(self):
        with TemporaryDirectory() as temporary, patch.object(probe, "BASE", Path(temporary).resolve()):
            root = Path(temporary).resolve()
            prior = probe.reserve_seeds(root / "first")
            (root / "seed-ledger.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "Missing seed ledger"):
                probe.reserve_seeds(root / "second")
            self.assertEqual(json.loads((root / "reservation-receipts/000000.json").read_text()), prior)
            self.assertFalse((root / "seed-ledger.jsonl").exists())

    def test_tampered_receipt_or_schedule_or_existing_attempt_fails_closed(self):
        for mutation in ("missing-central", "wrong-central", "wrong-schedule", "missing-attempt", "wrong-attempt"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary, \
                    patch.object(probe, "BASE", Path(temporary).resolve()):
                root = Path(temporary).resolve()
                row = probe.reserve_seeds(root / "first")
                receipt = root / "reservation-receipts/000000.json"
                if mutation == "missing-central":
                    receipt.unlink()
                elif mutation == "wrong-central":
                    changed = deepcopy(row); changed["seeds"][probe.CASES[0]] += 1
                    receipt.write_text(json.dumps(changed))
                elif mutation == "wrong-schedule":
                    changed = deepcopy(row); changed["sequence"] = False
                    receipt.write_text(json.dumps(changed))
                    (root / "seed-ledger.jsonl").write_text(json.dumps(changed) + "\n")
                else:
                    (root / "first").mkdir()
                    if mutation == "wrong-attempt":
                        (root / "first/seed-reservation.json").write_text('{}')
                before = (root / "seed-ledger.jsonl").read_bytes()
                with self.assertRaises(ValueError):
                    probe.reserve_seeds(root / "second")
                self.assertEqual((root / "seed-ledger.jsonl").read_bytes(), before)
                self.assertFalse((root / "reservation-receipts/000001.json").exists())

    def test_policy_proof_requires_both_nonempty_exact_reconstructions(self):
        collected = SimpleNamespace(rollouts=tuple(SimpleNamespace(behavior_checkpoint="tiny", decisions=[1, 2],
                                    estimates=[1, 2, 3]) for _ in range(2)))
        proofs = [dict(player=player, checkpoint="tiny", maximum_absolute_difference=0,
                       private_memory_reset=True, weights_unchanged=True, actions_resampled=2,
                       values_recomputed=3) for player in range(2)]
        probe.validate_policy_proofs(proofs, collected)
        for field, value in (("actions_resampled", 0), ("values_recomputed", 2),
                             ("maximum_absolute_difference", .00001), ("private_memory_reset", False)):
            changed = deepcopy(proofs); changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_policy_proofs(changed, collected)

    def test_check_mode_launches_and_writes_nothing(self):
        with patch.object(probe, "check_inputs", return_value={"runtime": {}}), \
                patch.object(probe, "reserve_seeds", side_effect=AssertionError("Check reserved seeds")), \
                patch.object(probe, "write", side_effect=AssertionError("Check wrote data")), \
                patch("builtins.print"):
            self.assertEqual(probe.main(["--check", "--data", "/unused"]), 0)


if __name__ == "__main__":
    unittest.main()
