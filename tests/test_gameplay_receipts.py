"""Synthetic receipt tamper checks only; no native runner or gameplay optimizer."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, ARMIES_MODEL_SCHEMA, weights_sha256
from experiments.learner.contextual_rollout import ContextualCollector
from experiments.learner.gameplay import CollectionReceipt, validate_collection_receipt
from experiments.learner.inference import digest, write_json
from tests.test_contextual_rollout import initial, request, resolve, finish_pair


class GameplayReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def fixture(self, directory, *, seat=0, day=1):
        torch.manual_seed(21)
        model = ContextualActorCritic(ModelConfig(schema=ARMIES_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        identities = ("learner", "opponent") if seat == 0 else ("opponent", "learner")
        collector = ContextualCollector((model, model), identities, learner_player=seat, opponent_pool="pool",
            interface=c.ARMIES_ENVIRONMENT, purpose="synthetic", origin="synthetic", policy_seeds=(7, 8))
        collector.start("5")
        pair = []
        for player in (0, 1):
            base = initial(player, episode="5", schema=c.ARMIES_SCHEMA)
            history = c.PlayerHistory("5", player, schema=c.ARMIES_SCHEMA)
            history.append(replace(base.frames[0], day=day))
            resolve(history, collector.act(request(history)))
            pair.append(history)
        collected = collector.finish(finish_pair(tuple(pair), "natural"), histories=tuple(h.frames for h in pair))
        # Deliberate synthetic labels for testing the pure admission boundary.
        # No native evidence or replay validity is asserted by this fixture.
        collected = replace(collected, rollouts=tuple(replace(item, purpose="training", origin="native")
                                                      for item in collected.rollouts))
        manifest = json.loads(collected.manifest_json); manifest["seed"] = 123
        collected = replace(collected, manifest_json=json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        folder = Path(directory).resolve()
        write_json(folder / "manifest.json", manifest)
        write_json(folder / "neural-rollout.json", collected.to_dict())
        weight = weights_sha256(model)
        receipt = CollectionReceipt(str(folder), "5", 123, seat, (7, 8), digest(folder / "neural-rollout.json"),
            digest(folder / "manifest.json"), "learner", weight, "opponent", weight, "pool")
        expected = dict(episode=5, seed=123, seat=seat, policy_seeds=(7, 8), learner_checkpoint="learner",
            learner_digest=weight, opponent_checkpoint="opponent", opponent_digest=weight,
            opponent_pool="pool", max_requests=256, max_days=30)
        return receipt, collected, expected

    def test_both_seats_match_exact_fresh_capture_and_receipt_is_immutable(self):
        for seat in (0, 1):
            with tempfile.TemporaryDirectory() as temp:
                receipt, collected, expected = self.fixture(temp, seat=seat)
                self.assertEqual(validate_collection_receipt(receipt, temp, collected, **expected), collected.rollouts[seat])
                with self.assertRaises(FrozenInstanceError): receipt.episode = "6"

    def test_schedule_frozen_pair_and_policy_seed_changes_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, collected, expected = self.fixture(temp)
            changes = dict(episode=6, seed=124, seat=1, policy_seeds=(8, 7), learner_checkpoint="other",
                learner_digest="1" * 64, opponent_checkpoint="other", opponent_digest="1" * 64, opponent_pool="other")
            for key, value in changes.items():
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_collection_receipt(receipt, temp, collected, **{**expected, key: value})
            with self.assertRaises(ValueError): validate_collection_receipt(None, temp, collected, **expected)

    def test_relabelled_diagnostic_artifact_does_not_match_fresh_capture_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, collected, expected = self.fixture(temp)
            diagnostic = replace(collected, rollouts=tuple(replace(item, purpose="diagnostic") for item in collected.rollouts))
            write_json(Path(temp) / "neural-rollout.json", diagnostic.to_dict())
            diagnostic_receipt = replace(receipt, artifact_sha256=digest(Path(temp) / "neural-rollout.json"))
            with self.assertRaisesRegex(ValueError, "training-purpose"):
                validate_collection_receipt(diagnostic_receipt, temp, diagnostic, **expected)
            # Changing labels back in the artifact cannot alter the receipt held
            # by the running collector, even if all artifact dataclasses validate.
            write_json(Path(temp) / "neural-rollout.json", collected.to_dict())
            with self.assertRaisesRegex(ValueError, "changed after capture"):
                validate_collection_receipt(diagnostic_receipt, temp, collected, **expected)

    def test_artifact_manifest_and_in_memory_edits_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, collected, expected = self.fixture(temp)
            path = Path(temp) / "manifest.json"
            original = path.read_bytes(); path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "changed after capture"):
                validate_collection_receipt(receipt, temp, collected, **expected)
            path.write_bytes(original)
            changed = replace(collected, audit_json=json.dumps({"changed": True}))
            with self.assertRaisesRegex(ValueError, "In-memory collection"):
                validate_collection_receipt(receipt, temp, changed, **expected)

    def test_actual_chosen_action_and_day_budgets_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, collected, expected = self.fixture(temp)
            with self.assertRaisesRegex(ValueError, "choice budget"):
                validate_collection_receipt(receipt, temp, collected, **{**expected, "max_requests": 1})
        with tempfile.TemporaryDirectory() as temp:
            receipt, collected, expected = self.fixture(temp, day=31)
            with self.assertRaisesRegex(ValueError, "day budget"):
                validate_collection_receipt(receipt, temp, collected, **expected)


if __name__ == "__main__": unittest.main()
