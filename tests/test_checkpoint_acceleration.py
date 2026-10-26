"""Frozen evaluation input gates; no native game or gameplay admission."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from tools import validate_checkpoint_acceleration as harness


class FrozenValidationTests(unittest.TestCase):
    def test_wrong_input_hash_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "after-policy.pt"
            path.write_bytes(b"original")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            harness.pinned(path, expected)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Pinned input changed"):
                harness.pinned(path, expected)

    def test_rejects_wrong_interface_update_and_total_budget(self):
        plan = {"source": {"interface": harness.INTERFACE}, "configuration": {
            "interface": harness.INTERFACE, "ordinary_choice_limit": 4960,
            "reserved_dialog_choices": 160, "max_total_choices": 5120,
            "max_days": 365, "torch_threads": 1}}
        report = {"cumulative_progress": dict(harness.PROGRESS),
            "plan_sha256": harness.PINS["plan.json"],
            "after_checkpoint": {"weights_sha256": harness.WEIGHTS}}
        harness.validate_configuration(plan, report)
        for section, key, value in (("source", "interface", "adventure-dialogs-v1"),
                                    ("configuration", "max_total_choices", 4968),
                                    ("configuration", "reserved_dialog_choices", 8)):
            changed = deepcopy(plan)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                harness.validate_configuration(changed, report)
        changed = deepcopy(report)
        changed["cumulative_progress"]["updates"] = 411
        with self.assertRaisesRegex(ValueError, "update412"):
            harness.validate_configuration(plan, changed)

    def test_loader_override_restores_strict_original_even_on_error(self):
        from experiments.learner import evaluate_builtin_opponent as evaluator
        original = evaluator.load_fixed
        replacement = object()
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            with harness.scoped_loader(replacement):
                self.assertIs(evaluator.load_fixed, replacement)
                raise RuntimeError("diagnostic failure")
        self.assertIs(evaluator.load_fixed, original)

    def test_strict_restore_rejects_missing_rng_without_mutating_caller(self):
        import torch
        from experiments.learner.builtin_preservation import verify_pair
        from tests.test_builtin_training_resume import ParentResumeTests
        fixture = ParentResumeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = deepcopy(fixture.raw)
        del raw["rng"]["torch_cpu"]
        broken = fixture.root / "missing-rng.pt"
        torch.save(raw, broken)
        before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
        with self.assertRaises((ValueError, KeyError)):
            verify_pair(broken, fixture.run / "execution/after-policy.pt", fixture.plan, fixture.report)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(random.getstate(), before_python)
