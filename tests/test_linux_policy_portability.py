"""Synthetic, inference-only portability checks; no game data or optimizer."""
from copy import deepcopy
import json
import platform
import stat
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    raise unittest.SkipTest('Optional torch dependency is not installed')
from experiments.learner import contextual as c
from experiments.learner.checkpoints import FrozenOpponentPool, save_policy_checkpoint, load_policy_checkpoint
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, DIALOGS_MODEL_SCHEMA
from experiments.learner.inference import source_identity
from tools import port_fixed_policy_linux as p


@unittest.skipUnless(str(torch.__version__) == '2.9.1+cpu' and platform.system() == 'Linux'
                     and platform.machine() == 'x86_64' and sys.version_info[:2] == (3, 12),
                     'Linux x86_64 CPython 3.12 CPU portability target only')
class LinuxPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original.pt'
        self.output = self.root / 'converted'
        model = ContextualActorCritic(ModelConfig(schema=DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        pool = FrozenOpponentPool(); pool.add('synthetic-opponent', model)
        source = source_identity(c.DIALOGS_ENVIRONMENT)
        save_policy_checkpoint(self.original, model, checkpoint='synthetic-original', source_metadata=source,
                               opponent_pool=pool, interface=c.DIALOGS_ENVIRONMENT)
        payload = torch.load(self.original, weights_only=True)
        payload['torch_version'] = payload['source_metadata']['torch'] = '2.9.1'
        payload['source_metadata']['python'] = '3.13.5 (synthetic macOS fixture)'
        torch.save(payload, self.original)
        self.payload = payload
        self.expected = dict(expected_sha256=p.digest(self.original), expected_bytes=self.original.stat().st_size,
                             expected_learner=payload['weights_sha256'], expected_opponents=list(pool.identities))

    def run_conversion(self):
        return p._convert(self.original, self.output, **self.expected)

    def test_exact_tensors_and_original_preserved_under_standard_loader(self):
        before = self.original.read_bytes()
        rng_before = torch.get_rng_state().clone()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer')):
            report = self.run_conversion()
        self.assertTrue(report['exact_learner_and_opponent_tensors'])
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(before, self.original.read_bytes())
        converted = self.output / 'fixed-policy-linux.pt'
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in (converted, self.output / 'portability.json'):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        loaded = load_policy_checkpoint(converted, interface=c.DIALOGS_ENVIRONMENT)
        self.assertEqual(loaded.checkpoint, 'synthetic-original-linux-cpu-v1')
        self.assertEqual(loaded.source_metadata['portability']['original_source_metadata'], self.payload['source_metadata'])
        self.assertEqual(report['converted']['sha256'], p.digest(converted))
        self.assertEqual(json.loads((self.output / 'portability.json').read_text()), report)
        self.assertEqual(report['cross_platform_behavior_equivalence'], 'not_established')
        with self.assertRaisesRegex(ValueError, 'version or provenance'):
            load_policy_checkpoint(self.original, interface=c.DIALOGS_ENVIRONMENT)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_conversion()

    def test_public_entry_point_refuses_non_authorized_original(self):
        with self.assertRaisesRegex(ValueError, 'size or SHA-256'):
            p.convert(self.original, self.output)
        self.assertFalse(self.output.exists())

    def test_private_conversion_inside_checkout_refused_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'outside the Git checkout'):
            p._convert(self.original, p.ROOT / 'private-conversion-test', **self.expected)

    def test_exact_comparison_checks_dtype_shape_and_signed_zero(self):
        for other in (torch.tensor([0.], dtype=torch.float64), torch.tensor([-0.]),
                      torch.tensor([[0.]]), torch.tensor([1.])):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, 'tensor'):
                p.assert_exact_state(torch.tensor([0.]), other)
        for tensor in (torch.tensor(1.), torch.empty(0), torch.tensor([True]),
                       torch.arange(12).reshape(3, 4).t(), torch.tensor([1.], dtype=torch.bfloat16)):
            p.assert_exact_state(tensor, tensor.clone())

    def test_hash_change_is_rejected_before_loading(self):
        self.original.write_bytes(self.original.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'size or SHA-256'):
            self.run_conversion()
        self.assertFalse(self.output.exists())

    def test_rebound_structural_runtime_source_and_tensor_corruption_rejected(self):
        def tensor_change(v):
            next(iter(v['model_state'].values())).view(-1)[0] += 1
        changes = [lambda v: v.update(resume={}), lambda v: v.update(torch_version='2.8.0'),
                   lambda v: v['contracts'].update(environment='wrong'),
                   lambda v: v['source_metadata'].update(python='3.12.0 wrong'),
                   lambda v: v['source_metadata']['source_sha256'].update({'experiments/learner/contextual_model.py':'0'*64}),
                   lambda v: v['opponent_pool'][0].update(weights_sha256='0'*64), tensor_change]
        for change in changes:
            with self.subTest(change=change):
                payload = deepcopy(self.payload); change(payload)
                torch.save(payload, self.original)
                self.expected.update(expected_sha256=p.digest(self.original), expected_bytes=self.original.stat().st_size)
                with self.assertRaises(ValueError): self.run_conversion()
                self.assertFalse(self.output.exists())
